from turtle import forward
from unittest.util import three_way_cmp
import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch.nn import init
import math

from scipy.spatial.distance import pdist,squareform
import numpy as np
eps = 1e-8
alpha = 10 #gumbel-GIB MNIST:1.8 5
# alpha = 1.01
# sigma_ = 1e-4 #gumbel-GIB
sigma_ = 1e-4
def pairwise_distances(x):
    assert(len(x.shape) == 2),"x should be two dimensional"
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t())+instances_norm+instances_norm.t()

def calculate_gram_mat(x,sigma):
    dist = pairwise_distances(x)
    return torch.exp(-dist/sigma)

def renyi_entropy(x,sigma):
    k = calculate_gram_mat(x,sigma)
    k = k/(eps+torch.trace(k))
    eigv = torch.abs(torch.linalg.eigvalsh(k))
    eig_pow = eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow)+eps)
    return entropy

class RenyiEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        with torch.no_grad():
            x_numpy = x.cpu().detach().numpy()
            k = squareform(pdist(x_numpy,'euclidean'))
            sigma = np.mean(np.mean(np.sort(k[:,:10],1)))
            sigma = max(sigma_,sigma)
        H = renyi_entropy(x,sigma=sigma**2)
        return H

class ShannonEntopyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        b = F.softmax(x,dim=1)*F.log_softmax(eps+x,dim=1)
        b = -1.0 * b.sum()
        return b

class prune_loss(nn.Module):
    def __init__(self,pr):
        super().__init__()
        self.pr = pr
    def forward(self,mask):
        hold_number = mask.sum()
        drop_number = (1-mask).sum()
        prune_rate = drop_number/(hold_number+drop_number)
        loss = (self.pr-prune_rate)**2 + 1e-4
        return loss.sqrt()

def calculate_sigma(Z_numpy):   

    if Z_numpy.dim()==1:
        Z_numpy = Z_numpy.unsqueeze(1)
    Z_numpy = Z_numpy.cpu().detach().numpy()
    k = squareform(pdist(Z_numpy, 'euclidean'))       # Calculate Euclidiean distance between all samples.
    sigma = np.mean(np.mean(np.sort(k[:, :10], 1)))
    if sigma < sigma_:
        sigma = sigma_
    return sigma 

def reyi_entropy(x,sigma):
    k = calculate_gram_mat(x,sigma)
    k = k/(torch.trace(k)+eps)
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow = eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x,y,s_x,s_y):
    x = calculate_gram_mat(x,s_x)
    y = calculate_gram_mat(y,s_y)
    k = torch.mul(x,y)
    k = k/(torch.trace(k)+eps)
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow =  eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))

    return entropy

def joint_entropy3(x,y,z,s_x,s_y,s_z):
    x = calculate_gram_mat(x,s_x)
    y = calculate_gram_mat(y,s_y)
    z = calculate_gram_mat(z,s_z)
    k = torch.mul(x,y)
    k = torch.mul(k,z)
    k = k/(torch.trace(k)+eps)
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow =  eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))

    return entropy

def calculate_MI(x, y, s_x, s_y):
    Hx = reyi_entropy(x, sigma=s_x)
    Hy = reyi_entropy(y, sigma=s_y)
    Hxy = joint_entropy(x, y, s_x, s_y)
    Ixy = Hx + Hy - Hxy
    return Ixy

class MILoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,X,E):
        s_X = calculate_sigma(X)**2
        s_E = calculate_sigma(E)**2
        mi = calculate_MI(X,E,s_X,s_E)
        return mi

class CMILoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y,d,E,classes_number,domain_number):
        Y = F.one_hot(y,classes_number).float()
        D = F.one_hot(d,domain_number).float()
        s_Y = calculate_sigma(Y)**2
        s_D = calculate_sigma(D)**2
        s_E = calculate_sigma(E)**2
        HDE = joint_entropy(D,E,s_D,s_E)
        HYE = joint_entropy(Y,E,s_Y,s_E)
        HE = reyi_entropy(E,sigma=s_E)
        HYDE = joint_entropy3(Y,D,E,s_Y,s_D,s_E)
        CI = HDE + HYE - HE - HYDE       
        return CI

class DiscreteGate(nn.Module):
    def __init__(self,args,out_features:int,th:float=10,th_a:float=.33,init_:str='uniform',const_:float=.1)->None:
        super().__init__()
        self.mask_type = args.mask_type
        self.th = th
        self.th_a = th_a 
        self.weight = Parameter(torch.Tensor(out_features))
        self.mask = Parameter(torch.ones_like(self.weight))
        self.mask.requires_grad = False
        assert(init_ in ['uniform','uniform-plus','cont'])
        self.init_ = init_
        self.const_ = const_
        assert(self.mask_type in ['tanh','sigmoid'])
        if self.mask_type in ['tanh']:
            self.gate_fn = torch.tanh
        if self.mask_type in ['sigmoid']:
            self.gate_fn = torch.sigmoid
        self.reset_parameters()
    def get_mask(self):
        return self.mask
    def reset_parameters(self):
        if self.init_ in ['uniform']:
            init.uniform_(self.weight,-self.const_,self.const_)
        if self.init_ in ['uniform-plus']:
            init.uniform_(self.weight,0,self.const_)
        if self.init_ in ['const']:
            init.constant_(self.weight,self.const_)
    def update(self):
        def get_mask_(a,m,th,th_a):
            b = 1-(-th_a<a).float()*(a<th_a).float()
            if sum(b*m)<th:
                return m
            else:
                return b*m
        values = self.gate_fn(self.weight).detach()
        self.mask *= get_mask_(values,self.mask,self.th,self.th_a)
        return self.mask
    def forward(self,input):
        self.input = input
        return self.mask*self.gate_fn(self.weight)*input
    def get_loss(self):
        return 1e0*torch.sqrt(((torch.abs(self.gate_fn(self.weight))-torch.ones_like(self.weight))**2).sum())

class gumbel_gate(nn.Module):
    def __init__(self,args,out_features):
        super().__init__()
        self.args = args
        self.in_features = out_features
        self.out_features = out_features
        self.weight = Parameter(torch.zeros(out_features))
        self.mask = Parameter(torch.ones_like(self.weight))
        self.mask.requires_grad = False
        self.rate = self.get_rate(args)

    def get_mask(self):
        return self.mask
    def get_rate(self,args):
        return args.feature_number_rate
    def forward(self,input):
        if self.training:
            zeros_like_weight = torch.zeros_like(self.weight)
            stack_weight = torch.stack((self.weight,zeros_like_weight))
            sampled_tensor = F.gumbel_softmax(stack_weight,tau=self.args.tau,hard=False,dim=0)
            self.sampled_tensor = stack_weight[0]
            return input*sampled_tensor[0],stack_weight[0]
        else:
            return input*self.sampled_tensor,self.mask
    def get_loss(self,mask):
        loss = prune_loss(self.rate)
        return loss(self.sampled_tensor)
    def update(self):
        self.mask.copy_((self.mask * (self.weight>0).float()).detach())
    def update_rate(self,domain_idx):
        self.rate = self.args.feature_number_rate_after