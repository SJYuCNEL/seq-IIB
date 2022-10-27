from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from GIB import MILoss,RenyiEntropyLoss,DiscreteGate,gumbel_gate,CMILoss

class StdMLP(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.ft1 = nn.Flatten()
        self.lin1 = nn.Linear(args.length*args.width*args.height,args.hidden_dim)
        self.do1 = nn.Dropout(args.dropp)
        self.lin2 = nn.Linear(args.hidden_dim,args.hidden_dim)
        self.do2 = nn.Dropout(args.dropp)
        self.lin3 = nn.Linear(args.hidden_dim,args.num_classes)
        self._loss = nn.CrossEntropyLoss()
        self._main = nn.Sequential(self.ft1,
                                    self.lin1,self.do1,nn.ELU(True),
                                    self.lin2,self.do2,nn.ELU(True),
                                    self.lin3)
    def forward(self,x):
        out = self._main(x)
        return out
    def eval_loss(self,y_,y):
        return self._loss(y_,y)
    
class GatedMLP(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.ft1 = nn.Flatten()
        self.lin1 = nn.Linear(args.length*args.width*args.height,args.hidden_dim)
        self.do1 = nn.Dropout(args.dropp)
        self.lin2 = nn.Linear(args.hidden_dim,args.hidden_dim)
        self.do2 = nn.Dropout(args.dropp)
        self.gate3 = DiscreteGate(args,args.hidden_dim)
        self.lin3 = nn.Linear(args.hidden_dim,args.num_classes)
        self._loss = nn.CrossEntropyLoss()
        self._loss_entropy = RenyiEntropyLoss()
        self._features = nn.Sequential(self.ft1,self.lin1,self.do1,nn.ELU(True),self.lin2,self.do2,nn.ELU(True))
        self._classifier = nn.Sequential(self.gate3,self.lin3)
        self._main = nn.Sequential(self._features,self._classifier)

    def get_mask(self):
        return self.gate3.get_mask()
    def update_mask(self):
        self.gate3.update()
    def get_gate_net(self):
        return self.gate3
    def forward(self,x):
        out = self._main(x)
        return out
    def eval_loss(self,y_,y):
        return self._loss(y_,y)
    def get_entropy_loss(self,x):
        encoder = self.gate3(self._features(x))
        return self._loss_entropy(encoder)

class seq_IIB(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.ft1 = nn.Flatten()
        self.length,self.width,self.height = args.length, args.width, args.height
        self.lin1 = nn.Linear(args.length*args.width*args.height,args.hidden_dim)
        self.do1 = nn.Dropout(args.dropp)
        self.lin2 = nn.Linear(args.hidden_dim,args.hidden_dim)
        self.do2 = nn.Dropout(args.dropp)
        self.gate3 = gumbel_gate(args,args.hidden_dim)
        self.lin3 = nn.Linear(args.hidden_dim,args.num_classes)
        self._loss = nn.CrossEntropyLoss()
        self._loss_entropy = RenyiEntropyLoss()
        self._loss_cmi = CMILoss()
        self._loss_mi = MILoss()
        self._features = nn.Sequential(self.ft1,self.lin1,self.do1,nn.ELU(True),self.lin2,self.do2,nn.ELU(True))
        self.rate = self.set_rate(args)
        
    def get_mask(self):
        return self.gate3.get_mask()
    def get_mask_information(self):
        mask = self.gate3.get_mask()
        number = mask.sum()
        return number
    def get_gate_net(self):
        return self.gate3
    def forward(self,x):
        out = self._features(x)
        out,mask = self.gate3(out)
        out = self.lin3(out)
        return out,mask
    def eval_loss(self,y_,y):
        return self._loss(y_,y)
    def get_entropy_loss(self,x):
        encoder,_ = self.gate3(self._features(x))
        return self._loss_mi(x.view(x.size(0),-1),encoder)
    def get_prune_loss(self,mask):
        return self.gate3.get_loss(mask)
    def update_mask(self):
        self.gate3.update()
    def update_rate(self):
        self.gate3.update_rate()
    def set_rate(self,args):
        self.gate3.get_rate(args)
    def get_cmi_loss(self,y,d,x_in,classes_number,domain_number):
        E,_ = self.gate3(self._features(x_in))
        return self._loss_cmi(y,d,E,classes_number,domain_number)

class IRM_MLP(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.ft1 = nn.Flatten()
        self.lin1 = nn.Linear(args.length*args.width*args.height,args.hidden_dim)
        self.do1 = nn.Dropout(args.dropp)
        self.lin2 = nn.Linear(args.hidden_dim,args.hidden_dim)
        self.do2 = nn.Dropout(args.dropp)
        self.classifier = nn.Linear(args.hidden_dim,args.num_classes)
        self._features = nn.Sequential(
            self.ft1,
            self.lin1,self.do1,nn.ELU(True),
            self.lin2,self.do2,nn.ELU(True),
            )
        self._main = nn.Sequential(self._features,self.classifier)
        self._loss = nn.CrossEntropyLoss()

    @staticmethod
    def _irm_penalty(logits,y):
        device = "cuda:0" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2]*scale,y[::2])
        loss_2 = F.cross_entropy(logits[1::2]*scale,y[1::2])
        grad_1 = torch.autograd.grad(loss_1,[scale],create_graph=True)[0]
        grad_2 = torch.autograd.grad(loss_2,[scale],create_graph=True)[0]
        result = torch.sum(grad_1*grad_2)
        return result

    def forward(self,input):
        output = self._main(input)
        return output
    
    def irm_loss(self,y_,y):
        return self._irm_penalty(y_,y)

    def eval_loss(self,y_,y):
        return self._loss(y_,y)

class IIB(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.ft1 = nn.Flatten()
        self.lin1 = nn.Linear(args.length*args.width*args.height,args.hidden_dim)
        self.do1 = nn.Dropout(args.dropp)
        self.lin2 = nn.Linear(args.hidden_dim,args.hidden_dim)
        self.do2 = nn.Dropout(args.dropp)
        self.featurizer = nn.Sequential(self.ft1,self.lin1,self.do1,nn.ELU(True),
                                    self.lin2,self.do2,nn.ELU(True))

        self.encoder = nn.Sequential(
            nn.Linear(args.hidden_dim,args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hidden_dim,args.hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.fc_mu = nn.Linear(args.hidden_dim,args.hidden_dim)
        self.fc_logvar = nn.Linear(args.hidden_dim,args.hidden_dim)
        self.inv_classifier = nn.Linear(args.hidden_dim,args.num_classes)
        self.env_classifier = nn.Linear(args.hidden_dim+1,args.num_classes)

        self._loss = nn.CrossEntropyLoss()

    def encoder_fun(self,feat):
        latent_z = self.encoder(feat)
        mu = self.fc_mu(latent_z)
        logvar = self.fc_logvar(latent_z)
        return mu,logvar

    def reparameterize(self,mu,logvar):
        if self.training:
            std = torch.exp(logvar/2)
            eps = torch.randn_like(std)
            return torch.add(torch.mul(std,eps),mu)
        else:
            return mu
    def forward(self,x):
        z = self.featurizer(x)
        mu,logvar = self.encoder_fun(z)
        z = self.reparameterize(mu,logvar)
        # domain_indx = torch.full((x.size(0),1),domain_idx[0])
        # embeddings = torch.cat([curr_dom_embed for curr_dom_embed in domain_indx]).to(x.device)
        return z,mu,logvar

    def eval_loss(self,z,y):
        return self._loss(self.inv_classifier(z),y)
    
    def ib_loss(self,mu,logvar):
        return -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    
    def env_loss(self,z,d_in,y):
        return F.cross_entropy(self.env_classifier(torch.cat([z,d_in.view(z.size(0),1)],1)),y)
    
    def predict(self,x):
        z = self.featurizer(x)
        mu,logvar = self.encoder_fun(z)
        z = self.reparameterize(mu,logvar)
        y = self.inv_classifier(z)
        return y

class IBIRM(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.ft1 = nn.Flatten()
        self.lin1 = nn.Linear(args.length*args.width*args.height,args.hidden_dim)
        self.do1 = nn.Dropout(args.dropp)
        self.lin2 = nn.Linear(args.hidden_dim,args.hidden_dim)
        self.do2 = nn.Dropout(args.dropp)
        self.classifier = nn.Linear(args.hidden_dim,args.num_classes)
        self._features = nn.Sequential(
            self.ft1,
            self.lin1,self.do1,nn.ELU(True),
            self.lin2,self.do2,nn.ELU(True),
            )
        self._loss = nn.CrossEntropyLoss()

    @staticmethod
    def _irm_penalty(logits,y):
        device = "cuda:0" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2]*scale,y[::2])
        loss_2 = F.cross_entropy(logits[1::2]*scale,y[1::2])
        grad_1 = torch.autograd.grad(loss_1,[scale],create_graph=True)[0]
        grad_2 = torch.autograd.grad(loss_2,[scale],create_graph=True)[0]
        result = torch.sum(grad_1*grad_2)
        return result

    def forward(self,input):
        inter_logits = self._features(input)        
        output = self.classifier(inter_logits)
        return inter_logits,output
    
    def irm_loss(self,y_,y):
        return self._irm_penalty(y_,y)

    def eval_loss(self,y_,y):
        return self._loss(y_,y)
    
    def var_loss(self,inter_logits):
        return inter_logits.var(dim=0).mean()