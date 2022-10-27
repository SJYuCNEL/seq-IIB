import imghdr
import imp
import torch
import torch.optim as optim

from model import StdMLP

def get_optimizer(model,lr,weight_decay):
    opt_cls = optim.Adam
    optimizer = opt_cls(model.parameters(),lr=lr,weight_decay=weight_decay)
    return optimizer

def get_optimizer_gate(model,lr,weight_decay):
    opt_cls = optim.Adam
    optimizer = opt_cls(model.get_gate_net().parameters(),lr=lr,weight_decay=weight_decay)
    return optimizer

def get_optimizer_list(model_list,lr,weight_decay):
    optimizer_list = []
    n_e = len(model_list)
    opt_cls = optim.Adam
    for e in range(n_e):
        optimizer_list.append(opt_cls(model_list[e].parameters(),lr=lr,weight_decay=weight_decay))
    return optimizer_list

def combine_models(model_list,x):
    return sum([model_i(x) for model_i in model_list])

def get_models(args,device):
    n_e = args.domain_number
    model_list = []
    for e in range(n_e):
        model_list.append(StdMLP(args).to(device))
    clip_value = 10.
    for model in model_list:
        for p in model.parameters():
            try:
                p.register_hook(lambda grad: torch.clamp(grad,-clip_value,clip_value))
            except:
                pass
    return model_list

import matplotlib.pyplot as plt
from matplotlib import style
def plot_mask_number(mask_list,epochs_number):
    x = list(range(epochs_number))
    font = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size': 20
    }
    style.use('seaborn-whitegrid')
    plt.figure(dpi=300)
    plt.title('The number of neurons with p>50%',fontdict=font)
    plt.plot(x,mask_list,c='tomato',label='Neurons number')
    plt.legend(prop=font)
    plt.xlabel('Epochs',font)
    plt.ylabel('Neurons',font)
    plt.show()
def plot_acc(acc,epochs_number):
    x = list(range(epochs_number))
    plt.figure(dpi=300)
    plt.plot(x,acc,'o-',label='Test Accuracy')
    plt.title('Variation of test accuracy with epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
import numpy as np
def imshow(img,transpose=True):
    # img = img/2+0.5
    npimg=img.numpy()
    plt.figure(dpi=300)
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.axis('off')
    plt.show()