from email.mime import image
from email.policy import default
from pickletools import optimize
from tkinter import BASELINE

from turtle import update
import torch
import numpy as np

from Interpretability import inter
from model import IBIRM, IIB, IRM_MLP, StdMLP,GatedMLP,seq_IIB
from util import get_optimizer,get_optimizer_gate,get_optimizer_list,combine_models,get_models, plot_acc, plot_mask_number, imshow
from data import load_data_more_mnist,load_data_mnist,load_data_more_mnist_2, load_data_more_mnist_gray

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
    
def try_gpu(i=0):
    if torch.cuda.device_count():
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def test(model,test_datasets,device,ways_to_train,domain_idx):
    if ways_to_train == 'IRMG':
        pass
    else:
        model.eval()
    with torch.no_grad():
        test_number = 0.0
        right_number = 0.0
        for x_te,d_te,y_te in test_datasets:
            x_te = x_te.to(device)
            y_te = y_te.to(device)
            if ways_to_train == "seq-IIB":
                y_te_,_ = model(x_te)
            elif ways_to_train == "IRMG":
                model_list = model
                y_te_ = combine_models(model_list[:(domain_idx+1)],x_te)
            elif ways_to_train == "IIB":
                y_te_ = model.predict(x_te)
            elif ways_to_train == "IBIRM":
                _,y_te_ = model(x_te)
            else:
                y_te_ = model(x_te)
            _,pd = torch.max(y_te_.data,1,keepdim=False)
            rt = (pd.view([-1])==y_te.view([-1])).float().sum()
            right_number += rt
            test_number += y_te.size(0)
        return right_number/test_number
    
def get_acc(model,datasets,domain_idx,device,ways_to_train):
    if domain_idx == -1:
        all_acc = test(model,datasets[-1],device,ways_to_train,domain_idx)
        return all_acc
    else:
        acc = []
        for i in range(domain_idx):
            env_acc = test(model,datasets[i],device,ways_to_train,domain_idx)
            acc.append(env_acc)
        print(acc)
        return sum(acc)/len(acc)

def load_checkpoint(model,checkpoint_PATH,optimizer):
    model_CKPT = torch.load(checkpoint_PATH+'.pkl')
    model.load_state_dict(model_CKPT['state_dict'])
    print('loading checkpoint!')
    optimizer.load_state_dict(model_CKPT['optimizer'])
    return model,optimizer

def train(args,model,train_data,test_data,device):
    lr,num_epochs,net_freeze_epoch,up_mask_epoch,ways_to_train,weight_decay = args.lr,args.num_epochs,args.net_freeze_epoch,args.up_mask_epoch,args.ways_to_train,args.weight_decay
    path = 'checkpoint/' + ways_to_train + args.experiment_time
    if args.domain_number == 4 :
        gray_data = load_data_more_mnist_gray(args,args.batch_size)
    if ways_to_train == 'IRMG':
        optimizer_list = get_optimizer_list(model,lr,weight_decay=weight_decay)
    else:
        optimizer = get_optimizer(model,lr=lr,weight_decay=weight_decay)
    change_domain = num_epochs/(len(train_data)-1)
    if ways_to_train == 'IRMG':
        model_list = model
    else:
        model.to(device)
    test_acc = 0
    domain_idx = 0
    as_one = args.as_one
    acc = []
    for epoch in range(num_epochs):
        if as_one == True:
            if epoch == 0:
                train_iter = train_data[-1]
                domain_idx = -1
                best_train = 0
                best_test = 0
        else:
            if epoch%change_domain == 0 and domain_idx-1 < len(train_data):
                train_iter = train_data[domain_idx]
                domain_idx += 1
                best_train = 0
                best_test = 0
            if epoch == change_domain:
                if ways_to_train == 'seq-IIB':
                        #Select the model that performs best in the first environment before freezing the parameters (not a necessary setting)
                        model,optimizer = load_checkpoint(model,path,optimizer)  
                        up_mask_epoch = args.up_mask_epoch_after
            
        if ways_to_train == 'ERM':
            pass
        elif ways_to_train == 'IRM':
            if epoch == 0:
                lambda0,lambda1 = 1,args.irm_loss
        elif ways_to_train == 'IRMG':
            pass
        elif ways_to_train == 'Gate':
            if epoch == 0:
                mask_list = []
                lambda0,lambda1 = args.lambda0before,args.lambda1before
            elif epoch == net_freeze_epoch:
                lambda0,lambda1 = args.lambda0after,args.lambda1after
                optimizer = get_optimizer_gate(model,lr=lr,weight_decay=weight_decay)
        elif ways_to_train == 'IIB':
            if epoch == 0:
                lambda_beta = args.lambda_beta
                lambda_inv_risks = args.lambda_inv_risks
        elif ways_to_train == 'IBIRM':
            if epoch == 0:
                penalty_weight = args.penalty_weight
                ib_penalty_weight = args.ib_penalty_weight
        elif ways_to_train == 'seq-IIB':
            if epoch == 0:
                mask_list = []
                lambda0,lambda1,lambda2,p = args.lambda0before,args.lambda1before,args.lambda2before,args.pbefore      
            elif epoch == net_freeze_epoch:
                lambda0,lambda1,lambda2,p = args.lambda0after,args.lambda1after,args.lambda2after,args.pafter
                optimizer = get_optimizer_gate(model,lr=lr,weight_decay=weight_decay)
        print("Epoch:{}/{}".format(epoch+1,num_epochs))
        if ways_to_train == 'IRMG':
            pass
        else:
            model.train()
        loss = 0
        loss_num = 0
        for x_in,d_in,y_in in train_iter:
            loss_num += 1
            x_in = x_in.to(device)
            y_in = y_in.to(device)
            d_in = d_in.to(device)
            if ways_to_train == 'seq-IIB':
                y_pred,mask = model(x_in)
            elif ways_to_train == 'IRMG':
                y_pred = combine_models(model_list[:(domain_idx)],x_in)
            elif ways_to_train == 'IIB':
                z,mu,logvar= model(x_in)
            elif ways_to_train == 'IBIRM':
                inter_logits,y_pred = model(x_in)
            else:
                y_pred = model(x_in)
            if ways_to_train == 'IRMG':
                loss_value = model_list[domain_idx-1].eval_loss(y_pred,y_in)
            elif ways_to_train == 'IIB':
                loss_value = model.eval_loss(z,y_in)
            else:
                loss_value = model.eval_loss(y_pred,y_in)
            if ways_to_train == 'ERM':
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
            elif ways_to_train == 'IRM':
                loss_value = lambda0*loss_value + lambda1*model.irm_loss(y_pred,y_in)
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
            elif ways_to_train == 'IRMG':
                optimizer_list[domain_idx-1].zero_grad()
                loss_value.backward()
                optimizer_list[domain_idx-1].step()
            elif ways_to_train == "Gate":
                loss_value = lambda0*loss_value+lambda1*model.get_entropy_loss(x_in)
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
                if epoch%up_mask_epoch==up_mask_epoch-1 :
                    model.update_mask()
                    m = (model.get_mask()>0.5).long().sum()
                    print(f"mask={m}",end=", ")
            elif ways_to_train == 'IIB':
                loss_value = loss_value + model.env_loss(z,d_in,y_in) + lambda_beta*model.ib_loss(mu,logvar) + lambda_inv_risks*(loss_value-model.env_loss(z,d_in,y_in))**2
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
            elif ways_to_train == 'IBIRM':
                loss_value = loss_value + penalty_weight*model.irm_loss(y_pred,y_in) + ib_penalty_weight * model.var_loss(inter_logits)
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
            elif ways_to_train == "seq-IIB":
                loss_value = lambda0*loss_value+lambda2*model.get_cmi_loss(y_in,d_in,x_in,args.classes_number,args.domain_number)+lambda1*model.get_entropy_loss(x_in)+p*model.get_prune_loss(mask)
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
                mask_number = model.get_mask_information()
                if epoch%up_mask_epoch==up_mask_epoch-1 :
                    model.update_mask()
            loss += loss_value
        if ways_to_train == "seq-IIB":
            mask = mask_number
            mask_list.append(mask_number.cpu())
            print(f"mask={mask}",end=",")
        elif ways_to_train == "Gate":
            m = (model.get_mask()>0.5).long().sum()
            mask_list.append(m.cpu())
        train_acc = get_acc(model,train_data,domain_idx,device,ways_to_train)
        train_test_acc = test(model,test_data,device,ways_to_train,domain_idx)
        acc.append(train_test_acc.cpu())
        if args.gray and args.domain_number==4:
            gray_acc = test(model,gray_data,device,ways_to_train,domain_idx)
        if best_train < train_acc:  #Save the best-performing models in the training set
            if ways_to_train == 'IRMG':
                for e in range(len(model_list)):
                    torch.save({
                        'epoch' : epoch+1,
                        'state_dict': model_list[e].state_dict(),
                        'best_loss' : loss/loss_num,
                        'optimizer' : optimizer_list[e].state_dict()},path+f'{e}'+'.pkl')
            else:
                torch.save({
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'best_loss' : loss/loss_num,
                    'optimizer' : optimizer.state_dict()},path+'.pkl')
            best_train = train_acc
            test_acc = train_test_acc
            print("Train_loss:{:.3f} ,Train_acc:{:.3f} ,Test_acc:{:.3f}".format(loss/loss_num,train_acc,test_acc))
        if best_test < train_test_acc:
            best_test = train_test_acc
            best_test_train = train_acc
            print("Train_loss:{:.3f} ,Train_acc:{:.3f} ,Test_acc:{:.3f}".format(loss/loss_num,train_acc,best_test))
        print("Train_loss:{:.3f} ,Train_acc:{:.3f},test_acc:{:.3f}".format(loss/loss_num,train_acc,train_test_acc))
        print("Best_train:{:.3f},test_acc:{:.3f} Best_test_train:{:.3f},Best_test:{:.3f} ".format(best_train,test_acc,best_test_train,best_test))
        if args.gray and args.domain_number==4:
            print("Gray acc: {:.3f}".format(gray_acc))
        if args.Inter == True and epoch == num_epochs-1:
            inter(args,model,test_data,device)
    if ways_to_train == 'Gate' or ways_to_train == 'seq-IIB':
        plot_mask_number(mask_list,num_epochs)
        plot_acc(acc,num_epochs)

def model_selection(args):
    if args.ways_to_train == 'ERM':
        return StdMLP(args)
    elif args.ways_to_train == 'IRM':
        return IRM_MLP(args)
    elif args.ways_to_train == 'IRMG':
        return get_models(args,try_gpu())
    elif args.ways_to_train == 'Gate':
        return GatedMLP(args)
    elif args.ways_to_train == 'seq-IIB':
        model = seq_IIB(args)
        return model
    elif args.ways_to_train == 'IIB':
        return IIB(args)
    elif args.ways_to_train == 'IBIRM':
        return IBIRM(args)

def dataset_selection(args):
    if args.domain_number == 2:
        train_data,test_data = load_data_mnist(args.batch_size)
    elif args.domain_number == 4:
        train_data,test_data = load_data_more_mnist(args)
    elif args.domain_number == 8:
        train_data,test_data = load_data_more_mnist_2(args.batch_size)
    return train_data,test_data