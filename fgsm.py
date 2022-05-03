"""
Created on Tue Apr 26 23:18:18 2022
@author: Ziming Huang
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import *

from robustbench.eval import benchmark


######################
## FGSM: Attack
######################

def fgsm(model, X, y, epsilon=1e-8):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()


def main():
    BATCH_SIZE = 100

    df_train = datasets.CIFAR10(root=r'/tmp/data',download=True, train=True, 
                                transform = transforms.ToTensor())
    df_test = datasets.CIFAR10(root=r'/tmp/data',download=True, train=False, 
                                transform = transforms.ToTensor())
    loader_train = DataLoader(df_train, batch_size = BATCH_SIZE, shuffle=True)
    loader_test = DataLoader(df_test, batch_size = BATCH_SIZE, shuffle=True)
    device = torch.device("cpu")


    nn1 = torch.load('cifar10_nn1_standard.pkl')
            
    ######################
    ## FGSM: Attack
    ######################
            
    print("FGSM: Attacked error rate:", 
          epoch_adv(dataloader=loader_test, hypothesis=nn1, 
                    optimizer=optim.SGD,  
                    device = device, 
                    mode = None,
                    algo_adv = fgsm,
                    epsilon = 8/255)[0])
    
    
    ######################
    ## FGSM: Training
    ######################
    
    
    nn1_fgsm = CNN()
            
    train_arr_fgsm = train_adv(loader_train,loader_test,model=nn1_fgsm,algo_adv=fgsm)
            
    
    
    ######################
    ## FGSM: Robuest Eval
    ######################
    
    clean_acc, robust_acc = benchmark(nn1,
                                      dataset='cifar10',
                                      threat_model='Linf',
                                      eps = 8/255
                                      )
    
    clean_acc_fgsm, robust_acc_fgsm = benchmark(nn1_fgsm,
                                      dataset='cifar10',
                                      threat_model='Linf',
                                      eps = 8/255
                                      )
    
    torch.save(nn1_fgsm, 'cifar10_nn1_fgsm.pkl')
    nn1_fgsm = torch.load('cifar10_nn1_fgsm.pkl')
    
main()
