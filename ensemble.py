# -*- coding: utf-8 -*-
"""
Created on Mon May  2 17:52:47 2022

@author: micke
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from utils import *
from nn1 import *

class MyEnsembleStacking(nn.Module):
    def __init__(self, modelA, modelB, modelC):
        super(MyEnsembleStacking, self).__init__()
        self.modelA = modelA
        for param in modelA.parameters():
            param.requires_grad = False
        self.modelB = modelB
        for param in modelB.parameters():
            param.requires_grad = False
        self.modelC = modelC
        for param in modelC.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(30,10)
    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x3 = self.modelC(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.classifier(F.relu(x))
        return x
    
    
class MyEnsembleAverage(nn.Module):
    def __init__(self, modelA, modelB, modelC):
        super(MyEnsembleAverage, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
    def forward(self, x):
        x1 = self.modelA(x)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x)
        x2 = x2.view(x2.size(0), -1)
        x3 = self.modelB(x)
        x3 = x3.view(x3.size(0), -1)
        return (x1+x2+x3)/3
    



BATCH_SIZE = 100

df_train = datasets.CIFAR10(root=r'/tmp/data',download=True, train=True, 
                            transform = transforms.ToTensor())
df_test = datasets.CIFAR10(root=r'/tmp/data',download=True, train=False, 
                            transform = transforms.ToTensor())
loader_train = DataLoader(df_train, batch_size = BATCH_SIZE, shuffle=True)
loader_test = DataLoader(df_test, batch_size = BATCH_SIZE, shuffle=True)
    
device = torch.device("cpu")

nn1_fgsm = torch.load('cifar10_nn1_fgsm_2.pkl')
nn1_pgd = torch.load('cifar10_nn1_pgd.pkl')
nn1_lireg = torch.load('cifar10_lip_reg.pkl')
# ens1 = MyEnsembleAverage(nn1_fgsm,nn1_pgd,nn1_lireg)
for model in (nn1_fgsm,nn1_pgd,nn1_lireg):
    for param in model.parameters():
        param.requires_grad = False

ens2 = MyEnsembleStacking(nn1_fgsm,nn1_pgd,nn1_lireg)
for param in ens2.parameters():
    print(param)
ens_arr = train_standard(loader_train=loader_train, loader_test=loader_test,
                                  model=ens2, n_epoch=5, opt=optim.SGD, lr=0.1)

ens2.parameters()



optim.SGD(nn1_pgd.parameters(), lr=0.1)

eval_standard(loader_test,ens2)

from robustbench.eval import benchmark
clean_acc_ens1, robust_acc_ens1 = benchmark(ens1,
                                  dataset='cifar10',
                                  threat_model='Linf',
                                  eps = 8/255
                                  )
clean_acc_ens2, robust_acc_ens2 = benchmark(ens2,
                                  dataset='cifar10',
                                  threat_model='Linf',
                                  eps = 8/255
                                  )