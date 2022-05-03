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

# class MyEnsemble(nn.Module):
#     def __init__(self, modelA, modelB, modelC):
#         super(MyEnsemble, self).__init__()
#         self.modelA = modelA
#         self.modelB = modelB
#         self.modelC = modelC
#         self.classifier = nn.Linear(30,10)
#     def forward(self, x):
#         x1 = self.modelA(x)
#         x1 = x1.view(x1.size(0), -1)
#         x2 = self.modelB(x)
#         x2 = x2.view(x2.size(0), -1)
#         x = torch.cat((x1, x2), dim=1)
#         x = self.classifier(F.relu(x))
#         return x
    
    
class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, modelC):
        super(MyEnsemble, self).__init__()
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
    
nn1_fgsm = torch.load('cifar10_nn1_fgsm_2.pkl')
nn1_pgd = torch.load('cifar10_nn1_pgd.pkl')
nn1_lireg = torch.load('cifar10_lip_reg.pkl')
ens1 = MyEnsemble(nn1_fgsm,nn1_pgd,nn1_lireg)

eval_standard(loader_test,ens1)

from robustbench.eval import benchmark
clean_acc_ens1, robust_acc_ens1 = benchmark(ens1,
                                  dataset='cifar10',
                                  threat_model='Linf',
                                  eps = 8/255
                                  )
