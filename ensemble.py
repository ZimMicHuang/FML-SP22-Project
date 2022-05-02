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

class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
    def forward(self, x):
        x1 = self.modelA(x)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(F.relu(x))
        return x
    
ens1 = MyEnsemble(nn1_pgd,nn1_fgsm)


# does not work
from robustbench.eval import benchmark
clean_acc_ens1, robust_acc_ens1 = benchmark(ens1,
                                  dataset='cifar10',
                                  threat_model='Linf',
                                  eps = 8/255
                                  )