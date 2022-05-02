# -*- coding: utf-8 -*-
"""
Created on Sun May  1 10:29:55 2022

@author: Think
"""
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from nn1 import *

######################
## PGM: Attack
######################

def pgd(model, X, y, eps, alpha, n):
    """ Construct pgd adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for i in range(n):        
        # gradient
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        # projection/clipping
        delta.data = (delta + alpha*X.shape[0]*delta.grad.data).clamp(-eps,eps)
        delta.grad.zero_()
    return delta.detach()


print(
      "PGD: Attacked error rate:", 
      epoch_adv(loader_test, nn1, optimizer, device,
                mode=None,
                algo_adv=pgd, 
                epsilon = 0.5, alpha = 0.1, n = 50)[0]
      )
