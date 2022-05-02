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




######################
## PGD: Training
######################


nn1_pgd = nn.Sequential(Flatten(), 
                    nn.Linear(3072,100), nn.ReLU(), 
                    nn.Linear(100,100), nn.ReLU(),
                    nn.Linear(100,100), nn.ReLU(),
                    nn.Linear(100,10)
                    ).to(device)

optimizer = optim.SGD(nn1_pgd.parameters(), lr=0.2)
print("Initializing standard NN training; learning rate 0.2")
for e in range(15):
    train_err, train_loss = epoch_adv(loader_train, nn1_pgd, optimizer, device, mode = "train",
                                      algo_adv = pgd,
                                      epsilon = 8/255,
                                      alpha = 0.1, n = 50)
    test_err, test_loss = epoch(loader_test, nn1_pgd, None, device, mode = "test")
    test_err_adv, test_loss_adv = epoch_adv(loader_test, nn1_pgd, optimizer, device, mode = "train",
                                      algo_adv = pgd,
                                      epsilon = 8/255,
                                      alpha = 0.1, n = 50)
    print('''epoch_adv {e}: Training Error: {train_err} 
          Test Error: {test_err} 
          Robust Error: {test_err_adv}''' 
              .format(e=e,train_err=train_err, test_err=test_err, test_err_adv= test_err_adv))


######################
## PGD: Robuest Eval
######################

from robustbench.eval import benchmark
clean_acc, robust_acc = benchmark(nn1,
                                  dataset='cifar10',
                                  threat_model='Linf',
                                  eps = 8/255
                                  )

clean_acc_fgsm, robust_acc_fgsm = benchmark(nn1_pgd,
                                  dataset='cifar10',
                                  threat_model='Linf',
                                  eps = 8/255
                                  )
