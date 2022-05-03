# -*- coding: utf-8 -*-
"""
Created on Sun May  1 10:28:56 2022

@author: Think
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import *
import torch.nn.functional as F



######################
## Architecture
######################

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, padding=1, stride=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
        self.fc2 = torch.nn.Linear(64, 10)
 
 
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 18 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    ######################
    ## Set Up
    ######################
    
    
    BATCH_SIZE = 100
    
    df_train = datasets.CIFAR10(root=r'/tmp/data',download=True, train=True, 
                                transform = transforms.ToTensor())
    df_test = datasets.CIFAR10(root=r'/tmp/data',download=True, train=False, 
                                transform = transforms.ToTensor())
    loader_train = DataLoader(df_train, batch_size = BATCH_SIZE, shuffle=True)
    loader_test = DataLoader(df_test, batch_size = BATCH_SIZE, shuffle=True)
        
    device = torch.device("cpu")
    
    
    
    ######################
    ## Architecture
    ######################
    
    nn1 = CNN()
    
    ######################
    ## Standard Training
    ######################
    optimizer = optim.SGD(nn1.parameters(), lr=0.1)
    nn1_standard_arr = train_standard(loader_train=loader_train, loader_test=loader_test,
                                      model=nn1, n_epoch=10, opt=optim.SGD, lr=0.1)
    
    ######################
    ## Standard Evaluation
    ######################
    
    eval_standard(loader_test,model=nn1)
    
    ######################
    ## Save Params
    ######################
    
    torch.save(nn1, 'cifar10_nn1_standard.pkl')
    nn1 = torch.load('cifar10_nn1_standard.pkl')
    
