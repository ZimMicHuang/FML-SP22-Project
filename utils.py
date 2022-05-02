# -*- coding: utf-8 -*-
"""
Created on Sun May  1 10:53:34 2022

@author: Think
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)    


def epoch(dataloader, hypothesis, optimizer, device, mode):
    loss = error = 0
    for train_x, train_y in dataloader:
        train_x,train_y = train_x.to(device), train_y.to(device)
        pred = hypothesis(train_x)
        loss_value = nn.CrossEntropyLoss()(pred,train_y)
        if mode == "train":
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
        error += (pred.max(dim=1)[1] != train_y).sum().item()
        loss += loss_value.item() * train_x.shape[0]
    return error / len(dataloader.dataset), loss / len(dataloader.dataset)


def epoch_adv(dataloader, hypothesis, optimizer, device, mode, 
              algo_adv, epsilon, **kwargs):
    loss = error = 0
    for train_x, train_y in dataloader:
        train_x,train_y = train_x.to(device), train_y.to(device)
        pred = hypothesis(train_x + algo_adv(hypothesis, train_x, train_y, epsilon, **kwargs))
        loss_value = nn.CrossEntropyLoss()(pred,train_y)
        if mode=="train":
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
        error += (pred.max(dim=1)[1] != train_y).sum().item()
        loss += loss_value.item() * train_x.shape[0]
    return error / len(dataloader.dataset), loss / len(dataloader.dataset)
