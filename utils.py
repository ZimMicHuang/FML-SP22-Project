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




######################
## Training
######################

def train_standard(loader_train,loader_test, model, n_epoch, opt, lr, device=torch.device("cpu")):
    optimizer = opt(model.parameters(), lr=lr) 
    print("Initializing standard NN training; learning rate {lr}".format(lr=lr))
    train_arr = []
    for e in range(n_epoch):
        train_err, train_loss = epoch(loader_train, model, optimizer, device, mode = "train")
        test_err, test_loss = epoch(loader_test, model, None, device, mode = "test")
        print("Epoch {e}: Training Error: {train_err} Test Error: {test_err}".format(e=e,train_err=train_err, test_err=test_err))
        train_arr.append([train_err, train_loss, test_err, test_loss])
    return train_arr

def train_adv(loader_train,loader_test, model, 
              algo_adv, epsilon=8/255,
              n_epoch = 10, opt = optim.SGD, lr=0.1, 
              device=torch.device("cpu"),**kwargs): #kwargs: alpha, n_step (inner)
         
    train_arr = []
    optimizer = opt(model.parameters(), lr=lr)
    print("Initializing standard NN training; learning rate {lr}".format(lr=lr))
    for e in range(n_epoch):
        train_err, train_loss = epoch_adv(loader_train, model, optimizer, device, 
                                          mode = "train",  
                                          algo_adv = algo_adv, 
                                          epsilon = epsilon) #kwargs:
        test_err, test_loss = epoch(loader_test, model, None, device, mode = "test")
        test_err_adv, test_loss_adv = epoch_adv(loader_test, model, optimizer, device, 
                                          mode = "test",
                                          algo_adv = algo_adv,
                                          epsilon = epsilon)
        print('''epoch_adv {e}: Training Error: {train_err} 
              Test Error: {test_err} 
              Robust Error: {test_err_adv}''' 
                  .format(e=e,train_err=train_err, test_err=test_err, test_err_adv= test_err_adv))
        train_arr.append([train_err, train_loss,test_err, test_loss,test_err_adv, test_loss_adv])
        
    return train_arr


######################
## Standard Evaluation
######################

def eval_standard(loader_test,model):
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    correct = total = 0
    
    with torch.no_grad():
        for data in loader_test:
            images, labels = data
            outputs = model(images)
            _, one_class_pred = torch.max(outputs.data, 1)
            _, multi_class_pred = torch.max(outputs, 1)
            for label, pred in zip(labels, multi_class_pred):
                if label == pred:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
            total += labels.size(0)
            correct += (one_class_pred == labels).sum().item()
    
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    print('Accuracy of the network on the 10000 test images: ' + str(correct / total))
    

