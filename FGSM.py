# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 23:18:18 2022

@author: Ziming Huang
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim



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

    
device = torch.device("cpu")



######################
## Architecture
######################

nn1 = nn.Sequential(Flatten(), 
                    nn.Linear(3072,100), nn.ReLU(), 
                    nn.Linear(100,100), nn.ReLU(),
                    nn.Linear(100,100), nn.ReLU(),
                    nn.Linear(100,10)
                    ).to(device)



######################
## Standard Training
######################

optimizer = optim.SGD(nn1.parameters(), lr=0.25)
print("Initializing standard NN training; learning rate 0.25")
for e in range(10):
    train_err, train_loss = epoch(loader_train, nn1, optimizer, device, mode = "train")
    test_err, test_loss = epoch(loader_test, nn1, None, device, mode = "test")
    print("Epoch {e}: Training Error: {train_err} Test Error: {test_err}".format(e=e,train_err=train_err, test_err=test_err))
optimizer = optim.SGD(nn1.parameters(), lr=0.1)
print("learning rate 0.25")
for e in range(10):
    train_err, train_loss = epoch(loader_train, nn1, optimizer, device, mode = "train")
    test_err, test_loss = epoch(loader_test, nn1, None, device, mode = "test")
    print("Epoch {e}: Training Error: {train_err} Test Error: {test_err}".format(e=e,train_err=train_err, test_err=test_err))
optimizer = optim.SGD(nn1.parameters(), lr=0.05)
print("learning rate 0.05")
for e in range(10):
    train_err, train_loss = epoch(loader_train, nn1, optimizer, device, mode = "train")
    test_err, test_loss = epoch(loader_test, nn1, None, device, mode = "test")
    print("Epoch {e}: Training Error: {train_err} Test Error: {test_err}".format(e=e,train_err=train_err, test_err=test_err))



######################
## Standard Evaluation
######################

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
correct = total = 0

with torch.no_grad():
    for data in loader_test:
        images, labels = data
        outputs = nn1(images)
        _, one_class_pred = torch.max(outputs.data, 1)
        _, multi_class_pred = torch.max(outputs, 1)
        # collect the correct predictions for each class
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




######################
## FGSM
######################

def fgsm(model, X, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()


