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
        self.classifier = nn.Linear(20,10)
    def forward(self, x):
        x1 = self.modelA(x)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x)
        x2 = x2.view(x2.size(0), -1)
        return (x1 + x2) / 2
    
    
ens1 = MyEnsemble(nn1_fgsm,nn1_pgd)
ens1.parameters()

# does not work
from robustbench.eval import benchmark
clean_acc_ens1, robust_acc_ens1 = benchmark(ens1,
                                  dataset='cifar10',
                                  threat_model='Linf',
                                  eps = 8/255
                                  )

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
correct = total = 0

with torch.no_grad():
    for data in loader_test:
        images, labels = data
        outputs = ens1(images) 
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