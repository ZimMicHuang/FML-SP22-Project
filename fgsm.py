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
from nn1 import *
from utils import *


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
## FGSM: Attack
######################

def fgsm(model, X, y, epsilon=1e-8):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()


print("FGSM: Attacked error rate:", epoch_adv(loader_test, nn1, device, fgsm, epsilon = 0.5)[0])


######################
## FGSM: Training
######################


