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


print("FGSM: Attacked error rate:", 
      epoch_adv(loader_test, nn1, optimizer,  device, 
                mode = None,
                algo_adv = fgsm,
                epsilon = 8/255)[0])


######################
## FGSM: Training
######################


nn1_fgsm = nn.Sequential(Flatten(), 
                    nn.Linear(3072,100), nn.ReLU(), 
                    nn.Linear(100,100), nn.ReLU(),
                    nn.Linear(100,100), nn.ReLU(),
                    nn.Linear(100,10)
                    ).to(device)

optimizer = optim.SGD(nn1_fgsm.parameters(), lr=0.2)
print("Initializing standard NN training; learning rate 0.2")
for e in range(15):
    train_err, train_loss = epoch_adv(loader_train, nn1_fgsm, optimizer, device, mode = "train",
                                      algo_adv = fgsm,
                                      epsilon = 8/255)
    test_err, test_loss = epoch(loader_test, nn1_fgsm, None, device, mode = "test")
    test_err_adv, test_loss_adv = epoch_adv(loader_test, nn1_fgsm, optimizer, device, mode = "train",
                                      algo_adv = fgsm,
                                      epsilon = 8/255)
    print('''epoch_adv {e}: Training Error: {train_err} 
          Test Error: {test_err} 
          Robust Error: {test_err_adv}''' 
              .format(e=e,train_err=train_err, test_err=test_err, test_err_adv= test_err_adv))
optimizer = optim.SGD(nn1_fgsm.parameters(), lr=0.1)
print("Initializing standard NN training; learning rate 0.1")
for e in range(15):
    train_err, train_loss = epoch_adv(loader_train, nn1_fgsm, optimizer, device, mode = "train",
                                      algo_adv = fgsm,
                                      epsilon = 8/255)
    test_err, test_loss = epoch(loader_test, nn1_fgsm, None, device, mode = "test")
    test_err_adv, test_loss_adv = epoch_adv(loader_test, nn1_fgsm, optimizer, device, mode = "train",
                                      algo_adv = fgsm,
                                      epsilon = 8/255)
    print('''epoch_adv {e}: Training Error: {train_err} 
          Test Error: {test_err} 
          Robust Error: {test_err_adv}''' 
              .format(e=e,train_err=train_err, test_err=test_err, test_err_adv= test_err_adv))
optimizer = optim.SGD(nn1_fgsm.parameters(), lr=0.025)
print("Initializing standard NN training; learning rate 0.05")
for e in range(15):
    train_err, train_loss = epoch_adv(loader_train, nn1_fgsm, optimizer, device, mode = "train",
                                      algo_adv = fgsm,
                                      epsilon = 8/255)
    test_err, test_loss = epoch(loader_test, nn1_fgsm, None, device, mode = "test")
    test_err_adv, test_loss_adv = epoch_adv(loader_test, nn1_fgsm, optimizer, device, mode = "train",
                                      algo_adv = fgsm,
                                      epsilon = 8/255)
    print('''epoch_adv {e}: Training Error: {train_err} 
          Test Error: {test_err} 
          Robust Error: {test_err_adv}''' 
              .format(e=e,train_err=train_err, test_err=test_err, test_err_adv= test_err_adv))


######################
## FGSM: Robuest Eval
######################

from robustbench.eval import benchmark
clean_acc, robust_acc = benchmark(nn1,
                                  dataset='cifar10',
                                  threat_model='Linf',
                                  eps = 8/255
                                  )

clean_acc_fgsm, robust_acc_fgsm = benchmark(nn1_fgsm,
                                  dataset='cifar10',
                                  threat_model='Linf',
                                  eps = 8/255
                                  )
