# -*- coding: utf-8 -*-
"""
Created on Mon May  2 19:57:29 2022

@author: micke
"""

   
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from matplotlib import pyplot as plt

# ===========================================================================================
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=tf)
test_set = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=tf)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'truck', 'ship')

n_training_sample = 50000
train_sample = SubsetRandomSampler(np.arange(n_training_sample, dtype=np.int64))
# Validation 取训练集中的[20000,20000+5000]作为验证集
# n_validation_sample = 5000
# validation_sample = SubsetRandomSampler(np.arange(n_training_sample, n_training_sample + n_validation_sample,dtype=np.int64))
# Testing 共10000,取前5000作为测试集
n_test_sample = 10000
test_sample = SubsetRandomSampler(np.arange(n_test_sample, dtype=np.int64))


# 开启shuffle就等于全集使用SubsetRandomSampler，都是随机采样,num_workers代表多线程加载数据,Windows上不能用(必须0),Linux可用
train_batch_size = 100
test_batch_size = 4
train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, sampler=train_sample, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, sampler=test_sample, num_workers=0)
#val_loader = torch.utils.data.DataLoader(train_set, batch_size=500, sampler=validation_sample, num_workers=0)

# ================================================================================================

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding = 1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding = 1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.globalavgpool = nn.AvgPool2d(8, 8)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout10 = nn.Dropout(0.1)
        self.fc = nn.Linear(256, 10)
 
    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn1(F.relu(self.conv2(x)))
        x = self.maxpool(x)
        x = self.dropout10(x)
        x = self.bn2(F.relu(self.conv3(x)))
        x = self.bn2(F.relu(self.conv4(x)))
        x = self.avgpool(x)
        x = self.dropout10(x)
        x = self.bn3(F.relu(self.conv5(x)))
        x = self.bn3(F.relu(self.conv6(x)))
        x = self.globalavgpool(x)
        x = self.dropout50(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
cnn = CNN()
# 如有GPU则自动使用GPU计算
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
cnn.to(device)
# ===================================================================================
# 损失函数
loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
# 优化器
optimizer = optim.Adam(cnn.parameters(), lr=0.001)     # Adam 优化算法是随机梯度下降算法的扩展式
 
# ==========================================================================================
def trainNet(epoch):
    print('Epoch {}'.format(epoch))
    # 加载数据集上边的方法解释了获取训练数据
    training_start_time = time.time()   # 开始时间，为了后边统计一个训练花费时间
    #循环训练 n_epochs是5,也就是重复扫 五遍样本数据,CIFAR10数据集将50000条训练数据分为了五个batch，所以这个地方不要有疑惑
    start_time = time.time()
    train_loss = 0
    for step,(x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
        # forward:前向传播
        outputs = cnn(x_batch)  #
        loss = loss_func(outputs, y_batch)
        train_loss += loss.item()
        # 在一个epoch里。每十组batchsize大小的数据输出一次结果，即以batch_size大小的数据为一组，到第10组，20组，30组...的时候输出
        if step % (len(train_loader)/100) == 0:
            print("epoch{}, {:d}% \t loss:{:.6f} took:{:.2f}s".format(epoch, int(100 * (step) / len(train_loader)),loss.item(), time.time()-start_time))
            start_time = time.time()
        #backward:后向传播
        optimizer.zero_grad()  # 将所有的梯度置零，原因是防止每次backward的时候梯度会累加
        loss.backward()  # 根据反向传播更新所有的参数
        optimizer.step()
    print("Training loss={}, took {:.2f}s".format(train_loss/(len(train_loader)),time.time() - training_start_time))  # 所有的Epoch结束，也就是训练结束，计算花费的时间

#使用以下方法保存和恢复网络参数
#torch.save(cnn, 'cifar10.pkl')
#cnn = torch.load('cifar10.pkl')

def test():
    correct = 0
    test_loss = 0
    cnn.eval()
    with torch.no_grad():
        for data in test_loader:
            # Forward pass
            x_batch,y_batch = data
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            out = cnn(x_batch)
            loss = loss_func(out, y_batch)
            predicted = torch.max(out, 1)[1]
            correct += (predicted == y_batch).sum().item()
            test_loss += loss.item()
        print("test loss = {:.2f}, Accuracy={:.6f}".format(test_loss / len(test_loader),correct/len(test_loader)/test_batch_size))  # 求验证集的平均损失是多少
    


# 执行整个训练过程
for epoch in range(1,11):
    trainNet(epoch)
    test()

# 统计每类的分类准确率
cnn.eval()
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_loader:
        x_batch, y_batch = data
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        out = cnn(x_batch)
        predicted = torch.max(out, 1)[1]
        c = (predicted == y_batch).squeeze()
        # 
        for i in range(test_batch_size):
            label = y_batch[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))