from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from trades import trades_loss

# parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
# parser.add_argument('--batch-size', type=int, default=128, metavar='N',
#                     help='input batch size for training (default: 128)')
# parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
#                     help='input batch size for testing (default: 128)')
# parser.add_argument('--epochs', type=int, default=76, metavar='N',
#                     help='number of epochs to train')
# parser.add_argument('--weight-decay', '--wd', default=2e-4,
#                     type=float, metavar='W')
# parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
#                     help='learning rate')
# parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
#                     help='SGD momentum')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training')
# parser.add_argument('--epsilon', default=0.031,
#                     help='perturbation')
# parser.add_argument('--num-steps', default=10,
#                     help='perturb number of steps')
# parser.add_argument('--step-size', default=0.007,
#                     help='perturb step size')
# parser.add_argument('--beta', default=6.0,
#                     help='regularization, i.e., 1/lambda in TRADES')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=100, metavar='N',
#                     help='how many batches to wait before logging training status')
# parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
#                     help='directory of model for saving checkpoint')
# parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
#                     help='save frequency')

# args = parser.parse_args()



def train(model, device, train_loader, optimizer, epoch, lr, epsilon, num_steps, beta):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=lr,
                           epsilon=epsilon,
                           perturb_steps=num_steps,
                           beta=beta)
        loss.backward()
        optimizer.step()

        # print progress
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy




BATCH_SIZE = 100

df_train = datasets.CIFAR10(root=r'/tmp/data',download=True, train=True, 
                            transform = transforms.ToTensor())
df_test = datasets.CIFAR10(root=r'/tmp/data',download=True, train=False, 
                            transform = transforms.ToTensor())
loader_train = DataLoader(df_train, batch_size = BATCH_SIZE, shuffle=True)
loader_test = DataLoader(df_test, batch_size = BATCH_SIZE, shuffle=True)
    
device = torch.device("cpu")

model = torch.load('cifar10_nn1_standard.pkl')
optimizer = optim.SGD(model.parameters(), lr=0.1)

train_arr_lipreg = []
for epoch in range(1, 10):
    
    # adversarial training
    train( model, device, loader_train, optimizer, epoch, lr=0.1, epsilon=8/255, num_steps=10, beta=3)

    # evaluation on natural examples
    print('================================================================')
    train_loss, training_accuracy = eval_train(model, device, loader_train)
    test_loss, test_accuracy = eval_test(model, device, loader_test)
    print('================================================================')
    
    train_arr_lipreg.append([training_accuracy,train_loss,test_accuracy,test_loss])

    # # save checkpoint
    # if epoch % args.save_freq == 0:
    #     torch.save(model.state_dict(),
    #                os.path.join(model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))
    #     torch.save(optimizer.state_dict(),
    #                os.path.join(model_dir, 'opt-wideres-checkpoint_epoch{}.tar'.format(epoch)))

torch.save(model, 'cifar10_lip_reg.pkl')

from robustbench.eval import benchmark
clean_acc_lipreg, robust_acc_lipreg = benchmark(model,
                                  dataset='cifar10',
                                  threat_model='Linf',
                                  eps = 8/255
                                  )

