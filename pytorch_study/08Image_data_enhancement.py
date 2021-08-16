#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   08Image_data_enhancement.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/8/16 16:00   SeafyLiang   1.0       08图像数据增强
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import os
import warnings
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
使用图像数据增强手段，提高06的图像数据分类模型准确度，方法就是常见的transforms图像数据增强手段

来源：【小白学习PyTorch教程】八、使用图像数据增强手段，提升CIFAR-10 数据集精确度
https://mp.weixin.qq.com/s?__biz=MzU2ODU3ODY0Nw==&mid=2247489361&idx=1&sn=745e4e8153e84e6917c5bf9f9f5e9a67&chksm=fc8a9069cbfd197f97845853886cca6de75a7358515b3de6fb83bc523be551e175be0f6efb4f&scene=178&cur_album_id=1904059205397839873#rd
"""

# 1. 加载数据集
# number of images in one forward and backward pass
batch_size = 128

# number of subprocesses used for data loading
# Normally do not use it if your os is windows
num_workers = 2

transform = transforms.Compose(
    # to tensor object
    [transforms.ToTensor(),
     #  mean = 0.5, std = 0.5
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = datasets.CIFAR10('./data/CIFAR10/',
                                 train=True,
                                 download=True,
                                 transform=transform)

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers)

val_dataset = datasets.CIFAR10('./data/CIFAR10',
                               train=True,
                               transform=transform)

val_loader = DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers)

test_dataset = datasets.CIFAR10('./data/CIFAR10',
                                train=False,
                                transform=transform)

test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=num_workers)

# declare classes in CIFAR10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# 之前的transform ’只是进行了缩放和归一，在这里添加RandomCrop和RandomHorizontalFlip
# define a transform to normalize the data

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # converting images to tensor
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # if the image dataset is black and white image, there can be just one number.
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


# 可视化具体的图像
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image


# obtain one batch of imges from train dataset
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()  # convert images to numpy for display

# plot the images in one batch with the corresponding labels
fig = plt.figure(figsize=(25, 4))

# display images
for idx in np.arange(10):
    ax = fig.add_subplot(1, 10, idx + 1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])


# 2. 建立常见的CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.main = nn.Sequential(
            # 3x32x32
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  # 3x32x32 (O = (N+2P-F/S)+1)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x16x16
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32x16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x8x8
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 3, padding=1),  # 64x8x8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128x4x4
            nn.BatchNorm2d(128),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        # Conv and Poolilng layers
        x = self.main(x)

        # Flatten before Fully Connected layers
        x = x.view(-1, 128 * 4 * 4)

        # Fully Connected Layer
        x = self.fc(x)
        return x


cnn = CNN().to(device)
print(cnn)
# torch.nn.CrossEntropyLoss对输出概率介于0和1之间的分类模型进行分类

# 3. 训练模型
# 超参数：Hyper Parameters
learning_rate = 0.001
train_losses = []
val_losses = []

# Loss function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)


# define train function that trains the model using a CIFAR10 dataset

def train(model, epoch, num_epochs):
    model.train()

    total_batch = len(train_dataset) // batch_size

    for i, (images, labels) in enumerate(train_loader):

        X = images.to(device)
        Y = labels.to(device)

        ### forward pass and loss calculation
        # forward pass
        pred = model(X)
        # c alculation  of loss value
        cost = criterion(pred, Y)

        ### backward pass and optimization
        # gradient initialization
        optimizer.zero_grad()
        # backward pass
        cost.backward()
        # parameter update
        optimizer.step()

        # training stats
        if (i + 1) % 100 == 0:
            print('Train, Epoch [%d/%d], lter [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, total_batch, np.average(train_losses)))

            train_losses.append(cost.item())


# def the validation function that validates the model using CIFAR10 dataset

def validation(model, epoch, num_epochs):
    model.eval()

    total_batch = len(val_dataset) // batch_size

    for i, (images, labels) in enumerate(val_loader):

        X = images.to(device)
        Y = labels.to(device)

        with torch.no_grad():
            pred = model(X)
            cost = criterion(pred, Y)

        if (i + 1) % 100 == 0:
            print("Validation, Epoch [%d/%d], lter [%d/%d], Loss: %.4f"
                  % (epoch + 1, num_epochs, i + 1, total_batch, np.average(val_losses)))

            val_losses.append(cost.item())


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(5, 5))
    plt.plot(train_losses, label='Train', alpha=0.5)
    plt.plot(val_losses, label='Validation', alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.legend()
    plt.grid(b=True)
    plt.title('CIFAR 10 Train/Val Losses Over Epoch')
    plt.show()


num_epochs = 20
for epoch in range(num_epochs):
    train(cnn, epoch, num_epochs)
    validation(cnn, epoch, num_epochs)
    torch.save(cnn.state_dict(), './data/Tutorial_3_CNN_Epoch_{}.pkl'.format(epoch + 1))

plot_losses(train_losses, val_losses)


# 4. 测试模型
def test(model):
    # declare that the model is about to evaluate
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_dataset:
            images = images.unsqueeze(0).to(device)

            # forward pass
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += 1
            correct += (predicted == labels).sum().item()

    print("Accuracy of Test Images: %f %%" % (100 * float(correct) / total))


# 经过图像数据增强。模型从60提升到了84。

# 5. 测试模型在哪些类上表现良好
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = cnn(images)

        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()

        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
