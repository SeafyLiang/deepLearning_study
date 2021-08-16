#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   05datasets_dataloader.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/8/16 14:57   SeafyLiang   1.0        05pytorch使用datasets和dataloader自定义数据集
"""
import torch
import matplotlib.pyplot as plt
# torchvision 的torch计算机视觉模块
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import random

"""
来源：【小白学习PyTorch教程】五、在 PyTorch 中使用 Datasets 和 DataLoader 自定义数据
https://mp.weixin.qq.com/s?__biz=MzU2ODU3ODY0Nw==&mid=2247489272&idx=1&sn=24d778103331464022543413387aabe0&chksm=fc8a91c0cbfd18d64208d332fe964d60aa244666e0dac86589a2f56ad3447466bfebb2c4d767&scene=178&cur_album_id=1904059205397839873#rd
"""
"""
有时候，在处理大数据集时，一次将整个数据加载到内存中变得非常难。

因此，唯一的方法是将数据分批加载到内存中进行处理，这需要编写额外的代码来执行此操作。对此，PyTorch 已经提供了 Dataloader 功能。
"""

# 一、DataLoader
# 下面显示了 PyTorch 库中DataLoader函数的语法及其参数信息。
"""
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
几个重要参数

dataset：必须首先使用数据集构造 DataLoader 类。
Shuffle ：是否重新整理数据。
Sampler ：指的是可选的 torch.utils.data.Sampler 类实例。采样器定义了检索样本的策略，顺序或随机或任何其他方式。使用采样器时应将 Shuffle 设置为 false。
Batch_Sampler ：批处理级别。
num_workers ：加载数据所需的子进程数。
collate_fn ：将样本整理成批次。Torch 中可以进行自定义整理。
"""


# # 1.1 加载内置MNIST数据集
# # MNIST 是一个著名的包含手写数字的数据集。下面介绍如何使用DataLoader功能处理 PyTorch 的内置 MNIST 数据集。
# # 对于 MNIST 数据集，下面使用了归一化技术。
# # ToTensor()能够把灰度范围从0-255变换到0-1之间。
# transform = transforms.Compose([transforms.ToTensor()])
# # 加载所需的数据集。使用 PyTorchDataLoader通过给定 batch_size = 64来加载数据。shuffle=True打乱数据。
# trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# # 为了获取数据集的所有图像，一般使用iter函数和数据加载器DataLoader。
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# print(images.shape)
# print(labels.shape)
# plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')


# 1.2 自定义数据集
# 创建一个包含 1000 个随机数的自定义数据集
class SampleDataset(Dataset):
    def __init__(self, r1, r2):
        randomlist = []
        for i in range(120):
            n = random.randint(r1, r2)
            randomlist.append(n)
        self.samples = randomlist

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


dataset = SampleDataset(1, 100)
print(dataset[100:120])
# 最后，将在自定义数据集上使用 dataloader 函数。将 batch_size 设为 12，并且还启用了num_workers =2 的并行多进程数据加载。
loader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=2)
for i, batch in enumerate(loader):
    print(i, batch)
