#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   01basic_opt.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/8/16 13:22   SeafyLiang   1.0          01Pytorch基本操作
"""
import torch
import numpy as np

"""
来源：【小白学习PyTorch教程】一、PyTorch基本操作
https://mp.weixin.qq.com/s?__biz=MzU2ODU3ODY0Nw==&mid=2247489209&idx=1&sn=e215e014d5a5b82aeb55afcfa62cc270&chksm=fc8a9181cbfd18975d457ef692a34f8c76e404c05f587e0e54a2f555bc24fbac578ebee66c97&scene=178&cur_album_id=1904059205397839873#rd
"""
# 一、张量
# 类似于NumPy 的n 维数组，此外张量也可以在 GPU 上使用以加速计算。
print("\n一、张量基本操作")
x = torch.empty(5, 3)  # 构建一个 5×3 的未初始化矩阵
print('5×3 的未初始化矩阵:', x)
x = torch.rand(5, 3)  # 构造一个随机初始化的矩阵
print('随机初始化的矩阵:', x)
x = torch.tensor([5.5, 3])  # 直接从数据构造张量
print('从数据构造张量:', x)
x = torch.LongTensor(3, 4)  # 创建一个统一的长张量
print('统一的长张量:', x)
x = torch.FloatTensor(3, 4)  # 浮动张量
print('浮动张量:', x)
x = torch.arange(10, dtype=torch.float)  # 在范围内创建张量
print('在范围内创建张量:', x)
x = x.view(2, 5)  # 重塑张量
print('重塑张量:', x)
x = x.view(5, -1)  # 根据张量的大小自动识别维度
print('根据张量的大小自动识别维度:', x)
x1 = torch.tensor([[1., 2., 3.], [4., 5., 6.]])  # 改变张量轴
print("x1: \n", x1)
print("\nx1.shape: \n", x1.shape)
print("\nx1.view(3, -1): \n", x1.view(3, -1))
print("\nx1.permute(1, 0): \n", x1.permute(1, 0))

# 二、PyTorch 和 NumPy的转换
print("\n二、PyTorch 和 NumPy的转换")
a = torch.ones(5)  # 将 Torch 张量转换为 NumPy 数组
print('Torch 张量:', a)
b = a.numpy()
print('将 Torch 张量转换为 NumPy 数组:', b)

a = np.ones(5)  # 将 NumPy 数组转换为 Torch 张量
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print('NumPy 数组：', a)
print('将 NumPy 数组转换为 Torch 张量：', b)

# 三、AutoGrad
# 该autograd包提供自动求导为上张量的所有操作

# 如果requires_grad=True，则 Tensor 对象会跟踪它是如何创建的
x = torch.tensor([1., 2., 3.], requires_grad=True)
print('x: ', x)
y = torch.tensor([10., 20., 30.], requires_grad=True)
print('y: ', y)
z = x + y
print('\nz = x + y')
print('z:', z)
# 因为requires_grad=True，z知道它是通过增加两个张量的产生z = x + y。
s = z.sum()
print(s)
# s是由它的数字总和创建的。当我们调用.backward()，反向传播从s开始运行。然后可以计算梯度。
s.backward()
print('x.grad: ', x.grad)
print('y.grad: ', y.grad)
# 下面例子是计算log(x)的导数为1 / x
x = torch.tensor([0.5, 0.75], requires_grad=True)
# 1 / x
y = torch.log(x[0] * x[1])
y.backward()
