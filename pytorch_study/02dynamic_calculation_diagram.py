#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   02dynamic_calculation_diagram.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/8/16 14:22   SeafyLiang   1.0          02动态计算图与GPU支持操作
"""
import torch

"""
【小白学习PyTorch教程】二、动态计算图和GPU支持操作
来源：https://mp.weixin.qq.com/s?__biz=MzU2ODU3ODY0Nw==&mid=2247489224&idx=1&sn=a8323507708c1f81b94f2ff4c87829ec&chksm=fc8a91f0cbfd18e65f56b5c2431611cc3f56d1887c163cd396114888e055ed259c779af4cd34&scene=178&cur_album_id=1904059205397839873#rd
"""

# 一、动态计算图
"""
在深度学习中使用 PyTorch 的主要原因之一，是我们可以自动获得定义的函数的梯度/导数。
当我们操作我们的输入时，会自动创建一个计算图。该图显示了如何从输入到输出的动态计算过程。
"""
# 只有浮动张量有梯度
x = torch.arange(1, 4, dtype=torch.float32, requires_grad=True)
print("X", x)

# X tensor([1., 2., 3.], requires_grad=True)
a = x + 2
b = a ** 2
c = b + 3
y = c.mean()
print("Y", y)
# Y tensor(19.6667, grad_fn=<MeanBackward0>)
# 我们可以通过backward()在最后一个输出上调用函数来对计算图执行反向传播，这样可以，计算了每个具有属性的张量的梯度requires_grad=True：
y.backward()
# 最后打印x.grad就可以查看对应梯度。
print("x.grad:", x.grad)

# 二、GPU支持操作
"""
「CPU 与 GPU的区别」

CPU	            GPU
中央处理器	    图形处理单元
几个核心	        多核
低延迟	        高吞吐量
适合串行处理	    适合并行处理
可以一次做一些操作	可以同时进行数千次操作

PyTorch 使用GPU，需要搭建NVIDIA 的CUDA和cuDNN。
"""
# 检查是否有可用的GPU
gpu_avail = torch.cuda.is_available()
print("Is the GPU available? %s" % str(gpu_avail))


# 创建一个张量并将其推送到GPU设备
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)
x = x.to(device)
print("X", x)
# cuda 旁边的零表示这是计算机上的第0个 GPU 设备。因此，PyTorch 还支持多 GPU 系统，
# Device cuda
# X tensor([1., 1., 1.], device='cuda:0')

