#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   03neural_network_model.py
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/8/16 14:30   SeafyLiang   1.0       03pytorch实现神经网络模型
"""
import torch
from torch import nn
from torch import optim
"""
来源：【小白学习PyTorch教程】三、Pytorch中的NN模块并实现第一个神经网络模型
https://mp.weixin.qq.com/s?__biz=MzU2ODU3ODY0Nw==&mid=2247489240&idx=1&sn=31c2b46b3119220031a72c367b24e6ff&chksm=fc8a91e0cbfd18f62f956bd14d23385bdff0dbc933d5a9898923b7404b1c374991af0cf4f5ca&scene=178&cur_album_id=1904059205397839873#rd
"""
# 一、nn.Linear
# nn.Linear是创建一个线性层。这里需要将输入和输出维度作为参数传递。

# linear接受nx10的输入并返回nx2的输出
linear = nn.Linear(10, 2)
example_input = torch.randn(3, 10)
example_output = linear(example_input)
print(example_input)
print(example_output)
# 输出如下
# tensor([[-0.1249, -0.8002],
#         [-1.0945, -0.2297],
#         [-0.3558,  0.8439]], grad_fn=<AddmmBackward>)

# 二、nn.Relu
# nn.Relu对线性的给定输出执行 relu 激活函数操作。

relu = nn.ReLU()
relu_output = relu(example_output)
print(relu_output)

# 输出如下
# tensor([[0.0000, 0.0000],
#         [0.0000, 0.0000],
#         [0.0000, 0.8439]], grad_fn=<ReluBackward0>)


# 三、nn.BatchNorm1d
# nn.BatchNorm1d是一种标准化技术，用于在不同批次的输入中保持一致的均值和标准偏差。
batchnorm = nn.BatchNorm1d(2)
batchnorm_output = batchnorm(relu_output)
print(batchnorm_output)

# 输出如下
# tensor([[ 0.0000, -0.7071],
#         [ 0.0000, -0.7071],
#         [ 0.0000,  1.4142]], grad_fn=<NativeBatchNormBackward>)

# 四、nn.Sequential
# nn.Sequential一次性创建一系列操作。和tensorflow中的Sequential完全一样。
mlp_layer = nn.Sequential(
    nn.Linear(5, 2),
    nn.BatchNorm1d(2),
    nn.ReLU()
)
test_example = torch.randn(5, 5) + 1
print("input: ")
print(test_example)
print("output: ")
print(mlp_layer(test_example))

# 输出如下
# input:
# tensor([[ 1.4617,  1.2446,  1.4919,  1.5978, -0.3410],
#         [-0.2819,  0.5567,  1.0113,  1.8053, -0.0833],
#         [ 0.2830,  1.0857,  1.2258,  2.6602,  0.1339],
#         [ 0.8682,  0.9344,  1.3715,  0.0279,  1.8011],
#         [ 0.6172,  1.1414,  0.6030,  0.3876,  1.3653]])
# output:
# tensor([[0.0000, 0.0000],
#         [0.0000, 1.3722],
#         [0.0000, 0.8861],
#         [1.0895, 0.0000],
#         [1.3047, 0.0000]], grad_fn=<ReluBackward0>)

# 在上面的模型中缺少了优化器，我们无法得到对应损失。
adam_opt = optim.Adam(mlp_layer.parameters(), lr=1e-1)
# 这里lr表示学习率，1e-1表示0.1
train_example = torch.randn(100, 5) + 1
adam_opt.zero_grad()
# 我们将使用1减去平均值，作为简单损失函数
cur_loss = torch.abs(1 - mlp_layer(train_example)).mean()
cur_loss.backward()
# 更新参数
adam_opt.step()
print(cur_loss.data)
# 输出如下
# tensor(0.7467)

# 虽然上面只是用了一个epoch，训练线性模型得到loss为0.8140，上面就是NN模型建立model的整个流程，

# 五、第一个神经网络模型
"""
下面实现第一个分类神经网络，其中一个隐藏层用于开发单个输出单元。
"""
# 定义所有层和批量大小以开始执行神经网络，如下所示
n_in, n_h, n_out, batch_size = 10, 5, 1, 10
# 由于神经网络包括输入数据的组合以获得相应的输出数据，我们将遵循以下相同的程序 -
x = torch.randn(batch_size, n_in)
y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])
# 创建顺序模型。使用下面代码，创建一个顺序模型 -
model = nn.Sequential(nn.Linear(n_in, n_h),
                      nn.ReLU(),
                      nn.Linear(n_h, n_out),
                      nn.Sigmoid())
# 借助梯度下降优化器构建损失函数，如下所示 -
# 构造损失函数
criterion = torch.nn.MSELoss()
# 构造优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# 使用给定代码行的迭代循环实现梯度下降模型 -
# 梯度下降
for epoch in range(50):
    # 正向传递：通过将x传递给模型来计算预测的y
    y_pred = model(x)

    # 计算loss
    loss = criterion(y_pred, y)

    # 梯度清0
    optimizer.zero_grad()

    # 反向传播，求解梯度
    loss.backward()

    # 更新模型参数
    optimizer.step()
    if epoch % 10 == 0:
        print('epoch: ', epoch, ' loss: ', loss.item())
