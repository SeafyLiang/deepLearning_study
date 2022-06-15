#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   10LSTM_reviews_emotion.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/6/15 22:06   SeafyLiang   1.0      10LSTM预测评论情感
"""
"""
来源：【小白学习PyTorch教程】十、基于大型电影评论数据集训练第一个LSTM模型
https://mp.weixin.qq.com/s?__biz=MzU2ODU3ODY0Nw==&mid=2247489418&idx=1&sn=dd61a60f00d43f8cc2bc7b32596e47e1&chksm=fc8a90b2cbfd19a423b62dff117a49b4a21eef58069786b2cce77bc4b1569a60ec92aaf1742e&scene=178&cur_album_id=1904059205397839873#rd
"""
"""
本文对原始IMDB数据集进行预处理，建立一个简单的深层神经网络模型，对给定数据进行情感分析。
"""
import numpy as np

# 数据集下载地址：https://github.com/udacity/deep-learning-v2-pytorch/tree/master/sentiment-rnn/data
# read data from text files
with open('data/reviews/reviews.txt', 'r') as f:
    reviews = f.read()
with open('data/reviews/labels.txt', 'r') as f:
    labels = f.read()

# 编码
# 在将数据输入深度学习模型之前，应该将其转换为数值，文本转换被称为「编码」，这涉及到每个字符转换成一个整数。在进行编码之前，需要清理数据。 有以下几个预处理步骤：
#
# 删除标点符号。
# 使用\n作为分隔符拆分文本。
# 把所有的评论重新组合成一个大串。
from string import punctuation

# remove punctuation
reviews = reviews.lower()
text = ''.join([c for c in reviews if c not in punctuation])
print(punctuation)
# !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~


# split by new lines and spaces
reviews_split = text.split('\n')
text = ' '.join(reviews_split)

# create a list of words
words = text.split()

# 建立字典并对评论进行编码
# 创建一个「字典」，将词汇表中的单词映射为整数。然后通过这个字典，评论可以转换成整数，然后再传送到模型网络。
from collections import Counter

word_counts = Counter(words)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)

vocab2idx = {vocab: idx for idx, vocab in enumerate(vocab, 1)}
print("Size of Vocabulary: ", len(vocab))
# Size of Vocabulary:  74072
encoded_reviews = []
for review in reviews_split:
    encoded_reviews.append([vocab2idx[vocab] for vocab in review.split()])
print("The number of reviews: ", len(encoded_reviews))
# The number of reviews:  25001

# 对标签进行编码
# Negative 和Positive应分别标记为0和1（整数）
splitted_labels = labels.split("\n")
encoded_labels = np.array([
    1 if label == "positive" else 0 for label in splitted_labels
])

# 删除异常值
# 应删除长度为0评论，然后，将对剩余的数据进行填充，保证所有数据具有相同的长度。
length_reviews = Counter([len(x) for x in encoded_reviews])
print("Zero-length reviews: ", length_reviews[0])
print("Maximum review length: ", max(length_reviews))
# Zero-length reviews:  1 Maximum review length:  2514
# reviews with length 0
non_zero_idx = [i for i, review in enumerate(encoded_reviews) if len(review) != 0]

# Remove 0-length reviews
encoded_reviews = [encoded_reviews[i] for i in non_zero_idx]
encoded_labels = np.array([encoded_labels[i] for i in non_zero_idx])


# 填充序列
# 下面要处理很长和很短的评论，需要使用0填充短评论，使其适合特定的长度，
#
# 并将长评论剪切为seq_length的单词。这里设置seq_length=200
def text_padding(encoded_reviews, seq_length):
    reviews = []

    for review in encoded_reviews:
        if len(review) >= seq_length:
            reviews.append(review[:seq_length])
        else:
            reviews.append([0] * (seq_length - len(review)) + review)

    return np.array(reviews)


seq_length = 200
padded_reviews = text_padding(encoded_reviews, seq_length)
print(padded_reviews[:12, :12])
# [[    0     0     0     0     0     0     0     0     0     0     0     0]
#  [    0     0     0     0     0     0     0     0     0     0     0     0]
#  [22382    42 46418    15   706 17139  3389    47    77    35  1819    16]
#  [ 4505   505    15     3  3342   162  8312  1652     6  4819    56    17]
#  [    0     0     0     0     0     0     0     0     0     0     0     0]
#  [    0     0     0     0     0     0     0     0     0     0     0     0]
#  [    0     0     0     0     0     0     0     0     0     0     0     0]
#  [    0     0     0     0     0     0     0     0     0     0     0     0]
#  [    0     0     0     0     0     0     0     0     0     0     0     0]
#  [   54    10    14   116    60   798   552    71   364     5     1   730]
#  [    0     0     0     0     0     0     0     0     0     0     0     0]
#  [    0     0     0     0     0     0     0     0     0     0     0     0]]

# 数据加载器
# 将数据按8:1:1的比例拆分为训练集、验证集和测试集，然后使用“TensorDataset”和“DataLoader”函数来处理评论和标签数据。
ratio = 0.8
train_length = int(len(padded_reviews) * ratio)

X_train = padded_reviews[:train_length]
y_train = encoded_labels[:train_length]

remaining_x = padded_reviews[train_length:]
remaining_y = encoded_labels[train_length:]

test_length = int(len(remaining_x) * 0.5)

X_val = remaining_x[: test_length]
y_val = remaining_y[: test_length]

X_test = remaining_x[test_length:]
y_test = remaining_y[test_length:]
print("Feature shape of train review set: ", X_train.shape)
print("Feature shape of   val review set: ", X_val.shape)
print("Feature shape of  test review set: ", X_test.shape)
# Feature shape of train review set:  (20000, 200)
# Feature shape of   val review set:  (2500, 200)
# Feature shape of  test review set:  (2500, 200)
import torch
from torch.utils.data import TensorDataset, DataLoader

batch_size = 50
device = "cuda" if torch.cuda.is_available() else "cpu"
train_dataset = TensorDataset(torch.from_numpy(X_train).to(device), torch.from_numpy(y_train).to(device))
valid_dataset = TensorDataset(torch.from_numpy(X_val).to(device), torch.from_numpy(y_val).to(device))
test_dataset = TensorDataset(torch.from_numpy(X_test).to(device), torch.from_numpy(y_test).to(device))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
data_iter = iter(train_loader)
X_sample, y_sample = data_iter.next()

# RNN模型的实现
# 到目前为止，包括标记化在内的预处理已经完成。现在建立一个神经网络模型来预测评论的情绪。
#
# 首先，「嵌入层」将单词标记转换为特定大小。
#
# 第二，一个 LSTM层，由hidden_size和num_layers定义。
#
# 第三，通过完全连接的层从LSTM层的输出映射期望的输出大小。
#
# 最后，sigmoid激活层以概率0到1的形式返回输出。

import torch.nn as nn
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers):
        super(Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # embedding and LSTM
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.5,
                            bidirectional=False)

        # 完连接层
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, token, hidden):
        batch_size = token.size(0)

        # embedding and lstm output
        out = self.embedding(token.long())
        out, hidden = self.lstm(out, hidden)

        # stack up lstm outputs
        out = out.contiguous().view(-1, self.hidden_dim)

        # fully connected layer
        out = self.fc(out)

        # reshape to be batch_size first
        out = out.view(batch_size, -1)

        # get the last batch of labels
        out = out[:, -1]

        return out

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)),
                Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)))


# vocab_size : 词汇量
# embedding_dim : 嵌入查找表中的列数
# hidden_dim : LSTM单元隐藏层中的单元数
# output_dim : 期望输出的大小
vocab_size = len(vocab) + 1  # +1 for the 0 padding + our word tokens
embedding_dim = 400
hidden_dim = 256
output_dim = 1
num_layers = 2

model = Model(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers).to(device)
print(model)
# Model(
#   (embedding): Embedding(74073, 400)
#   (lstm): LSTM(400, 256, num_layers=2, batch_first=True, dropout=0.5)
#   (fc): Sequential(
#     (0): Dropout(p=0.5, inplace=False)
#     (1): Linear(in_features=256, out_features=1, bias=True)
#     (2): Sigmoid()
#   )
# )

# 训练
# 对于损失函数，BCELoss被用于「二分类交叉熵损失」，通过给出介于0和1之间的概率进行分类。使用Adam优化器，学习率为0.001
#
# 另外，torch.nn.utils.clip_grad_norm_(model.parameters(), clip = 5)，防止了RNN中梯度的爆炸和消失问题clip是要剪裁最大值。

# Loss function and Optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
train_losses = []

# First checking if GPU is available
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

# move model to GPU, if available
if train_on_gpu:
    model.cuda()

for epoch in range(num_epochs):
    # error：cudnn RNN backward can only be called in training mode
    # cudnn和rnn模型冲突
    # 在epoch前边，加上下面这句话，这句话的意思就是不用cudnn加速训练，
    torch.backends.cudnn.enabled = False
    model.train()
    hidden = model.init_hidden(batch_size)

    for i, (review, label) in enumerate(train_loader):
        review, label = review.to(device), label.to(device)

        # Initialize Optimizer
        optimizer.zero_grad()

        hidden = tuple([h.data for h in hidden])

        # Feed Forward
        output = model(review, hidden)

        # Calculate the Loss
        loss = criterion(output.squeeze(), label.float())

        # Back Propagation
        loss.backward()

        # Prevent Exploding Gradient Problem
        nn.utils.clip_grad_norm_(model.parameters(), 5)

        # Update
        optimizer.step()

        train_losses.append(loss.item())

        # Print Statistics
        if (i + 1) % 100 == 0:

            ### Evaluation ###

            # initialize hidden state
            val_h = model.init_hidden(batch_size)
            val_losses = []

            model.eval()

            for review, label in valid_loader:
                review, label = review.to(device), label.to(device)
                val_h = tuple([h.data for h in val_h])
                output = model(review, val_h)
                val_loss = criterion(output.squeeze(), label.float())

                val_losses.append(val_loss.item())

            print("Epoch: {}/{} | Step {}, Train Loss {:.4f}, Val Loss {:.4f}".
                  format(epoch + 1, num_epochs, i + 1, np.mean(train_losses), np.mean(val_losses)))

# Training on GPU.
# Epoch: 1/10 | Step 100, Train Loss 0.6917, Val Loss 0.6834
# Epoch: 1/10 | Step 200, Train Loss 0.6812, Val Loss 0.6932
# Epoch: 1/10 | Step 300, Train Loss 0.6851, Val Loss 0.6936
# Epoch: 1/10 | Step 400, Train Loss 0.6873, Val Loss 0.6933
# Epoch: 2/10 | Step 100, Train Loss 0.6885, Val Loss 0.6932
# Epoch: 2/10 | Step 200, Train Loss 0.6893, Val Loss 0.6931
# Epoch: 2/10 | Step 300, Train Loss 0.6898, Val Loss 0.6937
# Epoch: 2/10 | Step 400, Train Loss 0.6901, Val Loss 0.6930
# Epoch: 3/10 | Step 100, Train Loss 0.6905, Val Loss 0.6931
# Epoch: 3/10 | Step 200, Train Loss 0.6908, Val Loss 0.6921
# Epoch: 3/10 | Step 300, Train Loss 0.6887, Val Loss 0.6752
# Epoch: 3/10 | Step 400, Train Loss 0.6825, Val Loss 0.5102
# Epoch: 4/10 | Step 100, Train Loss 0.6651, Val Loss 0.4977
# Epoch: 4/10 | Step 200, Train Loss 0.6461, Val Loss 0.4505
# Epoch: 4/10 | Step 300, Train Loss 0.6287, Val Loss 0.4594
# Epoch: 4/10 | Step 400, Train Loss 0.6110, Val Loss 0.4070
# Epoch: 5/10 | Step 100, Train Loss 0.5919, Val Loss 0.4713
# Epoch: 5/10 | Step 200, Train Loss 0.5739, Val Loss 0.4778
# Epoch: 5/10 | Step 300, Train Loss 0.5563, Val Loss 0.4320
# Epoch: 5/10 | Step 400, Train Loss 0.5401, Val Loss 0.4529
# Epoch: 6/10 | Step 100, Train Loss 0.5216, Val Loss 0.5586
# Epoch: 6/10 | Step 200, Train Loss 0.5037, Val Loss 0.5650
# Epoch: 6/10 | Step 300, Train Loss 0.4875, Val Loss 0.5755
# Epoch: 6/10 | Step 400, Train Loss 0.4729, Val Loss 0.5197
# Epoch: 7/10 | Step 100, Train Loss 0.4569, Val Loss 0.6315
# Epoch: 7/10 | Step 200, Train Loss 0.4418, Val Loss 0.6607
# Epoch: 7/10 | Step 300, Train Loss 0.4281, Val Loss 0.6868
# Epoch: 7/10 | Step 400, Train Loss 0.4154, Val Loss 0.7197
# Epoch: 8/10 | Step 100, Train Loss 0.4023, Val Loss 0.8735
# Epoch: 8/10 | Step 200, Train Loss 0.3899, Val Loss 0.8808
# Epoch: 8/10 | Step 300, Train Loss 0.3786, Val Loss 0.8340
# Epoch: 8/10 | Step 400, Train Loss 0.3681, Val Loss 0.8368
# Epoch: 9/10 | Step 100, Train Loss 0.3578, Val Loss 0.8589
# Epoch: 9/10 | Step 200, Train Loss 0.3480, Val Loss 0.8265
# Epoch: 9/10 | Step 300, Train Loss 0.3389, Val Loss 0.8693
# Epoch: 9/10 | Step 400, Train Loss 0.3301, Val Loss 0.8807
# Epoch: 10/10 | Step 100, Train Loss 0.3216, Val Loss 0.9990
# Epoch: 10/10 | Step 200, Train Loss 0.3135, Val Loss 1.0098
# Epoch: 10/10 | Step 300, Train Loss 0.3059, Val Loss 0.9556
# Epoch: 10/10 | Step 400, Train Loss 0.2986, Val Loss 0.9421
#
# Process finished with exit code 0
