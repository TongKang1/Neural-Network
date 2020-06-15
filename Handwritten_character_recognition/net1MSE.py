# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:51:39 2020

@author: Rock Kang
"""


import numpy as np
from math import floor
import torch

import time
time_start=time.time()

input_x = np.loadtxt("data_sheet/input_x.csv", delimiter=',',skiprows = 1)
output_y = np.loadtxt("data_sheet/output_y.csv", delimiter=',',skiprows = 1)

test_x = np.loadtxt("data_sheet/test_x.csv", delimiter=',',skiprows = 1)
test_y = np.loadtxt("data_sheet/test_y.csv", delimiter=',',skiprows = 1)


EPOCHS = 100
DEVICE = torch.device("cuda") 


import torch.nn.functional as F     # 激励函数都在这

class Net(torch.nn.Module):     # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.out = torch.nn.Linear(n_hidden, n_output)       # 输出层线性输出

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x = F.sigmoid(self.hidden(x))      # 激励函数(隐藏层的线性值)
        #x = F.sigmoid(self.out(x))               # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        x = self.out(x)
        return x

net = Net(n_feature=28*28, n_hidden=160, n_output=36) # 几个类别就几个 output

net.to(DEVICE)
print(net)  # net 的结构
"""
Net (
  (hidden): Linear (2 -> 10)
  (out): Linear (10 -> 2)
)
"""
x = torch.from_numpy(input_x)
x = x.to(DEVICE)
y = torch.from_numpy(output_y)
y = y.to(DEVICE)
test_data_x = torch.from_numpy(test_x)
test_data_x = test_data_x.to(DEVICE)
test_data_y = torch.from_numpy(test_y)
test_data_y = test_data_y.to(DEVICE)
# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # 传入 net 的所有参数, 学习率
# 算误差的时候, 注意真实值!不是! one-hot 形式的, 而是1D Tensor, (batch,)
# 但是预测值是2D tensor (batch, n_classes)
loss_func = torch.nn.MSELoss()    # 预测值和真实值的误差计算公式 (均方差)

accuracy = []

net.train()
for i in range(EPOCHS):

    for t in range(500):
        out = net(x.float())     # 喂给 net 训练数据 x, 输出分析值

        loss = loss_func(out.float(), y.float())     # 计算两者的误差
            
        optimizer.zero_grad()   # 清空上一步的残余更新参数值
        loss.backward()         # 误差反向传播, 计算参数更新值
        optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
    
    
    net.eval()
    correct = 0
    #est_loss =0
    with torch.no_grad():
        pred_y = net(test_data_x.float())
        #pred_yy = pred_y.cpu().detach().numpy()            
        #test_loss += F.nll_loss(pred_y, test_data_y.squeeze().long(), reduction = 'sum') # 将一批的损失相加
        pred = pred_y.max(1, keepdim = True)[1] # 找到概率最大的下标
        pred11 = pred.cpu().detach().numpy()
        for j in range(len(pred11)):
            correct += test_y[j,pred11[j]]
    
    #test_loss /= 1440
    accuracy.append(correct/ 14.40)
          
    print('EPO '+ str(i+1)+' accuracy is:',correct/ 14.4)
    
    
time_end=time.time()
print('time cost',time_end-time_start,'s')


torch.save(net,'model/sigmoid_MSE_model.pkl')
torch.save(net.state_dict(), 'model/sigmoid_MSE_model_conf.pkl')


import matplotlib.pyplot as plt

fig = plt.figure() 

plt.rcParams['savefig.dpi'] = 200 #图片像素
plt.rcParams['figure.dpi'] = 200 #分辨率
accuracy = np.array(accuracy)
np.savetxt("data_sheet/sigmoid_MSE.csv", accuracy, delimiter=',')
plt.plot(np.arange(EPOCHS), accuracy) 

plt.ylim((0,100))
plt.xlim((0,100))
plt.xlabel('epochs times')
plt.ylabel('accuracy(%)')
plt.title('sigmoid')
plt.grid()   
plt.show()
