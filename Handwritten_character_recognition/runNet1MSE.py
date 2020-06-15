# -*- coding: utf-8 -*-
"""
Created on Tue May 26 23:23:28 2020

@author: Rock Kang
"""



import torch
import numpy as np
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片


list1 = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F', \
         'G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V', \
         'W','X','Y','Z']

DEVICE = torch.device("cuda") 


import torch.nn.functional as F     # 激励函数都在这

class Net(torch.nn.Module):     # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.out = torch.nn.Linear(n_hidden, n_output)       # 输出层线性输出

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = self.out(x)               # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x

net = torch.load('model/relu_MSE_model.pkl')
net.to(DEVICE)

net.eval()



def read_noise_fig(percent = 0):
    input_img = np.zeros((5,28*28))
    for i in range(5):
        if(percent>0):
            input_img2 = mpimg.imread(r'myName/No'+str(i+1)+'_'+str(percent)+'_noise.png')
        else:
            input_img2 = mpimg.imread(r'myName/No'+str(i+1)+'.png')
        input_img[i,:] = input_img2.reshape(28*28)
        
    input_img = torch.from_numpy(input_img)
    input_img = input_img.to(DEVICE)
    with torch.no_grad():
        pred_y = net(input_img.float())
    pred = pred_y.max(1, keepdim = True)[1] # 找到概率最大的下标
    output = pred.cpu().detach().numpy()
    
    print('add '+str(percent)+'% noise')
    for i in range(len(output)):
        print('the No.'+str(i+1)+' picture is: '+ list1[output[i,0]])


read_noise_fig(5)
read_noise_fig(10)
read_noise_fig(15)
read_noise_fig(20)
read_noise_fig(30)
read_noise_fig(40)
    




