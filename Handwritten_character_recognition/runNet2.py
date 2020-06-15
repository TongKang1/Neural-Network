# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:11:27 2020

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

# 定义模型
class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #1*1*28*28
        self.conv1 = torch.nn.Conv2d(1, 7, 5) 
        self.conv2 = torch.nn.Conv2d(7, 14, 3) 
        self.fc1 = torch.nn.Linear(14 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 36)
        
    def forward(self, x):
        in_size = x.size(0)
        out= self.conv1(x) # 1* 10 * 24 *24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2) # 1* 6 * 12 * 12
        out = self.conv2(out) # 1* 20 * 10 * 10
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2) # 1* 12 * 5 * 5
        out = out.view(in_size, -1) # 1 * 2000
        out = self.fc1(out) # 1 * 500
        out = F.relu(out)
        out = self.fc2(out) # 1 * 10
        out = F.log_softmax(out, dim = 1)
        return out
    

CNN_model = ConvNet().to(DEVICE)
CNN_model.load_state_dict(torch.load('CNN_model_1_max_conf.pkl'))

CNN_model.eval()



def read_noise_fig(percent = 0):
    input_img2 = np.zeros((5,1,28,28))
    for i in range(5):
        if(percent>0):
            input_img2[i,0] = mpimg.imread(r'myName/No'+str(i+1)+'_'+str(percent)+'_noise.png')
        else:
            input_img2[i,0] = mpimg.imread(r'myName/No'+str(i+1)+'.png')
    
    input_img2 = torch.from_numpy(input_img2)
    input_img2 = input_img2.to(DEVICE)
    CNN_model.eval()
    CNN_model.eval()
    with torch.no_grad():
        pred_y = CNN_model(input_img2.float())
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
    




