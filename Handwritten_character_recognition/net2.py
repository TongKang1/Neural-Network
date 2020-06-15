# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:36:22 2020

@author: Rock Kang
"""


import numpy as np
import torch


input_x = np.loadtxt("data_sheet/input_x.csv", delimiter=',',skiprows = 1)
output_y = np.loadtxt("data_sheet/output_y_one.csv", delimiter=',',skiprows = 1)

test_x = np.loadtxt("data_sheet/test_x.csv", delimiter=',',skiprows = 1)
test_y = np.loadtxt("data_sheet/test_y_one.csv", delimiter=',',skiprows = 1)

input_x = input_x.reshape((5760,1,28,28))
output_y = output_y.reshape((5760,1))
test_x = test_x.reshape((1440,1,28,28))
test_y = test_y.reshape((1440,1))


####
BATCH_SIZE = 512 # 大概需要2G的显存
EPOCHS = 100 # 总共训练批次

DEVICE = torch.device( "cuda") 



x = torch.from_numpy(input_x)
x = x.to(DEVICE)
y = torch.from_numpy(output_y)
y = y.to(DEVICE)
test_data_x = torch.from_numpy(test_x)
test_data_x = test_data_x.to(DEVICE)
test_data_y = torch.from_numpy(test_y)
test_data_y = test_data_y.to(DEVICE)


import time
time_start=time.time()


import torch.nn.functional as F     # 激励函数都在这

# 定义模型
class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #1*1*28*28
        self.conv1 = torch.nn.Conv2d(1, 7, 5) 
        self.conv2 = torch.nn.Conv2d(7, 14, 3) 
        self.fc1 = torch.nn.Linear(14* 5 * 5, 120)
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
    
    
model = ConvNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())


print(model)  # net 的结构
"""
Net (
  (hidden): Linear (2 -> 10)
  (out): Linear (10 -> 2)
)
"""

accuracy = []

model.train()

for i in range(EPOCHS):

    for t in range(2):
        #data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(x.float())
        a = output.cpu().detach().numpy()
        loss = F.nll_loss(output.float(), y.squeeze().long())
        loss.backward()
        optimizer.step()

        
    model.eval()
    test_loss =0
    correct = 0
    with torch.no_grad():
        pred_y = model(test_data_x.float())
        #pred_yy = pred_y.cpu().detach().numpy()            
        test_loss += F.nll_loss(pred_y, test_data_y.squeeze().long(), reduction = 'sum') # 将一批的损失相加
        pred = pred_y.max(1, keepdim = True)[1] # 找到概率最大的下标
        correct += pred.eq(test_data_y.view_as(pred)).sum().item()
    
    test_loss /= 1440
    accuracy.append(correct/ 14.40)
    print('EPO '+ str(i+1)+' accuracy is:',correct/ 14.4)


time_end=time.time()
print('time cost',time_end-time_start,'s')

#save model
torch.save(model,'CNN_model_1_max.pkl')
torch.save(model.state_dict(), 'CNN_model_1_max_conf.pkl')

import matplotlib.pyplot as plt

fig = plt.figure() 

plt.rcParams['savefig.dpi'] = 200 #图片像素
plt.rcParams['figure.dpi'] = 200 #分辨率
accuracy = np.array(accuracy)
np.savetxt("data_sheet/max_pool_CNN.csv", accuracy, delimiter=',')
plt.plot(np.arange(EPOCHS), accuracy) 

plt.ylim((0,100))
plt.xlim((0,100))
plt.xlabel('epochs times')
plt.ylabel('accuracy(%)')
plt.title('CNN_1')
plt.grid()   
plt.show()


'''

import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片

filename = r'data_pngs/Train_0/0_0.png'
lena = mpimg.imread(filename) # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape #(512, 512, 3)
 
plt.imshow(lena,cmap='gray') # 显示灰度图片
plt.axis('off') # 不显示坐标轴
plt.show()

input_img = lena 
input_img = input_img.reshape((1,1,28,28))
input_img = torch.from_numpy(lena)


CNN_model = ConvNet()
CNN_model.load_state_dict(torch.load('CNN_model_1_conf.pkl'))

CNN_model.eval()
with torch.no_grad():
    result = CNN_model(input_img.float())

output = result.numpy()
print(output)
'''