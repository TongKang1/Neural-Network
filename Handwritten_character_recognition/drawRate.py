# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:20:23 2020

@author: Rock Kang
"""


import numpy as np
t = np.array([5,10,15,20,30,40])

input_rate1 = np.array([4,4,3,3,3,1])
input_rate2 = np.array([5,4,4,3,4,1])
input_rate3 = np.array([5,5,5,4,4,4])
input_rate4 = np.array([5,5,5,4,5,4])
input_rate5 = np.array([4,4,3,3,3,3])
input_rate6 = np.array([5,4,5,5,5,2])


'''
t = np.arange(100)

input_rate1 = np.loadtxt('data_sheet/sigmoid_MSE.csv', delimiter=',',skiprows = 0)
input_rate2 = np.loadtxt('data_sheet/relu_MSE.csv', delimiter=',',skiprows = 0)
input_rate3 = np.loadtxt('data_sheet/sigmoid_CE.csv', delimiter=',',skiprows = 0)
input_rate4 = np.loadtxt('data_sheet/relu_CE.csv', delimiter=',',skiprows = 0)
input_rate5 = np.loadtxt('data_sheet/max_pool_CNN.csv', delimiter=',',skiprows = 0)
input_rate6 = np.loadtxt('data_sheet/aver_pool_CNN.csv', delimiter=',',skiprows = 0)

'''

import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 200 #图片像素
plt.rcParams['figure.dpi'] = 200 #分辨率


plt.plot(t, input_rate1, '--', label = 'sigmoid+MSE')
plt.plot(t, input_rate2, '--', label = 'relu+MSE')
plt.plot(t, input_rate3, '--', label = 'sigmoid+CE')
plt.plot(t, input_rate4, '--', label = 'relu+CE')
plt.plot(t, input_rate5, '--', label = 'max_pool')
plt.plot(t, input_rate6, '--', label = 'aver_pool')


plt.legend()
plt.xlim(0,40)
plt.ylim(0,5.1)
plt.xlabel('noise rate(%)')
plt.ylabel('correct number')
plt.title('trend line')
plt.grid()   
plt.show()
