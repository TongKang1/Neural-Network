# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:26:27 2020

@author: Rock Kang
"""



import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np

input_x = np.zeros((1,28*28))
output_y = np.zeros((1,1))
test_x = np.zeros((1,28*28))
test_y = np.zeros((1,1))

filename = r'data_pngs/Train_'

list1 = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F', \
         'G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V', \
         'W','X','Y','Z']

for i in range(36):
    for n in range(200):
        lena = mpimg.imread(filename+list1[i]+'/'+list1[i]+'_'+str(n)+'.png') # 读取和代码处于同一目录下的 lena.png
        # 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
        lena.shape #(512, 512, 3)
        a = lena
        b = np.zeros((1,1))
        b[0,0] = i
        a = a.reshape((1,28*28))
        if(n<160):
            input_x = np.vstack((input_x,a))
            output_y = np.vstack((output_y,b))
        else:
            test_x = np.vstack((test_x,a))
            test_y = np.vstack((test_y,b))


np.savetxt("data_sheet/input_x.csv", input_x, delimiter=',')
np.savetxt("data_sheet/output_y_one.csv", output_y, delimiter=',')

np.savetxt("data_sheet/test_x.csv", test_x, delimiter=',')
np.savetxt("data_sheet/test_y_one.csv", test_y, delimiter=',')
    

plt.imshow(lena,cmap='gray') # 显示灰度图片
plt.axis('off') # 不显示坐标轴
plt.show()

