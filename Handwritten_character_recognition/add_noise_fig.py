# -*- coding: utf-8 -*-
"""
Created on Mon May 18 21:30:07 2020

@author: Rock Kang
"""


import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import random 
from PIL import Image


random.seed(0)

def add_noise(percent):
    for i in range(5):
        '''
        image_array = mpimg.imread(r'myName/No'+str(i+1)+'.png') # 读取和代码处于同一目录下的 lena.png
        image_array = image_array.reshape(28*28)
        for j in range(len(image_array)):
            if(random.random() < percent/100):
                image_array[j] = min(random.random()+0.3,image_array[i])
        
        image_array = image_array.reshape((28,28))
        image_array *= 255  # 变换为0-255的灰度值
        im = Image.fromarray(image_array)
        im = im.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
        im.save('myName/No'+str(i+1)+'_'+str(percent)+'_noise.png')
        '''
        image_array = mpimg.imread(r'myName/No'+str(i+1)+'_'+str(percent)+'_noise.png') # 读取和代码处于同一目录下的 lena.png
        #print(image_array)
        plt.imshow(image_array,cmap='gray')
        plt.axis('off') # 不显示坐标轴
        plt.show()

        
add_noise(5)
add_noise(10)
add_noise(15)
add_noise(20)
add_noise(30)
add_noise(40)