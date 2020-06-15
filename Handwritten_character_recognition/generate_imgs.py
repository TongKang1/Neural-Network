# Created on:   2018-01-09
# Updated on:   2018-09-03
# Author:       coneypo
# Blog:      http://www.cnblogs.com/AdaminXie/
# Github:       https://github.com/coneypo/Generate_handwritten_number
# 生成手写体数字


import random
import os
from PIL import Image, ImageDraw, ImageFont




random.seed(0)

path_img = "data_pngs/"
list1 = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F', \
         'G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V', \
         'W','X','Y','Z']

# 在目录下生成用来存放数字 1-9 的 9个文件夹，分别用 1-9 命名
def mkdir_for_imgs():
    for i in range(len(list1)):
        if os.path.isdir(path_img + "Train_" + list1[i]):
            pass
        else:
            print(path_img + "Train_" + list1[i])
            os.mkdir(path_img + "Train_" + list1[i])


# generate folders
# mkdir_for_imgs()


# 删除路径下的图片
def del_imgs():
    for i in range(len(list1)):
        dir_nums = os.listdir(path_img+ "Train_"  + list1[i])
        for tmp_img in dir_nums:
            if tmp_img in dir_nums:
                # print("delete: ", tmp_img)
                os.remove(path_img + "Train_" + list1[i] + "/" + tmp_img)
    print("Delete finish", "\n")





# 生成单张扭曲的数字图像
def generate_single(i):
    # 先绘制一个50*50的空图像
    im_50_blank = Image.new('RGB', (50, 50), (255, 255, 255))

    # 创建画笔
    draw = ImageDraw.Draw(im_50_blank)

    # 生成随机数1-9
    num = list1[i]

    # 设置字体，这里选取字体大小25
    font = ImageFont.truetype('simsun.ttc', 20)

    # xy是左上角开始的位置坐标
    draw.text(xy=(18, 11), font=font, text=num, fill=(0, 0, 0))

    # 随机旋转-10-10角度
    random_angle = random.randint(-10, 10)
    im_50_rotated = im_50_blank.rotate(random_angle)

    # 图形扭曲参数
    params = [1 - float(random.randint(1, 2)) / 100,
              0,
              0,
              0,
              1 - float(random.randint(1, 10)) / 100,
              float(random.randint(1, 2)) / 500,
              0.001,
              float(random.randint(1, 2)) / 500]

    # 创建扭曲
    im_50_transformed = im_50_rotated.transform((50, 50), Image.PERSPECTIVE, params)

    # 生成新的30*30空白图像
    im_30 = im_50_transformed.crop([11, 11, 39, 39])

    return im_30, num


# 生成手写体0-z存入指定文件夹
def generate_0toz(n):
    # 用cnt_num[1]-cnt_num[9]来计数数字1-9生成的个数，方便之后进行命名
    cnt_num = []
    for i in range(36):
        cnt_num.append(0)

    for m in range(n):
        # 调用生成图像文件函数
        for b in range(len(list1)):
            im, generate_num = generate_single(b)

            # 取灰度
            im_gray = im.convert('L')

            # 计数生成的数字1-9的个数,用来命名图像文件

                # 路径如 "F:/code/***/P_generate_handwritten_number/data_pngs/1/1_231.png"
                # 输出显示路径
            #print("Generate:", path_img + "Train_" + list1[b] + "/" + list1[b] + "_" + str(m) + ".png")
                # 将图像保存在指定文件夹中
            im_gray.save(path_img + "Train_" + list1[b] + "/" + list1[b] + "_" + str(m) + ".png")

    print("\n")
    # 输出显示1-9的分布
'''
# generate n times
mkdir_for_imgs()
del_imgs()

generate_0toz(200)
'''

import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片


im, generate_num = generate_single(1)
im_gray = im.convert('L')
im_gray.save('myName/No1.png')
lena = mpimg.imread(r'myName/No1.png') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理

im, generate_num = generate_single(29)
im_gray = im.convert('L')
im_gray.save('myName/No2.png')
lena = mpimg.imread(r'myName/No2.png') # 读取和代码处于同一目录下的 lena.png


im, generate_num = generate_single(2)
im_gray = im.convert('L')
im_gray.save('myName/No3.png')
lena = mpimg.imread(r'myName/No3.png') # 读取和代码处于同一目录下的 lena.png


im, generate_num = generate_single(20)
im_gray = im.convert('L')
im_gray.save('myName/No4.png')
lena = mpimg.imread(r'myName/No4.png') # 读取和代码处于同一目录下的 lena.png


im, generate_num = generate_single(3)
im_gray = im.convert('L')
im_gray.save('myName/No5.png')
lena = mpimg.imread(r'myName/No5.png') # 读取和代码处于同一目录下的 lena.png


lena.shape #(512, 512, 3)

plt.imshow(lena,cmap='gray') # 显示灰度图片
plt.axis('off') # 不显示坐标轴
plt.show()


