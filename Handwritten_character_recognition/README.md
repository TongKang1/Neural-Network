# 文件运行顺序

#数据预处理

Step1:随机生成训练神经网络所需的图片
    
    运行generate_imgs.py
    图片保存在data_pngs文件夹下（由于文件数目太多，已压缩打包）
    
Step2:读取全部图片与标签，生成训练用的标准数据集（.csv）

    运行png_read.py
    在data_sheet目录下，会生成对应的文件，以便网络读取
    
Step3:生成作业需要的字符图片，并加上不同程度的噪音
    
    运行add_noise_fig.py
    测试的图片保存在myName文件夹下
    
#网络训练

Step4:选取对应的网络训练
    
    选取net1.py 、net1MSE.py或net2.py其中的一个，依次运行
    （激活函数可在源码中修改）
    net1.py为损失度函数为交叉熵函数的BP网络
    net1MSE.py为损失度函数为平方和函数的BP网络
    net2.py为CNN网络(基于LeNet-5)
    运行过程中的准确率的变化过程会被保存到data_sheet文件夹下，网络参数会被保存在对应的.pkl文件中
    
    
#数据测试

Step5:作业要求的测试
    
    依次运行runNet1.py 、runNet1MSE.py和runNet2.py，测试训练好的模型
    
    运行drawRate.py可画出6个网络的准确率对比（可选）
    
    
