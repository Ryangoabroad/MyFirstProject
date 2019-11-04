# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 10:01:16 2019

@author: lxl
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pylab
import math
import cmath
from tensorflow.examples.tutorials.mnist import input_data
import cv2

  
#载入数据集
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

image= np.ones([50,784],dtype=np.float64)
fimage= np.ones([50,784],dtype=np.float64)

for i in range(0, 50):
    img = np.reshape(mnist.test.images[i], (28, 28))
    #将图像缩放（改变矩阵维数）
    img1 = cv2.resize(img, (28,28), interpolation = cv2.INTER_AREA)
    #将矩阵变成一维数组，添加进原始数据集，并进行二值化处理
    image[i] = np.reshape(img1, (1, 784))
    '''for j in range(0, 784):
        if image[i,j] < 0.5:
            image[i,j]=0
        else:
            image[i,j]=1'''
    #对处理后的图像进行傅里叶变换
    origin = np.reshape(image[i], (28, 28))
    #快速傅里叶变换算法得到频率分布
    f = np.fft.fft2(origin)
    #默认结果中心点位置是在左上角,
    #调用fftshift()函数转移到中间位置
    fshift = np.fft.fftshift(f) 
    #fft结果是复数, 其绝对值结果是振幅
    fimg = np.log(np.abs(fshift))
    #将矩阵变成一维数组，添加进傅里叶数据集
    fimage[i] = np.reshape(fimg, (1, 784))
    plt.subplot(121)
    plt.imshow(origin)
    plt.title('Original Fourier')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(fimg)
    plt.title('Fourier Fourier')
    plt.axis('off')
    plt.show()


'''for i in range(0, 1):
    image[i] = np.reshape(mnist.test.images[i], (1, 784))
    for j in range(0, 784):
        if image[i,j]<0.5:
            image[i,j]=0
        else:
            image[i,j]=1
    one_pic_arr = np.reshape(image[i], (28, 28))
    f = np.fft.fft2(one_pic_arr)
    fshift = np.fft.fftshift(f) 
    fimg = np.log(np.abs(fshift))
    plt.subplot(121)
    plt.imshow(one_pic_arr)
    plt.title('Original Fourier')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(fimg)
    plt.title('Fourier Fourier')
    plt.axis('off')
    plt.show()'''
    
'''for i in range(0, 10):
    image = np.reshape(mnist.test.images[i], (28, 28))
    resized = cv2.resize(image, (14,14), interpolation = cv2.INTER_AREA)
    plt.imshow(image)
    plt.show()
    plt.imshow(resized)
    plt.show()'''
    
    
    
