# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 10:01:16 2019

@author: lxl
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pylab
from tensorflow.examples.tutorials.mnist import input_data
  
#载入数据集
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
  
#设置每个批次的大小
batch_size=100
#计算一共有多少个批次
n_batch=mnist.train.num_examples//batch_size
  
#定义三个placeholder
x=tf.placeholder(tf.complex64,[None,784])
y=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)  #存放百分率
  
#创建一个多层神经网络模型
#第一个隐藏层
W1=tf.Variable(tf.truncated_normal([784,200],stddev=0.1))
b1=tf.Variable(tf.zeros([2000])+0.1)
L1=tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_drop=tf.nn.dropout(L1,keep_prob) #keep_prob设置工作状态神经元的百分率
#第二个隐藏层
W2=tf.Variable(tf.truncated_normal([2000,2000],stddev=0.1))
b2=tf.Variable(tf.zeros([2000])+0.1)
L2=tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
L2_drop=tf.nn.dropout(L2,keep_prob)
#第三个隐藏层
W3=tf.Variable(tf.truncated_normal([2000,1000],stddev=0.1))
b3=tf.Variable(tf.zeros([1000])+0.1)
L3=tf.nn.tanh(tf.matmul(L2_drop,W3)+b3)
L3_drop=tf.nn.dropout(L3,keep_prob)
#输出层
W4=tf.Variable(tf.truncated_normal([1000,10],stddev=0.1))
b4=tf.Variable(tf.zeros([10])+0.1)
prediction=tf.nn.softmax(tf.matmul(L3_drop,W4)+b4)

#定义交叉熵代价函数
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
optimizer = tf.train.GradientDescentOptimizer(0.2) # 梯度下降法（反向传播算法），学习速率为0.5
train = optimizer.minimize(loss) # 训练目标：最小化损失函数

#结果存放在一个布尔型列表中(argmax函数返回一维张量中最大的值所在的位置)
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
  
#求准确率(tf.cast将布尔值转换为float型)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#创建会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #初始化变量
    #训练次数
    for i in range(5):
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
        #测试数据计算出的准确率
        test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        print("Iter"+str(i)+",Testing Accuracy"+str(test_acc))
        
    for i in range(0, len(mnist.test.images)):
        result = sess.run(correct_prediction, feed_dict={x: np.array([mnist.test.images[i]]), y: np.array([mnist.test.labels[i]]),keep_prob:1.0})
        if not result:
          print('预测的值是：',sess.run(prediction, feed_dict={x: np.array([mnist.test.images[i]]), y: np.array([mnist.test.labels[i]]),keep_prob:1.0}))
          print('实际的值是：',sess.run(y,feed_dict={x: np.array([mnist.test.images[i]]), y: np.array([mnist.test.labels[i]]),keep_prob:1.0}))
          one_pic_arr = np.reshape(mnist.test.images[i], (28, 28))
          pic_matrix = np.matrix(one_pic_arr, dtype="float")
          plt.imshow(pic_matrix)
          pylab.show()
          break
 




