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
  
#载入数据集
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
  
#设置每个批次的大小
batch_size=100
#计算一共有多少个批次
n_batch=mnist.train.num_examples//batch_size
  
#定义三个placeholder
x=tf.placeholder(tf.complex64,[None,784])
y=tf.placeholder(tf.float32,[None,10])
#keep_prob=tf.placeholder(tf.float32)  #存放百分率

#将两个实数矩阵变成复数矩阵
'''a = tf.constant([[1, 2], [78, 3]], dtype=tf.float32)
a = a/tf.reduce_max(a)
b = tf.constant([[1, 3], [3, 1]], dtype=tf.float32)
c = tf.complex(a,b)

a1 = tf.constant([[1, 1], [1, 1]], dtype=tf.float32)
b1 = tf.constant([[1, -1], [-1, 1]], dtype=tf.float32)
c1 = tf.complex(a1,b1)

k=tf.matmul(c,c1)
k1=tf.abs(k)
with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(c1))
    print(sess.run(k))
    print(sess.run(k1))
    print(sess.run(a))'''


w1= np.ones([784,200],dtype=np.complex64)#根据神经元之间距离决定
w2= np.ones([200,200],dtype=np.complex64)#根据神经元之间距离决定
w3= np.ones([200,10],dtype=np.complex64)#根据神经元之间距离决定

#以mm做单位
ix=0.0
kx=0.0
for i in range(784):
    for k in range(200):
        r=math.sqrt((kx-ix)**2+30**2)
        #w1[i,k]=(0.03/(r**2))*(1/(2*math.pi*r)+1/(0.75*(10**(-3))*cmath.sqrt(-1)))*cmath.exp(2*math.pi*r*cmath.sqrt(-1)/(0.75*(10**(-3))))
        w1[i,k]=(30/(r**2))*(1/(2*math.pi*r)+1/((0.75)*cmath.sqrt(-1)))*cmath.exp(2*math.pi*r*cmath.sqrt(-1)/(0.75))
        kx=kx+0.4
    ix=ix+0.102
    kx=0

ix=0.0
kx=0.0
for i in range(200):
    for k in range(200):
        r=math.sqrt((kx-ix)**2+30**2)
        w2[i,k]=(30/(r**2))*(1/(2*math.pi*r)+1/((0.75)*cmath.sqrt(-1)))*cmath.exp(2*math.pi*r*cmath.sqrt(-1)/(0.75))
        kx=kx+0.4
    ix=ix+0.4
    kx=0

ix=0.0
kx=0.0
for i in range(200):
    for k in range(10):
        r=math.sqrt((kx-ix)**2+30**2)
        w3[i,k]=(30/(r**2))*(1/(2*math.pi*r)+1/((0.75)*cmath.sqrt(-1)))*cmath.exp(2*math.pi*r*cmath.sqrt(-1)/(0.75))        
        kx=kx+8
    ix=ix+0.4
    kx=0
        

'''phase=tf.Variable(tf.nn.sigmoid(tf.truncated_normal([1,7])))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(phase))'''

  
#创建一个多层神经网络模型
#第一个隐藏层
#amp1=tf.Variable(tf.nn.sigmoid(tf.truncated_normal([200])))
#phase1=tf.Variable(2*math.pi*tf.nn.sigmoid(tf.truncated_normal([200])))
amp1=tf.nn.sigmoid(tf.Variable(tf.truncated_normal([1,200])))
phase1=2*math.pi*tf.nn.sigmoid(tf.Variable(tf.truncated_normal([1,200])))
t1=tf.complex(amp1,phase1)
L1=tf.multiply(tf.matmul(x,w1),t1)
#L1_drop=tf.nn.dropout(L1,keep_prob) #keep_prob设置工作状态神经元的百分率，L1好像必须是floating，不能是复数
#第二个隐藏层
#amp2=tf.Variable(tf.nn.sigmoid(tf.truncated_normal([200])))
#phase2=tf.Variable(2*math.pi*tf.nn.sigmoid(tf.truncated_normal([200])))
amp2=tf.nn.sigmoid(tf.Variable(tf.truncated_normal([1,200])))
phase2=2*math.pi*tf.nn.sigmoid(tf.Variable(tf.truncated_normal([1,200])))
t2=tf.complex(amp2,phase2)
L2=tf.multiply(tf.matmul(L1,w2),t2)
#L2_drop=tf.nn.dropout(L2,keep_prob) #keep_prob设置工作状态神经元的百分率
#第三个隐藏层
#amp3=tf.Variable(tf.nn.sigmoid(tf.truncated_normal([200])))
#phase3=tf.Variable(2*math.pi*tf.nn.sigmoid(tf.truncated_normal([200])))
amp3=tf.nn.sigmoid(tf.Variable(tf.truncated_normal([1,200])))
phase3=2*math.pi*tf.nn.sigmoid(tf.Variable(tf.truncated_normal([1,200])))
t3=tf.complex(amp3,phase3)
L3=tf.multiply(tf.matmul(L2,w2),t3)
#L3_drop=tf.nn.dropout(L3,keep_prob)
#第四个隐藏层
#amp4=tf.Variable(tf.nn.sigmoid(tf.truncated_normal([200])))
#phase4=tf.Variable(2*math.pi*tf.nn.sigmoid(tf.truncated_normal([200])))
amp4=tf.nn.sigmoid(tf.Variable(tf.truncated_normal([1,200])))
phase4=2*math.pi*tf.nn.sigmoid(tf.Variable(tf.truncated_normal([1,200])))
t4=tf.complex(amp4,phase4)
L4=tf.multiply(tf.matmul(L3,w2),t4)
#L4_drop=tf.nn.dropout(L4,keep_prob)
#第五个隐藏层
#amp5=tf.Variable(tf.nn.sigmoid(tf.truncated_normal([200])))
#phase5=tf.Variable(2*math.pi*tf.nn.sigmoid(tf.truncated_normal([200])))
amp5=tf.nn.sigmoid(tf.Variable(tf.truncated_normal([1,200])))
phase5=2*math.pi*tf.nn.sigmoid(tf.Variable(tf.truncated_normal([1,200])))
t5=tf.complex(amp5,phase5)
L5=tf.multiply(tf.matmul(L4,w2),t5)
#L5_drop=tf.nn.dropout(L5,keep_prob)
#输出层
aout=tf.abs(tf.matmul(L5,w3))
sout=tf.multiply(aout,aout)
#pred=sout/tf.reduce_max(sout)
pred=sout

#定义交叉熵代价函数
#loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
loss=tf.reduce_mean(tf.square(y-pred))# 损失函数为mse
optimizer = tf.train.AdamOptimizer(0.2) # 梯度下降法（反向传播算法），学习速率为0.5
#optimizer = tf.train.GradientDescentOptimizer(0.2) # 梯度下降法（反向传播算法），学习速率为0.5
train = optimizer.minimize(loss) # 训练目标：最小化损失函数

#结果存放在一个布尔型列表中(argmax函数返回一维张量中最大的值所在的位置)
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(pred,1))
  
#求准确率(tf.cast将布尔值转换为float型)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


#创建会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #初始化变量
    #训练次数
    for i in range(30):
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_xs,y:batch_ys})
        #测试数据计算出的准确率和loss
        test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        test_los=sess.run(loss,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter"+str(i)+",accuracy："+str(test_acc))
        print("loss："+str(test_los))
    #训练后各层的t值
    t1_1=sess.run(t1,feed_dict={x:mnist.test.images,y:mnist.test.labels})
    t2_1=sess.run(t2,feed_dict={x:mnist.test.images,y:mnist.test.labels})
    t3_1=sess.run(t3,feed_dict={x:mnist.test.images,y:mnist.test.labels})
    t4_1=sess.run(t4,feed_dict={x:mnist.test.images,y:mnist.test.labels})
    t5_1=sess.run(t5,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        
       
        
   
    #for i in range(0, len(mnist.test.images)):
    for i in range(0, 5):
        result = sess.run(correct_prediction, feed_dict={x: np.array([mnist.test.images[i]]), y: np.array([mnist.test.labels[i]])})
        #if not result:
        print('预测的值是：',sess.run(pred, feed_dict={x: np.array([mnist.test.images[i]]), y: np.array([mnist.test.labels[i]])}))
        print('实际的值是：',sess.run(y,feed_dict={x: np.array([mnist.test.images[i]]), y: np.array([mnist.test.labels[i]])}))
        one_pic_arr = np.reshape(mnist.test.images[i], (28, 28))
        pic_matrix = np.matrix(one_pic_arr, dtype="float")
        plt.imshow(pic_matrix)
        pylab.show()
        #if not result:
            #break
 




