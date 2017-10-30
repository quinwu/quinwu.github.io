---
title: ML-CNN
date: 2017-08-21 15:37:00
tags:
---

### Convolutional Neural Network 

CNN 卷积神经网络是一种专门来处理具有类似的网格结构的数据的神经网络。

> Convolutional Networks are simply neural networks that use convolution in place of general matrix multiplication in at least one of their layers.
>
> 卷积网络是指那些至少在网络的一层中使用卷积运算来替代一般的矩阵乘法运算的神经网络。

### Convolution Operation

$$s(t) = \int x(a)w(t-a)da$$t

$s(t) = (x*w)(t)$

> first argument (function $x$) is often referred to as the $input$ 
>
> second argument (function $w$) as the $kernel$
>
> The output is sometimes referred to as the $feature \  map$ （特征映射）

$$
S(i,j) = (I * K)(i*j) = \sum_m \sum_n I(m,n)K(i-m,j-n)
$$

$$
S(i,j) = (K*I)(i,j) = \sum_m \sum_n I(i-m,j-n)K(m,n)
$$

cross-correlation
$$
S(i,j) =(I*K)(i,j) =\sum_m \sum_n I(i+m,j+n)K(m,n)
$$

### Motivation

##### sparse interactions

##### parameter sharing

##### equivariant representations



### 定义Convolutional Neural Network

##### Convolution Layer

##### Pooling Layer

##### Output Layer





### 参考文献

http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/

http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/

https://grzegorzgwardys.wordpress.com/2016/04/22/8/

https://www.zybuluo.com/hanbingtao/note/485480

http://www.cnblogs.com/pinard/p/6494810.html

http://www.jianshu.com/p/9c4396653324

https://github.com/manutdzou/manutdzou.github.io/blob/master/_posts/%E7%A7%91%E7%A0%94/2016-05-17-Why%20computing%20the%20gradients%20CNN%2C%20the%20weights%20need%20to%20be%20rotated.md

http://colah.github.io/posts/2014-07-Understanding-Convolutions/

http://mengqi92.github.io/2015/10/06/convolution/

http://neuralnetworksanddeeplearning.com/chap4.html