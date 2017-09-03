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

