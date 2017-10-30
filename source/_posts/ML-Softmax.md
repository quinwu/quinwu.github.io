---
title: 小记 Softmax 
date: 2017-09-14 20:40:15
categories: 笔记
tags:
  - Machine Learning
  - 小记系列
---

在数学，尤其是概率论和相关领域中，Softmax函数，或称归一化指数函数，是逻辑函数的一种推广。它能将一个含任意实数的$K$维的向量$z$『压缩』到另外一个$K$维实向量$\sigma(z)$中，使得每一个元素的范围都在(0,1)之间，并且所有的元素和为$1$。<!--more-->

### Softmax损失函数的由来

Softmax 概率函数可以表示为：


$$
P(y|x;\theta) = \prod ^k_{j=1} \left(\frac{e^{\theta^T_j x}}{\sum^k_{l=1} e^{\theta^T_lx}} \right) ^{\mathbb1 \left\{y=j\right\}}
$$


似然函数表示为：
$$
L(\theta) = \prod^m_{i=1} \prod^k_{j=1} \left(  \frac{e^{\theta^T_j x}}{\sum^k_{l=1} e^{\theta^T_l x}} \right)^{\mathbb1 \left\{y=j\right\}}
$$
$\log$似然为
$$
l(\theta) = \log L(\theta) = \sum^m_{i=1}\sum^k_{j=1} \mathbb1\left\{y=j\right\} \log \frac{e^{\theta^T_j x}}{\sum^k_{l=1} e^{\theta^T_l x}}
$$
我们要最大化似然函数，即求 $\max l(\theta)$。也就是下面的$\min J(\theta)$


$$
\begin{align}
J(\theta) = - \frac{1}{m} \left[ \sum_{i=1}^{m} \sum_{k=1}^{K} \mathbb 1 \left\{y^{(i)} = k\right\} \log \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)})}\right]
\end{align}
$$
### Softmax Gradient

对$J(\theta)$求梯度，得
$$
\begin{align}
\frac{\partial J(\theta)}{\partial \theta_j}  &= -\frac{1}{m} \frac{\partial}{\partial \theta_j} \left[  \sum_{i=1}^{m}\sum_{j=1}^{k} \mathbb1 \left\{  y^{(i) }=j\right\} \log \frac{e^{\theta^T_j x^{(i)}}}{\sum^k_{l=1}e^{\theta^T_lx^{(i)}} }    \right] \\
\\ &= -\frac{1}{m} \frac{\partial}{\partial \theta_j} \left[  \sum_{i=1}^{m}\sum_{j=1}^{k} \mathbb1 \left\{  y^{(i) }=j\right\}( \theta^T_jx^{(i)} - \log \sum^k_{l=1}e^{\theta^T_lx^{(i)} })    \right] \\
\\ &= -\frac{1}{m} \left[  \sum^m_{i=1} \mathbb1\left \{ y^{(i)} =j\right\}(x^{(i)} - \sum^{k}_{j=1} \frac{e^{\theta^T_j x^{(i)}}\cdot x^{(i)}}{\sum^k_{l=1} e^{\theta^T_lx^{(i)}}} ) \right] \\
\\ &= -\frac{1}{m} \left[  \sum^m_{i=1} x^{(i)} \mathbb1\left \{ y^{(i)} =j\right\}(1 - \sum^{k}_{j=1} \frac{e^{\theta^T_j x^{(i)}}}{\sum^k_{l=1} e^{\theta^T_lx^{(i)}}} ) \right] \\
\\ &= -\frac{1}{m} \left[  \sum^m_{i=1} x^{(i)}( \mathbb1\left \{ y^{(i)} =j\right\} -\sum^k_{j=1}\mathbb1\left \{ y^{(i)} =j\right\}  \frac{e^{\theta^T_j x^{(i)}}}{\sum^k_{l=1} e^{\theta^T_lx^{(i)}}} ) \right] \\
\\ &= -\frac{1}{m} \left[  \sum^m_{i=1} x^{(i)}( \mathbb1\left \{ y^{(i)} =j\right\} -  \frac{e^{\theta^T_j x^{(i)}}}{\sum^k_{l=1} e^{\theta^T_lx^{(i)}}} ) \right] \\
\\ &= -\frac{1}{m} \left[  \sum^m_{i=1} x^{(i)}( \mathbb1\left \{ y^{(i)} =j\right\} -  p(y^{(i)} = j | x^{(i)};\theta)\right] \\
\end{align}
$$




使用交叉熵作为我们的损失函数
$$
L = -\sum_j y_j\log p_j
$$
或者
$$
L = -\sum_jy_i \ln p_i
$$



$$
p_k = \frac{e^{f_k}}{\sum_j e^{f_j}}
$$

$$
L_i = -\log(\frac{e^{f_{y_i}}}{\sum_je^{f_i}})
$$

$$
L_i = -\log(p_{y_i})
$$

$$
\frac{\partial L_i }{ \partial f_k } = p_k - \mathbb{1}(y_i = k)
$$

### 参考文献

- [Softmax 回归](http://blog.csdn.net/u012328159/article/details/72155874)
- [ufldl](http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/)