---
title: 小记 Gradient Descent
date: 2017-09-02 19:55:59
categories: 技术向
tags:
  - Machine Learning
  - 小记系列
---


Gradient Descent（梯度下降法）是一个一阶最优化算法，又被称为最速下降法。通过Gradient Descent算法对函数上当前点对应梯度的反方向的指定步长进行迭代搜索，我们可以很快的找到一个Local Minimum（局部极小值）。如果是对当前点对应梯度的正方向进行迭代搜索，我们可以求得一个Local Maximum（局部最大值）。

<!--more-->

在应用到机器学习算法的时候，我们通常采用Gradient Descent来对所采用的算法进行训练，使得loss函数的目标值达到最优。

以前面提到过的Linear Regression为例，假设函数为
$$
h_\theta(x) = \theta^Tx = \sum_{j=0}^n\theta_j x_j
$$
对应的损失函数loss为
$$
J(\theta) = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})^2
$$

其中，$n$为训练样本的特征数量；$m$为训练样本数。

### Batch Gradient Descent

Batch Gradient Descent（批量梯度下降法，BGD）是 Gradient Descent（梯度下降法）的最基本形式，算法主要布置是：在更新每个参数的时候，都要使用所有的样本来进行更新。

对上述的损失函数loss求偏导：


$$
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j
$$

$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial{\theta_j}} {J(\theta) },\quad (  j = 0,1,\cdots,n)
$$

### Stochastic Gradient Descent

Stochastic Gradient Descent（随机梯度下降法，SGD）

由于Batch Gradient Descent（批量低度下降法，BGD）在更新每一个参数$\theta_j$时，都需要计算到所有训练样本，因此在训练的过程会随着样本数量的增大而变的异常缓慢。针对这个缺点，提出了Stochastic Gradient Descent（随机梯度下降法，SGD）。
$$
\begin{align}
  J(\theta) &= \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})^2   \\
 \\ &= \frac{1}{m} \sum_{i=1}^m \frac{1}{2} (h_\theta(x^{(i)}) - y^{(i)})^2  \\
\\  &=\frac{1}{m} \sum_{i=1}^m cost(\theta,(x^{(i)},y^{(i)})) \\
\end{align}
$$
即$cost(\theta,(x^{(i)},y^{(i)})) = \frac{1}{2} (h_\theta(x^{(i)}) - y^{(i)})^2$

利用每个样本的loss函数对$\theta$求偏导得到对应的梯度，来更新$\theta$
$$
\begin{align}
Repeat \ \{ \\
\\ for \ \ i &= 1，2，\cdots ，m \ \{ \\
\\ \theta_j &= \theta_j - \alpha (h_\theta(x^{(i)}) - y^{(i)})x^i_j \\
\\ (j & = 0，1，\cdots，n)  \\
\\ & \quad \} \\
\\ \}
\end{align}
$$

Stochastic Gradient Descent 是通过每个样本来迭代更新一次，在遇到样本数量$m$很大的时候，有可能仅仅通过其中的几万条或者几千条的样本就可以将$\theta$给迭代到最优了，对比Batch Gradient Descent每迭代一次都要用到所有的训练样本，效率上会有很大的提升。

但是Stochastic Gradient Descent伴随的一个问题是训练噪音比Batch Gradient Descent要多，使得每次的Stochastic Gradient Descent迭代并不是都朝着全局最优的方向优化。

比较下Batch Gradient Descent跟Stochastic Gradient Descent，先给出Batch Gradient Descent的表达式
$$
x_{t+1} = x_t - \eta_t \nabla f(x_t)
$$
同样的，给出Stochastic Gradient Descent的表达式定义
$$
x_{t+1} = x_t - \eta_t g_t
$$
其中$g_t$就是Stochastic Gradient，它满足$E(g_t) = \nabla f(x_t)$，也就是说虽然Stochastic Gradient有一定的随机性，但是从数学期望上来说，它是等于正确的导数的，虽然每一次的迭代并不是朝着全局最优的方向，但是在大的整体上是朝着全局最优方向的，最终的结果也往往在全局最优解附近。

虽然说Stochastic Gradient Descent会伴随的随机噪声的问题，但是在实际表现中，Stochastic Gradient Descent的表现要Batch Gradient Descent的表现要好的多，加了噪声的算法表现反而更好。这一点主要取决于Stochastic Gradient Descent的一些不错的性质，它能够自动逃离鞍点，自动逃离比较差的局部最优点。

关于具体的一些证明可以看一下参考文献部分第三个条目。

### Mini-batch Gradient Descent

Mini-batch Gradient Descent（小批量梯度下降法，MBGD）

|           Method            |             The difference             |
| :-------------------------: | :------------------------------------: |
|   Batch Gradient Descent    | Use all $m$ examples in each iteration |
| Stochastic Gradient Descent |   Use one example in each iteration    |
| Mini-batch Gradient Descent |   Use $b$ examples in each iteration   |



$$
\begin{align}
\theta_j &= \theta_j - \alpha \frac{1}{b} \sum_{k=i}^{i+b-1}(h_\theta(x^{(k)})-y^{(k)})x_j^{(k)} \\ 
\\ &s.t. \quad i = i + b
\end{align}
$$

### 参考文献

- [Couresera Machine Learning Andrew-Ng Large Scale Machine Learning](https://www.coursera.org/learn/machine-learning/lecture/DoRHJ/stochastic-gradient-descent)
- [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
- [为什么说随机最速下降法(SGD)是一个很好的方法？](https://zhuanlan.zhihu.com/p/27609238)
- [BGD，SGD，MBGD的对比](http://blog.csdn.net/lilyth_lilyth/article/details/8973972)