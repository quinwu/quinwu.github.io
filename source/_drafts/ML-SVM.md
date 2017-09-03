---
title: 小记 SVM
date: 2017-05-25 16:43:02
categorise: 技术向
tags:
  - Machine Learning
  - 小记系列
---

支持向量机（support vector machines）SVM是一种二类分类模型。它的基本模型是定义在特征空间上的间隔最大的线性分类器，间隔最大使它有别于感知机，支持向量机还包括核技巧（kernal trick）这使它成为实质上的非线性分类器。

### 间隔最大化

分类超平面对应于方程$w \cdot x + b = 0 $

当训练数据集线性可分的时候，存在无穷个分离超平面将两类数据正确分开。线性可分支持向量机利用间隔最大化求最优分离超平面，这时，解是唯一的。

#### 函数间隔

一般的，一个点距离分类超平面远近可以表示分类分类预测的确信程度。在分类超平面$w \cdot x + b = 0 $确定的情况下，$|w \cdot x + b|$ 能够相对的表示点$x$距离分类超平面的远近。$w \cdot x + b $的符号与标记$y$的符号是否一致能够表示分类是否预测正确。因此可以用 $y(w \cdot x + b)$来表示分类的正确性及确信度，这就是函数间隔（function margin）

> 对于给定的训练数据集$T$和超平面$(w,b)$，定义超平面$(w,b)$关于样本点$(x_i,y_i)$的函数间隔为
> $$
> \hat{\gamma_i} = y_i(w \cdot x_i + b)
> $$
> 定义超平面$(w,b)$关于训练集$T$的函数间隔为超平面$(w,b)$关于$T$中所有样本点$(x_i,y_i)$的函数间隔之最小值，
> $$
> \hat{\gamma} = \min_{i=1,\cdots,N} \hat{\gamma_i}
> $$
>

#### 几何间隔

函数间隔可以表示分类的正确性与确信度，但是如果在选择超平面的时候成比例的改变$w$，$b$，超平面并没有发生改变，函数间隔却发生了变化。这个时候，就需要我们对超平面的法向量$w$加以约束，使得间隔是确定的，这是的函数间隔就变成了几何间隔（geometric margin）

>其中$||w||$为$w$的$L2$范数
>
>对于给定的训练数据集$T$和超平面$(w,b)$，定义超平面$(w,b)$关于样本点$(x_i,y_i)$的几何间隔
>$$
>{\gamma_i} = y_i \lgroup \frac{w}{\| w \|} \cdot x_i + \frac{b}{\| w \|} \rgroup
>$$
>其中$\|w\|$为$w$的$L2$范数
>
>定义超平面$(w,b)$关于训练集$T$的函数间隔为超平面$(w,b)$关于$T$中所有样本点$(x_i,y_i)$的几何间隔之最小值，
>$$
>\gamma = \min_{i=1,\cdots,N}\gamma_i
>$$
>

因此函数间隔$\hat{\gamma_i}​$跟几何间隔$\gamma_i​$的关系为
$$
\gamma_i = \frac{\hat{\gamma_i}}{\| w \|}
$$

$$
\gamma = \frac{\hat{\gamma}}{\| w\|}
$$

#### 最大间隔分离超平面

$$
\max_{w,b} \quad \gamma
$$

$$
s.t. \quad y_i \lgroup \frac{w}{\| w \|} \cdot x_i + \frac{b}{\| w \|} \rgroup \ge \gamma ,\quad i = 1,2,\cdots,N
$$

### 对偶问题

$$
\max_{w,b} \quad \frac{2}{\| w \|}
$$

$$
s.t. \quad y_i(w^{T}x_i + b) \ge1,\quad i=1,2,\cdots,N
$$


$$
\min_{w,b} \quad \frac{1}{2} \| w \|^2
$$

$$
s.t. \quad y_i(w^{T}x_i + b) \ge1,\quad i=1,2,\cdots,N
$$


$$
L(w,b,\alpha) = \frac{1}{2}\|w\|^2 + \sum^N_{i=1}\alpha_i(1-y_i(w^Tx_i+b))
$$
其中$\alpha=(\alpha_1;\alpha_2;\cdots;N)$
$$
w = \sum_{i=1}^N\alpha_iy_ix_i
$$

$$
0=\sum^N_{i=1}\alpha_iy_i
$$

### SMO算法

SMO表示序列最小优化（Sequential Minimal Optimization）。SMO算法是将大优化问题分解为许多小优化问题来求解。SMO算法的目标是求出一系列$\alpha$跟$b$，一旦求出了$\alpha$，就很容易计算出权重向量$w$。

### 核函数

### 软间隔

### 支持向量回归



