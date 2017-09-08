---
title: 小记 SVM
tags:
  - Machine Learning
  - 小记系列
categorise: 笔记
date: 2017-09-08 16:43:02
---


支持向量机（support vector machines）SVM是一种二类分类模型。它的基本模型是定义在特征空间上的间隔最大的线性分类器，间隔最大使它有别于感知机，支持向量机还包括核技巧（kernal trick）这使它成为实质上的非线性分类器。<!--more-->

### 间隔最大化

分类超平面对应于方程$w \cdot x + b = 0 $

当训练数据集线性可分的时候，存在无穷个分离超平面将两类数据正确分开。线性可分支持向量机利用间隔最大化求最优分离超平面，这时，解是唯一的。

#####函数间隔

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

#####几何间隔

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

#####最大间隔分离超平面

$$
\max_{w,b} \quad \gamma
$$

$$
s.t. \quad y_i \lgroup \frac{w}{\| w \|} \cdot x_i + \frac{b}{\| w \|} \rgroup \ge \gamma ,\quad i = 1,2,\cdots,N
$$

#####对偶问题

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

#####SMO算法

SMO表示序列最小优化（Sequential Minimal Optimization）。SMO算法是将大优化问题分解为许多小优化问题来求解。SMO算法的目标是求出一系列$\alpha$跟$b$，一旦求出了$\alpha$，就很容易计算出权重向量$w$。

### 核函数

在上面的最大分隔超平面的讨论中，假设训练样本的数据集都是线性可分的，存在一个划分超平面可以将数据集给正确分类。 对于非线性的问题往往不好求解，我们还是希望能够转化为线性分类问题来求解，通过将样本从原始空间映射到一个更高的特征空间上，使得训练样本在高维的特征空间上线性可分。

> 幸运的是，如果原始空间是有限维的，那么一定存在一个高维特征空间使得样本可分。

令$\phi(x)$为将$x$映射后的特征向量，因此在高维特征空间中划分超平面的模型函数为
$$
f(x) = w^T\phi(x) + b
$$
其中$w$，$b$为参数模型。类比一般情况下的支持向量，最大间隔分离超平面的模型为
$$
\min_{x,b} \frac{1}{2} ||w||^2
$$

$$
s.t. \quad y_i(w^T\phi(x_i) + b) \ge 1,\quad i=1,2,\cdots,m
$$

其对偶问题是
$$
\max_{\alpha} \sum^{m}_{i=1}\alpha_i - \frac{1}{2}\sum^m_{i=1}\sum^m_{j=1}\alpha_i \alpha_j y_i y_j\phi(x_i)^T\phi(x_j)
$$

$$
s.t. \quad \sum^m_{i=1} \alpha_i y_i = 0，\quad \alpha_i \ge0,\quad i =1,2,\cdots,m
$$

求解上式涉及到$\phi(x_i)^T\phi(x_j)$的计算，这是样本$x_i$与$x_j$映射到特征空间之后的内积，特征空间的维数很高，直接求解上式的难度很大，为了避开这个障碍，我们可以设想这样的一个函数
$$
\kappa(x_i,x_j) = \langle\phi(x_i),\phi(x_j)\rangle = \phi(x_i)^T\phi(x_j)
$$
使得$x_i$与$x_j$在特征空间的内积等于它们在原始空间上通过函数$\kappa(\cdot,\cdot)$来计算，这里的函数$\kappa(\cdot,\cdot)$就是核函数（kernel function）。将核函数带入最大分隔超平面模型后，
$$
\max_{\alpha} \sum^{m}_{i=1}\alpha_i - \frac{1}{2}\sum^m_{i=1}\sum^m_{j=1}\alpha_i \alpha_j y_i y_j \kappa(x_i,x_j)
$$

$$
s.t. \quad \sum^m_{i=1} \alpha_i y_i = 0，\quad \alpha_i \ge0,\quad i =1,2,\cdots,m
$$

求解后，可以得到 
$$
\begin{align}
f(x) &= w^T\phi(x) + b \\
\\&= \sum_{i=1}^m\alpha_iy_i\phi(x_i)^T\phi(x) +b \\
\\ & = \sum_{i=1}^m \alpha_i y_i \kappa(x,x_i) + b
\end{align}
$$

##### 几种常见的核函数

| 名称       | 表达式                                      | 参数                                   |
| -------- | ---------------------------------------- | ------------------------------------ |
| 线性核      | $$\kappa(x_i,x_j)= x^T_ix_j$$            |                                      |
| 多项式核     | $\kappa(x_i,x_j) = (x^T_ix_j)^d$         | $d\ge1$为多项式的次数                       |
| 高斯核      | $$\kappa(x_i,x_j) = \exp(-\frac{||x_i-x_j||^2}{2 \sigma^2})$$ | $\sigma>0$为高斯核的带宽                    |
| 拉普拉斯核    | $$\kappa(x_i,x_j) = \exp(-\frac{||x_i-x_j||}{ \sigma})$$ | $\sigma>0$                           |
| Sigmoid核 | $$\kappa(x_i,x_j) = \tanh(\beta x_i^Tx_j+\theta)$$ | $\tanh$为双曲正切函数，$\beta >0 ,\theta <0$ |

> 上表来自周志华老师的《机器学习》一书

常见核函数的组合也是核函数

> todo

### 软间隔

之前在讨论函数间隔几何间隔的过程中，我们都是假设训练样本在样本空间中是线性可分的，一定存在一个超平面能够将两类不同的数据给完全的分隔开，但是在实际中的数据集往往不是这样的理想，这就允许我们的向量机在一定的样本上出错，因此引入了软间隔（soft marigin）的概念。

> 少一个软间隔的图

之前介绍的向量机形式都是要求所有的样本必须满足约束条件$y_i(w^Tx_i+b) \ge 1$，所有的样本都必须正确的划分，这种形式的向量机成为硬间隔（hard marigin），而软间隔（soft marigin）则允许某些样本不用满足上述的约束条件。

同样，在最大化软间隔的同时，我们也是希望不满足上述约束条件的样本尽可能的少，因此优化函数可以写为
$$
\min _{w,b}\frac{1}{2} ||w||^2 + C\sum^m_{i=1} \ell_{0/1}(y_i(w^Tx_i+b)-1)
$$
其中的$C > 0$是一个常数 ，$\ell_{0/1}$是`0/1损失函数`


$$
\begin{equation}
\ell_{0/1}(z) =  
\left \{
\begin{array}
 {r@{\quad:\quad}l} 
 1 & z < 0\\ 
 0  & else  
\end{array}
\right.
\end{equation}
$$

> 折页损失（hinge loss）

$$
\ell_{hinge}(z) = \max(0,1-z)
$$

若采用hinge loss，则优化函数可以写为
$$
\min _{w,b}\frac{1}{2} ||w||^2 + C\sum^m_{i=1}\max(0,1-y_i(w^Tx_i + b))
$$

> 指数损失（exponential loss）

$$
\ell_{exp}(z) = \exp(-z)
$$

> 对率损失（logistic loss ）

$$
\ell_{log}(z) = log(1+\exp(-z))
$$
