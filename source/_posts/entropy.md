---
title: 小记 Entropy
date: 2017-10-30 21:51:59
tags:
  - Mathematics
  - 小记系列
categories: 笔记
---

#### 熵（entropy）

在信息论与概率统计中，熵是表示随机变量不确定性的度量。设$X$是一个取有限个值的离散随机变量，其概率分布为：

<!--more--> 
$$
P(X=x_i) = p_i,\quad i=1,2,\cdots,n
$$
则随机变量的$X$的熵定义为：
$$
H(X) = -\sum^n_{i=1}p_ilog(p_i)
$$
熵越大，随机变量的不确定性也就越大。

若$X$是连续型的随机变量，其熵定义为：
$$
H(X) = -\int_{x \in X} p(x)\log(p(x))
$$

#### 相对熵（relative entropy）

相对熵又称KL散度（Kullback-Leibler divergence），KL距离，是两个随机分布间的距离的度量。记为$D_{KL}=(p||q)$，他度量当真实分布为$p$时，假设分布$q$的无效性。
$$
\begin{align}
D_{KL} (p||q) &= E_p[{\log\frac{p(x)}{q(x)}}] \\
\\ &= \sum_{x \in X} p(x) \log (\frac{p(x)}{q(x)}) \\
\\ &= \sum_{x \in X} [p(x)\log(p(x)) - p(x)\log(q(x))] \\
\\ &= -H(p) - \sum_{x \in X} p(x)\log(q(x)) \\
\\ &= -H(p) + E_p[-\log(q(x))] \\
\\ &= H_p(q) - H(p)
\end{align}
$$

#### 交叉熵（cross entropy）

假设有两个分布$p$，$q$，则在它们给定的样本集上的交叉熵定义为:
$$
\begin{align}
CEH(p,q) &= E_p[-\log(q)] \\
\\ &= -\sum_{x \in X}p(x)\log(q(x)) \\
\\ &= H_p(q) \\
\\ & = H(p) + D_{KL}(p||q)
\end{align}
$$
