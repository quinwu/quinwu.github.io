---
title: Matrix-derivation
tags:

---

向量求导的两种形式

- 分子布局 numerator layout
- 分母布局 denominator layout

以$\frac{\partial y}{\partial x}$为例，假设

###### 向量对标量

###### 标量对向量

###### 向量对向量

###### 矩阵对标量

###### 标量对矩阵




$$
df = tr(\frac{\partial f}{\partial X}^T dX)
$$




### 常用的矩阵微分运算法则

- 加减法：$d(X \pm Y) = dX \pm dY$
- 矩阵乘法：$d(XY) = dXY + XdY$
- 转置：$d(X^T) = (dX)^T$
- 迹：$dtr(X) = tr(dX)$
- 逆：$ dX^{-1} = -X^{-1}dXX^{-1}$  可在$XX^{-1} = I$两侧求微分来证明
- 行列式：$d|X|=tr(X^{\ast}dX)$



### 参考文献

- [闲话矩阵求导](http://xuehy.github.io/2014/04/18/2014-04-18-matrixcalc/)
- [矩阵求导术上](https://zhuanlan.zhihu.com/p/24709748)
- [矩阵求导术下](https://zhuanlan.zhihu.com/p/24863977)
- [矩阵求导](http://www.junnanzhu.com/?p=141)
- [矩阵微分](http://www.qiujiawei.com/matrix-calculus-1/)
- [矩阵微分](http://www.cnblogs.com/xuxm2007/p/3332035.html)
- https://zhuanlan.zhihu.com/p/25202034

