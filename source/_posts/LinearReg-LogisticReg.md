title: 小记 Linear Regression 与 Logistic Regression

data: 2017-5-3 20:55:50

categories: 技术向

tags:  

- Machine Learning
- 小记系列

---

> Linear Regression 线性回归
>
> Logistic Regression 对数几率回归

回归的本身是一种基于数据的建模，一般而言，`线性回归（Linear Regression）`相对比较简单，而`对数几率回归（Logistic Regression）`的内容要丰富很多。

线性回归很简单，给定一个样本集合  $D=(x_1,y_1),(x_2,y_2),\cdots,(x_m,y_m)$  这里的$x_i,y_i$都可以是高维向量，可以找到一个线性模拟$f(x_i)=wx_i+b$，只要确定了$w$跟$b$，这个线性模型就确定了，如何评估样本$y$与你的$f(x)$之间的差别，最常用的方法是最小二乘法。

也就是说，我们用一条直线去模拟当前的样本，同时预测未来的数据，即我们给定某一个$x$值，根据模型可以返回给我们一个$y$值，这就是线性回归。

对于`线性回归（Liner Regression）`而言，我们的任务是要确定出参数$w$跟$b$，使得线性函数$f(x_i)=wx_i+b$对于数据的拟合度足够高。这里我们用最小二乘法来评估样本$y$与估计值$f(x_i)$的拟合程度

为了表示方便，我们使用如下的形式表示假设函数，为了方便  ${h_{\theta}(x)}$   也可以记作  $h(x)$。

$$h_\theta(x) = \theta\_0 + \theta\_1x$$



> 代价函数 （cost function）



$${\mathop{min}\_{\theta\_0,\theta\_1}}   \frac{1}{2m}  \sum\_{i=0}^{m} {(h\_\theta(x^{(i)}) - y^{(i)})}^2 $$

$${\mathop{min}\_{\theta\_0,\theta\_1}}  {J(\theta\_0,\theta\_1) } $$

$$ {J(\theta\_0,\theta\_1) } $$

$$ \frac{1}{2m}\sum\_{i=0}^{m} $$

$ {(h\_\theta(x^{(i)} )  - y^{(i)}) }^2 $



$${\mathop{min} \_{\theta\_0,\theta\_1}}  \frac{1}{2m}\sum\_{i=0}^{m}  {(h\_\theta(x^{(i)} )  - y^{(i)}) }^2 $$



> J

$${J(\theta\_0,\theta\_1) } = \frac{1}{2m}\sum\_{i=0}^{m}{(h_\theta(x^{(i)} )  - y^{(i)}) }^2$$



$${\mathop{min}\_{\theta\_0,\theta\_1}} {J(\theta\_0,\theta\_1) } $$



> 梯度下降



Have some function  ${J(\theta_0,\theta_1) }$

want  ${\mathop{min}\_{\theta\_0,\theta\_1}} {J(\theta\_0,\theta\_1) } $

Outline:

* Start with some $\theta\_0,\theta\_1$
* keep changing  $\theta\_0,\theta\_1$ to reduce  ${J(\theta\_0,\theta\_1) }$ until we hopefully end up at ${\mathop{min}\_{\theta\_0,\theta\_1}} {J(\theta\_0,\theta\_1) } $

> Gradient descent algorithm

repeat until  convergence {

​	$\theta\_j$ := $\theta\_j - \alpha \frac{\partial}{\partial{\theta\_j}} {J(\theta\_0,\theta\_1) }$  *for j=0  and  j=1*

}



> j = 0

$$\frac{\partial}{\partial{\theta\_0}} {J(\theta\_0,\theta\_1)} =  \frac {1} {m} \sum\_{i = 1} ^ {m}{(h\_\theta(x^{(i)}) - y^{(i)})} $$



> j = 1

$$\frac{\partial}{\partial{\theta\_1}} {J(\theta\_0,\theta\_1)} =  \frac {1} {m} \sum\_{i = 1} ^ {m}{(h\_\theta(x^{(i)}) - y^{(i)})} x^{(i)} $$



  $$\theta\_0 := \theta\_0 - \alpha \frac {1} {m} \sum\_{i = 1} ^ {m}{(h\_\theta(x^{(i)}) - y^{(i)})} $$

  $$\theta\_1 := \theta\_1 - \alpha \frac {1} {m} \sum\_{i = 1} ^ {m}{(h\_\theta(x^{(i)}) - y^{(i)})}  x^{(i)} $$

----



> Linear Regression with Mulitiple Variables



$$h_\theta(x) = \theta\_0 + \theta\_1x$$

$$h_\theta(x) = \theta\_0 + \theta\_1 x\_1+ \theta\_2 x\_2+ \cdots+\theta\_n x\_n$$



> 差列向量的公式



$$ h\_\theta(x) = \theta\_0 x\_0 +  \theta\_1 x\_1 + \theta\_2 x\_2 +\cdots+ \theta\_n x\_n  =  \theta^T x$$









----



接下来介绍下`广义线性回归`，也很简单，我们不在只用线性函数来模拟数据，而是在外层添加了一个单调可微函数$g(z)$，即$f(x_i) = g(wx_i+b) $ ，如果 $ g=ln(x) $，则这个`广义线性回归`就变成了`对数线性回归`，其本质就是给原来线性变换加上了一个非线性变换，使得模拟的函数有非线性的属性。但本质上的参数还是线性的，主体是内部线性的调参。

`对数几率回归（Logistic Regression）`不是解决回归问题的，而是解决分类问题的。目的是要构造出一个分类器（Classifier）。`对数几率回归（Logistic Regression）`的关键并不在于回归，而在于对数几率函数。

对一个简单的二分类问题，实际上是样本点或者预测点到一个值域为 $[0,1]$的函数，函数值表示这个点分在`正类（postive）`或者`反类（negtive）`的概率，如果非常可能是`正类（postive）`，那么其概率值就逼近与1，如果非常可能是`反类（negtive）`其概率值就逼近与0。

> //补周志华 单位阶跃函数与对数几率函数的图

于是我们构造一个`sigmoid`函数 $$y=\frac{1}{1+ \mathrm{e}^{-z} } $$，



$$h\_\theta (x) = g (\theta^T x)$$

$$g(z) = \frac {1}{1 + \mathrm{e}^{-z}}$$

$$h\_\theta (x) = \frac {1}{1 + \mathrm{e}^{-\theta^T x}}$$







$$ J(\theta) = \frac{1}{2m}  \sum\_{i=0}^{m} {(h\_\theta(x^{(i)}) - y^{(i)})}^2 $$ 



$$ Cost (h\_\theta{x^(i)},y^(i)) = \frac{1}{2} (h\_\theta(x^{(i)}) - y^{(i)})}^2 $$





$$Cost(h\_\theta(x),y) = \frac{1}{2}(h\_\theta - y) ^ 2$$

$$Cost(h\_\theta,y) = -y\log(h\_\theta(x)) - (1-y) \log(1-h\_\theta(x))$$



> cost function



$${J(\theta)=-\frac{1}{m}\left[\sum\_{i=1}^my^{(i)}log(h\_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)}))\right]}$$











