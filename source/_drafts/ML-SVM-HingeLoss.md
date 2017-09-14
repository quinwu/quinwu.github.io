---
title: ML-SVM-HingeLoss
tags:

---



### SVM 以Hinge Loss 为例

##### Hinge Loss Function

$$
L_i = \sum_{j\ne y_i} [\max (0,x_iw_j - x_iw_{y_i} + \Delta)]
$$

- $i$  iterates over all N examples .
- $j$ iterates over all C classes .
- $L_i$ is loss for classifiying a single example $x_i$ (row vector) .
- $w_j$ is the weights (column vector) for computing the score of class $j$ .
- $y_i$ is the index of the correct class of $x_i$ .
- $\Delta$ is a margin parameter

$$
\nabla_{w} L_i 
  =
  \begin{bmatrix}
    \frac{dL_i}{dw_1} & \frac{dL_i}{dw_2} & \cdots & \frac{dL_i}{dw_C} 
  \end{bmatrix}
  = 
  \begin{bmatrix}
    \frac{dL_i}{dw_{11}} & \frac{dL_i}{dw_{21}} & \cdots & \frac{dL_i}{dw_{y_i1}} & \cdots & \frac{dL_i}{dw_{C1}} \\
    \vdots & \ddots \\
    \frac{dL_i}{dw_{1D}} & \frac{dL_i}{dw_{2D}} & \cdots & \frac{dL_i}{dw_{y_iD}} & \cdots & \frac{dL_i}{dw_{CD}} 
  \end{bmatrix}
$$

$$
\begin{align*}
L_i = &\max(0, x_{i1}w_{11} + x_{i2}w_{12} \ldots + x_{iD}w_{1D} - x_{i1}w_{y_i1} - x_{i2}w_{y_i2} \ldots - x_{iD}w_{y_iD} + \Delta) + \\
 &\max(0, x_{i1}w_{21} + x_{i2}w_{22} \ldots + x_{iD}w_{2D} - x_{i1}w_{y_i1} - x_{i2}w_{y_i2} \ldots - x_{iD}w_{y_iD} + \Delta) + \\
&\quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \vdots \\
&\max(0, x_{i1}w_{C1} + x_{i2}w_{C2} \ldots + x_{iD}w_{CD} - x_{i1}w_{y_i1} - x_{i2}w_{y_i2} \ldots - x_{iD}w_{y_iD} + \Delta)
\end{align*} 
$$

For a general case , if $(x_iw_1 - x_iw_{y_i} + \Delta )\gt 0$
$$
\begin{equation}
\frac{dL_i}{dw_{11}} = x_{i1}
\end{equation}
$$
using an indicator function:
$$
\begin{equation}
\frac{dL_i}{dw_{11}} = \mathbb{1}(x_iw_1 - x_iw_{y_i} + \Delta > 0) x_{i1}
\end{equation}
$$
同样的
$$
\begin{equation}
\frac{dL_i}{dw_{12}} = \mathbb{1}(x_iw_1 - x_iw_{y_i} + \Delta > 0) x_{i2} \\
\frac{dL_i}{dw_{13}} = \mathbb{1}(x_iw_1 - x_iw_{y_i} + \Delta > 0) x_{i3} \\
\vdots \\
\frac{dL_i}{dw_{1D}} = \mathbb{1}(x_iw_1 - x_iw_{y_i} + \Delta > 0) x_{iD}
\end{equation}
$$
因此
$$
\begin{align*}
\frac{dL_i}{dw_{j}} &= \mathbb{1}(x_iw_j - x_iw_{y_i} + \Delta > 0)
  \begin{bmatrix}
  x_{i1} \\
  x_{i2} \\
  \vdots \\
  x_{iD}
  \end{bmatrix}
\\
&= \mathbb{1}(x_iw_j - x_iw_{y_i} + \Delta > 0) x_i 
\end{align*}
$$
对于 $j = y_i$的特殊情况
$$
\begin{equation}
\frac{dL_i}{dw_{y_{i1}}} = -(\ldots) x_{i1}
\end{equation}
$$
The coefficent of $x_{i1}$ is the number of classes that meet the desire margin. Mathematically speaking, $\sum _{j\ne y_i}  \mathbb{1} (x_iw_j - x_iw_{y_i} + \Delta \gt 0)$

因此
$$
\begin{align*}
\frac{dL_i}{dw_{y_i}} &= - \sum_{j\neq y_i} \mathbb{1}(x_iw_j - x_iw_{y_i} + \Delta > 0)
  \begin{bmatrix}
  x_{i1} \\
  x_{i2} \\
  \vdots \\
  x_{iD}
  \end{bmatrix}
\\
&= - \sum_{j\neq y_i} \mathbb{1}(x_iw_j - x_iw_{y_i} + \Delta > 0) x_i 
\end{align*}
$$

### 