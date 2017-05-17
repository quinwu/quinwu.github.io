

####logistic regression cost function

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^m y^{(i)}\log(h_\theta(x^{(i)}) ) +(1-y^{(i)})\log(1-h_\theta(x^{(i)}))
$$

####neural network
$$
J(\Theta) = -\frac{1}{m}[\sum_{i=1}^m\sum_{k=1}^Ky_k^{(i)} \log(h_\Theta(x^{(i)}))_k + (1- y_k^{(i)})\log(1-(h_\Theta(x^{(i)}))_k) ]
$$

#### logistic regression cost function regularization
$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^m y^{(i)}\log(h_\theta(x^{(i)}) ) +(1-y^{(i)})\log(1-h_\theta(x^{(i)})) + \frac{\lambda}{2m} \sum_{j=1}^n \theta_j^2
$$

####neural network regularization
$$
J(\Theta) = -\frac{1}{m}[\sum_{i=1}^m\sum_{k=1}^Ky_k^{(i)} \log(h_\Theta(x^{(i)}))_k + (1- y_k^{(i)})\log(1-(h_\Theta(x^{(i)}))_k)] + \frac{\lambda}{2m} \sum_{l=1}^{L-1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_{l+1}}(\Theta_{ji}^{(l)})^2
$$

#### 代价函数与反向传播 Back propagation

一些标记:

* L 表示神经网络的总层数
* $S_4$表示第四层神经网络，不包括偏差单元`bias unit`
* k表示第几个输出单元



#### Feed forward computation  $h_\theta(x^{(i)})$

![Neural network model.)](C:\Users\nuctang\Desktop\1494568163(1).png)

```matlab
% computation h(x)
% input layerx
a1 = [ones(m,1) X];
% hidden layer
Z2 = a1*Theta1';
a2 = sigmoid(Z2);
a2 = [ones(size(a2,1),1) a2];
% output layer
Z3 = a2*Theta2';
a3 = sigmoid(Z3);
h = a3;
```


$$
J(\Theta) = -\frac{1}{m}[\sum_{i=1}^m\sum_{k=1}^Ky_k^{(i)} \log(h_\Theta(x^{(i)}))_k + (1- y_k^{(i)})\log(1-(h_\Theta(x^{(i)}))_k) ]
$$

```matlab

```

![back propagation](C:\Users\nuctang\Desktop\TIM截图20170512155549.png)

```matlab

```

