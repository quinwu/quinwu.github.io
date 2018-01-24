---
title: object-detection
tags:
---

### R-CNN

- 借助一个可以生成约2000个 region proposal 的 Selective Search 算法，R-CNN 可以对输入图像进行扫描，来获取可能出现的目标。
- 在每个 region proposal 上都运行一个 CNN 。
- 将每个 CNN 的输出都输入进 
  - 一个支持向量机SVM，对上述区域进行分类
  - 一个线性回归器，收缩目标周围的边界框，前提是这样的目标存在。

### SPPNet

### Fast R-CNN

- 在通过 Selective Search 选择 region proposal 之前，先对图像执行提取特征的工作，通过这种方法，后面只用对整个图像使用一个 CNN，之前的 R-CNN 网络需要在2000个重叠区域上分别运行2000个 CNN。
- 将支持向量机SVM替换为一个 Softmax 层，将神经网络进行拓展以用于预测工作。

### Faster R-CNN

- 引入 RPN( region proposal net ) 用一个快速神经网络代替了之前的慢速选择搜索算法 ( Selective Search )

RPN的工作原理：

- 在最后的 feature map 上使用一个 3*3 的滑动窗口在其上滑动，将 feature map 映射到一个更低的维度上（如 256 维）。
- 在每个滑动窗口的位置上，RPN 都可以基于 k 个固定比例的 anchor box（默认的边界框）生成多个可能的区域。
- 每个 region proposal 都由两部分组成：
  - 该区域的 objectness 分数
  - 4 个表征该区域边界框的坐标

### R-FCN

Region-based Fully Convolutional Net

### YOLO

You Only Look Once

### SSD

Single-Shot Detector

