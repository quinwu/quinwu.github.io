---
title: semantic-segmentation
tags:

---

### Semantic Segmentation using Fully Convolutional Networks

#### Segmantic Segmentation

- FCN 
- SegNet
- U-Net
- FC-Densenet E-Net & Link-Net
- RefineNet
- PSPNet
- Mask-RCNN
- DecoupledNet
- GAN-SS

#### Network Architectures

an **encoder** network followed by a **decoder** network

- The **encoder** network is usually is a pre-trained classification network like VGG/ResNet.
- The **decoder** network/mechanism is mostly where these architectures differ.

The task of **decoder** is to semantically project the discriminative feature ( lower resolution ) learnt by the encoder onto the piexl space (higher resolution) to get a dense classification.

Diffierent architectures employ different mechanisms as a part of the decoder mechanism.

- skip connection
- pyramid pooling



#### [Fully Convolution Networks ( FCNs )]()

| CVPR 2015 | Fully Convolutional Networks for segmantic Segmentation | Arxiv |
| --------- | ------------------------------------------------------- | ----- |
| CVPR 2015 | Fully Convolutional Networks for segmantic Segmentation | Arxiv |



#### Reference

- [Semantic Segmentation](https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html)