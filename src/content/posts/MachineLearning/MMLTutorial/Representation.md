---
title: 多模态深度学习入门之Representation
published: 2024-08-23
description: '多模态深度学习入门之Representation'
image: ''
tags: [多模态，导论]
category: '导论'
draft: true
---

##  Challenge 1：Representation

定义：反映各个元素之间、跨不同模式的跨模式交互的学习表示

有三个子挑战：

![image-20240823142124081](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20240823142124081.png)

### Sub-Challenge 1：Representation Fusion

定义：学习一个联合表示，该表示模型化了不同模态个体元素之间的跨模态交互。也就是将同源或者异源的数据进行Fusion在一起，变为一个特征。

当我们使用Unimodal Encoder的时候，通常如图

![basic Fusion](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20240823143255293.png)

#### Basic Fusion

基本的融合方法包含**加法融合，乘法融合和tensor融合**，加法融合就是简单的加起来，乘法融合包含基本的乘法融合（$$w(x_a×x_b)$$）和双线性融合($$w(x_a^T*x_b)$$)

tensor融合示意图如下，为了减少tensor fusion带来的维度爆炸问题，我们可以使用**低秩融合**的方法。 ![image-20240823144156039](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20240823144156039.png)

以上这些内容都是多项式形式的，没有跳出多项式的框架。$$Z = w_1 x_A + w_2 x_B + w_3 x_C + w_4 (x_A × x_C) + w_5 (x_A × x_C) + w_6 (x_B × x_C) + w_7 (x_A × x_B × x_C)$$

#### Gated Fusion

![image-20240823145435338](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20240823145435338.png)

$$z=g_a(x_a,x_b)*x_a+g_b(x_a,x_b)*x_b$$,这里的ga和gb可以看做是attention function，来决定哪个的权重更大，可以使用liner，nonliner和核函数.

#### Nonliner Fusion

一般得到fusion后再接多个全链接层，现在我们把fusion和全连接结合在一起，就是nonliner fusion，这是一种早期融合

#### Complex Fusion

这是一个困难的问题，想怎么做都可以，比如随机交换两个CNN网络的一些层

## Sub-Challenge 2: Representation Coordination

定义：学习通过跨模式交互协调的多模式上下文化表示

需要网络学习两个特征之间有多么相似$$L = g(f_A(\triangle), f_B(\bullet))$$，比如我们可以使用余弦相似度函数，Kernal相似度函数，CCA分析。

使用这种方法的例子比如自动编码器（对比学习），让标签一致的损失小，让标签不一致的损失大

还有视觉语义嵌入：![image-20240823151447474](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20240823151447474.png)

## Sub-Challenge 3:Representation Fisson

定义：学习反映多模式内部结构（例如数据分解或集群）的一组新的表示

![image-20240823151836824](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20240823151836824.png)

