---
title: "[[StableDiffusion3.pdf|StableDiffusion3]]"
description: ""
image: ""
published: 2024-07-12
tags:
  - 机器学习
  - 论文阅读
category: 论文阅读
draft: true
---

> [!PDF|yellow] [Page.2](StableDiffusion3.pdf#page=2&selection=25,0,28,0&color=yellow)
>
> > While specifying a forward path from data to noise leads to efficient training, it also raises the question of which path to choose.
>
>虽然指定前进的路径可以高效训练,但是如何指定路径?这个问题对于扩散模型非常重要.

例如,如果没有成功的从噪声中全部去噪,那么就会导致伪影等问题,此外,前向过程也影响着反向过程,从而影响采样效率。与需要多个积分步骤来模拟过程的曲线路径相比，直线路径可以通过单一步骤模拟，且不易积累误差。由于每个步骤都对应神经网络的评估，这对采样速度有直接影响。

一种特定的选择前向路径的方法叫做**Rectified-flow**,能够把数据和噪声在一个直线上连起来.虽然这个类别有更好的理论基础,但是并没有被广泛的应用.迄今为止,一些实验表名了他的一些有点,但是大多数局限于类条件模型.在这项工作中，我们通过在修正流模型中引入噪声尺度的重加权来改变这一点，这与噪声预测扩散模型类似。通过大规模研究，我们将新公式与现有的扩散公式进行比较，并展示其优势。

传统的t2i生成之后,文本被直接的输入到模型中是不理想的,并且设计了一种新的架构,结合了两者的token,让和他们的信息流可以互相移动