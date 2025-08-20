---
title: "[[BLIP2.pdf|BLIP2]]"
description: ""
image: ""
tags:
  - 机器学习
  - 论文阅读
category: 论文阅读
draft: true
published: 2024-10-21
last modified: 2024-10-21
---

# Abstract

Vision and Languague 的预训练模型的训练开销越来越大,本文提出了 BLIP2,一个高效的预训练策略,利用现成的冻结的预训练图像编码器和大语言模型来进行视觉 - 语言预训练,BLIP-2 使用一个轻量级的 Querying-transofrmer 来进行视觉和语言模态之间的链接.第一阶段从冻结的图像编码器中启动视觉 - 语言表示学习，第二阶段从冻结的语言模型中启动视觉到语言的生成学习。我们的 BLIP2 又快又好.

## Introduction

VLP 研究目前吸引了很多的注意,预训练模型也越来越大,但是需要很大的计算量.

> [!PDF|yellow] [Page.1](BLIP2.pdf.md#page=1&selection=55,0,59,19&color=yellow)
>
> > Vision-language research sits at the intersection between vision and language, therefore it is naturally expected that vision-language models can harvest from the readilyavailable unimodal models from the vision and natural language communities.
>
>VLP 在 v 和 l 之间,因此自然希望模型利用目前已经训练好的单模态模型.

> [!PDF|red] [Page.1](BLIP2.pdf.md#page=1&selection=139,0,139,44&color=red)
>
> > it is key to facilitate cross-modal alignment
>
>为了利用预训练的单模态模型,重要的是跨模态对齐

但是 LLM 在预训练阶段没有见过图片,隐藏已冻结参数使得夸模态对齐变得很困难.有人尝试过使用一个 Image2Text 的 loss,但是这被证明不够

为了在冻结参数的情况下达到高效的 vision language 对齐,我们提出了 Q-Former.他是一组轻量级的转换器,能够吧视觉特征使用一组可学习的参数提取出来.他也是 img 和 txt 之间的信息瓶颈,把最重要的视觉特征输入到 LLM 中. 我们在第一个阶段学习和文本最有关的视觉特征,在第二个预训练阶段，我们通过将 Q-Former 的输出连接到一个冻结的 LLM 来执行视觉到语言的生成学习，并训练 Q-Former，使其输出的视觉表示可以被 LLM 解释。

![image.png](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/20241021135603.png)

## Method

> [!PDF|red] [Page.2](BLIP2.pdf.md#page=2&selection=112,8,118,48&color=red)
>
> > In order to bridge the modality gap, we propose a Querying Transformer (Q-Former) pre-trained in two stages: (1) vision-language representation learning stage with a frozen image encoder and (2) vision-to-language generative learning stage with a frozen LLM. This section first introduces the model architecture of Q-Former, and then delineates the two-stage pre-training procedures

1. 使用冻结的 image encoder 学习视觉语言表征学习
2. 使用冻结的 LLM 学习视觉语言表征学习

### Model Architecture

他可以从图像 encoder 中提取固定数量的输出而不用考虑模型的输入分辨率.Q-Former 包含了两个 Transformer 子模块共享了同一个自注意力层.

1. 一个 image Transformer 从冻结的 encoder 中提取视觉特征
2. 一个 text Transformer 能作为 text encoder 和 decoder

![image.png](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/20241021140820.png)

使用 Bert 的参数进行初始化,

### 从冻结的 Image Encoder 中开始 Vl 表征学习

在表征学习阶段,我们把 Q-Former 和冻结的 image-encoder 链接在一起并且使用 image-text 图像对进行预训练,我们希望 Q-former 能够从图像中提取出文本的绝大多数信息.我们同时优化了三个训练目标并使用不同的 attention 屏蔽策略来控制他们的交互. Z 是输出的 query 表示

**Image-Text 对比学习**: 让两者的互信息最多,我们要将图像转换器的输出查询 Z 和来自文本转换器的 t 对齐,其中 t 是 CLS 的输出嵌入.因为 z 包含多个输出嵌入 (每个查询向量有一个) 我们首先计算了输出和 t 之间的成对相似性,然后选择最高的作为图像 - 文本相似性.为了避免信息泄露，我们采用单模态自注意力掩码，确保查询和文本之间无法直接看到彼此。由于使用了冻结的图像编码器，与端到端方法相比，我们可以在每个 GPU 上适应更多样本。因此，我们在 BLIP 中使用批内负样本，而不是动量队列。

> Cross attention
> 交叉注意力的思想是使一个序列能够“关注”另一个序列
> 一个序列作为 Q 定义了输出序列的长度,另一个序列提供 K&V
> 两个序列的维度必须一致,此外都一样
