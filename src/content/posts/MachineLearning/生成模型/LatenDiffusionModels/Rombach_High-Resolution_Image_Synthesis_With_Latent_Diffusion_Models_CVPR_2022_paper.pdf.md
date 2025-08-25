---
title: "[[Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf|Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper]]"
description: ""
image: ""
published: 2024-07-12
tags:
  - 机器学习
  - 论文阅读
category: 论文阅读
draft: true
---

# Abstract

> [!PDF|yellow] [Page.1](Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf#page=1&selection=37,0,42,0&color=yellow)
>
> > However, since these models typically operate directly in pixel space, optimization of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evaluations.
>
>直接在像素空间训练开销大且消耗昂贵,因此,我们使用强大的预训练自动编码器赖在隐藏空间中进行训练.

> [!PDF|yellow] [Page.1](Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf#page=1&selection=48,3,49,61&color=yellow)
>
> > reach a near-optimal point between complexity reduction and detail preservation, greatly boosting visual fidelity
>
>在复杂性降低和细节保留之间达到了平衡,显著减少了计算需求而且在多项任务中表现出强大的竞争力

# Introduction

图像生成是计算机视觉领域的重大发展,但是 DM 模型需要大量的算力,有十亿参数的自回归 Transformer,相比之下,GAN 显示出了局限性因为他们的对抗训练不容易扩展到多模态的领域.相比于 GAN,生成模型有如下的好处:

1. 避免模式崩溃和训练的不稳定性: 基于似然,扩散模型不容易出现 GAN 那样的模式崩溃,训练更稳定
2. 参数高效: 通过高效的参数共享机制,扩散模型能够以较少的参数量建模复杂的自然图像分布,而不需要像自回归那样需要十亿的参数.

## Departure to Latent Space

我们的方法通过分析已有的在像素空间中的扩散模型出发.对于大多数的似然模型来说,学习可以被分为两个阶段:

1. 感知压缩阶段: 在这个阶段首先移除高频信息,只学习有限的语义变化
2. 语义压缩阶段: 进一步学习数据的语义和结构概念,构建真正的模型
因此,我们希望找到一个感知上等下,但是计算更高效的空间,来生产更高分辨率的图像

我们将训练分为两个阶段: 首先训练一个自动编码器来提供低纬度的特征表示并且在表示上等价于原始的数据.重要的是,我们不依赖于过度的空间压缩,而是在隐藏空间中训练了扩散模型,能够表现出更好的缩放特性

> [!PDF|yellow] [Page.2](Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf#page=2&selection=94,0,95,48&color=yellow)
>
> > A notable advantage of this approach is that we need to train the universal autoencoding stage only once
>
>我们只需要训练一次自动编码器并且在其他的阶段可以复用他

# Method

> [!PDF|yellow] [Page.3](Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf#page=3&selection=243,0,248,3&color=yellow)
>
> > We propose to circumvent this drawback by introducing an explicit separation of the compressive from the generative learning phase (see Fig. 2).
>
>为了降低计算开销,我们提出了将压缩和生成阶段分离的方法

该方法的好处如下:

1. **计算效率提升**: 通过在低维的空间中进行采样,扩散模型的计算效率大幅度提升
2. **利用扩散模型的归纳偏置**：扩散模型从其 UNet 架构中继承了针对空间结构数据的归纳偏置，这使得它们在处理具有空间结构的数据时特别高效，从而减少了先前方法中需要的激进压缩水平（这些压缩通常会降低质量）。
3. **通用压缩模型**：获得的潜在空间不仅可以用于训练多个生成模型，还可以应用于其他下游任务，例如单图像的 CLIP 引导合成。

## Perceptual Image Compression

我们的压缩算法基于先前的研究并且包括一个由感知损失函数和基于 patch 的对比损失目标历来训练的自动编码器,能够避免模糊 (L2/1 损失) 而且引入局部真实感

更精确的说,给定一个图片 $x \in R^{HxWx3}$,encoder 编码 x 到潜在空间中 $z=\xi(x)$,然后 decoderD 把图片从潜在空间中恢复,其中z∈hxwxc,且encoder把图片以比率f进行下采样

这个部分本质上是一个tradeoff,减少计算的开销,只保留其中最重要和基础的一块

## Latent Diffusion Models
**潜在空间的生成模型**:有了高效的压缩器,我们接下来应该有一个高效的低维潜在空间,相较于高维空间,这个空间更适合用来构建似然生成,因为他更高效,也可以更关注语义信息.
与先前的依赖于自回归不同,基于attention的Transformer模型不同,我们可以利用模型的图像偏置来帮我们.

在潜在表示空间上做diffusion操作其主要过程和标准的扩散模型没有太大的区别，所用到的扩散模型的具体实现为 [time-conditional](https://zhida.zhihu.com/search?content_id=217622020&content_type=Article&match_order=1&q=time-conditional&zhida_source=entity) UNet。但是有一个重要的地方是论文为diffusion操作引入了[条件机制](https://zhida.zhihu.com/search?content_id=217622020&content_type=Article&match_order=1&q=%E6%9D%A1%E4%BB%B6%E6%9C%BA%E5%88%B6&zhida_source=entity)（Conditioning Mechanisms），通过cross-attention的方式来实现多模态训练，使得条件图片生成任务也可以实现。