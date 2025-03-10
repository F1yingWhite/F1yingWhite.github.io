---
title: "[[MachineLearning/生成模型/DiT/DiT.pdf|DiT]]"
description: ""
image: ""
published: 2024-11-26
tags:
  - 机器学习
  - 论文阅读
category: 论文阅读
draft: true
---
http://www.myhz0606.com/article/dit
# Abstract

我们训练了 LDM 通过把 UNet 换成了 Transformer 架构,叫做 Diffusion Transformers(DiT),

# Introduction

> [!PDF|yellow] [Page.2](MachineLearning/生成模型/DiT/DiT.pdf#page=2&selection=32,53,35,46&color=yellow)
>
> > . In contrast to the standard U-Net [49], additional spatial selfattention blocks, which are essential components in transformers, are interspersed at lower resolutions
>
>相比于传统的 UNet,DM 中使用的 UNet 添加了注意力机制,

在本文中,我们希望证明 UNet 的偏置并不重要,并且可以被 Transformer 所替代,本文提出的 DiT 很好的遵循了 ViT 的实践.此外,我们还研究了 trasnfromer 的复杂度 (以 Gflops 评估) 和模型样本质量 (FID 评估) 之间的关系.我们展示了在 LDM 框架下,成功使用了 Transformer 替换了 UNet 主干.

> [!PDF|yellow] [Page.3](MachineLearning/生成模型/DiT/DiT.pdf#page=3&selection=221,41,224,46&color=yellow)
>
> > In general, parameter counts can be poor proxies for the complexity of image models since they do not account for, e.g., image resolution which significantly impacts performance
>
>参数计算并不能很好的表示模型的复杂度,因为受到模型的生成图像像素的影响

# Diffusion Transformers

## Preliminaries

先来回顾一下 DDPM: 高斯噪声 DM 的前向过程逐渐加噪 $x_0 : q(x_t | x_0) = \mathbb{N}(x_t; \sqrt{\bar{a}_t} x_0, (1-\bar{a}_t)I)$,通过重参数化技巧,我们可以把他写为:$x_{t}=\sqrt{\bar{a}_{t} }x_{0}+\sqrt{ 1-\bar{a}_{t}\epsilon_{t} }$,其中 $\epsilon_{t}\sim N(0,I)$

DM 希望学习这个反向过程:$p_{\theta}(x_{t-1}|x_{t})=\mathbb{N}\left( \mu_{\theta}(x_{t}),\sum_{\theta}(x_{t}) \right)$,其中网络用来预测 $p_\theta$,逆过程模型通过对数似然的变分下界（variational lower bound）进行训练 $\mathcal{L}(\theta) = -p(x_0 | x_1) + \sum_t D_{KL}(q^*(x_{t-1} | x_t, x_0) \| p_\theta(x_{t-1} | x_t)),$,通过忽略无关参数,因为 q 和 p 都是高斯噪声,D_KL 可以通过均值和协方差来衡量.通过重参数化,loss 变为预测的噪声和实际噪声的 MSE Loss 来进行评估.

->实际中先使用简单的 MSE 训练预测网络,在通过完整的 Loss 来优化协方差来提升模型的多样性
**Classifier-free guidance**: 条件 dm 通过考虑额外的参数作为输入,比如输入的分类标签等.在这种情况下,反向过程变为 $p_{\theta}(x_{t-1}|xt,c)$ 其中噪声和协方差都和 c 有关.在这种情况下,classifier-free guidance 能够被用来鼓励采样过程发现 x,让,log(p(c|x)) 变大,
**Latent Diffusion Models**:1. 训练自动编码器来把像素空间变到离散的高维空间中,2.在高维空间训练 DDPM,

## Diffusion Transformer Design Space

我们使用原始的 Transformer,来保持其缩放特性.由于我们希望训练生成图像,因此 DiT 基于 ViT 架构![image.png](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/20241126102124.png)

**patchify**:DiT 的空间表示 z 大小为 32x32x4(进过 VAE),原始大小为 256x256x3,DiT 的第一层就是 Patchify,把输入变成 Token,通过线性变换,每个 token 有 d 维.我们还使用了位置编码

**DiT block**:tokens 被输入到一系列的 Transformer 中.除了噪声图片,DM 有时候还出来额外的条件信息,比如时间步 t,类别 c 和自然语言.我们探索了 4 中 Transformer 的变体来不同的处理条件.

- 条件控制: 我们简单的把 t 和 c 作为两个额外的 token 拼接到序列,这和 cls token 类似.最后再移除条件 token
- 交叉注意力机制: 我们拼接 t 和 c 的 embeddings 得到一个 len = 2 的序列,Transformer 被修改为再多头自注意力后加一个多头交叉注意力,带来 15% 的额外开销
...

**Transformer Decoder**: 在最后的 DiT 之后,我们需要解码我们的序列来得到一个预测的噪声对角协方差预测,这两个形状与原始输入一致,我们使用标准的线性解码器来进行解码,把每个 token 变为 pxpx2C 的 tensor,其中 C 是输入的通道维度.最后 rearrange tokens 变为原始的形状来得到预测噪声和协方差
