---
title: "[[MachineLearning/多模态/MLLM/LoRA/LoRA.pdf|LoRA]]"
description: ""
image: ""
tags:
  - 机器学习
  - 论文阅读
category: 论文阅读
draft: true
published: 2024-10-22
last modified: 2024-10-22
---

# Abstract

LLM 需要再大规模数据集上训练然后再下游任务上微调,如果全量 fine-tune 的话太贵了,我们提出低秩适应（LoRA），该方法冻结预训练模型的权重，并在 Transformer 架构的每一层中注入可训练的秩分解矩阵，从而大幅减少下游任务的可训练参数数量。

## Instruction

许多在自然语言上的研究依赖于在一个大数据集上做预训练,然后再许多下游任务上做应用.这样的适应叫做 fine-tune,微调的坏处是和原来的模型的大小一模一样.

许多人希望修改模型的部分参数或者使用外部模块来进行学习,但是目前的技术不行.

> [!PDF|yellow] [Page.2](MachineLearning/多模态/MLLM/LoRA/LoRA.pdf#page=2&selection=4,0,6,0&color=yellow)
>
> > We take inspiration from Li et al. (2018a); Aghajanyan et al. (2020) which show that the learned over-parametrized models in fact reside on a low intrinsic dimension.
>
>我们学习到的过参数化模型实际上位于一个低内在维度上。

> 过参数化是指模型的参数数量超过了训练数据的数量或复杂度。这种情况使得模型能够更灵活地拟合训练数据，甚至捕捉到数据中的噪声，从而提高了拟合能力，但也可能导致过拟合。

> 低内在维度是指数据在其特征空间中的有效维度较低，即尽管数据可能在高维空间中表示，但其本质上可以通过较少的维度来描述。

我们假设模型适应过程中的权重变化也具有低“内在秩”.

## 问题陈述

对于全量微调,预训练的权重 $\Phi_{0}$ 需要被更新为 $\Phi_{0}+\Delta\Phi$ 通过重复的最大化以下语言目标:

$$
	\max_{\Phi}\sum_{(x,y)\in Z}\sum_{t=1}^{|y|}\log(P_{\Phi}(y_{t}|x,y_{<t}))
$$

对于全量微调的缺点是对于每一个下游任务,我们都需要学习一组参数,因此如果预训练模型很大,管理和部署这些模型是很有挑战的.

在本文中,我们提出了一种更加参数高效的方法,参数的增长 $\Delta \Phi = \Delta \Phi(\Theta)$ 被一个更小的参数 $\Theta$ 所编码.因此优化任务就变成了优化θ

$$
	\max_{\Theta}\sum_{(x,y)\in Z}\sum_{t=1}^{|y|}\log(P_{\Phi+\Delta \Phi(\Theta)}(y_{t}|x,y_{<t}))
$$

## 我们的方法

### 低秩参数化更新矩阵

当训练一个密集参数的矩阵的时候这个矩阵是满秩的,当适应到特定任务之后,预训练模型有一个低秩的矩阵,尽管随机投射到较小的子空间，但仍然可以有效地学习.受此启发,我们认为权重在更新的过程中也有内在的排名.对于一个预训练的权重矩阵,我们通过用低秩分解表示后者来约束它的更新

$$
W_{0}+\Delta W=W_{0}+BA
$$

其中 B∈R^dxr,A∈R^rxk,其中 r<<min(d,k),在训练过程中,w0 是冻结的,AB 包含可学习参数.修改后的模型如下

$$
h=W_{0}x+\Delta W_{x}=W_{0}x+BAx
$$
![image.png](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/20241022171123.png)
我们使用

