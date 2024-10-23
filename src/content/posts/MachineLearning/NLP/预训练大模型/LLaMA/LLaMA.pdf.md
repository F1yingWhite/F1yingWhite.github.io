---
title: "[[LLaMA.pdf|LLaMA]]"
description: ""
image: ""
tags:
  - 机器学习
  - 论文阅读
category: 论文阅读
draft: true
published: 2024-10-23
---

# Introduction

> [!PDF|yellow] [Page.1](LLaMA.pdf#page=1&selection=71,0,72,48&color=yellow)
>
> > These efforts are based on the assumption that more parameters will lead to better performance.
>
>更多的参数带来更多的性能,但是在计算资源有限的情况下,最佳性能并不能有大模型实现,而是在更多数据上训练的小模型实现的

> [!PDF|red] [Page.1](LLaMA.pdf#page=1&selection=77,0,83,14&color=red)
>
> > The objective of the scaling laws from Hoffmann et al. (2022) is to determine how to best scale the dataset and model sizes for a particular training compute budget
>
>因此需要确定如何在特定计算资源的情况下均衡数据集和模型大小

但是这个目标忽略了推理开销,因此在这个情况下,我们需要模型的推理速度最快而不是训练速度最快,训练时间长的小模型在推理上所花费的开销要更小.

> [!PDF|red] [Page.1](LLaMA.pdf#page=1&selection=113,0,116,42&color=red)
>
> > The focus of this work is to train a series of language models that achieve the best possible performance at various inference budgets, by training on more tokens than what is typically used
>
>本研究的重点是在使用更多的 token 进行训练,在不同的推理预算下训练一系列的语言模型,

本模型的范围从 7B~65B,叫做 LLaMA.目前 13B 的 LLaMA 已经超过的 GPT3,并且达到了 10 倍推理速度.在更高的模型大小上,我们的 65B 的模型已经能和 540B 的模型抗衡.

我们只使用了公开数据进行训练,在文章的后部分,我们阐述我们对 Transformer 做出的一系列修改和我们的训练方法.

# Approach

## Pre-train Data

> [!PDF|yellow] [Page.2](LLaMA.pdf#page=2&selection=81,0,132,5&color=yellow)
>
> > Dataset Sampling prop. Epochs Disk size CommonCrawl 67.0% 1.10 3.3 TB C4 15.0% 1.06 783 GB Github 4.5% 0.64 328 GB Wikipedia 4.5% 2.45 83 GB Books 4.5% 2.23 85 GB ArXiv 2.5% 1.06 92 GB StackExchange 2.0% 1.03 78 GB
>
>训练数据如图所示

## Tokenizer

我们使用 BPE 算法对数据进行 tokenize.值得注意的是,我们把所有的数字都拆分为个位数,对于未知的 UTF-8 字符把他们退回到字节级别进行分解。

> UTF-8 编码的格式如下: 使用 1~6 个字节进行编码,是一种前缀码

总的来说,我们的训练数据包含大约 1.4T 的 token,对于绝大多数数据,我们只用一遍来进行训练,除了 Wikipedia 和 Books 训练了两轮

## Architecture

我们的模型基于原始的 Transformer,我们的对于 Transformer 的模型做出了一定的改进,下面是我们的主要的一些区别

1. pre-normalization\[GPT3]: 为了提高训练的稳定性,我们对 Transformer 的每个子层输入都进行了 normalize,而不是对输出进行 normalize
2. SwinGLU 激活函数\[PaLM]: 我们替换使用 SwiGLU 来替换 RELU 的非线性层
3. Rotary Embeddings\[GPTNeo]: 我们删除了绝对位置嵌入,而是使用旋转位置嵌入来替代.

## Optimizer

我们使用 AdamW 优化器,余弦学习率调度器,最后的学习率是开始的 10%,使用 0.1 的权重衰减和 1 的梯度裁剪

## 加速方法

我们使用了一些优化方法来加速训练的速度,首先,我们使用高效的**casual 多头注意力机制**来减少内存和时间开销.
