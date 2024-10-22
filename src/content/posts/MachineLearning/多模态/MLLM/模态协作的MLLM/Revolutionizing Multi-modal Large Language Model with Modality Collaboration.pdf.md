---
title: "[[Revolutionizing Multi-modal Large Language Model with Modality Collaboration.pdf|Revolutionizing Multi-modal Large Language Model with Modality Collaboration]]"
published: 2024-10-17
last modified: 2024-10-18 11:58
description: ""
image: ""
tags:
  - 机器学习
  - 论文阅读
category: 论文阅读
draft: true
---

# Abstract

> [!PDF|yellow] [Page.1](Revolutionizing%20Multi-modal%20Large%20Language%20Model%20with%20Modality%20Collaboration.pdf#page=1&selection=44,7,46,30&color=yellow)
> > mPLUG-Owl2 utilizes a modularized network design, with the language decoder acting as a universal interface for managing different modalities.
>
> 使用模块化网络设计,使用一个语言 encoder 来作为统一的接口管理不同的模态

## Introduction

> [!PDF|yellow] [Page.2](Revolutionizing%20Multi-modal%20Large%20Language%20Model%20with%20Modality%20Collaboration.pdf#page=2&selection=3,0,6,20&color=yellow)
> > Previous studies [27, 63] in multi-modal learning suggest that different modalities can effectively collaborate, thereby enhancing the performance of both text and multi-modal tasks simultaneously
>
> 先前关于 MLLM 的论文说明了不同的模态能够互相合作,来增强文本与多模态能力.

但是当前的多模态学习使用跨模态的对其方法,把视觉编码器的视觉特征映射到冻结的 LLM 中,通过利用保留的语言能力来进行多模态任务的开发.这种方式阻止了模态的潜在合作.虽然 fine-tune 显著提升了多模态的性能,但是减少了文本任务的表现,因此我们认为:

> [!PDF|red] [Page.2](Revolutionizing%20Multi-modal%20Large%20Language%20Model%20with%20Modality%20Collaboration.pdf#page=2&selection=19,32,24,36&color=red)
> > As illustrated in Figure 1, the challenge of modality collaboration in MLLMs is from applying a single module to balance the gain of modality collaboration and modality interference, where modalities may interfere with each other on a large number of instruction datasets across multiple modalities.
>
> MLLMs 中模态协作的挑战在于应用单一模块来平衡模态协作的收益与模态干扰，其中模态可能在多个模态的指令数据集上相互干扰。

我们的模型采用模块化网络设计，考虑了模态协作和模态干扰，使用语言解码器作为管理多模态信号的通用接口。特定来说,模型结合了某些共享模块,以促进模态协作开发,并且引入了一个模态自适应模块作为不同模态之间的枢纽.因此视觉和语言被投影到一个共享的语义空间中来进行跨模态的交互,而所提出的模块有助于保留模态特定特征。我们的新型架构通过模态自适应模块**屏蔽了信息密度不同的模态间干扰**，使其能够有效协同捕捉共享信息。此外，我们引入了一种创新的两阶段训练模式，包括**视觉 - 语言预训练**和**联合视觉 - 语言指令微调**。这一模式分两个阶段训练视觉编码器，使其能够更有效地捕捉低级和高级语义视觉信息。

## Related Work

**MLLM**: 有 3 中方法来构建多模态基础模型,
> [!PDF|yellow] [Page.2](Revolutionizing%20Multi-modal%20Large%20Language%20Model%20with%20Modality%20Collaboration.pdf#page=2&selection=95,17,98,18&color=yellow)
> > For instance, Flamingo [2] is a forerunner in this area, using a frozen vision encoder and a large language model equipped with gated cross-attention for crossmodality alignment
>
> 使用冻结的视觉编码器并且使用一个门注意力来进行跨模态对齐

> [!PDF|yellow] [Page.2](Revolutionizing%20Multi-modal%20Large%20Language%20Model%20with%20Modality%20Collaboration.pdf#page=2&selection=100,0,104,33&color=yellow)
> > In contrast, PaLM-E [16] integrates extracted visual features directly through linear layers into the pre-trained PaLM [12] model, which boasts 520 billion parameters, thereby leading to robust performance across numerous real-world applications.
> >
>相比之下这人直接把视觉特征通过线性层投影到模型中,但是这种方法是创建了冗长的视觉特征

> [!PDF|yellow] [Page.2](Revolutionizing%20Multi-modal%20Large%20Language%20Model%20with%20Modality%20Collaboration.pdf#page=2&selection=100,0,104,33&color=yellow)
> > In contrast, PaLM-E [16] integrates extracted visual features directly through linear layers into the pre-trained PaLM [12] model, which boasts 520 billion parameters, thereby leading to robust performance across numerous real-world applications.
>
> 后续的开发了 Q-former 来减少视觉特征的长度

> [!PDF|red] [Page.2](Revolutionizing%20Multi-modal%20Large%20Language%20Model%20with%20Modality%20Collaboration.pdf#page=2&selection=120,30,121,46&color=red)
> >  overlooking the unique granularities between vision and language modalitie
>
> 需要注意的是上述的方法直接把视觉特征和 LLM 进行对齐,认为视觉和语言特征是等价的,忽略了视觉和语言形态之间的差别.

**Instruction Tuning With MLLMS**: 指令微调通过优化预训练的大型语言模型，使其能够理解并遵循自然指令，从而提升其在零样本情况下对未见任务的泛化能力。

> 在大规模无监督数据上训练之后，再经过有监督微调和对齐之后就可以完成很多任务。尽管如此，面对垂直领域的应用，大模型依然需要微调才能获得更好地应用结果。而大模型的微调有很多方式，包括指令微调、有监督微调、提示工程等。其中，指令微调（Instruction Tuning）作为改进模型可控性最重要的一类方法

同时,一些模型使用图像注释的方式 (也就是物体边界框,图像描述和区域描述) 来提示 GPT 生成指令和回复.

## Methodology

![image.png](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/20241018095256.png)

我们的模型如上,主要包括一个 Visual Encoder,Visual Abstractor,Text embedding Layer,Language embedding/Decoder,其中 text 的部分与传统的大模型一致.此外,我们还引入了模态自适应层来使用不同的模态

### Architecture

> [!PDF|yellow] [Page.3](Revolutionizing%20Multi-modal%20Large%20Language%20Model%20with%20Modality%20Collaboration.pdf#page=3&selection=47,0,51,0&color=yellow)
> > Specifically, we utilize ViT-L/14 as the vision encoder and LLaMA-2-7B [58] as the language decoder.
>
> 我们使用 Vit-L/14 作为视觉编码器,LLaMA-2-7B 作为语言 decoder

视觉编码器把输入 HxW 的特征图变为 H/14xWx14 大小的 tokens,视觉特征然后与 text token 进行 combine 在输入到语言 decoder 中把视觉 - 语言任务转变为语言生成任务.

**但是随着图片大小的变大,encoder 序列会变得很长,此外,如果图像中存在冗余,会导致噪声的引入**.因此我们引入了 Visual Abstractor 模块,装配上固定的可学习 queries 来从图片中提取高维语义特征.特定来说,我们输入提取到的 visual token 序列 $\mathcal{I} = \left[ I_1, I_2, \cdots, I_P \right] \in \mathbb{R}^{P \times d}$ 和一个固定数量的可学习查询 $\mathcal{Q}\in \mathbb{R}^{Kxd}$ 来输入到 visual Abstractor 中.visual encoder 包换一系列的层,第 i 个层的输出这样计算:

$$
\begin{align}
    C^i &= \text{Attn}(V^i, [\mathcal{I}; V^i], [\mathcal{I}; V^i]), \tag{1} \\
    V^{i+1} &= \text{SwiGLU}(C^i W_1) W_2. \tag{2}
\end{align}
$$

w 是可学习的参数,我们设置 V0=Q 来初始化这个过程,此外,为了增加细粒度的能力,我们向 I 和 V 中加入了正弦位置嵌入来提示位置信息.这样就可以把时间复杂度 O((P+L)^2)>>O((K+L)^2)l 了.特别是当多张图片和 text 很短的情况.

> $C^i = \text{Attn}(V^i, [\mathcal{I}; V^i], [\mathcal{I}; V^i])$ 这里分别是 Qkv,通过拼接 I 和 V,可以从图像和语言的组合中提取关键信息

### 模态自适应模块

先前的方法尝试把视觉投影到语言 space 中

> [!PDF|yellow] [Page.3](Revolutionizing%20Multi-modal%20Large%20Language%20Model%20with%20Modality%20Collaboration.pdf#page=3&selection=337,1,342,18&color=yellow)
> > However, this strategy can cause a mismatch in granularity , where image features often contain fruitful semantic information compared to the discrete semantic information within text embedding features.
>
> 然而，这种策略可能导致粒度不匹配，与文本嵌入特征中的离散语义信息相比，图像特征通常包含丰富的语义信息。这些方法忽略了模态各自的特征

因此我们提出了自己的方法 (模态自适应块)MAM,

> [!PDF|red] [Page.3](Revolutionizing%20Multi-modal%20Large%20Language%20Model%20with%20Modality%20Collaboration.pdf#page=3&selection=346,8,349,24&color=red)
> > hich decouples vision-language representations by projecting visual features and language features into a shared semantic space while preserving the distinctive properties of each modality.
>
> 把视觉 - 语言表示投影到共享的语义空间中同时保留各自模态的特征,从而解耦视觉 - 语言表示

通常给定一个视觉 - 语言序列 $X \in \mathbb{R}^{L_{V}+L_{T} \times d}$ 和一个模态指示器 $M \in {0,1}^{L_V+L_T}$ ,我们首先定义模态特定操作

$$
\phi(X, M, m) = X \odot \mathbf{1}_{\{M = m\}},

$$

其中 m∈{0,1}表示模态,给定前一个模态的输入,我们首先把不同的模态归一化到同样的量级:

$$
H_{l-1}=LN_{V}(\Phi(H_{l-1},M,0))+LN_{T}(\Phi(H_{l-1},M,1))
$$

然后我们修改 attention 层如下:

$$
\begin{align}
H_l^Q = \tilde{H}_{l-1} W_l^Q, \\
H_l^K = \phi(\tilde{H}_{l-1}, M, 0) W_l^{K_0} + \phi(\tilde{H}_{l-1}, M, 1) W_l^{K_1}, \\
H_l^V = \phi(\tilde{H}_{l-1}, M, 0) W_l^{V_0} + \phi(\tilde{H}_{l-1}, M, 1) W_l^{V_1}, \\
C_l = \text{Softmax} \left( \frac{H_l^Q {H_l^K}^\top}{\sqrt{d}} \right) H_l^V,
\end{align}
$$

通过这种方式，我们可以在共享语义空间中计算这两种模态之间的相似性，同时通过不同的值投影层保留每种模态的独特特性。此外，通过解耦键和值的投影矩阵，我们可以避免两种模态之间的干扰，特别是与粒度不匹配相关的干扰。同样，我们也通过使用不同的层归一化层来建模这些特性。最后，为了促进同一特征空间内的模态协作，我们为两种模态保留了一个共享的前馈网络（FFN）.这种设计既保留了每个模态的独特性，又有效地实现了模态之间的协作

### 训练方式

我们采用 2 步训练法则,包括预训练和视觉指令微调,旨在对齐语言和视觉,然后再指令微调阶段进行微调,

> [!PDF|red] [Page.4](Revolutionizing%20Multi-modal%20Large%20Language%20Model%20with%20Modality%20Collaboration.pdf#page=4&selection=422,14,426,35&color=red)
> > However, we find that simply freezing a pretrained vision encoder and training a vision-language projector to align visual data with language models can limit their capacity to interpret complex visual information, such as scene text and visual knowledge.
>
> 但是我们发现简单的冻结 vision encoder 然后训练视觉 - 语言投影来让两者对齐限制复杂视觉的能力

为了解决这个问题,我们让 vison encoder 在整个训练阶段都可以被训练,这样的策略让模型更高效捕获低纬和高维的特征,

> [!PDF|yellow] [Page.4](Revolutionizing%20Multi-modal%20Large%20Language%20Model%20with%20Modality%20Collaboration.pdf#page=4&selection=434,45,439,1&color=yellow)
> >  Specifically, for the pre-training stage, we enable the vision encoder, visual abstractor, and a part of the modality-adaptive module to be trainable, while keeping the pre-trained language model frozen.
>
> pretrain 训练方式

> [!PDF|yellow] [Page.4](Revolutionizing%20Multi-modal%20Large%20Language%20Model%20with%20Modality%20Collaboration.pdf#page=4&selection=445,0,447,61&color=yellow)
> > Based on this, we adopt a joint training approach by tuning the whole model during the instruction tuning stage, incorporating both text and multi-modal instructions.
>
> instruction fine-turn 训练方式

## Experiments
6层的visual abstractor,7B的llama-2,adamw优化器,余弦学习率调度器预热步长1k,最大学习率1e-4.对于vision encoder,使用layer-wise learning rate decay(0.9)来保持低层次的视觉表示.在指令微调阶段，我们对整个模型进行了 **1轮** 训练，学习率设定为 **2e-5**，批次大小为 **256**。此外，图像分辨率从 **224×224** 增加到 **448×448**。在实验中，层次学习率衰减同样被应用，并且对保持良好的视觉表示至关重要。