---
title: "[[MachineLearning/多模态/多尺度/ContIG/Taleb_ContIG_Self-Supervised_Multimodal_Contrastive_Learning_for_Medical_Imaging_With_Genetics_CVPR_2022_paper.pdf|Taleb_ContIG_Self-Supervised_Multimodal_Contrastive_Learning_for_Medical_Imaging_With_Genetics_CVPR_2022_paper]]"
description: ""
image: ''
published: 2024-11-06
tags:
  - 机器学习
  - 论文阅读
category: 论文阅读
draft: True
---

# Abstract

本篇论文讲述了一个从大量的无标签医疗图片和基因型中学习的自监督学习算法。我们用对比学习算法把医学图片和基因型进行了对齐，我们设计了一个方法，将每个个体的多模态信息集成到同一个模型中，并且能够在不同个体的可用模态各异的情况下进行端到端的处理。

# Introduction

> [!PDF|yellow] [Page.2](MachineLearning/多模态/多尺度/ContIG/Taleb_ContIG_Self-Supervised_Multimodal_Contrastive_Learning_for_Medical_Imaging_With_Genetics_CVPR_2022_paper.pdf#page=2&selection=19,0,24,22&color=yellow)
>
> > Unlabelled medical images carry valuable information about organ structures, and an organism’s genome is the blueprint for biological functions in the individual’s body. Clearly, integrating these distinct yet complementary data modalities can help create a more holistic picture of physical and disease trait
>
>重要性

我们希望有如下算法:

1. 在不需要昂贵专家标注的情况下学习语义数据表示，
2. 高效地端到端地整合这些数据模态，
3. 解释发现的跨模态对应关系（关联）。

> [!PDF|yellow] [Page.2](MachineLearning/多模态/多尺度/ContIG/Taleb_ContIG_Self-Supervised_Multimodal_Contrastive_Learning_for_Medical_Imaging_With_Genetics_CVPR_2022_paper.pdf#page=2&selection=92,1,95,51&color=yellow)
>
> > In fact, we are not aware of any prior work that leverages self-supervised representation learning on combined imaging and genetic modalities
>
>影像 + 基因型的先驱

我们的贡献如下:

1. 提出了子监督学习方法能够从多模态的影像 + 基因型数据中进行学习。在一个端到端的模型中进行表征学习
2. 我们将梯度基解释性算法应用于更好地理解图像和遗传模态之间的跨模态对应关系（关联）

# Method

## 基因数据模态（介绍 dna）

DNA 的基本组成单位是核苷酸，它们编码着有机体发育所需的生物功能。由四种核苷酸——腺嘌呤（A）、胸腺嘧啶（T）、胞嘧啶（C）和鸟嘌呤（G）组成的长链序列构成了基因组——这就是构建有机体所需的“配方” .基因组中只有相对较小的部分编码蛋白质，而其余部分则具有调控或结构功能。然而，随着世代的更替，基因突变会发生，例如用一个核苷酸替代另一个核苷酸，比如将 A 替换为 C。这些基因变化中的一些可能会改变身体特征（例如眼睛颜色），或者引发疾病（例如阿尔茨海默症）。基因分型是测量这些基因变化的过程。最常测量的变化类型是单核苷酸多态性（SNPs），即在基因组的特定位置上，单对核苷酸发生了改变。

人类有三十亿个碱基对，但是通常只测量了一小部分，即使有很长的序列知道，处理原始数据依然不可行，因为只有一小部分携带有效信息，其他的只是增加了噪声。因此大多数研究只聚集在一个子集上，通常是几十到几百万个 SNP.而人类大部分的 SNP 都是相同的，只有一小部分发生了变化，因此我们在本研究中只考虑 3 个模态发生的变化。

复杂形状被多个因果关系影响，包括常见的基因型变体。比如身高，由全基因组的大多数决定。许多的疾病也是复杂性状，为了最好地编码与复杂性状相关的基因结构，我们使用**多基因风险评分（PGS）**，PGS 将许多大多数常见的 SNPs 汇聚成一个单一分数，反映个体对特定疾病的遗传易感性。这些单个 SNP 的权重根据它们与疾病的关联强度来确定。通过使用针对不同性状和疾病的多种 PGS，我们可以获得个体复杂性状易感性的多维视角。

最后，我们还采用了全基因组的均匀采样横截面，通过包括每个研究中基因分型的每 k 个 SNP。这些原始 SNP 大多数是常见变异（由于生物学采样过程），并且提供了个体遗传组成的广泛表示。这种表示可能包含群体结构信息，如祖先，同时也标记了高度多样的功能信息。

这三种基因模态——PGS、Burden Scores 和原始 SNP——捕捉了互补的方面，结合起来可以全面描述个体的遗传易感性。

## 图像和基因型的对比学习

我们假设一个有 N 个多模态样本的数据，每个包含一个基因和图像对，图像标记为 $x_{v}^i$,基因未 $x_{gm}^i$,其中 i∈{1..N},m∈{1..M}，我们根据 batchsize b>1 进行分组。其中基因型的类别数量因人而异

我们的算法对每个模态都有一个 encoder，得到的 d 维度向量通过投影层（2 个 MLP)

**两个模态间的对比损失**：我们对于第 i 个 pair，让第 i 个基因型是其他图像的负类，同理，让第 i 个图像时其他基因型的负类。因此损失是两个 parts 的和
1. 图像 - 基因型 L(v,g)
2. 基因型 - 图像 L(g,v)
**多个模态的相似度**：对于多组的基因模态，就是两两配对的和

## 基因特征可解释性

给定一组多模态元组，(x_v,x_g1,....,x_gm)，我们进行特征解释来理解每个基因特征的贡献，标准的深度学习可解释性方法在这个场景下并不直接适用，因为它们要求输入与输出之间存在简单的一对一关系，相反，我们利用一个固定的批次进行参考，这个批次包含了图像和基因模态，并且定义了解释器函数。。。

然后，我们可以使用诸如积分梯度（Integrated Gradients）或 DeepLift 等标准特征归因方法来解释 xx 中所有元素对整个批次损失的贡献。

缺失值的情况可以类似于第 3.2 节中的“外部”聚合方案来处理，即只需省略相应的模态。
