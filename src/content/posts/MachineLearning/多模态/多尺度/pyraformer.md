---
title: Pyraformer
published: 2024-10-04
description: ''
image: ''
tags: [机器学习,多模态,多尺度,论文阅读]
category: '论文阅读'
draft: false
---
## Abstract
实践中我们需要建立一个灵活但简洁的模型，可以捕获广泛的时间依赖关系,在本篇文章中，我们提出了PAM，其中尺度间树结构总结了不同分辨率下的特征，尺度内相邻连接建模了不同范围内的时间依赖关系。

## Introduction
时间序列预测的主要挑战在于如何构建一个powerful但是简单的模型，如果把short-term和long-term都纳入考虑是精确预测的关键，值得注意的是长期的注意比短期的要更为困难一些.并且学习long-term，模型的输入也很长，因此需要注意时间复杂度和空间复杂度。

在这篇文章中，我们提出了一种新颖的金字塔注意力模式来建立长短期注意力之间的联系并且有着低时空复杂度。特定来说，我们如图fig1d基于attention传递信息，图中的联系被分为inter-scale和intra-scale联系。尺度间的联系构建了原始序列的多分辨率表示。尺度间的联系通过把自己和邻居联系在一起，来捕获想同分辨率下的时间依赖关系。

1. 我们提出了一种能够捕获不同分辨率的网络
2. 时间复杂度低
3. 准确率高

![image-20241004120304419](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20241004120304419.png)

## Related work

### 稀疏transformer

在NLP领域，已经有很多的方法来降低Transformer的复杂度

### Hierarchical Transformer

## Method

时间序列预测任务可以被看为预测未来的M步：$z_{t+1:t+M}$基于前面的L步$z_{t-L+1：t}$和相关的协变量,在本文中，我们提出了Pyraformer。

![image-20241004121948094](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20241004121948094.png)

我们的PAM模块如图1d所示，这种多分辨率结构已经被证明是计算机视觉中long-range交互建模的有效工具。主要分为两部分，inter-scale和intra-scale。

Inter-scale链接形成了一个C-ary-tree,每个父节点有C个孩子节点。比如我们把金字塔的最佳尺度和原始序列的每小时的观测值链接起来，在粗尺度上就可以看做是时间序列的每日/月、周的特征。因此金字塔提供了原始时间序列的多尺度特征。此外，通过简单的尺度内连接相邻的节点，更容易在粗粒度中捕获长期依赖关系。

与原始attention不同，在PAM中每个节点只和有限数量的节点进行attention，具体来说，$n_l^s$表示在scale s中第l个节点，其中s从1~S表示从第到顶的scale。每个节点能够与一些邻居节点交互，包括3类，分别是具有相同规模的A nodes（包括自己），C children和P parent
$$
N^{(s)}{l} &=& A^{(s)}{l}\cup C^{(s)}{l}\cup P^{(s)}{l}\\
A^{(s)}{l}&=&{n^{(s)}{j}:|j-l|\leq\frac{A-1}{2},1\leq j\leq\frac{L}{C^{s-1}}}\\
C^{(s)}{l}&=&{n^{(s-1)}{j}:(l-1)C<j\leq lC} if s\geq2 else \emptyset\\
P^{(s)}{l}&=&{n^{(s+1)}{j}:j=\left[\frac{l}{C}\right]} if s\leq S-1 else \emptyset
$$
然后注意力机制如下所示

$$y_i = \sum_{\ell\in N^{(s)}*{l}}\frac{\exp(q_i k^T*{\ell}/\sqrt{d_K})v_{\ell}} {\sum_{\ell\in N_l^{(s)}}\exp(q_i k^T_{\ell}/\sqrt{d_K})'}$$

### CSCM模块

coarser-scale construction module主要初始化节点以便后续的PAM模型。具体来说，就是在c个孩子上使用kernel c和步长C的卷积

![image-20241004125835027](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20241004125835027.png)