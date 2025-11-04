---
title: "[[DyT.pdf|DyT]]"
description: normalization到底是干嘛的
image: ""
published: 2025-04-01
tags:
  - 论文阅读
category: 论文阅读
draft: false
---

# Abstract

在过去 normalization 模块总是被认为是重要的，其中 layer norm 是最流行的一种，但是通过观察发现 trasnfromer 中的 normalization 总是充当着 tanh 的作用，我们发现不需要 normalization 也能达到很好的效果。

>[!tip]
>
> $$
> \tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
> $$

归一化层的广泛应用，主要源于其在优化过程中表现出的实证优势，它们不仅能提升模型性能，更能显著加速训练收敛并增强稳定性。虽然人们替换了卷积或者注意力，但是仍然保留了归一化层。

我们的研究始于发现：LN 层实际上相当于类似 tanh 的方式映射输入到输出并且在缩放输入激活值的同时压缩极端数值。因此我们提出了 DyT,$DyT(x)=tanh(\alpha x)$,其中 $\alpha$ 是个可学习参数。值得注意的是，与归一化层不同，DyT 无需计算激活统计量即可同时实现这两种效果。

我们直接用 DyT 替代了 LN 层并且不修改任何超参，发现效果甚至更好，还提升了计算和训练的速度。

# Background

给定一个输入 x：（B，T，C），其中 T 是 token 数目，C 是每个 token 的维度，通常正则化层是：

$$
\text{normalization}(\boldsymbol{x})=\boldsymbol{\gamma}*\left(\frac{\boldsymbol{x}-\boldsymbol{\mu}}{\sqrt{\boldsymbol{\sigma}^2+\epsilon}}\right)+\boldsymbol{\beta}
$$

其中 $\gamma和\beta$ 都是可学习的 vector，$\mu和\epsilon$ 是输入的均值和方差，

batch normalization 是第一个现代的归一化层，他计算了 batch 和 token 维度的方差和均值。$\mu_{k}=\frac{1}{BT}\sum_{i,j}x$ $\sigma^2_{k}=\frac{1}{BT}\sum(x-\mu)^2$.

layer norm 和 RMSNorm 是 transformer 中使用最主要的俩归一化层。其中 layer norm 使用 $\mu_{k}=\frac{1}{C}\sum_{k}x$ $\sigma^2_{k}=\frac{1}{C}\sum(x-\mu)^2$.而 RMSNorm 直接让平均值为 0，现在大多数的 LM 用的都是 RMSNorm

## 归一化层做了什么

我们使用了在 ImageNet 上训练的 ViT-B,Difusion Transformer 和在 LibiSpeech 上训练的 wav2vecTransformer

我们对 lazynrom 的 input 和 output 做了可视化（因为输入输出的维度不变，我们可以一一对应）。

对于这三个模型，在早期的 LN 中，输入输出几乎是线性的，而到了后面和 tanh 很像。人们可能预期层归一化（LN）会对输入张量进行线性变换，因为减去均值并除以标准差均属于线性操作。LN 以逐 token 的方式执行归一化，仅对每个 token 的激活值进行线性变换。由于不同 token 具有不同的均值和标准差，这种线性变换并不会在输入张量的所有激活值上形成整体线性关系。然而，实际呈现的非线性变换竟与缩放后的 tanh 函数高度相似，这一发现仍令我们感到意外。

我们观察到在中央部分仍然以线性为主，但是还是有很多超出范围的值。归一化层对这些数值的核心作用在于将其压缩至相对温和的区间，使其分布与主体数据更为一致。这正是归一化层无法被简单仿射变换层替代的关键所在。我们推测，这种对极端值的非线性非等比压缩效应，正是归一化层不可或缺的重要原因。

**基于 token 和 channel 的归一化**：LN 层是如何在对每个 token 归一化的同时又以非线性的方式压缩极值？我们根据 token 和 channel 进行了可视化，我们可以观察到：任一 token 的所有数据点确实都严格排列成一条直线。然而，由于每个 token 具有不同的方差，这些直线的斜率也各不相同。输入范围较小的 token 通常方差较小，归一化层会使用较小的标准差对其激活值进行缩放，因此产生的直线斜率更大。这些不同斜率的直线整体叠加后，最终形成了类似 tanh 函数的 S 形曲线。
不同通道的输入值在 x 轴上呈现显著的范围差异，各自构成整体 tanh 曲线的不同区段。特定通道（如红、绿、粉色所示）表现出更极端的输入值分布，这些值经过 LN 层后被显著压缩。

# DyT

被 tanh 和 NL 的相似激发，我们提出了 DyT 作为替代，给定一个 tensor x，DyT 被定义为：

$$
DyT(x)=\gamma*\tanh(\alpha x)+\beta
$$

其中，α是一个可学习的标量参数，它能根据输入值的范围动态调整缩放比例，从而适应不同的输入尺度变化——这也正是我们将该操作命名为 " 动态 Tanh"（Dynamic Tanh）的原因。γ和β则是可学习的通道级向量参数（与各类归一化层中的参数作用相同），它们使输出值能够重新缩放至任意范围。尽管这些参数有时被视为独立的仿射变换层，但为了保持统一性（如同归一化层通常包含此类参数的做法），我们将其归为 DyT 层的组成部分。具体实现请参见算法 1。

```python
#input x has the shape of [B, T, C] 
# B: batch size, T: tokens, C: dimension
class DyT(Module): 
	def __init__(self, C, init_α): 
		super().__init__() 
		self.α = Parameter(ones(1) * init_α)
		self.γ = Parameter(ones(C)) 
		self.β = Parameter(zeros(C)) 
	def forward(self, x): 
		x = tanh(self.alpha * x) 
		return self.γ * x + self.β
```

token 上线性变换，channel 上 tanh

## **核心观察**（论文第 3 节）

- **现象**：作者发现，在 Transformer 的深层 Layer Norm（LN）中，输入 - 输出的映射呈现**类 tanh 的 S 形曲线**（图 2、图 3）。
    - LN 的原始操作是线性的（对每个 token 的激活值减去均值、除以标准差），但**不同 token 的统计量（均值/方差）不同**，导致整体输入 - 输出的散点图呈现非线性。
    - 极端值（远离 0 的输入）会被 LN 压缩到一定范围内（类似 tanh 的饱和区），而中心区域（接近 0 的输入）近似线性变换。

## 2. **DyT 公式的推导**

- **动机**：直接用 tanh 函数模拟 LN 的非线性行为，但需解决两个问题：
    1. **动态缩放**：不同 channel 的输入范围不同（图 4 右），需自适应调整 tanh 的输入尺度。
    2. **仿射变换**：LN 的输出通常接可学习的缩放（γ）和偏置（β），需保留这一设计。
- **公式设计**：
    DyT(x)=γ∗tanh⁡(αx)+βDyT(x)=γ∗tanh(αx)+β
    - **αα**：**可学习的标量参数**，控制 tanh 的输入范围（动态调整非线性区域的斜率）。
    - **γ,βγ,β**：与 LN 相同的**逐 channel 仿射参数**，恢复输出的尺度和偏移。
