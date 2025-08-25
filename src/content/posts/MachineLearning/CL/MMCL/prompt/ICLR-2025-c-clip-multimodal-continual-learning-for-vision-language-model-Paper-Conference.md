---
title: ICLR-2025-c-clip-multimodal-continual-learning-for-vision-language-model-Paper-Conference
description: ""
image: ""
published: 2025-08-22 15:36:09
tags:
  - 论文阅读
  - 多模态
  - 持续学习
category: 论文阅读
draft: true
updated: 2025-08-23 13:44:47
---

# Abstract

1. 因为 CLIP 表现出很强的 zero-shot 能力，因此许多的工作都采用了 Prompt Design。将预训练的模型冻结只训练 prompt。相 1.比之下，多模态任务更具挑战性，因为特定领域的表现不佳往往需要同时对视觉编码器和文本编码器进行微调（？）。
2. 我们提出了新的 VLCL 的 benchmark 标准
3. 此外，基于正则化的 CL 忘得更少因为他们学的更少。在持续学习过程中逐渐失去了可塑性。因此，我们寻求利用多模态表示学习的特性来实现新旧知识的同时学习，克服过去的权衡。

VLCL 的目标是持续学习一个函数 $f \circ g: \mathcal{V} \times \mathcal{C} \rightarrow \mathcal{Y}$，该函数能够为所有已见任务中的每一对图像 - 文本描述（image-caption pair）分配正确的标签。具体而言，在阶段 $t$，挑战在于最小化新数据集 $\mathcal{D}^t$ 上的损失函数 $\ell$（例如对称交叉熵），同时保留来自先前任务的知识，并可能在早期学习的基础上进一步提升（Aljundi, 2019），如下所示：

$$
\begin{aligned}
& \min_{\theta,\varphi,\epsilon} \mathbb{E}_{(v,c)\sim\mathcal{D}^t}[\ell(f_\theta(v),g_\varphi(c))] + \sum \epsilon_j \\
& \text{s.t. } \mathbb{E}_{(v,c)\sim\mathcal{D}^j}[\ell(f_\theta(v),g_\varphi(c)) - \ell(f_{\theta^{t-1}}(v),g_{\varphi^{t-1}}(c))] \leq \epsilon_j; \forall j \in \{1,2,...,t-1\}.
\end{aligned}
\quad (1)
$$

其中，最后一项 $\epsilon = \{\epsilon_j\}$ 是一个松弛变量，可用于表示在第 $j$ 个旧任务的数据集 $\mathcal{D}^j$ 上的遗忘（$\epsilon_j > 0$）或向后知识迁移（$\epsilon_j \leq 0$）。

# Method：C-CLIP

持续学习的核心在于保持先前的任务的情况下提升新任务的性能。

1. 在不存储或重放数据的基础上，总体思路是利用当前数据来约束就任务中的输入输出关系。我们采用 LORA 微调方法
2. 第二个目标，目前的 CL 算法很大程度上限制了模型自己的可塑性，因此我们提出了对比知识巩固方法。

## 针对遗忘的 Lora 集成

由于模型在推理时无法访问任务标识，难以决定使用哪一组 LoRA 参数。更重要的是，存储所有先前任务的 LoRA 会导致日益严重的内存问题。因此，我们提出在每个持续学习阶段结束时，将当前的 LoRA 参数 $\{\theta_{\text{LoRA}}, \varphi_{\text{LoRA}}\}$ 整合到主干网络中，如下所示：

$$
\{\theta^t, \varphi^t\} = \{\theta^{t-1} + \alpha \cdot \theta_{\text{LoRA}}, \varphi^{t-1} + \alpha \cdot \varphi_{\text{LoRA}}\},
\quad (2)
$$

通过实验我们发现我们的 lora 集成能够有效的防止 zero-shot 能力退化，但在微调数据集上的性能确很差，因此我们提出了一种对比知识巩固来提升模型的可塑性和稳定性

## 对比学习防止遗忘

我们的 lora 方法简单的将新模型和就特征空间对其，天然的会在新的数据集上表现很差，并忽略了多模态特性。因此我们的想法是优化 CLIP，让他从旧模型中学到一个更好的特征空间，而不是仅仅和他对齐。具体而言，对于每个图文对，会被新旧模型同时映射到深度特征空间，

$$
\mathcal{L}_{\text{CLIP}}^t = -\frac{1}{2N} \sum_{i=1}^{N} \left( \log \frac{\exp\left(\mathbf{z}_{v,i}^{\mathrm{T}} \mathbf{z}_{c,i}^t / \tau\right)}{\sum_{j=1}^{N} \exp\left(\mathbf{z}_{v,i}^{\mathrm{T}} \mathbf{z}_{c,j}^t / \tau\right)} + \log \frac{\exp\left(\mathbf{z}_{c,i}^{\mathrm{T}} \mathbf{z}_{v,i}^t / \tau\right)}{\sum_{j=1}^{N} \exp\left(\mathbf{z}_{c,i}^{\mathrm{T}} \mathbf{z}_{v,j}^t / \tau\right)} \right)
$$

传统的 CLIP 损失如上，而我们的 CKC 方法希望计算当前模型通过一次映射后新空间与旧空间的相似度

$$
\mathcal{L}_{\text{CKC}}^t = -\frac{1}{2N} \sum_{i=1}^{2N} \left( \log \frac{\exp\left(\tilde{\mathbf{h}}_i^{\mathrm{T}} \tilde{\mathbf{z}}_i^{t-1} / \tau\right)}{\sum_{j=1}^{2N} \exp\left(\tilde{\mathbf{h}}_i^{\mathrm{T}} \tilde{\mathbf{z}}_j^{t-1} / \tau\right)} + \log \frac{\exp\left(\tilde{\mathbf{z}}_i^{t-1\,\mathrm{T}} \tilde{\mathbf{h}}_i^t / \tau\right)}{\sum_{j=1}^{2N} \exp\left(\tilde{\mathbf{z}}_i^{t-1\,\mathrm{T}} \tilde{\mathbf{h}}_j^t / \tau\right)} \right),
\quad (5)
$$

其中：

$$
\tilde{\mathbf{h}}_i^t = \frac{[h_\psi(f_{\theta^t}(\mathbf{v}_i)),\; h_\psi(g_{\varphi^t}(\mathbf{c}_i))]}{\|[h_\psi(f_{\theta^t}(\mathbf{v}_i)),\; h_\psi(g_{\varphi^t}(\mathbf{c}_i))]\|},\quad
\tilde{\mathbf{z}}_i^{t-1} = \frac{[f_{\theta^{t-1}}(\mathbf{v}_i),\; g_{\varphi^{t-1}}(\mathbf{c}_i)]}{\|[f_{\theta^{t-1}}(\mathbf{v}_i),\; g_{\varphi^{t-1}}(\mathbf{c}_i)]\|},
$$

- **避免简单对齐的局限**：直接强制新旧特征完全对齐会限制模型学习新任务的能力
- **保持连接但允许差异**：通过投影器让两个空间保持联系，但又允许新空间有更好的表达能力
