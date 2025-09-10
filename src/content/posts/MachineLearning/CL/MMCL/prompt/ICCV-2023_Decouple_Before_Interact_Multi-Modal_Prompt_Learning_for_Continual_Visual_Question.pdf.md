---
title: Decouple_Before_Interact_Multi-Modal_Prompt_Learning_for_Continual_Visual_Question
description: ""
image: ""
published: 2025-08-21 09:40:03
tags:
  - 论文阅读
  - 持续学习
category: 论文阅读
draft: false
updated: 2025-08-21 11:17:05
---

# Abstract

我们提出看了 TRIPLET 方法，建立在预训练的 VL 模型上，包含解耦提示和提示交互策略来捕捉模态之间的复杂交互。解耦提示包含一组可学习的参数从不同方面解耦，提示词交互策略负责建模提示与输入的交互

直接使用现有的单模态持续学习方法，没有考多模态特性，也没有考虑复杂的模态交互关系。

![image.png](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/20250821095224090.png)

# Methods

对于基于 transformer 的 VQA 模型通常有 3 个 encoder，visual encoder/language encoder 和 fusion encoder。给定问题 q 和图像 v，公式可以表示为：

$$
\hat{y}(v, q) = \mathcal{F}\left(\mathrm{FT}\left([\mathrm{VT}(v); \mathrm{TT}(q)]\right)[0]\right)
$$

其中，$\mathrm{VT}$ 和 $\mathrm{TT}$ 分别是预训练的视觉变换器编码器和文本变换器编码器，用于对图像 $v$ 和问题 $q$ 进行编码。$\mathrm{FT}(\cdots)[0]$ 将多模态特征进行融合，并将第一个融合后的特征输入到分类器 $\mathcal{F}(\cdot)$ 中以预测答案 $a$。

我们的目标是设计一些提示词和交互策略来解决 CL-VQA 问题。所以我们的公式被改为了：

$$
\hat{y}(v, q) = \mathcal{F}\left(\mathrm{FT}\left([P^{(f)}; \mathrm{VT}([P^{(v)}; v]); \mathrm{TT}([P^{(q)}, q])]\right)[0]\right),
\quad (2)
$$

其中，$P^{(v)}$、$P^{(q)}$ 和 $P^{(f)}$ 分别表示视觉提示、问题提示和融合提示。

**选择性深度解耦**
我们以逐层的形式对提示进行解耦，并将其附加到选定的层上。与将提示附加到所有选定的多头注意力（MHA）层 不同，本文采用一种替换式策略，仅在部分 MHA 层中添加提示，从而更加节省内存。给定一个包含 $K$ 层的变换器 $T$，即 $T([P; x]) = (\mathrm{L}_K \circ \mathrm{L}_{K-1} \cdots \circ \mathrm{L}_0)([P; x])$，可以按层进行分解：

$$
\bar{h}_k^P = \alpha_k \cdot h_k^P + (1 - \alpha_k) \cdot P_k,
\quad (3)
$$

$$
[h_{k+1}^\mathrm{CLS}; h_{k+1}^P; h_{k+1}^x] = \mathrm{L}_k([h_k^\mathrm{CLS}; \bar{h}_k^P; h_k^x]),
$$

其中，$[h_0^\mathrm{CLS}; \bar{h}_0^P; h_0^x] = [\mathrm{CLS}, P_0, x]$ 是原始输入，而 $\mathrm{L}_K$ 的输出被视为模型的最终输出。此外，$\alpha_k \in \{0, 1\}$ 是一个预定义的开关，用于控制是使用输出提示特征 $h_k^P$ 还是第 $k$ 层特定的提示 $P_k$ 作为输入

**互补解耦**
遵循互补设计原则 ，每个提示被进一步划分为两部分：一个通用提示（G-Prompt），用于提取任务不变的知识；以及一个专家提示（E-Prompt），用于提取特定任务的知识。例如，视觉提示 $P^{(v)} = \{G^{(v)}; \{E^{(v)}\}\}$ 由所有任务共享的 G-Prompt $G^{(v)}$ 和专为第 $t$ 个任务设计的 E-Prompt $E_t^{(v)}$ 组成。当第 $t$ 个任务到来时，我们训练提示 $P_t^{(m)} = \{G^{(m)}; E_t^{(m)}\}$，其中 $m = v, q, f$。

在我们的实现中，我们将上述三种解耦设计相结合。也就是说，我们为三种模态分别设置了三组提示，每组提示包含逐层的深度提示，而每个逐层的深度提示又包含一个 G-Prompt 和一组 E-Prompts。总结来说，所有可学习的提示包括：

$$
P^{(m)} = \left\{G_k^{(m)} \in \mathbb{R}^{L_G \times D}\right\} \bigcup \left\{E_{t,k}^{(m)} \in \mathbb{R}^{L_E \times D}\right\},
\quad (4)
$$

其中，下标 $t$ 表示任务，$k$ 表示第 $k$ 个多头注意力（MHA）层，$L_G / L_E$ 分别表示 G-Prompt / E-Prompt 的长度，$D$ 为嵌入维度。

## 提示词交互

通过所提出的解耦提示，我们需要交互策略来将它们全部训练在一起。我们首先采用查询与匹配策略（Query-and-Match Strategy）来匹配输入特征与相关的任务特定提示。我们进一步引入模态交互策略（Modality-Interaction Strategy）和任务交互策略（Task-Interaction Strategy）来促进提示间的交互。前者会鼓励不同模态提示间的相互传播，从而增强模型性能。后者会使提示较少受到序列任务的影响，从而减少灾难性遗忘。

**查询与匹配策略**
由于我们解耦的提示中包含任务特定的提示，因此需要准确的任务特定键来将输入特征与这些提示关联起来。我们将文献中的“查询与匹配”（Query-and-Match）策略扩展到多模态领域，通过一个查询匹配损失 $\mathcal{L}_{qm}$ 来训练对应的任务特定键 $\boldsymbol{u}_t^{(m)}$，使得 $\boldsymbol{u}_t^{(m)}$ 更接近来自任务 $t$ 的样本，而远离其他任务的样本。首先，给定 $(\boldsymbol{v}, \boldsymbol{q})$，查询通过冻结的变换器（见公式 (1)）获得：

$$
\boldsymbol{h}^{(v)} = \mathrm{VT}(\boldsymbol{v}), \quad \boldsymbol{h}^{(q)} = \mathrm{TT}(\boldsymbol{q}), \quad \boldsymbol{h}^{(f)} = \mathrm{FT}([\boldsymbol{h}^{(v)}, \boldsymbol{h}^{(q)}]),
$$

$$
\mathsf{q}^{(v)} = \boldsymbol{h}^{(v)}[0], \quad \mathsf{q}^{(q)} = \boldsymbol{h}^{(q)}[0], \quad \mathsf{q}^{(f)} = \boldsymbol{h}^{(f)}[0],
$$

其中 $\boldsymbol{h}[0]$ 表示从向量中选择第一个元素，即选择 $\boldsymbol{h}^{\mathrm{CLS}}$，如公式 (3) 所示。使用余弦相似度 $\gamma$，查询匹配损失 $\mathcal{L}_{qm}$ 定义为：

$$
\mathcal{L}_{qm}(D_t) = -\sum_{(\boldsymbol{v}, \boldsymbol{q}) \in D_t} \sum_{m \in \{v,q,f\}} \gamma\left(\boldsymbol{u}_t^{(m)}, \mathsf{q}^{(m)}\right).
\quad (5)
$$

这里也就是为每一个任务计算一个特征

**模态交互策略**
我们提出一种提示模态交互机制，作为不同模态提示之间的桥梁。我们引入如下交互映射：

$$
\hat{P}_{t,k}^{(f)} = \boldsymbol{W}_{t,k}^{(v)} \otimes P_{k,t}^{(v)} + \boldsymbol{W}_{t,k}^{(q)} \otimes P_{t,k}^{(q)} + \boldsymbol{W}_{t,k}^{(v,q)} \otimes \left(P_{t,k}^{(v)} \odot P_{t,k}^{(q)}\right),
\quad (6)
$$

其中 $\odot$ 表示逐元素乘法，$\otimes$ 表示矩阵乘法，$\boldsymbol{W}^{(\cdot)}$ 是可学习的交互矩阵。在本文中，我们对这些交互矩阵的秩进行约束，令 $\boldsymbol{W} = \boldsymbol{U} \otimes \boldsymbol{V}^\top$，其中 $\boldsymbol{U}, \boldsymbol{V} \in \mathbb{R}^{D \times d}$ 是两个低秩矩阵。我们使用以下 $\mathcal{L}_{mod}$ 来处理这种模态交互：

$$
\mathcal{L}_{mod}(D_t) = -\sum_k \gamma\left(\hat{P}_{t,k}^{(f)}, P_{t,k}^{(f)}\right).
\quad (7)
$$

这个损失函数鼓励生成的融合提示 $\hat{P}_{t,k}^{(f)}$ 与原始的融合提示 $P_{t,k}^{(f)}$ 保持一致性，确保交互是有意义的。

**任务交互策略**
由于我们的基于提示学习的方法建立在冻结的预训练模型之上，不同任务的表示共享相同的语义空间。因此，提示在不同任务之间共享不变的语义空间，以与预训练模型对齐，这导致了不同任务之间具有不变的提示模态交互结构。为此，我们引入任务交互约束 $\mathcal{L}_{task}$ 来调节这种不变结构，具体如下：

$$
\mathcal{L}_{task}(D_t) = \sum_{m,t,k} \left( \left\| \boldsymbol{W}_{t,k}^{(m)} - \langle \boldsymbol{W}_{t,k}^{(m)} \rangle_{t-1} \right\|_F^2 \right),
\quad (8)
$$

其中，$\|\cdot\|_F$ 表示 Frobenius 范数，而 $\langle \boldsymbol{W}_k^{(m)} \rangle_{t-1}$ 是在训练第 $(t-1)$ 个任务时缓存的 $\boldsymbol{W}_k^{(m)}$ 的副本。通过约束 W 的变化幅度，防止新任务的学习过度改变已学习的模态交互模式，从而**减少灾难性遗忘**。

### 4.2.3 训练与推理

**训练**
当一个新任务 $t$ 到来时，我们将 $\mathcal{F}$ 实例化为一个分类器 $g_t$（一个全连接层），并分配任务特定的查询键 $(\boldsymbol{u}_t^{(v)}, \boldsymbol{u}_t^{(q)}, \boldsymbol{u}_t^{(f)})$ 和提示 $(E_t^{(v)}, E_t^{(q)}, E_t^{(f)})$。然后，解耦提示、交互矩阵、分类器和查询键通过以下联合损失函数进行共同训练：

$$
\begin{align}
\mathcal{L}(D_t) = \sum_{(\boldsymbol{v},\boldsymbol{q},y)\in D_t} \ell_{\mathrm{CE}}(\hat{y}(\boldsymbol{v},\boldsymbol{q}), y) \\
+ \lambda_1 \mathcal{L}_{qm}(D_t) + \lambda_2 \mathcal{L}_{mod}(D_t) + \lambda_3 \mathcal{L}_{task}(D_t),
\end{align}
\quad (9)
$$

其中 $\hat{y}(\boldsymbol{v},\boldsymbol{q})$ 是网络的预测结果（见公式 (2)），$y$ 是目标答案，$\ell_{\mathrm{CE}}(\hat{y}, y)$ 是交叉熵损失，$\lambda_{(\cdot)}$ 是超参数。

**推理**
在推理阶段，给定一个输入样本 $(\boldsymbol{v}, \boldsymbol{q})$，我们选择最佳匹配的任务索引 $\arg\max_t \gamma\left(\boldsymbol{u}_t^{(m)}, \mathsf{Q}^{(m)}\right)$。然后选择对应的提示 $P_{t^{(m)}}^{(m)}$，并将其输入到相应的变换器中。最后，选择对应的分类器 $g_{t^{(\cdot)}}$ 来预测答案。
