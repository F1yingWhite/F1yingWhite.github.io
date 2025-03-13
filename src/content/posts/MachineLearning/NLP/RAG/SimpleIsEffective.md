---
title: SimpleIsEffective
description: ""
image: ""
published: 2025-03-12
tags:
  - 论文阅读
category: 论文阅读
draft: false
---

https://arxiv.org/pd/2410.20724

ICLR2025

# Abstract

现有的 rag 方法还困扰在大多在检索效果和效率之间找到最佳平衡。我们提出了 subgraphRAG，该方法使用了一个 MLP 和一个三元组打分算法来为高效的进行子图检索。检索到的子图的大小可以被灵活的调整来匹配查询的需求以及下游的 llm 的功能，我们的方法在没有 fineturn 的情况下达到了 sota

# Intruduction

相比 text-rag，graph-rag 提供了更高效的额外知识，减少了信息的冗余并且可以被灵活的更改。传统的 text-rag 无法提供很好的复杂推理支持。而 KG-RAG 则遇到了显著的性能问题。此外，检索到的信息结构必须涵盖回答问题的证据，单纯的扩大上下文窗口可能会导致噪声，应该修剪冗余的结构信息，从而提升 llm 的准确率。许多方法还依赖与多次调用 llm 来进行迭代搜索，导致了复杂度爆炸，此外，巨大的搜索空间和较小的上下文窗口为大模型推理带来了许多的限制。

*设计准则*：我们承认模型复杂度和推理能力之间有固有的平衡，为了高效的在 KG 上搜索，检索模块应该尽量的保持轻量，可变并且有一定的推理能力。我们的算法遵循了先检索后推理的范式（搜索一般有 LLM 搜索，GNN 和启发式），

# Method

我们提出的框架。给定一个查询 $q$,首先从 $\mathbb{G}$ 中提取出一个子图 $\mathbb{G}_g$ 然后 llm 根据子图给出答案。相比现有方法，我们的方法在三点上进行了改进：

1. 提取的子图应该包含问题 q 的所有的 evidence，同时保证大小约束，可以根据下游的 llm 的上下文窗口大小动态调整。
2. 提取过程的速度应该要快
3. 我们设计了提示词来进行推理

## 高效灵活的子图检索

我们定义子图检索问题并且缩小到一个可以被高效解决的问题，有一个 llm$\mathbb{P}(G_{q},q)$ 接受子图和查询，给定一个查询和对应的回答，最好的子图线索为：$G_{q}^*=argmax_{G_{q}\in G}\mathbb{P}(A_{q}|G_{q},q)$,但是解决这个问题时不可能的因为我们不知道 A，因此我们希望学习一个子图检索器能够生成未来的查询。

假设子图检索器是一个分布，给定问答对 D，子图检索能够被定义为如下任务

$$
\max_\theta\mathbb{E}_{(q,\mathcal{A}_q)\sim\mathcal{D},\mathcal{G}_q\sim\mathbb{Q}_\theta(\mathcal{G}_q|q,\mathcal{G})}\mathbb{P}(\mathcal{A}_q\mid\mathcal{G}_q,q).
$$

这个公式因为 llm 的复杂度因此很难被解出来，而且需要 llm 能够输出他的逻辑，更不用说计算梯度了。为了计算出这个公式，我们改变了策略。如果我们直到最优子图，最大似然估计能被用来训练检索器

$$
\max_{\theta}\mathbb{E}_{(q,\mathcal{A}_{q})\thicksim\mathcal{D}}\mathbb{Q}_{\theta}(\mathcal{G}_{q}^{*}\mid\mathcal{G},q)
$$

但是获取一个最优子图时困难的并且需要依赖 llm。相反，我们使用问答对来构建子图并且基于 MLE 来训练检索器。

$$
\max_{\theta}\mathbb{E}_{(q,\mathcal{A}_{q})\thicksim\mathcal{D}}\mathbb{Q}_{\theta}(\hat{\mathcal{G}_{q}}\mid\mathcal{G},q)
$$

这里的 $\hat{G}$ 可以是最短路径链接答案和实体。这个公式表明采样的子图不一定要遵循固定的路径方式，可以被拆除三元组的乘积的形式。

## 三元组分解

我们提出了一个检索器，允许一些潜在变量的情况下对三元组进行分解 $$

\mathbb{Q}_\theta(\mathcal{G}_q\mid\mathcal{G},q)=\prod_{\tau\in\mathcal{G}_q}p_\theta(\tau\mid z_\tau,q)\prod_{\tau\in\mathcal{G}\setminus\mathcal{G}_q}\left(1-p_\theta\left(\tau\mid z_\tau,q\right)\right)$$

>[!TIP]
>这个公式其实就是只获取子图中的三元组而不取其他的三元祖，每个三元组都计算一次概率

使用 MLE 可以得到新的公式，计算得到 z 之后，我们能够并行的从图中检索三元组，前 k 个三元组还能自由组合并且选择数目以适应不同的 llm。

以往的子图检索算法通常都专注启发式搜索，这限制了子图空间的灵活性，影响了 RAG 的效果。我们的 z(G,q) 建模了三元组在给定图下和 q 之间的关系，通常 GNN 是一个建模的选择，但是 gnn 在表现形式上具有很大的缺点，比如无法计算实体和主题实体之间的距离。

受到启发，我们使用DDE来作为z来建模r和q之间的关系。给定一个主题实体t，让s0作为热度编码便是e是否属于$T_q$
