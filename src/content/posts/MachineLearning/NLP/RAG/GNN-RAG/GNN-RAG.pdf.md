---
title: "[[GNN-RAG.pdf|GNN-RAG]]"
description: ""
image: ""
published: 2024-12-17
tags:
  - 机器学习
  - 论文阅读
  - RAG
category: 论文阅读
draft: false
---

# Abstract

知识图谱是人类手工设计的 3 元组,GNN 已经在 KGQA 领域被广泛使用,因为他们能处理复杂图谱中的信息.在本次工作中,我们提出了 GNN-RAG,吧 LLM 和 GNN 的归因能力结合在一起.

1. 首先 GNN 在一个密集子图上进行推理寻找答案
2. 连接提取到的问题实体和候选实体的答案的最短路径，该路径被转换为自然语言
3. 输入到 llm 中进行推理。

在我们的论文中 gnn 被用来充当从密集子图中提取信息的角色，我们还开发了检索增强技术

# Introduction

知识图谱通过多跳练习，能够很好的用于知识密集型任务比如 QA。而 RAG 的效果很大程度上取决于检索到的信息。知识图谱存储了复杂的图信息，从中检索信息需要高效的图处理算法，而取出不相关的信息会导致大模型困惑。现

> [!PDF|red] [Page.2](MachineLearning/NLP/RAG/GNN-RAG/GNN-RAG.pdf#page=2&selection=100,0,103,74&color=red)
>
> > Existing retrieval methods that rely on LLMs to retrieve relevant KG information (LLM-based retrieval) underperform on multi-hop KGQA as they cannot handle complex graph information [Baek et al., 2023, Luo et al., 2024] or they need the internal knowledge of very large LMs, e.g., GPT-4, to compensate for missing information during KG retrieval [Sun et al., 2024].
>
>有的方法借助大模型来检索相关信息在多跳任务中效果不佳，因为模型无法理解复杂的图信息，或者需要超大规模的 lm 来弥补其中缺少的信息。

虽然 GNN 不像 LM 那样能够理解自然语言，但是有很强的检索能力。

KGQA 大体上分为两类,语义解析 SP 和信息检索 IR 方法,SP 方法把给定问题转化为逻辑形式的查询,然后从 KG 中查询得到答案,这种方法需要真实的逻辑查询来进行训练,标注耗时,且给出的语句可能无法执行.IR 方法聚焦于弱监督的 KGQA,只提供 question-answer 对.IR 方法检索 KG 信息,然后输入一个 KG 子图.

现有的方法主要分为两类，分别是使用潜在的 图信息来增强 lm，但是由于模态不匹配导致效果较差另一种是使用语言话的图描述，这种方法在插入的数据量较大的时候会引入噪声。

![image.png](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/20241217092235.png)

# 背景

## KGQA

给定一个问题，KGQA 需要从中提取能够回答问题的实体

## 检索和推理

KG 包含了很多节点和边，因此要为一个问题取一个合适的小的子图。在理想情况下所有正确的答案都被包含在这个子图中作为模型的输入，然后由 llm 推理得到正确的答案。

## GNN

KGQA 可以被认为是一个节点分类问题，每个实体都可以被分类为 answer 或者 non-answer，而 GNN 可以被用来进行节点分类。

# GNN-RAG
