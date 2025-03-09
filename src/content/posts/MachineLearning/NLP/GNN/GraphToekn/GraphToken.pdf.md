---
title: "[[MachineLearning/NLP/GNN/GraphToekn/GraphToken.pdf|GraphToken]]"
description: ""
image: ""
published: 2025-03-05
tags:
  - GNN
  - 机器学习
category: 论文阅读
draft: false
---

> [!PDF|red] [Page.1](MachineLearning/NLP/GNN/GraphToekn/GraphToken.pdf#page=1&selection=32,0,34,11&color=red)
>
> > How can we best encode structured data into sequential form for use in large language models (LLMs)?
>
>要解决的问题

我们提出了一种参数高效的算法来为 llm 展示结构化数据。我们的方法能够针对结构化数据的编码，改善图推理任务。通常的 llm 只把顺序文本作为输入，但是最近的工作已经把输入扩展到了空间和时间模式。

目前的 LLM 存在幻觉和知识新鲜度的问题，llm 在有新的知识辅助的时候，它们能够调整参数信息，有效的考虑新的信息。通过 rag 能够很好的使用结构化数据来丰富 llm 的知识。

如何在 llm 中展示结构化数据是一件非常重要的事情。当前主要的手段是通过手工设计的基于文本的序列化。这种方法会导致模型解码的复杂度：模型必须首先理解知识的结构才能来使用这些信息。我们需要更高效的方法来表示结构化数据。

重新训练大模型可以提升能力但是太消耗资源，微调需要领域相关知识和人类专业知识。

> [!PDF|yellow] [Page.2](MachineLearning/NLP/GNN/GraphToekn/GraphToken.pdf#page=2&selection=20,48,22,31&color=yellow)
>
> > ur method, GraphToken, learns an encoding function that generates fine-tuned soft-token prompts.
>
>我们的方法提出了使用 GNN 来得到一种编码方法来得到 soft-token，并且只需要训练 GraphToken 就可以了。

# 方法

在考虑如何把结构化数据传给 llm 的时候有两种选择：

1. 编码为词性标记并进行 llm 嵌入
2. 通过网络把他编码为连续表示，跳过标记嵌入
但是结构化数据通常没有顺序
