---
title: "[[GeminiInMedicine.pdf|GeminiInMedicine]]"
description: ""
image: ""
tags:
  - 机器学习
  - 论文阅读
category: 论文阅读
draft: true
published: 2024-10-21
last modified: 2024-10-21
---

> [!PDF|red] [Page.4](GeminiInMedicine.pdf#page=4&selection=11,9,15,98&color=red)
>
> > owever, while the generalist approach is an meaningful research direction for medicine, real world considerations present trade-offs and requirements for task-specific optimizations which are at odds with each other. In this work, we do not attempt to build a generalist medical AI system. Rather, we introduce a family of models, each optimized for different capabilities and application-specific scenarios, considering factors such as training data, compute availability, and inference latency.
>
>本文的模型并没有尝试训练一个通用的 ai 模型,而是针对每一种场景构建了一个模型

> [!PDF|yellow] [Page.7](GeminiInMedicine.pdf#page=7&selection=125,43,129,24&color=yellow)
>
> >  To overcome this, we generate two novel datasets with self-training as described below: MedQA-R (Reasoning), which extends MedQA with synthetically generated reasoning explanations, or “Chain-of-Thoughts” (CoTs), and MedQA-RS (Reasoning and Search), which extends MedQA-R with instructions to use web search results as additional context to improve answer accuracy.
>
>我们构建了两个新的数据集用自学习方法,一个是使用思维连来扩展数据集,另一个是使用网络搜索来作为辅助扩大搜索准确率.
