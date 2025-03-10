---
title: "[[MachineLearning/NLP/RAG/ReasoningOnGraphs/REASONING ON GRAPHS.pdf|REASONING ON GRAPHS]]"
description: ""
image: ""
published: 2025-03-06
tags:
  - 论文阅读
category: 论文阅读
draft: false
---

>[!tip]
>这篇论文在推理上进行了优化

# Abstract

现在的方法把 KG 当做一个真实知识基准并且过度重视他们的结构化信息的重要性。我们的方法首先生成推理路径充当可信任的规划，然后利用这些计划从 kg 中获取可信任的推理路径。

# Introduction

为了释放 llm 的推理潜力，有人提出了首先提供一个 plan 然后一步步执行推理的方案。通过这种方法，llm 将复杂问题拆解为一系列小问题并且一步步解决他们。但是 llm 还是会有幻觉和知识的缺乏。因此需要 rag.

我们的方法：rog 首先生成关系路径来充当可信任的 plan 能够充当解释和检索

1) 规划优化，我们将 KG 中的知识提炼到 LLM 中，生成忠实的关系路径作为规划；
2) 检索推理优化，我们使 LLM 能够基于检索路径进行忠实推理，并生成可解释的结果

# Related Work

有人用提示词来引导 llm 进行多部推理，但是幻觉和知识的匮乏导致了推理的准确性。KGQA 主要考虑从 KG 中检索相关知识提高推理性能。

# 我们的方法

## 目标

简而言之，我们的目标就是优化给定知识图谱，问题的时候最大输出正确答案的概率

$$\begin{aligned}&P_\theta(a|q,\mathcal{G})=\sum_{z\in\mathcal{Z}}P_\theta(a|q,z,\mathcal{G})P_\theta(z|q)\end{aligned}$$

其中 $\mathcal{Z}$ 是所有的路径集合

## 优化框架

虽然我们的计划生成很有吸引力，但是 llm 对 kg 没有一点了解。因此 llm 不能直接生成路径。此外，llm 可能不能正确的理解路径。为了处理这个问题，我们设计了两个指令微调任务

1. 将 KG 的知识提炼到 llm 中来生成可靠的 plan
2. 检索推理优化：让 llm 基于 path 来推理。
