---
title: KGRAG导论
description: ""
image: ""
published: 2025-03-11
tags:
  - 论文阅读
category: 论文阅读
draft: false
---

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10387715

主要介绍了 3 中 llm：

1. KG-enhanced LLM：在 LLMs 的预训练和推理阶段融入知识图谱（KGs），或用于增强对 LLMs 所学知识的理解；
2. LLM-augmented KG，利用 LLMs 完成不同的 KG 任务，如嵌入、补全、构建、图到文本生成和问答；
3. 协同的 llm+kg:LLMs 和 KGs 在其中扮演同等重要的角色，以互利的方式共同增强，实现由数据和知识驱动的双向推理。

# Introdcution

llm 缺少新鲜知识与可解释性，虽然有 CoT 这种增强解释的方法，但是还是受到幻觉问题的影响，这在高风险环境中非常致命。KGs 就是来解决这个问题的，KG 是一种结构化的知识表现形式，他们有符号推理能力。然而，kg 很难被构造，并且很难动态的随着真实世界被更改，并且经常忽略了冗余的信息，

## KG-enhanced LLM

这个部分可以被分为 3 类：

1. 在预训练阶段来帮助大模型进行训练
2. 在推理阶段帮助大模型获取最新的知识而不用预训练
3. 使用 llm 来增强大模型的解释性。
## 在预训练阶段增强大模型
现有的大模型训练通常依赖于自监督学习在大规模的语料下。
...
## 在llm的输入中集成KG
给定一个只是图谱和对应的语句，