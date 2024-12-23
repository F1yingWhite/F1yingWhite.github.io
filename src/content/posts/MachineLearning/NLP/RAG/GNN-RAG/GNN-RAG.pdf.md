---
title: "[[MachineLearning/NLP/RAG/GNN-RAG/GNN-RAG.pdf|GNN-RAG]]"
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

知识图谱是人类手工设计的 3 元组,GNN 已经在 KGQA 领域被广泛使用,因为他们能处理复杂图谱中的信息.在本次工作中,我们提出了 GNN-RAG,吧 LLM 和 GNN 的归因能力结合在一起. 首先,GNN 从复杂的 GK 子图中见检索制定问题的候选答案.然后,提取 KG 中连接问题实体和答案候选的最短路径来表示 KG 推理路径,提取出的路径被语言化输入到 llm 里面.

RAG 是一种减轻 llm 幻觉的方法,通过增强上下文 (最新和精确的信息).比如,输入 RAG 的变为 "Knowledge:x->h, Question:xxx",

RAG的效果很大程度上取决于KG取回的facts.问题是kg存储了复杂的信息.而取回无用的信息会造成大模型的困惑.

# Introduction
KGQA大体上分为两类,语义SP解析和信息IR检索方法,SP方法把给定问题转化为逻辑形式的查询,然后从KG中查询得到答案,这种方法需要真实的逻辑查询来进行训练,标注耗时,且给出的语句可能无法执行.IR方法聚焦于弱监督的KGQA,只提供question-answer对.IR方法检索KG信息,然后输入一个KG子图.![image.png](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/20241217092235.png)
