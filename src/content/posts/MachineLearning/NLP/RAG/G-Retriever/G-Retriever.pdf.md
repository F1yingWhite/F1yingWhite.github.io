---
title: "[[MachineLearning/NLP/RAG/G-Retriever/G-Retriever.pdf|G-Retriever]]"
description: ""
image: ""
published: 2024-12-06
tags:
  - 机器学习
  - 论文阅读
category: 论文阅读
draft: true
---

# Abstract

给定一个有文本属性的图,我们让用户能和他们的图进行对他,也就是使用会话界面向图进行提问.针对用户的问题,我们的方法提供文本回复并且突出显示图中的相关内容.虽然现有的方法集成了 LLM 和 GNN,但是他们大多关注传统的图任务或者简单的图查询和小图和合成图.相比之下,我们开发了一个灵活的问答框架,针对现实世界的文本图,适用于多个应用场景,包括场景图理解,常识推理和知识图谱推理.为了实现这个目标,我们引入了图问答基准测试和第一个通用 RAG 方案.

> [!PDF|yellow] [Page.1](MachineLearning/NLP/RAG/G-Retriever/G-Retriever.pdf#page=1&selection=67,6,67,73&color=yellow)
>
> > can be fine-tuned to enhance graph understanding via soft prompting
>
> 该方法通过软提示微调,以增强图的理解能力.

# Introduction

## 先前的工作: 让你能和图聊天

现有的 LLM 和 GNN 结合的工作大多聚集于常规的图像任务比如 node,edge 和 graph 的分类,或者在小图和合成图上进行问答.对比之下,我们开发了一套灵活的问题回答框架,使用统一的问答接口来进行问答,表示图形交互的飞跃.

## 解决幻觉问题

> [!PDF|red] [Page.2](MachineLearning/NLP/RAG/G-Retriever/G-Retriever.pdf#page=2&selection=197,0,198,56&color=red)
>
> > LLMs are prone to hallucination, a phenomenon where the generated content is factually inaccurate or nonsensical
>
>大模型容易产生幻觉,这是一种在生成内容上不和逻辑或者胡乱编造的现象.

> [!PDF|yellow] [Page.2](MachineLearning/NLP/RAG/G-Retriever/G-Retriever.pdf#page=2&selection=202,16,212,2&color=yellow)
>
> > In particular, we employ a baseline method that adapts MiniGPT-4 to graphs, where a frozen LLM interacts with a trainable GNN that encodes graph data as a soft prompt, as in GraphToken .
>
>  使用 GNN 来提取 GraphToken

## 图大模型的效率和可扩展性

> [!PDF|yellow] [Page.2](MachineLearning/NLP/RAG/G-Retriever/G-Retriever.pdf#page=2&selection=223,1,226,55&color=yellow)
>
> > Recent research endeavors have explored translating graphs into natural language, such as by flattening nodes and edges into a text sequence, enabling their processing by LLMs for graph-based tasks
>
>最近的研究尝试把图翻译为自然语言,比如把 node 和 edge 展开为语言序列,但是这种方法有很大的扩展性问题.如果有几千个节点,那么展开之后就会有巨大的 token 开销,一种方法就是缩短 text sequence 让输入匹配 llm 的 token 限制,但是这会导致信息丢失

## G-Retriever

在这一章节中,我们提出了 G-Retriever,结合了 LLM,GNN 和 RAG 的优点,为了保持高效微调和 LLM 的预训练能力,我们冻结了 LLM 并用 SoftPrompt 的方法.我们的方法还把输出的大小缩放到 LLM 窗口大小之内.

G-Retriever 包括了 4 个部分,indexing,retrieval,subgraph construction 和 generation.

## Indexing

我们通过使用预训练的 LM 生成节点和图嵌入来启动 RAG 方法。然后将这些嵌入存储在最近邻数据结构中。

考虑 xn 作为节点 n 的文字表示,使用预训练的 LM,比如 Bert,zn=LM(xn)

## Retrieval

对于检索,我们使用同样的方式来对 query 做操作,zq=LM(xq),保证了一致性

为了找到最相关的 node 和 edge,我们使用 knn,

$$
v_{k} = argtopk_{n \in V}Cos(z_{q},z_{n})
$$

## Subgraph Construction

这一步是为了构建包含足够多的 node 和 edge,同时保持 graph 不会太大.这会筛选节点和边使得信息相关同时提升效率.这一步是通过求解 PCST 来得到的.

prize-collecting steiner tree 问题旨在找到一个连通子图能最大化节点的值同时最小化边的值 (值是上面的 cos()),具体来说，首先把 top k 的边和节点从高到低排序，然后其他的置为 0，也就是说 node 的值如下：

$$
\begin{aligned}
 & \mathrm{prize}(n)=
\begin{cases}
k-i, & \mathrm{if~}n\in V_k\mathrm{~and~}n\text{ is the top }i\mathrm{~node,} \\
0, & \text{otherwise.} & & 
\end{cases}
\end{aligned}
$$

边同理，我们的目标是得到一个子图，能够优化节点和边的总奖励，减去与子图大小相关的成本。

$$
S^*=\underset{S\text{ is connected}}{\operatorname*{\operatorname*{argmax}}}\sum_{n\in V_S}\mathrm{prize}(n)+\sum_{e\in E_S}\mathrm{prize}(e)-\mathrm{cost}(S),
$$

其中这里的 cost 计算为,这里的 ce 是事先指定的每条边的值

$$
cost(S) = |E_{S}|\times{C_{e}}
$$

最后我们才用了一种近线性复杂度的算法来解决这个问题。

## Answer Generation

S\* 表示我们构建的子图,我们使用 graph Encoder 来处理这个图,比如使用 GAT,

$$
h_{G} = POOL(Gnn{N}(S))
$$

这里的 pool 是平均池化，然后过一个 MLP，这里的 MLP 主要用来对齐图 token 和 llmtoken，然后把 S\* 给展开，和图 token 一起输入。（算法能够同时利用图结构数据的抽象表示和自然语言文本的具体信息）
