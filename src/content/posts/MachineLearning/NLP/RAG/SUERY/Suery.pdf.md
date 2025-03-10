---
title: "[[MachineLearning/NLP/RAG/SUERY/Suery.pdf|Suery]]"
description: ""
image: ""
published: 2024-12-17
tags:
  - 机器学习
  - 论文阅读
  - RAG
  - 导论
category: 论文阅读
draft: false
---

# Introduction

RAG 用来为 LLM 添加领域相关知识，减轻幻觉问题。

传统 RAG 缺点:

1. 忽略联系关系
2. 冗余信息
3. 缺少全局信息
GraphRAG 就是为了解决这些缺点的,利用实现构建的图谱,提供了文本信息的抽象和总结,减少了文本长度和冗余.通过查询子图,我们能够有全局的理解.
GraphRAG 的顺序如下:
1. graph-based Indexing
2. graph-guided Retrieval
3. graph-enhanced generation

# Comparison with Related Techniques

## RAG

RAG 把外部知识和 LLM 进行结合来提升大模型的效果,聚合领域相关知识.

总体上看,GraphRAG 是 RAG 的一个分支,从相关知识图谱而不是文本中获取数据.但相比 Text Rag,GraphRAG 考虑了联系的关系.在构建知识图谱的过程中,信息已经被提炼和精简,提升了信息的精度.

## LLMs On Graphs

虽然大模型是处理纯文本和非欧几里得复杂数据的,图和数学也被引入到大模型中.目前的研究主要把 llm 和 gnn 结合来提升对图像数据的下游任务的能力,比如 node 分类,边预测.

与这些方法不同,GraphRAG 聚焦于使用查询从外部的知识图谱中使用查询得到信息.也就是一个是对图建模,一个是提升大模型的问答能力.

## KBQA

KBQA 旨在基于外部的知识提供用户的查询,提高实时性.先前的方法主要分为两类,基于 IR 的方法和 SP 方法,IR 方法从 KG 中检索信息来提升 LLM 的效果.而 SP 方法为每条 query 生成一个逻辑表示然后在 KG 中进行查询.

GraphRAG 和 KBQA 很接近,基于 IR 的 KBQA 就是 GraphRAG 的一个子集.

# Preliminaries

## Text-Attributed Graphs

缩写为 TAGs,nodes 和 edges 都是文本,能被缩写为 $\mathcal{G}=(V, \mathcal{E}, \mathcal{A}, \{ \mathbf{x}_v \}_{v \in V}, \{ \mathbf{e}_{i,j} \}_{i,j \in \mathcal{E}})$,其中 $\mathcal{V}$ 是 node 集合,$\mathcal{E}\subseteq \mathcal{V}\times \mathcal{V}$ 是边集合,A 是邻接矩阵,x 和 e 是节点和边的文字属性。

## GNN

Graph Neural Networks 是一种为了建模图数据的深度学习框架,正式的来说,每个 node 的表示 $h_{i}^{(l-1)}$ 在地 l 层的更新是同过聚合邻居和 edges 的信息:

$$
h_i^{(l)} = UPD(h_i^{(l - 1)}, AGG_{j \in N(i)} MSG(h_i^{(l - 1)}, h_j^{(l - 1)}, e_{i,j}^{(l - 1)}))
$$

Ni 是邻居节点,MSG 表示信息函数,根据邻居,边和自身计算信息.AGG 是聚合函数,把所有的信息加起来,是一种置换不变函数,比如 sum,max,min.UPD 是更新函数,来更新属性.

然后一个 readout 函数比如 mean,max pooling 来获得全局信息表示.

在 GraphRAG 中,GNN 能用来在检索阶段获得图的全局表示

## LM

language model 主要用于文本理解,主要分为两类,一种是判别模型,主要预测条件概率 p(y|x),主要用与分类和语义理解中,而生成模型比如 GPT 主要用来预测联合概率比如翻译和文本生成.

早期 RAG 主要关注预训练.最近随着 llama 和 gpt,RAG 的重点转变为提升信息检索,来提升效率和减轻幻觉

# Overview of GraphRAG

![](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/20241217153248.png)

GraphRAG 是一种利用外部知识图谱来提升情景理解能力的和回答更精确的方法.GraphRAG 的目标是从数据库中检索更相关的信息,来提升下游任务的效果.这个过程可以被概括为:$a=argmaxp(a|q,G)$.其中 a 是给定问题 q 和 TAG G 的最佳答案.这个过程可以被写为一个连续的过程: 一个图检索器 $p_{\theta}(G|q,\mathcal{G})$ 和一个答案生成器 $p_{\phi}(a|q,G)$.其中 G 是子图.这个过程可以被写为

$$
\begin{align}
p(a|q, G) = \sum_{G \subseteq \mathcal{G}} p_{\phi}(a|q,G) p_{\theta}(G|q,\mathcal{G}) \\
 \approx p_{\phi}(a|q, G^*) p_{\theta}(G^*|q, G),
\end{align}
$$

G\* 是最优子图.我们认为 GraphRAG 分为 3 个主要步骤,如上图

## Graph-based Indexing(G-Indexing)

G-Indexing 步骤主要用于 RAG 的初期,旨在识别或构建一个与下游任务相匹配的图数据库 G，并在其上建立索引。也就是数据库构建过程.索引过程通常包括映射节点和边的属性、在相连节点之间建立指针以及对数据进行组织，以支持快速遍历和检索操作。索引决定了后续检索阶段的粒度，在提高查询效率方面起着至关重要的作用。

## Graph-Guided Retrieval(G-Retrieval)

在 G-indexing 后,G-Retrieval 主要关注根据用户输入来从数据库中提取相关信息.这个阶段可以被写为:

$$
\begin{align}
G^*=G-Retriever(q,\mathcal{G}) \\
=argmax p_{\theta}(G|q,\mathcal{G}) \\
=argmax sim(q,G)
\end{align}
$$

其中 sim 是用来测量 q 和 g 的语义相似度的.

## Graph-Enhanced Generation

...

# Graph-Baed Indexing

这一节我们讨论 Indexing 方法的选择

## Indexing

### Graph Indexing

最常用的方法 用来提高信息的检索速度,也就是给定一个节点,他的邻居和 edge 很容易被访问到.,传统的检索方法有 BFS 和 Shortest Path algorithms.

### Text Indexing

text Indexing 是把图数据转变为文本描述来优化查询过程,这些描述存在 text corpus 中,能用各种技术加速.这里设计一部分的 graph 转 text 技术.

### Vector Indexing

向量索引将图数据转换为向量表示形式，以提高检索效率，便于快速检索以及有效地进行查询处理。

# Graph-Guided Retrieval

检索有两个主要问题:

1. 爆炸性的候选子图: 随着图变大,候选子图也变大，需要启发式搜索算法来有效地探索和检索相关的子图。
2. 不充分的相似度测量: 需要算法理解文本和图像信息

## LM Based Retriever

...

## GNN Based Retriever

GNN 通常把图数据编码然后根据和查询的相似度给分,比如 GNN-RAG,给每个实体打分,然后选择超过 threshold 的实体

# 检索范式

在 GraphRAG 中,有多种检索范式,比如一次检索,交互检索和多次检索.一次检索旨在一次操作就获取全部的信息,交互检索基于前一次的结果来进行后面的检索,逐渐找到最相关的信息,

## 单次检索

旨在通过一次查询就查到所有的信息,一类单词查询方法使用嵌入相似度来检索最相关信息,另一类通过预定义的规则来提取特定的三元组,路径或者子图,比如 G-Retriever 就是用 PCST 算法来检索最相关信息

## 交互查询

非自适应的交互查询通常人工设计迭代次数,每一次在上一次的基础上继续查询,而自适应查询让模型自己判断是否需要停止.

# 检索粒度

可以分为 nodes,三元组,path 和子图,每种检索粒度有自己的优缺点.

## Nodes

节点能够对单个元素进行精确检索,对有针对性的查询很有效,

## 三元组

三元组的这种结构化格式便于进行清晰且有条理的数据检索，在需要理解实体之间的关系以及上下文关联性至关重要的场景中颇具优势

## Path

路径粒度数据的检索可被视为对实体间关系序列的捕捉，能增强对上下文的理解以及推理能力。GraphRAG 中，由于路径能够捕捉图内复杂的关系以及上下文依赖关系，所以检索路径有着显著优势。然而，随着图规模的增大，可能的路径会呈指数级增长，这使得计算复杂度不断攀升，从而导致路径检索颇具挑战性。

## SubGraph

检索子图有着显著优势，因为它能够捕捉图内全面的关系上下文。这种粒度使得图检索增强生成（GraphRAG）能够提取并分析嵌入在更大结构中的复杂模式、序列以及依赖关系，有助于获得更深入的见解以及对语义关联更细致入微的理解。

## 检索增强

为了保证检索的质量,研究者提出了增强用户查询和图谱检索.在本文中,我们
