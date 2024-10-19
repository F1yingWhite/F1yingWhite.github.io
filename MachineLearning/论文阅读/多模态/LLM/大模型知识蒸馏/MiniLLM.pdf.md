---
title: "[[MachineLearning/论文阅读/多模态/LLM/大模型知识蒸馏/MiniLLM.pdf|MiniLLM]]"
description: ""
image: ""
tags:
  - 机器学习
  - 论文阅读
  - LLM
category: 论文阅读
draft: false
published: 2024-10-19 10:42
last modified: 2024-10-19 11:34
---

# Abstract

> [!PDF|yellow] [Page.1](MachineLearning/论文阅读/多模态/LLM/大模型知识蒸馏/MiniLLM.pdf#page=1&selection=39,0,41,52&color=yellow)
>
> > However, previous KD methods are primarily applied to white-box classification models or training small models to imitate black-box model APIs like ChatGPT.
>
>先前的知识蒸馏方法主要采用白盒分类模型或者训练小模型来模仿黑盒模型 api，如 ChatGPT。

我们使用反向的 Kullback-Leibler 散度来替代正向的 KL 散度 (KLD),这对于生成语言模型的 KL 更为合适.然后我们推到了一种有效的优化方案

## Introduction

随着大模型的发展,一种有效的技术来减轻他们的计算量就是知识蒸馏,也就是我们从大的老师模型的监督下训练小模型.有两种训练方法,一种是黑盒蒸馏,也就是只能访问教师模型的输出文本,另一种是白盒蒸馏,能够看见中间的隐藏状态.

> [!PDF|yellow] [Page.2](MachineLearning/论文阅读/多模态/LLM/大模型知识蒸馏/MiniLLM.pdf#page=2&selection=52,0,66,34&color=yellow)
>
> > However, white-box KD approaches are mostly studied for small (< 1B parameters) language understanding models [SDCW19, WWD+20], while white-box KD for LLMs is yet to be explored.
>
>黑盒模型在通过大模型 API 进行微调上取得了很好的效果,而白盒模型随着开源大模型的增加而更加的有用,元就这弄够通过输出的分布以及中间状态获得更高的效果.但是白盒蒸馏的方法常用来训练小模型 (<1B),白盒 LLM 的方法尚未探索.

我们认为标准的 KD 目标对于 LLM 是次优的,给定教师模型的分布 p(y|x) 和学生模型的分布 $q_{\theta}(y|x)$,标准的 KD 目标本至少是最小化教师与学生分布之间的近似正向 KL 散度,叫 $KL(p||q_\theta)$,这迫使 q 覆盖 p 的所有模式,对与文本分类任务,KL 散度的效果较好,因为输出空间优限制的数量类别,而对于开放性的问题,**输出空间非常复杂,因此 p(y|x) 要比 q(y|x) 包含复杂得多的模式,最小化 KL 散度可能会导致 q 在 p 的空白区域分配不合理的高概率,生成在 p 下不可能产生的样本**.

为了减轻这个问题,我们提出了最小化反向 KLD,KL(q||p),常用在计算机视觉和强化学习中.相较于 KL(p||q),KL(q||p).最小化 KL\[qθ||p] 使 qθ能够寻找 p 的主要模式，并对 p 的空白区域赋予较低的概率.这意味着学生模型能够避免学习教师模型中太多的长尾变体,专注正确内容的生成,这在需要正确性的场景非常重要.

## Method

我们考虑条件文本生成，其中模型对从分布 px 中采样的提示 x 产生响应 y = {yt}T T =1 条件，这是 llm 执行任务的典型方式。传统的蒸馏优化了如下目标:

$$
KL[p||q_{\theta}]=\mathbb{E}_{x \sim p_{x},y \sim p'}\log \frac{p(y|x)}{q_{\theta}(y|x)}
$$

但是如果 q 的表达能力不足,就会过度强调 p 的空白区域

### MiniLLM: 使用反向 KLD 蒸馏知识

我们考虑最小化反向 KLD:

$$
\begin{align}
\theta = \arg\min_{\theta} \mathcal{L}(\theta) = \arg\min_{\theta} \text{KL}[q_{\theta} || p]\\
= \arg\min_{\theta} \left[ - \mathbb{E}_{x \sim p_x, y \sim q_{\theta}} \log \frac{p(y|x)}{q_{\theta}(y|x)} \right].
\end{align}
$$

最小化反向 KLD 以及被证明会导致 mode-seeking 行为 (在生成模型中),也就是 $q_{\theta}$ 能够分配到 p 的高概率模式而忽略小概率,我们首先探究了 LLM 中的这一性质,与正向最小化 KLD 的序列级别 KD 不同,MINILLM 最小化反向 KLD

> [!PDF|red] [Page.3](MachineLearning/论文阅读/多模态/LLM/大模型知识蒸馏/MiniLLM.pdf#page=3&selection=334,4,355,9&color=red)
>
> >  MINILLM that minimizes reverse KLD does not force qθ to fit all y sampled from the teacher distribution p. Instead, it encourages the student to generate samples preferred by the teacher within its own capacities, which is more possible to achieve.
>
>不强制 q 符合从教师模型中采样得到的所有 y,而是孤立学生模型在能力范围内找到教师偏好的样本.

### 策略梯度优化
