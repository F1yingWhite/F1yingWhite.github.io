---
title: "[[MiniLLM.pdf|MiniLLM]]"
description: ""
image: ""
tags:
  - 机器学习
  - 论文阅读
  - LLM
category: 论文阅读
draft: true
published: 2024-10-19
last modified: 2024-10-21 12:30
---

# Abstract

> [!PDF|yellow] [Page.1](MiniLLM.pdf#page=1&selection=39,0,41,52&color=yellow)
>
> > However, previous KD methods are primarily applied to white-box classification models or training small models to imitate black-box model APIs like ChatGPT.
>
>先前的知识蒸馏方法主要采用白盒分类模型或者训练小模型来模仿黑盒模型 api，如 ChatGPT。

我们使用反向的 Kullback-Leibler 散度来替代正向的 KL 散度 (KLD),这对于生成语言模型的 KL 更为合适.然后我们推到了一种有效的优化方案

## Introduction

随着大模型的发展,一种有效的技术来减轻他们的计算量就是知识蒸馏,也就是我们从大的老师模型的监督下训练小模型.有两种训练方法,一种是黑盒蒸馏,也就是只能访问教师模型的输出文本,另一种是白盒蒸馏,能够看见中间的隐藏状态.

> [!PDF|yellow] [Page.2](MiniLLM.pdf#page=2&selection=52,0,66,34&color=yellow)
>
> > However, white-box KD approaches are mostly studied for small (< 1B parameters) language understanding models [SDCW19, WWD+20], while white-box KD for LLMs is yet to be explored.
>
>黑盒模型在通过大模型 API 进行微调上取得了很好的效果,而白盒模型随着开源大模型的增加而更加的有用,元就这弄够通过输出的分布以及中间状态获得更高的效果.但是白盒蒸馏的方法常用来训练小模型 (<1B),白盒 LLM 的方法尚未探索.

> [!PDF|red] [Page.2](MiniLLM.pdf#page=2&selection=129,2,201,35&color=red)
>
> > For text classification tasks, KL[p||qθ ] works well because the output space usually consists of a finite number of classes such that both p(y|x) and qθ (y|x) have few modes. However, for open-ended text generation tasks, which is usually the case of LLM applications, the output spaces are much more complex and p(y|x) can contain many more modes than what qθ (y|x) can express due to the limited model capacity. Minimizing forward KLD causes qθ to assign unreasonably high probabilities to the void regions of p [MG19] and produces very unlikely samples under p during free-run generation [Hus15].
>
> 我们认为标准的 KD 目标对于 LLM 是次优的,给定教师模型的分布 p(y|x) 和学生模型的分布 $q_{\theta}(y|x)$,标准的 KD 目标本至少是最小化教师与学生分布之间的近似正向 KL 散度,叫 $KL(p||q_\theta)$,这迫使 q 覆盖 p 的所有模式,对与文本分类任务,KL 散度的效果较好,因为输出空间优限制的数量类别,而对于开放性的问题,**输出空间非常复杂,因此 p(y|x) 要比 q(y|x) 包含复杂得多的模式,最小化 KL 散度可能会导致 q 在 p 的空白区域分配不合理的高概率,生成在 p 下不可能产生的样本**.

为了减轻这个问题,我们提出了最小化反向 KLD,KL(q||p),常用在计算机视觉和强化学习中.相较于 KL(p||q),KL(q||p).**最小化 KL\[qθ||p] 使 qθ能够寻找 p 的主要模式，并对 p 的空白区域赋予较低的概率**.这意味着学生模型能够避免学习教师模型中太多的长尾变体,专注正确内容的生成,这在需要正确性的场景非常重要.

> [!PDF|red] [Page.2](MiniLLM.pdf#page=2&selection=268,0,299,14&color=red)
>
> > Compared to KL[p||qθ ], minimizing KL[qθ ||p] causes qθ to seek the major modes of p, and assign low probabilities to p’s void region

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

> 这里的概率就是模型输出真实回答的概率

最小化反向 KLD 以及被证明会导致 mode-seeking 行为 (在生成模型中),也就是 $q_{\theta}$ 能够分配到 p 的高概率模式而忽略小概率,我们首先探究了 LLM 中的这一性质,与正向最小化 KLD 的序列级别 KD 不同,MINILLM 最小化反向 KLD

> [!PDF|red] [Page.3](MiniLLM.pdf#page=3&selection=334,4,355,9&color=red)
>
> >  MINILLM that minimizes reverse KLD does not force qθ to fit all y sampled from the teacher distribution p. Instead, it encourages the student to generate samples preferred by the teacher within its own capacities, which is more possible to achieve.
>
>不强制 q 符合从教师模型中采样得到的所有 y,而是孤立学生模型在能力范围内找到教师偏好的样本.

### 策略梯度优化

**梯度推导**: 我们发现上面的目标损失函数能够使用策略梯度优化解决

$$
\nabla \mathcal{L}(\theta) = - \mathbb{E}_{x \sim p_{\text{data}}, y \sim q_{\theta}(\cdot | x)} \sum_{t=1}^{T} (R_t - 1) \nabla \log q_{\theta}(y_t | y_{<t}, x),
$$

其中 T=|y|,$R_t=\sum_{t'=t}^{T}\log \frac{p(y'|y_{<t'},x)}{q_{\theta}(y'|y<{t'},x)}$ 是 $r_{t'}=\log \frac{p(y'|y_{<t'},x)}{q_{\theta}(y'|y<{t'},x)}$ 的累计用来衡量每一步的生成质量.直观上，生成的文本应该通过增大 $p(yt∣y<t,x)$ 来在教师分布中保持高概率，但同时通过降低 $qθ(yt∣y<t,x)$ 来保持多样性。公式 (2) 中的期望值通过蒙特卡罗采样计算.

>$p(y_t|y<t,x)$ 指的是在给定上下文和先前生成的序列的情况下,给定当前单词的概率

**单步分解**:

> [!PDF|yellow] [Page.4](MiniLLM.pdf#page=4&selection=4,5,13,0&color=yellow)
>
> > as found that the single-step generation quality rt is critical to the training variance because the error in the front tokens accumulates along the whole sentence.
>
>有人发现单步的生成质量对训练方差至关重要,因为前面的误差会累计到整个句子中

为了更关注 rt,我们重写了损失梯度来把 rt 从 Rt 中分离,并且直接计算 $\mathbb{E}_{y_{t}\sim q_{\theta}(t)}[r_{t}]$,

$$
\begin{align*} \nabla \mathcal{L}(\theta) &= \underbrace{\mathbb{E}_{\substack{\mathbf{x}\sim p_{\text{data}}(\cdot)\ \mathbf{y}\sim q_{\theta}(\cdot|\mathbf{x})}}\left[-\sum_{t=1}^{T}\nabla_{\theta}\mathbb{E}_{y_t\sim q_{\theta}(t)}[r_t]\right]}_{\text{(Single-step gradient)}} + \underbrace{\mathbb{E}_{\substack{\mathbf{x}\sim p_{\text{data}}(\cdot)\ \mathbf{y}\sim q_{\theta}(\cdot|\mathbf{x})}}\left[-\sum_{t=1}^{T}R_{t+1}\nabla\log q_{\theta}(y_t|y_{<t},x)\right]}_{\text{(Long-term gradient)}} \ &= (\nabla \mathcal{L})_{\text{Single}} + (\nabla \mathcal{L})_{\text{Long}}, \end{align*}
$$

其中 $q_{\theta}t=q_{\theta}(\cdot|y_{<t},x)$,注意单步损失能够直接通过词汇表求和,而不是使用蒙特卡洛采样并且对 $\theta$ 可导.这个算法更加关注单步生成质量,减少了方差加速了收敛.

**Teacher-Mixed Sampling**: 我们观察到**reward hacking**当训练第一个公式的时候,有时候会产生教师给高分的退化句子 y(比如重复短语).特别是在小模型训练的时候.为了创建一个更好的采样分布,我们把老师和学生的分布融合在一起

> reward hacking: 也就是通过意想不到的策略来获得高分,比如这里就通过不断地重复短语来得到高分.

$$
\bar{p}(y_{t|y_{<t},x})=\alpha p(y_{t})+(1-\alpha)q_{\theta}(y_{t})
$$

这样能够减轻 reward hacking 的强度,

> [!PDF|red] [Page.17](MiniLLM.pdf#page=17&selection=377,1,418,1&color=red)
>
> >  qθ (y|x) log p(y|x) qθ (y|x) ∇ log qθ (y|x)dydx
>
>这里用到了概率分布的对数技巧,也就是 $\nabla q_\theta(y|x) = q_\theta(y|x) \nabla \log q_\theta(y|x)$
>因为 logx 的倒数为 $\nabla x\cdot \frac{1}{x}$

> [!PDF|red] [Page.17](MiniLLM.pdf#page=17&selection=440,1,456,0&color=red)
>
> > log p(y|x) qθ (y|x)
>
>这里,因为时间序列步相当于 $logp(y∣x)=log(p(y1​∣x)⋅p(y2​∣y1​,x)…p(yT​∣y<T​,x))$,而 log 又是相加,所以可以写为下面的形式
