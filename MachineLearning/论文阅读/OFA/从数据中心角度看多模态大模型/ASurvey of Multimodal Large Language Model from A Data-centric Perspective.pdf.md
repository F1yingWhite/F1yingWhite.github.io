---
title: "[[ASurvey of Multimodal Large Language Model from A Data-centric Perspective.pdf|ASurvey of Multimodal Large Language Model from A Data-centric Perspective]]"
published: 2024-10-17
description: ""
image: ""
tags:
  - 机器学习
  - 论文阅读
category: 论文阅读
draft: false
---

> [!PDF|yellow] [Page.2](MachineLearning/论文阅读/OFA/从数据中心角度看多模态大模型/ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=2&selection=60,0,61,22&color=yellow)
> > Most existing MLLMs focus on modifying model architecture to explore the use of information from multiple modality
> 
> 许多的任务目前考虑修改模型的架构来高效的利用信息

> [!PDF|yellow] [Page.2](MachineLearning/论文阅读/OFA/从数据中心角度看多模态大模型/ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=2&selection=83,40,84,47&color=yellow)
> >  data also significantly impacts the success of MLLMs
> 
> 对大模型来说数据也很重要,包括数据质量和数量

> [!PDF|red] [Page.2](MachineLearning/论文阅读/OFA/从数据中心角度看多模态大模型/ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=2&selection=104,3,105,56&color=red)
> > Our discussion answers three key questions from a data-centric perspective at different stages of MLLMs:
> 1. 怎么样选择大模型的数据,需要选择高质量,异构的数据,模型的不同阶段也有不同的数据需求
> 2. 数据如何影响大模型的
> 3. 怎么评价数据对多模态大模型的影响

> [!PDF|yellow] [Page.4](MachineLearning/论文阅读/OFA/从数据中心角度看多模态大模型/ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=4&selection=62,0,64,10&color=yellow)
> > One fundamental characteristic is the scaling law, which describes how the performance of large language models improves as they scale in terms of model size, training data, and computational resources.
> 
> 大模型的一个特点就是缩放定律,揭露了随着模型的规模,数据量和计算规模的夸大,大模型的性能如何提升.实证证据表明，随着模型规模和预训练数据量的增加，LLMs在下游任务上的表现得到改善。该幂律关系表明，规模更大的模型在更多数据的训练下，能够捕捉到更复杂的模式，并在新任务中表现出更好的泛化能力。

> [!PDF|yellow] [Page.4](MachineLearning/论文阅读/OFA/从数据中心角度看多模态大模型/ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=4&selection=77,0,78,44&color=yellow)
> > Another intriguing property of LLMs is the emergence of abilities that were not explicitly trained for, often referred to as emergent abilities
> 
> 另一个令人瞩目的能力就是llm的涌现能力,LLm能从预训练数据中捕获并且利用复杂的语言和知识,从而执行超出原始目标任务的能力.

> [!PDF|yellow] [Page.4](MachineLearning/论文阅读/OFA/从数据中心角度看多模态大模型/ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=4&selection=84,58,89,75&color=yellow)
> >  Furthermore, LLMs have demonstrated the capability of in-context learning [24], where they can perform tasks based on a few examples provided in the input prompt, without the need for explicit fine-tuning. This highlights the models’ ability to rapidly adapt to new tasks and generalize from limited examples.
> 
> 此外，LLMs还展示了**上下文学习**的能力，即在输入提示中提供少量示例时，能够基于这些示例执行任务，而无需显式微调。
