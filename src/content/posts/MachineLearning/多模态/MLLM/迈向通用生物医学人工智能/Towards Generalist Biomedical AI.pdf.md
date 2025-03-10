---
title: "[[Towards Generalist Biomedical AI.pdf|Towards Generalist Biomedical AI]]"
description: ""
image: ""
tags:
  - 机器学习
  - 论文阅读
category: 论文阅读
draft: true
published: 2024-10-18
last modified: 2024-10-18 16:13
---

> [!PDF|yellow] [Page.1](Towards%20Generalist%20Biomedical%20AI.pdf#page=1&selection=99,51,100,22&color=yellow)
>
> >  we first curate MultiMedBench, a new multimodal biomedical benchmark.
>
> 我们首先开发了一个新的 benchmark 方法 (MUutilMedBench),包含 14 个不同的任务比如医学问题,放射学报告生成和总结.

> [!PDF|note] [Page.1](Towards%20Generalist%20Biomedical%20AI.pdf#page=1&selection=103,39,108,2&color=note)
>
> > Med-PaLM M is a large multimodal generative model that flexibly encodes and interprets biomedical data including clinical language, imaging, and genomics with the same set of model weights.
>
> 我们还提出了一个新的大型多模态生成模型，能够用同一套模型权重灵活地编码和解释包括临床语言、影像和基因组学在内的生物医学数据。

# Introduction

> [!PDF|yellow] [Page.1](Towards%20Generalist%20Biomedical%20AI.pdf#page=1&selection=132,66,133,80&color=yellow)
>
> > This bounds performance and utility of these narrow, single-task, unimodal, specialist AI systems in real-world applications.
>
> 单模态的坏处

> [!PDF|yellow] [Page.2](Towards%20Generalist%20Biomedical%20AI.pdf#page=2&selection=84,0,93,1&color=yellow)
>
> > In this work, we detail our progress towards such a generalist biomedical AI system - a unified model that can interpret multiple biomedical data modalities and handle many downstream tasks with the same set of model weights.
>
> 目标是创建一个通用的 AI 模型,能够解释多种医学模态数据和处理不同的下游任务使用同一套权重.其中的一个挑战就是这方面的评价指标缺失,因此我们使用了 MutilMedBench,有 14 个不同的任务上打指标.

我们开发了一个多模态的医学模型,总的来说,

> [!PDF|red] [Page.2](Towards%20Generalist%20Biomedical%20AI.pdf#page=2&selection=102,0,105,62&color=red)
>
> > In particular, Med-PaLM M is a flexible multimodal sequence-to-sequence architecture that can easily incorporate and interleave various types of multimodal biomedical information. Further, the expressiveness of the modality-agnostic language decoder enables the handling of various biomedical tasks in a simple generative framework with a unified training strategy
>
> 这是一个 seq2seq 架构的模型,能够把轻易地整合多种多模态的生物信息.模态无关的语言解码器也能让他在一个简单的生成框架中处理不同的医学任务

我们的贡献如下;

1. 提出了 MutilMedBench
2. 第一个通用人工智能医学 AI 模型
3. 全新的涌现能力的现象
4. 人类对 AI 效果的评估

## Related Work

### 基础模型,多模态和 Generalists

> [!PDF|yellow] [Page.3](Towards%20Generalist%20Biomedical%20AI.pdf#page=3&selection=74,0,76,56&color=yellow)
>
> > Visual foundation models such as CLIP [30] are made possible by training on language-labeled visual datasets [25, 31], which are easier to collect from large-scale internet data than classification datasets with pre-determined class labels
>
> CLIP 可以训练基于语言标注的视觉数据集,使得从网络上获得大规模的数据集更简单.

> [!PDF|yellow] [Page.3](Towards%20Generalist%20Biomedical%20AI.pdf#page=3&selection=79,53,80,60&color=yellow)
>
> > Further, the flexibility of language also enables a wide range of task specifications all via one unified output space
> 语言的灵活性可以让模型通过一个统一的范式进行不同任务.

我们的模型也从大量的在 vision-language dataset 上 pretrain 的模型进行收益,然后再医学领域上进行微调.Med-PaLM M 则是专为生物医学领域设计的通用模型，通过微调和对齐 PaLM-E 通用模型构建而成。

## 生物学中的多模态基础模型

> [!PDF|yellow] [Page.4](Towards%20Generalist%20Biomedical%20AI.pdf#page=4&selection=32,40,33,82&color=yellow)
>
> > owever, all these efforts are pretrained models and as such they require further task-specific data and finetuning to enable downstream application
>
> 之前的模型依赖于下游任务微调,而我们的模型同时在多个任务上进行㜕并且不需要更多的参数更新

> [!PDF|yellow] [Page.4](Towards%20Generalist%20Biomedical%20AI.pdf#page=4&selection=35,37,37,15&color=yellow)
>
> > . LLaVA-Med [47] is perhaps most similar to our effort. The authors use PubMed and GPT-4 [48] to curate a multimodal instruction following dataset and finetune a LLaVA model with it.
>
> LLaVa-Med 也是一个很好的模型,到时候可以看一下

## MutilMedBench

> [!PDF|important] [Page.4](Towards%20Generalist%20Biomedical%20AI.pdf#page=4&selection=60,45,63,54&color=important)
>
> >  It measures the capability of a general-purpose biomedical AI to perform a variety of clinically-relevant tasks. The benchmark covers a wide range of data sources including medical questions, radiology reports, pathology, dermatology, chest X-ray, mammography, and genomics. Tasks in MultiMedBench vary across the following axes:
>
> 它包括了: 医学问答,放射学诊断报告,病理,胸部 X 光,基因组学,皮肤科,乳腺摄影等数据

输入输出的格式如下:

- 任务类型: 问答,报告生成和总结,视觉问题问答,医学图像分类和基因变异识别
- 模态: 文本,放射图,病理,皮肤病,乳腺摄影和基因组学
- 输出格式: 包括分类在内的所有开放性生成问题

**那么数据集在哪下载呢 (?)**

## Med-PaLM: 医学多面手的概念验证

**Pathways LM**: 这是一种只包含解码器的 Transformer 的 LLM,，该模型通过 Pathways 进行训练。Pathways 是一个大规模的机器学习加速器编排系统，可高效地在 TPU 集群上进行训练。

**PaLM-E**: 是一个多模态语言模型,能处理多模态的序列包括文本,视觉和传感器信号,PaLM-E 使用了 VIT 和 PaLM,体现了很强的性能,能够结合图片,文本以及传感器信号.

## 把他们放到一起:Med-PaLM M

Med-PaLM M 是通 MutilMedBench 进行微调并且调整模型以适应医学领域开发的,下面是模型的细节

**数据集和预处理**: 我们的数据集使用 224x224x3 的图片大小,同时在必要时通过填充保持原始的宽高比。对于灰度图像，我们通过在通道维度上堆叠相同图像将其转换为三通道图像。

**指令任务提示和一次性范例**: 我们通过改变指令通化市训练混合任务来对模型进行微调,<font color="#9bbb59">我们为不同的任务提供不同的指令来在一个统一的框架中执行任务</font>.提示包括指令,相关的上下文以及问题.为了让模型更好的遵循指令,对于主要任务,我们在任务提示中添加了仅包含文本的“单样本示例”，以条件化语言模型的预测。这个单样本示例通过部分输入输出对来帮助提示模型。比如把图片替换为一个占位符\<img>

## Evaluation
