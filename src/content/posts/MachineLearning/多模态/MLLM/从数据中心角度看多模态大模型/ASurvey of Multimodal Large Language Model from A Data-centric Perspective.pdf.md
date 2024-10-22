---
title: "[[ASurvey of Multimodal Large Language Model from A Data-centric Perspective.pdf|ASurvey of Multimodal Large Language Model from A Data-centric Perspective]]"
published: 2024-10-17
description: ""
image: ""
tags:
  - 机器学习
  - 论文阅读
category: 论文阅读
draft: true
---

> [!PDF|yellow] [Page.2](ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=2&selection=60,0,61,22&color=yellow)
>
> > Most existing MLLMs focus on modifying model architecture to explore the use of information from multiple modality
>
> 许多的任务目前考虑修改模型的架构来高效的利用信息

> [!PDF|yellow] [Page.2](ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=2&selection=83,40,84,47&color=yellow)
>
> > data also significantly impacts the success of MLLMs
>
> 对大模型来说数据也很重要,包括数据质量和数量

> [!PDF|red] [Page.2](ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=2&selection=104,3,105,56&color=red)
>
> > Our discussion answers three key questions from a data-centric perspective at different stages of MLLMs:
>
> 1. 怎么样选择大模型的数据,需要选择高质量,异构的数据,模型的不同阶段也有不同的数据需求
> 2. 数据如何影响大模型的
> 3. 怎么评价数据对多模态大模型的影响

# LLM 的三大能力

> [!PDF|yellow] [Page.4](ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=4&selection=62,0,64,10&color=yellow)
>
> > One fundamental characteristic is the scaling law, which describes how the performance of large language models improves as they scale in terms of model size, training data, and computational resources.
>
> 大模型的一个特点就是缩放定律,揭露了随着模型的规模,数据量和计算规模的夸大,大模型的性能如何提升.实证证据表明，随着模型规模和预训练数据量的增加，LLMs 在下游任务上的表现得到改善。该幂律关系表明，规模更大的模型在更多数据的训练下，能够捕捉到更复杂的模式，并在新任务中表现出更好的泛化能力。

> [!PDF|yellow] [Page.4](ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=4&selection=77,0,78,44&color=yellow)
>
> > Another intriguing property of LLMs is the emergence of abilities that were not explicitly trained for, often referred to as emergent abilities
>
> 另一个令人瞩目的能力就是 llm 的涌现能力,LLm 能从预训练数据中捕获并且利用复杂的语言和知识,从而执行超出原始目标任务的能力.

> [!PDF|yellow] [Page.4](ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=4&selection=84,58,89,75&color=yellow)
>
> > Furthermore, LLMs have demonstrated the capability of in-context learning [24], where they can perform tasks based on a few examples provided in the input prompt, without the need for explicit fine-tuning. This highlights the models’ ability to rapidly adapt to new tasks and generalize from limited examples.
>
> 此外，LLMs 还展示了**上下文学习**的能力，即在输入提示中提供少量示例时，能够基于这些示例执行任务，而无需显式微调。

> [!PDF|yellow] [Page.4](ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=4&selection=90,0,91,14&color=yellow)
>
> > The development and deployment of LLMs typically involve three key stages: training, adaptation, and evaluation
>
> 包含三个阶段,训练,泛化和评估.其中训练阶段主要关注模型从大规模的无标签语料中学习通用的语言表示,捕捉自然语言的基本结构,适应阶段是通过领域适应和任务微调技术将模型整合到特定的领域中,这一步对优化模型在目标任务上的应用非常重要,最后通过评估阶段对其进行评估

> [!PDF|yellow] [Page.5](ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=5&selection=25,90,28,19&color=yellow)
>
> > By integrating LLMs with multimodal projectors, MLLMs can process various types of information, demonstrating strong understanding and analytical capabilities to address downstream tasks across various modalities
>
> 将 LLM 与其他的多种模态数据结合,能使得多模态大模型能够在不同模态的下游任务中进行理解.

> [!PDF|yellow] [Page.7](ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=7&selection=20,0,20,82&color=yellow)
>
> > Analyzing LLMs and MLLMs from a data-centric perspective offers several advantages
>
> 从数据中心角度来处理有很多的好处
>
> 1. 能够帮助研究人员识别训练数据中的潜在偏差与局限
> 2. 通过精心处理和扩充数据集,能够保证数据的多样性和代表性
> 3. 数据中心的方法能够帮助研究人员基于训练数据更好的对模型的能力和局限进行评估,通过改变训练数据来进行模型性能的洞察,
> 4. 为高效数据学习提供了机会,比如少样本学习和提示工程能够帮助模型在数据有限的情况下进行训练
> 5. 有助于开发更具有可解释性的模型

> [!PDF|blue] [Page.8](ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=8&selection=80,0,80,89&color=yellow)
>
> > CommonCrawl project serves as the most commonly used start-point for large-scale webpages
>
> 这些网站包括 CommonCrawl,wudaocorpora crwal,

> [!PDF|important] [Page.10](ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=10&selection=59,0,86,64&color=important)
>
> > In the medical domain, pre-training data comes from online medical websites, knowledge bases, and in-hospital database systems. Online medical websites such as Qianwen Health and PubMed contribute to datasets like Huatuo-26M [ 153 ] and MedQuAD [19]. Knowledge bases like Wikipedia also contribute to Huatuo-26M [ 153] and MedHop [ 278 ]. In-hospital database systems, including electronic health record systems (EHR) and ICU-specific clinical information systems, are used in MIMIC-IV [ 118 ]. Multimodal datasets in radiology, such as MIMIC-CXR-JPG [ 119 ] and PADCHEST [25], are generated from in-hospital radiology reports.
>
> very import! 到时候可以从这个

### 过滤

> [!PDF|yellow] [Page.10](ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=10&selection=121,93,123,7&color=yellow)
>
> > For English-only datasets, commonly used tools like Langdetect and FastText operate at the document level.
>
> 对与文本过滤,可以使用**语言过滤**和**内容过滤**, 低于特定阈值的会被移除,比如英文数据集常用 Langdetect 和 fastText 用作文档过滤.
>
> 对于中文数据聚集,WuDaoCorpora 采用启发式规则规律橱包含连续 10 个以上非中文字符的网页
>
> 对于代码,只需要限定文件扩展名即可

另一种方式为基于内容的过滤,可以过滤掉有害和干扰内容,比如脏话和 html 标签,不完整句子等内容

> [!PDF|yellow] [Page.11](ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=11&selection=56,0,57,72&color=yellow)
>
> > For image-level filtering, the most fundamental step is to remove images with excessively low resolution, as these images often fail to convey effective information.
>
> 对于图像来说.最基本的就是删除分辨率过低的图像,因为这些图像没有效信息,还需要过滤宽高比不合适的一级有害内容的.可以训练一个二分类模型来达到这种效果.还需要对数据进行脱敏

### 去重

先前的研究发现有很多的重复数据,通过对数据集进行去重,可以防止记忆化问题(隐私,模型质量),还可以降低训练成本

> 更多的重复数据可能会导致模型逐字输出记忆化数据

现有的去重方法可以分为:精确去重,语义去重和近似去重

1. 精确去重去除完全一致的字符串
2. 语义去重使用模型进行去重
3. 近似去重针对文档级别的重复,使用哈希等算法移除近似文档

### 增强

数据增强主要集中在两个方面,增强 x 模态数据和增强文本.

> [!PDF|yellow] [Page.12](ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=12&selection=184,59,207,2&color=yellow)
>
> > Traditional data augmentation methods for single modality data have been discussed thoroughly in previous work [ 27 , 69, 132 , 277 , 295 , 323].
>
> 在本文中不考虑单模态数据增强

对于视觉-语言模型,增加图像-字幕数据集的质量对于训练大模型至关重要,能够避免因为文本质量太差而丢失信息.而提升图像分辨率也能够大幅提升图像的性能,最近的模型将图像分辨率从 224-336-896 等大小,大幅提升了图像的能力.

> [!PDF|note] [Page.13](ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=13&selection=78,47,80,96&color=note)
>
> > However, a fundamental challenge remains within the MLLM architecture: models using lower-resolution inputs struggle to detect fine details, whereas those with higher resolutions may underperform in tasks requiring a broader global understanding
>
> 低分辨率的模型缺少细节,高分辨率模型泛用性差

## Data-center pre-training

模型的预训练可以被分为两个不同的阶段

第一个阶段是使用文本数据 pre-trainging llm backbone 和预训练模态特定的 encoder.这样的目标是让模型有一个较强的基础能力.

第二个阶段是使用多模态数据进行训练.这个阶段使用输入投影器将不同的特征投影到一个统一的 LLM 嵌入空间,让文本与非文本信息有一个统一的表示.

> [!PDF|yellow] [Page.16](ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=16&selection=97,0,98,92&color=yellow)
>
> > Some researchers consider employing a selective training approach during the second phase, which involves keeping certain components of the model frozen while training specific parts.
>
> 部分研究者在第二个阶段冻结了部分模型的参数 保留第一个阶段的训练知识并且降低计算成本

### Domain mixture

语言 model 的性能受到预训练数据组成的影响,先前的方法通过启发式手段或者基于下游任务优化领域权重,并且可能会对下游任务过拟合

### 模态融合

在预训练模型中,确定多模态数据的最佳比例对于提升模型在不同任务上非常重要.
随着发展,目前的视频 llm 模型朝着多分支训练的方法现进行训练.比如视觉-语言,音频-字幕等,这些方法利用不同模态的相关性,扩展了模型的理解和学习能力.这种模块化训练方法提供了灵活性,可以进行部分数据的预训练.

### 质量选择

由于数据分布的不一致,使用所有的数据进行训练是次优的,因此数据的选择变得非常重要

> [!PDF|red] [Page.17](ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=17&selection=144,3,146,42&color=red)
>
> > Unlike pure text datasets, data selection for multimodal datasets must consider the alignment between different modalities, in addition to metrics like perplexity used for single-modality evaluation [269].
>
> 相较于文本数据,多模态数据集的选择必须考虑不同模态之间的对齐

数据选择有主动学习和预训练选择方法,主动学习方法动态的在训练过程中选择数据,pretrain 方法在训练开始前评估并且选择了所有的数据.

对于选择的数据分布,可以只考虑单点数据的质量,也可以考虑总体数据分布.分布无关的方法上可以使用 clip 分数的前 30%的数据进行训练,也可以使用最近提出的一些复杂选择方法

> [!PDF|yellow] [Page.18](ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=18&selection=19,3,25,38&color=yellow)
>
> > For instance, Mahmoud et al . [188] use the difference between original and synthetic captions generated by a small model to evaluate image-text pair alignment.
>
> 复杂选择方法在这里

## 数据为中心的适应

adaptation 对与预训练的大模型和下游任务的对齐是十分关键的.
自监督预训练为大模型提供了广泛的文本和多模态信息理解,而监督微调(SFT)和基于人类反馈的强化学习(SLHF)让这些模型在特定应用中符合人类的规范.前者依赖于在多个模态下使用精心选择的数据集,后者着重人类的判断,量或者突出了高质量数据集的重要性

### 数据中心的自监督微调

supervised fine-tune 成为了多模态大模型适应不同领域特定任务不可或缺的技术,通过精心设计的 Instruction 来让模型从数据集中学习必要的能力.

在自监督微调阶段探索了不同的策略,包括微调:LLM backbone,modality encoders 和 input projector 中的全部或者部分,这种选择性的方法在适应特定任务和保留预训练期间获得的通用知识之间取得了平衡。为了实现有效的监督微调，使用了与目标领域和任务相匹配的多样化数据集，这些数据集通常包括多模态的指令-响应对和仅文本的 SFT 数据。

> [!PDF|yellow] [Page.18](ASurvey%20of%20Multimodal%20Large%20Language%20Model%20from%20A%20Data-centric%20Perspective.pdf#page=18&selection=63,0,63,79&color=yellow)
>
> > The curation of self-supervised fine-tuning datasets usually includes four step
>
> 预训练的步骤包括:
>
> 1. 从不同数据源中获得 X-text 数据对
> 2. 处理数据
> 3. 将 X-text 数据对处理为指令-响应的形式,包括纯设计指令和转变为回答
> 4. 根据微调的需要从中选择高质量的指令

指令-响应对的生成。在收集和处理原始 X-文本数据集之后，下一步是基于原始数据集生成指令-响应对数据集。监督微调阶段的目标通常是增强模型在下游任务中的能力。因此，已有工作会针对不同的下游任务生成指令-响应对，包括描述性任务、问答任务、推理任务和分类任务。基于任务分类，前人的工作选择不同的数据源，并采用多种方法生成指令-响应对。

LLM 的主题知识来源于预训练,而指令微调的部分目的是让模型学习如何在特定的任务上出色的与人类进行交互.

## 未来方向

### MLLM 的数据处理系统

为 MLLM 处理数据涉及到多个模态的复杂步骤,需要为 MLLM 设计专门的自动化处理流程.需要具有处理多模态数据的能力,涵盖图像,音频等.

### MLLM pre-training 的数据数量分析

数据数量如何影响多模态大语言模型的涌现能力仍然是空白的.或者可以去探索数据评估方面

### MML pre-training 的数据质量分析

需要开发代理模型来评估数据的质量.代理模型与原始模型的关系仍然需要进一步的研究.

### 数据评估

尽管引入了各种数据评估指标，但针对多模态数据的综合性指标仍然缺乏。多模态数据评估更具挑战性，因其包含多种数据类型，往往涉及多个任务和模态

### 数据质量提升对于指令微调的影响
迫切需要设计专门针对指令微调数据的评估指标。另一种实用的方法是利用大语言模型（LLMs）进行数据评估，借助其在预训练阶段积累的大量知识。然而，虽然使用LLMs评估数据具有成本效益，但这些方法通常缺乏可解释性。

### MLLM终身学习

防止在不同阶段中学习后产生灾难性遗忘