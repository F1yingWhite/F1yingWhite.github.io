---
title: "[[nextgpt.pdf|nextgpt]]"
description: ""
image: ""
tags:
  - 机器学习
  - 论文阅读
category: 论文阅读
draft: true
published: 2024-10-22
last modified: 2024-10-22
---

# Abstract & Instruction

> [!PDF|yellow] [Page.1](nextgpt.pdf.md#page=1&selection=44,1,48,41&color=yellow)
>
> > By leveraging the existing well-trained high-performing encoders and decoders, NExTGPT is tuned with only a small amount of parameter (1%) of certain projection layers
>
>修改的参数量小

> [!PDF|yellow] [Page.1](nextgpt.pdf.md#page=1&selection=51,12,52,37&color=yellow)
>
> > Moreover, we introduce a modalityswitching instruction tuning (MosIT)
>
>引入了模态切换指令并且手动创建高质量数据集

> [!PDF|yellow] [Page.1](nextgpt.pdf.md#page=1&selection=103,0,105,51&color=yellow)
>
> > With such intuition, the purely text-based LLMs have recently been endowed with other modal understanding and perception capabilities of image, video, audio, etc
>
>未来导向

一个值得注意的方法就是把预训练的 encoders(来自其他模态) 和文本 LLM 进行对齐.但是他们绝大多数都关注了输入侧的多模态内容理解.

> [!PDF|red] [Page.1](nextgpt.pdf.md#page=1&selection=121,44,126,59&color=red)
>
> > We emphasize that natural human cognition and communication indispensably require seamless transitions between any modalities of information. This makes the exploration of any-to-any MM-LLMs critical, i.e., the ability to accept inputs in any modality and deliver responses in any appropriate modality.
>
>人类能在多模态之间无缝切换,那么大模型也要接受任意的输入和做出任意模态的输出

> [!PDF|yellow] [Page.2](nextgpt.pdf.md#page=2&selection=103,44,105,43&color=yellow)
>
> >  First, the information transfer between different modules is entirely based on discrete texts produced by the LLM
>
>之前的研究的信息传递完全依赖于 LLM 的文本,不可避免的映入了噪声.更严重的是这些系统利用预训练的工具只做推理.

由于缺少端到端的训练,多模态的理解能力是被严重限制的.因此我们提出了 Next-GPT,一种 OFA 的 MM-LLM 来接受任意的输入包括 text,image,video 和 audio.

![image.png](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/20241022093337.png)

我们的模型分为 3 步:

1. 利用已有的编码器对各种输入模态编码,然后通过**投影层被投影为类语言表示**
2. 利用现有的开元大模型来进行语义理解和推理,LLM 不仅生成文本信号,还生成了控制指令来标记是否输出以及输出哪个模态
3. 经过投影之后,产生的多模态信号被路由到对应的 encoder 中并最终输出对应的模态
为了再三层中实现特征对齐,我们仅微调输入和输出的 projection layer,**采用编码端以 LLM 为中心的对齐方式和解码端的指令跟随对齐方式，最小化计算开销以确保更高的效率。** 此外我们还采用 loar 微调技术对整个系统进行指令微调,对整个系统进行指令微调,更新输入输出投影以及部分 LLM 参数

# Overall Architecture

我们的 GPT 一共包含三个架构:encoder,LLM 和 decoder

**Muitimodal Encoding Stage**: 首先,我们利用预训练模型来 encoder 多种输入,然后使用投影层,把不同的输入投影到类语言表示层.

> [!PDF|yellow] [Page.3](nextgpt.pdf.md#page=3&selection=120,16,123,1&color=yellow)
>
> > ere we take advantage of the ImageBind (Girdhar et al., 2023), which is a unified high-performance encoder across six modalities.
>
>ImageBind 可以同时接受 6 种模态的输入..?

**LLM 理解和推理阶段**: 我们使用 Vicuna 作为我们的大模型,这个模型在任务中被广泛利用.LLM 接收来自不同模态的输入表示并且进行语义理解和推理,它输出文本特征和每个模态的指示信号
**多模态生成阶段**: 在接收到来自大语言模型（LLM）的具有特定指令的多模态信号后，基于 Transformer 的输出投影层将信号标记表示映射为能够被后续多模态解码器理解的表示形式。

> [!PDF|yellow] [Page.4](nextgpt.pdf.md#page=4&selection=0,35,4,97&color=yellow)
>
> > NExT-GPT: Any-to-Any Multimodal LLM Table 1. Summary of NExT-GPT system configuration. Only 1% of parameters need updating during fine-tuning.
>
>我们的模型只需要微调输入输出层以及 lora 微调 LLM 的一小部分即可.这部分只占全参数的 1%

## 轻量的多模态对齐学习

为了建立不同模态之间的 gap 并且保证理解不同的语义,由于我们的解耦设计,我们的训练量只需要在两个 projection 层上进行对齐.

### 编码侧以 LLM 为中心的模态对齐

许多现有的架构基于 Transformer 架构的多模态 encoder 并且生成 patch 级别的网格特征 (比如图像,音频),它们通过线性层将多模态特征直接投影到文本特征空间，使其能够被核心大语言模型理解。但是这样通常无法与语言文本很好的对应,因为语言通常表示额外的概念,导致 MLLM 次优.因此我们设计了一种可学习的 token 来网格级特征层次化的聚类为语义概念标记.最后把概念化的表示输入到 LLM 中去

> [!PDF|red] [Page.4](nextgpt.pdf.md#page=4&selection=186,43,190,52&color=red)
>
> > However, we note that the patch-based feature units might not best coincide with the intricate textual token semantics, as intuitively the language tokens always encapsulate separate concepts. This may result in suboptimal information perception

为了完成对齐,<font color="#9bbb59">我们使用了一种 X-to-text 的生成任务在 X-caption 对上进行训练,也就是给定一个 X,来提示 LLM 来生成对应的文本描述.</font>也是非常经典的方法了.

>这里的具体方法需要参考 GroupViT 这篇文章中的内容

![image.png](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/20241022104211.png)

### 解码器侧指令对齐

这部分主要是吧 LLM 的输出跟预训练模型进行对齐.但是

> [!PDF|red] [Page.4](nextgpt.pdf.md#page=4&selection=224,18,226,49&color=red)
>
> > However, performing a full-scale alignment process between each diffusion model and the LLM would entail a significant computational burden.
>
>在 LLM 和每个模型之间进行全量的对齐会产生很大的计算负担

因此我们采用了一种更高效的方法,也就是解码器侧遵从指令的对齐,我们让 LLM 输出三种特别的 token 而不是生成直接的文本指令.

1. \[img] 作为图片的信号 token
2. \[audio] 作为音频信号 token
3. \[ViD] 作为 video 信号
这些 token 为下游的任务带来了很多的信息如果 LLM 识别到一种特殊的 token,那么就激活对应的模态,否则就停止

我们注意到扩散模型只依靠文本作为表示,这种和我们大模型输出的 token 有较大的区别,导致模型无法很好的解释 LLM 的命令,因此，一方面，我们考虑将 LLM 的模态信号标记表示（经过每个基于 Transformer 的投影层后）作为去噪过程中的条件输入，来引导扩散模型生成合适的图像、视频或音频。另一方面，我们还提出最小化投影后的信号标记表示与扩散模型中的条件文本表示之间的距离，以加速对齐学习。需要注意的是，所有扩散模型的主干网络（如 U-Net）保持冻结状态，这也确保了训练的轻量化。

![image.png](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/20241022104410.png)

## 模态切换指令微调

虽然在 encoding 和 decoding 部分进行了对齐,但是完全理解人类的问题依然比较困难,因此需要进一步指令微调.我们使用 LoRA 微调技术,仅仅微调 LLM 和两个投影层的一小部分.当接受到一个输入的时候,模型输出对应的 output 还有 signal,优化基于金标准和模型的输出进行,此外,我们还根据生成的图片的区别进行了解码端 projection 的微调.

## Input Projection 细节

通过多模态编码器，我们可以获取分块级别的多模态标记，表示为 $\mathbf{X}^* = \{\mathbf{x}_i^*\}_{i=1}^{N^*}$，其中 $* \in \{i, a, v\}$ 分别表示图像、音频和视频。为了简洁起见，我们省略了模式特定的符号。不同于现有的通过线性投影层直接嵌入多模态标记到大型语言模型 (LLM) 的方法，我们提出了一种多阶段分组机制，将分块级标记分组为概念级标记，以便促进后续的跨模态交互。形式上，我们应用 $L$ 个分组阶段，在每个阶段中，我们随机初始化 $M_l$ 个可学习的概念标记 $\mathbf{C}^l = \{c_j^l\}_{j}^{M_l}$。然后，我们将输入特征 $\mathbf{X}^l$ 和 $\mathbf{C}^l$ 连接起来并输入到 transformer 层中：

$$ \hat{\mathbf{C}}^l, \hat{\mathbf{X}}^l = \text{Transformer}([\mathbf{C}^l; \mathbf{X}^l]) $$

其中 $\mathbf{X}^1 = \mathbf{X}$，$[;]$ 表示连接操作。在每个 $l$ 分组块中，我们基于特征相似性将更新后的 $M_l$ 个概念标记 $\hat{\mathbf{X}}^l$ 分组为 $M_{l+1}$ 个新概念标记 $\hat{\mathbf{X}}^{l+1}$。

具体而言，我们首先通过 Gumbel-softmax 计算 $\hat{\mathbf{C}}^l$ 和 $\hat{\mathbf{X}}^l$ 之间的相似性矩阵 $\mathbf{A}^l$：

$$ \mathbf{A}^l = \text{Softmax}((\text{Norm}(\hat{\mathbf{C}}^l)\text{Norm}(\hat{\mathbf{X}}^l) + G) / \tau) $$

其中 $G$ 是从 Gumbel(0, 1) 分布中独立同分布采样的随机变量，$\tau$ 是可学习的显著性系数，用来帮助找到更合适的分配边界。我们计算分组并通过 argmax 对所有组进行一次性操作来分配概念标记。由于一次性分配操作的 argmax 不可微分，我们采用了直通技巧来计算分配矩阵 $\hat{\mathbf{A}}^l = \text{Onehot}(\text{Argmax}(\mathbf{A}^l)) + \mathbf{A}^l - \text{Sg}(\mathbf{A}^l)$，其中 $\text{Sg}(.)$ 是停止梯度操作符。最后，我们将特征整合为更新后的概念标记：

$$ \mathbf{X}^{l+1} = \hat{\mathbf{C}}^l + \text{MLP}(\hat{\mathbf{A}}^l, \hat{\mathbf{X}}^l) $$

经过 $L$ 个阶段的分组后，我们可以获得 $M_L$ 个概念标记 $\mathbf{X}^L$，这些标记被输入到 LLM 中进行感知和推理。
