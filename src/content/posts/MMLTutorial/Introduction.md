---
title: 多模态深度学习入门
published: 2024-08-22
description: '多模态深度学习入门导论'
image: ''
tags: [多模态，导论]
category: '导论'
draft: true

---

## Multimodal Machine Learning:A Survey and Taxonomy

### abstract

我们对世界的感官是多模态的，模态指的是当一个事件发生的时候是怎么被感知到的，当有多种方式被感知的时候，就叫做多模态问题。为了让AI来了解我们周围的世界需要让和他结合多种模态的数据到一起。MML旨在建立一个能够结合多种模态信息模型。

目前，多模态的挑战主要有：*representation, translation, alignment, fusion, and co-learning*。

### 基本分类

1. **表示 (Representation)**  
    第一个基本挑战是学习如何表示和总结多模态数据，以充分利用多种模态的互补性和冗余性。多模态数据的异质性使得构建这种表示形式变得具有挑战性。例如，语言通常是符号性的，而音频和视觉模态则表示为信号。
2. **转换 (Translation)**  
    第二个挑战是如何将一种模态的数据转换（映射）为另一种模态的数据。不仅数据是异质的，而且模态之间的关系往往是开放式的或主观的。例如，对一幅图像的描述方式可能有多种正确选项，且可能不存在一个完美的转换。
3. **对齐 (Alignment)**  
    第三个挑战是识别来自两种或多种不同模态的（子）元素之间的直接关系。例如，我们可能希望将一个食谱中的步骤与展示菜肴制作过程的视频进行对齐。应对这一挑战需要衡量不同模态之间的相似性，并处理可能存在的长距离依赖和模糊性。
4. **融合 (Fusion)**  
    第四个挑战是将来自两种或多种模态的信息结合起来进行预测。例如，在音频-视觉语音识别中，将唇部运动的视觉描述与语音信号融合，以预测所说的单词。来自不同模态的信息可能具有不同的预测能力和噪声拓扑结构，且至少有一个模态的数据可能缺失。
5. **共学习 (Co-learning)**  
    第五个挑战是跨模态、它们的表示形式及其预测模型之间转移知识。这在共同训练、概念落地和零样本学习的算法中得以体现。共学习探讨了如何利用从一种模态中学习的知识来帮助训练另一种模态的计算模型。

### MML历史

最早的MML研究是AVSR视听语音识别，为了提升语音识别的准确率。在这个任务中发现模态之间的交互作用是补充性的，而非互补性的。两种模态中捕捉到的是相同的信息，这提高了多模态模型的鲁棒性，但在无噪声的情况下并没有提升语音识别性能。

第二个重要的MML使用来源于多媒体内容检索领域，第三类出现的应用是在2000年左右出现，旨在了解人类在社交活动中的多模态行为。

### 多模态表示

在计算模型中表示原始数据一直是计算机领域的一个困难，我们使用feature和representation表示一个实体的向量。一个多模态的表示就是多个实体的向量集和。表示多模态的数据有很多困难，如何从多源异构的数据源中表示多种模态，如何处理不同程度的噪声，如何处理缺失的数据。

好的数据表示对于模型的表现非常重要，Bengio提出了好的数据表示的几个指标：smoothness,temporal和spatial coherence，sparsity and natural clustering，目前，单模态的数据表示已经被广泛的研究了，在过去的10年内，任务已经变成了数据驱动。为了更好的理解我们的任务，我们提出了两种数据多模态的表示方法：==joint和coordinated==,其中joint数据表示方法能够将单模态的数据特征组合到一个共同的表示空间，coordinated表示方法分别处理单个型号，但在它们之间强制某些相似性约束，将它们带入我们称之为“协调空间”的领域。。

数学上，joint的表示方法如下:$$x_m=f(x_1,...,x_n)$$,而coordinated表示发如下$$f(x_1)~g(x_2)$$。每个模态都有一个映射方法，能够将他从一个模态投影到多模态空间中。每个模态的投影方法相互独立，但是最终的空间是协调的。这种方法有最小余弦距离，最大相似度等

![image-20240822161319021](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20240822161319021.png)

#### Joint表示法

joint表示法把所有特征全部扔进去结合投影到一个空间中。联合表示法是最常被使用的方法，最简单的方法就是concat在一起，在本章中我们探讨一些使用网络的更高级的表示方法。

**神经网络**：为了让网络能够学习到一种方式来表示数据， 网络首先通过特定的任务来进行训练，因为网络越深数据越抽象，通常使用倒数1~2层的数据来表示数据。每种模态都从几个独立的神经层开始，随后是一个将这些模态映射到联合空间的隐藏层。

因为模型需要很多标记数据，通常在自动编码器和无监督训练方法上使用预训练，或者在别人训练好的模型上进行微调。

使用网络的好处是其具有更加优越的性能和无监督预训练表示。但是性能的提升依赖于数据量的大小，还有个缺点是无法处理数据的缺失。

**概率图模型**：TODO

**序列表示法**：目前为止我们只讨论了显示定长的数据，但是我们经常需要表示变长的数据格式比如video，audio。循环神经网络和他们的变体比如LSTM通常用与处理变长的数据。与神经网络类似，RNN的隐藏层状态也可被看做是数据的表示。

#### Coordinated表示法

与将多个模态投射到一个空间不一样，我们学习每个模态的单独表示，但是对他们施加一定的约束

**相似度模型**最小化模态在coordinated空间之间的距离。比如我们把dog和dog的图片的距离变小，和猫的图片距离变大，深度学习在这个任务中的好处是可以进行一个端到端的表示。

**结构化coordinated空间模型**对模态特征的表现之间增加了额外约束，常用与把高维数据压缩为紧凑的二进制码，对于相似的对象拥有相似的二进制码。夸模态哈希的思想是为夸模态检索创建这样的代码。TODO

目前这种方法一般只限于两个模态，而joint方法适用于多个模态。

### Translation

翻译就是给定一个模态的实体的情况下生成另一个模态的实体，比如有一张图片，现在需要生成一段句子来描述他

目前主要分为两种类型：example-based和generative。exmaple-based的模型使用字典来进行模态之间的转换，generative模型能够进行模态的转换。前者从字典中找出最合适的目标，而后者是直接生成。

![image-20240823100931970](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20240823100931970.png)



#### Example-based

基于Example的算法被他的训练数据（dictionary）严格限制。他也分类两类，一类是Retrieval-based（基于检索），另一类是*combination*-based models

**retrieval-based**模型是最简单的翻译，他依赖于找到字典中和输入样本最近的sample

**Combination-baed**模型不是简单的从字典中检索结果，而是将结果进行combine得到新的结果

基于Example的模型的最大问题就是模型本身只是一个字典，是的模型变得庞大而且推断速度较为缓慢。而且期待从源中找到完全符合的sample是不显示的。

#### Generative approaches

面向多模态翻译的生成方法构建了能在给定单模态源实例的情况下执行多模态翻译的模型。这是一个具有挑战性的问题，因为它不仅要求模型能够理解源模态，还要求它能生成目标序列或信号。正如下面讨论的那样，这也使得这类方法的评估变得极其困难，因为可能正确的答案空间非常大。

我们可以吧模态翻译转变为3个大类：**grammer-based,encoder-decoder和continuous generation**.基于语法的模型通过使用语法来限制目标域从而简化任务，例如，根据“主语-宾语-动词”模板生成有限制的句子。编码器-解码器模型首先将源模态编码为潜在表示，然后由解码器使用该表示生成目标模态。连续生成模型基于源模态输入流连续生成目标模态，最适合于在时间序列之间进行翻译，例如文本到语音。

**grammer-based**依赖于预定义的语法来生成特定模态。这些模型从检测源模态中的高级概念开始，比如图像中的物体或视频中的动作。这些检测到的概念随后与一个基于预定义语法的生成过程相结合，从而产生目标模态。这种方法比较原始，基于语法的方法的一个优点是，由于它们使用预定义的模板和受限的语法，因此更有可能生成在语法上（对于语言而言）或逻辑上正确的目标实例。然而，这限制了它们只能产生公式化的而非创造性的翻译。此外，基于语法的方法依赖于复杂的管道来进行概念检测，每个概念都需要单独的模型和单独的训练数据集。

**Encoder-decoder**模型基于端到端的神经网络训练，是目前最流行的模态翻译方法，首先使用一个encoder把输入转换为vector，然后再使用decoder转换为对应的模态。这种方法效果好，但是有人认为这只是在记忆训练数据并且需要大量的数据来进行训练。

**continuous generation model**一般用于序列转换，比如语音转文本等

#### 如何评估效果

一个模态翻译的问题是，如何评估转换效果的好坏是非常主观的，幸运的是当前有很多自动度量方法

### Alignment

我们将多模态对齐定义为在两种或多种模态实例的子组件之间找到关系和对应。比如给一个图片和一个标题，需要找到标题的队友词语在图中的位置。我们将模态对齐分为implicit和explicit。在explicit对其中，我们明显感兴趣的是对其不同模态的子组件，比如把食谱步骤和视屏对其，而隐式对齐是作为另一个任务的中间步骤，比如基于文本描述的图像检索可以把对齐作为其中的一个步骤

#### Explicit alignment

显示对其的一个重要部分就是相似度度量，大多数的方法依赖于相似度走位基本模块，可以分为无监督和弱监督

#### Implicit alignment

相比于显示对其，隐式对齐通常作为另一个任务的中间的步骤（或则latent），这类模型并不显式地对齐数据，也不依赖于有监督的对齐样本，而是在模型训练过程中学习如何潜在地对齐数据。

解决这一问题的一种非常流行的方法是通过注意力机制，这种方法允许解码器专注于源实例的子组件。这与传统的编码器-解码器模型不同，在传统模型中所有的源子组件会被一起编码。注意力模块会指导解码器更多地关注待翻译源中的目标子组件——比如图像中的区域、句子中的单词。例如，在图像标题生成任务中，并不是使用卷积神经网络（CNN）来编码整幅图像，而是通过注意力机制让解码器（通常是循环神经网络，RNN）在生成每个连续单词时聚焦于图像的特定部分。学习应聚焦图像哪一部分的注意力模块通常是一个浅层的神经网络，并且与目标任务（如翻译）一起端到端地训练。

多模态对齐面临多个困难：

1) 明确标注了对齐关系的数据集很少；
2) 设计模态间的相似性度量很困难；
3) 可能存在多种可能的对齐方式，并非所有一模态中的元素在另一模态中都有对应。

### Fusion

多模态融合是机器学习领域的一个重要主题，包括早期、晚期以及混合融合方法。从技术术语来讲，多模态融合是指整合来自多个模态的信息以预测一个结果度量的概念：通过分类预测一个类别（例如，快乐与悲伤），或通过回归预测一个连续值（例如，情感的积极性）。

多模态融合的好处在于

1. 通过多个模态同时观察一个对象能让结果预测变得更加鲁邦
2. 访问多个模态可能使我们能够捕捉互补信息——这是单个模态单独无法显现的。
3. 即使其中一个模态缺失，多模态系统仍然可以运行，例如当一个人不说话时仅从视觉信号中识别情绪

我们将多模态融合分为两个大类：模型无关和模型有关

**模型无关方法**：早期融合在特征提取后立即整合特征（通常是简单地拼接它们的表示）。早期融合可以被视为多模态研究人员进行多模态表示学习的初步尝试——因为它能够学习利用每种模态的低级特征之间的相关性和交互作用。此外，它只需要训练一个模型，与晚期融合和混合融合相比，使得训练流程更为简便。

而晚期融合则是在每种模态做出决策（例如，分类或回归）之后进行整合。晚期融合使用各模态的决策值并通过融合机制（如平均法 [181]、投票方案 [144]、基于通道噪声的加权 [163] 和信号方差 [53] 或学习得到的模型 [68], [168]）进行融合。它允许为每种模态使用不同的模型，因为不同的预测器可以更好地建模每个单独的模态，从而提供了更大的灵活性。此外，当一种或多种模态缺失时，它更容易进行预测，甚至在没有并行数据的情况下也可以进行训练。但是，晚期融合忽略了模态间的低级交互作用。

最后，混合融合结合了早期融合的输出和各个单一模态预测器的结果。模型无关方法的一个优点是几乎可以使用任何单一模态的分类器或回归器来实现。

**模型有关方法**：基于MKL、图模型和神经网络，目前最常使用的是神经网络，深度神经网络方法在数据融合方面的一大优势是它们可以从大量数据中学习。其次，最近的神经架构允许对多模态表示组件和融合组件进行端到端的训练。最后，它们与非神经网络系统相比表现出良好的性能，并且能够学习其他方法难以处理的复杂决策边界。

神经网络方法的主要缺点是缺乏可解释性。很难确定预测依据什么，哪些模态或特征发挥了重要作用。此外，神经网络需要大量的训练数据才能取得成功。

多模态融合一直是广泛研究的主题，提出了大量的方法来解决这个问题，包括模型无关的方法、图模型、多重核学习以及各种类型的神经网络。每种方法都有自己的优缺点，有些更适合小型数据集，而其他方法则在噪声环境中表现更好。最近，神经网络已成为解决多模态融合非常流行的方式，但图模型和多重核学习仍然被使用，特别是在有限的训练数据或模型可解释性很重要的任务中。

尽管有了这些进步，多模态融合仍面临以下挑战：

1. 信号可能在时间上不同步（可能是密集的连续信号和稀疏事件）；
2. 构建能够利用补充信息而不仅仅是互补信息的模型很困难；
3. 每个模态在不同的时间点可能会表现出不同类型和不同程度的噪声。

### Co-learning

![image-20240823115405854](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20240823115405854.png)

我们分类中的最后一个多模态挑战是协同学习——即通过利用另一种（资源丰富）模态的知识来辅助建模一种（资源匮乏）的模态。当其中一种模态资源有限——例如缺乏标注数据、输入噪声大或标签不可靠时，这个问题尤为突出。我们将这个挑战称为协同学习，因为通常帮助模态只在模型训练期间使用，在测试阶段则不使用。

根据它们的训练资源，我们可以将协同学习方法分为三种类型：并行数据、非并行数据和混合数据。

- **并行数据**方法要求训练数据集中的不同模态观察直接相关联。换句话说，当多模态观察来自于同一实例时，如在一个视听语音数据集中，视频样本和语音样本都是来自同一个说话者。
- **非并行数据**方法并不需要不同模态之间的直接关联。这些方法通常通过类别重叠来实现协同学习。例如，在零样本学习中，传统的视觉对象识别数据集可以与来自维基百科的第二个仅文本的数据集结合起来，以提高视觉对象识别的泛化能力。
- **混合数据**设置下，模态之间通过共享模态或数据集相连。一个模态可以通过另一个模态与第三个共享模态联系起来。

**parallel data**：在并行数据中，数据来源于同一组实体。协同学习和迁移学习是并行数据的两大方法，协同训练是在一个多模态问题中当我们只有少量标注样本时创造更多标注训练样本的过程[21]。基本算法是在每个模态中建立弱分类器，用未标注数据的标签来相互引导。虽然协同训练是一种生成更多标注数据的强大方法，但它也可能导致偏差的训练样本，从而产生过拟合。迁移学习是另一种利用并行数据进行协同学习的方式。将一个模态的表示信息转移到另一个模态上。这不仅导致了多模态表示，而且也产生了更好的单模态表示，在测试时只使用一个模态[

**非并行数据**：依赖非并行数据的方法并不需要模态之间有共享实例，只需要有共享的类别或概念即可。非并行协同学习方法在学习表示时可以提供帮助，允许更好地理解语义概念，甚至可以执行未知物体的识别。