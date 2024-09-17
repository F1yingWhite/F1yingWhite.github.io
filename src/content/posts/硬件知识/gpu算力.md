---
title: GPU算力 
published: 2024-09-15
description: ''
image: ''
tags: [gpu算力]
category: '硬件'
draft: false 
---

## GPU算力标准
大模型使用4090不能训练，但推理使用4090甚至性价比还高一点，如果4090极致优化，甚至性价比达到H100的两倍

事实上，H100和4090的差距就在通信和内存上，算力差距不大

| ()          | H100       | 4090       |
| ----------- | ---------- | ---------- |
| Tensor FP16 | 989 Tflops | 330 Tflops |
| Tensor FP32 | 495 Tflops | 83 Tflops  |
| 容量        | 80G        | 24G        |
| 内存带宽    | 3.35TB/s   | 1TB/s      |
| 通信带宽    | 900GB/s    | 64GB/s     |
| 通信延时    | 1us        | 10us       |

NVIDIA 的算力表里面油水很多，比如 H100 TF16 算力写的是 1979 Tflops，但那是加了 sparsity（稀疏）的，稠密的算力只有一半；4090 官方宣传 Tensor Core 算力高达 1321 Tflops，但那是 int8 的，FP16 直只有 330 Tflops

回到大模型训练所需的总算力，其实很简单，**6 \* 模型的参数量 \* 训练数据的 token 数就是所有训练数据过一遍所需的算力。这里的 6 就是每个 token 在模型正向传播和反向传播的时候所需的乘法、加法计算次数。**

一堆矩阵相乘，简单来想就是左边若干个神经元，右边若干个神经元，组成一个[完全二分图](https://zhida.zhihu.com/search?q=完全二分图&zhida_source=entity&is_preview=1)。选出其中任意一个左边的神经元 l 和右边的神经元 r，正向传播的时候：

1. l 把它的输出乘上 l 和 r 之间的权重 w，发给 r；
2. r 不可能只连一个神经元吧，总要把多个 l 的加到一起，这就是 reduce，需要一次加法。

反向传播的时候：

1. r 把它收到的梯度乘上 l 和 r 之间的权重 w，发给 l；
2. l 也不可能只连一个 r，需要把梯度 reduce 一下，做个加法；
3. 别忘了权重 w 需要更新，那就要计算 w 的梯度，把 r 收到的梯度乘上 l 正向传播的输出（activation）；
4. 一个 batch 一般有多个 sample，权重 w 的更新需要把这些 sample 的梯度加到一起。

一共 3 次乘法，3 次加法，不管 Transformer 多复杂，矩阵计算就是这么简单，其他的向量计算、softmax 之类的都不是占算力的主要因素，估算的时候可以忽略。

## 训练大模型的算力标准

有了模型训练需要的总算力，除以每个GPU的理论算例，在处理GPU有效算力的利用比率，就得到了GPU-hours。比如LLaMA2 70B的GPU-hour就是1.7M GPU-hours，用一个GPU要算200年。

那么按照2048张4090算，这2048张GPU怎么通信就成了大问题。一般有 tensor parallelism、pipeline parallelism、data parallelism 几种并行方式，分别在模型的层内、模型的层间、训练数据三个维度上对 GPU 进行划分。三个并行度乘起来，就是这个训练任务总的 GPU 数量。

### Data parallelism(数据并行)
就是每个GPU分别计算不同的输入数据，并计算各自的梯度，最后汇总，取个平均，广播给每个GPU分别更新（torch中的dataparallel）。当然大模型中肯定不行，因为一个GPU放不下整个大模型，训练需要的内存包括模型参数、反向传播的梯度、优化器所用的内存、正向传播的中间状态（activation）。

优化器所用的内存其实也很简单，如果用最经典的 Adam 优化器，它需要用 32 位浮点来计算，否则单纯使用 16 位浮点来计算的误差太大，模型容易不收敛。

### Pipeline parallelism（流水线并行）
这种方式叫做模型并行，模型不是很多层吗？那就分成号机组，穿成一条链,但是呢这样会导致只有一个GPU在干活，当然可以用mini-batch来加速流水。

其次，pipeline 的相邻流水级（pipeline stage）之间是要通信的，级数越多，通信的总数据量和总时延就越高。

最后，要让这样的 pipeline 流起来，batch size 需要等于 Transformer 里面的层数，一般是几十，再乘以 data parallelism 的并行数，batch size 会很大，影响模型收敛的速度或模型收敛后的精度。

### Tensor parallelism（张量并行）
在模型的层内划分，也就是把一层内的 attention 计算和 Feed Forward Network 划分到多个 GPU 上处理。

Tensor、Pipeline、Data Parallelism 就像是这样的不可能三角，相互牵制，只要集群规模够大，模型结构仍然是 Transformer，就很难逃出内存容量和网络带宽的魔爪。