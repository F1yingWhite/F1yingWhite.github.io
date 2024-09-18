---
title: 混合专家模型
published: 2024-09-17
description: ''
image: ''
tags: [机器学习,多模态,论文阅读]
category: '多模态'
draft: false 
---
[混合专家模型详解](https://huggingface.co/blog/zh/moe)
## 总结
混合专家模型 (MoEs):

与稠密模型相比， 预训练速度更快
与具有相同参数数量的模型相比，具有更快的 推理速度
需要 大量显存，因为所有专家系统都需要加载到内存中
在 微调方面存在诸多挑战，但 近期的研究 表明，对混合专家模型进行 指令调优具有很大的潜力。