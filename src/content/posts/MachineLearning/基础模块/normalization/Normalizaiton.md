---
title: Normalizaiton
description: ""
image: ""
tags:
  - 机器学习
  - 论文阅读
category: 论文阅读
draft: false
published: 2024-10-23
last modified: 2024-10-23
---

# AdaLN

 自适应 [层归一化](https://so.csdn.net/so/search?q=%E5%B1%82%E5%BD%92%E4%B8%80%E5%8C%96&spm=1001.2101.3001.7020)（Adaptive Layer Normalization，adaLN）是一种归一化技术，用于 [深度学习模型](https://so.csdn.net/so/search?q=%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A8%A1%E5%9E%8B&spm=1001.2101.3001.7020) 中特征的标准化。它结合了传统的层归一化（Layer Normalization, LN）和自适应学习的特性，以提高模型在不同任务和数据集上的表现。

$$
LN(x) =\gamma \left( x-\frac{\mu(x)}{\sigma(x)} \right)+\beta
$$
