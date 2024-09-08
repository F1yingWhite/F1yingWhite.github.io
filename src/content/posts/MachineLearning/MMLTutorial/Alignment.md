---
title: 多模态深度学习入门之Alignment
published: 2024-08-23
description: '多模态深度学习入门之Alignment'
image: ''
tags: [多模态，导论]
category: '导论'
draft: true
---

## Challenge 2:Alignment

定义：根据数据结构构建，识别和建模多个模式的所有元素之间的跨模式连接

![image-20240823160939571](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20240823160939571.png)

### Sub-Challenge 1：Connections

定义：在多个模态元素之间识别联系

两个模态之间的信息可能具有重叠，这部分就是他们的公用信息

### Sub-Challenge 2: Aligned Representations

定义：对所有跨模态连接进行建模互动以学习更好的表征

1. Segmented elements 
2. List of elements (with position encodings) 
3. Early fusion
4. All elements are connected
5. Same modeling method for all  interactions (similarity kernels)

一般可以使用attention model或者mutilmodal Transformer