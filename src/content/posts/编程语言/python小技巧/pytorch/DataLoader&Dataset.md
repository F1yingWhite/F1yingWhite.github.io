---
title: DataLoader&Dataset
description: ""
image: ""
published: 2025-03-04
tags:
  - python
  - pytorch
category: " pytorch"
draft: false
---

# Dataset

Dataset 是一个抽象类，所有表示从键到数据样本映射的数据集都应继承此类。所有子类都应重写 :meth:`__getitem__` 方法，以支持根据给定键获取数据样本。子类还可以选择性地重写 :meth:`__len__` 方法，该方法应返回数据集的大小，子类还可以选择性地实现 :meth:`__getitems__` 方法，以加速批量样本的加载。该方法接受一个批次样本的索引列表，并返回样本列表。

# DataLoader

在 Dataloader 中结合了一个 dataset 和一个采样器，为给定的数据集提供可迭代对象。

Args:

	dataset:需要加载的数据集
	batch_size int：默认为 1
	shuffle bool:是否在每个 epoch 中打乱数据
	sampler ：自定义采样策略
	num_workers：用于数据加载的子进程，默认为 0,0 表示会在主线成中加载
	pin_memory： 如果是 True，那么每次加载将到同样的内存中
	collate_fn:将一个列表合并为一个小 batch 的 tenseor
	drop_last:如果 batchsize 不是整数，那么扔掉最后一个
	persistent_workers:在结束一个 epoch 后是否保持这些进程存活。
	timeout:是用来设置数据读取的超时时间的，但超过这个时间还没读取到数据的话就会报错。

>[!TIP]
>使用 `collate_fn` 的主要原因是为了处理那些不能简单地通过堆叠来组合的数据结构。例如，如果你的数据样本是不同长度的序列，那么简单地堆叠它们会导致错误。在这种情况下，你可能需要使用 `padding` 来确保所有序列具有相同的长度，然后使用 `pack_padded_sequence` 和 `pad_packed_sequence` 来处理这些序列。
