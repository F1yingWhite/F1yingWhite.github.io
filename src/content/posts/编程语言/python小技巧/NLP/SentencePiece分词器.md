---
title: SentencePiece
description: ""
image: ""
tags:
  - python
  - language
category: language
draft: false
published: 2024-10-24
---

SentencePiece 实现了**subword**单元（例如，字节对编码 (BPE)）和 unigram 语言模型），并可以直接从原始句子训练字词模型 (subword model)。 这使得我们可以制作一个不依赖于特定语言的预处理和后处理的纯粹的端到端系统。

# SentencePiece 特性

## Token 数量是预先确定的

神经网络翻译模型通常用固定的词汇表进行操作,与大多数假设无线词汇量的不一样,sentencepiece 预先确定词汇表的大小,比如 8k,16k 或者 32k

## 从原始句子开始训练

SentencePiece 的实现速度足够快，可以从原始句子训练模型。 这对于训练中文和日文的 tokenizer 和 detokenizer 很有用，因为在这些词之间不存在明确的空格。

## 空格被视为基本符号

自然语言处理的第一步是文本 tokenization。SentencePiece 把输入文本是做一系列的 Unicode 字符,使用元符号 U+2581 转义空格,这样就可以不产生歧义的转义句子,否则 Tokenize(“World.”) == Tokenize(“World .”)
