---
title: KV-Cache
description: ""
image: ""
published: 2025-02-02
tags:
  - LLM
category: 论文阅读
draft: false
---


https://zhuanlan.zhihu.com/p/630832593

1. KV Cache 节省了 Self-Attention 层中哪部分的计算？
2. KV Cache 对 MLP 层的计算量有影响吗？
3. KV Cache 对 block 间的数据传输量有影响吗？

# 什么是 KV Cache

大模型优化中一个常用的技术就是 kv cache,可以在不影响任何计算精度的前提下通过空间换时间,提升推理性能.

# 背景

生成式模型的推理过程很有特点,我们给定一个输入文本,模型会输出一个回答 (假设长度是 n),那么实际上模型进行了 n 次推理,也就是 llm 每次推理给出一个 token,输出 token 会与输入 token 拼在一起,作为下一次推理的输入,这样推理直到遇到终止符.

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("/WORK/Test/gpt", torchscript=True).eval()

# tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("/WORK/Test/gpt")
in_text = "Lionel Messi is a"
in_tokens = torch.tensor(tokenizer.encode(in_text))

# inference
token_eos = torch.tensor([198]) # line break symbol
out_token = None
i = 0
with torch.no_grad():
    while out_token != token_eos:
        logits, _ = model(in_tokens)
        out_token = torch.argmax(logits[-1, :], dim=0, keepdim=True)
        in_tokens = torch.cat((in_tokens, out_token), 0)
        text = tokenizer.decode(in_tokens)
        print(f'step {i} input: {text}', flush=True)
        i += 1

out_text = tokenizer.decode(in_tokens)
print(f' Input: {in_text}')
print(f'Output: {out_text}')
```

输出为:

```
step 0 input: Lionel Messi is a player
step 1 input: Lionel Messi is a player who
step 2 input: Lionel Messi is a player who has
step 3 input: Lionel Messi is a player who has been
step 4 input: Lionel Messi is a player who has been a
step 5 input: Lionel Messi is a player who has been a key
step 6 input: Lionel Messi is a player who has been a key part
step 7 input: Lionel Messi is a player who has been a key part of
step 8 input: Lionel Messi is a player who has been a key part of the
step 9 input: Lionel Messi is a player who has been a key part of the team
step 10 input: Lionel Messi is a player who has been a key part of the team's
step 11 input: Lionel Messi is a player who has been a key part of the team's success
step 12 input: Lionel Messi is a player who has been a key part of the team's success.
step 13 input: Lionel Messi is a player who has been a key part of the team's success.

 Input: Lionel Messi is a
Output: Lionel Messi is a player who has been a key part of the team's success.
```

注意这里的输出,每次推理后 token 都变长了,有没有办法让推理过程中的 flops 基本恒定不变呢?

## 原理

在上述的推理中,每个 step 中输入一个 token 序列,经过 embedding 层把 token 变为三维张量\[b,s,h],经过一通计算，最后经 logits 层将计算结果映射至词表空间，输出张量维度为\[b, s, vocab_size]。

当前轮输出 token 与输入 tokens 拼接，并作为下一轮的输入 tokens，反复多次。可以看出第 i+1 轮输入数据只比第 i 轮输入数据新增了一个 token，其他全部相同！因此第 i+1 轮推理时必然包含了第 i 轮的部分计算。KV Cache 的出发点就在这里，缓存当前轮可重复利用的计算结果，下一轮计算时直接读取缓存结果，就是这么简单，不存在什么 Cache miss 问题。
