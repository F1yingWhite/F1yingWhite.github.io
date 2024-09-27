---
title: AttentionIsAllYouNeed
published: 2024-07-22
description: ''
image: ''
tags: [机器学习,NLP]
category: '论文阅读'
draft: false 
---

# Attention
## Attention本质是什么

Attention机制就是从关注全部到关注重点.当我们看一张图片的时候,并没有看到全部内容,而是将注意力集中在了图片的焦点上.也就是说,我们的视觉系统是一种Attention机制,将有限的注意力集中在重点信息上,从而节省资源,快速获得最有效的信息.

为什么有Self-Attention layer?

> 因为RNN不能并行化,做seq2seq的时候太慢,因此想要有一个可以做到seq2seq的并行化架构

Attention机制具有以下三种有点:

- 参数小:跟CNN/RNN比,复杂度小

- 速度快:解决了RNN不能并行计算的问题,Attention每一步不依赖上一步的结果

- 效果好:在Attention之前,一直有个问题:长距离的信息会被弱化,就好像记忆差的人记不住过去.而Attention能够挑选重点,即使文本长,也能从中抓住重点而不丢失重要的信息.

  ![image-20240722143859518](https://p.ipic.vip/p3syrb.png)

  图书管（source）里有很多书（value），为了方便查找，我们给书做了编号（key）。当我们想要了解漫威（query）的时候，我们就可以看看那些动漫、电影、甚至二战（美国队长）相关的书籍。为了提高效率，并不是所有的书都会仔细看，针对漫威来说，动漫，电影相关的会看的仔细一些（权重高），但是二战的就只需要简单扫一下即可（权重低）。当我们全部看完后就对漫威有一个全面的了解了。

## Self-Attention

![image-20240722145101718](https://p.ipic.vip/3h97ug.png)

图中的$a=wx$,是x的Embedding,qkb是都等于$w_ix$,这里的w不共享,这里除以根号d是为了减少数值.然后使用Soft-max计算一下得到a_head

![image-20240722145324992](https://p.ipic.vip/zyta17.png)

![image-20240722145404337](https://p.ipic.vip/tw8rr2.png)

这样就可以得到全局的注意力了,这样我们就可以平行计算出所有的$b_i$了.这样我们就可以用矩阵来计算了.

### 矩阵实现

这里没有下标就表示是拼起来的一整个.

$Q/K/V = W_{q/k/v}A$这里的I表示a1~a4矩阵,然后计算aii,aii的计算由:$K^T@Q$得到,然后b的矩阵可以由V@A就可以得到了.

## Multi-head self-attention

![image-20240722150739213](https://p.ipic.vip/h616i9.png)

把那个头给分裂开,然后可以计算得到多个bi,bi可以contact在一起然后乘一个权重就输出为和原来维度一样的向量了.

### Positional Encoding

![image-20240722150931470](https://p.ipic.vip/osdala.png)

可以加入位置参数,这个参数可以自己学也可以直接设置.

## Transformer

<img src="https://p.ipic.vip/f3j6fl.png" alt="image-20240722151714617" style="zoom:50%;" />

在Transformer中大概就是这样.BatchNorm是对每个batch的相同dim做norm,而layer是对每个batch做nrom.Transformer的结构像RNN,所以这里用layernorm.
```python
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn, tensor


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim=512, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout),
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, x):  # x的维度为batch,num,dim
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # (b, n, dim*3) ---> 3 * (b, n, dim) 这里生成kqv
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)  # q, k, v   (b, h, n, dim_head(64))

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = self.attend(dots)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)), PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


```