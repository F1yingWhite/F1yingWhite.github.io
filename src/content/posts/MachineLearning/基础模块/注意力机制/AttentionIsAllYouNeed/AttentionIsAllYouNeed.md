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

## Attention 本质是什么

Attention 机制就是从关注全部到关注重点.当我们看一张图片的时候,并没有看到全部内容,而是将注意力集中在了图片的焦点上.也就是说,我们的视觉系统是一种 Attention 机制,将有限的注意力集中在重点信息上,从而节省资源,快速获得最有效的信息.

为什么有 Self-Attention layer?

> 因为 RNN 不能并行化,做 seq2seq 的时候太慢,因此想要有一个可以做到 seq2seq 的并行化架构

Attention 机制具有以下三种有点:

- 参数小: 跟 CNN/RNN 比,复杂度小

- 速度快: 解决了 RNN 不能并行计算的问题,Attention 每一步不依赖上一步的结果

- 效果好: 在 Attention 之前,一直有个问题: 长距离的信息会被弱化,就好像记忆差的人记不住过去.而 Attention 能够挑选重点,即使文本长,也能从中抓住重点而不丢失重要的信息.

  ![image-20240722143859518](https://p.ipic.vip/p3syrb.png)

  图书管（source）里有很多书（value），为了方便查找，我们给书做了编号（key）。当我们想要了解漫威（query）的时候，我们就可以看看那些动漫、电影、甚至二战（美国队长）相关的书籍。为了提高效率，并不是所有的书都会仔细看，针对漫威来说，动漫，电影相关的会看的仔细一些（权重高），但是二战的就只需要简单扫一下即可（权重低）。当我们全部看完后就对漫威有一个全面的了解了。

## Self-Attention

![image-20240722145101718](https://p.ipic.vip/3h97ug.png)

图中的 $a=wx$,是 x 的 Embedding,qkb 是都等于 $w_ix$,这里的 w 不共享,这里除以根号 d 是为了减少数值.然后使用 Soft-max 计算一下得到 a_head

![image-20240722145324992](https://p.ipic.vip/zyta17.png)

![image-20240722145404337](https://p.ipic.vip/tw8rr2.png)

这样就可以得到全局的注意力了,这样我们就可以平行计算出所有的 $b_i$ 了.这样我们就可以用矩阵来计算了.

### 矩阵实现

这里没有下标就表示是拼起来的一整个.

$Q/K/V = W_{q/k/v}A$ 这里的 I 表示 a1~a4 矩阵,然后计算 aii,aii 的计算由:$K^T@Q$ 得到,然后 b 的矩阵可以由 V@A 就可以得到了.

## Multi-head Self-attention

![image-20240722150739213](https://p.ipic.vip/h616i9.png)

把那个头给分裂开,然后可以计算得到多个 bi,bi 可以 contact 在一起然后乘一个权重就输出为和原来维度一样的向量了.

### Positional Encoding

![image-20240722150931470](https://p.ipic.vip/osdala.png)

可以加入位置参数,这个参数可以自己学也可以直接设置.

## Transformer

<img src="https://p.ipic.vip/f3j6fl.png" alt="image-20240722151714617" style="zoom:50%;" />

在 Transformer 中大概就是这样.BatchNorm 是对每个 batch 的相同 dim 做 norm,而 layer 是对每个 batch 做 nrom.Transformer 的结构像 RNN,所以这里用 layernorm.

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

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert  image_height % patch_height ==0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))					# nn.Parameter()定义可学习参数
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)        # b c (h p1) (w p2) -> b (h w) (p1 p2 c) -> b (h w) dim
        b, n, _ = x.shape           # b表示batchSize, n表示每个块的空间分辨率, _表示一个块内有多少个值

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)  
        x = torch.cat((cls_tokens, x), dim=1)               # 将cls_token拼接到patch token中去       (b, 65, dim)
        x += self.pos_embedding[:, :(n+1)]                  # 加位置嵌入（直接加）      (b, 65, dim)
        x = self.dropout(x)

        x = self.transformer(x)                                                 # (b, 65, dim)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]                   # (b, dim)

        x = self.to_latent(x)                                                   # Identity (b, dim)
        print(x.shape)

        return self.mlp_head(x)                                                 #  (b, num_classes)


```

# 常看常新

https://www.cnblogs.com/GreenOrange/p/18279948

现在我们来看后面的 docoder 以及从 nlp 的角度进行讲解.

![image.png](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/20241025134452.png)

首先看输入输出,有一个 input,一个 Output 和一个 Output Probabilities,比如翻译,input 就是英文,output 是汉语,概率就是下一个词的概率.词是一个一个出来的而不是一起出来的.第 n 个词的预测需要前 n-1 个词.为了不让模型抄答案,我们让 masked 模型预测第 n 个词的时候,把后面的词语盖住.对于输出的第一个词,我们需要加一个\<start>让所有的句子都是 start 开头.

因为现代的词嵌入技术可以反应出词的相似程度,我们把**词向量和其他词求内积,就可以得到每个词向量和其他词向量的相似程度**.

KQV 就是通过 embedding 之后的词向量,如果按照之前的方法直接 $X \cdot X^T$ 就限制了数据分布,而且如果嵌入训练的不好会影响模型性能,因此我们又加入了三个独立的线性层用来计算 kqv.

现在我们使用 $Q \cdot K^T$ 就可以得到相似度矩阵,为了保证梯度稳定,我们进行一个归一化,均值为 0,方差为 d,也就是

$$
\frac{Q \cdot K^T}{\sqrt{ d }}
$$

然后加个 softmax 就是权重矩阵了.然后乘 V 得到加权后的注意力矩阵 (对每一行进行 softmax,表示跟每一个的相似度,向量中每一行是一个单词)![image.png](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/20241025140252.png)

## 多头注意力机制

在向量 X（经过 Encoding 之后的词向量）进入自注意力机制模块前将他”断开“，不同的维度进入不同的自注意力机制模块进行相同的的运算，例如”你“这个词，假设它的词嵌入向量 Y 是 512 维的 \[a0, a1, a2, ···， a510, a511]，我们只使用两个头，也就是 h=2，那么就将 Y 截断，\[a0, a1, ···, a254, a255] 进入第一个自注意力机制模块进行计算，\[a256, a257, ···, a510, a511] 进入第二个模块经行同样的计算，在各自计算完成后拼接（concat）起来，再通过一个全连接层增加模型的复杂度。事实上，这样做是很有必要的，这样可以训练多个注意力矩阵提取不同维度的信息，增加了模型的复杂度，同时通过拆分维度把计算量分成一小块一小块的了，提高了并行性。

至此，我们走完了 Multi-Head Attention 这个模块。

## Masked Mutil-Head Attetnion

1. 我们输入 A read apple
2. output(shift right) 自动添加一个\<start>
3. 第一轮得到单词 一个 的概率最大
4. output 输入 `<start>一个`
5. ...
6. 最后输出一个 `<end>` 结束

可以看到，结果是一个一个输出的，第 n 个词的输出需要依赖前 n-1 个词的输入，训练过程也是一样

1. 我们在 `Inputs` 端输入 "A red apple"。
2. `Outputs(shift right)` 端会自动输入一个 `<start>` 作为起始标记。
3. 解码器依据输入在经过一系列的变化，但是实际情况下，如果训练的不够，`Output Probabilities` 输出结果很可能不是 " 一个 "，而是其他的，我们就用交叉熵损失函数来计算损失值（量化它的输出与标准答案“一个”的差异），根据这个来调整网络的参数。
4. `Outputs(shift right)` 端会自动输入 `<start> 一个`，**注意，不是 `<start>` 加上 `Output Probabilities` 输出的不标准的答案，而是标准答案**，这个方法叫**Teacher forcing**，试想如果第一个输出就不对，用错的结果继续生成的也只能是错误的结果，最后随着训练的继续只能越错越多，十分不利于模型的收敛，因此我们的输入端是要求输入标准答案的。也正是因为有了这种机制，我们让模型去预测 `一个` 的同时，也能让模型去预测 `红色的`，因为训练过程中的输入不依赖上一步的输出，这也就为并行计算提供了可能。
5. 一直重复 3，4 步骤直至句子结束
![image.png](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/20241027152035.png)
也就是说,在训练的时候 (比如训练翻译),我们输入了 a red apple,在 outputs 端输入一个红色的苹果,然后他就就会生成上面的内容,为了防止偷看,我们就把后面的盖掉,然后一行一行的看,相当于得到了 4 个损失.
