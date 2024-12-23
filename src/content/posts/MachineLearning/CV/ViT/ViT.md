---
title: ViT
published: 2024-07-22
description: ''
image: ''
tags: [机器学习,CV]
category: '论文阅读'
draft: false 
last modified: 2024-10-21
---

::github{repo="google-research/vision_transformer"}

# 简介

ViT 是 Google 提出的把 Transformer 应用在图像分类的模型.因为模型简单效果好可扩展性强,所以成为了 Transformer 在 CV 领域的里程碑

ViT 论文最核心的结论就是当有足够多的数据进行预训练,ViT 就会超过 CNN,突破 Transformer 没有归纳偏置的限制,可以在下游任务重获得较好的迁移学习效果.

而数据量少的时候,ViT 比同等大小的 ResNet 小一点,因为 Transformer 缺少归纳偏置 (一种先验知识,是 CNN 预先设计好的假设).CNN 有两种归纳偏置: 其一是局部性,即在图片上相邻的区域具有相似特征,一种是平移不变形,$f(g(x))=g(f(x))$,其中 f 是平移,g 是卷积,当 CNN 有了上面的归纳偏置,就有了很多的先验知识,需要相对少的数据就可以学习一个好模型.

# ViT 结构

![ViT架构](https://p.ipic.vip/x1mzr1.png)

ViT 首先把图片划分为多个 Patch(16*16),然后把每个 Patch 投影为固定长度的向量送入 Transformer 中.后续 encoder 的操作就和原始 Transformer 一模一样.但是因为对图片分类，因此在输入序列中加入一个特殊的 token，该 token 对应的输出即为最后的类别预测.实际代码中的切分和映射是是直接使用一个卷积来实现的 ``self.proj=nn.Conv2d(in_c,embed_dim,kernel_size=patch_size,stride=patch_size)``.可以直接完成切分 + 映射,这里可以填 3-768-16-16,然后 ``x=self.proj.flatten(2).transpose(1,2)``

一个 ViT block 如下

1. patch embedding: 比如输入图片大小为 224x224,把图片分为固定大小的 patch,patch 大小为 16x16,每个图片就变成了 196 个 patch,即输入序列长度为 196,每个 patch 的维度是 16x16x3=768 个,线性投影层的维度是 768x768,所以每个 patch 出来还是 768,即具有 196 个 token,每个 token 维度 768.最终维度是 197x768,到目前为止,已经把视觉问题变成了 seq2seq 问题.
2. positional encoding:vit 需要加入位置编码,可以理解为一张表,一共 N 行,N 的大小和输入序列长度相同,每一行代表一个向量.每一行代表一个向量，向量的维度和输入序列 embedding 的维度相同（768）。注意位置编码的操作是 sum，而不是 concat。加入位置编码信息之后，维度依然是**197x768**
3. 多头注意力机制:LN 输出维度依然是 197x768。多头自注意力时，先将输入映射到 q，k，v，如果只有一个头，qkv 的维度都是 197x768，如果有 12 个头（768/12=64），则 qkv 的维度是 197x64，一共有 12 组 qkv，最后再将 12 组 qkv 的输出拼接起来，输出维度是 197x768，然后在过一层 LN，维度依然是**197x768**
4. MLP：将维度放大再缩小回去，197x768 放大为 197x3072，再缩小变为**197x768**

这样就是一个标准 block 了,输入和输出大小都一样,因此可以堆叠多个 block.对于分类,是这样的:

```python
import torch
import torch.nn as nn

class ViTClassifier(nn.Module):
    def __init__(self, num_classes, hidden_dim):
        super(ViTClassifier, self).__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # 假设 x 的形状是 [batch_size, num_patches + 1, hidden_dim]
        cls_token_output = x[:, 0]  # 取出 [CLS] token 的输出 只要每个的第一个
        logits = self.classifier(cls_token_output)  # 线性层映射到类别数
        return logits

# 示例用法
num_classes = 10
hidden_dim = 768
model = ViTClassifier(num_classes, hidden_dim)

# 假设输入张量的形状是 [batch_size, num_patches + 1, hidden_dim]
input_tensor = torch.randn(32, 197, 768)  # batch_size=32
output = model(input_tensor)
print(output.shape)  # 应该是 [32, num_classes]
```

实际开发中的做法是：基于大数据集上训练，得到一个预训练权重，然后再在小数据集上 Fine-Tune。

完整代码如下:

```python
class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_c=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 representation_size=None,
                 distilled=False,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 embed_layer=PatchEmbed,
                 norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        # 每个patch的图像维度 = embed_dim
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # token的个数为1
        self.num_tokens = 2 if distilled else 1
        # 设置激活函数和norm函数
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        # 对应的将图片打成patch的操作
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # 设置分类的cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # distilled 是Deit中的 这里为None
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # pos_embedding 为一个可以学习的参数
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # 使用nn.Sequential进行构建，ViT中深度为12
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
```
