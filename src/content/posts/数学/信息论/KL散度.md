---
title: KL散度
published: 2024-10-08
description: ''
image: ''
tags: [数学]
category: '数学'
draft: false
---
## KL散度简介
KL散度的概念来自于概率论和信息论中，KL散度又叫相对熵，互熵等。用来衡量两个概率分布的区别

KL散度的概念是建立在熵（Entropy）的基础上的，以离散随机变量为例，给出熵的定义:

如果一个随机变量x的取值可能是$x={x_1...x_n}$,对应的概率为$p_i=p(X=x_i)$,那么随机变量x的熵定义为：
$$
H(X) = -\sum^n_{i=1}{p(x_i)\log{p(x_i)}}
$$
> 规定如果p(x_i)=0,那么$p(x_i)\log{p(x_i)}=0$

如果有两个随机变量P，Q,且概率为p(x),q(x),则p相当q的相对熵为：
$$
D_{KL}(p||q)=\sum^n_{i=1}{p(x)log\frac{p(x)}{q(x)}}
$$
之所以叫相对熵，是因为可以通过两随机变量的交叉熵和信息熵推导到

如果从定义出发，交叉熵就是$H(p,q)-H(p)=\sum^n_{i=1}{p(x)log\frac{p(x)}{q(x)}}$
