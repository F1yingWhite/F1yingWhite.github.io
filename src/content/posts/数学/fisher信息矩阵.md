---
title: fisher信息矩阵
description: ""
image: ""
published: 2025-08-18
tags:
  - 数学
category: 数学
draft: false
---

https://blog.csdn.net/wxc971231/article/details/135591016

# 前言

机器学习本身常常被设计为参数化概率模型 $p(x|\theta)$,同过优化损失函数 $\mathcal{L}$ 来最大化参数表示的似然函数.由于似然函数本身就是一个概率分布,当参数 $\theta$ 更新为 $\theta+d$ 的时候,可以用 KL 散度 $KL[p(x|\theta)||p(x|\theta+d)]$ 来度量模型所描述的映射的关系变化程度.这里有两个空间

1. 分布空间: 模型似然 $p(x|\theta)$ 取值的空间,可以用 KL 散度衡量差异
2. 参数空间: 参数向量 $\theta$ 的取值空间.且该空间相对光滑,局部上类似欧几里得空间.
梯度下降的公式为:$\theta_{k+1} = \theta_{k}-\alpha \nabla_{\theta}\mathcal{L}(\theta+d)$,注意到参数更新的方向是梯度方向,这是参数空间中 $\theta_{k}$ 局部最陡峭的下降方向.

# 海森矩阵

海森矩阵就是由某个多元函数的二阶偏导构成的方阵,描述了函数的局部曲率,

$$
\mathbf{H}_f(\boldsymbol{x}) =
\begin{bmatrix}
\dfrac{\partial^2 f}{\partial x_1^2} & \dfrac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_1 \partial x_n} \\
\dfrac{\partial^2 f}{\partial x_2 \partial x_1} & \dfrac{\partial^2 f}{\partial x_2^2} & \cdots & \dfrac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\dfrac{\partial^2 f}{\partial x_n \partial x_1} & \dfrac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}_{n \times n}
$$

这里要求 f 在展开位置附近二阶连续可导,由于二阶偏导可以交换,H 是对称矩阵.

# Fisher 信息矩阵

考虑极大似然估计的过程，目标分布由 $\theta_{m \times 1} = [\theta_1, \theta_2, \dots, \theta_m]^T$ 参数化，首先从目标分布中收集样本 $x_i \sim p(x|\theta)$ 构成数据集 $\mathcal{X} = \{x_1, x_2, \dots, x_n\}$，记为 $X \sim p(X|\theta)$，然后写出对数似然函数：

$$
L(X|\theta) = \sum_{i=1}^{n} \log p(x_i|\theta)
$$

通过最大化对数似然函数来得到估计参数，即：

$$
\hat{\theta} = \arg\max_{\theta} \sum_{i=1}^{n} \log p(x_i|\theta)
$$

在取得极值处，应该有梯度为 0，即：

$$
\nabla_\theta L(X|\theta) = \sum_{i=1}^{n} \nabla_\theta \log p(x_i|\theta)
$$

$$
\nabla_\theta L(X|\theta)\big|_{\theta=\hat{\theta}} = \mathbf{0}
$$

基于以上观察，我们将关于任意样本 $x$ 的对数似然函数的梯度定义为 **得分函数（score function）**，利用它评估 $\hat{\theta}$ 估计的良好程度：

$$
s(x|\theta)_{m\times1} = \nabla_\theta L(x|\theta) = \nabla_\theta \log p(x|\theta)
$$

注意得分函数是标量对向量 $\theta$ 求导，因此 $s(x|\theta)$ 是和 $\theta$ 相同尺寸的向量。得分函数的重要性质是其期望为 0，即：

$$
\mathbb{E}_{x \sim p(x|\theta)}[s(x|\theta)] = \mathbf{0}
$$

证明如下：

$$
\begin{aligned}
\mathbb{E}_{p(x|\theta)}[s(x|\theta)] 
&= \mathbb{E}_{p(x|\theta)}[\nabla \log p(x|\theta)] \\
&= \int \nabla \log p(x|\theta) \, p(x|\theta) \, dx \\
&= \int \frac{\nabla p(x|\theta)}{p(x|\theta)} \, p(x|\theta) \, dx \\
&= \int \nabla p(x|\theta) \, dx \\
&= \nabla \int p(x|\theta) \, dx \\
&= \nabla 1 \\
&= \mathbf{0}_{m \times 1}
\end{aligned}$$
因此可以用大量样本的socre来评估估计值$\hat{\theta}$的质量,也就是期望越接近0,估计越精确.

进一步地，参数估计值 $\hat{\theta}$ 的置信度可以由大量样本的 $s(x|\theta)$ 协方差来描述，这是一个围绕期望值的不确定度的度量。将以上期望为 0 的结果代入协方差计算公式，得到

$$

\mathbb{E}_{p(x|\theta)}\left[(s(x|\theta) - \mathbf{0}) \cdot (s(x|\theta) - \mathbf{0})^T\right]

$$

由于得分函数 $s(x|\theta)$ 是尺寸为 $m \times 1$ 的向量，以上协方差是尺寸 $m \times m$ 的协方差矩阵，**这就是 费舍尔信息矩阵（Fisher information matrix）的定义**，它描述了极大似然估计参数值的置信度（不确定度）的信息，整理如下：

$$

\begin{aligned}

\mathbf{F}

&= \mathbb{E}_{p(x|\theta)}[s(x|\theta) \cdot s(x|\theta)^T] \\

&= \mathbb{E}_{p(x|\theta)}[\nabla_\theta \log p(x|\theta) \cdot (\nabla_\theta \log p(x|\theta))^T] \\

&\approx \frac{1}{n} \sum_{i=1}^{n} \nabla_\theta \log p(x_i|\theta) \cdot (\nabla_\theta \log p(x_i|\theta))^T

\end{aligned}

$$
# 海森矩阵和费舍尔矩阵的关系
**费舍尔信息矩阵是对数似然函数的海森矩阵$H_{\log(p(x|\theta))}$的期望的负值**,也就是$$
F = -\mathbb{E}_{p(x|\theta)}[H_{\log(p(x|\theta))}]
$$

证明如下

首先根据定义，利用求导法则展开对数似然函数的海森矩阵 $\mathbf{H}_{\log p(x|\theta)}$：

$$
\begin{aligned}
\mathbf{H}_{\log p(x|\theta)}
&= \frac{\partial}{\partial \theta} \left( \frac{\partial \log p(x \mid \theta)}{\partial \theta} \right) \\
&= \frac{\partial}{\partial \theta} \left( \frac{\nabla_\theta p(x \mid \theta)}{p(x \mid \theta)} \right) \\
&= \frac{
    \mathbf{H}_{p(x|\theta)} p(x \mid \theta) - \nabla_\theta p(x \mid \theta) \nabla_\theta p(x \mid \theta)^T
}{
    p(x \mid \theta) p(x \mid \theta)
} \\
&= \frac{\mathbf{H}_{p(x|\theta)} p(x \mid \theta)}{p(x \mid \theta)^2}

- \frac{\nabla_\theta p(x \mid \theta) \nabla_\theta p(x \mid \theta)^T}{p(x \mid \theta)^2} \\
&= \frac{\mathbf{H}_{p(x|\theta)}}{p(x \mid \theta)}
- \left( \frac{\nabla_\theta p(x \mid \theta)}{p(x \mid \theta)} \right)
\left( \frac{\nabla_\theta p(x \mid \theta)}{p(x \mid \theta)} \right)^T
\end{aligned}

$$

进一步取关于 $p(x \mid \theta)$ 的期望，得到：

$$
\begin{aligned}
\mathbb{E}_{p(x \mid \theta)} \left[ \mathbf{H}_{\log p(x|\theta)} \right]
&= \mathbb{E}_{p(x \mid \theta)} \left[
    \frac{\mathbf{H}_{p(x|\theta)}}{p(x \mid \theta)}
    - \left( \frac{\nabla_\theta p(x \mid \theta)}{p(x \mid \theta)} \right)
    \left( \frac{\nabla_\theta p(x \mid \theta)}{p(x \mid \theta)} \right)^T
\right] \\
&= \mathbb{E}_{p(x \mid \theta)} \left[ \frac{\mathbf{H}_{p(x|\theta)}}{p(x \mid \theta)} \right]
- \mathbb{E}_{p(x \mid \theta)} \left[
    \left( \frac{\nabla_\theta p(x \mid \theta)}{p(x \mid \theta)} \right)
    \left( \frac{\nabla_\theta p(x \mid \theta)}{p(x \mid \theta)} \right)^T
\right] \\
&= \int \frac{\mathbf{H}_{p(x|\theta)}}{p(x \mid \theta)} p(x \mid \theta) \, dx
- \mathbb{E}_{p(x \mid \theta)} \left[ \nabla_\theta \log p(x \mid \theta) \cdot \nabla_\theta \log p(x \mid \theta)^T \right] \\
&= \mathbf{H}_{\int p(x|\theta)\,dx} - \mathbf{F} \\
&= \mathbf{H}_1 - \mathbf{F} \\
&= -\mathbf{F}
\end{aligned}
$$

利用以上关系，在一些优化方法中可以用 $\mathrm{F}$ 估计 $\mathrm{H}$，前者往往计算成本较低
