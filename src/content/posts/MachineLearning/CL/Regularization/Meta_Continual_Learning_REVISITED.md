---
title: Meta_Continual_Learning_REVISITED
description:
image: ""
published: 2025-08-19 00:00:00
tags:
  - 论文阅读
  - 持续学习
category: 论文阅读
draft: false
updated: 2025-08-20 19:18:55
---

# Abstract

通常的持续学习至今都使用了基于正则化的方法,这些方法都归结为依赖于模型权重的海森矩阵近似.但是这些方法在知识迁移和遗忘之间存在次优的均衡.另一类元持续学习方法要求先前的任务梯度和当前的任务梯度一致.在本文中,我们把元持续学习和正则化的方法连接起来,

# Introduction

持续学习在非平稳分布的数据集上进行训练,在 CL 中的 i.i.d 假设可能会导致在能行遗忘,引起先前的学习性能的下降

>[!TIP]
>i.i.d 是指一组随机变量中每个变量的概率分布都相同，且这些随机变量互相独立

今年来有许多的工作尝试解决灾难性遗忘,正则化方法是其中的一个重要分支,旨在保留和先前的任务相关的重要参数的权重以维持其性能.这些正则化方法的主要区别是他们对于近似海森矩阵的方法有所不同.尽管这些方法努力改进海森近似以保留先前任务的重要权重，但它们对先前任务估计的海森矩阵在后续训练中保持不变且无法更新。随着模型参数逐渐偏离最初计算海森矩阵的点，由于泰勒展开的截断误差增加，未改变的估计海森矩阵会逐渐失去准确性。

与在每个任务结束后显示的计算海森矩阵不同,元持续学习方法使用双层优化获得的超梯度隐式地利用二阶海森信息.超梯度的计算涉及从内存缓冲区中抽取的一批样本，该缓冲区存储了来自先前任务的样本，这使得 Meta-CL 能够利用来自先前任务的最新二阶信息。然而，值得注意的是，使用从内存缓冲区抽取的样本可能无法提供先前任务的完整表示，甚至可能缺乏某些任务的数据。这可能导致海森估计中存在错误信息。因此，隐式估计的海森矩阵可能存在相当大的变化，最终导致先前任务性能的下降。

# 基于正则化的持续学习的统一框架

基于正则化的连续学习方法分析。连续学习考虑一个包含 $N$ 个任务的序列 $[T^1, T^2, \cdots, T^N]$，其中第 $i$ 个任务包含 $N^i$ 个样本，即 $T^i = (\mathbf{X}^i, \mathbf{Y}^i) = \{(x_n^i, y_n^i)\}_{n=0}^{N^i}$。这里，$T^j$ 表示当前的训练任务，$T^{[1:j]}$ 表示模型迄今为止所经历的所有前 $j$ 个任务，$\mathcal{L}^i(\theta)$ 表示参数 $\theta$ 在任务 $T^i$ 上的经验风险。连续学习的目标是学习一个具有参数 $\theta$ 的模型，使其最小化所有已见任务 $T^{[1:j]}$ 的平均经验风险，即 $\frac{1}{j}(\sum_{i=1}^{j} \mathcal{L}^i(\theta))$。

然而，在基于正则化的连续学习（CL）中，我们无法访问先前学习任务 $T^{[1:j-1]}$ 的样本数据。因此，模型参数 $\theta$ 不能直接优化以最小化之前任务对应的经验风险总和，即 $\sum_{i=1}^{j-1} \mathcal{L}^i(\theta)$。因此，正则化类的连续学习方法选择在训练当前任务 $T^j$ 时，对先前任务的经验风险进行近似（。具体地，在最简单的情形下，当 $j=2$ 且正在训练任务 $T^2$ 时，基于正则化的方法的目标是最小化 $\frac{1}{2}(\mathcal{L}_1^{\text{prox}} + \mathcal{L}^2)$，其中 $\mathcal{L}_1^{\text{prox}}(\theta)$ 是通过在 $\hat{\theta}^1$ 处对 $\mathcal{L}^1(\theta)$ 进行泰勒展开得到的近似，形式为：

$$
\mathcal{L}_1^{\text{prox}} = \mathcal{L}^1(\hat{\theta}^1) + (\theta - \hat{\theta}^1)^\top \nabla \mathcal{L}^1(\hat{\theta}^1) + \frac{1}{2}(\theta - \hat{\theta}^1)^\top \nabla^2 \mathcal{L}^1(\hat{\theta}^1)(\theta - \hat{\theta}^1)
$$

其中，$\hat{\theta}^1$ 表示在完成任务 $T^1$ 训练后模型的参数。对于更一般的情况，当 $j > 2$ 时，我们可以将前 $(j-1)$ 个任务的经验损失近似为：

$$
\mathcal{L}_{j-1}^{\text{prox}}(\theta) = \sum_{i=1}^{j-1} \underbrace{\mathcal{L}^i(\hat{\theta}^i)}_{(a)} + \underbrace{(\theta - \hat{\theta}^i)^\top \nabla_\theta \mathcal{L}^i(\hat{\theta}^i)}_{(b)} + \underbrace{\frac{1}{2}(\theta - \hat{\theta}^i)^\top \mathbf{H}^i (\theta - \hat{\theta}^i)}_{(c)}
\tag{1}
$$

其中，(a) 是在最优参数处的损失值，(b) 是一阶导数项，(c) 是二阶海森矩阵项，用于捕捉参数变化带来的曲率影响。这里的 a 和 $\theta$ 无关和 b 的导数为 0 都可以被忽略,因此,我们可以得出命题:

**命题 1**：在基于正则化的连续学习中，若模型参数 $\theta$ 在邻域集合 $\cup_{i=1}^{j-1} \mathcal{N}^i$ 内搜索，其中 $\mathcal{N}^i = \{\theta : d(\theta, \hat{\theta}^i) < \delta^i\}$，则 $\theta$ 的迭代更新规则近似为：

$$
\theta := \theta - \alpha (\mathbf{H}^1 + \mathbf{H}^2 + \cdots + \mathbf{H}^{j-1})^{-1} \nabla_\theta \mathcal{L}^j(\theta). \tag{2}
$$

## 连续学习中海森矩阵的作用

: 令 $\mathbf{H} = \sum_{i=1}^{j-1} \mathbf{H}^i$ 表示公式 (2) 中的海森部分。该更新规则表明，海森矩阵 $\mathbf{H}$ 在优化过程中起着关键作用。具体而言，通过将海森矩阵 $\mathbf{H}$ 的逆与梯度 $\nabla_\theta \mathcal{L}^j(\theta)$ 相乘，梯度在曲率较大的方向上被抑制，即对应于 $\mathbf{H}$ 大特征值的特征向量方向；而在曲率较小的方向上则被放大。在基于正则化的连续学习算法中，海森矩阵的高曲率方向通常与存储先前任务知识的重要权重对齐，这有助于抑制这些方向上的更新幅度以减少遗忘；而低曲率方向则有利于模型适应当前任务。

>[!NOTE]
> 假设海森矩阵 $\mathbf{H}$ 有特征值分解：(这里海森矩阵是对称阵,那么根据谱定理一定能特征分解且都是实数特征值,特征向量还是正交的)
>
> $$\mathbf{H} = U \Lambda U^\top$$
>
> 其中：
> - $U = [\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_n]$ 是特征向量矩阵
> - $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$ 是对角特征值矩阵
> - $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_n \geq 0$
> 
> 更新公式中的关键项是：
>
> $$\mathbf{H}^{-1} \nabla_\theta \mathcal{L}^j(\theta) = U \Lambda^{-1} U^\top \nabla_\theta \mathcal{L}^j(\theta)$$
>
> 将梯度投影到特征向量方向上：
>
> $$\nabla_\theta \mathcal{L}^j(\theta) = \sum_{i=1}^n (\nabla_\theta \mathcal{L}^j(\theta)^\top \mathbf{u}_i) \mathbf{u}_i = \sum_{i=1}^n g_i \mathbf{u}_i$$
>
> 其中 $g_i = \nabla_\theta \mathcal{L}^j(\theta)^\top \mathbf{u}_i$ 是梯度在第 $i$ 个特征向量方向上的分量。
>
> $$\mathbf{H}^{-1} \nabla_\theta \mathcal{L}^j(\theta) = \sum_{i=1}^n \frac{g_i}{\lambda_i} \mathbf{u}_i$$
>
> **" 在大特征值方向上被抑制 "** 指的是：
>
> - **大特征值方向**：当 $\lambda_i$ 很大时，$\frac{g_i}{\lambda_i}$ 变得很小
> - **小特征值方向**：当 $\lambda_i$ 很小时，$\frac{g_i}{\lambda_i}$ 变得很大

**统一框架下的基于正则化的持续学习**: 鉴于海森矩阵在模型更新中的重要作用,各种正则化的方法主要致力于改进海森矩阵的近似,比如 EWC 和 On-EWC 使用对角 fihser 矩阵来近似海森矩阵,此外还有因子分解拉普拉斯近似等方法.

# 从海森近似角度重新审视元持续学习

具体而言，虽然基于正则化的方法显式地近似二阶海森矩阵，但 Meta-CL 通过超梯度的计算来利用二阶信息。

![image.png](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/20250819144426709.png)

## 元学习

Riemer 等人提出了元学习框架,旨在最小化所有已见任务 $T^{[1:j]}$ 的经验风险,该方法中,所有的先前任务可以有一个记忆缓冲区 $\mathcal{M}$ 来访问,该缓冲区保存了之前的任务数据,具体来说,该任务的近似方案被表述为:

$$
\begin{align}
\min_{\theta} \mathcal{L}^{[1:j]}(\theta_{(K)}) \quad \text{s.t.} \quad \theta_{(K)} = U_K(\theta; \mathcal{T}^j), \\
where U_K(\theta; \mathcal{T}^j) = \underbrace{U \circ \cdots \circ U}_{K \text{ inner steps}} (U(\theta_{(0)}; \mathcal{T}^j_{(0)}))， U(\theta_{(k)}; \mathcal{T}^j_{(k)}) = [\theta - \beta \nabla_\theta \mathcal{L}^j_{(k)}(\theta)]|_{\theta=\theta_{(k-1)}} \tag{3}
\end{align}
$$

$\mathcal{L}^{[1:j]}(\theta_{(K)})$ 表示参数 $\theta_{(K)}$ 在所有 $j$ 个任务 $T^{[1:j]}$ 上的经验风险，实际中该值基于当前任务 $T^j$ 和记忆缓冲区 $\mathcal{M}$ 中的数据进行估计，$\theta_{(k)}$ 表示任务 $T^j$ 学习过程中第 $k$ 步内部更新的模型参数，而 $U(\cdot)$ 表示内部循环优化中的一次 SGD 更新。

在外部循环中，我们从 $T^j$ 和记忆缓冲区 $\mathcal{M}$ 中随机采样数据 $\epsilon_b$。然后计算损失 $\mathcal{L}^{[1:j]}(\theta_{b(K)})$，并得到关于 $\theta_b$ 的超梯度 $g^{\epsilon_b}_{\theta_b} := \partial \mathcal{L}(\theta_{b(K)}; \epsilon_b)/\partial \theta_b$。最后，我们使用超梯度更新 $\theta_b$。需要注意的是，不同的 Meta-CL 方法通常采用不同的更新梯度 $g^{\epsilon_b}_{\theta_b}$ 近似方式。<font color="#92d050">说白了就是内循环中,拷贝当前模型的一个备份,进行 n 次训练后再拿到外循环中计算一个大的 loss 来更新原始模型</font>

## 海森近似角度重新审视元持续学习

为了解决上面的公式 3 中的双重优化,我们需要计算 $\mathcal{L}^{[1:j]}(\theta_{(K)})$ 关于 $\theta$ 的超梯度，即：

$$
\frac{\partial \mathcal{L}^{[1:j]}(\theta_{(K)})}{\partial \theta} = \frac{\partial \mathcal{L}^{[1:j]}(\theta_{(K)})}{\partial \theta_{(K)}} \frac{\partial \theta_{(K)}}{\partial \theta},
$$

其中第一项在 $\theta$ 附近的泰勒近似为：

$$
\frac{\partial \mathcal{L}^{[1:j]}(\theta_{(K)})}{\partial \theta_{(K)}} = \nabla_{\theta_{(K)}} \mathcal{L}^{[1:j]}(\theta_{(K)}) \approx \nabla_{\theta_{(K)}} \mathcal{L}^{[1:j]}(\theta) + \mathbf{H}_M^j (\theta_{(K)} - \theta) + (\theta_{(K)} - \theta)^T \otimes \mathbf{T} \otimes (\theta_{(K)} - \theta).
$$

注意，$\mathbf{H}_M^j = \nabla_{\theta_{(K)}}^2 \mathcal{L}^{[1:j]}(\theta)$ 和 $\mathbf{T}$ 分别表示海森矩阵和三阶对称张量，$\otimes$ 表示克罗内克积。通过上述近似，并为简化起见假设在优化 $T^j$ 时仅进行一次内循环更新（即 $K=1$），

这里我们希望找到一个最优解 $\theta^*$ 来达到最小损失,也就是 $\frac{\partial \mathcal{L}^{[1:j]}(\theta_{(K)})}{\partial \theta}=0$,近似 $\frac{\partial \mathcal{L}^{[1:j]}(\theta_{(K)})}{\partial \theta_{(K)}}=0$,

**命题 2**：对于具有单次内循环适应的 Meta-CL，假设 $\theta_{(K)}$ 位于最优模型参数 $\theta^* = \arg\min_\theta \mathcal{L}^{[1:j]}(\theta_{(K)})$ 的 $\epsilon$- 邻域 $N(\theta^*, \epsilon)$ 内，$\mathcal{L}$ 是 $\mu$- 光滑的，且 $\beta < \sqrt{\delta / |\nabla_\theta \mathcal{L}^j(\theta) - (\nabla_\theta \mathcal{L}^j(\theta))^2|}$，其中 $\delta$ 是一个小数。令 $\alpha = \beta^2$，则迭代更新规则近似为：

$$
\theta := \theta - \alpha (\mathbf{H}_M^j)^{-1} \nabla_\theta \mathcal{L}^j(\theta).
$$

上述表示这个超梯度方法实际上隐式等价与使用海森矩阵并遵循基于正则化的方法的归一迭代.比对俩个命题,我们能发现：Meta-CL 利用从记忆缓冲区 $\mathcal{M}$ 中采样的样本，隐式地计算海森矩阵 $\mathbf{H}_M^j$，以近似公式 (2) 中的 $(\mathbf{H}^1 + \mathbf{H}^2 + \cdots + \mathbf{H}^{j-1})$。与基于正则化的方法估计并固定每个单独的 $\mathbf{H}^i$ 不同，Meta-CL 隐式地计算一个海森矩阵 $\mathbf{H}_M^j$，使其能够获得最新的二阶信息。

尽管 Meta-CL 通过记忆缓冲区 $\mathcal{M}$ 利用最新的海森信息具有优势，但它也可能引入错误信息。例如，从 $\mathcal{M}$ 中抽取的样本可能无法充分代表先前任务，或可能缺少特定任务的实例。这种不充分性可能导致估计的海森矩阵在这些先前任务的重要权重方向上错误地具有较小的曲率，从而无法抑制该方向上的更新。因此，这可能导致这些任务的先前知识丢失，并导致遗忘。

# 元学习中的方差缩减

 对于元学习中随机抽取策略带来的高方差,本文提出了方差缩减元持续学习,这种超梯度策略的方差缩减等价于在海森矩阵上添加正则化项.

## 方差缩减元持续学习

VR-MCL 旨在通过减少超梯度的方差来降低其隐式估计的海森矩阵的方差。在在线持续学习设置中，任务中的样本是以小批量形式接收的，且无法访问完整的样本集，这给这些技术的实施带来了挑战.

![image.png](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/20250819201106631.png)

基于此,我们提出了基于动量的方差减小元持续学习方法,该方法不用计算全部的梯度,而是保留先前步骤中的模型.其中 $\hat{\mathbf{g}}_{\theta_{b-1}}^{\epsilon_b-1}$ 表示动量分量，而 $\mathbf{g}_{\theta_{b-1}}^{\epsilon_b}$ 和 $\mathbf{g}_{\theta_b}^{\epsilon_b}$ 分别表示使用数据 $\epsilon_b$ 在先前参数 $\theta_{b-1}$ 和当前参数 $\theta_b$ 上计算得到的超梯度。在 VR-MCL 中，具有降低方差的最终更新梯度如公式 (4) 所示，其中 $r$ 表示动量比例系数。最终的梯度为:

$$
\hat{\mathbf{g}}_{\theta_b}^{\epsilon_b} = \mathbf{g}_{\theta_b}^{\epsilon_b} + r\left(\hat{\mathbf{g}}_{\theta_{b-1}}^{\epsilon_{b-1}} - \mathbf{g}_{\theta_{b-1}}^{\epsilon_b}\right).
\quad (4)
$$

为了理解为什么公式 (4) 能够产生更小的梯度方差，我们首先考察误差项 $\Delta_b := \hat{\mathbf{g}}_{\theta_b}^{\epsilon_b} - G_{\theta_b}$(这里的 G 应该是在全部数据上的梯度)。该项衡量了当我们使用 $\hat{\mathbf{g}}_{\theta_b}^{\epsilon_b}$ 作为梯度方向，而非真实的（但未知的）全批量梯度方向 $G_{\theta_b}$ 时所产生的误差。通过证明 $\mathbb{E}[\|\Delta_b\|^2]$ 随时间递减，我们可以展示 VR-MCL 在方差降低方面的有效性 1。将公式 (4) 代入 $\Delta_b$，可得如下表达式：

$$
\Delta_b = r\Delta_{b-1} + (1-r)(\mathbf{g}_{\theta_b}^{\epsilon_b} - G_{\theta_b}) + r(\mathbf{g}_{\theta_b}^{\epsilon_b} - \mathbf{g}_{\theta_{b-1}}^{\epsilon_b} - (G_{\theta_b} - G_{\theta_{b-1}})).
\quad (5)
$$

第二项 $(1-r)(\mathbf{g}_{\theta_b}^{\epsilon_b} - G_{\theta_b})$ 可以通过调整动量比例 $r$ 的值来调节；第三项 $\mathbf{g}_{\theta_b}^{\epsilon_b} - \mathbf{g}_{\theta_{b-1}}^{\epsilon_b} - (G_{\theta_b} - G_{\theta_{b-1}})$ 在 $\mu$- 光滑损失函数 $\mathcal{L}$ 的假设下，其期望量级为 $O(\|\theta_b - \theta_{b-1}\|) = O(\alpha\|\hat{\mathbf{g}}_{\theta_{b-1}}^{\epsilon_{b-1}}\|)$。因此，我们有 $\|\Delta_b\| = r\|\Delta_{b-1}\| + S$，其中 $S$ 是一个受 $r$ 和学习率 $\alpha$ 影响的常数。由于 $\|\Delta_b\|$ 会持续减小，直到收敛到 $S/(1-r)$，选择合适的 $r$ 和 $\alpha$ 使得 $S/(1-r)$ 尽可能小，即可实现我们期望的方差降低效果。

## VR-MCL 的理论证明

**定理 1（VR-MCL 的遗憾界）**。在线持续学习中的遗憾定义为 $\mathbf{CR}_j = \tilde{F}(\theta) - F(\theta^*)$，其中 $\tilde{F}(\theta) = \sum_{i=1}^{j} \mathcal{L}^i(\hat{\theta}^i)$，$F(\theta^*) = \min_\theta \sum_{i=1}^{j} \mathcal{L}^i(\theta)$。假设目标函数 $F$ 具有 $\varphi$-Lipschitz 海森矩阵，且是 $\mu$- 强凸、$G$-Lipschitz、$\eta$- 光滑的，并满足附录 A 中的四个假设条件，$\sigma$ 和 $M$ 分别是假设 3 和假设 4 中的两个大常数，$T$ 表示训练迭代次数。那么对于任意 $\delta \in (0,1)$，以至少 $1-\delta$ 的概率，我们有：

$$
\mathbf{CR}_j \leq (\log T + 1)(F(\theta_1) - F(\theta^*)) + \frac{LD^2(\log T + 1)^2}{2}
+ \frac{LD^2(\log T + 1)^2}{2} + (16LD^2 + 16\sigma D + 4M)\sqrt{2T\log(8T/\delta)} = \tilde{O}(\sqrt{T}).
$$

**定理 1 说明**，在随机设置下，只要满足一些温和的假设条件，VR-MCL 算法能够以高概率（w.h.p.）达到近似最优的 $\tilde{O}(\sqrt{T})$ 遗憾界，这在在线优化问题中具有理论上的优越性。该结果从理论上验证了所提出的 VR-MCL 算法的有效性。详细证明见附录 A。
