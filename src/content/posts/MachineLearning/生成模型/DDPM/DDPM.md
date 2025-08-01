---
title: DDPM概率扩散去噪模型
published: 2024-08-14
description: ''
image: ''
tags: [图像生成]
category: '论文阅读'
draft: false
---

# 正态分布/高斯分布

我们投 n 个骰子,重复几亿次,他们的和就可以成为正态分布了.如果某个随机变量收到很多因素影响,但是不受到任何一个决定性,就是正态分布

# 生成图片逻辑

生成一个和需要生成的图片大小一样的高斯噪声 ->多次 Denoise,denoise 的次数是人为规定的,这个过程叫做 Reverse Process(逆过程),这个是把同一个 denoise 模块多次使用了,因为不同输入的图片的清晰度不一样,使用同一个模型的效果不一定很好

![image-20240814094245684](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20240814094245684.png)

我们确实只有一个 denoise 模块,但是我们在输入的时候会多加一个输入,指使当前是第几次/当前 denoise 的严重程度,越大表示噪声的权重越大

![image-20240814094755222](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20240814094755222.png)

在 denoise 内部其实是对输入图片的噪声进行预测,然后使用输入图片减去预测的噪声从而更加清晰,之所以不用端到端是因为根据实验结果这样更好,因为产生满足噪声分布的数据更简单

![image-20240814094958002](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20240814094958002.png)

那么怎么训练呢?怎么得到 Noise Predicter 的输入图片和对应的噪声呢?其实就是对一个图片不断的使用噪声,这个过程就叫做 forward process,也叫 diffusion process,这样昨晚后就有对应的 input 和 output 了.自给自足,train 下去就结束了

![image-20240814095436536](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20240814095436536.png)

想要通过语言来控制 Text-to-image,其实就是在 denoise 的时候把 text 加进去,把 text 加进去就结束了![image-20240814100057528](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20240814100057528.png)

# 加噪过程

$$x_t=\sqrt{\beta} \times \epsilon + \sqrt{1-\beta} \times x$$

其中 $$\epsilon$$

是噪声,$$\beta$$

是系数,控制比例.每一步中加噪用到的 $$\epsilon$$

都是标准正态分布重新采样的,且每一步的 $$\beta$$

是从 0 到 1 逐渐变大的,因为扩散速度越来越快.

为了加速推导.我们设置 $$\alpha_t=1-\beta_t$$

,从而上面的式子变 $$x_t = \sqrt{1 - \alpha_t} \times \epsilon_t + \sqrt{\alpha_t} \times x_{t-1}$$

.使用这个公式可以迭代 t 次得到图像.为了加速迭代,有如下推导:

把上一时刻的公式带入公式中,可以得到新的式子,通过化简得到:$$x_t = \sqrt{a_t(1-a_{t-1})} \times \epsilon_{t-1} + \sqrt{1-a_t} \times \epsilon_t + \sqrt{a_ta_{t-1}} \times x_{t-2}$$

.这个相当于把一颗骰子投掷 2 次然后相加.但是如果我们有两个骰子,就可以一次投掷完了.同时投掷两颗骰子所得点数的概率分布等于单独投一颗两次的概率分布,所以我们只需要知道叠加后的概率分布,就只采样一次就可以了

> 卷积: 对两个概率分布计算卷积操作,实际上是计算这两个分布的所有可能的组合情况,并且得到这些组合的概率分布,也就是的加后的概率分布

现在需要把两次采样变成一次采样,就需要知道两个正态分布的联合概率分布.对于 $$\sqrt{a_t(1-a_{t-1})} \times \epsilon_{t-1}$$

,其中 $$\epsilon_{t-1}$$

服从 N(0,1),那么乘一个系数之后符合 $$N\left(\sqrt{α_t(1-α_{t-1})}, (\sqrt{α_t(1-α_{t-1})}σ)^2\right)$$

,也就变成了 $$N\left(0, (α_t-a_tα_{t-1})\right)$$

.这是正态分布基本性质,后面的符合 $$N(0,1-\alpha_t)$$

,两个相加符合 $$N(0,1-\alpha_t\alpha_{t-1})$$

,因此上面的 x_t 可以写为 $$x_t = \sqrt{1-\alpha_t\alpha_{t-1}} \times ε + \sqrt{\alpha_t\alpha_{t-1}} \times x_{t-2}$$

,其中 $$\epsilon$$

是标准正态分布,这种技巧叫做重参数化技巧.

使用数学归纳法可以得到从原图像到任意图像的关系

$$x_t = \sqrt{1 - \bar{\alpha}_t} \times \epsilon + \sqrt{\bar{\alpha}_t} \times x_0$$

其中，

$$\bar{\alpha}_t = a_t a_{t-1} a_{t-2} a_{t-3} \ldots a_2 a_1$$

# 反向过程

从 x_t 恢复得到 x_0,首先看贝叶斯公式:$$ P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

,其中 pa 是先验概率,比如小明坐地铁的概率,p(a|b) 也是概率,但是是在 b 事件发生后对 a 的修正,pb 是证据,p(b|a) 是表示 a 发生的前提下,b 事件发生的概率,叫似然,可以看做 b 对 a 的归因力度,支持 a 事件.

因为 xt-1 到 xt 加了一个随机变量,是一个随机过程,那么 xt 到 xt-1 也是一个随机过程.我们的目标是在知道 xt 的情况下推导出 xt-1 的图像.我们用 p(xt-1|xt) 表示在知道 xt 的情况下前一时刻 xt-1 的概率. 使用贝叶斯定理得到:

$$ P(x_{t-1}|x_t) = \frac{P(x_t|x_{t-1})P(x_{t-1})}{P(x_t)} $$

,其中 p(x) 表示的事从 x0 得到这个图片的概率.也就是 p(x|x0)

我们来看 p(xt|xt-1),首先 $$x_t = \sqrt{1 - \alpha_t} \times \epsilon_t + \sqrt{\alpha_t} \times x_{t-1}$$

,右边这一坨可以看做 $$N(\sqrt{a_t}x_{t-1},1-a_t)$$

的正态分布

再来看 p(xt|x0),$$x_t = \sqrt{1 - \bar{\alpha}_t} \times \epsilon + \sqrt{\bar{\alpha}_t} \times x_0$$

,那么就符合 $$N(\sqrt{\bar{a}_t}x_{t-1},1-\bar{a}_t)$$

对于 p(xt-1|x0) 也是如此.那么我们可以写出来这些东西的式子,然后带入贝叶斯公式得到 ![image-20240814110438269](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20240814110438269.png)

然后我们需要把右边的式子转换为 xt-1 的概率密度的式子.最终结果如下: 图中是一个高斯分布的概率密度函数，其 LaTeX 表示如下：

$$f(x_{t-1}|x_t, \bar{\alpha}_t) = \frac{1}{\sqrt{2\pi}\left(\frac{\sqrt{1-\bar{\alpha}_t}}{1-\bar{\alpha}_t}\right)^2} e^{-\frac{1}{2}\left[\frac{(x_{t-1}-\frac{\sqrt{1-\bar{\alpha}_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t+\frac{\sqrt{1-\bar{\alpha}_{t-1}}(1-\bar{\alpha}_t)}{1-\bar{\alpha}_t}x_0)^2}{\left[2\left(\frac{\sqrt{1-\bar{\alpha}_t}\sqrt{1-\bar{\alpha}_{t-1}}}{\sqrt{1-\bar{\alpha}_t}}\right)\right]^2}\right]}$$

,也就是 $$\left(x_{t-1} \mid x_t, x_0\right) \sim \mathcal{N} \left(\frac{\sqrt{a_t}(1-\bar{a}_{t-1})}{1-\bar{a}_t} x_t + \frac{\sqrt{\bar{a}_{t-1}}(1-a_t)}{1-\bar{a}_t} x_0, \left(\frac{\sqrt{1-a_t}\sqrt{1-\bar{a}_{t-1}}}{\sqrt{1-\bar{a}_t}}\right)^2\right)$$

我们就确定了给定 Xt 的情况下,xt-1 的概率分布.从 xT 开始一直迭代到 x0.但是我们发现其中包含了 x_0.别急,我们通过 $$x_t = \sqrt{1 - \bar{\alpha}_t} \times \epsilon + \sqrt{\bar{\alpha}_t} \times x_0$$

得到 x0 用 xt 的表示,就得到了不含 x0 的式子

$P(x_{t-1}|x_t, x_0) \sim N\left(\frac{\sqrt{a_t}(1-\bar{a}_{t-1})}{1-\bar{a}_t}x_t + \frac{\sqrt{\bar{a}_{t-1}}(1-a_t)}{1-\bar{a}_t}\times\frac{x_t - \sqrt{1-\bar{a}_t}\times\epsilon}{\sqrt{\bar{a}_t}}, (\sqrt{\frac{\beta_t(1-\bar{a}_{t-1})}{1-\bar{a}_t}})^2\right)$

$$x_t = \sqrt{1 - \bar{\alpha}_t} \times \epsilon + \sqrt{\bar{\alpha}_t} \times x_0$$

这个式子表示任意 xt 都可以看做从 x0 直接加噪得来的,然后只要知道了加入的噪声,就能得到他前一时刻 xt-1 的概率分布.因此我们只需要预测噪声就好了,然后采个样就好了.

在刚开始的时候,a 接近 0,所以直接用高斯噪声就好了

# TODO

[]VAE

[]GAN

# 论文阅读

概率扩散模型是一个参数化的卡尔可夫链,使用变分推理,在有限时间内创造一个 data.该链的转换学习是为了逆转扩散过程.

contribution: 使用参数化的扩散模型来生成高质量的图像

概率扩散模型和其他的隐变量模型的区别是近似后延分布,是逐渐添加高斯噪声的过程.阿巴阿巴

我们忽略正向传播方差可以被学习的事实并且设置他们为常量,因此,前向过程没有可以被学习的参数.

# 证据下界

数学上,我们可以认为我们观测到的变量可以被一个联合分布 p(x,z) 表示.
