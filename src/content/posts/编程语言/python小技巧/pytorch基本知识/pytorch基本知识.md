---
title: pytorch基本知识
published: 2024-07-16
description: ""
image: ""
tags:
  - python
category: language
draft: false
---

# Normalization

Batch Normalization(批标准化),和普通的数据标准化类似,是把分散的数据统一的一种做法,也是优化神经网络的一种方法,具有统一规格的数据, 能让机器学习更容易学习到数据之中的规律.Batch normalization 也可以被看做一个层面. 在一层层的添加神经网络的时候, 我们先有数据 X, 再添加全连接层, 全连接层的计算结果会经过 激励函数 成为下一层的输入, 接着重复之前的操作. Batch Normalization (BN) 就被添加在每一个全连接和激励函数之间.

计算结果在进入激活函数前的值非常重要,也就是数据的分布对与激活函数来说很重要.大部分数据在一个去建立才能有效的进行传递.

三维医学图像处理中,现存不足时经常遇到的问题,模型应该在 batch size 和 patch size 之间做出权衡.Unet 中应该优先考虑 patch_size,保证模型能获得足够的信息来进行推理,但是 batch size 的最小值应该大于等于 2,因为我们需要保证训练过程中优化的鲁棒性.在保证 patch size 的情况下如果现存有多余,再增加 batch size.因为 batch size 都比较小,所以大多使用 Instance Norm 而不是 BN.

- BatchNorm：batch 方向做归一化，算 NxHxW 的均值，对小 batchsize 效果不好；BN 主要缺点是对 batchsize 的大小比较敏感，由于每次计算均值和方差是在一个 batch 上，所以如果 batchsize 太小，则计算的均值、方差不足以代表整个数据分布。
- LayerNorm：channel 方向做归一化，算 CxHxW 的均值，主要对 RNN(处理序列) 作用明显，目前大火的 Transformer 也是使用的这种归一化操作；
- InstanceNorm：一个 channel 内做归一化，算 H*W 的均值，用在风格化迁移；因为在图像风格化中，生成结果主要依赖于某个图像实例，所以对整个 batch 归一化不适合图像风格化中，因而对 HW 做归一化。可以加速模型收敛，并且保持每个图像实例之间的独立。
- GroupNorm：将 channel 方向分 group，然后每个 group 内做归一化，算 (C//G)HW 的均值；这样与 batchsize 无关，不受其约束，在分割与检测领域作用较好。

# 混合精度训练

日常的使用通常使用单精度浮点表示 (float32),单精度和半精度的表示如下,与双精度表示,半精度仅有 16bit

![image-20240728162624237](https://p.ipic.vip/qisxcy.png)

所谓混合精度训练,就是单精度和半精度混合,float16 和 float 相比内存少,计算快

- 内存少: 只需要一般的精度,memory-bandwidht 减半,模型的 batch size 可以更大,训练的时候多卡的交互 (通信量) 减少,减少等待时间,加快数据沟通
- 计算快:GPU 针对 16fp 进行优化,吞吐量可以达到单精度的 2~8 倍

那为什么要用混合精度呢?

- 数据溢出 Underflow:fp16 的范围下载,大概 $2^{-24}到65504$ 之间,对于深度学习来说,最大的问题在于后期的梯度计算的时候会发生梯度消失
- 舍入误差:fp16weight:$2^{-3}$,gradient:$2^{-14}$,那么这俩加起来还是 $2^{-3}$,这就是舍入误差.

为了解决这个问题,我们的方法如下:

- fp32 备份权重,主要解决舍入误差,可以概括为：weights, activations, gradients 等数据在训练中都利用 FP16 来存储，同时拷贝一份 FP32 的 weights，用于更新,确保在更新的时候是在 float32 下进行的
- loss scale: 主要解决 underflow 问题,由于链式法则的存在,loss 上的 scale 会作用在梯度上.只有在进行更新的时候，才会将 scaled-gradient 转化为 fp32，同时将 scale 抹去。
- 提高算数精度: 在某些模型中，fp16 矩阵乘法的过程中，需要利用 fp32 来进行矩阵乘法中间的累加 (accumulated)，然后再将 fp32 的值转化为 fp16 进行存储。 换句不太严谨的话来说，也就是利用 **利用 fp16 进行乘法和存储，利用 fp32 来进行加法计算**。 这么做的原因主要是为了减少加法过程中的舍入误差，保证精度不损失。因此只有特定 gpu 才可以这么做

# 如何使用混合精度?

`torch.amp` 提供了混合精度的便利方法，其中一些操作使用 `torch.float32`,对于精度较低的浮点数则使用 `torch.bfloat16` 或者 `torch.float16`.一些算子，如线性层和卷积，在 `lower_precision_fp` 中速度快得多。其他算子，如归约（reductions），通常需要 `float32` 的动态范围。混合精度尝试将每个算子与其适当的数据类型匹配。通常混合精度训练会同时使用 `torch.atuocast` 和 `torch.cuda.amp.GradScaler`.

当 GPU 饱和的时候，混合精度能够提供最大的加速，小型网络可能是 cpu 瓶颈，混合精度对性能的提升不一定会很高。如果没有 `torch.cuda.amp`，默认的网络将以 `torch.float32` 运行。添加 `torch.autocast` 的实例充当上下文管理器，能够让脚本的某些区域一混合精度运行。在这些区域中，CUDA 算子将以 autocast 为每个算子选择何种精度

```python
for epoch in range(0): # 0 epochs, this section is for illustration only
    for input, target in zip(data, targets):
        # Runs the forward pass under ``autocast``.
        with torch.autocast(device_type=device, dtype=torch.float16):
            output = net(input)
            # output is float16 because linear layers ``autocast`` to float16.
            assert output.dtype is torch.float16

            loss = loss_fn(output, target)
            # loss is float32 because ``mse_loss`` layers ``autocast`` to float32.
            assert loss.dtype is torch.float32

        # Exits ``autocast`` before backward().
        # Backward passes under ``autocast`` are not recommended.
        # Backward ops run in the same ``dtype`` ``autocast`` chose for corresponding forward ops.
        loss.backward()
        opt.step()
        opt.zero_grad() # set_to_none=True here can modestly improve performance
```

## 添加 GradScaler

梯度缩放有助于在使用混合精度训练的时候，梯度因为幅度过小而变为 0（下溢）

```python
# Constructs a ``scaler`` once, at the beginning of the convergence run, using default arguments.
# If your network fails to converge with default ``GradScaler`` arguments, please file an issue.
# The same ``GradScaler`` instance should be used for the entire convergence run.
# If you perform multiple convergence runs in the same script, each run should use
# a dedicated fresh ``GradScaler`` instance. ``GradScaler`` instances are lightweight.
scaler = torch.amp.GradScaler("cuda")

for epoch in range(0): # 0 epochs, this section is for illustration only
    for input, target in zip(data, targets):
        with torch.autocast(device_type=device, dtype=torch.float16):
            output = net(input)
            loss = loss_fn(output, target)

        # Scales loss. Calls ``backward()`` on scaled loss to create scaled gradients.
        scaler.scale(loss).backward()

        # ``scaler.step()`` first unscales the gradients of the optimizer's assigned parameters.
        # If these gradients do not contain ``inf``s or ``NaN``s, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(opt)

        # Updates the scale for next iteration.
        scaler.update()

        opt.zero_grad() # set_to_none=True here can modestly improve performance
```

此外我们还可以设置自动混合精度

```python
use_amp = True

net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)
scaler = torch.amp.GradScaler("cuda" ,enabled=use_amp)

start_timer()
for epoch in range(epochs):
    for input, target in zip(data, targets):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            output = net(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
end_timer_and_print("Mixed precision:")
```

## 检查/修改梯度

由 `scaler.scale(loss).backward()` 产生的梯度都是经过缩放的。由 `scaler.scale(loss).backward()` 产生的梯度都是经过缩放的。如果你希望在 `backward()` 和 `scaler.step(optimizer)` 之间修改或检查参数的 `.grad` 属性，应首先使用 [scaler.unscale_(optimizer)](https://pytorch.ac.cn/docs/stable/amp.html#torch.cuda.amp.GradScaler.unscale_) 取消缩放。

```python
for epoch in range(0): # 0 epochs, this section is for illustration only
    for input, target in zip(data, targets):
        with torch.autocast(device_type=device, dtype=torch.float16):
            output = net(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()

        # Unscales the gradients of optimizer's assigned parameters in-place
        scaler.unscale_(opt)

        # Since the gradients of optimizer's assigned parameters are now unscaled, clips as usual.
        # You may use the same value for max_norm here as you would without gradient scaling.
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)

        scaler.step(opt)
        scaler.update()
        opt.zero_grad() # set_to_none=True here can modestly improve performance
```

# 梯度截断

在网络变深的时候反向传播容易引起梯度消失和梯度爆炸，对于梯度爆炸的问题解决方法之一就是梯度裁剪，也就是设置一个梯度大小的上限。

梯度裁剪的函数为：`torch.nn.utils.clip_grad_norm_(_parameters_, _max_norm_, _norm_type=2.0_, _error_if_nonfinite=False_, _foreach=None_)`，其中 parameters: 网络参数 max_norm: 该组网络参数梯度的范数上线 norm_type: 范数类型。

>[!tip]
> 梯度本身是一个向量，因此这里是 L2 范数

# 梯度累计

在深度学习训练的时候，数据的 batch size 大小受到 GPU 内存限制，batch size 大小会影响模型最终的准确性和训练过程的性能。在 GPU 内存不变的情况下，模型越来越大，那么这就意味着数据的 batch size 只能缩小，这个时候，梯度累积（Gradient Accumulation）可以作为一种简单的解决方案来解决这个问题。

```python
ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
with ctx:
	res = model(X)
	loss = loss_fct(res,label)

scaler.scale(loss).backward()  # 梯度累计到grad里面

if (step + 1) % args.accumulation_steps == 0:
	scaler.unscale_(optimizer)
	torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

	scaler.step(optimizer)
	scaler.update()

	optimizer.zero_grad(set_to_none=True)#减少内存使用
```
