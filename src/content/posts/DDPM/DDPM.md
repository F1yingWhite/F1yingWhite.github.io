---
title: DDPM
published: 2024-08-14
description: ''
image: ''
tags: [图像生成]
category: '论文阅读'
draft: false 
---

## 生成图片逻辑

生成一个和需要生成的图片大小一样的高斯噪声->多次Denoise,denoise的次数是人为规定的,这个过程叫做Reverse Process(逆过程),这个是把同一个denoise模块多次使用了,因为不同输入的图片的清晰度不一样,使用同一个模型的效果不一定很好

![image-20240814094245684](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20240814094245684.png)

我们确实只有一个denoise模块,但是我们在输入的时候会多加一个输入,指使当前是第几次/当前denoise的严重程度,越大表示噪声的权重越大

![image-20240814094755222](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20240814094755222.png)

在denoise内部其实是对输入图片的噪声进行预测,然后使用输入图片减去预测的噪声从而更加清晰,之所以不用端到端是因为根据实验结果这样更好,因为产生满足噪声分布的数据更简单

![image-20240814094958002](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20240814094958002.png)

那么怎么训练呢?怎么得到Noise Predicter的输入图片和对应的噪声呢?其实就是对一个图片不断的使用噪声,这个过程就叫做forward process,也叫diffusion process,这样昨晚后就有对应的input和output了.自给自足,train下去就结束了

![image-20240814095436536](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20240814095436536.png)

想要通过语言来控制Text-to-image,其实就是在denoise的时候把text加进去,把text加进去就结束了![image-20240814100057528](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/image-20240814100057528.png)
