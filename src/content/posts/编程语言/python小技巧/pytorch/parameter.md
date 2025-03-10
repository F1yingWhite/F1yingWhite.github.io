---
title: parameter
description: ""
image: ""
published: 2025-03-05
tags:
  - python
  - pytorch
category: " pytorch"
draft: false
---

parameter 实际上也是 tensor，也就是多维矩阵，是 variable 类中的一个特殊类，当我们创建一个 model 时，parameter 会自动累加到 parameter 列表中。

```python
import torch.nn as nn
a = nn..Parameter(torch.zeros(100,20))#这里进行初始化
```
