---
title: FastApi
description: ""
image: ""
tags:
  - language
  - python
  - web
category: language
draft: false
published: 2024-10-29
---

# 前言

## 并发 async/await

有些函数需要使用 await,然后使用 async 的 def(只能在 async 创建的函数内使用 await)

如果使用了第三方库,而且不知道第三方支不支持 await,那么使用 def 就好了,否则使用 async

>**注意**：你可以根据需要在路径操作函数中混合使用 `def` 和 `async def`，并使用最适合你的方式去定义每个函数。FastAPI 将为他们做正确的事情。无论如何，在上述任何情况下，FastAPI 仍将异步工作，速度也非常快。但是，通过遵循上述步骤，它将能够进行一些性能优化。

### 异步代码

异步就是编程语言告诉计算机他需要再代码中的某个点等待执行完成一些事情,这些事情包括一些相对较慢的 IO,比如网络请求,api 远程调用等等

```python
async def get_burgets(number:int):
	return burgets

@app.get("/burgers")
async def read_burgers():
	burgers = await get_burgets(2)
	return burgets
```
