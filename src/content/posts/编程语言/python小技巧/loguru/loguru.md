---
title: loguru
description: ""
image: ""
published: 2025-01-05
tags:
  - python
  - 日志
category: python
draft: false
---

# Take the Tour

## 开箱即用

loguru 的核心理念就是这里只有一个 logger,为了使用方便，logger 在使用时，是提前配置好的，并且开始是默认输出至 stderr（但是这些完全是可以再进行配置的），而且打印出的 log 信息默认是配置了颜色的。

## 日志的配置

使用 `add()` 函数

```python
import sys
from loguru import logger
 
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
 
logger.debug("This's a new log message")
```
## 输出到文件
可以通过传入一个文件名字串或文件路径,loguru就会自己创建一个日志文件.如果不想在控制台也输出日志信息，因为logger是默认输出至stderr的，所以只需要在之前把它给remove掉就好了：
```python
from loguru import logger
 
logger.remove(handler_id=None)
 
logger.add("runtime.log")       # 创建了一个文件名为runtime的log文件
 
logger.debug("This's a log message in file")
```
