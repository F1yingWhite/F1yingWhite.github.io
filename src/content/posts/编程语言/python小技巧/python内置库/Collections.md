---
title: Collections
description: ""
image: ""
published: 2025-08-11
tags:
  - python
category: python
draft: false
---

# Namedtuple

```python
import collections

if __name__ == "__main__":
    Point = collections.namedtuple("Point", ["x", "y"])
    p = Point(10, 20)
    print(p.x, p, p.y)
```

`namedtuple` 用来创建一个自定义的 tuple 对象，而且规定的了 tuple 的元素个数，可以用属性而不是索引来获取某个元素

## Defaultdict

使用 `dict` 的时候经常会出现 `KeyError`，我们可以使用 `defaultdict` 来让 key 不存在的时候返回默认值。注意默认值是调用函数返回的，而函数在创建 `defaultdict` 对象时传入。

```python
dd = collections.defaultdict(lambda: Point(0, 0))
dd["a"] = Point(1, 2)
print(dd["a"], dd["b"])
```

当然了，在常规的 `dict` 中我们也可使用 `dd.get()` 的方式来获取默认值

## OrderedDict

使用 `dict` 时，Key 是无序的。在对 `dict` 做迭代时，我们无法确定 Key 的顺序。而 OrderedDict 则是有序的，会按照插入的顺序进行排序

```python
od = collections.OrderedDict()
od["a"] = Point(1, 2)
od["b"] = Point(3, 4)
for key, value in od.items():
	print(key, value)
```

## ChainMap

`ChainMap` 可以传入多个 `dict` 组成一个逻辑上的 `dict`，查找的时候会根据传入 dict 的顺序进行查找

```python
from collections import ChainMap
import os, argparse

# 构造缺省参数:
defaults = {
    'color': 'red',
    'user': 'guest'
}

# 构造命令行参数:
parser = argparse.ArgumentParser()
parser.add_argument('-u', '--user')
parser.add_argument('-c', '--color')
namespace = parser.parse_args()
command_line_args = { k: v for k, v in vars(namespace).items() if v }

# 组合成ChainMap:
combined = ChainMap(command_line_args, os.environ, defaults)

# 打印参数:
print('color=%s' % combined['color'])
print('user=%s' % combined['user'])s
```
