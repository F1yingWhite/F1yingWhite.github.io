---
title: python语法
published: 2024-09-04
description: ""
image: ""
tags:
  - python
category: language
draft: false
---

# *args 和**kwargs

```python
def tests(a,b,*args,**kwargs):
    for arg in args:
        print(arg)
    for key in kwargs:
        print(key,kwargs[key])

test(1,2,3,4,a=1,b=2)
# 3,4,a a=1,b b=2
```

简单来说,args 可以接受任意数量的参数,是以一个 tuple 的形式进行存储的,而**kwargs 是用于接收有参数名的参数的

# 装饰器

python 的装饰器就是有类似@的形式

首先我们需要认识以下内容

```python
def hi(name="yasoob"):
    return "hi " + name
print(hi())
# 我们甚至可以将一个函数赋值给一个变量，比如
greet = hi
print(greet())
# output: 'hi yasoob'
del hi
print(hi())
#outputs: NameError
print(greet())
#outputs: 'hi yasoob'
```

在 python 中函数也是一个对象,了解了这个之后,我们就可以明白如何在函数中返回函数!

```python
def hi(name="yasoob"):
    def greet():
        return "now you are in the greet() function"

    def welcome():
        return "now you are in the welcome() function"

    if name == "yasoob":
        return greet
    else:
        return welcome
```

此外,函数也可以作为参数被传入另一个函数

```python
def hi():
    return "hi yasoob!"

def doSomethingBeforeHi(func):
    print("I am doing some boring work before executing hi()")
    print(func())
```

有了上述知识,我们就可以了解装饰器如何工作的了

```python
def warps(func):
    def wrapTheFunction(*args,**kwargs):
        print("1")
        func(*args,**kwargs)
        print("2")
    return wrapTheFunction

def func():
    print("1")

func = warps(func)
# 这部分类似
@warps
def func():
    print("1")
print(func.__name__)#这里的名字被warpTheFunction替换了,为了修正,我们需要做出如下更正

from functools import wraps
def MyWarps(func):
    @wraps(func)
    def wrapTheFunction(*args,**kwargs):
        print("1")
        func(*args,**kwargs)
        print("2")
    return wrapTheFunction

@MyWarps
def func():
    print("a")
```

# Dataclass 注解

假设我们有一个场景,需要一个数据对象来存储一些运动员信息,可以使用 tuple 或者 dict 实现但 tuple 需要记住位置,用 dict 的语法不太优雅,

```python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int

p = Person("John", 36)
print(p.name)#这样非常优雅
```

# Python 内置类型

TypedDict: 用于定义带有具体键值对的字典,类似 struct 和 interface

```python
from typing import TypedDict

class Person(TypedDict):
	name: string
	age: int

def get_person() -> person:
	reutrn {"name":"飞白","age":18}
```

Literal: 表示具体的值而不是类型

```python
def get_status() -> Literal["success","failure"]:
	reutrn "success"
```

optional 表示 Union,可以有多个类型

```python
from typing import Optional

def find_item(item_id: int) -> Optional[str]:
    if item_id == 0:
        return None
    return "Item"
```
