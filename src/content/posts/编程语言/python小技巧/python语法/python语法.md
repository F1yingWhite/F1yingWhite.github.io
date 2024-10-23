---
title: python语法
published: 2024-09-04
description: ''
image: ''
tags: [python,language]
category: 'language'
draft: false
---

## *args和**kwargs
```python
def tests(a,b,*args,**kwargs):
    for arg in args:
        print(arg)
    for key in kwargs:
        print(key,kwargs[key])

test(1,2,3,4,a=1,b=2)
# 3,4,a a=1,b b=2
```
简单来说,args可以接受任意数量的参数,是以一个tuple的形式进行存储的,而**kwargs是用于接收有参数名的参数的

## 装饰器
python的装饰器就是有类似@的形式
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
在python中函数也是一个对象,了解了这个之后,我们就可以明白如何在函数中返回函数!
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


