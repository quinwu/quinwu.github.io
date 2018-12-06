---
title: Python __slots__ 
date: 2018-12-06 16:43:31
tags:
  - Python
categories: 笔记
---


### Saving Space with the `__slots__` Class Attribute

By default, Python stores instance attributes in a per-instance `dict` named `__dict__`.

<!--more--> 

为了使用底层的散列表提升访问速度，字典会消耗大量内存。如果处理上百万个属性不多的实例，通过`__slots__`类属性，能节省大量的内存，方法是让解释器在元组中存储实例属性，而不是用字典。

> If you are dealing with millions of instances with few attributes, the `__slots__` class attribute can save a lot of memory, by letting the interpreter store the instance attributes in a `tuple` instead of a `dict`.

To define `__slots__` , you create a class with that name and assign it an iterable of `str` with identifiers for the instance attributes. I like to use a `tuple`for that, because it conveys the message that the `__slots__` definition cannot change.

> By define `__slots__` in the class, you are telling the interpreter : "These are all the instance attribute in this class." Python then stores them in a tuple-like structure in each instance, avoiding the memory overhead of the per-instance `__dict__`. This can make a huge differnce in memory usage if you have millions of instances active at the same time.

在类中定义`__slots__` 属性后，实例不能再有`__slots__`中所列名称之外的其他属性。 不要使用 `__slots__` 属性禁止类的用户新增实例属性。`__slots__` 是用于优化的，不是为了约束程序员。

### The Problems with `__slots__`

- 每个子类都要定义 `__slots__` 属性，因为解释器会忽略继承的 `__slots__` 属性

- 实例只能拥有 `__slots__` 中列出的属性，除非把 `__dict__` 加入`__slots__` 中，这样做就失去了节省内存的功效

- 如果不把 `__weakref__` 加入 `__slots__` ，实例就不能作为弱引用的目标。

   