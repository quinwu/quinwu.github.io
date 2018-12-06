---
title: Python中下划线的用法
date: 2018-01-24 17:20:30
tags:
  - Python
categories: 笔记
---

## Python中的下划线

Python 中的 `_` 的不同用法绝大部分都是一种惯例约定。

<!--more--> 



### 单个下划线（ `_` ）

##### 解释器中

`_` 符号代表交互解释器中的最后一次执行语句的结果。

##### 作为名称使用

`_` 用做被抛弃的名称。表示在后续的代码中，这个变量不会被使用到。



### 单下划线前缀（ `_XX` ）

单下划线做前缀的名称约定了这个名称是『私有』的，但是这样的实例变量是可以在外部访问到的，但是按照 Python 的约定，看到单下划线前缀这样的变量时，可以理解为『 虽然我可以在外部被访问，但是请把我当做一个内部变量（ `private` ）来使用，不要随意访问 』

Python 程序员约定使用一个下划线前缀编写『受保护』的属性，如 `self._x`。

Python 解释器不会对使用单个下划线的属性名做特殊处理，不过这是很多 Python 程序员严格遵守的约定，他们不会在类外访问这种属性。

> Attributes with a single _ prefix are called "protected" in some corners of the Python documentation. The pracitce of "protecting" attributes by convention with the form `self._x` is widespread, but calling that a "protected" attribute is not so common. Some even call that a "private" attribute.

单下划线前缀的通常被用于模块中，在一个模块中以单下划线开头的变量和函数被默认视为内部函数，使用 `from a_module import *` 导入时，任何以单下划线开头的名称都不会被导入。如果使用 `import a_module` 这样导入模块时，仍然可以使用 `a_module._var` 这样的方式来访问到对象。



### 双下划线前缀的名称 （ `__XX` ）

通常用在类中，如果想要使得类的内部属性不被外部随意访问到，可以把属性名字加上双下划线前缀，在 Python 中的实例变量如果以双下划线作为前缀，则表示改变量表示一个私有变量 （ `private` ），只有在内部可以访问，外部是无法访问的。

双下划线前缀命名的形式并不是一种惯例：对解释器而言，Python 会改写这些名称，以免与子类中定义的这些名称发生冲突。任何 `__method` 这种形式的标识符，都会被 Python 解释器在文本上替换为 `_classname__method`，`classname` 是所属类的名字。

也就是说，双下划线开头的实力变量并不是一定不能从外部进行访问的，仍然可以通过 `_classname__method` 形式访问到类中的内部变量 `__method` ，但是通常不建议这么做。



### 前后缀都为双下划线的名称（ `__XX__` ）

表示 Python 中的`特殊方法名`，这也只是一种惯例，表示 Python 系统的名称不会跟用户的自定义名称发生冲突。通常我们可以自己重写这些方法，比如在类里面经常会重写 `__init__()` 方法。

特殊变量是可以直接访问的，不是 `private` 变量。



###  参考

- [segmentfault](https://segmentfault.com/a/1190000002611411)
- [Python 的类的下划线命名有什么不同?](https://www.zhihu.com/question/19754941)

