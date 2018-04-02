---
title: conda下安装caffe2踩坑
date: 2018-04-02 21:45:16
categories: 踩坑
tags:
  - caffe2
  - 搞机指南
---

最近在GPU服务器上安装 `caffe2` 遇到了一些问题，这里把踩坑的过程跟解决的方法记录下来。

<!--more--> 

#### System Information

- Operating System: Ubuntu 14.04.4

  > kwu@Ubuntu:~$ cat /proc/version
  > Linux version 4.4.0-31-generic (buildd@lgw01-43) (gcc version 4.8.4 (Ubuntu 4.8.4-2ubuntu1~14.04.3) ) #50~14.04.1-Ubuntu SMP Wed Jul 13 01:07:32 UTC 2016

- cuda version: 8.0.61

  > kwu@Ubuntu:~$ cat /usr/local/cuda/version.txt
  > CUDA Version 8.0.61

- cudnn versino: 7.1.1

  > kwu@Ubuntu:~$ cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
  >
  > define CUDNN_MAJOR 7
  >
  > define CUDNN_MINOR 1
  >
  > define CUDNN_PATCHLEVEL 1
  >
  > --
  >
  > define CUDNN_VERSION    (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)
  >
  > include "driver_types.h"

- gcc version: 4.8.4

  > kwu@Ubuntu:~$ gcc --version
  > gcc (Ubuntu 4.8.4-2ubuntu1~14.04.4) 4.8.4
  > Copyright (C) 2013 Free Software Foundation, Inc.


此外还需要安装 [nccl2](https://developer.nvidia.com/nccl) ，选择合适的版本。我选择的是 `nccl-repo-ubuntu1404-2.1.15-ga-cuda8.0_1-1_amd64.deb`

### 踩坑之旅

看到上面的系统信息，很自然的在 [caffe.ai](https://caffe2.ai/docs/getting-started.html?platform=ubuntu&configuration=compile) 上选择

> conda install -c caffe2 caffe2-cuda8.0-cudnn7-gcc4.8

下面是 `caffe2` 的文档说明的关于 `gcc` 版本不同的命令格式：

> If your gcc version is older than 5 (less than 5) (you can run `gcc --version` to find out), then append ‘-gcc4.8’ to the package name. For example, run `conda install -c caffe2 caffe2-gcc4.8` or `conda install -c caffe2 caffe2-cuda9.0-cudnn7-gcc4.8` instead of the commands below.

由于国内 `anaconda` 默认源访问速度的原因，我这里的 `anaconda` 选择的是 [tuna.tsinghua.edu.cn](https://mirror.tuna.tsinghua.edu.cn/help/anaconda/) 的安装源。

出现的问题：

> from caffe2.python import core
> WARNING:root:This caffe2 python run does not have GPU support. Will run in CPU only mode.
> WARNING:root:Debug message: /home/kwu/anaconda3/lib/python3.6/site-packages/caffe2/python/../../../../libcaffe2_protos.so: undefined symbol:_ZNK6google8protobuf7Message11GetTypeNameEv
> CRITICAL:root:Cannot load caffe2.python. Error: /home/kwu/anaconda3/lib/python3.6/site-packages/caffe2/python/caffe2_pybind11_state.cpython-36m-x86_64-linux-gnu.so: undefined symbol: _ZN6google8protobuf8internal26fixed_address_empty_stringE

### 解决

首先是 `gcc` 版本的原因

> kwu@Ubuntu-P100:~$ python
> Python 3.6.4 |Anaconda, Inc.| (default, Jan 16 2018, 18:10:19)
> [GCC 7.2.0] on linux
> Type "help", "copyright", "credits" or "license" for more information.
>
> \> \> \>

这里看到，`Anaconda libraries have been built with gcc 5+`

所以我们选择的 `conda` 命令应该是

> conda install -c caffe2 caffe2-cuda8.0-cudnn7

随后发现，问题依旧存在。

查看 `caffe2` 在  `github` 上的 `issues` 发现`@pjh5`  已经解答过这个问题了。[#1980](https://github.com/caffe2/caffe2/issues/1980)

我们需要把 `tuna.tsinghua.edu.cn` 的源给换成 `default` 源。

原因在这里：

>  so the error before is that some packages are compiled using gcc < 5 in Tsinghua channel

当然了，`default` 存在的问题就是访问速度过慢，有的 `conda` 包下载不下来，这个时候就需要记下所需的 `conda` 包的版本号，去 `Anaconda` 的官网手动下载，再使用`conda install --use-local packages ` 安装即可。

比如我这里是

- `ffmpeg-3.4-h7264315_0.tar.bz2`
- `mkl-2018.0.2-1.tar.bz2`
- `libprotobuf-3.4.1-h5b8497f_0.tar.bz2`
- `opencv-3.3.1-py36h6cbbc71_1.tar.bz2`

四个第三方包下载不下来。

本地安装

- `conda install --use-local ffmpeg-3.4-h7264315_0.tar.bz2 `
- `conda install --use-local mkl-2018.0.2-1.tar.bz2`
- `conda install --use-local libprotobuf-3.4.1-h5b8497f_0.tar.bz2`
- `conda install --use-local opencv-3.3.1-py36h6cbbc71_1.tar.bz2`

就可以正常使用 `caffe2` 了。