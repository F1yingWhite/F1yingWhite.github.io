---
title: docker入门
description: ""
image: ""
published: 2025-01-28
tags:
  - docker
category: Linux工具
draft: false
---

# 为什么要用 Docker

软件开发最大的一个问题就是如何配置环境,比如一个 python 应用,必须要 python,还有依赖,有时候还有环境变量.代码能在一台机器上跑,在其他机器上可能跑不了.

docker 是 linux 容器的一种封装,提供简单易用的容器接口,把应用程序与该程序的依赖打包在一个文件中, 运行这个文件,就会生成一个虚拟容器,程序在这个容器里运行,就好像在真实的物理机上运行.

docker 能够

1. 提供一次性的环境
2. 提供弹性的云服务.docker 可以随时开关,适合运维和缩容.
3. 组件微服务架构.通过多个容器,可以在本机模拟为服务架构.

# 创建一个基本镜像

docker 把应用程序和依赖打包在 image 文件里面,只有通过这个文件,才能生成 docker 容器,image 文件可以看做是容器的模版.docker 会根据 image 文件生成容器的实例,同一个 image 文件能够生成多个实例.

image 是一个二进制文件,实际开发中,一个 image 文件往往继承另一个 image 文件,加上一些个性化设置而成,比如可以在 ubuntu 的 image 上加入 apache 服务器.一般我们会在别人的 image 上加一些自己的定制,而不是全部自己制作,为了方便共享,我们可以把 image 文件发布到网上的仓库.

## Docker 镜像

使用 `docker search keyword` 搜索镜像

使用 `docker images` 查看当前所有的文件

使用 `docker pull [IMAGE_NAME]:[TAG]` 拉取镜像,默认使用 latest 版本

使用 `docker rmi [image]` 删除镜像

## Docker 容器

docker 启动容器有两种方式,

1. 基于镜像启动一个容器
2. 将一个终止的容器启动

### 基于镜像新建容器并启动

`docker run IMAGE:TAG`,举个例子🌰，比如想启动一个 `ubuntu` 容器, 并通过终端输出 `Hello world`：`docker run ubuntu:latest /bin/echo 'Hello world'`

这个命令在输出完后就会停止,如果想要交互的运行容器,需要执行 `docker run -t -i ubuntu:latest /bin/bash`,这里的 -t 是 docker 分配一个伪终端,-i 是让容器保持打开

在实际使用中,需要让容器以后台的方式运行,这里我们就需要添加 `-d` 参数来保持容器的运行.后台运行的容器可以使用 `docker logs` 来查看日志

### 查看容器

image 生成的容器实例本质上也是一个文件,被称为容器文件,也就是说容器一旦运行就会存在两个文件 image 和容器文件.关闭容器并不会删除容器文件,只是让容器停止运行.可以使用 `docker ps -a` 来查看所有的容器.想要删除容器文件可以使用 `docker rm [containerID]` 来删除指定的容器

### 进入容器

docker 可以使用 exec 命令进入容器,比如 `docker exec -it [name] /bin/bash`,这里退出容器的时候容器不会停止

### 导出容器

可以使用 `docker export` 导出容器,比如 `docker export 9e8d11aeef0c > redis.tar`,然后使用 `docker import` 可以将快照导入为新的容器.`cat redis.tar | docker import - test/redis:v1.0`

### 清除容器

使用 `docker rm [id/name]` 可以删除已经停止的容器,使用 `-f` 可以强制删除.

如果想要删除所有已经停止的容器,直接使用 `docker container prune` 即可

# 数据管理

## 什么是数据卷

简单来说数据卷就是一个可供一个或多个容器使用的特殊目录,用于持久化数据以及容器之间的共享,他以正常的目录存在于宿主机上,并且数据集不会随着容器被删除而被删除.

- 数据卷可以在容器之间共享和重用；
- 对数据卷的修改会立刻生效；
- 更新数据卷不会影响镜像；
- 数据卷默认一直存在，即使容器被删除；

## Volume 方式

docker 管理宿主机文件系统的一部分,默认位于/var/lib/docker/volumes 下,也是最常用的方式.

## Dockerfile 文件

学会使用 image 文件后,如何生成 image 文件?如果想要推广自己的软件,就必须要自己制作 image 文件.这就需要使用 dockerfile 文件,他是一个文本文件,用来配置 image,docker 根据该文件生成二进制的 image 文件.

Dockerfile 是一个被用来构建 Docker 镜像的文本文件，该文件中包含了一行行的指令（Instruction），这些指令对应着修改、安装、构建、操作的命令，每一行指令构建一层（layer），层层累积，于是有了一个完整的镜像

创建一个叫做 `Dockerfile` 的文件,写入

```dockerfile
FROM nginx
RUN echo '这是一个本地构建的nginx镜像' > /usr/share/nginx/html/index.html
```

这里的 from 是定制时需要的基础镜像,run 后面是所需要的命令

:::warning

**注意**：Dockerfile 的指令每执行一次都会在 docker 上新建一层。所以过多无意义的层，会造成镜像膨胀过大。例如：

```dockerfile
FROM centos
RUN yum -y install wget
RUN wget -O redis.tar.gz "http://download.redis.io/releases/redis-5.0.3.tar.gz"
RUN tar -xvf redis.tar.gz
```

以上执行会创建 3 层镜像。可简化为以下格式：

```dockerfile
FROM centos
RUN yum -y install wget \
    && wget -O redis.tar.gz "http://download.redis.io/releases/redis-5.0.3.tar.gz" \
    && tar -xvf redis.tar.gz
```

:::

### 构建镜像

在 dockerfile 目录下执行构建可以得到一个镜像

```bash
docker build -t nginx:v3 .
```

这里有个 `.`,是上下文路径,docker 在构建镜像的时候,有时候会使用到本地的文件,docker build 在得知这个路径后会将路径下的所有内容打包。

# Docker Compose

compose 是用于定义和运行多容器的 docker 应用程序的工具,通过 compose,可以用 yaml 文件来配置应用程序需要的所有服务,然后用一个命令就可以启动 yaml 中配置的所有服务.

1. 使用 dockerfile 定义环境
2. 使用 docker-compose.yml 定义应用程序的服务
3. 最后使用 docker-compose up 命令来启动并运行整个应用程序

