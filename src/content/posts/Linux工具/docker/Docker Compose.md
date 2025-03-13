---
title: Docker Compose
description: ""
image: ""
published: 2025-02-18
tags:
  - Linux
  - docker
category: Linux工具
draft: false
---

https://docs.docker.com/compose/gettingstarted/#step-2-define-services-in-a-compose-file

# 简介

docker compose 使用 `compose.yaml` 或者 `docker-compose.yaml`(早期) 文件来定义

## CLI

docker 允许你使用 `docker compose` 命令来管理 compose.yaml 文件中定义的各种容器的生命周期

```shell
docker compose up#启动任务
docker compose down#停止任务
docker compose ps#查看任务的情况
```

## 为什么要用 Docker Compose

- 简化控制，docker compose 允许在单个 yaml 文件中定义和管理多个容器
- 易于共享
- compose 会缓存容器的配置，当启动未跟改的服务的时候，compose 会复用现有的容器
- 可移植性好

# 快速开始

# Docker Compose

## 初始化

compose 是用于定义和运行多容器的 docker 应用程序的工具,通过 compose,可以用 yaml 文件来配置应用程序需要的所有服务,然后用一个命令就可以启动 yaml 中配置的所有服务.

1. 使用 dockerfile 定义环境
2. 使用 docker-compose.yml 定义应用程序的服务
3. 最后使用 docker-compose up 命令来启动并运行整个应用程序
4. 创建一个 flask 应用,使用 redis 计算 count 数
5. 写依赖文件 `requirements.txt`
6. 创建 `Dockerfile`

```yaml
# syntax=docker/dockerfile:1
FROM python:3.10-alpine
WORKDIR /code
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
RUN apk add --no-cache gcc musl-dev linux-headers
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000
COPY . .
CMD ["flask", "run", "--debug"]
```

>[!TIP]
>- 创建 py310 镜像
>- 设置工作目录为/code
>- 设置一些 flask 环境变量
>- 复制 `requirements.txt` 并且下载依赖
>- 拷贝当前目录到工作目录
>- 运行

## 在 Compose 中定义服务

compose 文件简化了应用堆栈的控制

创建 `compose.yaml` 文件

```yaml
services:
  web:
    build: .
    ports:
      - "8000:5000"
  redis:
    image: "redis:alpine"
```

这里定义了两个服务，web 和 redis，web 服务在当前目录构建 dockerfile 并且吧 8000 端口映射到 5000 上（flask 默认端口）。

## 运行 Compose 服务

在当前目录使用 `docker compose up` 命令直接运行

## 使用 Compose Watch

编辑项目目录中的 `compose.yaml` 文件，使用 `watch` 功能，以便您可以预览正在运行的 Compose 服务。这些服务会在您编辑并保存代码时自动更新：

```yaml
services:
  web:
    build: .
    ports:
      - "8000:5000"
    develop:
      watch:
        - action: sync
          path: .
          target: /code
  redis:
    image: "redis:alpine"
```

每当文件发生更改时，Compose 会将文件同步到容器内 `/code` 下的相应位置。文件复制完成后，捆绑器会更新正在运行的应用程序，而无需重新启动。

>[!Note]
>这里使用了 `--debug` 来让 flask 在.py 文件更改的时候自动更新

## 使用 Watch 来运行 Docker Compose

现在，使用 `docker compose wath` 或者 `docker compose up --watch` 来重启服务

## 拆分 Services

可以同时撰写多个 compose file 文件来定义复杂的工作流

1. 在工作目录下创建 `infra.yaml` 文件
2. 从 `compose.yaml` 目录中拷贝 redis service 到 `infra.yaml` 文件中，确保在文件顶部添加了 `service`

```yaml
services:
	redis:
		image: "redis:alpine"
```

3. 在 `compose.yaml` 中在顶部添加 include

```yaml
include:
   - infra.yaml
services:
  web:
    build: .
    ports:
      - "8000:5000"
    develop:
      watch:
        - action: sync
          path: .
          target: /code
``` 
