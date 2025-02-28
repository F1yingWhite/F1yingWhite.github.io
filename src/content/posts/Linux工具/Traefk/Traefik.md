---
title: Trafik
description: ""
image: ""
published: 2025-02-18
tags:
  - Linux
  - web
category: " Linux"
draft: false
---

https://doc.traefik.io/traefik/

# 介绍

## 边路由

Traefik 是边路由，这意味着他是平台的门，并且它拦截和路由每个传入的请求：它知道所有逻辑和确定哪些服务处理的所有规则（基于路径，主机，标题等）。

![](https://doc.traefik.io/traefik/assets/img/traefik-concepts-1.png)

## 自动路由发现

传统的反向代理服务器需要一个配置文件来给出每一个服务的可能路由，而 Traefik 有 services **自己获取**

# 使用 Docker 部署 Traefik

创建如下的 compose.yaml 文件并使用 `docker compose up -d reverse-proxy` 启动服务

```yaml
version: '3'

services:
  reverse-proxy:
    # The official v3 Traefik docker image
    image: traefik:v3.3
    # Enables the web UI and tells Traefik to listen to docker
    command: --api.insecure=true --providers.docker
    ports:
      # The HTTP port
      - "80:80"
      # The Web UI (enabled by --api.insecure=true)
      - "8080:8080"
    volumes:
      # So that Traefik can listen to the Docker events
      - /var/run/docker.sock:/var/run/docker.sock
```

然后可以查看 `http://localhost:8080/api/rawdata` 来看元数据

## 发现并创建新的服务

现在我们在 compose.yaml 中添加如下新的服务

```yaml
services:
  ...
  whoami:
    # A container that exposes an API to show its IP address
    image: traefik/whoami
    labels:
      - "traefik.http.routers.whoami.rule=Host(`whoami.docker.localhost`)"
```

然后使用 `docker compose up -d whoami`，Traefik 会自动发现这个服务，然后使用 `curl -H Host:whoami.docker.localhost http://127.0.0.1` 可以得到信息

## 自动均衡负载

现在我们开启了多个 instance，Traefik 会自动发现他们并且在他们之间进行负载均衡

`docker compose up -d --scale whoami=2`

现在我们来仔细看看一个 compose

```yaml
services:

  traefik:
    image: "traefik:v3.3"
    container_name: "traefik"
    command:
      #- "--log.level=DEBUG"
      - "--api.insecure=true"#用于在 8080 端口查看配置
      - "--providers.docker=true"#获取来自 docker 中的信息
      - "--providers.docker.exposedbydefault=false"
      - "--entryPoints.web.address=:80"#打开并接受 http 流量
    ports:
      - "80:80"
      - "8080:8080"
    volumes:
      - "/var/run/docker.sock:/var/run/docker.sock:ro"

  whoami:
    image: "traefik/whoami"
    container_name: "simple-service"
    labels:
      - "traefik.enable=true"#暴露这个容器
      - "traefik.http.routers.whoami.rule=Host(`whoami.localhost`)"#
      - "traefik.http.routers.whoami.entrypoints=web"#只允许来自web端点的流量
```

# HTTPS & TLS