---
title: Tmux
published: 2024-09-08
description: ""
image: ""
tags:
  - Linux
category: Linux工具
draft: false
---

## 基本概念

tmux采用C/S模型构建，输入tmux命令就相当于开启了一个服务器，此时默认将新建一个会话，然后会话中默认新建一个窗口，窗口中默认新建一个面板。会话、窗口、面板之间的联系如下：

一个tmux `session`（会话）可以包含多个`window`（窗口），窗口默认充满会话界面，因此这些窗口中可以运行相关性不大的任务。

一个`window`又可以包含多个`pane`（面板），窗口下的面板，都处于同一界面下，这些面板适合运行相关性高的任务，以便同时观察到它们的运行情况。

## 会话session

### 新建会话

语法为`tmux new -s session-name`，也可以简写为`tmux`

```shell
tmux #创建一个无名称的会话
tmux new -s demo #新建一个叫demo的会话
```

### 断开会话

会话中操作了一段时间，我希望断开会话同时下次还能接着用，怎么做？此时可以使用detach命令。

```shell
tmux detach
```

除了使用命令,我们也可以直接使用tmux自带的快捷键组合`Ctrl(control)+b` + `d`，三次按键就可以断开当前会话。

### 进入之前的会话

断开会话后，想要接着上次留下的现场继续工作，就要使用到tmux的attach命令了，语法为`tmux attach-session -t session-name`，可简写为`tmux a -t session-name` 或 `tmux a`。通常我们使用如下两种方式之一即可：

```shell
tmux a # 默认进入第一个会话
tmux a -t demo # 进入到名称为demo的会话
```

#### 关闭会话

会话的使命完成后，一定是要关闭的。我们可以使用tmux的kill命令，kill命令有`kill-pane`、`kill-server`、`kill-session` 和 `kill-window`共四种，其中`kill-session`的语法为`tmux kill-session -t session-name`。如下：

```shell
tmux kill-session -t demo # 关闭demo会话
tmux kill-server # 关闭服务器，所有的会话都将关闭
```

### 查看所有会话

```shell
tmux ls#查看所有的会话
```

## 窗口操作
ctrl+b c=>新的窗口
ctrl+b p=>上一个窗口
ctrl+b n=>下一个窗口
ctrl+b w=>选择窗口

tmux split-window / ctrl+b % =>划分为上下两个
tmux split-window -h / ctrl+b " =>划分为左右
ctrl+d 关闭窗口

## 自定义配置
安装tqm启动插件后在~/.config/tmux/tmux.conf开启配置

```conf
unbind C-b
set -g prefix M-b  # 将前缀设置为Alt + b
bind C-a send-prefix

bind -n M-Left select-pane -L
bind -n M-Right select-pane -R
bind -n M-Up select-pane -U
bind -n M-Down select-pane -D
# bind -n M-p previous-window
# bind -n M-n next-window
```