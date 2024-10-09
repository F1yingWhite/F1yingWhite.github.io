---
title: uv使用
published: 2024-09-18
description: '一起来替代conda吧！'
image: ''
tags: [python]
category: 'python'
draft: false
lang: ''
---
:::warning
注意,uv的使用大多需要科学上网,如果服务器没有魔法最好不要使用
:::

## Install
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh # macos/linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex" # win
```
安装完毕后使用`uv`命令查看是否安装正确，可以使用`uv python install`来安装python的最新版本,或者使用`uv python install 3.12`等，可以使用`uv python list`来查看目前的python
:::important
通过uv安装的python不会自动激活，需要使用uv run或者创建并激活一个虚拟环境来使用python
:::
## Run scripts
使用`uv run *.py`来运行特定的python文件,如果运行的python文件存在依赖关系，直接运行会报错，uv 更喜欢按需创建这些环境，而不是使用具有手动管理依赖项的长期虚拟环境。这需要显式声明脚本所需的依赖项，比如
```bash
uv run --with rich test.py # 单个依赖
uv run --with rich --with opencv-python .\1.py #可以使用多个--with来确定多个依赖
uv run --python 3.10 test.py #指定python版本
```
所以单个文件的时候还是老老实实用普通的python吧！

## using tools
很多python的包提供了applicaiton可以当作工具被使用,比如ruff静态代码检查
```bash
uvx ruff # 自动下载依赖
uv tool run ruff #上下等价,uvx is provided as an alias for convenience.
uv tool install ruff
uv tool upgrade ruff
```

## Working on projetcts
uv项目的依赖和rust类似，都存放在pyproject.toml文件下
### Creating a new project
```bash
uv init hello-world
# 或者
cd helloworld
uv init
```
uv会自动创建如下文件：
```
.
├── .python-version
├── README.md
├── hello.py
└── pyproject.toml
```
然后`uv run hello.py`就可以运行啦！
#### pyproject.toml
pyproject.toml介绍了元数据
```toml
[project]
name = "hello-world"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
dependencies = []
```
:::tip
关于pyproject.toml的更多信息可以在[官方文档中](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)查看
:::
#### .python-version
包含了python的版本
#### .venv
包含了虚拟环境
#### uv.lock
是一个跨平台锁文件，其中包含有关项目依赖关系的确切信息。与用于指定项目广泛需求的 pyproject.toml 不同，lockfile 包含安装在项目环境中的精确解析版本。这个文件应该签入到版本控制中，允许跨计算机进行一致和可重复的安装。是一个人类可读的 TOML 文件，但是由 uv 管理，不应手动编辑。

### Managing dependencies
可以使用`uv`在`pyproject.toml`中添加依赖，这会自动更新lockfire和venv
```bash
uv add requests
uv remote requests
```

## pip interface
uv 为常见的 pip、 pip-tools 和 viralenv 命令提供了一个插件替换。这些命令直接与虚拟环境一起工作，与 uv 的主要接口不同，uv 的主要接口是自动管理虚拟环境的
### python environments
uv支持创建虚拟环境，会把环境写在.venv中
```bash
uv venv
uv venv my-name --python=3.10
uv pip install ruff
```
如果需要激活环境
```bash
.venv\Scripts\activate # win
source .venv/bin/activate # linux/mac
```
使用uv下载package的方法和pip一样，只要前面加个uv就好了