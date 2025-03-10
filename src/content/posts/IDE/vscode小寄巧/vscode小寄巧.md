---
title: vscode小寄巧
published: 2024-08-09
description: ''
image: ''
tags: []
category: 'IDE'
draft: false 
---
## 主题

我个人喜欢用的主题为ayu-light-border和github-dark

## 字体

设置字体为jetbrains-mono-nerd并且开启连字.可以设置字体粗细

## 光标


## Python
### pylance

pylance是个非常司马又非常好用的插件,如果代码提示`Enumeration of workspace source files is taking a long time. Consider opening a sub-folder instead. Learn more`,使用如下方法解决[链接](https://github.com/microsoft/pyright/blob/main/docs/configuration.md#sample-config-file)原因为当前目录下又太多的文件,索引扫描非常慢,在当前目录下创建pyrightconfig.json,然后输入需要忽略的文件夹即可.本文中使用的格式化工具为black
```json
{
    "exclude": [
        "assets",
        "./data",
        "checkpoints",
        ".idea",
        "**/__pycache__",
        "logs"
    ]
}
```

### isort
这个插件用来对import进行自动排序,通过isort+pylance能实现自动排序的效果,在setting.json中设置

```json
{
  "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": "always"
        },
  },
  "isort.args": [
      "--profile",
      "black"
  ],
  "black-formatter.args": [
      "--line-length=200",
  ],
}
```

