---
title: electron在wayland下的问题
description: ""
image: ""
published: 2025-03-10
tags:
  - Linux
  - wayland
category: wayland
draft: false
---
对于vscode和obsidain这样的election应用，在wayland下会出现中文输入法的问题，因此我们需要添加神秘代码 
```
Exec=/usr/bin/code --ozone-platform-hint=auto --enable-wayland-ime %F
```

对于chrome浏览器，我们需要额外添加代码：
在`~/.config/chrome-flags.conf`中添加
```
--ozone-platform=wayland
--enable-wayland-ime
--disable-features=WaylandFractionalScaleV1
```