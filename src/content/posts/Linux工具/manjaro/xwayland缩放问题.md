---
title: xwayland缩放问题
description: ""
image: ""
published: 2025-03-10
tags:
  - Linux
  - wayland
category: wayland
draft: false
---
对与非整数倍缩放，wayland会出现xwayland应用模糊的问题，这个时候我们需要添加神秘代码：
在`hyprland.conf`中添加如下代码：
```
xwayland { force_zero_scaling = true }
env = GDK_SCALE,2
env = XCURSOR_SIZE,32
```
对于微信添加如下代码
```bash
Exec=env QT_IM_MODULE=fcitx QT_SCALE_FACTOR=1.5 'QT_QPA_PLATFORM=wayland;xcb' /opt/wechat/wechat %U
```
这个方法适用于两块屏幕dpi一致的场景

https://bbs.archlinuxcn.org/viewtopic.php?id=14839
新的办法：`sudo nvim ~/.Xresources`添加如下
Xft.dpi: 120
执行`xrdb -merge ~/.Xresources`即可改变所有的xwayland应用的缩放