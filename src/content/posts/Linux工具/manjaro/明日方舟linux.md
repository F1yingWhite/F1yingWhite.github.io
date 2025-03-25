---
title: 明日方舟linux
description: ""
image: ""
published: 2025-03-16
tags:
  - Linux
category: Linux
draft: false
---

https://jedsek.xyz/posts/other/linux-arknights/

```bash
sudo docker run -d --privileged  \
  -v ~/vms/redroid11:/data   \
  -p 5555:5555  \
  --name redroid11   \
  redroid/redroid:11.0.0-latest   \
  androidboot.redroid_width=1920   \
  androidboot.redroid_height=1080   \
  androidboot.redroid_gpu_mode=host   \
  androidboot.redroid_dpi=480 \
  androidboot.redroid_fps=120  \
  androidboot.use_memfd=true
```

随后再打开另外一个终端, 输入如下内容:

```bash
adb connect localhost:5555
scrcpy -s localhost:5555 --audio-codec=raw --print-fps -b 2048m --audio-bit-rate=2048m
```

在 [明日方舟官网](https://ak.hypergryph.com/) 下载安卓版本的安装包, 然后:

```bash
adb install ~/Downloads/arknights-hg-2221.apk
```

在 scrcpy 的显示窗口中, 我们按住鼠标左键从下往上拉, 在 `app-launcher` 界面点击 `明日方舟` 即可打开游戏并进行下载

如果是 1920x1080 的分辨率, 你可能看不到 `app-launcher`, 所以可以更改分辨率, 比如改成 4k :
