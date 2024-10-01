---
title: argparse
published: 2024-10-01
description: 'python解析命令行输入'
image: ''
tags: [python]
category: 'python'
draft: false
---
## argparse

该命令的使用方式如下：
```python
import argparse

parser = argparse.ArgumentParser(description='Patch extraction for WSI')
parser.add_argument('-d', '--dataset', type=str, default='TCGA-lung', help='Dataset name')
parser.add_argument('-e', '--overlap', type=int, default=0, help='Overlap of adjacent tiles [0]')
parser.add_argument('-f', '--format', type=str, default='jpeg', help='Image format for tiles [jpeg]')
parser.add_argument('-v', '--slide_format', type=str, default='svs', help='Image format for tiles [svs]')
parser.add_argument('-j', '--workers', type=int, default=4, help='Number of worker processes to start [4]')
parser.add_argument('-q', '--quality', type=int, default=70, help='JPEG compression quality [70]')
parser.add_argument('-s', '--tile_size', type=int, default=224, help='Tile size [224]')
parser.add_argument('-b', '--base_mag', type=float, default=20, help='Maximum magnification for patch extraction [20]')
parser.add_argument('-m', '--magnifications', type=int, nargs='+', default=(0,), help='Levels for patch extraction [0]')
parser.add_argument('-o', '--objective', type=float, default=20, help='The default objective power if metadata does not present [20]')
parser.add_argument('-t', '--background_t', type=int, default=15, help='Threshold for filtering background [15]')  
args = parser.parse_args()
print(args.magnifications)
```
其中，使用-d或者--dataset；进行输入。dataset是名称.后续使用的时候就可以直接`python 1.py --d 114`
