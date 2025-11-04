---
zotero-key: XMIY94KJ
zt-attachments:
  - "179"
title: Visual Instruction Tuning
citekey: liuVisualInstructionTuning2023
published: 2025-11-04 10:15:05
updated: 2025-11-04 10:31:36
---

# Visual Instruction Tuning

[Zotero](zotero://select/library/items/XMIY94KJ) [attachment](<file:///Users/flyingwhite/Zotero/storage/DX23DRJQ/Liu%20et%20al.%20-%202023%20-%20Visual%20Instruction%20Tuning.pdf>)

![image.png](https://picture-bed-1325530970.cos.ap-nanjing.myqcloud.com/20251104102948876.png)

主要是使用 CLIP 的 ViT 来当 visual encoder,然后使用一个投影层吧 ViT 的输出映射到 LLM 的 embedding 空间中。

其中训练氛围两步

1.  冻结 ViT 和 LLM，只训练投影层，用来对其特征空间
2. 训练投影层和 LLM
