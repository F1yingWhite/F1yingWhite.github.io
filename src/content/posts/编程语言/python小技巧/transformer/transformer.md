---
title: transformer
description: ""
image: ""
published: 2025-03-21
tags:
  - python
category: python
draft: false
---

```python
from transformers import Trainer, TrainerCallback
from pydantic import BaseModel
from typing import Optional

class ModelArguments(BaseModel):
    tokenizer_name: Optional[str] = "malteos/PubMedNCL"
    model_name: Optional[str] = None

class DataArguments(BaseModel):
    dataset_path: str
    max_seq_length: int = 128

class TrainingArguments(BaseModel):
    learning_rate: float = 5e-5
    batch_size: int = 32
#使用如下可以把命令行参数解析为预定义的类
parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
```
