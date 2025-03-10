---
title: module
description: ""
image: ""
published: 2025-03-04
tags:
  - python
  - pytorch
category: " pytorch"
draft: false
---

# 模型参数统计

```python
def print_trainable_params(model):
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        num_params = param.numel()#统计参数数量

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param
```

# 模型的保存于加载

```python
def _save_checkpoint(model, optimizer, cur_epoch, args, is_best=False):
    """
     能够最大限度的减少存储消耗,只保留需要训练的参数，在使用预训练模型的时候非常好用
    """
    os.makedirs(f"{args.output_dir}/{args.dataset}", exist_ok=True)

    param_grad_dic = {k: v.requires_grad for (k, v) in model.named_parameters()}
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "config": args,
        "epoch": cur_epoch,
    }
    path = f"..."
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, path))
    torch.save(save_obj, path)

def _reload_best_model(model, args):
    """
    Load the best checkpoint for evaluation.
    """
    checkpoint_path = f"..."

    print("Loading checkpoint from {}.".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    return model

def _reload_model(model, checkpoint_path):
    """
    Load the best checkpoint for evaluation.
    """

    print("Loading checkpoint from {}.".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    return model
```

# 参数获取

## model.named_parameters()

```python
params = [p for _, p in model.named_parameters() if p.requires_grad]
for name, parameter in model.named_parameters(): 
	print(name, parameter.requires_grad)
```

## model.parameters()

迭代打印 model.parameters() 将会打印每一次迭代元素的 param 而不会打印名字，这是它和 named_parameters 的区别，两者都可以用来改变 requires_grad 的属性。

 下面这段代码加载了一个预训练模型并且冻结了非线性层

```python
weights_dict = torch.load(weights_path, map_location=device)
load_weights_dict = {k: v for k, v in weights_dict.items() if 'fc' not in k}
model.load_state_dict(load_weights_dict, strict=False)

# 冻结特征提取层参数
for name, param in model.named_parameters():
	param.requires_grad = False if 'fc' not in name else True
pg = [p for p in model.parameters() if p.requires_grad]#注意这里只返回了参数而没有返回参数名！
optimizer = torch.optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=4E-5)
```

## model.state_dict()

pytorch 中的 state_dict 是一个简单的 python 的字典对象,将每一层与它的对应参数建立映射关系.(如 model 的每一层的 weights 及偏置等等)

>[!tip]
>（1）只有那些参数可以训练的 layer 才会被保存到模型的 state_dict 中,如卷积层,线性层等等，像什么池化层、BN 层这些本身没有参数的层是没有在这个字典中的；
>
>（2）这个方法的作用一方面是方便查看某一个层的权值和偏置数据，另一方面更多的**是在模型保存**的时候使用。

```python
import torch

class net(torch.nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

n = net()

print(type(n.state_dict()))#<class 'collections.OrderedDict'>
for k, v in n.state_dict().items():
    print(k, v.size())
#fc.weight torch.Size([1, 10])
#fc.bias torch.Size([1])
```
