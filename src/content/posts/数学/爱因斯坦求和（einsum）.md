---
title: 爱因斯坦求和(enisum)
published: 2024-09-27
description: ''
image: ''
tags: [数学]
category: '数学'
draft: false 
---


> 在数学中，特别是将线性代数套用到物理中时，爱因斯坦求和约定是一种标记的约定，又称爱因斯坦标记法，在处理坐标的方程中特别有用

## einsum记法
在pytorch中的点积/外积/转置/矩阵乘法都是可以用einsum记法来表示的，包括复杂张量运算在内的优雅方式。会熟练运用einsum后，可以不用看库函数，还可以更加迅速的编写紧凑，高效的代码，可忽略不必要的张良转置和变形，甚至还可以忽略生成中间张量。事实上，pytorch引入的能够自动生成GPU代码并且为特定输入尺寸自动调节的张量理解就类似于einsum领域的特定语言。

比如我们有一个矩阵
$
A_{2 \times 2} = \begin{pmatrix}
1 & 2 \\
3 & 4 
\end{pmatrix}
$
我们想对A的行求和，用公式表示就是
$
B_i =\sum_jA_{ij}=\begin{pmatrix}
    3\\7
\end{pmatrix}
$
对于这个求和符号，爱因斯坦说看着多余，就省略了，于是式子就变成了$B_i=A_{ij}$,在torch中就可以标记为`torch.einsum("ij->i",a)`这样就等于上面的矩阵了

## 在numpy和pytorch中的用法
einsum在numpy中叫`numpy.einsum`，torch中叫`torch.einsum`,使用一致的函数签名`einsum(euqation,operands)`

比如我们有如下式子
```python
A = torch.Tensor(range(2*3*4)).view(2,3,4)
C = torch.einsum("ijk->jk",A)
```
他就等于$c_{jk}=A_{ijk}$,我们再补充\sum符号，那么就得到了
$$
    c_{jk}=\sum_i{A_{ijk}}
$$
也就是把第一维度合起来了，相当于把长方形拍平,但是用for循环写就难啦~
```python
i, j, k = A.shape[0], A.shape[1], A.shape[2] # 得到 i, j, k
C_ = torch.zeros(j, k) # 初始化 C_ , 用来保存结果
for i_ in range(i): # 遍历 i
    for k_ in range(k): # 遍历 j
        for j_ in range(j): # 遍历 k
            C_[j_][k_] += A[i_][j_][k_] # 求和
```
也就是说，我们需要对缺少的那几个字母添加\sum符号,比如$B_{ji}=A_{ij}$,他左右符号一致，所以不用添加\sum，所以就是个转置的过程

下面是一个难一点的
```python
A = torch.Tensor(range(2*3*4*5)).view(2, 3, 4, 5)
B = torch.Tensor(range(2*3*7*8)).view(2, 3, 7, 8)
C = torch.einsum("ijkl,ijmn->klmn", A, B)
```
> 如果等式右边有多个矩阵，则用逗号分割

对于这个式子一眼看过去应该是一个矩阵乘法,我们先写出表达式$C_{klmn}=A_{ijkl}B_{ijmn}$,然后添加\sum,也就是$C_{klmn}=A_{ijkl}B_{ijmn}$,然后添加缺少的ij,得到
$$
  C_{klmn}=\sum_{i}\sum_{j}A_{ijkl}B_{ijmn}
$$
>注意这个是数字相乘

```python
i,j,k,l,m,n = A.shape[0],A.shape[1],A.shape[2],A.shape[3],B.shape[2],B.shape[3]
C_ = torch.zeros(k,l,m,n)
for i_ in range(i):
    for j_ in range(j):
        for k_ in range(k):
            for l_ in range(l):
                for m_ in range(m):
                    for n_ in range(n):
                        # 由于有求和符号，所以用+=
                        C_[k_][l_][m_][n_] += A[i_][j_][k_][l_]*B[i_][j_][m_][n_]
```