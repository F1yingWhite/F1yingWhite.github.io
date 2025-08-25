---
title: RDF和SPARQL
description: ""
image: ""
published: 2025-03-11
tags:
  - 杂项
category: 杂项
draft: false
---

# RDF

关系型数据库是最流行的数据库，将数据抽象成行和列的表格，但是现实世界并不是表格而是网络，RDF 就是图数据库的一种描述形式，或者是一种协议，他使用三元组的方式描述事物之间的关系。RDF 要求事物之间的联系（谓语）必须有明确的定义。RDF 要求每套谓语必须有一个明确的 URL，通过 URL 区分不同的谓语。RDF 官方定义了一套常用的谓语，URL 如下。

```url
[https://www.w3.org/1999/02/22-rdf-syntax-ns](https://www.w3.org/1999/02/22-rdf-syntax-ns)
```

比如：小明是学生可以写成 RDF 三元组的形式：

```
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns>

小明 rdf:type 学生.

```

相同主语的描述可以合并

```txt
John_Lennon      a 艺术家 .
Paul_McCartney   a 艺术家 .
Ringo_Starr      a 艺术家 .
George_Harrison  a 艺术家 .
Please_Please_Me a 专辑 ;
                 :name "Please Please Me" ;
                 :date "1963" ;
                 :artist "甲壳虫" ;
                 :track Love_Me_Do .
Love_Me_Do       a Song ;
                 :name "Love Me Do" ;
                 :length 125 .
```

# SPARQL 查询语言

SPARQL 是 RDF 数据库的查询语言，跟 SQL 语法类似，核心思想为根据给定的谓语动词，从三元组提取符合条件的主语或宾语。

```sql
SELECT <variables>
WHERE {
   <graph pattern>
}
```

这里的 variables 就是需要提取的主语或宾语，graph pattern 就是要查询的三元组模式，比如我们要查询所有的专辑。

```sql
SELECT ?album
WHERE {
   ?album rdf:type :Album .
}
```
