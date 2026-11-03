---
id: default
name: Research 文献笔记
version: 1.0.0
author: F1yingWhite
description: 文献信息自动更新，个人理解保留，默认不公开。
contract: 8
language: liquid
filename: '{{ zt.citationKey | default: zt.key }}{% suffix %}'
frontmatter:
  - key: title
    expr: zt.title
    merge: keep
  - key: citekey
    expr: zt.citationKey
    merge: replace
  - key: authors
    expr: 'zt.authors | map: "fullName"'
    merge: replace
  - key: doi
    expr: zt.DOI
    merge: replace
  - key: source
    expr: zt.url
    merge: replace
  - key: type
    value: 文献笔记
    merge: keep
  - key: status
    value: 待读
    merge: keep
  - key: publish
    value: false
    merge: keep
  - key: description
    value: ""
    merge: keep
  - key: category
    value: 文献笔记
    merge: keep
  - key: tags
    value: [论文阅读]
    merge: keep
  - key: created
    expr: 'zt.dateAdded | date: "%Y-%m-%d"'
    merge: keep
---
# {{ zt.title }}

{% managed %}
## 文献信息

- 作者：{{ zt.authorsShort }}
- 年份：{{ zt.date.year }}
- [Zotero]({{ zt.backlink }}) {{ zt.attachments | map: "fileLink" | compact | join: " " }}
{% if zt.DOI %}- [论文来源](https://doi.org/{{ zt.DOI }}){% elsif zt.url %}- [论文来源]({{ zt.url }}){% endif %}

## 摘录和标注

```zotquery
views:
  - type: annotations
    name: 摘录和标注
    from: annotations
    filters: item == this.item
    sort:
      - property: sortIndex
        direction: ASC
```
{% endmanaged %}

## 一句话概括

## 研究问题

## 核心方法

## 关键证据

## 局限与疑问

## 与已有工作的关系

## 我的理解与后续实验

--- zotlit:annotation ---
{% bq %}
[!note] {% if zt.pageLabel and zt.pageLabel != "" %}Page [{{ zt.pageLabel }}]({{ zt.backlink }}){% else %}[Zotero]({{ zt.backlink }}){% endif %}

{{ zt.imgLink | embed }}{{ zt.text }}{% if zt.citation %} {{ zt.citation }}{% endif %}
{% if zt.comment %}

{{ zt.comment }}
{% endif %}
{% endbq %}
