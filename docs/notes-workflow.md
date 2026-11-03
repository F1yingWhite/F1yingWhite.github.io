# 笔记与博客工作流

日常笔记保存在独立的 Obsidian 库，博客仓库只接收明确选择过的公开副本。Zotero 管理文献、原始 PDF 和标注，ZotLit 在 Obsidian 中提供文献库、PDF 阅读和标注入口。

## 在 Obsidian 中使用

1. 在 Research 库打开 `00 首页`，按主题进入笔记。
2. 文献笔记可使用 `ZotLit: Open PDF` 打开 Zotero 的原始附件，在右侧记录自己的理解。
3. 整理好要公开的正文、摘要、分类和标签后，执行 `Research Blog Publisher: 选择当前笔记发布`，确认固定的网址路径。
4. 打开左侧“博客发布”，检查选中的文章和公开文件。点击“导出并预览”，在浏览器确认结果。
5. 点击“提交并部署…”，确认后才会提交公开副本并推送到 GitHub。

选择笔记本身不会发布。取消选择会停止后续导出，但不会撤下已经存在的线上文章。迁移前已经公开的文章保留原始快照，只有重新选择才会以整理后的笔记更新它。

## 属性约定

```yaml
title: 笔记标题
note-id: 一个固定的 UUID
publish: false
slug: 一个固定的文章路径
description: 给读者看的简短摘要
category: 多模态
tags: [论文阅读, 多模态]
status: 整理中
```

`publish` 必须是布尔值 `true` 才会选择导出，字符串 `"true"` 不生效。文献笔记以 `zotero-key` 为身份，普通笔记使用 `note-id`；移动文件或改标题不会改变其身份。公开过的 `slug` 不能隐式改变。

首次发布可用 `blog-date: YYYY-MM-DD` 指定文章日期，否则使用首次导出日期。后续发布保持原日期。`created` 表示笔记创建日期，`updated` 表示笔记修订日期。

ZotLit 模板将自动内容放在 managed 区域，自己的理解写在区域外。迁移来的旧笔记保留原正文；更新元信息可以使用 `Update literature note metadata`。不要对有手写内容的旧笔记直接执行整篇覆盖。

## 导出规则

- Markdown 和 Obsidian 图片会复制到 `public/notes-assets`，远程图片会下载成固定资源。公开资源名由图片内容哈希生成。
- 双链和相对 Markdown 笔记链接转换到已经公开或本次明确选择的文章。未公开目标会阻止导出，引用它不会自动公开它。
- Zotero 和本机附件链接转换为公开的论文来源。没有公开来源、仍含本机路径或未解析引用标记的笔记会停止导出。
- Obsidian 私密注释、HTML 注释和动态查询块不进入公开副本。代码示例中的普通双链仍作为代码保留。
- `[@citekey]` 引用从文献笔记元信息生成链接和参考文献，不会导出被引用笔记的私人正文。
- 模板、插件配置、缓存、原始 PDF 和未选择的笔记均不进入博客。已有公开副本不会因为源笔记被移动、删除或取消选择而自动撤下。

## 命令行

本机设置放在不受 Git 跟踪的 `.notes-publish.local.json`：

```json
{ "vault": "/你的/Obsidian/Research" }
```

也可通过 `NOTES_VAULT` 环境变量或 `--vault` 指定库。

```sh
pnpm new-post '新笔记标题'         # 在私有收件箱创建，默认不公开
pnpm notes:list                 # 本机笔记列表
pnpm notes:plan                 # 只检查，不修改公开文件
pnpm notes:publish --write       # 导出选择的笔记，不推送
pnpm build                      # 校验公开快照并构建
pnpm notes:deploy --confirmed    # 检查、构建、提交公开文件并推送
pnpm notes:unpublish --slug '文章路径'         # 撤稿预览
pnpm notes:unpublish --slug '文章路径' --write # 明确撤下公开副本，随后部署
```

部署只提交公开文章、公开图片和发布清单；暂存区存在其他改动时会停止。源笔记不参与 Git 提交。公开副本如果被手工编辑，校验会停止覆盖，避免丢失修改。

## 安装集成

`integrations/obsidian-blog-publisher` 是本仓库自带的 Obsidian 插件，将其内容放入私有库的 `.obsidian/plugins/research-blog-publisher`，启用后在设置中填写仓库和 Node.js 的本机路径。

`integrations/zotlit/zotlit-profile.default.md` 是 ZotLit 2.2 的文献模板，可放入配置的 ZotLit 模板文件夹。新文献默认进入待读目录，默认不公开；旧笔记不自动覆盖。

在 Obsidian 中修改 Zotero 标注需要 Zotero 10 保持运行、开启本机通信，并由你在 Zotero 中批准 ZotLit 编辑权限。同一条标注在一个应用中编辑、保存后再切换。

## 迁移后的同步状态

原库完整备份后已迁入独立 Research 库。新库暂未启用 Remotely Save，旧远端配置保留，避免直接将目录重组解释为云端批量删除；跨设备同步需要先核对远端目标。博客发布与笔记库同步各自独立。
