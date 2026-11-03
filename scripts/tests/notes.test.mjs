import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { stringify } from "yaml";
import { buildPlan, applyPlan, digest, validSlug } from "../lib/notes.mjs";

async function fixture(t) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "notes-test-"));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const repo = path.join(root, "blog"), vault = path.join(root, "vault");
  const config = { repo, vault, output: "src/content/posts", assets: "public/notes-assets", manifest: "notes-publication.json", outputPath: path.join(repo, "src/content/posts"), assetsPath: path.join(repo, "public/notes-assets"), manifestPath: path.join(repo, "notes-publication.json") };
  await fs.mkdir(config.outputPath, { recursive: true }); await fs.mkdir(vault);
  await fs.writeFile(config.manifestPath, '{"version":1,"entries":{}}\n');
  const note = async (relative, data, body = "正文。\n") => { const file = path.join(vault, relative); await fs.mkdir(path.dirname(file), { recursive: true }); await fs.writeFile(file, `---\n${stringify({ title: relative, "note-id": relative, slug: relative.replace(/\.md$/, ""), ...data })}---\n\n${body}`); return file; };
  return { config, note };
}
test("only boolean true selects a note, and a dry run leaves public snapshots unchanged", async (t) => {
  const { config, note } = await fixture(t);
  await note("a.md", { publish: false }); await note("b.md", { publish: "true" }); await note("c.md", {});
  const existing = Buffer.from("existing public body"); await fs.writeFile(path.join(config.outputPath, "old.md"), existing);
  await fs.writeFile(config.manifestPath, JSON.stringify({ version: 1, entries: { "note:old": { slug: "old", file: "old.md", sha256: digest(existing) } } }));
  const plan = await buildPlan(config); assert.equal(plan.selected.length, 0); assert.equal(plan.writes.size, 0);
  await applyPlan(plan); assert.deepEqual(await fs.readFile(path.join(config.outputPath, "old.md")), existing);
});
test("private linked notes cannot be implicitly published; code examples remain code", async (t) => {
  const { config, note } = await fixture(t);
  await note("private.md", { title: "私密", publish: false }); await note("a.md", { publish: true }, "[[私密]]\n");
  await assert.rejects(buildPlan(config), /尚未公开/);
  await note("a.md", { publish: true }, "```md\n[[私密]]\n```\n");
  const plan = await buildPlan(config); assert.match(plan.writes.get("src/content/posts/a.md").toString(), /\[\[私密\]\]/);
});
test("selected links, private comments, local images, source links, and repeated headings export correctly", async (t) => {
  const { config, note } = await fixture(t);
  const png = Buffer.from("89504e470d0a1a0a00000000", "hex");
  await fs.mkdir(path.join(config.vault, "附件")); await fs.writeFile(path.join(config.vault, "附件/图.png"), png);
  await note("b.md", { title: "乙", publish: true }, "乙正文");
  await note("目录/a.md", { title: "甲", publish: true, source: "https://arxiv.org/abs/1234.56789" }, "# 甲\n\n%% 私密想法 %%\n\n[Zotero](zotero://select/library/items/ABCDEFGH) [attachment](file:///Users/person/paper.pdf)\n\n[[乙#段落|另见乙]]\n\n![图](../附件/图.png)\n");
  const plan = await buildPlan(config), text = plan.writes.get("src/content/posts/目录/a.md").toString();
  assert.doesNotMatch(text, /私密想法|file:\/\/|zotero:\/\/|# 甲\n/); assert.match(text, /论文来源/); assert.match(text, /\[另见乙\]\(\/posts\/b\/#段落\)/); assert.match(text, /\/notes-assets\/[a-f0-9]+\.png/);
  await applyPlan(plan); const again = await buildPlan(config); assert.equal(again.writes.size, 0);
});
test("publication cannot traverse directories or overwrite another route", async (t) => {
  for (const slug of ["../secret", "/absolute", "a/../b", "a?x=y", "a\\b", "a%2fb"]) assert.throws(() => validSlug(slug));
  const { config, note } = await fixture(t); await note("a.md", { publish: true, slug: "same" }); await note("b.md", { publish: true, slug: "same" });
  await assert.rejects(buildPlan(config), /冲突/);
});
test("asset symlinks outside the vault and unresolved citations stop the entire plan", async (t) => {
  const { config, note } = await fixture(t);
  const outside = path.join(config.repo, "secret.png"); await fs.writeFile(outside, Buffer.from("89504e470d0a1a0a", "hex")); await fs.symlink(outside, path.join(config.vault, "secret.png"));
  await note("a.md", { publish: true }, "![[secret.png]]\n"); await assert.rejects(buildPlan(config), /找不到库内图片/);
  await note("a.md", { publish: true }, "引用 [@missing2026]。\n"); await assert.rejects(buildPlan(config), /不存在/);
  assert.deepEqual(await fs.readdir(config.outputPath), []);
});
test("source edits after preview and hand-edited public copies are protected", async (t) => {
  const { config, note } = await fixture(t); const file = await note("a.md", { publish: true });
  const plan = await buildPlan(config); await fs.appendFile(file, "新内容\n"); await assert.rejects(applyPlan(plan), /笔记在预览后/);
  await applyPlan(await buildPlan(config)); await fs.appendFile(path.join(config.outputPath, "a.md"), "手工编辑\n"); await assert.rejects(buildPlan(config), /手工修改/);
});
test("citations keep locators and include only referenced public bibliographic fields", async (t) => {
  const { config, note } = await fixture(t); await note("ref.md", { title: "论文", publish: false, citekey: "author2026", source: "https://example.com/paper", authors: ["Author"] }, "私密论文理解");
  await note("a.md", { publish: true }, "参见 [see @author2026, p. 3]。\n");
  const plan = await buildPlan(config), content = plan.writes.get("src/content/posts/a.md").toString();
  assert.match(content, /see \[1\]\(#reference-1\), p\. 3/); assert.match(content, /## 参考文献/); assert.doesNotMatch(content, /私密论文理解/);
});
