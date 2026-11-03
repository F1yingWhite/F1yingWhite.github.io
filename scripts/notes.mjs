#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { stringify } from "yaml";
import { loadConfig, scanNotes, buildPlan, applyPlan, safePath, digest, validSlug } from "./lib/notes.mjs";
import { validateSnapshots } from "./validate-public-notes.mjs";

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2), command = args[0] ?? "plan";
const option = (name) => { const i = args.indexOf(`--${name}`); return i < 0 ? undefined : args[i + 1]; };
const output = (value) => console.log(JSON.stringify(value, null, 2));
try {
  const config = await loadConfig(repo, option("vault"));
  if (command === "list") {
    output((await scanNotes(config.vault)).map((n) => ({ path: n.relative, title: n.data.title, publish: n.data.publish === true, slug: n.data.slug, type: n.data.type, status: n.data.status })));
  } else if (command === "new") {
    const title = option("title") ?? args[1];
    if (!title || title.startsWith("--")) throw new Error("用法：pnpm new-post '笔记标题'");
    const relative = `收件箱/${title.replace(/[\\/:*?"<>|]/g, "-")}.md`, file = safePath(config.vault, relative);
    await fs.mkdir(path.dirname(file), { recursive: true });
    await fs.writeFile(file, `---\n${stringify({ title, "note-id": randomUUID(), created: new Date().toISOString().slice(0, 10), type: "知识笔记", status: "整理中", publish: false, description: "", tags: [], category: "", slug: "" })}---\n\n# ${title}\n\n`, { flag: "wx" });
    output({ created: relative });
  } else if (command === "unpublish") {
    const slug = validSlug(option("slug")), manifest = JSON.parse(await fs.readFile(config.manifestPath, "utf8"));
    const pair = Object.entries(manifest.entries).find(([, entry]) => entry.slug === slug);
    if (!pair) throw new Error("该文章不在发布清单中");
    const [id, entry] = pair, notes = await scanNotes(config.vault);
    if (notes.some((n) => n.data.publish === true && (n.data["zotero-key"] ? `zotero:${n.data["zotero-key"]}` : `note:${n.data["note-id"]}`) === id)) throw new Error("请先取消这篇源笔记的 publish，再撤下公开副本。");
    const file = safePath(config.outputPath, entry.file), current = await fs.readFile(file);
    if (digest(current) !== entry.sha256) throw new Error("公开副本有手工修改，已停止撤稿");
    const result = { action: "unpublish", slug, file: path.relative(repo, file), applied: args.includes("--write") };
    if (args.includes("--write")) { delete manifest.entries[id]; await fs.rm(file); await fs.writeFile(config.manifestPath, JSON.stringify(manifest, null, 2) + "\n"); }
    output(result);
  } else if (["plan", "publish", "deploy"].includes(command)) {
    const plan = await buildPlan(config, { note: option("note") });
    const pending = execFileSync("git", ["status", "--porcelain=v1", "--no-renames", "--untracked-files=all", "-z", "--", config.output, config.assets, config.manifest], { cwd: repo, encoding: "utf8" }).split("\0").filter(Boolean).map((line) => ({ status: line.slice(0, 2), file: line.slice(3) }));
    const pendingHashes = await Promise.all(pending.map(async (entry) => [entry.status, entry.file, await fs.readFile(safePath(repo, entry.file)).then(digest, () => null)]));
    plan.summary.token = digest(Buffer.from(JSON.stringify([plan.summary.token, pendingHashes])));
    if (option("expect") && option("expect") !== plan.summary.token) throw new Error("笔记或公开文件在检查后发生变化，请重新检查改动。");
    if (command === "deploy") {
      if (config.maintenance) throw new Error("笔记迁移验证进行中，部署暂时关闭；阅读和编辑不受影响。");
      if (!args.includes("--confirmed")) throw new Error("部署需要确认本次公开改动：先运行 notes:plan，再使用 notes:deploy --confirmed。");
      const git = (a) => execFileSync("git", a, { cwd: repo, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }).trim();
      if (git(["branch", "--show-current"]) !== "main") throw new Error("请在 main 分支部署");
      if (git(["diff", "--cached", "--name-only"])) throw new Error("Git 暂存区已有其他改动，请先处理，避免混入本次发布");
      await applyPlan(plan);
      await validateSnapshots();
      execFileSync(process.execPath, ["node_modules/astro/astro.js", "build"], { cwd: repo, stdio: ["ignore", "pipe", "pipe"], maxBuffer: 20 * 1024 * 1024 });
      execFileSync(path.join(repo, "node_modules/.bin/pagefind"), ["--site", "dist"], { cwd: repo, stdio: ["ignore", "pipe", "pipe"] });
      execFileSync(process.execPath, ["scripts/verify-built-site.mjs"], { cwd: repo, stdio: ["ignore", "pipe", "pipe"] });
      const scope = [config.output, config.assets, config.manifest];
      const changed = git(["status", "--porcelain", "--", ...scope]);
      if (!changed) { output({ ...plan.summary, deployed: false, message: "公开副本没有变化，无需部署" }); }
      else { git(["add", "-A", "--", ...scope]); git(["commit", "-m", "docs: publish selected research notes"]); git(["push", "origin", "main"]); output({ ...plan.summary, deployed: true, commit: git(["rev-parse", "--short", "HEAD"]) }); }
    } else { if (command === "publish" && args.includes("--write")) await applyPlan(plan); output({ ...plan.summary, pending, applied: command === "publish" && args.includes("--write") }); }
  } else throw new Error(`未知命令：${command}`);
} catch (error) { console.error(error.message); process.exitCode = 1; }
