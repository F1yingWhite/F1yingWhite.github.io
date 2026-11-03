import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { digest, safePath } from "./lib/notes.mjs";

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export async function validateSnapshots() {
  const config = JSON.parse(await fs.readFile(path.join(repo, "notes.config.json"), "utf8"));
  const manifest = JSON.parse(await fs.readFile(safePath(repo, config.manifest), "utf8"));
  const root = safePath(repo, config.output), known = new Set();
  for (const entry of Object.values(manifest.entries)) {
    const file = safePath(root, entry.file); known.add(file);
    if (digest(await fs.readFile(file)) !== entry.sha256) throw new Error(`公开副本与发布记录不一致：${entry.file}。请从 Obsidian 导出，而非手改副本。`);
  }
  async function scan(dir) {
    for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
      if (entry.name === ".obsidian") throw new Error("公开文章目录中不能包含 Obsidian 配置");
      const file = path.join(dir, entry.name);
      if (entry.isSymbolicLink()) throw new Error("公开文章目录中不能使用本机软链接");
      if (entry.isDirectory()) await scan(file);
      else if (/\.mdx?$/.test(entry.name) && !known.has(file)) throw new Error(`存在未经发布流程选择的文章：${path.relative(root, file)}`);
    }
  }
  await scan(root); return known.size;
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { console.log(`Verified ${await validateSnapshots()} public note snapshots.`); } catch (error) { console.error(error.message); process.exitCode = 1; }
}
