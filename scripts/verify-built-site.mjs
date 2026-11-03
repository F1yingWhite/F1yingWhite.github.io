import fs from "node:fs/promises";
import path from "node:path";
const root = path.resolve("dist"), cache = new Map(); let count = 0;
async function scan(dir) {
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) await scan(file);
    else if (entry.name.endsWith(".html")) {
      const html = await fs.readFile(file, "utf8");
      let css = [...html.matchAll(/<style[^>]*>([\s\S]*?)<\/style>/g)].map((m) => m[1]).join("\n");
      for (const match of html.matchAll(/<link[^>]+href="([^\"]+\.css)"[^>]*>/g)) {
        if (!match[1].startsWith("/_astro/")) continue;
        if (!cache.has(match[1])) cache.set(match[1], await fs.readFile(path.join(root, match[1].slice(1)), "utf8"));
        css += cache.get(match[1]);
      }
      for (const marker of ["--page-bg:", "--card-bg:", ".card-base{", ".float-panel-closed{"]) if (!css.includes(marker)) throw new Error(`${path.relative(root, file)} 缺少全局样式 ${marker}`);
      count++;
    }
  }
}
await scan(root);
const manifest = JSON.parse(await fs.readFile("notes-publication.json", "utf8"));
for (const entry of Object.values(manifest.entries)) await fs.access(path.join(root, "posts", entry.slug, "index.html"));
console.log(`Verified global styles on ${count} pages and all published article routes.`);
