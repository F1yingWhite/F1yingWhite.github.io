import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { parse, stringify } from "yaml";
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkMath from "remark-math";

export const digest = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const exists = async (file) => fs.access(file).then(() => true, () => false);
const posix = (s) => s.split(path.sep).join("/");
const imageExtensions = new Set([".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif", ".svg", ".tif", ".tiff"]);
const noteId = (note) => note.data["zotero-key"] ? `zotero:${note.data["zotero-key"]}` : note.data["note-id"] ? `note:${note.data["note-id"]}` : null;
const parser = unified().use(remarkParse).use(remarkMath);
const walk = (node, fn) => { if (fn(node) !== false) for (const child of node.children ?? []) walk(child, fn); };
const inside = (root, file) => { const rel = path.relative(root, file); return rel === "" || (!rel.startsWith(`..${path.sep}`) && rel !== ".." && !path.isAbsolute(rel)); };
export function safePath(root, relative) {
  if (typeof relative !== "string" || !relative || relative.includes("\\") || relative.includes("\0") || path.isAbsolute(relative)) throw new Error("路径必须是库内的相对路径");
  const file = path.resolve(root, relative);
  if (!inside(path.resolve(root), file)) throw new Error("路径越出了允许目录");
  return file;
}
export function validSlug(value) {
  if (typeof value !== "string" || !value || /[?#%\\\u0000-\u0020]/u.test(value) || value.startsWith("/") || value.endsWith("/") || value.split("/").some((v) => !v || v === "." || v === "..")) throw new Error("slug 必须是固定的站内文章路径，不能包含空格、查询参数或 ..");
  return value;
}
export function readNote(text, relative = "note.md") {
  text = text.replaceAll("\r\n", "\n");
  const match = text.match(/^---\n([\s\S]*?)\n---(?:\n|$)/);
  let data;
  try { data = match ? parse(match[1], { uniqueKeys: true }) : {}; } catch (error) { throw new Error(`${relative}: YAML 无法解析：${error.message}`); }
  if (!data || typeof data !== "object" || Array.isArray(data)) throw new Error(`${relative}: 属性必须是 YAML 对象`);
  return { relative, data, body: match ? text.slice(match[0].length) : text, sourceHash: digest(Buffer.from(text)) };
}
export async function loadConfig(repo, vaultOverride) {
  const config = JSON.parse(await fs.readFile(path.join(repo, "notes.config.json"), "utf8"));
  const local = await fs.readFile(path.join(repo, ".notes-publish.local.json"), "utf8").then(JSON.parse, () => ({}));
  const vault = vaultOverride ?? process.env.NOTES_VAULT ?? local.vault;
  if (!vault) throw new Error("尚未设置笔记库：请在 .notes-publish.local.json 中设置 vault，或使用 NOTES_VAULT。");
  const root = await fs.realpath(path.resolve(vault));
  return { ...config, maintenance: local.maintenance === true, repo: path.resolve(repo), vault: root, outputPath: safePath(repo, config.output), assetsPath: safePath(repo, config.assets), manifestPath: safePath(repo, config.manifest) };
}
export async function scanNotes(vault) {
  const notes = [];
  async function scan(dir) {
    for (const entry of (await fs.readdir(dir, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
      if (entry.name.startsWith(".")) continue;
      const file = path.join(dir, entry.name);
      if (entry.isSymbolicLink()) continue;
      if (entry.isDirectory()) await scan(file);
      else if (entry.name.endsWith(".md")) notes.push({ ...readNote(await fs.readFile(file, "utf8"), posix(path.relative(vault, file))), file });
    }
  }
  await scan(vault);
  return notes;
}
function dateString(value, fallback) {
  const v = value ?? fallback;
  if (typeof v !== "string" || !/^\d{4}-\d{2}-\d{2}(?:[ T].*)?$/.test(v) || Number.isNaN(Date.parse(v))) throw new Error("日期必须是 YYYY-MM-DD 格式");
  return v;
}
function publicSource(data) {
  if (data.doi) return `https://doi.org/${String(data.doi).replace(/^https?:\/\/(?:dx\.)?doi\.org\//, "")}`;
  const value = data.source ?? (data.arxiv ? `https://arxiv.org/abs/${data.arxiv}` : null);
  if (!value) return null;
  const url = new URL(value);
  if (!["https:", "http:"].includes(url.protocol) || url.username || url.password) throw new Error("论文来源需要公开的 HTTP(S) 链接");
  return url.href;
}
const encodeUrl = (url) => url.split("/").map(encodeURIComponent).join("/");
const labelText = (s) => String(s).replaceAll("[", "\\[").replaceAll("]", "\\]").replaceAll("\n", " ");
const normalizedTitle = (s) => String(s).replace(/[*_`]/g, "").trim().toLowerCase();
function stripComments(body) {
  body = body.replace(/%%\/?zt-managed%%/g, "");
  const ranges = [];
  walk(parser.parse(body), (node) => { if (["code", "inlineCode", "math", "inlineMath"].includes(node.type)) ranges.push([node.position.start.offset, node.position.end.offset]); });
  return body.replace(/%%[\s\S]*?%%|<!--[\s\S]*?-->/g, (text, offset) => ranges.some(([a, b]) => offset >= a && offset < b) ? text : "");
}
async function assetBytes(config, from, target, options) {
  if (/^https?:\/\//i.test(target)) {
    let url = new URL(target);
    const cache = path.join(config.vault, ".research/publish-cache", digest(Buffer.from(url.href)));
    if (await exists(cache)) return fs.readFile(cache);
    let response;
    for (let redirects = 0; redirects < 6; redirects++) {
      if (url.protocol !== "https:" || url.username || url.password || /^(?:localhost|127\.|0\.|10\.|192\.168\.|169\.254\.|172\.(?:1[6-9]|2\d|3[01])\.|\[|.*\.local$)/i.test(url.hostname)) throw new Error("远程图片必须来自公开 HTTPS 地址");
      response = await (options.fetch ?? fetch)(url, { redirect: "manual", signal: AbortSignal.timeout(20000) });
      if (![301, 302, 303, 307, 308].includes(response.status)) break;
      url = new URL(response.headers.get("location"), url);
    }
    if (!response?.ok) throw new Error(`图片下载失败：HTTP ${response?.status ?? "unknown"}`);
    const chunks = []; let size = 0;
    for await (const chunk of response.body) { size += chunk.length; if (size > 20 * 1024 * 1024) throw new Error("单张图片超过 20 MB"); chunks.push(chunk); }
    const bytes = Buffer.concat(chunks);
    await fs.mkdir(path.dirname(cache), { recursive: true }); await fs.writeFile(cache, bytes);
    return bytes;
  }
  const clean = decodeURIComponent(target.split(/[?#]/)[0]);
  const candidates = [path.resolve(path.dirname(from.file), clean)];
  try { candidates.push(safePath(config.vault, clean.replace(/^\//, ""))); } catch { /* The note-relative candidate can still be inside the vault. */ }
  const candidate = await Promise.any(candidates.map(async (file) => { const real = await fs.realpath(file); if (!inside(config.vault, real)) throw new Error("图片实际位置位于笔记库外"); if (!(await fs.stat(real)).isFile()) throw new Error("图片不是普通文件"); return real; })).catch(() => null);
  if (!candidate) throw new Error(`找不到库内图片：${clean}`);
  return fs.readFile(candidate);
}
function imageExtension(bytes, target) {
  if (bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) return ".png";
  if (bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255) return ".jpg";
  if (/^GIF8[79]a/.test(bytes.subarray(0, 6).toString())) return ".gif";
  if (bytes.subarray(0, 4).toString() === "RIFF" && bytes.subarray(8, 12).toString() === "WEBP") return ".webp";
  if (bytes.subarray(4, 12).toString().includes("ftypavif")) return ".avif";
  if (["49492a00", "4d4d002a"].includes(bytes.subarray(0, 4).toString("hex"))) return ".tiff";
  if (path.extname(target.split(/[?#]/)[0]).toLowerCase() === ".svg" && /<svg[\s>]/i.test(bytes.toString()) && !/<script|<foreignObject|\bon\w+\s*=|javascript:|<!ENTITY/i.test(bytes.toString())) return ".svg";
  throw new Error("附件不是支持的图片，或 SVG 包含动态内容");
}
function resolveNote(notes, from, target) {
  let plain = decodeURIComponent(target).replace(/\.md$/, "");
  const relative = posix(path.normalize(path.join(path.dirname(from.relative), plain)));
  const exact = notes.filter((n) => [plain, relative].includes(n.relative.replace(/\.md$/, "")));
  if (exact.length === 1) return exact[0];
  const matches = notes.filter((n) => [n.data.title, path.basename(n.relative, ".md"), ...(Array.isArray(n.data.aliases) ? n.data.aliases : [])].includes(plain));
  if (matches.length !== 1) throw new Error(`笔记链接${matches.length ? "不唯一" : "不存在"}：${plain}`);
  return matches[0];
}
async function renderNote(note, context) {
  const { config, notes, routes, assets, options } = context;
  let body = stripComments(note.body);
  const tree = parser.parse(body), edits = [], pending = [], cited = new Map();
  const edit = (node, value) => edits.push({ start: node.position.start.offset, end: node.position.end.offset, value });
  const image = async (target, alt) => {
    const bytes = await assetBytes(config, note, target, options), ext = imageExtension(bytes, target);
    const name = `${digest(bytes).slice(0, 24)}${ext}`;
    assets.set(posix(path.relative(config.repo, path.join(config.assetsPath, name))), bytes);
    return `![${labelText(alt ?? "图片")}](/notes-assets/${name})`;
  };
  const noteLink = (target, anchor = "", label) => {
    const found = resolveNote(notes, note, target), slug = routes.get(noteId(found));
    if (!slug) throw new Error(`链接指向尚未公开的笔记「${found.data.title ?? found.relative}」；请改为普通文字，或明确选择同时发布。`);
    return `[${labelText(label ?? found.data.title ?? target)}](/posts/${encodeUrl(slug)}/${anchor})`;
  };
  const citation = (raw) => {
    const refs = [...raw.matchAll(/@([\p{L}\p{N}_:.-]+)/gu)].map((m) => m[1]);
    const labels = refs.map((key) => {
      const matched = notes.filter((n) => n.data.citekey === key);
      if (matched.length !== 1) throw new Error(`引用 ${key} ${matched.length ? "不唯一" : "不存在"}`);
      const ref = matched[0], source = publicSource(ref.data);
      if (!source) throw new Error(`引用 ${key} 缺少公开来源链接`);
      if (!cited.has(key)) cited.set(key, { note: ref, source, number: cited.size + 1 });
      return `[${cited.get(key).number}](#reference-${cited.get(key).number})`;
    });
    const rendered = new Map(refs.map((key, index) => [key, labels[index]]));
    return raw.slice(1, -1).replace(/@([\p{L}\p{N}_:.-]+)/gu, (_match, key) => rendered.get(key));
  };
  walk(tree, (node) => {
    if (node.type === "code" && ["zotquery", "dataview", "dataviewjs"].includes(node.lang)) { edit(node, ""); return false; }
    if (node.type === "paragraph" && node.children.some((n) => n.type === "link") && node.children.every((n) => n.type === "link" && /^(?:zotero|file|obsidian):/i.test(n.url) || n.type === "text" && !n.value.trim())) {
      const source = publicSource(note.data); edit(node, source ? `论文来源：[${labelText(note.data.title)}](${source})` : ""); return false;
    } else if (node.type === "image") pending.push(image(node.url, node.alt).then((value) => edit(node, value)));
    else if (node.type === "link") {
      const label = (node.children ?? []).map((n) => n.value ?? "").join("") || "来源";
      if (/^(?:zotero|file|obsidian):/i.test(node.url)) { const source = publicSource(note.data); edit(node, source ? `[${labelText(label)}](${source})` : labelText(label === "attachment" || label === "Zotero" ? "" : label)); }
      else if (!/^(?:[a-z][a-z0-9+.-]*:|\/|#)/i.test(node.url)) {
        const hashAt = node.url.indexOf("#"), target = hashAt < 0 ? node.url : node.url.slice(0, hashAt), anchor = hashAt < 0 ? "" : node.url.slice(hashAt);
        if (target.endsWith(".md")) edit(node, noteLink(target, anchor, label));
        else if (imageExtensions.has(path.extname(target).toLowerCase())) pending.push(image(node.url, label).then((value) => edit(node, value)));
        else if (/\.pdf$/i.test(target)) { const source = publicSource(note.data); if (!source) throw new Error("PDF 本机附件缺少公开论文来源"); edit(node, `[${labelText(label)}](${source})`); }
        else throw new Error(`尚未处理的本机附件链接：${target}，请改为公开链接或普通文字`);
      }
    } else if (node.type === "html" && /(?:file:|zotero:|obsidian:|<script|<iframe|\bon\w+\s*=)/i.test(node.value)) throw new Error("正文 HTML 含本机链接或动态脚本，请先整理");
    else if (node.type === "text") {
      const raw = body.slice(node.position.start.offset, node.position.end.offset);
      const pattern = /(!?)\[\[([^\]\n]+)\]\]|\[[^\]\n]*@[\p{L}\p{N}_:.-]+[^\]\n]*\]/gu;
      for (const match of raw.matchAll(pattern)) {
        const start = node.position.start.offset + match.index, end = start + match[0].length;
        if (match[2]) {
          const [targetWithAnchor, alias] = match[2].split("|"), at = targetWithAnchor.indexOf("#"), target = at < 0 ? targetWithAnchor : targetWithAnchor.slice(0, at), anchor = at < 0 ? "" : targetWithAnchor.slice(at);
          if (imageExtensions.has(path.extname(target).toLowerCase())) pending.push(image(target, alias).then((value) => edits.push({ start, end, value })));
          else if (/\.pdf$/i.test(target)) { const source = publicSource(note.data); if (!source) throw new Error("PDF 双链缺少公开论文来源"); edits.push({ start, end, value: `[${labelText(alias ?? path.basename(target, ".pdf"))}](${source})` }); }
          else edits.push({ start, end, value: noteLink(target, anchor, alias) });
        } else edits.push({ start, end, value: citation(match[0]) });
      }
    }
  });
  await Promise.all(pending);
  edits.sort((a, b) => b.start - a.start);
  let last = body.length;
  for (const e of edits) { if (e.end > last) throw new Error("正文链接范围重叠，导出已停止"); body = body.slice(0, e.start) + e.value + body.slice(e.end); last = e.start; }
  body = body.replace(/^\s*\n/, "");
  const firstHeading = body.match(/^# (.+)\n(?:\n|$)/);
  if (firstHeading && normalizedTitle(firstHeading[1]) === normalizedTitle(note.data.title)) body = body.slice(firstHeading[0].length);
  if (/(?:file:\/\/|zotero:\/\/|obsidian:\/\/|\/Users\/|[A-Z]:\\Users\\)/i.test(body)) throw new Error("公开正文仍含本机路径或专用协议，请先清理");
  if (/[]/u.test(body)) throw new Error("正文仍含未解析的引用标记，请先核对来源");
  if (cited.size) body += "\n\n## 参考文献\n\n" + [...cited.values()].map(({ note: ref, source, number }) => `<span id="reference-${number}"></span>${number}. ${labelText((ref.data.authors ?? []).join(", "))}${ref.data.authors?.length ? ". " : ""}[${labelText(ref.data.title)}](${source}).`).join("\n\n");
  return body.trim() + "\n";
}
export async function buildPlan(config, options = {}) {
  config = { ...config, vault: await fs.realpath(config.vault) };
  const manifestText = await fs.readFile(config.manifestPath, "utf8").catch(() => '{"version":1,"entries":{}}\n');
  const manifest = JSON.parse(manifestText);
  if (manifest.version !== 1 || !manifest.entries || Array.isArray(manifest.entries)) throw new Error("发布清单格式错误");
  const notes = await scanNotes(config.vault), selected = notes.filter((n) => n.data.publish === true && (!options.note || options.note === n.relative || options.note === noteId(n)));
  if (options.note && !selected.length) throw new Error("指定笔记不存在，或尚未明确设置 publish: true");
  const ids = new Set(), routes = new Map(), owners = new Map(), candidates = [], assets = new Map(), writes = new Map(), expected = new Map(), actions = [];
  for (const [id, entry] of Object.entries(manifest.entries)) { validSlug(entry.slug); safePath(config.outputPath, entry.file); if (owners.has(entry.slug)) throw new Error("发布清单中有重复网址"); routes.set(id, entry.slug); owners.set(entry.slug, id); }
  for (const n of selected) {
    const id = noteId(n); if (!id) throw new Error(`${n.relative}: 缺少 note-id 或 zotero-key`);
    if (ids.has(id)) throw new Error("选中的笔记有重复的文献标识"); ids.add(id);
    if (notes.filter((other) => noteId(other) === id).length > 1) throw new Error(`${n.relative}: 库内存在重复文献笔记`);
    if (typeof n.data.title !== "string" || !n.data.title.trim()) throw new Error(`${n.relative}: 缺少标题`);
    const old = manifest.entries[id], slug = validSlug(n.data.slug ?? old?.slug);
    if (old && old.slug !== slug) throw new Error(`${n.relative}: 已公开文章的网址不能隐式改变`);
    if (owners.has(slug) && owners.get(slug) !== id) throw new Error("slug 与另一篇公开文章冲突");
    if (n.data.tags !== undefined && (!Array.isArray(n.data.tags) || n.data.tags.some((s) => typeof s !== "string"))) throw new Error(`${n.relative}: tags 必须是文字列表`);
    routes.set(id, slug); owners.set(slug, id); candidates.push({ note: n, id, slug, old, file: old?.file ?? `${slug}.md` });
  }
  for (const c of candidates) {
    const { note: n, id, slug, old, file } = c;
    try {
      const content = await renderNote(n, { config, notes, routes, assets, options });
      const meta = { title: n.data.title.trim(), published: dateString(n.data["blog-date"], old?.published ?? (options.today ?? new Date().toISOString().slice(0, 10))), description: typeof n.data.description === "string" ? n.data.description : "", image: "", tags: n.data.tags ?? [], category: n.data.category ?? "", draft: false, slug };
      if (n.data.lang) meta.lang = n.data.lang;
      if (n.data.updated) meta.updated = dateString(n.data.updated);
      if (/(?:file:\/\/|zotero:\/\/|obsidian:\/\/|\/Users\/|[A-Z]:\\Users\\)/i.test(JSON.stringify(meta))) throw new Error("公开属性仍含本机路径或专用协议，请先清理");
      if (n.data.image) {
        const bytes = await assetBytes(config, n, n.data.image, options), ext = imageExtension(bytes, n.data.image), name = `${digest(bytes).slice(0, 24)}${ext}`;
        assets.set(posix(path.relative(config.repo, path.join(config.assetsPath, name))), bytes); meta.image = `/notes-assets/${name}`;
      }
      const bytes = Buffer.from(`---\n${stringify(meta, { lineWidth: 0 })}---\n\n${content}`), target = safePath(config.outputPath, file);
      const current = await fs.readFile(target).catch(() => null), currentHash = current ? digest(current) : null;
      if (old && currentHash !== old.sha256) throw new Error("公开副本被手工修改，已停止覆盖；请先核对差异");
      if (!old && current) throw new Error("目标文件已存在，禁止覆盖未受管理的文章");
      const action = { type: currentHash === digest(bytes) ? "unchanged" : current ? "update" : "add", source: n.relative, slug, file: posix(path.relative(config.repo, target)) };
      actions.push(action); expected.set(action.file, currentHash);
      if (action.type !== "unchanged") writes.set(action.file, bytes);
      manifest.entries[id] = { slug, file, source: n.relative, published: meta.published, sha256: digest(bytes), legacy: false };
    } catch (error) { throw new Error(`${n.relative}: ${error.message}`); }
  }
  for (const [file, bytes] of assets) {
    const current = await fs.readFile(safePath(config.repo, file)).catch(() => null);
    if (!current || !current.equals(bytes)) { writes.set(file, bytes); expected.set(file, current ? digest(current) : null); }
  }
  const serialized = JSON.stringify(manifest, null, 2) + "\n";
  if (selected.length && serialized !== manifestText) { writes.set(config.manifest, Buffer.from(serialized)); expected.set(config.manifest, await exists(config.manifestPath) ? digest(Buffer.from(manifestText)) : null); }
  const token = digest(Buffer.from(JSON.stringify([manifestText, selected.map((n) => [n.relative, n.sourceHash]), [...writes].map(([file, bytes]) => [file, digest(bytes)])])));
  return { config, notes, selected, actions, writes, expected, manifest, manifestHash: digest(Buffer.from(manifestText)), summary: { token, scanned: notes.length, selected: selected.length, retained: Object.keys(manifest.entries).length - selected.length, added: actions.filter((a) => a.type === "add").length, updated: actions.filter((a) => a.type === "update").length, unchanged: actions.filter((a) => a.type === "unchanged").length, assets: assets.size, actions, files: [...writes.keys()] } };
}
export async function applyPlan(plan) {
  const manifestNow = await fs.readFile(plan.config.manifestPath).catch(() => Buffer.from('{"version":1,"entries":{}}\n'));
  if (digest(manifestNow) !== plan.manifestHash) throw new Error("发布清单在预览后已更改，请重新检查发布改动");
  for (const note of plan.selected) { const current = (await fs.readFile(note.file, "utf8")).replaceAll("\r\n", "\n"); if (digest(Buffer.from(current)) !== note.sourceHash) throw new Error("笔记在预览后已更改，请重新检查发布改动"); }
  const originals = new Map();
  for (const [relative, expected] of plan.expected) {
    const target = safePath(plan.config.repo, relative), current = await fs.readFile(target).catch(() => null);
    if ((current ? digest(current) : null) !== expected) throw new Error("公开文件在预览后已更改，请重新检查发布改动");
    originals.set(relative, current);
  }
  const written = [];
  try {
    for (const [relative, bytes] of plan.writes) {
      const target = safePath(plan.config.repo, relative); await fs.mkdir(path.dirname(target), { recursive: true });
      const temporary = `${target}.tmp-${crypto.randomUUID()}`; await fs.writeFile(temporary, bytes); await fs.rename(temporary, target); written.push(relative);
    }
  } catch (error) {
    for (const relative of written.reverse()) { const target = safePath(plan.config.repo, relative), original = originals.get(relative); if (original) await fs.writeFile(target, original); else await fs.rm(target, { force: true }); }
    throw error;
  }
  return plan.summary;
}
