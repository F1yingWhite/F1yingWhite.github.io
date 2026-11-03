const { Plugin, ItemView, Notice, Modal, Setting, PluginSettingTab, TFile } = require("obsidian");
const { execFile, spawn } = require("child_process");
const { promisify } = require("util");
const { randomUUID } = require("crypto");
const path = require("path");
const run = promisify(execFile);
const VIEW = "research-blog-publisher";

class PublishView extends ItemView {
  constructor(leaf, plugin) { super(leaf); this.plugin = plugin; this.busy = false; }
  getViewType() { return VIEW; }
  getDisplayText() { return "博客发布"; }
  getIcon() { return "send"; }
  async onOpen() { await this.refresh(); }
  async perform(callback) {
    if (this.busy) return;
    this.busy = true;
    try { await callback(); } catch (error) { new Notice(error.message, 12000); }
    finally { this.busy = false; }
  }
  async refresh() {
    const el = this.contentEl; el.empty(); el.createEl("h2", { text: "博客发布" });
    el.createEl("p", { text: "笔记默认留在本地。只有明确选中的笔记才会导出；已有线上版本一直保留，撤稿需要单独操作。" });
    new Setting(el).setName("当前笔记").setDesc("在编辑器中打开笔记后，使用命令“选择当前笔记发布”或“取消当前笔记选择”。")
      .addButton((b) => b.setButtonText("检查改动").onClick(() => this.perform(() => this.refresh())));
    try {
      this.plan = await this.plugin.command("plan");
      const p = this.plan;
      el.createEl("p", { text: `已选 ${p.selected} 篇 · 新增 ${p.added} 篇 · 更新 ${p.updated} 篇 · 保留 ${p.retained} 篇公开快照` });
      if (!p.selected) el.createEl("p", { text: "目前没有选中笔记。整理后的笔记尚未自动公开。" });
      for (const a of p.actions) el.createEl("p", { text: `${({ add: "新增", update: "更新", unchanged: "无变化" })[a.type]}：${a.source}` });
      if (p.pending.length) { el.createEl("h3", { text: "待提交的公开文件" }); const list = el.createEl("ul"); for (const f of p.pending) list.createEl("li", { text: f.file }); }
      new Setting(el).setName("导出并预览").setDesc("将已选笔记转换为公开副本，构建后在浏览器中预览。此步不会推送到 GitHub。")
        .addButton((b) => b.setButtonText("导出并预览").setDisabled(!p.selected && !p.pending.length).onClick(() => this.perform(async () => {
          await this.plugin.command("publish", ["--write", "--expect", p.token]);
          new Notice("正在构建预览…", 5000); await this.plugin.preview(); await this.refresh();
        })));
      new Setting(el).setName("部署公开副本").setDesc("确认公开文件后，提交并推送到博客仓库；GitHub Pages 随后自动部署。")
        .addButton((b) => b.setButtonText("提交并部署…").setDisabled(!p.pending.length && !p.added && !p.updated).onClick(() => {
          const modal = new Modal(this.app); modal.contentEl.createEl("h2", { text: "确认部署到公开博客" });
          modal.contentEl.createEl("p", { text: `将处理已选的 ${p.selected} 篇笔记及上方列出的公开文件。只有博客副本、图片和发布清单会提交。` });
          new Setting(modal.contentEl).addButton((button) => button.setButtonText("取消").onClick(() => modal.close()))
            .addButton((button) => button.setButtonText("提交并推送").setCta().onClick(() => { modal.close(); this.perform(async () => {
              new Notice("正在构建并部署…", 6000); const result = await this.plugin.command("deploy", ["--confirmed", "--expect", p.token]);
              new Notice(result.deployed ? `已推送 ${result.commit}，等待 GitHub Pages 部署。` : result.message, 10000); await this.refresh();
            }); })); modal.open();
        }));
      new Setting(el).setName("查看线上博客").addButton((b) => b.setButtonText("打开博客").onClick(() => require("electron").shell.openExternal("https://f1yingwhite.github.io/")));
    } catch (error) { el.createEl("p", { text: error.message }); }
  }
}
class PublisherSettings extends PluginSettingTab {
  constructor(app, plugin) { super(app, plugin); this.plugin = plugin; }
  display() {
    this.containerEl.empty();
    new Setting(this.containerEl).setName("博客仓库").setDesc("本机博客仓库的完整路径，仅保存在本机。")
      .addText((t) => t.setValue(this.plugin.settings.repoPath).onChange(async (value) => { this.plugin.settings.repoPath = value; await this.plugin.saveData(this.plugin.settings); }));
    new Setting(this.containerEl).setName("Node.js").setDesc("Node.js 可执行文件的完整路径。")
      .addText((t) => t.setValue(this.plugin.settings.nodePath).onChange(async (value) => { this.plugin.settings.nodePath = value; await this.plugin.saveData(this.plugin.settings); }));
  }
}
module.exports = class ResearchPublisher extends Plugin {
  async onload() {
    this.settings = { repoPath: "", nodePath: "/opt/homebrew/bin/node", ...(await this.loadData()) };
    this.registerView(VIEW, (leaf) => new PublishView(leaf, this));
    this.addRibbonIcon("send", "博客发布", () => this.openView());
    this.addCommand({ id: "open-publisher", name: "打开博客发布", callback: () => this.openView() });
    this.addCommand({ id: "select-current", name: "选择当前笔记发布", callback: () => this.selectCurrent(true) });
    this.addCommand({ id: "deselect-current", name: "取消当前笔记选择", callback: () => this.selectCurrent(false) });
    this.addSettingTab(new PublisherSettings(this.app, this));
  }
  onunload() { if (this.server) this.server.kill(); }
  async command(command, args = []) {
    if (!this.settings.repoPath) throw new Error("请先在插件设置中指定博客仓库。");
    try {
      const { stdout } = await run(this.settings.nodePath, [path.join(this.settings.repoPath, "scripts/notes.mjs"), command, "--vault", this.app.vault.adapter.getBasePath(), ...args], { cwd: this.settings.repoPath, maxBuffer: 20 * 1024 * 1024, env: { ...process.env, PATH: `/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:${process.env.PATH ?? ""}` } });
      return JSON.parse(stdout);
    } catch (error) { throw new Error((error.stderr || error.message).slice(-5000)); }
  }
  async openView() {
    let leaf = this.app.workspace.getLeavesOfType(VIEW)[0];
    if (!leaf) { leaf = this.app.workspace.getLeaf("tab"); await leaf.setViewState({ type: VIEW }); }
    this.app.workspace.revealLeaf(leaf); await leaf.view.refresh();
  }
  async selectCurrent(selected) {
    const file = this.app.workspace.getActiveFile();
    if (!(file instanceof TFile) || file.extension !== "md") { new Notice("先打开一篇 Markdown 笔记。"); return; }
    if (/^(?:模板|整理记录)\//.test(file.path) || file.path === "00 首页.md" || file.path === "工作流与发布说明.md") { new Notice("导航、模板和整理记录不参与公开发布。"); return; }
    if (!selected) { await this.app.fileManager.processFrontMatter(file, (data) => { data.publish = false; }); new Notice("已取消后续发布选择；已有线上文章仍保留。"); return; }
    const meta = this.app.metadataCache.getFileCache(file)?.frontmatter ?? {};
    const modal = new Modal(this.app); modal.contentEl.createEl("h2", { text: "选择笔记发布" });
    modal.contentEl.createEl("p", { text: "选择只让它进入待发布列表，导出和部署需在发布面板中执行。网址一经公开应保持不变。" });
    let slug = meta.slug ?? file.basename.toLowerCase().replace(/[^\p{L}\p{N}._-]+/gu, "-").replace(/^-|-$/g, "");
    new Setting(modal.contentEl).setName("文章网址路径").addText((t) => t.setValue(slug).onChange((v) => { slug = v.trim(); }));
    new Setting(modal.contentEl).addButton((b) => b.setButtonText("取消").onClick(() => modal.close()))
      .addButton((b) => b.setButtonText("加入待发布").setCta().onClick(async () => {
        if (!slug || /[?#%\\\s]/u.test(slug) || slug.split("/").some((part) => !part || part === "." || part === "..")) { new Notice("请输入有效、固定的文章路径。"); return; }
        await this.app.fileManager.processFrontMatter(file, (data) => { data.publish = true; data.slug = slug; if (!data.title) data.title = file.basename; if (!data["note-id"]) data["note-id"] = randomUUID(); });
        modal.close(); new Notice("已加入待发布列表。"); await this.openView();
      })); modal.open();
  }
  async preview() {
    try {
      for (const args of [["scripts/validate-public-notes.mjs"], ["node_modules/astro/astro.js", "build"], ["node_modules/pagefind/lib/runner/bin.cjs", "--site", "dist"], ["scripts/verify-built-site.mjs"]]) {
        await run(this.settings.nodePath, args, { cwd: this.settings.repoPath, maxBuffer: 20 * 1024 * 1024 });
      }
    }
    catch (error) { throw new Error(`预览构建失败：${(error.stderr || error.stdout || error.message).slice(-5000)}`); }
    if (!this.server || this.server.exitCode !== null) {
      this.server = spawn(this.settings.nodePath, ["node_modules/astro/astro.js", "preview", "--host", "127.0.0.1", "--port", "4327"], { cwd: this.settings.repoPath, stdio: ["ignore", "pipe", "pipe"] });
      await new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error("本地预览未能启动，请确认 4327 端口可用。")), 15000);
        this.server.stdout.on("data", (bytes) => { if (bytes.toString().includes("127.0.0.1")) { clearTimeout(timer); resolve(); } });
        this.server.once("error", (error) => { clearTimeout(timer); reject(error); });
        this.server.once("exit", (code) => { clearTimeout(timer); reject(new Error(`预览服务退出：${code}`)); });
      });
    }
    await require("electron").shell.openExternal("http://127.0.0.1:4327/");
  }
};
