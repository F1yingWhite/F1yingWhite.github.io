import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
execFileSync(process.execPath, [fileURLToPath(new URL("./notes.mjs", import.meta.url)), "new", ...process.argv.slice(2)], { stdio: "inherit" });
