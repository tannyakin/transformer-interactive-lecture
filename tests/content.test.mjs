// Writing rule for this site: no em dashes anywhere a student can read.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, extname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const SKIP_DIRS = new Set([".git", ".superpowers", "node_modules", ".playwright-mcp"]);
const EXTS = new Set([".html", ".js", ".mjs", ".css", ".md"]);
const EM_DASH = "2014";

function* walk(dir) {
  for (const name of readdirSync(dir)) {
    if (SKIP_DIRS.has(name)) continue;
    const full = join(dir, name);
    if (statSync(full).isDirectory()) yield* walk(full);
    else if (EXTS.has(extname(name))) yield full;
  }
}

test("no em dashes in any site file", () => {
  const hits = [];
  for (const file of walk(ROOT)) {
    readFileSync(file, "utf8").split("\n").forEach((line, i) => {
      if (line.includes(EM_DASH)) hits.push(`${relative(ROOT, file)}:${i + 1}`);
    });
  }
  assert.deepEqual(hits, [], "Replace these em dashes with a comma, colon or full stop");
});
