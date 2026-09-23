// Part 7 (MHA, MQA, GQA): the KV cache walkthrough and the G slider must match the notes.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const M = require("../js/core/math.js");
require("../js/lecture-2/part-7.js");
const defs = globalThis.Lecture.defs;
const P7 = globalThis.Lecture.l2p7;
const EM = String.fromCharCode(0x2014);

function block(steps, key) {
  for (const s of steps) for (const b of s.blocks) if (b.key === key) return b;
  throw new Error("no block with key " + key);
}

test("KV cache walkthrough: 524,288 bytes per token, 0.5 MB, 2 GB, 40 GB", () => {
  const def = defs["l2-kv-cache"];
  const { steps } = def.build(def.input);
  assert.equal(block(steps, "perToken").value, 524288);
  assert.ok(block(steps, "perToken").lines.some((l) => (l.t || l).includes("524,288 bytes")));
  assert.equal(block(steps, "perTokenMB").value, 0.5);
  assert.equal(block(steps, "conversation").value, 2);
  assert.equal(block(steps, "users").value, 40);
  assert.ok(steps.length >= 4 && steps.length <= 7);
});

test("KV cache walkthrough asks for the 4,096-token size and the right choice is 2 GB", () => {
  const def = defs["l2-kv-cache"];
  const { steps } = def.build(def.input);
  const asking = steps.filter((s) => s.predict);
  assert.equal(asking.length, 1);
  const p = asking[0].predict;
  assert.equal(p.choices[p.answer], "About 2 GB");
  assert.ok(asking[0].blocks.some((b) => b.key === "conversation"));
});

test("G slider: names and cache sizes follow the notes", () => {
  assert.deepEqual(P7.G_OPTIONS, [1, 2, 4, 8, 16, 32]);
  assert.equal(P7.attnName(1, 32).short, "MQA");
  assert.equal(P7.attnName(8, 32).short, "GQA");
  assert.equal(P7.attnName(32, 32).short, "MHA");
  assert.equal(P7.fmtBytes(P7.cacheForG(1)), "64 MB");
  assert.equal(P7.fmtBytes(P7.cacheForG(8)), "512 MB");
  assert.equal(P7.fmtBytes(P7.cacheForG(32)), "2 GB");
  // h = 32, G = 8: query heads 0 to 3 share KV head 0, 28 to 31 share KV head 7.
  assert.equal(P7.groupOf(3, 8, 32), 0);
  assert.equal(P7.groupOf(4, 8, 32), 1);
  assert.equal(P7.groupOf(31, 8, 32), 7);
});

test("generation counter: one new K/V pair per step with the cache, all of them without", () => {
  assert.deepEqual(P7.genTotals(P7.GEN_PROMPT, P7.GEN_PROMPT), { withCache: 0, without: 0 });
  assert.deepEqual(P7.genTotals(3, 2), { withCache: 1, without: 3 });
  assert.deepEqual(P7.genTotals(7, 2), { withCache: 5, without: 3 + 4 + 5 + 6 + 7 });
  assert.equal(P7.fmtBytes(7 * 524288), "3.5 MB");
});

test("Part 7 text has no em dash and the page wiring is consistent", () => {
  const def = defs["l2-kv-cache"];
  assert.ok(!(JSON.stringify(def.build(def.input)) + def.setup).includes(EM));
  const frag = fileURLToPath(new URL("../.fragments/l2-p7.html", import.meta.url));
  if (!existsSync(frag)) return;
  const html = readFileSync(frag, "utf8");
  assert.ok(!html.includes(EM));
  assert.ok(html.includes('data-walk="l2-kv-cache"'));
  for (const id of ["p7-1", "p7-2", "p7-3", "p7-4", "p7-5", "l2p7-generate", "l2p7-groups"]) assert.ok(html.includes(`id="${id}"`), id);
});
