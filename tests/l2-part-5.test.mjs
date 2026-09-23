// Lecture 2, Part 5: the cost of attention must show exactly the notes' numbers.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";

const require = createRequire(import.meta.url);
require("../js/core/math.js");
require("../js/lecture-2/part-5.js");
const defs = globalThis.Lecture.defs;
const P5 = globalThis.Lecture.l2p5;
const FRAG = readFileSync(new URL("../.fragments/l2-p5.html", import.meta.url), "utf8");

function block(steps, key) {
  for (const s of steps) for (const b of s.blocks) if (b.key === key) return b;
  throw new Error("no block with key " + key);
}
const text = (b) => b.lines.map((l) => (typeof l === "string" ? l : l.t)).join("\n");

test("the memory walkthrough reaches 10,000,000,000 scores, 20,000,000,000 bytes, about 20 GB", () => {
  const def = defs["l2-attn-memory"];
  const { steps } = def.build(def.input);
  assert.equal(block(steps, "scores").value, 10000000000);
  assert.match(text(block(steps, "scores")), /100,000 × 100,000 = 10,000,000,000/);
  assert.equal(block(steps, "bytes").value, 20000000000);
  assert.match(text(block(steps, "bytes")), /10,000,000,000 × 2 bytes = 20,000,000,000 bytes/);
  assert.equal(block(steps, "gb").value, 20);
  assert.equal(block(steps, "double").value, 4);
  assert.equal(block(steps, "tenfold").value, 100);
});

test("the memory walkthrough has 4 to 7 steps and asks for the 4x before revealing it", () => {
  const def = defs["l2-attn-memory"];
  const { steps } = def.build(def.input);
  assert.ok(steps.length >= 4 && steps.length <= 7);
  const asking = steps.filter((s) => s.predict);
  assert.equal(asking.length, 1);
  const q = asking[0].predict;
  assert.equal(q.choices[q.answer], "4 times");
  assert.ok(asking[0].blocks.some((b) => b.key === "double"));
});

test("the crossover is n = 6d: 24,576 for d = 4,096", () => {
  assert.equal(P5.crossover(4096), 24576);
  for (const d of [512, 1024, 4096, 8192]) {
    const n = P5.crossover(d);
    assert.equal(P5.attnPerToken(n, d), P5.ffnPerToken(d));
  }
  assert.deepEqual(P5.attnOps(10, 4), { qk: 800, sv: 800 });
});

test("the growth table in the page matches n squared", () => {
  const rows = [...FRAG.matchAll(/<tr><td class="num">([\d,]+)<\/td><td class="num">([\d,]+)<\/td>/g)];
  assert.equal(rows.length, 4);
  for (const [, n, sq] of rows) {
    const v = Number(n.replace(/,/g, ""));
    assert.equal(sq, P5.scores(v).toLocaleString("en-US"));
  }
});

test("the fragment has its subparts, walkthrough and interactives, and no em dash", () => {
  for (const id of ["p5-1", "p5-2", "p5-3", "p5-4", "p5-5"]) assert.ok(FRAG.includes(`id="${id}"`), id);
  assert.ok(FRAG.includes('data-walk="l2-attn-memory"'));
  for (const id of ["l2p5-grow", "l2p5-cross", "l2p5-flash"]) assert.ok(FRAG.includes(`id="${id}"`), id);
  const glossary = new Set(["FLOPs", "FFN", "SRAM", "HBM", "KV cache"]);
  for (const [, term] of FRAG.matchAll(/data-term="([^"]+)"/g)) assert.ok(glossary.has(term), term);
  const def = defs["l2-attn-memory"];
  assert.ok(!(FRAG + JSON.stringify(def.build(def.input)) + def.setup).includes(String.fromCharCode(0x2014)));
});
