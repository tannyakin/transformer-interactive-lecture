// Lecture 2, Part 6: sparse attention must show exactly the notes' numbers.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";

const require = createRequire(import.meta.url);
require("../js/core/math.js");
require("../js/lecture-2/part-6.js");
const defs = globalThis.Lecture.defs;
const P6 = globalThis.Lecture.l2p6;
const PAGE = readFileSync(new URL("../lecture-2.html", import.meta.url), "utf8");
const FRAG = PAGE.slice(PAGE.indexOf('id="p6"'), PAGE.indexOf('id="p7"'));

function block(steps, key) {
  for (const s of steps) for (const b of s.blocks) if (b.key === key) return b;
  throw new Error("no block with key " + key);
}

test("n = 8: full attention is 64 pairs, a window of one each side is 22", () => {
  const def = defs["l2-window-pairs"];
  const { steps } = def.build(def.input);
  assert.equal(block(steps, "full").value, 64);
  assert.deepEqual(block(steps, "countsA").values, [2, 3, 3, 3]);
  assert.deepEqual(block(steps, "rowCounts").values, [2, 3, 3, 3, 3, 3, 3, 2]);
  assert.equal(block(steps, "windowTotal").value, 22);
  const grid = block(steps, "windowGrid").rows;
  assert.deepEqual(grid[0], ["■", "■", "·", "·", "·", "·", "·", "·"]);
  assert.deepEqual(grid[3], ["·", "·", "■", "■", "■", "·", "·", "·"]);
});

test("n = 100,000 with window 512: 10,000,000,000 vs 51,200,000, about 195 times fewer, linear growth", () => {
  const def = defs["l2-window-pairs"];
  const { steps } = def.build(def.input);
  const big = block(steps, "big").value;
  assert.equal(big.full, 10000000000);
  assert.equal(big.window, 51200000);
  assert.equal(Math.round(big.ratio), 195);
  assert.deepEqual(block(steps, "doubled").value, { fullGrowth: 4, winGrowth: 2 });
});

test("the pairs walkthrough has 4 to 7 steps and two predict stops with computed answers", () => {
  const def = defs["l2-window-pairs"];
  const { steps } = def.build(def.input);
  assert.ok(steps.length >= 4 && steps.length <= 7);
  const asking = steps.filter((s) => s.predict);
  assert.equal(asking.length, 2);
  assert.equal(asking[0].predict.choices[asking[0].predict.answer], "22");
  assert.equal(asking[1].predict.choices[asking[1].predict.answer], "2 times");
});

test("Longformer's grid matches the notes: 34 pairs, band 22, row counts as drawn", () => {
  const both = P6.longformerMask(8, "both");
  assert.deepEqual(P6.rowCounts(both), [8, 3, 4, 4, 4, 4, 4, 3]);
  assert.equal(P6.countPairs(both), 34);
  assert.equal(P6.countPairs(P6.longformerMask(8, "local")), 22);
  // Row "teddy bear" in the notes: [CLS], cute, teddy, and the next token.
  assert.deepEqual(both[3], [true, false, true, true, true, false, false, false]);
});

test("pattern explorer counts on 16 tokens", () => {
  const count = (o) => P6.countPairs(P6.patternKinds(16, o, 7).map((r) => r.map(Boolean)));
  assert.equal(count({ window: true }), 46);
  assert.equal(count({ strided: true }), 64);
  assert.equal(count({ global: true }), 31);
  assert.equal(count({ random: true }), 32);
  assert.equal(count({}), 0);
  assert.equal(count({ window: true, strided: true, global: true, random: true }) <= 256, true);
});

test("receptive field: Mistral 7B reaches 32 × 4,096 = 131,072, about 131,000", () => {
  assert.equal(P6.reach(32, 4096), 131072);
  const shown = FRAG.match(/id="l2p6-mistral">([^<]+)/)[1];
  assert.ok(shown.includes(`32 × 4,096 = ${P6.reach(32, 4096).toLocaleString("en-US")}`));
});

test("the fragment has its subparts, walkthrough and interactives, and no em dash", () => {
  for (const id of ["p6-1", "p6-2", "p6-3", "p6-4", "p6-5", "p6-6", "p6-7"]) assert.ok(FRAG.includes(`id="${id}"`), id);
  assert.ok(FRAG.includes('data-walk="l2-window-pairs"'));
  for (const id of ["l2p6-long", "l2p6-pat", "l2p6-reach"]) assert.ok(FRAG.includes(`id="${id}"`), id);
  const glossary = new Set(["[CLS]", "SWA", "CNN", "GPT"]);
  for (const [, term] of FRAG.matchAll(/data-term="([^"]+)"/g)) assert.ok(glossary.has(term), term);
  const def = defs["l2-window-pairs"];
  assert.ok(!(FRAG + JSON.stringify(def.build(def.input)) + def.setup).includes(String.fromCharCode(0x2014)));
});
