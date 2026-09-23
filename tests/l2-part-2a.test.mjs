// Lecture 2, Part 2 sections 2.1 to 2.3: numbers must match the notes.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const M = require("../js/core/math.js");
require("../js/lecture-2/part-2a.js");
const defs = globalThis.Lecture.defs;
const P2 = globalThis.Lecture.l2p2a;
const FRAGMENT = readFileSync(new URL("../.fragments/l2-p2a.html", import.meta.url), "utf8");

const r3 = (xs) => xs.map((x) => M.round(x, 3));
function block(steps, key) {
  for (const s of steps) for (const b of s.blocks) if (b.key === key) return b;
  throw new Error("no block with key " + key);
}

test("sinusoidal walkthrough: denominators and PE(0), PE(1), PE(2)", () => {
  const def = defs["l2-sinusoidal"];
  const { steps } = def.build(def.input);
  assert.deepEqual(r3(block(steps, "denoms").values), [1, 100]);
  assert.deepEqual(r3(block(steps, "pe0").values), [0, 1, 0, 1]);
  assert.deepEqual(r3(block(steps, "pe1").values), [0.841, 0.54, 0.01, 1]);
  assert.deepEqual(r3(block(steps, "pe2").values), [0.909, -0.416, 0.02, 1]);
});

test("sinusoidal walkthrough: cat at position 1", () => {
  const def = defs["l2-sinusoidal"];
  const { steps } = def.build(def.input);
  assert.deepEqual(block(steps, "emb").values, [0.2, 0.5, -0.1, 0.3]);
  assert.deepEqual(r3(block(steps, "catInput").values), [1.041, 1.04, -0.09, 1.3]);
});

test("sinusoidal predict: the slow hand barely moves", () => {
  const def = defs["l2-sinusoidal"];
  const { steps } = def.build(def.input);
  assert.ok(steps.length >= 4 && steps.length <= 7);
  const asking = steps.filter((s) => s.predict);
  assert.equal(asking.length, 1);
  assert.match(asking[0].predict.choices[asking[0].predict.answer], /slow hand/);
  assert.ok(asking[0].blocks.some((b) => b.key === "pe2"));
});

test("similarity walkthrough: PE(3) and the five dot products", () => {
  const def = defs["l2-pe-similarity"];
  const { steps } = def.build(def.input);
  assert.deepEqual(r3(block(steps, "pe3").values), [0.141, -0.99, 0.03, 1]);
  assert.deepEqual(r3(block(steps, "d01").values), [1.54]);
  assert.deepEqual(r3(block(steps, "dist1").values), [1.54, 1.54]);
  assert.deepEqual(r3(block(steps, "d02").values), [0.584]);
  assert.deepEqual(r3(block(steps, "d03").values), [0.01]);
  const S = block(steps, "simMatrix").rows.map(r3);
  assert.deepEqual(S, [
    [2, 1.54, 0.584, 0.01],
    [1.54, 2, 1.54, 0.584],
    [0.584, 1.54, 2, 1.54],
    [0.01, 0.584, 1.54, 2],
  ]);
});

test("similarity predicts: same at distance 1, lower at distance 2", () => {
  const def = defs["l2-pe-similarity"];
  const { steps } = def.build(def.input);
  assert.ok(steps.length >= 4 && steps.length <= 7);
  const asking = steps.filter((s) => s.predict);
  assert.equal(asking.length, 2);
  assert.match(asking[0].predict.choices[asking[0].predict.answer], /^Exactly 1\.540/);
  assert.match(asking[1].predict.choices[asking[1].predict.answer], /^Lower/);
});

test("the learned table has rows 0 to 511 and nothing past them", () => {
  assert.equal(P2.TABLE_ROWS, 512);
  assert.equal(P2.learnedRow(0).length, 8);
  assert.ok(Array.isArray(P2.learnedRow(511)));
  assert.equal(P2.learnedRow(512), null);
  assert.equal(P2.learnedRow(600), null);
  assert.deepEqual(P2.offsets(2, 6), [-2, -1, 0, 1, 2, 3]);
});

test("fragment has the exact subpart ids, interactives and walkthroughs", () => {
  for (const id of ["p2-1", "p2-2", "p2-3"]) assert.ok(FRAGMENT.includes(`<div class="subpart" id="${id}">`), id);
  for (const id of ["l2p2a-wall", "l2p2a-clocks", "l2p2a-heatmaps", "l2p2a-rotation", "l2p2a-bias"]) assert.ok(FRAGMENT.includes(`id="${id}"`), id);
  for (const w of ["l2-sinusoidal", "l2-pe-similarity"]) assert.ok(FRAGMENT.includes(`data-walk="${w}"`), w);
  for (const t of ["BERT", "GPT", "RoPE", "LLM", "T5", "ALiBi"]) assert.ok(FRAGMENT.includes(`<abbr data-term="${t}">${t}</abbr>`), t);
});

test("no em dashes in Part 2a text", () => {
  let text = FRAGMENT;
  for (const id of ["l2-sinusoidal", "l2-pe-similarity"]) text += JSON.stringify(defs[id].build(defs[id].input)) + defs[id].setup;
  assert.ok(!text.includes(String.fromCharCode(0x2014)));
});
