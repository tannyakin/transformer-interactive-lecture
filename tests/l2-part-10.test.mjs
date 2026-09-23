// Part 10 walkthroughs must show exactly the notes' numbers.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const M = require("../js/core/math.js");
require("../js/lecture-2/part-10.js");
const defs = globalThis.Lecture.defs;
const P = globalThis.Lecture.l2p10;
const EM = String.fromCharCode(0x2014);

function block(steps, key) {
  for (const s of steps) for (const b of s.blocks) if (b.key === key) return b;
  throw new Error("no block with key " + key);
}
const built = (id) => defs[id].build(defs[id].input);

test("RoBERTa batches: 256,000,000 vs 4,000,000,000, ratio about 15.6", () => {
  const { steps } = built("l2-roberta-batches");
  assert.equal(block(steps, "bert").raw, 256000000);
  assert.deepEqual(block(steps, "bert").values, ["256,000,000"]);
  assert.equal(block(steps, "roberta").raw, 4000000000);
  assert.deepEqual(block(steps, "roberta").values, ["4,000,000,000"]);
  assert.equal(M.round(block(steps, "ratio").raw, 1), 15.6);
  const p = steps.filter((s) => s.predict);
  assert.equal(p.length, 1);
  assert.match(p[0].predict.choices[p[0].predict.answer], /15\.6/);
});

test("dynamic masking: 10 masks seen 4 times, C(20,3) = 6,840 / 6 = 1,140", () => {
  const { steps } = built("l2-dynamic-masking");
  assert.deepEqual(block(steps, "static").values, [10, 4]);
  assert.deepEqual(block(steps, "dynamic").values, [40, 1]);
  assert.deepEqual(block(steps, "masked").values, [3]);
  assert.equal(block(steps, "ways").raw, 1140);
  const text = JSON.stringify(steps);
  assert.ok(text.includes("C(20, 3) = (20 × 19 × 18) / (3 × 2 × 1) = 6,840 / 6 = 1,140"));
  const p = steps.find((s) => s.predict).predict;
  assert.equal(p.choices[p.answer], "1,140");
});

test("KL distillation: terms, 0.0833 from rounded terms, 0.0834 unrounded, later 0.0011", () => {
  const { steps } = built("l2-kl-distill");
  assert.deepEqual(block(steps, "logs").values.map((x) => M.round(x, 3)), [-0.47, 0.336, -0.693]);
  assert.deepEqual(block(steps, "terms").values, [-0.1175, 0.2355, -0.0347]);
  assert.equal(M.round(block(steps, "kl-early").values[0], 4), 0.0833);
  assert.equal(M.round(block(steps, "kl-early").raw, 4), 0.0834);
  assert.equal(M.round(block(steps, "kl-late").values[0], 4), 0.0011);
  const preds = steps.filter((s) => s.predict).map((s) => s.predict);
  assert.equal(preds.length, 2);
  assert.match(preds[0].choices[preds[0].answer], /always 0 or more/);
  assert.match(preds[1].choices[preds[1].answer], /towards 0/);
});

test("student slider runs from the early to the later student and KL falls", () => {
  const { teacher, early, late } = defs["l2-kl-distill"].input;
  assert.deepEqual(P.studentAt(early, late, 0), early);
  assert.deepEqual(P.studentAt(early, late, 1).map((x) => M.round(x, 3)), late);
  let prev = Infinity;
  for (let i = 0; i <= 10; i++) {
    const s = P.studentAt(early, late, i / 10);
    assert.ok(Math.abs(s.reduce((a, b) => a + b, 0) - 1) < 1e-9, "student still sums to 1");
    const k = M.klDivergence(teacher, s).total;
    assert.ok(k < prev && k >= 0);
    prev = k;
  }
});

test("masks: 3 distinct positions; static cycles through 10 copies", () => {
  for (let i = 0; i < 50; i++) {
    const m = P.drawMask(20, 3);
    assert.equal(new Set(m).size, 3);
    assert.ok(m.every((x) => x >= 0 && x < 20));
  }
  const copies = Array.from({ length: 10 }, (_, i) => [i, i + 1, i + 2]);
  const seen = new Set();
  for (let e = 0; e < 40; e++) seen.add(P.maskKey(P.maskForEpoch("static", e, copies, Math.random, 20, 3)));
  assert.equal(seen.size, 10);
});

test("every walkthrough has 4 to 7 steps and 1 or 2 predict stops", () => {
  for (const id of ["l2-roberta-batches", "l2-dynamic-masking", "l2-kl-distill"]) {
    const { steps } = built(id);
    assert.ok(steps.length >= 4 && steps.length <= 7, id);
    const n = steps.filter((s) => s.predict).length;
    assert.ok(n >= 1 && n <= 2, id);
    for (const s of steps) if (s.predict) assert.ok(s.predict.answer < s.predict.choices.length);
  }
});

test("fragment has the subparts and mounts every Part 10 walkthrough", () => {
  const html = readFileSync(new URL("../.fragments/l2-p10.html", import.meta.url), "utf8");
  for (let i = 1; i <= 8; i++) assert.ok(html.includes(`id="p10-${i}"`), "p10-" + i);
  for (const id of ["l2-roberta-batches", "l2-dynamic-masking", "l2-kl-distill"]) {
    assert.ok(html.includes(`data-walk="${id}"`), id);
  }
  for (const href of ["#p2-6", "#p3", "#p4", "#p6"]) assert.ok(html.includes(`href="${href}"`), href);
  assert.ok(!html.includes(EM));
});

test("no em dash in any Part 10 walkthrough text", () => {
  for (const id of ["l2-roberta-batches", "l2-dynamic-masking", "l2-kl-distill"]) {
    const def = defs[id];
    assert.ok(!(JSON.stringify(def.build(def.input)) + def.setup).includes(EM), id);
  }
});
