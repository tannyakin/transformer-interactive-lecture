// The Lecture 1 scaled dot-product attention walkthrough must show numbers
// that match an independent, plain-loop computation of the same example.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const M = require("../js/core/math.js");
require("../js/lecture-1.js");
const def = globalThis.Lecture.defs["l1-sdpa"];

const r3 = (xs) => xs.map((x) => (isFinite(x) ? Math.round(x * 1000) / 1000 : x));
function block(steps, key) {
  for (const s of steps) for (const b of s.blocks) if (b.key === key) return b;
  throw new Error("no block with key " + key);
}

// Written from scratch with loops, sharing no code with math.js or lecture-1.js.
function reference({ q, K, V, query, mask }) {
  const scores = [];
  for (const k of K) {
    let s = 0;
    for (let i = 0; i < q.length; i++) s += q[i] * k[i];
    scores.push(s);
  }
  const scaled = scores.map((s) => s / Math.sqrt(q.length));
  const masked = scaled.map((s, j) => (mask && j > query ? -Infinity : s));
  const exps = masked.map((s) => Math.exp(s));
  let sum = 0;
  for (const e of exps) sum += e;
  const weights = exps.map((e) => e / sum);
  const out = [];
  for (let c = 0; c < V[0].length; c++) {
    let acc = 0;
    for (let j = 0; j < V.length; j++) acc += weights[j] * V[j][c];
    out.push(acc);
  }
  return { scores, scaled, masked, exps, weights, out };
}

const plainInput = def.input;
const maskedInput = Object.assign({}, def.input, def.twist.input);

test("SDPA without a mask matches the independent computation", () => {
  const { steps } = def.build(plainInput);
  const ref = reference(plainInput);
  assert.deepEqual(block(steps, "scores").values, ref.scores);
  assert.deepEqual(r3(block(steps, "scaled").values), r3(ref.scaled));
  assert.deepEqual(r3(block(steps, "exps").values), r3(ref.exps));
  assert.deepEqual(r3(block(steps, "weights").values), r3(ref.weights));
  assert.deepEqual(r3(block(steps, "output").values), r3(ref.out));
});

test("SDPA without a mask shows the expected hand-checkable values", () => {
  const { steps } = def.build(plainInput);
  assert.deepEqual(block(steps, "scores").values, [1, 0, 2]);
  assert.deepEqual(r3(block(steps, "scaled").values), [0.5, 0, 1]);
  assert.deepEqual(r3(block(steps, "weights").values), [0.307, 0.186, 0.506]);
  assert.deepEqual(r3(block(steps, "output").values), [0.814, 0.693]);
});

test("the causal mask twist hides the future word and renormalises", () => {
  const { steps } = def.build(maskedInput);
  const ref = reference(maskedInput);
  const masked = block(steps, "masked").values;
  assert.equal(masked[2], -Infinity);
  assert.deepEqual(r3(masked.slice(0, 2)), [0.5, 0]);
  assert.deepEqual(r3(block(steps, "weights").values), r3(ref.weights));
  assert.deepEqual(r3(block(steps, "weights").values), [0.622, 0.378, 0]);
  assert.deepEqual(r3(block(steps, "output").values), [0.622, 0.378]);
});

test("weights add up to 1 with and without the mask", () => {
  for (const input of [plainInput, maskedInput]) {
    const w = block(def.build(input).steps, "weights").values;
    assert.ok(Math.abs(w.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  }
});

test("one predict stop, asked before the softmax weights are revealed", () => {
  for (const [input, winner] of [[plainInput, 2], [maskedInput, 0]]) {
    const { steps } = def.build(input);
    const asking = steps.filter((s) => s.predict);
    assert.equal(asking.length, 1);
    assert.equal(asking[0].predict.answer, winner);
    assert.ok(asking[0].blocks.some((b) => b.key === "weights"));
    assert.equal(asking[0].predict.choices.length, input.words.length);
  }
});

test("each step lights one diagram box, in the order the maths runs", () => {
  const { steps } = def.build(plainInput);
  assert.deepEqual(steps.map((s) => s.box), ["mm1", "scale", "mask", "softmax", "mm2"]);
  assert.equal(typeof def.panel, "function");
});

test("math.js softmax agrees with the reference for this example", () => {
  const ref = reference(plainInput);
  assert.deepEqual(r3(M.softmax(ref.masked).weights), r3(ref.weights));
});

test("no em dash in any step, setup or takeaway", () => {
  for (const input of [plainInput, maskedInput]) {
    const text = JSON.stringify(def.build(input)) + def.setup + def.title + def.twist.label + def.twist.offLabel;
    assert.ok(!text.includes(String.fromCharCode(0x2014)));
  }
});
