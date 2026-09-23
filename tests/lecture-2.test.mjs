// Walkthrough definitions for Lecture 2 must show exactly the notes' numbers.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const M = require("../js/core/math.js");
require("../js/lecture-2/part-2.js");
const defs = globalThis.Lecture.defs;

const r3 = (xs) => xs.map((x) => M.round(x, 3));
// Blocks carry a `key` so tests can find them without depending on step order.
function block(steps, key) {
  for (const s of steps) for (const b of s.blocks) if (b.key === key) return b;
  throw new Error("no block with key " + key);
}

test("ALiBi walkthrough shows the notes' bias, scores and weights", () => {
  const def = defs["l2-alibi"];
  const { steps } = def.build(def.input);
  assert.deepEqual(r3(block(steps, "bias").values), [-1.5, -1, -0.5, 0]);
  assert.deepEqual(r3(block(steps, "adjusted").values), [0.5, 0, 2.5, 2]);
  assert.deepEqual(r3(block(steps, "exps").values), [1.649, 1, 12.182, 7.389]);
  assert.deepEqual(r3(block(steps, "weights").values), [0.074, 0.045, 0.548, 0.333]);
  assert.deepEqual(r3(block(steps, "compare").ghost), [0.197, 0.072, 0.534, 0.197]);
});

test("ALiBi asks the learner to predict the winning key before softmax is revealed", () => {
  const def = defs["l2-alibi"];
  const { steps } = def.build(def.input);
  const asking = steps.filter((s) => s.predict);
  assert.equal(asking.length, 1);
  assert.equal(asking[0].predict.answer, 2);
  assert.ok(asking[0].blocks.some((b) => b.key === "weights"), "the predict sits on the step that reveals the weights");
});

test("the twist reruns ALiBi with slope 0.1", () => {
  const def = defs["l2-alibi"];
  const { steps } = def.build(Object.assign({}, def.input, def.twist.input));
  assert.deepEqual(r3(block(steps, "bias").values), [-0.3, -0.2, -0.1, 0]);
  assert.deepEqual(r3(block(steps, "weights").values), [0.165, 0.067, 0.546, 0.222]);
});

test("no step text contains an em dash", () => {
  const def = defs["l2-alibi"];
  for (const input of [def.input, Object.assign({}, def.input, def.twist.input)]) {
    const built = def.build(input);
    const text = JSON.stringify(built) + def.setup;
    assert.ok(!text.includes("2014"));
  }
});

/* ---------------- 2.4 T5 bias ---------------- */
const P = globalThis.Lecture.l2p2;

test("T5 walkthrough shows the notes' biases, scores, sum and weights", () => {
  const def = defs["l2-t5"];
  const { steps } = def.build(def.input);
  assert.ok(steps.length >= 4 && steps.length <= 7);
  assert.deepEqual(block(steps, "distance").values, [3, 2, 1, 0]);
  assert.deepEqual(r3(block(steps, "bias").values), [-0.5, 0, 0.5, 1]);
  assert.deepEqual(r3(block(steps, "adjusted").values), [1.5, 1, 3.5, 3]);
  assert.deepEqual(r3(block(steps, "exps").values), [4.482, 2.718, 33.115, 20.086]);
  assert.deepEqual(r3(block(steps, "sum").values), [60.401]);
  assert.deepEqual(r3(block(steps, "weights").values), [0.074, 0.045, 0.548, 0.333]);
  assert.deepEqual(r3(block(steps, "compare").ghost), [0.197, 0.072, 0.534, 0.197]);
});

test("T5 asks where the distance-0 key ends up, and the answer is a clear second", () => {
  const def = defs["l2-t5"];
  const { steps } = def.build(def.input);
  const asking = steps.filter((s) => s.predict);
  assert.equal(asking.length, 1);
  assert.equal(asking[0].predict.answer, 1);
  assert.ok(asking[0].blocks.some((b) => b.key === "weights"));
});

test("T5 buckets: exact up to 15, logarithmic after, everything from 128 on in bucket 31", () => {
  for (let d = 0; d < 16; d++) assert.equal(P.t5Bucket(d), d);
  assert.equal(P.t5Bucket(16), 16);
  assert.equal(P.t5Bucket(90), P.t5Bucket(95), "90 and 95 share a bucket");
  assert.notEqual(P.t5Bucket(1), P.t5Bucket(2));
  for (const d of [128, 500, 5000, 100000]) assert.equal(P.t5Bucket(d), 31);
  let prev = 0;
  for (let d = 0; d < 400; d++) {
    const b = P.t5Bucket(d);
    assert.ok(b >= prev && b <= 31, "buckets never go down");
    prev = b;
  }
  assert.equal(P.t5BucketRange(31).hi, Infinity);
  assert.deepEqual(P.t5BucketRange(5), { lo: 5, hi: 5 });
});

/* ---------------- 2.5 ALiBi extras ---------------- */
test("ALiBi slopes for 8 heads run 1/2 down to 1/256", () => {
  assert.deepEqual(P.alibiSlopes(8), [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256]);
});

test("T5 and ALiBi scores differ by exactly 1, so their weights match", () => {
  const c = P.shiftCompare();
  assert.deepEqual(r3(c.t5Adj), [1.5, 1, 3.5, 3]);
  assert.deepEqual(r3(c.alAdj), [0.5, 0, 2.5, 2]);
  assert.deepEqual(r3(c.gaps), [1, 1, 1, 1]);
  assert.deepEqual(r3(c.t5W), r3(c.alW));
});

/* ---------------- 2.6 RoPE ---------------- */
test("RoPE walkthrough: cases A, B, C match the notes", () => {
  const def = defs["l2-rope"];
  const { steps } = def.build(def.input);
  assert.ok(steps.length >= 4 && steps.length <= 7);
  assert.deepEqual(r3(block(steps, "A-q").values), [0.5, 0.866]);
  assert.deepEqual(r3(block(steps, "A-k").values), [0.866, 0.5]);
  assert.deepEqual(r3(block(steps, "A-score").values), [0.866]);
  assert.deepEqual(r3(block(steps, "B-q").values), [-0.866, 0.5]);
  assert.deepEqual(r3(block(steps, "B-k").values), [-0.5, 0.866]);
  assert.deepEqual(r3(block(steps, "B-score").values), [0.866]);
  assert.deepEqual(r3(block(steps, "C-q").values), [-0.5, 0.866]);
  assert.deepEqual(r3(block(steps, "C-k").values), [0.866, 0.5]);
  assert.deepEqual(r3(block(steps, "C-score").values), [0]);
  assert.deepEqual(block(steps, "summary").rows.map((r) => r[4]), [0.866, 0.866, 0]);
});

test("RoPE predicts: case B keeps 0.866, case C gives 0", () => {
  const def = defs["l2-rope"];
  const { steps } = def.build(def.input);
  const asking = steps.filter((s) => s.predict);
  assert.equal(asking.length, 2);
  assert.equal(asking[0].predict.choices[asking[0].predict.answer], "0.866");
  assert.ok(asking[0].blocks.some((b) => b.key === "B-score"), "the first predict sits on case B");
  assert.equal(asking[1].predict.choices[asking[1].predict.answer], "0");
});

test("RoPE decay starts at 1, fades with distance, and fades later with base 500,000", () => {
  assert.equal(M.round(P.ropeDecay(0, 128, 10000), 6), 1);
  assert.ok(P.ropeDecay(256, 128, 10000) < P.ropeDecay(10, 128, 10000));
  assert.ok(P.ropeDecay(1000, 128, 500000) > P.ropeDecay(1000, 128, 10000));
  assert.deepEqual([0, 1, 2, 3].map((i) => M.round(P.ropeTheta(i, 8, 10000), 6)), [1, 0.1, 0.01, 0.001]);
});

test("no em dash in the T5 or RoPE walkthrough text", () => {
  for (const id of ["l2-t5", "l2-rope"]) {
    const def = defs[id];
    const text = JSON.stringify(def.build(def.input)) + def.setup + JSON.stringify(def.setupBlocks(def.input));
    assert.ok(!text.includes("2014"), id);
  }
});
