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
    assert.ok(!text.includes("—"));
  }
});
