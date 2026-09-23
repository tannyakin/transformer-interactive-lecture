// Part 4 (the FFN): walkthrough and interactive numbers must match the notes.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const M = require("../js/core/math.js");
require("../js/lecture-2/part-4.js");
const defs = globalThis.Lecture.defs;
const P4 = globalThis.Lecture.l2p4;
const EM = String.fromCharCode(0x2014);

const r3 = (xs) => xs.map((x) => M.round(x, 3));
function block(steps, key) {
  for (const s of steps) for (const b of s.blocks) if (b.key === key) return b;
  throw new Error("no block with key " + key);
}
const twisted = (def) => Object.assign({}, def.input, def.twist.input);

test("FFN by hand: xW1 = [1,1,-2,4], ReLU [1,1,0,4], output [3,-1], residual [4,1]", () => {
  const def = defs["l2-ffn"];
  const { steps } = def.build(def.input);
  assert.deepEqual(block(steps, "hidden").values, [1, 1, -2, 4]);
  assert.deepEqual(block(steps, "activated").values, [1, 1, 0, 4]);
  assert.deepEqual(r3(block(steps, "out").values), [3, -1]);
  assert.deepEqual(r3(block(steps, "residual").values), [4, 1]);
  assert.ok(steps.length >= 4 && steps.length <= 7);
});

test("FFN asks which features survive before ReLU is revealed", () => {
  const def = defs["l2-ffn"];
  const { steps } = def.build(def.input);
  const asking = steps.filter((s) => s.predict);
  assert.equal(asking.length, 1);
  const p = asking[0].predict;
  assert.equal(p.choices[p.answer], "Features 1, 2 and 4");
  assert.ok(asking[0].blocks.some((b) => b.key === "activated"));
});

test("FFN twist x = [1, -2]: xW1 = [1,-3,2,0], ReLU [1,0,2,0], a different set switches on", () => {
  const def = defs["l2-ffn"];
  const { steps } = def.build(twisted(def));
  assert.deepEqual(block(steps, "hidden").values, [1, -3, 2, 0]);
  assert.deepEqual(block(steps, "activated").values, [1, 0, 2, 0]);
  const p = steps.find((s) => s.predict).predict;
  assert.equal(p.choices[p.answer], "Features 1 and 3");
});

test("parameter share: 4d^2 = 1,048,576 and 2 d d_ff = 2,097,152 at d = 512, and 2:1 holds", () => {
  assert.equal(P4.attnParams(512), 1048576);
  assert.equal(P4.ffnParams(512, 2048), 2097152);
  for (const d of [128, 768, 4096]) assert.equal(P4.ffnParams(d, 4 * d) / P4.attnParams(d), 2);
  assert.equal(Math.round(P4.swigluParams(512, (8 / 3) * 512)), 2097152, "SwiGLU with 8/3 d_ff keeps the budget");
});

test("activations: ReLU cuts negatives, GELU and Swish let a little through", () => {
  assert.equal(P4.relu1(-0.5), 0);
  assert.ok(P4.gelu(-0.5) < 0 && P4.gelu(-0.5) > -0.2);
  assert.ok(Math.abs(P4.gelu(3) - 3) < 0.01);
  assert.equal(M.round(P4.swish(1), 3), 0.731);
});

test("MoE router picks exactly 2 distinct experts out of 8, repeatably", () => {
  const a = P4.route("cat@1", 8, 2);
  const b = P4.route("cat@1", 8, 2);
  assert.deepEqual(a.top, b.top);
  assert.equal(new Set(a.top).size, 2);
  assert.ok(Math.abs(a.probs.reduce((s, p) => s + p, 0) - 1) < 1e-9);
});

test("no em dash in the FFN walkthrough text", () => {
  const def = defs["l2-ffn"];
  for (const input of [def.input, twisted(def)]) {
    assert.ok(!(JSON.stringify(def.build(input)) + def.setup + def.title).includes(EM));
  }
});
