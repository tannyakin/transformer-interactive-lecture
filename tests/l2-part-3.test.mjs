// Part 3 (Add & Norm): walkthroughs must show exactly the notes' numbers.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const M = require("../js/core/math.js");
require("../js/lecture-2/part-3.js");
const defs = globalThis.Lecture.defs;
const P3 = globalThis.Lecture.l2p3;
const EM = String.fromCharCode(0x2014);

const r3 = (xs) => xs.map((x) => M.round(x, 3));
function block(steps, key) {
  for (const s of steps) for (const b of s.blocks) if (b.key === key) return b;
  throw new Error("no block with key " + key);
}
const twisted = (def) => Object.assign({}, def.input, def.twist.input);

test("residual: [2, 1] + [0.3, -0.2] = [2.3, 0.8], and without it the identity is lost", () => {
  const def = defs["l2-residual"];
  const { steps } = def.build(def.input);
  assert.deepEqual(r3(block(steps, "with").values), [2.3, 0.8]);
  assert.deepEqual(r3(block(steps, "without").values), [0.3, -0.2]);
  const asking = steps.filter((s) => s.predict);
  assert.equal(asking.length, 1);
  assert.equal(asking[0].predict.choices[asking[0].predict.answer], "[2.3, 0.8]");
  assert.ok(steps.length >= 4 && steps.length <= 7);
});

test("layer norm: mean 5, centred [-3,-1,1,3], variance 5, std 2.236, normalised values", () => {
  const def = defs["l2-layernorm"];
  const { steps } = def.build(def.input);
  assert.deepEqual(block(steps, "mean").values, [5]);
  assert.deepEqual(block(steps, "centered").values, [-3, -1, 1, 3]);
  assert.deepEqual(block(steps, "squares").values, [9, 1, 1, 9]);
  assert.deepEqual(block(steps, "variance").values, [5]);
  assert.deepEqual(r3(block(steps, "std").values), [2.236]);
  assert.deepEqual(r3(block(steps, "normalized").values), [-1.342, -0.447, 0.447, 1.342]);
  assert.deepEqual(r3(block(steps, "y").values), [-1.342, -0.447, 0.447, 1.342]);
  const p = steps.find((s) => s.predict).predict;
  assert.equal(p.choices[p.answer], "5");
});

test("layer norm twist (x 10): mean 50, deviations [-30,-10,10,30], std 22.36, same result, predict first", () => {
  const def = defs["l2-layernorm"];
  const { steps } = def.build(twisted(def));
  assert.deepEqual(def.twist.input.x, [20, 40, 60, 80]);
  assert.deepEqual(block(steps, "mean").values, [50]);
  assert.deepEqual(block(steps, "centered").values, [-30, -10, 10, 30]);
  assert.equal(P3.sig4(block(steps, "std").values[0]), "22.36");
  assert.deepEqual(r3(block(steps, "normalized").values), [-1.342, -0.447, 0.447, 1.342]);
  const asking = steps.filter((s) => s.predict);
  assert.equal(asking.length, 2);
  const scale = asking[1].predict;
  assert.ok(scale.choices[scale.answer].startsWith("Exactly the same"));
  assert.ok(asking[1].blocks.some((b) => b.key === "normalized"), "the scale predict sits on the step that reveals x-hat");
});

test("RMSNorm: squares, mean 30, rms 5.477, [0.365, 0.730, 1.095, 1.461]", () => {
  const def = defs["l2-rmsnorm"];
  const { steps } = def.build(def.input);
  assert.deepEqual(block(steps, "squares").values, [4, 16, 36, 64]);
  assert.deepEqual(block(steps, "meanSquare").values, [30]);
  assert.deepEqual(r3(block(steps, "rms").values), [5.477]);
  assert.deepEqual(r3(block(steps, "normalized").values), [0.365, 0.73, 1.095, 1.461]);
  const p = steps.find((s) => s.predict).predict;
  assert.ok(p.choices[p.answer].startsWith("All positive"));
  const tw = def.build(twisted(def)).steps;
  assert.deepEqual(r3(block(tw, "normalized").values), [0.365, 0.73, 1.095, 1.461]);
});

test("gradient highway: 0.5^10 = 0.00098 on the plain path, 1 along the shortcut", () => {
  assert.equal(P3.fmtSmall(P3.gradPlain(0.5, 10)), "0.00098");
  assert.equal(P3.gradShortcut(10), 1);
  assert.equal(P3.gradShortcut(24), 1);
});

test("the block diagram: post-norm puts two norms on the highway, pre-norm none", () => {
  assert.equal(P3.normsOnHighway("post"), 2);
  assert.equal(P3.normsOnHighway("pre"), 0);
  assert.equal(P3.blockModel("post").adds.length, 2, "Add & Norm appears twice per block");
});

test("no em dash in any Part 3 walkthrough text", () => {
  for (const id of ["l2-residual", "l2-layernorm", "l2-rmsnorm"]) {
    const def = defs[id];
    const inputs = [def.input].concat(def.twist ? [twisted(def)] : []);
    for (const input of inputs) {
      const text = JSON.stringify(def.build(input)) + def.setup + def.title;
      assert.ok(!text.includes(EM), id);
    }
  }
});
