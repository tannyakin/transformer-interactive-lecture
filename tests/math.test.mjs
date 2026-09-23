// Every expected value below is printed in the Lecture 2 notes.
// If one of these fails, a walkthrough would show a student a wrong number.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const M = require("../js/core/math.js");
const r = (xs, dp = 3) => xs.map((x) => M.round(x, dp));

test("sinusoidal PE for d = 4 matches positions 0 to 3", () => {
  assert.deepEqual(r(M.sinusoidalPE(0, 4)), [0, 1, 0, 1]);
  assert.deepEqual(r(M.sinusoidalPE(1, 4)), [0.841, 0.54, 0.01, 1]);
  assert.deepEqual(r(M.sinusoidalPE(2, 4)), [0.909, -0.416, 0.02, 1]);
  assert.deepEqual(r(M.sinusoidalPE(3, 4)), [0.141, -0.99, 0.03, 1]);
});

test("PE dot products fall with distance and repeat at equal distance", () => {
  const pe = (p) => M.sinusoidalPE(p, 4);
  assert.equal(M.round(M.dot(pe(0), pe(1))), 1.54);
  assert.equal(M.round(M.dot(pe(1), pe(2))), 1.54);
  assert.equal(M.round(M.dot(pe(2), pe(3))), 1.54);
  assert.equal(M.round(M.dot(pe(0), pe(2))), 0.584);
  assert.equal(M.round(M.dot(pe(0), pe(3))), 0.01);
});

test("adding PE(1) to the embedding for cat", () => {
  assert.deepEqual(r(M.add([0.2, 0.5, -0.1, 0.3], M.sinusoidalPE(1, 4))), [1.041, 1.04, -0.09, 1.3]);
});

test("T5 bias softmax", () => {
  const s = M.softmax([1.5, 1, 3.5, 3]);
  assert.deepEqual(r(s.exps), [4.482, 2.718, 33.115, 20.086]);
  assert.equal(M.round(s.sum), 60.401);
  assert.deepEqual(r(s.weights), [0.074, 0.045, 0.548, 0.333]);
});

test("ALiBi bias, adjusted softmax and the no-ALiBi comparison", () => {
  const bias = M.alibiBias(3, [0, 1, 2, 3], 0.5);
  assert.deepEqual(r(bias), [-1.5, -1, -0.5, 0]);
  const s = M.softmax(M.add([2, 1, 3, 2], bias));
  assert.equal(M.round(s.sum), 22.22);
  assert.deepEqual(r(s.weights), [0.074, 0.045, 0.548, 0.333]);
  const raw = M.softmax([2, 1, 3, 2]);
  assert.equal(M.round(raw.sum), 37.582);
  assert.deepEqual(r(raw.weights), [0.197, 0.072, 0.534, 0.197]);
});

test("RoPE scores depend only on distance", () => {
  const score = (a, b) => M.round(M.dot(M.rotate([1, 0], a), M.rotate([1, 0], b)));
  assert.equal(score(60, 30), 0.866);
  assert.equal(score(150, 120), 0.866);
  assert.equal(score(120, 30), 0);
});

test("residual add and vanishing gradient", () => {
  assert.deepEqual(r(M.add([2, 1], [0.3, -0.2])), [2.3, 0.8]);
  assert.equal(M.round(Math.pow(0.5, 10), 5), 0.00098);
});

test("layer norm, including scale invariance", () => {
  const ln = M.layerNorm([2, 4, 6, 8]);
  assert.equal(ln.mean, 5);
  assert.deepEqual(ln.centered, [-3, -1, 1, 3]);
  assert.equal(ln.variance, 5);
  assert.equal(M.round(ln.std), 2.236);
  assert.deepEqual(r(ln.normalized), [-1.342, -0.447, 0.447, 1.342]);
  assert.deepEqual(r(M.layerNorm([20, 40, 60, 80]).normalized), [-1.342, -0.447, 0.447, 1.342]);
});

test("RMSNorm", () => {
  const n = M.rmsNorm([2, 4, 6, 8]);
  assert.deepEqual(n.squares, [4, 16, 36, 64]);
  assert.equal(n.meanSquare, 30);
  assert.equal(M.round(n.rms), 5.477);
  assert.deepEqual(r(n.normalized), [0.365, 0.73, 1.095, 1.461]);
});

test("FFN by hand, and the twist input", () => {
  const W1 = [[1, -1, 0, 2], [0, 1, -1, 1]];
  const W2 = [[1, 0], [0, 1], [1, 1], [0.5, -0.5]];
  const a = M.ffn([1, 2], W1, W2);
  assert.deepEqual(a.hidden, [1, 1, -2, 4]);
  assert.deepEqual(a.activated, [1, 1, 0, 4]);
  assert.deepEqual(a.out, [3, -1]);
  const b = M.ffn([1, -2], W1, W2);
  assert.deepEqual(b.hidden, [1, -3, 2, 0]);
  assert.deepEqual(b.activated, [1, 0, 2, 0]);
});

test("causal mask gives future keys exactly zero", () => {
  const s = M.softmax([1, 3, -Infinity, -Infinity]);
  assert.equal(M.round(s.sum), 22.804);
  assert.deepEqual(r(s.weights), [0.119, 0.881, 0, 0]);
});

test("KV cache for a Llama 2 7B sized model", () => {
  const perToken = M.kvCacheBytes({ layers: 32, kvHeads: 32, headDim: 128 });
  assert.equal(perToken, 524288);
  assert.equal(M.kvCacheBytes({ layers: 32, kvHeads: 32, headDim: 128, tokens: 4096 }), 2147483648);
});

test("number of possible masks, C(20, 3)", () => {
  assert.equal(M.choose(20, 3), 1140);
});

test("KL divergence early and late in distillation", () => {
  const teacher = [0.25, 0.7, 0.05];
  const early = M.klDivergence(teacher, [0.4, 0.5, 0.1]);
  assert.deepEqual(r(early.terms, 4), [-0.1175, 0.2355, -0.0347]);
  assert.equal(M.round(early.total, 4), 0.0834);
  assert.equal(M.round(M.klDivergence(teacher, [0.27, 0.68, 0.05]).total, 4), 0.0011);
});

test("display formatting never shows negative zero or float noise", () => {
  assert.equal(M.fmt(-1.5, 1), "−1.5");
  assert.equal(M.fmt(-0.00001, 3), "0.000");
  assert.equal(M.fmt(0.1 + 0.2, 1), "0.3");
  assert.equal(M.fmtInt(10000000000), "10,000,000,000");
});
