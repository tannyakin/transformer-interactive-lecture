// Lecture 2, Part 9 (BERT): every walkthrough number must match the notes.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const EM = String.fromCharCode(0x2014);
require("../js/core/math.js");
const { defs, P9 } = require("../js/lecture-2/part-9.js");

function block(steps, key) {
  for (const s of steps) for (const b of s.blocks) if (b.key === key) return b;
  throw new Error("no block with key " + key);
}
const built = (id) => defs[id].build(defs[id].input);

test("BERT-Base parameter count matches the notes", () => {
  const { steps, takeaway } = built("l2-bert-params");
  assert.equal(block(steps, "attn").value, 2359296);
  assert.equal(block(steps, "ffn").value, 4718592);
  assert.deepEqual(block(steps, "layerShare").values.map((v) => +v.toFixed(3)), [0.333, 0.667]);
  assert.equal(block(steps, "stack").value, 12 * 7077888);
  assert.equal(Math.round(block(steps, "stack").value / 1e6), 85);
  assert.equal(block(steps, "tok").value, 30522 * 768);
  assert.equal(+(block(steps, "tok").value / 1e6).toFixed(1), 23.4);
  assert.equal(+(block(steps, "posSeg").value / 1e6).toFixed(1), 0.4);
  assert.equal(Math.round(block(steps, "total").value / 1e6), 109);
  const text = JSON.stringify(steps);
  for (const s of ["2,359,296", "4,718,592", "7.08 million", "85 million", "23.4 million", "0.4 million", "109 million"]) assert.ok(text.includes(s), s);
  assert.ok(takeaway.includes("110M"));
});

test("BERT params predicts: FFN is twice attention, and the FFN is the biggest piece", () => {
  const { steps } = built("l2-bert-params");
  const asks = steps.filter((s) => s.predict);
  assert.equal(asks.length, 2);
  assert.equal(asks[0].predict.choices[asks[0].predict.answer], "About twice as many");
  assert.match(asks[1].predict.choices[asks[1].predict.answer], /^FFN/);
  assert.ok(steps.length >= 4 && steps.length <= 7);
});

test("MLM on 200 tokens: 30 chosen, 24 masked, 3 random, 3 unchanged", () => {
  const { steps } = built("l2-mlm");
  assert.equal(block(steps, "chosen").value, 30);
  assert.equal(block(steps, "mask").value, 24);
  assert.deepEqual(block(steps, "split").value, [24, 3, 3]);
  assert.equal(block(steps, "predict").value, 30);
  assert.deepEqual(block(steps, "shares").values.map((v) => +v.toFixed(3)), [0.12, 0.015, 0.015, 0.85]);
  const asks = steps.filter((s) => s.predict);
  assert.equal(asks.length, 2);
  assert.equal(asks[0].predict.choices[asks[0].predict.answer], "24");
  assert.match(asks[1].predict.choices[asks[1].predict.answer], /^30/);
});

test("MLM panel layout marks exactly 24 / 3 / 3 cells", () => {
  const kinds = P9.mlmLayout(defs["l2-mlm"].input);
  const count = (k) => kinds.filter((x) => x === k).length;
  assert.deepEqual([count("mask"), count("random"), count("keep"), count("plain")], [24, 3, 3, 170]);
});

test("sentiment pipeline follows the video's ten steps", () => {
  const { steps, takeaway } = built("l2-bert-sentiment");
  assert.equal(block(steps, "lower").value, "this teddy bear is so cute!");
  assert.deepEqual(block(steps, "tokens").rows[0], ["this", "teddy", "bear", "is", "so", "cute", "!"]);
  const padded = block(steps, "padded");
  assert.equal(padded.value, 6);
  assert.deepEqual(padded.rows[0], ["[CLS]", "this", "teddy", "bear", "is", "so", "cute", "!", "[SEP]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]"]);
  assert.deepEqual(block(steps, "positions").rows[2], Array.from({ length: 15 }, (_, i) => i));
  const seg = block(steps, "segments");
  assert.deepEqual(seg.value, [9, 6]);
  assert.equal(seg.rows[3].join(""), "AAAAAAAAABBBBBB");
  assert.equal(block(steps, "sum").value, 15);
  assert.equal(block(steps, "sentiment").value, 1);
  const asks = steps.filter((s) => s.predict);
  assert.equal(asks.length, 2);
  assert.equal(asks[0].predict.choices[asks[0].predict.answer], "6");
  assert.equal(asks[1].predict.choices[asks[1].predict.answer], "[CLS]");
  assert.ok(steps.length <= 7);
  assert.ok(takeaway.includes("pretrained"));
});

test("sizes table matches the notes", () => {
  const rows = P9.SIZES.map((m) => [m.name, m.L, m.H, m.A, m.params / 1e6]);
  assert.deepEqual(rows, [
    ["BERT-Tiny", 2, 128, 2, 4],
    ["BERT-Mini", 4, 256, 4, 11],
    ["BERT-Small", 4, 512, 8, 30],
    ["BERT-Medium", 8, 512, 8, 42],
    ["BERT-Base", 12, 768, 12, 110],
    ["BERT-Large", 24, 1024, 16, 340],
  ]);
});

test("input example: segments and positions match the notes", () => {
  assert.equal(P9.INPUT.tokens.join(" "), "[CLS] my dog is cute [SEP] he likes playing [SEP]");
  assert.equal(P9.INPUT.segments.join(""), "AAAAAABBBB");
  assert.deepEqual(P9.INPUT.positions, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
});

test("bank: left to right cannot resolve either sentence, both sides resolves both", () => {
  for (const s of P9.BANK) {
    assert.equal(P9.bankView(s, "left").resolved, false);
    assert.equal(P9.bankView(s, "both").resolved, true);
    assert.equal(s.words[s.focus], "bank");
  }
});

test("masking sampler picks 15% and splits roughly 80 / 10 / 10", () => {
  const tokens = P9.MASK_SENTENCE.split(" ");
  const rand = P9.rng(1);
  const tally = { mask: 0, random: 0, keep: 0 };
  for (let r = 0; r < 2000; r++) {
    const out = P9.sampleMask(tokens, rand);
    const chosen = out.filter((t) => t.kind !== "plain");
    assert.equal(chosen.length, Math.round(0.15 * tokens.length));
    for (const t of chosen) {
      tally[t.kind]++;
      if (t.kind === "mask") assert.equal(t.shown, "[MASK]");
      if (t.kind === "random") assert.notEqual(t.shown, t.orig);
      if (t.kind === "keep") assert.equal(t.shown, t.orig);
    }
  }
  const all = tally.mask + tally.random + tally.keep;
  assert.ok(Math.abs(tally.mask / all - 0.8) < 0.02);
  assert.ok(Math.abs(tally.random / all - 0.1) < 0.02);
  assert.ok(Math.abs(tally.keep / all - 0.1) < 0.02);
});

test("NSP draws label pairs honestly and builds the notes' input", () => {
  const rand = P9.rng(7);
  let isNext = 0;
  for (let r = 0; r < 1000; r++) {
    const d = P9.nspDraw(rand);
    const truth = P9.NSP_TEXT.find((p) => p[0] === d.a)[1];
    assert.equal(d.label, d.b === truth ? "IsNext" : "NotNext");
    if (d.label === "IsNext") isNext++;
  }
  assert.ok(isNext > 420 && isNext < 580);
  const p = P9.pairTokens("the man went to the store", "he bought milk");
  assert.equal(p.tokens.join(" "), "[CLS] the man went to the store [SEP] he bought milk [SEP]");
});

test("part 9 files contain no em dashes", () => {
  for (const f of ["../js/lecture-2/part-9.js", "../css/l2/part-9.css", "../.fragments/l2-p9.html"]) {
    assert.ok(!readFileSync(new URL(f, import.meta.url), "utf8").includes(EM), f);
  }
  for (const id of ["l2-bert-params", "l2-mlm", "l2-bert-sentiment"]) {
    assert.ok(!(JSON.stringify(built(id)) + defs[id].setup).includes(EM), id);
  }
});
