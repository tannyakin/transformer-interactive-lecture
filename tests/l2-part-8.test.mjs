// Part 8 (architecture families): causal masking must match the notes exactly.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const M = require("../js/core/math.js");
require("../js/lecture-2/part-8.js");
const defs = globalThis.Lecture.defs;
const P8 = globalThis.Lecture.l2p8;
const EM = String.fromCharCode(0x2014);
const r3 = (xs) => xs.map((x) => (isFinite(x) ? M.round(x, 3) : x));

function block(steps, key) {
  for (const s of steps) for (const b of s.blocks) if (b.key === key) return b;
  throw new Error("no block with key " + key);
}

test("causal mask walkthrough shows the notes' mask, exps, sum and weights", () => {
  const def = defs["l2-causal-mask"];
  const { steps } = def.build(def.input);
  assert.deepEqual(block(steps, "maskRow").values, [0, 0, -Infinity, -Infinity]);
  assert.deepEqual(block(steps, "masked").values, [1, 3, -Infinity, -Infinity]);
  assert.deepEqual(r3(block(steps, "exps").values), [2.718, 20.086, 0, 0]);
  assert.equal(M.round(block(steps, "sum").value, 3), 22.804);
  assert.deepEqual(r3(block(steps, "weights").values), [0.119, 0.881, 0, 0]);
  assert.ok(steps.length >= 4 && steps.length <= 7);
});

test("causal mask walkthrough stops to ask what e to the minus infinity is", () => {
  const def = defs["l2-causal-mask"];
  const { steps } = def.build(def.input);
  const asking = steps.filter((s) => s.predict);
  assert.equal(asking.length, 1);
  assert.equal(asking[0].predict.choices[asking[0].predict.answer], "0");
  assert.ok(asking[0].blocks.some((b) => b.key === "exps"));
});

test("the full 4 x 4 causal mask blocks everything above the diagonal", () => {
  const def = defs["l2-causal-mask"];
  const { steps } = def.build(def.input);
  const rows = block(steps, "mask").rows;
  rows.forEach((row, q) => row.forEach((v, k) => assert.equal(v, k <= q ? 0 : -Infinity)));
});

test("each family carries the right mask pattern", () => {
  const byId = Object.fromEntries(P8.FAMILIES.map((f) => [f.id, f]));
  assert.deepEqual(byId.encoder.masks.map((m) => m.kind), ["bidirectional"]);
  assert.deepEqual(byId.decoder.masks.map((m) => m.kind), ["causal"]);
  assert.deepEqual(byId.encdec.masks.map((m) => m.kind), ["bidirectional", "causal", "cross"]);
  const causal = P8.maskMatrix("causal", 4);
  assert.deepEqual(causal.map((r) => r.filter(Boolean).length), [1, 2, 3, 4]);
  assert.ok(P8.maskMatrix("bidirectional", 4).flat().every(Boolean));
  assert.ok(P8.maskMatrix("cross", 4, 4).flat().every(Boolean));
});

test("Part 8 text has no em dash and the page wiring is consistent", () => {
  const def = defs["l2-causal-mask"];
  const text = JSON.stringify(def.build(def.input)) + def.setup + JSON.stringify(P8.FAMILIES);
  assert.ok(!text.includes(EM));
  const frag = fileURLToPath(new URL("../.fragments/l2-p8.html", import.meta.url));
  if (!existsSync(frag)) return;
  const html = readFileSync(frag, "utf8");
  assert.ok(!html.includes(EM));
  assert.ok(html.includes('data-walk="l2-causal-mask"'));
  for (const id of ["p8-1", "p8-2", "p8-3", "p8-4", "p8-5", "l2p8-mask", "l2p8-families"]) assert.ok(html.includes(`id="${id}"`), id);
});
