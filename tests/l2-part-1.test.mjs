// Lecture 2, Part 1: order disappears without position.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const M = require("../js/core/math.js");
require("../js/lecture-2/part-1.js");
const defs = globalThis.Lecture.defs;
const P1 = globalThis.Lecture.l2p1;
const FRAGMENT = readFileSync(new URL("../lecture-2.html", import.meta.url), "utf8");

const r3 = (xs) => xs.map((x) => M.round(x, 3));
function block(steps, key) {
  for (const s of steps) for (const b of s.blocks) if (b.key === key) return b;
  throw new Error("no block with key " + key);
}

test("dog to man scores the same in both sentences without position", () => {
  const def = defs["l2-order"];
  const { steps } = def.build(def.input);
  assert.deepEqual(r3(block(steps, "qA").values), [1, 0.5]);
  assert.deepEqual(r3(block(steps, "kA").values), [1, 1]);
  assert.deepEqual(r3(block(steps, "scoreA").values), [1.5]);
  assert.deepEqual(r3(block(steps, "scoreB").values), [1.5]);
});

test("the second grid is the first with rows and columns reordered", () => {
  const def = defs["l2-order"];
  const { steps } = def.build(def.input);
  const A = block(steps, "gridA");
  const B = block(steps, "gridB");
  assert.deepEqual(A.rows, [[1, 1, 1.5], [1, 0, 1], [1.25, 0.5, 1.5]]);
  A.rowLabels.forEach((q, r) =>
    A.colLabels.forEach((k, c) => {
      assert.equal(B.rows[B.rowLabels.indexOf(q)][B.colLabels.indexOf(k)], A.rows[r][c]);
    })
  );
});

test("adding position breaks the tie", () => {
  const def = defs["l2-order"];
  const { steps } = def.build(def.input);
  assert.deepEqual(r3(block(steps, "posScores").values), [1.75, 2]);
  const a = P1.scoreGrid(P1.SENTENCES.a, true).flat().sort();
  const b = P1.scoreGrid(P1.SENTENCES.b, true).flat().sort();
  assert.notDeepEqual(a, b);
});

test("one predict stop, answered by 'exactly the same'", () => {
  const def = defs["l2-order"];
  const { steps } = def.build(def.input);
  assert.ok(steps.length >= 4 && steps.length <= 7);
  const asking = steps.filter((s) => s.predict);
  assert.equal(asking.length, 1);
  assert.match(asking[0].predict.choices[asking[0].predict.answer], /^Exactly 1\.5$/);
  assert.ok(asking[0].blocks.some((b) => b.key === "scoreB"));
});

test("fragment has the subparts, interactives and walkthrough", () => {
  for (const id of ["p1-1", "p1-2", "p1-3", "p1-4", "l2p1-shuffle", "l2p1-links"]) assert.ok(FRAGMENT.includes(`id="${id}"`), id);
  assert.ok(FRAGMENT.includes('data-walk="l2-order"'));
  for (const t of ["RNN", "LSTM"]) assert.ok(FRAGMENT.includes(`<abbr data-term="${t}">${t}</abbr>`), t);
});

test("no em dashes in Part 1 text", () => {
  const def = defs["l2-order"];
  assert.ok(!(JSON.stringify(def.build(def.input)) + def.setup + FRAGMENT).includes(String.fromCharCode(0x2014)));
});
