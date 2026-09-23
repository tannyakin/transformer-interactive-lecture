// Part 11: one token through a modern LLM, with links back to each part.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";

const require = createRequire(import.meta.url);
require("../js/core/math.js");
require("../js/lecture-2/part-11.js");
const defs = globalThis.Lecture.defs;
const P = globalThis.Lecture.l2p11;
const EM = String.fromCharCode(0x2014);

test("the recipe covers the notes' steps 1 to 7 and 3a to 3h", () => {
  const ids = P.RECIPE.map((r) => r.id);
  assert.deepEqual(ids, ["1", "2", "3a", "3b", "3c", "3d", "3e", "3f", "3g", "3h", "4", "5", "6", "7"]);
});

test("the Llama walkthrough has 6 steps, every recipe line shown once, and two predicts", () => {
  const def = defs["l2-llama-token"];
  const { steps } = def.build(def.input);
  assert.equal(steps.length, 6);
  const shown = steps.flatMap((s) => s.blocks.filter((b) => b.type === "lines").flatMap((b) => b.lines.map((l) => l.t)));
  assert.equal(shown.length, P.RECIPE.length);
  const preds = steps.filter((s) => s.predict).map((s) => s.predict);
  assert.equal(preds.length, 2);
  assert.match(preds[0].choices[preds[0].answer], /rotating Q and K/);
  assert.match(preds[1].choices[preds[1].answer], /cache/);
});

test("each step links back to the part that taught it", () => {
  const def = defs["l2-llama-token"];
  const text = JSON.stringify(def.build(def.input));
  for (const href of ["#p2-6", "#p3", "#p4", "#p7", "#p8"]) assert.ok(text.includes(`href=\\"${href}\\"`), href);
  assert.ok(!(text + def.setup).includes(EM));
});

test("fragment has the subparts, the walkthrough and the toggle table", () => {
  const html = readFileSync(new URL("../lecture-2.html", import.meta.url), "utf8");
  for (let i = 1; i <= 5; i++) assert.ok(html.includes(`id="p11-${i}"`), "p11-" + i);
  assert.ok(html.includes('data-walk="l2-llama-token"'));
  assert.ok(html.includes('id="l2p11-then-now"'));
  assert.ok(html.includes("6 × N × D"));
  assert.ok(!html.includes(EM));
});
