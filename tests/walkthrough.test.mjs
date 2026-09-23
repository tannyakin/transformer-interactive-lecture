import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const W = require("../js/core/walkthrough.js");

const words = (n) => Array.from({ length: n }, (_, i) => "w" + i).join(" ");

test("a short or empty step still stays up for four seconds", () => {
  assert.equal(W.dwellMs("", 1), 4000);
  assert.equal(W.dwellMs("Add the bias.", 1), 4000);
});

test("longer narration earns more reading time", () => {
  assert.equal(W.dwellMs(words(50), 1), 5500);
});

test("speed divides the dwell time", () => {
  assert.equal(Math.round(W.dwellMs(words(50), 1.5)), 3667);
  assert.equal(W.dwellMs("", 0.75), 4000 / 0.75);
});

test("HTML tags are not counted as words", () => {
  assert.equal(W.dwellMs(`<b>${words(50)}</b> <span class="x"></span>`, 1), 5500);
});

const def = {
  id: "demo",
  title: "Demo",
  setup: "Here is the problem.",
  input: { k: 2 },
  build: (input) => ({
    steps: [
      { title: "One", say: "first", blocks: [{ type: "note", html: "a" }] },
      { title: "Two", say: "second", blocks: [{ type: "note", html: "k=" + input.k }] },
    ],
    takeaway: "What we learned.",
  }),
};

test("assemble wraps built steps with a setup card and a takeaway card", () => {
  const steps = W.assemble(def, def.input);
  assert.equal(steps.length, 4);
  assert.deepEqual(steps.map((s) => s.kind), ["setup", "step", "step", "takeaway"]);
  assert.equal(steps[0].say, "Here is the problem.");
  assert.equal(steps[3].say, "What we learned.");
});

test("the takeaway card keeps the final numbers on screen", () => {
  const steps = W.assemble(def, { k: 7 });
  assert.deepEqual(steps[3].blocks, steps[2].blocks);
  assert.equal(steps[3].blocks[0].html, "k=7");
});

test("the active player is the last one touched while it is still visible", () => {
  const a = { ratio: 0.9, touchedAt: 0 };
  const b = { ratio: 0.4, touchedAt: 50 };
  assert.equal(W.pickActive([a, b]), b);
});

test("a touched player that has scrolled away gives way to the most visible one", () => {
  const a = { ratio: 0.9, touchedAt: 0 };
  const b = { ratio: 0.1, touchedAt: 50 };
  const c = { ratio: 0.5, touchedAt: 0 };
  assert.equal(W.pickActive([a, b, c]), a);
});

test("no visible player means nothing is active", () => {
  assert.equal(W.pickActive([{ ratio: 0, touchedAt: 9 }, { ratio: 0.2, touchedAt: 0 }]), null);
  assert.equal(W.pickActive([]), null);
});
