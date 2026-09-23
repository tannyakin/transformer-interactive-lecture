import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { createStore, KEY } = require("../js/core/store.js");

function memoryStorage(initial) {
  const data = new Map(initial ? [[KEY, initial]] : []);
  return {
    getItem: (k) => (data.has(k) ? data.get(k) : null),
    setItem: (k, v) => data.set(k, String(v)),
    raw: () => data.get(KEY),
  };
}

test("progress survives a round trip through storage", () => {
  const storage = memoryStorage();
  const a = createStore(storage);
  a.markPart("l2", "p3");
  a.markWalk("l2", "l2-alibi");
  a.setLast("l2", "p3");
  a.setPref("speed", 1.5);

  const b = createStore(storage);
  assert.equal(b.isPartDone("l2", "p3"), true);
  assert.equal(b.isPartDone("l2", "p4"), false);
  assert.equal(b.isWalkDone("l2", "l2-alibi"), true);
  assert.equal(b.getLast("l2"), "p3");
  assert.equal(b.getPref("speed", 1), 1.5);
  assert.equal(JSON.parse(storage.raw()).v, 1);
});

test("invalid JSON in storage starts fresh instead of breaking the page", () => {
  const s = createStore(memoryStorage("{not json"));
  assert.equal(s.isPartDone("l2", "p1"), false);
  assert.equal(s.getLast("l2"), null);
  assert.equal(s.getPref("speed", 1), 1);
});

test("an unknown save version is treated as empty", () => {
  const s = createStore(memoryStorage(JSON.stringify({ v: 99, l2: { parts: { p1: true } } })));
  assert.equal(s.isPartDone("l2", "p1"), false);
});

test("storage that throws on every call never throws out of the store", () => {
  const hostile = {
    getItem() { throw new Error("SecurityError"); },
    setItem() { throw new Error("QuotaExceededError"); },
  };
  const s = createStore(hostile);
  assert.doesNotThrow(() => {
    s.markPart("l2", "p2");
    s.markWalk("l2", "x");
    s.setLast("l2", "p2");
    s.setPref("speed", 2);
  });
  // Still remembered for the rest of this page visit.
  assert.equal(s.isPartDone("l2", "p2"), true);
  assert.equal(s.getLast("l2"), "p2");
});

test("no storage at all behaves like an empty in-memory store", () => {
  const s = createStore(null);
  s.markPart("l1", "p1");
  assert.equal(s.isPartDone("l1", "p1"), true);
});
