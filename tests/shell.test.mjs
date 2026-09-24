import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const S = require("../js/core/shell.js");

// tops: each part's distance from the top of the viewport; line: the reading line.
test("the current part is the last one whose top has passed the reading line", () => {
  assert.equal(S.currentIndex([-900, -200, 150, 900], 240), 2);
  assert.equal(S.currentIndex([-900, -200, 400, 900], 240), 1);
});

test("before the first part (still in the hero) there is no current part", () => {
  assert.equal(S.currentIndex([500, 1400, 2600], 240), -1);
});

test("a part exactly on the reading line counts as current", () => {
  assert.equal(S.currentIndex([-100, 240, 900], 240), 1);
});

test("the device tip shows on phone-sized screens until it has been dismissed", () => {
  assert.equal(S.needsDeviceTip(375, false), true);
  assert.equal(S.needsDeviceTip(699, false), true);
  assert.equal(S.needsDeviceTip(375, true), false);
});

test("the device tip stays away on tablet, laptop and desktop widths", () => {
  assert.equal(S.needsDeviceTip(700, false), false);
  assert.equal(S.needsDeviceTip(820, false), false);
  assert.equal(S.needsDeviceTip(1440, false), false);
});

test("sub-section lists parse from the data-sub attribute", () => {
  assert.deepEqual(S.parseSubs("p2-1:Learned positions|p2-2:Sinusoidal: clock hands"), [
    { id: "p2-1", title: "Learned positions" },
    { id: "p2-2", title: "Sinusoidal: clock hands" },
  ]);
  assert.deepEqual(S.parseSubs(""), []);
  assert.deepEqual(S.parseSubs(undefined), []);
});
