# Phase 1: Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared foundation every lecture page will use: tested maths helpers, a progress store, the new shell (slim top bar, contents drawer, theme, resume), and the walkthrough engine, proven on a `lecture-2.html` skeleton with the ALiBi walkthrough.

**Architecture:** Classic `defer` scripts share one `window.Lecture` namespace. Pure logic (`math.js`, `store.js`, the engine's pacing and step assembly, walkthrough definitions) also exports through `module.exports` so Node's built-in test runner can check it with no dependencies. The shell reads the page's own `<section class="part">` elements, so the drawer and part counter never drift from the content.

**Tech Stack:** HTML, CSS custom properties, vanilla JS (ES2020), Node 24 `node --test` for tests, Playwright MCP for browser QA.

**Spec:** `docs/specs/2026-09-23-lecture-2-design.md`

## Global Constraints

- No build step. Pages must work from GitHub Pages and when opened from disk (`file://`), so no ES modules and no fetch of local files.
- No em dash (U+2014) anywhere in site files. Plain, human tone in all copy.
- Worked example numbers must equal the notes exactly (3 decimal places unless the notes show otherwise).
- Theme preference key stays `transformer-theme`. Progress key is `unipod-lectures-v1`.
- Every storage access is wrapped in try/catch.
- `prefers-reduced-motion: reduce` turns off autoplay and count-up animation.
- Commit messages are one conventional line, nothing else. Each phase ends with docs updated, commit, push.

## Review Focus

1. Storage blocked or corrupted (private window, bad JSON): the page must load normally with no progress, never throw. Pinned by store tests for a throwing storage and for invalid JSON.
2. Two walkthroughs visible at once: arrow keys must drive only the active one. Pinned by the engine's `pickActive` unit test.
3. A learner scrolls away mid-autoplay: the timer must stop and resume from where it was, not restart the step. Pinned by the browser QA script in Task 5.
4. A twist toggled mid-play: the walkthrough must rebuild from its first real step with the new numbers, and the old timer must not fire into the new steps. Pinned by the browser QA script in Task 5.
5. Negative zero and float noise in displayed numbers (`-0.000`, `0.30000000000000004`): must display as `0.000` and `0.3`. Pinned by `fmt` tests.

---

### Task 1: Maths helpers and number checks

**Files:**
- Create: `js/core/math.js`
- Create: `tests/math.test.mjs`
- Create: `package.json` (only `{"private": true, "scripts": {"test": "node --test tests/"}}`, no dependencies)

**Interfaces:**
- Produces `Lecture.math` (browser) and `module.exports` (Node) with:
  `round(x, dp=3) -> number`, `fmt(x, dp=3) -> string` (typographic minus U+2212, no negative zero), `fmtInt(n) -> string` (thousands commas),
  `dot(a, b)`, `add(a, b)`, `scale(a, k)`, `vecMat(x, W)`, `relu(v)`,
  `softmax(scores) -> {exps, sum, weights}` (unshifted, `-Infinity` gives 0),
  `sinusoidalPE(pos, d, base=10000) -> number[]`, `rotate(v, deg) -> [x, y]`,
  `alibiBias(m, keys, mu) -> number[]`,
  `layerNorm(x, eps=1e-5) -> {mean, centered, variance, std, normalized}`,
  `rmsNorm(x, eps=0) -> {squares, meanSquare, rms, normalized}`,
  `ffn(x, W1, W2) -> {hidden, activated, out}`,
  `klDivergence(p, q) -> {terms, total}`, `choose(n, k)`,
  `kvCacheBytes({layers, kvHeads, headDim, tokens=1, bytes=2}) -> number`.

- [ ] **Step 1: Write the failing tests** in `tests/math.test.mjs`, one `test()` per worked example, loading the module with `createRequire(import.meta.url)("../js/core/math.js")`. Expected values, all from the notes:
  - PE(0..3), d = 4: `[0,1,0,1]`, `[0.841,0.54,0.01,1]`, `[0.909,-0.416,0.02,1]`, `[0.141,-0.99,0.03,1]`; dot products (0,1) (1,2) (2,3) = 1.54, (0,2) = 0.584, (0,3) = 0.01; "cat" + PE(1) = `[1.041,1.04,-0.09,1.3]`.
  - T5: softmax `[1.5,1,3.5,3]` exps `[4.482,2.718,33.115,20.086]`, sum 60.401, weights `[0.074,0.045,0.548,0.333]`.
  - ALiBi: `alibiBias(3,[0,1,2,3],0.5)` = `[-1.5,-1,-0.5,0]`; adjusted sum 22.22; weights equal T5's; raw `[2,1,3,2]` sum 37.582 weights `[0.197,0.072,0.534,0.197]`.
  - RoPE: dot(rotate([1,0],60), rotate([1,0],30)) = 0.866, (150,120) = 0.866, (120,30) = 0.
  - Residual `[2,1]+[0.3,-0.2]` = `[2.3,0.8]`; `0.5^10` rounds to 0.00098 at 5 dp.
  - LayerNorm `[2,4,6,8]`: mean 5, variance 5, std 2.236, normalized `[-1.342,-0.447,0.447,1.342]`; `[20,40,60,80]` gives the same normalized vector.
  - RMSNorm `[2,4,6,8]`: meanSquare 30, rms 5.477, normalized `[0.365,0.73,1.095,1.461]`.
  - FFN with the notes' W1, W2: x = [1,2] gives hidden `[1,1,-2,4]`, activated `[1,1,0,4]`, out `[3,-1]`; x = [1,-2] gives hidden `[1,-3,2,0]`, activated `[1,0,2,0]`.
  - Causal mask: softmax `[1,3,-Infinity,-Infinity]` sum 22.804 weights `[0.119,0.881,0,0]`.
  - KV cache: 32 layers, 32 heads, 128 dim, 2 bytes = 524,288 per token; × 4,096 tokens = 2,147,483,648.
  - `choose(20,3)` = 1140. KL early: terms `[-0.1175,0.2355,-0.0347]` (4 dp), total 0.0834 (4 dp); KL late total 0.0011.
  - `fmt(-1.5,1)` = "−1.5", `fmt(-0.00001,3)` = "0.000", `fmt(0.1+0.2,1)` = "0.3", `fmtInt(10000000000)` = "10,000,000,000".
- [ ] **Step 2: Run** `node --test tests/`. Expected: FAIL, cannot find module `../js/core/math.js`.
- [ ] **Step 3: Implement** `js/core/math.js` as an IIFE taking `root` (`window` or `globalThis`), attaching `root.Lecture.math` and assigning `module.exports` when `module` exists. `round` adds a sign-aware `Number.EPSILON` nudge and maps `-0` to `0`.
- [ ] **Step 4: Run** `node --test tests/`. Expected: all math tests PASS.

### Task 2: Progress store and writing-rule check

**Files:**
- Create: `js/core/store.js`
- Create: `tests/store.test.mjs`
- Create: `tests/content.test.mjs`

**Interfaces:**
- Produces `Lecture.store` = `createStore(storage)` bound to `window.localStorage` (looked up inside try/catch), and `createStore` exported for tests.
- Store methods: `markPart(lec, part)`, `isPartDone(lec, part) -> bool`, `markWalk(lec, id)`, `isWalkDone(lec, id) -> bool`, `setLast(lec, part)`, `getLast(lec) -> string|null`, `getPref(key, fallback)`, `setPref(key, value)`.
- Saved shape: `{"v":1,"prefs":{},"l2":{"parts":{},"walk":{},"last":null}}`.

- [ ] **Step 1: Write failing tests.** `store.test.mjs`: round trip with an in-memory fake storage; invalid JSON in storage returns defaults and `isPartDone` is false; a storage whose `getItem` and `setItem` throw never throws out of any method; `v` other than 1 is treated as empty. `content.test.mjs`: walks `*.html`, `js/**/*.js`, `css/**/*.css`, `README.md`, `docs/**/*.md` and fails listing `file:line` for any U+2014.
- [ ] **Step 2: Run** `node --test tests/`. Expected: store tests FAIL (module missing), content test PASS.
- [ ] **Step 3: Implement** `js/core/store.js`: read parses inside try/catch and validates `v === 1`; every write is `try { storage.setItem(...) } catch (e) {}`; a missing storage (null) behaves like an empty in-memory one for the page's lifetime.
- [ ] **Step 4: Run** `node --test tests/`. Expected: all PASS.

### Task 3: Walkthrough engine

**Files:**
- Create: `js/core/walkthrough.js`
- Create: `css/walkthrough.css`
- Create: `tests/walkthrough.test.mjs`

**Interfaces:**
- Consumes: `Lecture.math.fmt`, `Lecture.store.{markWalk,isWalkDone,getPref,setPref}`.
- Produces:
  - `Lecture.defs` (object): definitions registered by lecture files, keyed by id.
  - `Lecture.Walkthrough.mount(el, def) -> controller` with `goto(i)`, `next()`, `back()`, `play()`, `pause()`, `restart()`, `setTwist(on)`.
  - Auto-mount of every `[data-walk="<id>"]` element on `DOMContentLoaded`.
  - Pure, exported for Node: `dwellMs(text, speed) -> ms` = `max(4000, 2500 + 60 × words) / speed`, words counted after stripping HTML tags; `assemble(def, input) -> steps[]` = setup step, built steps, takeaway step (`kind` = "setup" | "step" | "takeaway"); `pickActive(players) -> player|null` choosing the last interacted player if still visible, else the most visible one.
- Definition shape: `{ id, title, setup, build(input) -> {steps, takeaway}, input, twist?: {label, input}, panel?: (el) => ({render(index, step, input)}) }`. Step: `{ title, say, blocks: Block[], predict?: {ask, choices, answer, why, hint} }`. Block: `vector {label, values, dp, op, hl, tone, cols, fresh}`, `matrix {rows, rowLabels, colLabels, hl, dp}`, `lines {lines: (string | {t, hl})[]}`, `bars {labels, values, hl, ghost}`, `note {html, tone}`.
- A `predict` on step k is asked before step k is revealed (while step k−1 is on screen).

- [ ] **Step 1: Write failing tests** for `dwellMs` (empty text gives 4000; 50 words gives 5500; 50 words at 1.5x gives 3666.67; tags are not counted), `assemble` (length = built + 2, first kind "setup", last kind "takeaway", takeaway step keeps the last built step's blocks), and `pickActive` (last interacted wins when visible; falls back to highest visibility ratio; returns null when none visible).
- [ ] **Step 2: Run** `node --test tests/`. Expected: FAIL, module missing.
- [ ] **Step 3: Implement the engine.** Rendering builds the DOM contract below, timing uses one `requestAnimationFrame` loop per playing player that tracks elapsed time so a pause resumes mid-step, visibility uses one IntersectionObserver (thresholds 0, 0.25, 0.6, 1) that autoplays at 0.6 unless the learner paused or already finished, and pauses under 0.25. Keyboard: a single document listener sends Left and Right to `pickActive(...)` unless focus is in a form field or a dialog is open; Space works only when focus is on the player root. Reduced motion hides the play button, never autoplays, and skips count-ups. Reaching the takeaway calls `store.markWalk`. Speed cycles 0.75, 1, 1.5 and is saved with `setPref("speed")`. `setTwist` cancels the running timer before rebuilding.

  DOM contract (classes the CSS and QA rely on): `.wt` root (`tabindex="0"`, `role="region"`), `.wt-head` with `.wt-kicker`, `.wt-title`, optional `.wt-twist` button; `.wt-body` holding optional `.wt-panel` and `.wt-stage`; `.wt-step-label`; `.wt-blocks`; `.wt-say` (`aria-live="polite"`); `.wt-predict`; `.wt-controls` with `.wt-back`, `.wt-play`, `.wt-next` (ring via `--wt-progress`), `.wt-dots` (buttons), `.wt-speed`, `.wt-restart`.
- [ ] **Step 4: Write `css/walkthrough.css`** using tokens from `css/tokens.css` (Task 4): a two-column body at 900px and up, tabular mono numbers, bracketed vector rows, blue highlight for inputs, teal for results, a conic-gradient ring on Next, predict choices as large tap targets (at least 44px), and a reduced-motion block.
- [ ] **Step 5: Run** `node --test tests/`. Expected: all PASS.

### Task 4: Shell (tokens, top bar, drawer, theme, resume)

**Files:**
- Create: `css/tokens.css` (colour, type and spacing tokens for both themes, moved from `styles.css` and extended)
- Create: `css/shell.css` (top bar, L1/L2 switch, part button, progress line, drawer, resume toast, part anatomy: opener, takeaway card, bridge)
- Create: `js/core/shell.js`

**Interfaces:**
- Consumes: `Lecture.store`.
- Page contract: `<body data-lecture="l2">`; each part is `<section class="part" id="pN" data-title="...">` with an optional `data-sub` list of `id:title` pairs for drawer sub-entries; each part ends with `.part-bridge`. The top bar contains `#part-button`, `#progress-fill`, `#theme-toggle`, and the drawer is `<dialog id="contents">`.
- Produces: `Lecture.shell.currentPart() -> {id, index, title}`.

- [ ] **Step 1: Implement `shell.js`**: theme toggle using key `transformer-theme` (a tiny inline script in each page's `<head>` applies it before first paint); the drawer list is built from `.part` sections with ticks from the store; current part is the last part whose top is above 30% of the viewport, computed in a `requestAnimationFrame`-throttled scroll handler that also updates the progress line and `store.setLast`; bridges are observed and mark their part done; the resume toast appears on load when `getLast` is set, is not the first part, and the page is scrolled to the top.
- [ ] **Step 2: Write `tokens.css` and `shell.css`.** Use the impeccable skill to critique the shell before moving on.

### Task 5: `lecture-2.html` skeleton with the ALiBi walkthrough

**Files:**
- Create: `lecture-2.html`
- Create: `js/lecture-2/part-2.js` (ALiBi definition only in this phase)
- Create: `tests/lecture-2.test.mjs`

**Interfaces:**
- Consumes: everything above. Registers `Lecture.defs["l2-alibi"]`.

- [ ] **Step 1: Write the failing test**: build ALiBi with the default input and check the bias row is `[-1.5,-1,-0.5,0]`, the adjusted row is `[0.5,0,2.5,2]`, the weights are `[0.074,0.045,0.548,0.333]` and the comparison weights are `[0.197,0.072,0.534,0.197]`; build with the twist (μ = 0.1) and check the bias row is `[-0.3,-0.2,-0.1,0]`. The softmax step carries a `predict` whose answer is key 2.
- [ ] **Step 2: Run** `node --test tests/`. Expected: FAIL.
- [ ] **Step 3: Implement** `part-2.js` and `lecture-2.html`: hero, all 11 part openers with their questions and bridges taken from the notes, and the ALiBi walkthrough inside Part 2.
- [ ] **Step 4: Run** `node --test tests/`. Expected: all PASS.
- [ ] **Step 5: Browser QA** with Playwright at 375px and 1280px, dark and light: the drawer opens and closes with Esc, ticks appear after a bridge scrolls into view, the walkthrough autoplays at 60% visibility, pauses when scrolled away and resumes mid-step, the twist rebuilds with μ = 0.1 numbers, the predict stop blocks autoplay until answered, and there are no console errors.

### Task 6: Document, commit, push

- [ ] Update `README.md` (new file layout, how to run tests) and add a Phase 1 entry to `docs/CHANGELOG.md`.
- [ ] Run `node --test tests/` one final time. Expected: all PASS.
- [ ] Commit `feat: add walkthrough engine, progress store and new lecture shell` and push to `origin main`.
