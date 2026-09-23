# Lecture 2 and site redesign: design spec

Date: 2026-09-23
Status: awaiting review
Source notes: `../Lecture2_Notes_Positional_Encoding_to_RoBERTa.md` (in the parent LLM LECTURES folder)

## 1. Goal

Add Lecture 2 ("From Attention Is All You Need to modern LLMs") to the interactive lecture site, and redesign the whole site around self-study.

Success looks like this: a student working alone can open Lecture 2, follow the same order of ideas as the notes, watch every worked example play out step by step (or step through it themselves), get asked to predict at the key moments, and come back tomorrow to find their place and progress remembered.

### Agreed decisions

| Decision | Choice |
|---|---|
| Audience | self-study students (not live projection) |
| Structure | one page per lecture: `index.html` (Lecture 1), `lecture-2.html` |
| Navigation | slim top bar + contents drawer, no row of text pills |
| Walkthroughs | autoplay with reading-time pacing, manual controls, predict stops |
| Architecture | approach C: one generic walkthrough engine + custom visual panels where a picture teaches more |
| Lecture 1 | full redesign to match, all existing demos kept |
| Writing | no em dashes, plain human tone, numbers exactly as in the notes |
| Delivery | still no build step, still GitHub Pages, still opens from `file://` |

## 2. Page anatomy

Every part of a lecture follows the same shape:

1. **Opener**: part number, title, and the single question the part answers.
2. **Concept blocks**: short prose, each paired with something to interact with.
3. **Worked example walkthroughs** (section 4).
4. **"What did the math just say?"** takeaway card after each walkthrough.
5. **Bridge**: the notes' closing sentence for the part, plus a "Continue to Part N" button.

### Lecture 2 map (● walkthrough, ◆ interactive)

| Part | Content |
|---|---|
| Hero + 1. Order disappears | ◆ swap "dog bites man" / "man bites dog", attention scores stay identical |
| 2. Positional encoding | 2.1 learned: ◆ ask for position 600, hit the 512 wall. 2.2 sinusoidal: ◆ clock hands, ● PE(0..3) for d = 4, ● dot products building the similarity band. 2.3 relative idea. 2.4 ● T5 bias. 2.5 ● ALiBi with predict stop, ◆ slope slider (μ = 0.1 twist). 2.6 RoPE: ◆ drag m and n to rotate arrows, ● cases A, B, C, ◆ long-term decay curve. 2.7 summary timeline |
| 3. Add & Norm | ● residual, ◆ gradient highway (0.5ⁿ vs 1 + F′ over a layers slider), ● LayerNorm with ×10 twist, ● RMSNorm, ◆ post-norm vs pre-norm diagram flip |
| 4. Feedforward | ● FFN by hand with [1, −2] twist, ◆ attention vs FFN parameter bar, ◆ SwiGLU knob vs ReLU switch, ◆ MoE router |
| 5. Cost of attention | ◆ n² grid slider, ● 20 GB memory calculation, ◆ 6d crossover chart |
| 6. Sparse attention | ◆ window grid builder counting pairs live (22 vs 64), ◆ Longformer pattern, ◆ receptive field across layers |
| 7. MHA, MQA, GQA | ● KV cache for Llama 2 7B, ◆ G slider morphing MHA to GQA to MQA with live cache size |
| 8. Architecture families | ● causal masking, ◆ three family cards each showing its mask |
| 9. BERT | ● where 110M parameters live, ◆ input builder (token + segment + position), ● MLM 15/80/10/10 with re-roll, ● 10-step sentiment pipeline |
| 10. After BERT | ● RoBERTa 15.6× sequences, ● C(20,3) = 1,140, ● KL distillation loss with student sliding toward teacher, family table |
| 11. The bridge | ◆ 2017 vs today toggle, ● one token through Llama 3 with links back to each part |
| End | filterable glossary, inline abbreviation tooltips, further reading |

Part 2 is large, so its sub-sections appear as indented entries in the contents drawer.

## 3. Navigation and shell

**Top bar** (sticky, one line on every screen size):

```
[logo] [L1|L2]        3/11 · Add & Norm ▾        [theme]
▬▬▬▬▬▬▬▬▬▬▬▬░░░░░░░░░░░░░░░░░░░░░░░░░░  <- reading progress
```

- The L1/L2 segmented control is two real links.
- The current-part button shows `n/total · name` on desktop and `n/total` on phones. It opens the contents drawer.
- Reading progress is the scroll position through the page.

**Contents drawer**: a native `<dialog>` (gives Esc, focus trap and backdrop for free). It lists every part with a tick when finished and a marker on the current one, plus a "Pick up where you left off" row when there is saved progress.

**Progress store** (`js/core/store.js`): one localStorage key, every read and write wrapped in try/catch so a private window or blocked storage just means no memory, never a broken page.

```json
{ "v": 1, "l2": { "parts": { "p3": true }, "walk": { "l2-alibi": true }, "last": "p3" } }
```

A part counts as finished when its bridge scrolls into view. A walkthrough counts as finished when its last step is reached.

**Inline glossary**: abbreviations in the prose are wrapped in `<abbr data-term>`. Hover or tap shows the one-line definition from the glossary table, so there is one source for every definition.

## 4. The walkthrough player

### What the learner sees

```
┌ Worked example · ALiBi ─────────────────────────┐
│ Step 2 of 5 · Add the bias to each score        │
│                                                 │
│   raw     [ 2.0   1.0   3.0   2.0 ]             │
│   bias  + [-1.5  -1.0  -0.5   0.0 ]  (glows)    │
│   new     [ 0.5   0.0   2.5   2.0 ]  (counts up)│
│                                                 │
│ The farthest key lost the most. That is the     │
│ penalty growing with distance.                  │
│                                                 │
│ ◀ Back   ❚❚   Next ◔   ●●○○○   1x   ↺            │
└─────────────────────────────────────────────────┘
```

Optional visual panel sits beside the numbers on wide screens and above them on phones.

### Behaviour

- **Autoplay** starts when the player is at least 60% in view, unless the learner already finished it. It pauses when the player leaves view.
- **Pacing**: each step stays up for `max(4s, 2.5s + 60ms × words in the narration)`, divided by speed (0.75x, 1x, 1.5x). A ring on the Next button shows time left.
- **Manual control**: Back, Next and the step dots all jump immediately and pause autoplay. Play resumes it. Restart goes back to the setup card.
- **Keyboard**: Left and Right step, Space plays or pauses, but only for the active player (the one last interacted with or most in view), so two players never fight over the keys.
- **Predict stops**: at one or two key steps, autoplay halts and asks a question with 2 to 4 choices. A right answer shows why, then autoplay continues after a short beat. A wrong answer shows a hint and lets them try again, with "show me" as a way out. The result is never graded or stored as a score.
- **Twists**: examples that have a "change one number" twist in the notes get a toggle that reruns the walkthrough with the new input. Values are computed, not typed, so the twist is always correct.
- **Reduced motion**: no autoplay and no count-up animation. The player becomes a plain stepper.
- **Screen readers**: narration is an `aria-live="polite"` region, controls are real buttons with labels.

### Engine API

```js
Walkthrough.mount(el, {
  id: "l2-alibi",                  // progress key
  title: "ALiBi, by hand",
  setup: "A query at position 3 looks at keys 0 to 3...",
  panel: optionalPanelFactory,     // (el) => { render(stepIndex, data) }
  build: (input) => ({ steps, takeaway }),   // computed from input
  input: { mu: 0.5, raw: [2, 1, 3, 2] },
  twist: { label: "Try μ = 0.1", input: { mu: 0.1 } }
});
```

A step is `{ title, blocks, say, predict? }`. Block types:

| Block | Renders |
|---|---|
| `vector` | a labelled row of cells, optional operator, highlight indices, "result" tone |
| `matrix` | a grid with highlighted cells (masks, attention grids) |
| `lines` | a list of arithmetic lines, e.g. `e^1.5 = 4.482` |
| `bars` | a probability or weight bar chart |
| `note` | a short callout |

`predict` is `{ ask, choices, answer, why, hint }`.

### Visual panels (approach C)

Panels implement `render(stepIndex, data)` and nothing else. Planned panels: clock hands (sinusoidal), rotating arrows (RoPE), growing n² grid, sliding window grid, head grouping (MHA/GQA/MQA), FFN feature switches, residual stream, student vs teacher bars (KL).

## 5. Visual direction

Keep the current identity and refine it, rather than replace it:

- Dark notebook default with a light theme. Newsreader serif for reading, JetBrains Mono with tabular numbers for all maths.
- Existing role colours keep their meaning: teal = encoder, orange = decoder, red = attention, blue = accent. Inside walkthroughs: blue marks the values being used in this step, teal marks the newly computed result. Negative numbers always carry a minus sign, never colour alone.
- Reading column capped around 68 characters. Walkthrough stage can go wider.
- Chapter openers get a large numeral and the part's question, so parts are easy to tell apart while scrolling.
- The impeccable and design skills will be used during the build to critique and polish each phase.

## 6. File structure

```
index.html                 Lecture 1
lecture-2.html             Lecture 2
css/tokens.css             colours, type, spacing, both themes
css/shell.css              top bar, drawer, layout, part anatomy
css/walkthrough.css        player, blocks, predict stops
css/lecture-1.css          Lecture 1 diagrams (moved from styles.css)
css/lecture-2.css          Lecture 2 panels and interactives
js/core/math.js            pure helpers: softmax, pe, rotate, layerNorm, rmsNorm, ffn, kl, choose
js/core/store.js           progress persistence
js/core/shell.js           theme, top bar, drawer, part tracking, resume, glossary tooltips
js/core/walkthrough.js     the engine
js/panels/*.js             visual panels
js/lecture-1.js            Lecture 1 demos (from script.js, step player moved onto the engine)
js/lecture-2/part-*.js     walkthrough definitions and interactives per part
tests/math.test.mjs        numeric checks against the notes
tests/content.test.mjs     writing rule checks
```

Scripts are classic `defer` scripts sharing one `window.Lecture` namespace, not ES modules, so the site still works when opened straight from disk. `math.js` also exports through `module.exports` when present so Node can test it.

## 7. Correctness and testing

- **Numbers**: every worked example value in the notes was recomputed on 2026-09-23 and all match. One rounding note: the KL "early" loss is 0.0833 when summing the rounded terms and 0.0834 unrounded; the walkthrough shows the rounded terms and says so.
- `tests/math.test.mjs` (`node --test`, no dependencies) asserts every printed value in the notes against `math.js`.
- `tests/content.test.mjs` fails the build if any HTML or JS file contains an em dash.
- Browser QA per phase with Playwright: 375px and 1280px widths, both themes, reduced motion, keyboard only, no console errors.

## 8. Build order

Each phase ends with: docs updated (README and `docs/CHANGELOG.md`), a one-line conventional commit, and a push.

1. Foundation: CSS split, `math.js` + tests, store, shell (top bar, drawer, theme), walkthrough engine with one sample walkthrough.
2. Lecture 1 moved onto the new shell; its step player moved onto the engine.
3. `lecture-2.html` skeleton + Parts 1 and 2.
4. Parts 3 and 4.
5. Parts 5, 6 and 7.
6. Parts 8 and 9.
7. Parts 10 and 11, glossary, further reading.
8. Full QA pass and polish.

## 9. Out of scope

Accounts or server-side progress, quizzes with scores, a course home page, and editing the lecture notes themselves.
