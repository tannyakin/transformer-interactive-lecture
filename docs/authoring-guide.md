# Authoring guide: adding lecture content

This is how every part of a lecture page is built. Read it before touching a lecture file. The design reasons live in `docs/specs/2026-09-23-lecture-2-design.md` and `PRODUCT.md`.

## Writing rules (tested)

- **No em dashes** anywhere (U+2014). `npm test` fails if one appears. Use a comma, colon, full stop or brackets.
- Plain, warm, human English. Talk to the reader as "you". Explain why before what. No marketing words.
- Every number must match the lecture notes exactly. Compute numbers in code, never type them into narration by hand.
- Wrap the first use of an abbreviation in each part in `<abbr data-term="GQA">GQA</abbr>`. The glossary script adds the tooltip.

## Page anatomy

Each part is a `<section class="part" id="pN" data-title="...">` with an opener, a body and a bridge. The opener and bridge are in the page already; content goes in the body:

```html
<div class="subpart" id="p3-1">
  <h3>Add: the residual connection</h3>
  <div class="prose">
    <p>...</p>
  </div>
  <div class="formula">output = x + F(x)</div>
  <div data-walk="l2-residual"></div>          <!-- a worked example -->
  <figure class="ix" id="l2p3-highway">...</figure>  <!-- an interactive -->
</div>
```

Shared components (styled in `css/shell.css`, do not restyle them):

| Class | Use |
|---|---|
| `.prose` | reading text, capped at a comfortable line length |
| `.formula` | a displayed formula; scrolls sideways on phones; `<span class="hl">` highlights part of it |
| `.aside` + `.aside-label` | a short tip or "from the lecture video" note |
| `.table-wrap > table` | every table; `td.num` right-aligns numbers |
| `.ix`, `.ix-title`, `.ix-hint`, `.ix-controls`, `.ix-field`, `.ix-stage`, `.ix-readout` | an interactive figure with its own controls |
| `.chips` + `button.chip[aria-pressed]` | a small set of toggle choices |

## Worked examples (walkthroughs)

Register a definition on `Lecture.defs` and put `<div data-walk="<id>"></div>` in the page. The engine mounts it, autoplays it at reading pace, and handles controls, keyboard, progress and reduced motion.

```js
defs["l2-example"] = {
  id: "l2-example",
  title: "Layer norm, by hand",
  input: { x: [2, 4, 6, 8] },
  twist: { label: "Multiply the input by 10", offLabel: "Back to the original", input: { x: [20, 40, 60, 80] } }, // optional
  setup: "What the learner is about to compute, in one or two sentences.",
  setupBlocks: (input) => [ /* optional blocks shown on the setup card */ ],
  panel: (el) => ({ render(index, step, input) { /* optional custom visual beside the numbers */ } }),
  build(input) {
    // compute everything from input with Lecture.math
    return {
      steps: [
        { title: "Find the mean", say: "Narration for this step.", blocks: [ /* ... */ ] },
        { title: "...", say: "...", blocks: [], predict: { ask, choices, answer, why, hint } },
      ],
      takeaway: "What did the math just say? One short paragraph.",
    };
  },
};
```

- 4 to 7 steps. Each step shows one idea. Repeat earlier rows so the reader keeps context.
- One or two `predict` stops per example, at the moment worth guessing. A predict on step k is asked before step k is revealed. `answer` is the index of the right choice; compute it.
- Give each block that holds a checked number a `key` so tests can find it.
- Panel `render` gets `index` 0 for the setup card and the last index for the takeaway.

Block types:

| type | fields |
|---|---|
| `vector` | `label`, `values`, `dp`, `op` ("+", "=", ...), `hl` (indices), `tone` ("result", "muted"), `cols` (captions), `fresh` (animate in), `key` |
| `matrix` | `rows`, `rowLabels`, `colLabels`, `hl` ([[r, c]]), `muted` ([[r, c]]), `dp`, `key` |
| `lines` | `lines`: strings or `{ t, hl }`; HTML allowed (for example `e<sup>1.5</sup>`) |
| `bars` | `labels`, `values` (0 to 1), `hl`, `ghost` (comparison values), `label`, `ghostLabel`, `dp`, `key` |
| `note` | `html`, `tone` ("warn") |

Numbers render with a true minus sign and the fewest decimals that show every value exactly (override with `dp`). Strings pass through, so `"\u2212\u221e"` or `"\u25a0"` work in a matrix.

## Files and ownership

| What | Where |
|---|---|
| Definitions and interactives for a part | `js/lecture-2/part-N.js` (IIFE, see `part-2.js`) |
| Styles for a part | `css/l2/part-N.css`, class names prefixed `l2pN-` |
| Tests for a part | `tests/l2-part-N.test.mjs` |
| Shared maths | `js/core/math.js` (add helpers only with tests) |

Interactive code must run only in the browser: guard with `if (typeof document === "undefined") return;` after registering definitions, and set up on `DOMContentLoaded`. Canvas drawings listen for the `lecture:theme` event to redraw in the new colours. Read colours from CSS variables, never hard-code them.

## Quality bar

- Works at 375px wide with no sideways page scroll (wide content scrolls inside its own box).
- Both themes. Colour is never the only signal.
- `prefers-reduced-motion: reduce` gets no animation.
- Keyboard reachable controls, real `<button>` and `<input>` elements, labels on everything.
- `npm test` passes.
