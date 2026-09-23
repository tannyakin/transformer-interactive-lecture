# Changelog

A running record of what changed in the lecture site and why. Newest first.

## 2026-09-23: Lecture 2 design spec

- Added `docs/specs/2026-09-23-lecture-2-design.md`, the agreed design for adding Lecture 2 and redesigning the site for self-study.
- Key decisions: one page per lecture, a slim top bar with a contents drawer in place of the text pills, an autoplaying walkthrough player with predict stops, and a hybrid build (one generic walkthrough engine plus custom visual panels).
- Every worked example number in the Lecture 2 notes was recomputed and matches. The only difference is a rounding note on the KL loss (0.0833 from rounded terms, 0.0834 unrounded).

## 2026-09-23: Phase 1, the shared foundation

- `js/core/math.js`: tested maths helpers. Every worked-example number in the Lecture 2 notes is checked by `tests/math.test.mjs`.
- `js/core/store.js`: progress store (finished parts and walkthroughs, last position, speed). Survives blocked or corrupted storage.
- `js/core/walkthrough.js` + `css/walkthrough.css`: the walkthrough engine. Autoplays at reading pace, pauses when scrolled away, asks predict questions, supports "change one number" twists, keyboard and reduced motion.
- `js/core/shell.js` + `css/shell.css` + `css/tokens.css`: slim top bar with Lecture 1/2 switch, current part button, contents drawer with ticks, reading progress, welcome-back toast. Body font moved from Newsreader to Literata; faint text colours raised to pass WCAG AA.
- `lecture-2.html`: page skeleton with all 11 parts, their questions and bridges.
- `PRODUCT.md`: design context (audience, personality, anti-references, principles).
- Fixed: walkthroughs mounted before later scripts registered their definitions, because deferred scripts run while `readyState` is already "interactive".
- Run the checks with `npm test` (Node 20 or newer, no dependencies).
