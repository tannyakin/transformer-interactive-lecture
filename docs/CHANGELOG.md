# Changelog

A running record of what changed in the lecture site and why. Newest first.

## 2026-09-24: Best-on-a-bigger-screen pop-up

- New one-time dialog on phone-sized screens (under 700px wide) reading "Best on a laptop or tablet", explaining that the wide diagrams and step-by-step figures need room and that phones still work but some figures will be tight. Built in `js/core/shell.js` so both lectures get it, styled in `css/shell.css` with existing tokens, and it uses a native `<dialog>` (focus trapped, Escape closes it).
- Dismissal is remembered in the progress store (`deviceTipSeen`), so returning visitors are not nagged. Tablets, laptops and desktops never see it.
- The show/hide rule is a pure function, `needsDeviceTip`, with two new tests (137 total, all passing). Checked live: shows on first phone visit, saves the dismissal, stays gone after reload, never appears at 1280px.

## 2026-09-24: Label the decoder's cross-attention box by name

- The decoder's second sublayer box in Lecture 1, Part 3's architecture diagram was titled generically "Multi-Head Attention," same as the encoder's self-attention box, with only a small "Q: decoder · K, V: encoder" subtext hinting it was actually cross-attention. The masked self-attention box right above it already spells out its type ("Masked Multi-Head Attention"), so this one was the odd one out. Retitled it "Multi-Head Cross-Attention" to match, so the diagram is unambiguous on its own without reading the subtext.
- Followed up by also retitling the encoder's self-attention box, from generic "Multi-Head Attention" to "Multi-Head Self-Attention," so all three attention boxes in the diagram now name their own type at a glance. Checked Lecture 2 for the same failure mode (a diagram with several similar-looking boxes where only some spell out what they are) and found none; Lecture 2 doesn't redraw this architecture, it links back to it.

## 2026-09-24: Explain why each piece of the encoder-decoder stack exists

- Lecture 1, Part 3's architecture diagram already drew and labelled every piece (input/output embedding, positional encoding, encoder self-attention, Add & Norm, feed forward, masked self-attention, cross-attention, linear, softmax), but never said why each one is there. Added a lead-in paragraph on why the architecture splits into two stacks at all, plus an 8-item ordered explanation list (`.l1-why`, new CSS mirroring Lecture 2's `.l2p11-four` pattern) walking through the reasoning for every element in diagram order.
- Verified in-browser at desktop width and confirmed all 135 tests still pass (including the em-dash scanner over the new prose).

## 2026-09-23: Visual design pass, code review, and build-scratch cleanup

- Visual design pass: checked the shipped pages against the design system's own rules (checked-contrast token comment, no eyebrow/gradient-text/side-stripe-border patterns, purposeful rather than decorative use of backdrop-blur) and confirmed every text/background pairing in `css/tokens.css` clears WCAG AA (4.87:1 to 16.74:1). Spot-checked the hero, a worked-example walkthrough card, a comparison table, and Lecture 1's live attention demo in both themes and at 375px and 1440px widths. No changes needed; the build already met the bar.
- Code review pass: verified all 18 JS modules share the same IIFE/export wrapper, every `<script>`/`<link>` reference on both pages resolves to a file that exists (and vice versa, no orphans), no duplicate `id` attributes on either page (203 on Lecture 2, 84 on Lecture 1), no debug leftovers (`console.log`, `TODO`, `FIXME`), and dynamic content is written with `textContent` rather than interpolated into `innerHTML`. No findings.
- Removed the gitignored `.fragments/` and `.superpowers/` build-scratch directories now that both lectures are complete and `lecture-2.html` is the stable, committed artifact. This broke 9 tests that read the fragment files directly instead of the assembled page; repointed them at `lecture-2.html` (slicing to the relevant `<section>` for the two tests that whitelist every glossary term in scope) and dropped two now-pointless `existsSync` skip-guards so their assertions actually run. All 135 tests pass.

## 2026-09-23: Cross-link precision pass

- Audited every internal `#pN` link in `lecture-2.html` against the final subpart ids now that all 11 parts are built. Fixed cases where the visible text already named a subsection ("Part 3.5", "Part 4.6", "Part 10.1 lesson") but the link only pointed at the whole part, plus upgraded several "taught in" table references (RMSNorm, SwiGLU, GQA, FlashAttention, sliding-window attention, decoder-only shape, the KV cache callback) to their exact subparts.
- Verified with a script that all 64 internal part cross-links resolve to a real `id` on the page; none dangle.
- Applied the same fixes to the underlying `.fragments/l2-p5.html`, `l2-p10.html` and `l2-p11.html` sources and reassembled, so the fix survives a future rebuild.

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

## 2026-09-23: Lecture 2 complete, all 11 parts

- All 11 parts of Lecture 2 are built and spliced into `lecture-2.html`: Why order disappears, Positional encoding (learned, sinusoidal, T5, ALiBi, RoPE), Add & Norm, the feedforward network, the cost of attention, sparse attention, MHA/MQA/GQA, the three architecture families, BERT, After BERT (RoBERTa, DistilBERT, ALBERT, ELECTRA, DeBERTa, ModernBERT), and the bridge to modern LLMs.
- 21 worked-example walkthroughs across the lecture, each autoplaying at reading pace with predict stops and, where the notes invite it, a "change one number" twist. Every number is computed in code and checked against the notes by `tests/*.test.mjs`.
- 42 hand-built interactive figures (inline SVG/HTML, theme-aware via CSS variables): the 512-token wall, sinusoidal clock hands, PE heatmaps, T5 bucket staircase, ALiBi slope explorer, RoPE rotating arrows, the gradient highway, FFN feature switches, the n² growth grid, sliding-window pair counter, KV cache generation, the G slider from MHA to GQA to MQA, the three-family mask switcher, BERT's input builder and MLM sampler, the RoBERTa masking counter, the KL distillation slider, and the "one token through Llama 3" walk.
- Glossary: 52 terms from the notes' Part 12 table, with hover/focus/tap tooltips on every `<abbr data-term>` in the text, plus a filterable glossary list and further-reading section.
- Lecture 1 (`index.html`) moved onto the same shell: slim top bar, contents drawer, resume toast, and its scaled dot-product attention step player rebuilt as a walkthrough (`l2-sdpa`... `l1-sdpa`) with a predict stop and a causal-mask twist. `styles.css` and `script.js` retired.
- Fixed during QA: two phone-width overflow bugs (Part 9's WordPiece line, Lecture 1's equations grid), a page-level sideways-scroll bug caused by unpositioned `.sr-only` labels inside scroll containers, duplicate SVG marker IDs when the pre/post-norm diagram appears twice on Part 3, and inconsistent British/American spelling of "normalization".
- Full suite: `node --test "tests/*.test.mjs"` passes 135 of 135, including a site-wide no-em-dash scan over every `.html`, `.js`, `.mjs`, `.css` and `.md` file.
- Verified in a real browser: every walkthrough stepped through all steps and twists, every interactive control exercised, both themes, reduced motion, 375px and 1280px widths, no console errors, no broken in-page links, no duplicate element ids.
