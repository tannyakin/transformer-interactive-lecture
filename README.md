# AI UniPod LLM lectures: interactive notebooks

Interactive, self-study pages for the AI UniPod LLM lecture series (University of Lagos).
Every worked example plays out step by step at reading pace, can be stepped by hand, and
stops to ask you to predict at the moments that matter. Your place and progress are
remembered in your browser.

No build step and no dependencies. Plain HTML, CSS and JavaScript that works on GitHub Pages
and when opened straight from disk.

## The lectures

- **Lecture 1, The Transformer** (`index.html`): why attention beats recurrence, query, key and
  value, the encoder-decoder stack with an animated token flow, scaled dot-product attention
  worked by hand (with a causal mask twist), multi-head attention, the positional encoding
  explorer, key equations, a filterable glossary and the base vs big table. Based on
  ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).
- **Lecture 2, From attention to modern LLMs** (`lecture-2.html`): positional encoding in depth
  (sinusoidal, T5 bias, ALiBi, RoPE), Add & Norm, the feedforward network, the cost of attention,
  sparse attention, MHA to GQA, model families, BERT and what came after. Based on Stanford
  CME295, Lecture 2.

Both pages share one shell: a slim top bar with a Lecture 1 / Lecture 2 switch, a contents
drawer with ticks for finished parts, reading progress, a light and dark theme, and a
"pick up where you left off" prompt.

## File layout

```
index.html                 Lecture 1
lecture-2.html             Lecture 2
css/tokens.css             colours, type, spacing, both themes
css/shell.css              top bar, drawer, part anatomy, shared components
css/walkthrough.css        the worked example player
css/lecture-1.css          Lecture 1 diagrams and figures
css/l2/                    Lecture 2 styles per part
js/core/math.js            pure maths helpers (tested in Node)
js/core/store.js           progress saved in localStorage
js/core/walkthrough.js     the worked example engine
js/core/shell.js           theme, top bar, drawer, part tracking, resume
js/lecture-1.js            Lecture 1 walkthrough (l1-sdpa) and interactive figures
js/lecture-2/part-*.js     Lecture 2 walkthroughs and interactives per part
tests/*.test.mjs           numeric and writing-rule checks
docs/authoring-guide.md    how to add a part, a walkthrough or an interactive
```

Scripts are classic `defer` scripts sharing one `window.Lecture` namespace (not ES modules),
so the pages still work from `file://`.

## Running it locally

Open `index.html` directly in a browser, or serve the folder so fonts and paths behave exactly
as on the web:

```bash
python -m http.server 8000
# then visit http://localhost:8000
```

## Tests

You need Node 18 or newer. No packages to install.

```bash
npm test
```

This runs `node --test` over `tests/*.test.mjs`: every worked example's numbers are checked
against an independent computation or the lecture notes, and a content check fails if any
site file contains an em dash.

## Hosting on GitHub Pages

1. Push this folder to a GitHub repository.
2. In the repo, go to **Settings, then Pages**.
3. Under **Build and deployment**, set **Source** to "Deploy from a branch".
4. Choose the branch (for example `main`) and the root folder.
5. Save. GitHub publishes the site at `https://<username>.github.io/<repo>/` within a minute or two.

## Credits

Lecture 1 is redrawn and reinterpreted from Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J.,
Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). "Attention Is All You Need."
*NeurIPS 2017.* Lecture 2 follows Stanford CME295, Lecture 2, "Transformer-Based Models & Tricks"
by Afshine and Shervine Amidi.
