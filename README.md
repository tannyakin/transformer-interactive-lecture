# The Transformer: an interactive lecture notebook

A single page, animated walkthrough of the Transformer architecture from
["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017),
built as a source of truth reference for AI UniPod students.

No build step and no dependencies, just `index.html`, `styles.css`, and `script.js`.

## What is inside

- A live, auto-cycling attention visualization in the hero
- A side-by-side animated comparison of recurrent processing versus attention
- An interactive query, key, and value demo. Click any word to see it re-weight the sentence
- The full encoder-decoder diagram with a "run the flow" button that animates one token's path through the network
- A step-by-step player for scaled dot-product and multi-head attention
- A draggable positional encoding wave visualizer
- A filterable glossary and the base-vs-big hyperparameter table
- A light and dark theme toggle, keyboard-accessible controls, and reduced-motion support

## Running it locally

Open `index.html` directly in a browser, or serve the folder so relative paths and fonts behave normally:

```bash
python -m http.server 8000
# then visit http://localhost:8000
```

## Hosting on GitHub Pages

1. Push this folder to a GitHub repository.
2. In the repo, go to **Settings, then Pages**.
3. Under **Build and deployment**, set **Source** to "Deploy from a branch".
4. Choose the branch (for example `main`) and the root folder.
5. Save. GitHub will publish the site at `https://<username>.github.io/<repo>/` within a minute or two.

## Credits

Redrawn and reinterpreted from Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). "Attention Is All You Need." *NeurIPS 2017.*
