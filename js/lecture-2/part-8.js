// ============================================================
// Lecture 2, Part 8: the three architecture families.
// One walkthrough (causal masking for query 1) and two
// interactives: the 4 x 4 causal mask you can poke row by row,
// and one figure that switches between encoder-only,
// decoder-only and encoder-decoder with each one's masks.
// ============================================================
(function (root) {
  "use strict";

  const L = (root.Lecture = root.Lecture || {});
  const M = L.math;
  const defs = (L.defs = L.defs || {});
  const f = (x, dp) => M.fmt(x, dp === undefined ? 3 : dp);
  const keyCols = (n) => Array.from({ length: n }, (_, i) => "key " + i);
  const argmax = (xs) => xs.reduce((best, x, i) => (x > xs[best] ? i : best), 0);
  // Integers and ±∞ print bare, like the notes; everything else to 3 places.
  const show = (x) => (!isFinite(x) || Number.isInteger(x) ? f(x, 0) : f(x));

  // true = the query in this row may attend to the key in this column.
  function maskMatrix(kind, rows, cols) {
    const c = cols === undefined ? rows : cols;
    return Array.from({ length: rows }, (_, q) => Array.from({ length: c }, (_, k) => (kind === "causal" ? k <= q : true)));
  }
  // The additive form used before softmax: 0 means allowed, −∞ means blocked.
  const causalRow = (q, n) => Array.from({ length: n }, (_, k) => (k <= q ? 0 : -Infinity));

  /* ---------------- 8.1 causal masking worked example ---------------- */
  defs["l2-causal-mask"] = {
    id: "l2-causal-mask",
    title: "Causal masking, by hand",
    input: { raw: [1, 3, 2, 0], query: 1 },
    setup:
      "Query 1 has already been scored against all four keys, so its raw scores are ready. " +
      "Keys 2 and 3 come after it in the sentence. The mask must make sure it cannot use them, however useful they look.",
    setupBlocks: (input) => [{ type: "vector", key: "raw", label: "raw score", values: input.raw, cols: keyCols(input.raw.length) }],
    build(input) {
      const { raw, query } = input;
      const n = raw.length;
      const keys = raw.map((_, k) => k);
      const cols = keyCols(n);
      const full = keys.map((q) => causalRow(q, n));
      const mask = causalRow(query, n);
      const masked = M.add(raw, mask);
      const soft = M.softmax(masked);
      const noMask = M.softmax(raw);
      const future = keys.filter((k) => k > query);
      const past = keys.filter((k) => k <= query);
      const win = argmax(soft.weights);
      const bestFuture = future.reduce((b, k) => (raw[k] > raw[b] ? k : b), future[0]);

      const rawRow = { type: "vector", label: "raw score", values: raw, cols };
      const maskRow = { type: "vector", key: "maskRow", label: "mask row " + query, values: mask };
      const maskedRow = { type: "vector", key: "masked", label: "masked", values: masked, tone: "result" };
      const sumLine = { t: `sum = ${f(soft.sum)}`, hl: true };

      return {
        steps: [
          {
            title: `Read row ${query} of the mask`,
            say:
              `The causal mask has one row per query. Row ${query} holds a 0 for key ${past.join(" and key ")}, which changes nothing, ` +
              `and −∞ for key ${future.join(" and key ")}, the ones that come later in the sentence.`,
            blocks: [
              {
                type: "matrix",
                key: "mask",
                rows: full,
                rowLabels: keys.map((q) => "query " + q),
                colLabels: cols,
                hl: keys.map((k) => [query, k]),
                muted: keys.filter((q) => q !== query).flatMap((q) => keys.map((k) => [q, k])),
                fresh: true,
              },
              Object.assign({}, maskRow, { cols, fresh: true }),
            ],
          },
          {
            title: "Add the mask to the raw scores",
            say:
              "Adding 0 leaves a score alone. Adding −∞ to any number gives −∞, however big the number was. " +
              `Key ${bestFuture}'s raw score of ${show(raw[bestFuture])} is wiped out.`,
            blocks: [
              rawRow,
              Object.assign({}, maskRow, { op: "+", hl: future }),
              Object.assign({}, maskedRow, { op: "=", fresh: true, hl: future }),
            ],
          },
          {
            title: "Raise e to each score",
            say:
              `Softmax starts with e to the power of each score. e<sup>−∞</sup> is exactly 0, so the blocked keys drop out before anything is added up. ` +
              `The total is ${f(soft.sum)}.`,
            predict: {
              ask: "Softmax now needs e to the power of −∞. What is it?",
              choices: ["−∞", "0", "1", "It breaks the calculation"],
              answer: 1,
              why:
                "e to a large negative power is a tiny positive number, and the more negative the power, the closer it gets to 0. " +
                "At −∞ it is exactly 0. That is the whole trick: a blocked score becomes a zero before the sum.",
              hint: "Think about e<sup>−10</sup>, then e<sup>−100</sup>. Which way are they heading?",
            },
            blocks: [
              maskedRow,
              {
                type: "lines",
                key: "sum",
                value: soft.sum,
                fresh: true,
                lines: masked.map((a, k) => ({ t: `e<sup>${show(a)}</sup> = ${show(soft.exps[k])}` })).concat([sumLine]),
              },
              { type: "vector", key: "exps", label: "eˣ", values: soft.exps, dp: 3, fresh: true },
            ],
          },
          {
            title: "Divide by the total to get the weights",
            say:
              `Each weight is its e<sup>x</sup> divided by ${f(soft.sum)}. Query ${query} splits its attention between ` +
              past.map((k) => (k === query ? `itself (${f(soft.weights[k])})` : `key ${k} (${f(soft.weights[k])})`)).join(" and ") +
              `. Key ${future.join(" and key ")} get exactly 0.`,
            blocks: [
              {
                type: "lines",
                lines: soft.exps.map((e, k) => ({ t: `key ${k}:  ${show(e)} / ${f(soft.sum)} = ${show(M.round(soft.weights[k], 3))}`, hl: k === win })),
              },
              { type: "bars", key: "weights", labels: cols, values: soft.weights, hl: [win], fresh: true },
            ],
          },
          {
            title: "What the mask took away",
            say:
              `Without the mask, key ${bestFuture} would have taken ${f(noMask.weights[bestFuture])} of the attention, because its raw score was high. ` +
              "The mask does not care how relevant a key looks. If it is in the future, it gets zero.",
            blocks: [
              {
                type: "bars",
                key: "compare",
                label: "with the mask",
                ghostLabel: "no mask",
                labels: cols,
                values: soft.weights,
                ghost: noMask.weights,
                hl: future,
              },
            ],
          },
        ],
        takeaway:
          `Token ${query} can only draw on ${past.filter((k) => k !== query).map((k) => "token " + k).join(" and ")} and itself. ` +
          `Tokens ${future.join(" and ")} might have been relevant (the raw score for token ${bestFuture} was ${show(raw[bestFuture])}), ` +
          "but they are in the future, so they get exactly zero.",
      };
    },
  };

  /* ---------------- the three families ---------------- */
  const EN = ["That", "is", "good", "."];
  const DE = ["Das", "ist", "gut", "."];
  const FAMILIES = [
    {
      id: "encoder",
      name: "Encoder-only",
      role: "understanding",
      flow: {
        input: "The movie was great",
        blocks: [{ cls: "is-enc", name: "Encoder", sub: "× N layers, bidirectional" }],
        output: "a rich vector for every token, then one score per class: positive",
      },
      masks: [{ kind: "bidirectional", title: "Self-attention, bidirectional", rows: ["The", "movie", "was", "great"], cols: ["The", "movie", "was", "great"] }],
      points: [
        "Uses only the encoder stack.",
        "<b>No causal mask.</b> Every token sees every other token, both left and right. This is called <b>bidirectional</b> attention.",
        "Output: a rich vector for each token, understood in full context.",
        "Cannot easily generate text, because it never learned to predict the next word from the left.",
      ],
      best:
        "Understanding tasks. Classifying text (spam or not, positive or negative review), finding names and places in text " +
        '(<abbr data-term="NER">NER</abbr>, Named Entity Recognition), finding answer spans in a passage, and producing <b>embeddings</b> ' +
        'for search and <abbr data-term="RAG">RAG</abbr> (Retrieval-Augmented Generation).',
      examples: '<abbr data-term="BERT">BERT</abbr>, and its descendants <abbr data-term="RoBERTa">RoBERTa</abbr>, <abbr data-term="DeBERTa">DeBERTa</abbr>, ModernBERT.',
    },
    {
      id: "decoder",
      name: "Decoder-only",
      role: "generation",
      flow: {
        input: "The cat sat",
        blocks: [{ cls: "is-dec", name: "Decoder", sub: "× N layers, causal" }],
        output: "the next token: down. It is added to the input and the model goes again.",
      },
      masks: [{ kind: "causal", title: "Self-attention, causal", rows: ["The", "cat", "sat", "down"], cols: ["The", "cat", "sat", "down"] }],
      points: [
        "Uses only the decoder stack, without the cross-attention part.",
        "<b>Causal mask</b> on every layer. Each token only sees the past.",
        "Trained on one simple objective: predict the next token.",
      ],
      best:
        "Generating text, and, as it turned out with scale, almost everything else: answering questions, writing code, reasoning, translation, summarization.",
      examples: 'The <abbr data-term="GPT">GPT</abbr> family, Llama, Mistral, Claude, Gemini, Qwen, DeepSeek.',
    },
    {
      id: "encdec",
      name: "Encoder-decoder",
      role: "transforming one text into another",
      flow: {
        input: "That is good .",
        blocks: [
          { cls: "is-enc", name: "Encoder", sub: "× N layers, bidirectional" },
          { cls: "is-dec", name: "Decoder", sub: "× N layers, causal + cross-attention", input: "output so far: Das ist" },
        ],
        output: "the next output token: gut",
      },
      masks: [
        { kind: "bidirectional", title: "Encoder self-attention, bidirectional", rows: EN, cols: EN },
        { kind: "causal", title: "Decoder self-attention, causal", rows: DE, cols: DE },
        { kind: "cross", title: "Cross-attention: decoder queries (rows), encoder keys (columns)", rows: DE, cols: EN },
      ],
      points: [
        "The full original design.",
        "The encoder reads the input with bidirectional attention.",
        "The decoder generates the output with causal self-attention, plus <b>cross-attention</b>, where decoder queries look at the encoder's outputs as keys and values.",
        "A natural fit for tasks where there is a clear input and a clear output.",
      ],
      best: "Translation, summarization, and other \"turn this text into that text\" tasks.",
      examples: '<abbr data-term="T5">T5</abbr>, <abbr data-term="BART">BART</abbr>, and the original 2017 Transformer (built for translation).',
    },
  ];

  L.l2p8 = { maskMatrix, causalRow, FAMILIES };

  if (typeof module !== "undefined" && module.exports) module.exports = defs;
  if (typeof document === "undefined") return;

  /* =========================================================
     Browser side
     ========================================================= */
  const ARROW_DOWN = '<svg class="l2p8-arrow" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v15M6 13l6 6 6-6"/></svg>';

  function initMask(fig) {
    const words = ["The", "cat", "sat", "down"];
    const n = words.length;
    const grid = fig.querySelector(".l2p8-maskgrid");
    const strip = fig.querySelector(".l2p8-strip");
    const out = fig.querySelector(".ix-readout");
    let locked = 2;

    let html = '<span class="l2p8-corner"></span>';
    words.forEach((w, k) => (html += `<span class="l2p8-colhead">key ${k}<small>${w}</small></span>`));
    words.forEach((w, q) => {
      html += `<button type="button" class="l2p8-rowbtn" data-q="${q}" aria-pressed="false">query ${q}<small>${w}</small></button>`;
      causalRow(q, n).forEach((v, k) => {
        html += `<span class="l2p8-mcell ${v === 0 ? "is-open" : "is-blocked"}" data-q="${q}">${M.fmt(v, 0)}</span>`;
      });
    });
    grid.innerHTML = html;

    function show(q) {
      grid.querySelectorAll("[data-q]").forEach((el) => el.classList.toggle("is-row", Number(el.dataset.q) === q));
      grid.querySelectorAll(".l2p8-rowbtn").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.q) === locked)));
      strip.innerHTML = words
        .map((w, k) => `<span class="l2p8-tok ${k < q ? "is-seen" : k === q ? "is-self" : "is-hidden"}">${w}</span>`)
        .join("");
      const seen = words.slice(0, q).map((w) => `"${w}"`);
      const hidden = words.slice(q + 1).map((w) => `"${w}"`);
      const list = (xs) => (xs.length < 2 ? xs.join("") : xs.slice(0, -1).join(", ") + " and " + xs[xs.length - 1]);
      let text = `<b>Query ${q} ("${words[q]}")</b> may look at ` + (seen.length ? `${list(seen)} and itself` : "only itself") + ". ";
      text += hidden.length
        ? `${list(hidden)} ${hidden.length === 1 ? "is" : "are"} in the future, so ${hidden.length === 1 ? "its score is" : "their scores are"} set to −∞ and ${hidden.length === 1 ? "it gets" : "they get"} zero attention.`
        : "It is the last token, so nothing is blocked: it sees the whole sentence.";
      out.innerHTML = text;
    }

    grid.addEventListener("click", (e) => {
      const el = e.target.closest("[data-q]");
      if (!el) return;
      locked = Number(el.dataset.q);
      show(locked);
    });
    grid.addEventListener("pointerover", (e) => {
      const el = e.target.closest("[data-q]");
      if (el && e.pointerType === "mouse") show(Number(el.dataset.q));
    });
    grid.addEventListener("pointerleave", () => show(locked));
    grid.addEventListener("focusin", (e) => {
      const el = e.target.closest("[data-q]");
      if (el) show(Number(el.dataset.q));
    });
    grid.addEventListener("focusout", () => show(locked));
    show(locked);
  }

  function maskHTML(m) {
    const mat = maskMatrix(m.kind, m.rows.length, m.cols.length);
    let g = `<div class="l2p8-mini" style="--n:${m.cols.length}" aria-hidden="true"><span></span>`;
    m.cols.forEach((c) => (g += `<span class="l2p8-mini-col">${c}</span>`));
    mat.forEach((row, q) => {
      g += `<span class="l2p8-mini-row">${m.rows[q]}</span>`;
      row.forEach((ok) => (g += `<span class="l2p8-mini-cell ${ok ? "is-open" : "is-blocked"}"></span>`));
    });
    g += "</div>";
    const open = mat.flat().filter(Boolean).length;
    const sr = `<span class="sr-only">${open} of ${mat.flat().length} cells open.</span>`;
    return `<figure class="l2p8-maskfig"><figcaption>${m.title}</figcaption>${g}${sr}</figure>`;
  }

  function flowHTML(fam) {
    const fl = fam.flow;
    const io = (t) => `<p class="l2p8-io">${t}</p>`;
    const block = (b) => `<div class="l2p8-block ${b.cls}"><b>${b.name}</b><small>${b.sub}</small></div>`;
    if (fam.blocks === undefined && fl.blocks.length === 2) {
      const [enc, dec] = fl.blocks;
      return (
        `<div class="l2p8-flow is-two">` +
        `<div class="l2p8-col">${io("input: " + fl.input)}${ARROW_DOWN}${block(enc)}</div>` +
        `<div class="l2p8-cross"><span>cross-attention</span><small>encoder outputs become the keys and values</small>` +
        '<svg class="l2p8-arrow is-side" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 12h15M13 6l6 6-6 6"/></svg></div>' +
        `<div class="l2p8-col">${io(dec.input)}${ARROW_DOWN}${block(dec)}${ARROW_DOWN}${io(fl.output)}</div>` +
        `</div>`
      );
    }
    return `<div class="l2p8-flow"><div class="l2p8-col">${io("input: " + fl.input)}${ARROW_DOWN}${block(fl.blocks[0])}${ARROW_DOWN}${io(fl.output)}</div></div>`;
  }

  function initFamilies(fig) {
    const chips = Array.from(fig.querySelectorAll("[data-family]"));
    const stage = fig.querySelector(".ix-stage");
    const out = fig.querySelector(".ix-readout");
    function render(id) {
      const fam = FAMILIES.find((x) => x.id === id);
      chips.forEach((c) => c.setAttribute("aria-pressed", String(c.dataset.family === id)));
      stage.innerHTML =
        `<div class="l2p8-fam">${flowHTML(fam)}<div class="l2p8-masks">${fam.masks.map(maskHTML).join("")}</div></div>`;
      out.innerHTML =
        `<p class="l2p8-famname"><b>${fam.name}</b>: ${fam.role}.</p>` +
        `<ul class="l2p8-points">${fam.points.map((p) => `<li>${p}</li>`).join("")}</ul>` +
        `<dl class="l2p8-facts"><dt>Best at</dt><dd>${fam.best}</dd><dt>Examples</dt><dd>${fam.examples}</dd></dl>`;
    }
    chips.forEach((c) => c.addEventListener("click", () => render(c.dataset.family)));
    render("encoder");
  }

  function init() {
    const mask = document.getElementById("l2p8-mask");
    if (mask) initMask(mask);
    const fams = document.getElementById("l2p8-families");
    if (fams) initFamilies(fams);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})(typeof window !== "undefined" ? window : globalThis);
