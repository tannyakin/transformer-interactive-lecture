// ============================================================
// Lecture 1: the Transformer. Worked example definitions for the
// walkthrough engine, plus the page's own interactive figures:
// hero attention, RNN vs attention, the QKV sentence demo, the
// encoder-decoder flow, multi-head attention and the positional
// encoding explorer. Theme, progress and navigation live in the
// shell; canvases redraw on the "lecture:theme" event.
// ============================================================
(function (root) {
  "use strict";

  const L = (root.Lecture = root.Lecture || {});
  const M = L.math;
  const defs = (L.defs = L.defs || {});
  const f = (x, dp) => M.fmt(x, dp === undefined ? 3 : dp);
  const num = (x) => (Number.isInteger(x) ? f(x, 0) : f(x, 3));
  const vecText = (v, fmt) => "[" + v.map(fmt || num).join(", ") + "]";
  const argmax = (xs) => xs.reduce((best, x, i) => (x > xs[best] ? i : best), 0);

  /* ============================================================
     Worked example: scaled dot-product attention, by hand
     ============================================================ */
  // One box of the diagram per step, in the order the maths runs.
  const SDPA_BOXES = ["mm1", "scale", "mask", "softmax", "mm2"];

  const SDPA_SVG =
    '<svg class="l1-sdpa-svg" viewBox="112 20 318 410" role="img" aria-label="Scaled dot-product attention diagram">' +
    '<defs><marker id="l1-arrowP" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="l1-arrowhead"/></marker></defs>' +
    '<text x="140" y="35" text-anchor="middle" class="l1-t-label" font-size="13">Q</text>' +
    '<text x="260" y="35" text-anchor="middle" class="l1-t-label" font-size="13">K</text>' +
    '<text x="380" y="35" text-anchor="middle" class="l1-t-label" font-size="13">V</text>' +
    '<line x1="140" y1="42" x2="140" y2="63" class="l1-flow" marker-end="url(#l1-arrowP)"/>' +
    '<line x1="260" y1="42" x2="260" y2="63" class="l1-flow" marker-end="url(#l1-arrowP)"/>' +
    '<line x1="380" y1="42" x2="380" y2="345" class="l1-flow"/>' +
    '<g class="l1-step-g" data-box="mm1"><rect class="l1-box l1-box-attn" x="140" y="65" width="120" height="38" rx="7"/>' +
    '<text x="200" y="88" text-anchor="middle" class="l1-t-label" font-size="12">MatMul</text>' +
    '<text x="270" y="88" class="l1-t-sub" font-size="9.5">q · k</text></g>' +
    '<line x1="200" y1="103" x2="200" y2="118" class="l1-flow" marker-end="url(#l1-arrowP)"/>' +
    '<g class="l1-step-g" data-box="scale"><rect class="l1-box l1-box-neutral" x="140" y="120" width="120" height="36" rx="7"/>' +
    '<text x="200" y="142" text-anchor="middle" class="l1-t-label" font-size="12">Scale</text>' +
    '<text x="270" y="142" class="l1-t-sub" font-size="9.5">÷ √d_k</text></g>' +
    '<line x1="200" y1="156" x2="200" y2="171" class="l1-flow" marker-end="url(#l1-arrowP)"/>' +
    '<g class="l1-step-g" data-box="mask"><rect class="l1-box l1-box-neutral" x="140" y="173" width="120" height="36" rx="7" stroke-dasharray="4 3"/>' +
    '<text x="200" y="195" text-anchor="middle" class="l1-t-label" font-size="12">Mask (optional)</text>' +
    '<text x="270" y="195" class="l1-t-sub" font-size="9" data-mask-note>off in the encoder</text></g>' +
    '<line x1="200" y1="209" x2="200" y2="224" class="l1-flow" marker-end="url(#l1-arrowP)"/>' +
    '<g class="l1-step-g" data-box="softmax"><rect class="l1-box l1-box-attn" x="140" y="226" width="120" height="36" rx="7"/>' +
    '<text x="200" y="248" text-anchor="middle" class="l1-t-label" font-size="12">SoftMax</text>' +
    '<text x="270" y="248" class="l1-t-sub" font-size="9.5">weights</text></g>' +
    '<line x1="200" y1="262" x2="200" y2="325" class="l1-flow" marker-end="url(#l1-arrowP)"/>' +
    '<line x1="380" y1="345" x2="362" y2="345" class="l1-flow" marker-end="url(#l1-arrowP)"/>' +
    '<g class="l1-step-g" data-box="mm2"><rect class="l1-box l1-box-attn" x="140" y="327" width="220" height="38" rx="7"/>' +
    '<text x="250" y="351" text-anchor="middle" class="l1-t-label" font-size="12">MatMul with V</text></g>' +
    '<line x1="200" y1="365" x2="200" y2="385" class="l1-flow" marker-end="url(#l1-arrowP)"/>' +
    '<text x="200" y="403" text-anchor="middle" class="l1-t-flow" font-size="11" font-weight="600">output</text>' +
    '<text x="200" y="417" text-anchor="middle" class="l1-t-sub" font-size="9" font-style="italic">weighted sum of V</text>' +
    "</svg>";

  const BOX_NAMES = {
    mm1: "MatMul of the query with the keys",
    scale: "Scale",
    mask: "Mask",
    softmax: "SoftMax",
    mm2: "MatMul with the values",
  };

  // Draws the SDPA diagram beside the numbers and lights the box for the current step.
  function sdpaPanel(el) {
    el.classList.add("l1-sdpa-panel");
    el.innerHTML = SDPA_SVG;
    const svg = el.querySelector("svg");
    const boxes = {};
    SDPA_BOXES.forEach((b) => (boxes[b] = svg.querySelector('[data-box="' + b + '"]')));
    const maskNote = svg.querySelector("[data-mask-note]");
    return {
      render(index, step, input) {
        const kind = step ? step.kind : "setup";
        const cur = step && step.box ? SDPA_BOXES.indexOf(step.box) : -1;
        const all = kind === "takeaway";
        SDPA_BOXES.forEach((b, i) => {
          const skipped = b === "mask" && !input.mask;
          boxes[b].classList.toggle("is-active", all ? !skipped : i === cur);
          boxes[b].classList.toggle("is-ahead", kind === "step" && i > cur);
        });
        maskNote.textContent = input.mask ? "causal mask on" : "off in the encoder";
        const where = kind === "step" && cur >= 0 ? "Now at: " + BOX_NAMES[step.box] + "." : "";
        svg.setAttribute("aria-label", "Scaled dot-product attention diagram. " + where);
      },
    };
  }

  defs["l1-sdpa"] = {
    id: "l1-sdpa",
    title: "Scaled dot-product attention, by hand",
    input: {
      words: ["the", "cat", "sat"],
      query: 1,
      q: [1, 0, 1, 0],
      K: [
        [1, 1, 0, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
      ],
      V: [
        [1, 0],
        [0, 1],
        [1, 1],
      ],
      mask: false,
    },
    twist: { label: "Turn on the causal mask", offLabel: "Back to no mask", input: { mask: true } },
    setup:
      "Take a three word sentence, <b>the cat sat</b>, and follow one query: the one for <b>cat</b>. " +
      "The query, keys and values have already come out of their learned linear layers. " +
      "They are tiny, with d<sub>k</sub> = 4, so you can check every number by hand.",
    setupBlocks: (input) => [
      { type: "vector", key: "q", label: "q (" + input.words[input.query] + ")", values: input.q, cols: ["d1", "d2", "d3", "d4"] },
      { type: "matrix", key: "K", rows: input.K, rowLabels: input.words.map((w) => "k " + w) },
      { type: "matrix", key: "V", rows: input.V, rowLabels: input.words.map((w) => "v " + w) },
    ],
    panel: sdpaPanel,
    build(input) {
      const { words, q, K, V, query, mask } = input;
      const me = words[query];
      const dk = q.length;
      const rootDk = Math.sqrt(dk);
      const scores = K.map((k) => M.dot(q, k));
      const scaled = scores.map((s) => s / rootDk);
      const future = words.map((_, j) => j).filter((j) => mask && j > query);
      const masked = scaled.map((s, j) => (future.includes(j) ? -Infinity : s));
      const soft = M.softmax(masked);
      const w = soft.weights;
      const out = V[0].map((_, c) => w.reduce((s, wj, j) => s + wj * V[j][c], 0));
      const top = argmax(scores);
      const win = argmax(w);

      const scoreRow = { type: "vector", key: "scores", label: "q · k", values: scores, cols: words };
      const scaledRow = { type: "vector", key: "scaled", label: "÷ √" + dk, values: scaled, tone: "result" };
      const maskedRow = { type: "vector", key: "masked", label: "after mask", values: masked, tone: "result", hl: future };

      const shownSum = M.round(w.reduce((a, x) => a + M.round(x, 3), 0), 3);
      const sumNote = shownSum === 1 ? "" : " (the rounded values add to " + f(shownSum) + " only because of rounding)";
      const byHand = V[0].map((_, c) => M.round(w.reduce((s, wj, j) => s + M.round(wj, 3) * V[j][c], 0), 3));
      const drift = byHand.some((x, c) => x !== M.round(out[c], 3));
      const futureWords = future.map((j) => "<b>" + words[j] + "</b>").join(" and ");

      const maskStep = mask
        ? {
            title: "Hide the future with the mask",
            box: "mask",
            say:
              `In the decoder, the model is writing <b>${me}</b> right now and ${futureWords} has not been written yet. ` +
              "The mask sets every future score to −∞, and softmax will turn that into a weight of exactly 0.",
            blocks: [
              Object.assign({}, scaledRow, { tone: null, cols: words }),
              Object.assign({}, maskedRow, { op: "=", fresh: true }),
            ],
          }
        : {
            title: "The mask: skipped in the encoder",
            box: "mask",
            say:
              "This is encoder self-attention, so nothing is hidden: every word may look at every other word, " +
              "including the ones after it. The scores pass straight through. Use the causal mask button above to see the decoder's version.",
            blocks: [
              Object.assign({}, maskedRow, { cols: words }),
              { type: "note", html: "No mask here. The decoder's first attention layer is the only one that uses it." },
            ],
          };

      return {
        steps: [
          {
            title: "Compare the query with every key",
            box: "mm1",
            say:
              "A dot product multiplies matching positions and adds them up, so it is large when two vectors point the same way. " +
              `The key for <b>${words[top]}</b> lines up with the query best and scores ${num(scores[top])}. ` +
              "In matrix form this is MatMul(Q, K<sup>T</sup>), every query against every key at once. We are following one row of it.",
            blocks: [
              {
                type: "lines",
                fresh: true,
                lines: K.map((k, j) => ({
                  t: `q · k<sub>${words[j]}</sub> = ${q.map((x, i) => num(x) + "×" + num(k[i])).join(" + ")} = ${num(scores[j])}`,
                  hl: j === top,
                })),
              },
              Object.assign({}, scoreRow, { fresh: true, hl: [top] }),
            ],
          },
          {
            title: "Scale by √d<sub>k</sub>",
            box: "scale",
            say:
              "Longer vectors give bigger dot products, and big scores push softmax to hand nearly everything to one word. " +
              `Dividing by √d<sub>k</sub> = √${dk} = ${num(rootDk)} keeps the scores in a calm range. ` +
              "The real model has d<sub>k</sub> = 64, so it divides by 8.",
            blocks: [
              scoreRow,
              { type: "lines", lines: scores.map((s, j) => ({ t: `${words[j]}:  ${num(s)} / ${num(rootDk)} = ${num(scaled[j])}` })) },
              Object.assign({}, scaledRow, { op: "=", fresh: true }),
            ],
          },
          maskStep,
          {
            title: "Softmax turns scores into weights",
            box: "softmax",
            say:
              `Softmax raises e to each score, then divides by the total, ${f(soft.sum)}. ` +
              `The weights add up to 1${sumNote}, and <b>${words[win]}</b> takes the biggest share: ${f(w[win])}.`,
            predict: {
              ask: `Before softmax runs: which word will <b>${me}</b> pay the most attention to?`,
              choices: words.slice(),
              answer: win,
              why: mask
                ? `The mask hid ${futureWords}, so of the words that are left, <b>${words[win]}</b> has the biggest score (${num(masked[win])}). Softmax never changes the order of the scores.`
                : `<b>${words[win]}</b> has the biggest scaled score, ${num(masked[win])}. Softmax never changes the order of the scores, it only turns them into shares of 1.`,
              hint: "Softmax keeps the order of the scores. Find the biggest score that is not −∞.",
            },
            blocks: [
              {
                type: "lines",
                lines: masked
                  .map((s, j) => ({ t: `e<sup>${num(s)}</sup> = ${f(soft.exps[j])}` }))
                  .concat([{ t: `sum = ${f(soft.sum)}`, hl: true }]),
              },
              { type: "vector", key: "exps", label: "eˣ", values: soft.exps, dp: 3 },
              { type: "bars", key: "weights", labels: words, values: w, hl: [win], fresh: true },
            ],
          },
          {
            title: "Blend the values by those weights",
            box: "mm2",
            say:
              `The output for <b>${me}</b> is every value vector, scaled by its weight and added up. ` +
              `It leans towards v<sub>${words[win]}</sub> because that word won the most attention.` +
              (mask ? ` ${futureWords} adds nothing, because its weight is 0.` : ""),
            blocks: [
              {
                type: "lines",
                fresh: true,
                lines: words
                  .map((word, j) => ({ t: `${f(w[j])} × v<sub>${word}</sub> = ${f(w[j])} × ${vecText(V[j])}` }))
                  .concat([{ t: `output = ${vecText(out, (x) => f(x))}`, hl: true }]),
              },
              { type: "vector", key: "output", label: "output (" + me + ")", values: out, dp: 3, tone: "result", fresh: true },
            ].concat(
              drift
                ? [{ type: "note", html: "Worked with the unrounded weights, so adding up the rounded numbers by hand can be off by 0.001." }]
                : []
            ),
          },
        ],
        takeaway: mask
          ? "The mask changed nothing about the arithmetic. It only set future scores to −∞ before softmax, so their weights became exactly 0 " +
            "and their share went to the words already written. That is how the decoder is kept from peeking ahead while it generates."
          : `Score, scale, softmax, blend. The word <b>${me}</b> came out as a mix of the whole sentence, weighted by how well each key matched its query. ` +
            "Every attention box in the transformer does exactly this, with d<sub>k</sub> = 64, every query at once, and eight heads side by side.",
      };
    },
  };

  if (typeof module !== "undefined" && module.exports) module.exports = defs;
  if (typeof document === "undefined") return;

  /* ============================================================
     Browser side: the page's interactive figures
     ============================================================ */
  const svgNS = "http://www.w3.org/2000/svg";
  const REDUCE = root.matchMedia && root.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (id) => document.getElementById(id);
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function svgEl(tag, attrs, cls) {
    const e = document.createElementNS(svgNS, tag);
    Object.keys(attrs || {}).forEach((k) => e.setAttribute(k, attrs[k]));
    if (cls) e.setAttribute("class", cls);
    return e;
  }

  // Run fn once, the first time el is at least `ratio` visible.
  function onceVisible(el, ratio, fn) {
    if (!el) return;
    if (!("IntersectionObserver" in root)) return;
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            io.disconnect();
            fn();
          }
        });
      },
      { threshold: ratio }
    );
    io.observe(el);
  }

  /* ---------- shared: a row of word chips inside an SVG ---------- */
  function buildWordRow(svg, words, o) {
    svg.replaceChildren();
    const items = [];
    let x = 8;
    words.forEach((w) => {
      const width = Math.max(36, w.length * o.charW + 28);
      items.push({ word: w, x, width });
      x += width + 10;
    });
    const totalWidth = x - 10 + 8;
    const rowY = o.top;
    const height = o.height;
    svg.setAttribute("viewBox", `0 0 ${totalWidth} ${rowY + height + 14}`);
    const lines = svgEl("g", {}, "l1-lines");
    svg.append(lines);
    const groups = [];
    const centers = [];
    items.forEach((it, i) => {
      const g = svgEl("g", {}, o.groupClass);
      const rect = svgEl("rect", { x: it.x, y: rowY, width: it.width, height, rx: 8 }, "l1-word-rect");
      const text = svgEl("text", { x: it.x + it.width / 2, y: rowY + height / 2 + 4.5, "text-anchor": "middle" }, "l1-word-text");
      text.textContent = it.word;
      g.append(rect, text);
      if (o.onPick) {
        g.setAttribute("tabindex", "0");
        g.setAttribute("role", "button");
        g.setAttribute("aria-pressed", "false");
        g.setAttribute("aria-label", `Make "${it.word}" the query`);
        g.addEventListener("click", () => o.onPick(i));
        g.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            o.onPick(i);
          }
        });
      }
      svg.append(g);
      groups.push(g);
      centers.push({ x: it.x + it.width / 2, top: rowY });
    });
    return { groups, centers, lines };
  }

  function drawAttentionLines(row, qi, weights) {
    row.lines.replaceChildren();
    const q = row.centers[qi];
    weights.forEach((w, j) => {
      if (j === qi || w < 0.025) return;
      const t = row.centers[j];
      const midx = (q.x + t.x) / 2;
      const curveH = Math.min(q.top, t.top) - 16 - Math.abs(q.x - t.x) * 0.05;
      const path = svgEl("path", { d: `M${q.x},${q.top} Q${midx},${curveH} ${t.x},${t.top}` }, "l1-attn-line");
      path.style.strokeWidth = (1 + w * 8.5).toFixed(2);
      path.style.strokeOpacity = Math.min(1, 0.14 + w * 1.05).toFixed(2);
      row.lines.append(path);
    });
  }

  // Hand-shaped, illustrative weights: nearby words get some attention,
  // and a few pairs get a boost so the pattern tells a story.
  function illustrativeWeights(n, qi, boosts) {
    const w = Array.from({ length: n }, (_, i) => Math.exp(-((i - qi) * (i - qi)) / (2 * 1.5 * 1.5)));
    if (boosts) Object.keys(boosts).forEach((k) => (w[+k] += boosts[k]));
    const sum = w.reduce((a, b) => a + b, 0);
    return w.map((x) => x / sum);
  }

  const SENTENCES = [
    {
      label: "the tired animal",
      words: "the animal didn't cross the street because it was too tired".split(" "),
      boosts: { 7: { 1: 2.6 }, 10: { 1: 1.3, 7: 0.9 } },
      flagship: 7,
    },
    {
      label: "the poured water",
      words: "she poured water from the pitcher until it was full".split(" "),
      boosts: { 7: { 5: 2.4, 2: 0.7 } },
      flagship: 7,
    },
    {
      label: "the trophy and suitcase",
      words: "the trophy doesn't fit in the suitcase because it is too big".split(" "),
      boosts: { 8: { 1: 2.5 } },
      flagship: 8,
    },
  ];

  /* ---------- hero: attention refocusing on its own ---------- */
  function initHero() {
    const svg = $("l1-hero-sentence");
    if (!svg) return;
    const tag = $("l1-hero-tag");
    const btn = $("l1-hero-toggle");
    const cycle = [
      { s: 0, q: 7 },
      { s: 0, q: 10 },
      { s: 1, q: 7 },
      { s: 2, q: 8 },
    ];
    let idx = 0;
    let shown = -1;
    let row = null;
    let timer = null;
    let userPaused = false;
    let inView = false;

    function render() {
      const step = cycle[idx];
      const s = SENTENCES[step.s];
      if (step.s !== shown) {
        row = buildWordRow(svg, s.words, { charW: 8.2, height: 36, top: 56, groupClass: "l1-hero-word" });
        shown = step.s;
      }
      row.groups.forEach((g, i) => g.classList.toggle("is-focus", i === step.q));
      const weights = illustrativeWeights(s.words.length, step.q, s.boosts[step.q]);
      drawAttentionLines(row, step.q, weights);
      const top = weights.reduce((b, x, i) => (i !== step.q && x > weights[b] ? i : b), step.q === 0 ? 1 : 0);
      tag.textContent = `example ${idx + 1} of ${cycle.length}`;
      svg.setAttribute(
        "aria-label",
        `The sentence "${s.words.join(" ")}". The word "${s.words[step.q]}" attends most strongly to "${s.words[top]}".`
      );
    }
    function next() {
      idx = (idx + 1) % cycle.length;
      render();
    }
    function start() {
      if (timer || REDUCE || userPaused || !inView) return;
      timer = setInterval(next, 3200);
    }
    function stop() {
      clearInterval(timer);
      timer = null;
    }
    function syncBtn() {
      if (REDUCE) {
        btn.textContent = "Next example";
        btn.setAttribute("aria-label", "Show the next example");
        return;
      }
      btn.textContent = userPaused ? "Play" : "Pause";
      btn.setAttribute("aria-label", userPaused ? "Play the attention animation" : "Pause the attention animation");
    }
    btn.addEventListener("click", () => {
      if (REDUCE) return next();
      userPaused = !userPaused;
      if (userPaused) stop();
      else start();
      syncBtn();
    });
    render();
    syncBtn();
    if ("IntersectionObserver" in root) {
      new IntersectionObserver(
        (entries) => {
          entries.forEach((e) => {
            inView = e.isIntersecting;
            if (inView) start();
            else stop();
          });
        },
        { threshold: 0.4 }
      ).observe(svg);
    }
  }

  /* ---------- Part 1: recurrent vs attention ---------- */
  function initCompare() {
    const rnnSvg = $("l1-rnn-svg");
    const attnSvg = $("l1-attn-svg");
    if (!rnnSvg || !attnSvg) return;
    const N = 5;
    const X = [34, 92, 150, 208, 266];
    const Y = 75;
    const R = 15;

    function build(svg, mode) {
      svg.replaceChildren();
      const edges = [];
      if (mode === "chain") {
        for (let i = 0; i < N - 1; i++) {
          edges.push(svg.appendChild(svgEl("line", { x1: X[i] + R, y1: Y, x2: X[i + 1] - R, y2: Y }, "l1-rnn-edge")));
        }
      } else {
        for (let i = 0; i < N; i++) {
          for (let j = i + 1; j < N; j++) {
            const curveY = Y - 20 - (X[j] - X[i]) * 0.12 * ((i + j) % 2 === 0 ? 1 : 0.55);
            const d = `M${X[i]},${Y - R} Q${(X[i] + X[j]) / 2},${curveY} ${X[j]},${Y - R}`;
            edges.push(svg.appendChild(svgEl("path", { d }, "l1-attn-edge")));
          }
        }
      }
      const nodes = [];
      for (let i = 0; i < N; i++) {
        nodes.push(svg.appendChild(svgEl("circle", { cx: X[i], cy: Y, r: R }, mode === "chain" ? "l1-rnn-node" : "l1-attn-node")));
        const t = svgEl("text", { x: X[i], y: Y + 34, "text-anchor": "middle" }, "l1-node-label");
        t.textContent = "w" + (i + 1);
        svg.append(t);
      }
      return { nodes, edges };
    }

    const rnn = build(rnnSvg, "chain");
    const attn = build(attnSvg, "mesh");
    const pairs = attn.edges.length;
    const rnnMeta = $("l1-rnn-meta");
    const attnMeta = $("l1-attn-meta");
    const rnnBtn = $("l1-rnn-replay");
    const attnBtn = $("l1-attn-replay");
    let rnnRunning = false;

    async function playRnn() {
      if (rnnRunning) return;
      rnnRunning = true;
      rnnBtn.disabled = true;
      rnn.nodes.forEach((n) => n.classList.remove("on"));
      for (let i = 0; i < N; i++) {
        await new Promise((r) => setTimeout(r, REDUCE ? 0 : 480));
        rnn.nodes[i].classList.add("on");
        rnnMeta.textContent = `step ${i + 1} of ${N}`;
      }
      rnnMeta.textContent = `${N} steps, one after another`;
      rnnBtn.disabled = false;
      rnnRunning = false;
    }

    function playAttn() {
      attn.nodes.forEach((n) => n.classList.remove("on"));
      attn.edges.forEach((e) => e.classList.remove("on"));
      requestAnimationFrame(() => {
        attn.nodes.forEach((n) => n.classList.add("on"));
        attn.edges.forEach((e) => e.classList.add("on"));
        attnMeta.textContent = `1 step, all ${pairs} pairs at once`;
      });
    }

    rnnBtn.addEventListener("click", playRnn);
    attnBtn.addEventListener("click", playAttn);
    onceVisible($("l1-compare"), 0.5, () => {
      playRnn();
      playAttn();
    });
  }

  /* ---------- Part 2: click a word to make it the query ---------- */
  function initQkv() {
    const svg = $("l1-qkv-sentence");
    if (!svg) return;
    const bars = $("l1-qkv-bars");
    const readout = $("l1-qkv-readout");
    const chips = Array.from(document.querySelectorAll("#l1-qkv-chips .chip"));
    let sIdx = 0;
    let row = null;

    function select(qi) {
      const s = SENTENCES[sIdx];
      row.groups.forEach((g, i) => {
        g.classList.toggle("is-query", i === qi);
        g.setAttribute("aria-pressed", String(i === qi));
      });
      const weights = illustrativeWeights(s.words.length, qi, s.boosts[qi]);
      drawAttentionLines(row, qi, weights);
      const order = weights.map((w, i) => ({ w, i })).sort((a, b) => b.w - a.w);
      bars.replaceChildren(
        ...order.map(({ w, i }) => {
          const li = document.createElement("li");
          li.className = "l1-weight" + (i === qi ? " is-self" : "");
          const pct = Math.round(w * 100);
          li.innerHTML = '<span class="l1-weight-word"></span><span class="l1-weight-track"><span class="l1-weight-fill"></span></span><span class="l1-weight-val"></span>';
          li.querySelector(".l1-weight-word").textContent = s.words[i] + (i === qi ? " (itself)" : "");
          li.querySelector(".l1-weight-val").textContent = pct + "%";
          li.querySelector(".l1-weight-fill").style.setProperty("--w", pct + "%");
          return li;
        })
      );
      const top = order.find((o) => o.i !== qi) || order[0];
      readout.innerHTML =
        `Query: <b class="math">"${s.words[qi]}"</b>. Its strongest match is <b class="math">"${s.words[top.i]}"</b> at ${Math.round(top.w * 100)}%. ` +
        `That is how the model works out what a word like "${s.words[qi]}" is really pointing to.`;
    }

    function showSentence(i) {
      sIdx = i;
      chips.forEach((c, k) => c.setAttribute("aria-pressed", String(k === i)));
      row = buildWordRow(svg, SENTENCES[i].words, {
        charW: 8,
        height: 34,
        top: 44,
        groupClass: "l1-qkv-word",
        onPick: select,
      });
      select(SENTENCES[i].flagship);
    }

    chips.forEach((c, k) => c.addEventListener("click", () => showSentence(k)));
    showSentence(0);
  }

  /* ---------- Part 3: one token's trip through the network ---------- */
  function initArchitecture() {
    const svg = $("l1-arch-svg");
    if (!svg) return;
    const runBtn = $("l1-arch-run");
    const resetBtn = $("l1-arch-reset");
    const status = $("l1-arch-status");
    const pulse = $("l1-arch-pulse");
    const IDLE = "Press run to watch data move through the network.";
    const NODES = Array.from(svg.querySelectorAll(".l1-box[id]"));
    const PATH = [
      { id: "l1-enc-embed", x: 260, y: 77.5, status: "1 · The encoder turns each input token into a vector and adds its positional encoding." },
      { id: "l1-enc-pe", x: 260, y: 127 },
      { id: "l1-enc-attn", x: 260, y: 182.5, status: "2 · Self-attention: every input word looks at every other input word." },
      { id: "l1-enc-addnorm1", x: 260, y: 234.5 },
      { id: "l1-enc-ffn", x: 260, y: 286.5 },
      { id: "l1-enc-addnorm2", x: 260, y: 338.5 },
      { id: "", x: 260, y: 371 },
      { id: "", x: 470, y: 371, status: "3 · The encoder's output becomes K and V for the cross-attention in every decoder layer." },
      { id: "", x: 470, y: 287 },
      { id: "l1-dec-cross", x: 680, y: 286.5 },
      { id: "l1-dec-embed", x: 680, y: 77.5, status: "4 · Meanwhile the decoder embeds the tokens it has written so far." },
      { id: "l1-dec-pe", x: 680, y: 127 },
      { id: "l1-dec-mask", x: 680, y: 182.5, status: "5 · Masked self-attention: each position sees only itself and the positions before it." },
      { id: "l1-dec-addnorm1", x: 680, y: 234.5 },
      { id: "l1-dec-cross", x: 680, y: 286.5, status: "6 · Cross-attention: the decoder's queries meet the encoder's keys and values." },
      { id: "l1-dec-addnorm2", x: 680, y: 338.5 },
      { id: "l1-dec-ffn", x: 680, y: 390.5 },
      { id: "l1-dec-addnorm3", x: 680, y: 442.5 },
      { id: "l1-dec-linear", x: 680, y: 495, status: "7 · A linear layer scores every word in the vocabulary, and softmax turns the scores into probabilities." },
      { id: "l1-dec-softmax", x: 680, y: 547 },
    ];
    let runId = 0;

    function clear() {
      NODES.forEach((n) => n.classList.remove("is-active"));
      pulse.classList.remove("show");
    }

    function hop(a, b, dur, id) {
      return new Promise((resolve) => {
        const start = performance.now();
        function frame(now) {
          if (id !== runId) return resolve(false);
          const p = Math.min(1, (now - start) / dur);
          const e = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
          pulse.setAttribute("cx", a.x + (b.x - a.x) * e);
          pulse.setAttribute("cy", a.y + (b.y - a.y) * e);
          if (p < 1) requestAnimationFrame(frame);
          else resolve(true);
        }
        requestAnimationFrame(frame);
      });
    }

    function arrive(pt) {
      const el = pt.id && $(pt.id);
      if (el) el.classList.add("is-active");
      if (pt.status) status.innerHTML = "<b>" + pt.status + "</b>";
    }

    async function run() {
      const id = ++runId;
      clear();
      runBtn.disabled = true;
      if (REDUCE) {
        PATH.forEach(arrive);
        status.innerHTML = "<b>Done.</b> Every lit box is on one token's round trip through the network.";
        runBtn.disabled = false;
        return;
      }
      pulse.classList.add("show");
      pulse.setAttribute("cx", PATH[0].x);
      pulse.setAttribute("cy", PATH[0].y);
      arrive(PATH[0]);
      for (let i = 1; i < PATH.length; i++) {
        const ok = await hop(PATH[i - 1], PATH[i], 420, id);
        if (!ok) return;
        arrive(PATH[i]);
      }
      pulse.classList.remove("show");
      status.innerHTML = "<b>Done.</b> Every lit box is on one token's round trip through the network.";
      runBtn.disabled = false;
    }

    function reset() {
      runId++;
      clear();
      runBtn.disabled = false;
      status.textContent = IDLE;
    }

    runBtn.addEventListener("click", run);
    resetBtn.addEventListener("click", reset);
    status.textContent = IDLE;
    if (!REDUCE) onceVisible(svg, 0.4, run);
  }

  /* ---------- Part 4: multi-head attention ---------- */
  function initMultiHead() {
    const fig = $("l1-mh");
    if (!fig) return;
    const D_MODEL = 512;
    const stack = $("l1-mh-stack");
    const concatNote = $("l1-mh-concat-note");
    const readout = $("l1-mh-readout");
    const headChips = Array.from(fig.querySelectorAll("[data-heads]"));
    const stageChips = Array.from(fig.querySelectorAll("[data-stage]"));
    const STAGE_BOXES = {
      project: ["l1-mh-lin-q", "l1-mh-lin-k", "l1-mh-lin-v"],
      attend: ["l1-mh-stack"],
      concat: ["l1-mh-concat"],
      mix: ["l1-mh-linout"],
    };
    let h = 8;
    let stage = "project";

    function drawStack() {
      stack.replaceChildren();
      const cards = Math.min(h, 4);
      for (let c = cards - 1; c >= 0; c--) {
        const front = c === 0;
        const rect = svgEl(
          "rect",
          { x: 588 + c * 10, y: 126 + c * 7, width: 184, height: 66, rx: 8 },
          "l1-box " + (front ? "l1-box-attn" : "l1-box-neutral")
        );
        if (!front) rect.setAttribute("opacity", String(0.85 - c * 0.15));
        stack.append(rect);
      }
      const lines = [
        ["Scaled Dot-Product", 152, "l1-t-label", 11.5],
        ["Attention", 166, "l1-t-label", 11.5],
        [h === 1 ? "× 1 (a single head)" : `× ${h} heads, in parallel`, 182, "l1-t-sub", 9.5],
      ];
      lines.forEach(([t, y, cls, size]) => {
        const e = svgEl("text", { x: 680, y, "text-anchor": "middle", "font-size": size }, cls);
        e.textContent = t;
        stack.append(e);
      });
      concatNote.textContent = `${h} × ${D_MODEL / h} = ${D_MODEL}`;
    }

    function render() {
      const dk = D_MODEL / h;
      headChips.forEach((c) => c.setAttribute("aria-pressed", String(+c.dataset.heads === h)));
      stageChips.forEach((c) => c.setAttribute("aria-pressed", String(c.dataset.stage === stage)));
      Object.keys(STAGE_BOXES).forEach((s) =>
        STAGE_BOXES[s].forEach((id) => {
          const el = $(id);
          if (el) el.classList.toggle("is-active", s === stage);
        })
      );
      drawStack();
      const heads = h === 1 ? "the single head" : `each of the ${h} heads`;
      const TEXT = {
        project:
          `Q, K and V each go through their own learned linear layer, once per head. With d<sub>model</sub> = ${D_MODEL} and h = ${h}, ` +
          `${heads} works on vectors of length d<sub>k</sub> = ${D_MODEL} / ${h} = <b>${dk}</b>.`,
        attend:
          h === 1
            ? "One head runs the scaled dot-product attention you just stepped through, on the full-width vectors. It can only learn one way of looking at the sentence."
            : `All ${h} heads run the scaled dot-product attention you just stepped through, side by side, each on its own ${dk}-number slice. One head might track grammar, another which noun "it" refers to.`,
        concat: `The ${h} head output${h === 1 ? " is" : "s are"} glued end to end: ${h} × ${dk} = <b>${D_MODEL}</b> numbers per token, the same width we started with.`,
        mix: `One more linear layer, W<sup>O</sup>, mixes what the heads found into a single ${D_MODEL}-number vector per token. The paper's base model uses h = 8, so d<sub>k</sub> = 64.`,
      };
      readout.innerHTML = TEXT[stage];
    }

    headChips.forEach((c) =>
      c.addEventListener("click", () => {
        h = +c.dataset.heads;
        render();
      })
    );
    stageChips.forEach((c) =>
      c.addEventListener("click", () => {
        stage = c.dataset.stage;
        render();
      })
    );
    render();
  }

  /* ---------- Part 5: positional encoding explorer ---------- */
  function initPositional() {
    const canvas = $("l1-pe-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const iInput = $("l1-pe-i");
    const dInput = $("l1-pe-d");
    const iVal = $("l1-pe-i-val");
    const dVal = $("l1-pe-d-val");
    const tokensRow = $("l1-pe-tokens");
    const simStatus = $("l1-pe-status");
    const heatmap = $("l1-pe-heatmap");
    const heatA = $("l1-pe-heat-a");
    const heatB = $("l1-pe-heat-b");
    const labelA = $("l1-pe-label-a");
    const labelB = $("l1-pe-label-b");
    const TOKENS = 16;
    const N_POS = 80;
    const H = 240;
    let selected = [];

    function cosine(a, b) {
      const na = Math.sqrt(M.dot(a, a));
      const nb = Math.sqrt(M.dot(b, b));
      return M.dot(a, b) / (na * nb);
    }

    function renderTokens() {
      tokensRow.replaceChildren();
      for (let pos = 0; pos < TOKENS; pos++) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "l1-pe-token";
        btn.textContent = String(pos);
        const slot = selected.indexOf(pos);
        if (slot === 0) btn.classList.add("is-a");
        if (slot === 1) btn.classList.add("is-b");
        btn.setAttribute("aria-pressed", String(slot >= 0));
        btn.setAttribute("aria-label", "Position " + pos + (slot === 0 ? ", first pick" : slot === 1 ? ", second pick" : ""));
        btn.addEventListener("click", () => {
          if (selected.length >= 2) selected = [];
          selected.push(pos);
          renderTokens();
          compare();
          draw();
          const again = tokensRow.children[pos];
          if (again) again.focus({ preventScroll: true });
        });
        tokensRow.append(btn);
      }
    }

    function heatRow(container, vec) {
      container.replaceChildren(
        ...vec.map((v) => {
          const cell = document.createElement("span");
          const t = Math.max(-1, Math.min(1, v));
          cell.style.background = t < 0 ? "var(--encoder)" : "var(--decoder)";
          cell.style.opacity = (Math.abs(t) * 0.85 + 0.15).toFixed(2);
          return cell;
        })
      );
    }

    function compare() {
      if (selected.length < 2) {
        heatmap.hidden = true;
        simStatus.textContent =
          selected.length === 0
            ? "Pick a position above, then a second one, to compare them."
            : `Now pick a second position to compare with position ${selected[0]}.`;
        return;
      }
      const d = +dInput.value;
      const [a, b] = selected;
      const va = M.sinusoidalPE(a, d);
      const vb = M.sinusoidalPE(b, d);
      const sim = cosine(va, vb);
      const label = sim > 0.9 ? "almost identical" : sim > 0.6 ? "fairly similar" : sim > 0.3 ? "somewhat different" : "clearly different";
      const note =
        a === b
          ? "It is the same position, so the encodings match exactly."
          : Math.abs(a - b) <= 2
          ? "They sit close together, so their encodings are still alike."
          : "They sit further apart, so their encodings point in noticeably different directions.";
      simStatus.innerHTML = `Position <b>${a}</b> and position <b>${b}</b>: cosine similarity <b class="math">${f(sim, 2)}</b>, ${label}. ${note}`;
      heatmap.hidden = false;
      labelA.textContent = "position " + a;
      labelB.textContent = "position " + b;
      heatRow(heatA, va);
      heatRow(heatB, vb);
    }

    function fit() {
      const w = canvas.getBoundingClientRect().width || 600;
      const dpr = root.devicePixelRatio || 1;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function draw() {
      const i = +iInput.value;
      const d = +dInput.value;
      const w = canvas.getBoundingClientRect().width || 600;
      ctx.clearRect(0, 0, w, H);
      const col = {
        border: cssVar("--border"),
        faint: cssVar("--ink-faint"),
        enc: cssVar("--encoder"),
        dec: cssVar("--decoder"),
        accent: cssVar("--accent"),
        attn: cssVar("--attn"),
        surface: cssVar("--surface"),
      };
      const freq = 1 / Math.pow(10000, (2 * i) / d);
      const mx = 6;
      const plotW = w - mx * 2;
      const amp = H / 2 - 34;
      const xFor = (pos) => mx + (pos / N_POS) * plotW;
      const yFor = (pos, fn) => H / 2 - fn(pos * freq) * amp;

      ctx.strokeStyle = col.border;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, H / 2);
      ctx.lineTo(w, H / 2);
      ctx.stroke();

      function plot(fn, color, dash) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.2;
        ctx.lineJoin = "round";
        ctx.setLineDash(dash || []);
        ctx.beginPath();
        for (let pos = 0; pos <= N_POS + 1e-6; pos += 0.15) {
          const x = xFor(pos);
          const y = yFor(pos, fn);
          if (pos === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }
      plot(Math.sin, col.enc);
      plot(Math.cos, col.dec, [7, 4]);

      ctx.fillStyle = col.faint;
      ctx.globalAlpha = 0.6;
      for (let pos = 0; pos < TOKENS; pos++) {
        ctx.beginPath();
        ctx.arc(xFor(pos), yFor(pos, Math.sin), 2.6, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      selected.forEach((pos, k) => {
        const color = k === 0 ? col.accent : col.attn;
        const x = xFor(pos);
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.5;
        ctx.lineWidth = 1.4;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(x, 12);
        ctx.lineTo(x, H - 12);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
        ctx.fillStyle = color;
        [Math.sin, Math.cos].forEach((fn) => {
          ctx.beginPath();
          ctx.arc(x, yFor(pos, fn), 5, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.font = "600 11px 'JetBrains Mono', monospace";
        ctx.fillText(String(pos), x + 5, 56 + k * 14);
      });

      function chip(text, x, y, color) {
        ctx.font = "600 11.5px 'JetBrains Mono', monospace";
        const width = ctx.measureText(text).width;
        const left = Math.max(2, Math.min(x, w - width - 8));
        ctx.fillStyle = col.surface;
        ctx.globalAlpha = 0.92;
        ctx.fillRect(left - 5, y - 12, width + 10, 17);
        ctx.globalAlpha = 1;
        ctx.fillStyle = color;
        ctx.fillText(text, left, y);
      }
      chip("sin, even dimension (solid)", mx + 4, 18, col.enc);
      chip("cos, odd dimension (dashed)", mx + 4, 36, col.dec);
      chip(`positions 0 to ${N_POS}, dots mark 0 to ${TOKENS - 1}`, w - 290, H - 10, col.faint);

      iVal.textContent = "i = " + i;
      dVal.textContent = "d_model = " + d;
      canvas.setAttribute(
        "aria-label",
        `Sine and cosine waves for dimension pair i = ${i} with d_model = ${d}. One full wave spans about ${f((2 * Math.PI) / freq, 0)} positions.`
      );
    }

    iInput.addEventListener("input", draw);
    dInput.addEventListener("input", () => {
      const maxI = Math.max(0, +dInput.value / 2 - 1);
      iInput.max = String(maxI);
      if (+iInput.value > maxI) iInput.value = String(maxI);
      draw();
      compare();
    });
    let resizeRaf = 0;
    root.addEventListener("resize", () => {
      cancelAnimationFrame(resizeRaf);
      resizeRaf = requestAnimationFrame(() => {
        fit();
        draw();
      });
    });
    document.addEventListener("lecture:theme", draw);
    renderTokens();
    compare();
    fit();
    draw();
    // Web fonts can arrive after the first paint; redraw so canvas labels use them.
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(draw);
  }

  /* ---------- Part 7: glossary filter ---------- */
  function initGlossary() {
    const input = $("l1-gloss-search");
    if (!input) return;
    const items = Array.from(document.querySelectorAll(".l1-gloss-item"));
    const status = $("l1-gloss-status");
    const empty = $("l1-gloss-empty");
    function apply() {
      const q = input.value.trim().toLowerCase();
      let shown = 0;
      items.forEach((it) => {
        const match = !q || it.textContent.toLowerCase().includes(q);
        it.hidden = !match;
        if (match) shown++;
      });
      empty.hidden = shown !== 0;
      status.textContent = q ? `Showing ${shown} of ${items.length} terms.` : `${items.length} terms.`;
    }
    input.addEventListener("input", apply);
    apply();
  }

  function init() {
    initHero();
    initCompare();
    initQkv();
    initArchitecture();
    initMultiHead();
    initPositional();
    initGlossary();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})(typeof window !== "undefined" ? window : globalThis);
