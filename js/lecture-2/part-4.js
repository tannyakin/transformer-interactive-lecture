// ============================================================
// Lecture 2, Part 4: the position-wise feedforward network.
// Walkthrough: the FFN by hand, with a panel of hidden features.
// Interactives: the shape of the journey, position-wise, the
// parameter share, activation curves, the SwiGLU gate and the
// MoE router. Every number is computed, never typed.
// ============================================================
(function (root) {
  "use strict";

  const L = (root.Lecture = root.Lecture || {});
  const M = L.math;
  const defs = (L.defs = L.defs || {});

  /* ---------------- formatting helpers ---------------- */
  function dpFor(xs) {
    for (let dp = 0; dp <= 3; dp++) {
      if (xs.every((v) => Math.abs(M.round(v, dp) - M.round(v, 3)) < 1e-9)) return dp;
    }
    return 3;
  }
  const num = (v) => M.fmt(v, dpFor([v]));
  const vstr = (xs) => "[" + xs.map((v) => M.fmt(v, dpFor(xs))).join(", ") + "]";
  const par = (v) => (v < 0 ? "(" + num(v) + ")" : num(v));

  /* ---------------- pure helpers (tested) ---------------- */
  const attnParams = (d) => 4 * d * d;
  const ffnParams = (d, dff) => 2 * d * dff;
  const swigluParams = (d, dff) => 3 * d * dff;

  // erf by Abramowitz and Stegun 7.1.26 (error below 1.5e-7), enough for a plot.
  function erf(x) {
    const s = Math.sign(x);
    const a = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * a);
    const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
    return s * y;
  }
  const relu1 = (x) => Math.max(0, x);
  const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const swish = (x) => x * sigmoid(x);

  // A stand-in router: a fixed score per (token, expert) pair, so the demo is repeatable.
  function hash(str) {
    let h = 2166136261;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return (h >>> 0) / 4294967295;
  }
  function route(token, experts, k) {
    const scores = Array.from({ length: experts }, (_, e) => 4 * hash(token + "#" + e) - 2);
    const probs = M.softmax(scores).weights;
    const order = probs.map((p, i) => i).sort((a, b) => probs[b] - probs[a]);
    return { probs, top: order.slice(0, k).sort((a, b) => a - b) };
  }

  function onFeatures(hidden) {
    return hidden.map((v, i) => (v > 0 ? i : -1)).filter((i) => i >= 0);
  }
  function featureLabel(set) {
    if (set.length === 0) return "None of them";
    if (set.length === 4) return "All four";
    if (set.length === 1) return "Only feature " + (set[0] + 1);
    const n = set.map((i) => i + 1);
    return "Features " + n.slice(0, -1).join(", ") + " and " + n[n.length - 1];
  }

  L.l2p4 = { attnParams, ffnParams, swigluParams, relu1, gelu, swish, sigmoid, route, onFeatures, featureLabel };

  /* ---------------- 4.5 FFN by hand ---------------- */
  const W1 = [
    [1, -1, 0, 2],
    [0, 1, -1, 1],
  ];
  const W2 = [
    [1, 0],
    [0, 1],
    [1, 1],
    [0.5, -0.5],
  ];

  defs["l2-ffn"] = {
    id: "l2-ffn",
    title: "The FFN, by hand",
    input: { x: [1, 2], W1, W2 },
    twist: { label: "Change x to [1, −2]", offLabel: "Back to x = [1, 2]", input: { x: [1, -2] } },
    setup:
      "A tiny FFN with d_model = 2 and d_ff = 4, biases set to 0. The token vector x goes out to 4 hidden features " +
      "through W1, through ReLU, and back down to 2 numbers through W2. Watch the panel: it shows which hidden features switch on.",
    setupBlocks: (input) => [
      { type: "vector", key: "x", label: "x", values: input.x },
      { type: "matrix", label: "W1", rows: input.W1, rowLabels: ["row 1", "row 2"], colLabels: ["c1", "c2", "c3", "c4"] },
      { type: "matrix", label: "W2", rows: input.W2, rowLabels: ["row 1", "row 2", "row 3", "row 4"], colLabels: ["d1", "d2"] },
    ],
    panel: (el) => featurePanel(el),
    build(input) {
      const { x } = input;
      const r = M.ffn(x, input.W1, input.W2);
      const res = M.add(x, r.out);
      const on = onFeatures(r.hidden);
      const off = r.hidden.map((v, i) => i).filter((i) => !on.includes(i));
      const top = r.hidden.reduce((b, v, i) => (v > r.hidden[b] ? i : b), 0);
      const correct = featureLabel(on);
      const choices = Array.from(new Set([correct, "All four", featureLabel([top]), "None of them"])).slice(0, 3).sort();
      const cols = ["f1", "f2", "f3", "f4"];

      const xRow = { type: "vector", label: "x", values: x };
      const hRow = { type: "vector", key: "hidden", label: "x · W1", values: r.hidden, cols, hl: off };
      const aRow = { type: "vector", key: "activated", label: "h = ReLU", values: r.activated, cols, hl: on };
      const oRow = { type: "vector", key: "out", label: "FFN(x)", values: r.out, tone: "result" };

      return {
        steps: [
          {
            title: "Expand: x · W1",
            say:
              "Each hidden feature is x[0] times the top-row value plus x[1] times the bottom-row value of its column. " +
              "Two numbers become four.",
            blocks: [
              xRow,
              {
                type: "lines",
                fresh: true,
                lines: r.hidden.map((h, j) => ({
                  t: "column " + (j + 1) + ": " + num(x[0]) + "×" + par(input.W1[0][j]) + " + " + num(x[1]) + "×" + par(input.W1[1][j]) + " = " + num(h),
                  hl: h <= 0,
                })),
              },
              Object.assign({}, hRow, { fresh: true }),
            ],
          },
          {
            title: "ReLU: switch off the negatives",
            say:
              "max(0, ·) keeps positives and turns every negative into 0. " +
              (off.length ? featureLabel(off).replace("Only feature", "Feature") + " switched off." : "Nothing was negative, so nothing switched off."),
            predict: {
              ask: "After ReLU, which hidden features stay on?",
              choices,
              answer: choices.indexOf(correct),
              why:
                "ReLU keeps only the positive values. x · W1 was " + vstr(r.hidden) + ", so the features that stay on are " +
                on.map((i) => i + 1).join(", ") + ".",
              hint: "ReLU is max(0, value). A negative number, or zero, gives 0: that feature is off.",
            },
            blocks: [
              hRow,
              { type: "lines", lines: [{ t: "max(0, " + vstr(r.hidden) + ") = " + vstr(r.activated), hl: true }] },
              Object.assign({}, aRow, { fresh: true }),
            ],
          },
          {
            title: "Shrink: h · W2",
            say:
              "Each output dimension adds up h times one column of W2. A feature that is off contributes nothing, " +
              "so only the switched-on features write into the result.",
            blocks: [
              aRow,
              {
                type: "lines",
                fresh: true,
                lines: r.out.map((o, k) => {
                  const terms = r.activated.map((h, j) => num(h) + "×" + par(input.W2[j][k]));
                  const parts = r.activated.map((h, j) => h * input.W2[j][k]);
                  return {
                    t: "output dim " + (k + 1) + ": " + terms.join(" + ") + " = " + parts.map(num).join(" + ").replace(/\+ −/g, "− ") + " = " + num(o),
                    hl: true,
                  };
                }),
              },
              Object.assign({}, oRow, { fresh: true }),
            ],
          },
          {
            title: "Add the residual",
            say:
              "The FFN's output is added back onto x, exactly like the residual after attention. Then this goes to the norm " +
              "(or, in pre-norm, the norm was applied to x before step 1).",
            blocks: [
              xRow,
              Object.assign({}, oRow, { op: "+" }),
              { type: "vector", key: "residual", label: "x + FFN(x)", values: res, op: "=", tone: "result", fresh: true },
            ],
          },
        ],
        takeaway:
          x[1] < 0
            ? "Flipping one sign in x switched on a completely different set of features: " + featureLabel(on).toLowerCase() +
              " instead of 1, 2 and 4. That on and off switching is what lets the FFN respond differently to different kinds of tokens."
            : "Two numbers went out to four features, ReLU switched off feature " + off.map((i) => i + 1).join(", ") +
              ", and the features left on wrote their rows of W2 back into the result. Try the twist: change one sign in x.",
      };
    },
  };

  if (typeof module !== "undefined" && module.exports) module.exports = defs;
  if (typeof document === "undefined") return;

  /* ============================================================
     Browser side
     ============================================================ */
  const REDUCE = root.matchMedia && root.matchMedia("(prefers-reduced-motion: reduce)").matches;
  function ready(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else fn();
  }
  function setChips(fig, attr, value) {
    fig.querySelectorAll(".chip[" + attr + "]").forEach((c) => c.setAttribute("aria-pressed", String(c.getAttribute(attr) === String(value))));
  }

  /* ---------- walkthrough panel: the four hidden features ---------- */
  function featurePanel(el) {
    el.classList.add("l2p4-fpanel");
    return {
      render(index, step, input) {
        const r = M.ffn(input.x, input.W1, input.W2);
        const phase = index === 0 ? 0 : Math.min(index, 3); // 0 wait, 1 values, 2 on/off, 3+ write-back
        let s = '<p class="l2p4-fhead">The four hidden features</p><ol class="l2p4-feats">';
        r.hidden.forEach((h, j) => {
          const isOn = r.activated[j] > 0;
          const state = phase < 2 ? "is-wait" : isOn ? "is-on" : "is-off";
          const badge = phase < 2 ? (phase === 0 ? "waiting" : "value " + num(h)) : isOn ? "on" : "off";
          const row = input.W2[j];
          const write =
            phase >= 3
              ? isOn
                ? "writes " + num(r.activated[j]) + " × " + vstr(row) + " = " + vstr(row.map((w) => w * r.activated[j]))
                : "writes nothing"
              : "value row " + vstr(row);
          s +=
            `<li class="l2p4-feat ${state}"><span class="l2p4-fname">feature ${j + 1}</span>` +
            `<span class="l2p4-fbadge">${badge}</span>` +
            `<span class="l2p4-fval">${phase >= 1 ? "x·W1 = " + num(h) : "&nbsp;"}</span>` +
            `<span class="l2p4-fwrite">${write}</span></li>`;
        });
        s += "</ol>";
        if (phase >= 3) s += `<p class="l2p4-fsum">sum of what was written: ${vstr(r.out)}</p>`;
        el.innerHTML = s;
      },
    };
  }

  /* ---------- 4.1 the shape of the journey ---------- */
  const SHAPE_TEXT = [
    "The token arrives as a vector of <b>512</b> numbers (d_model in the original paper).",
    "<b>W1</b> expands it to <b>2048</b> numbers, four times bigger: lots of room for separate features.",
    "<b>ReLU</b> switches the negative features off. The dim blocks are features that are off for this token.",
    "<b>W2</b> shrinks the result back to <b>512</b>, so it can be added straight back onto the token.",
  ];
  function drawShape(stage) {
    const cols = [
      { x: 40, n: 8, label: "x", size: "512" },
      { x: 140, n: 32, label: "x·W1", size: "2048" },
      { x: 225, n: 32, label: "ReLU", size: "2048", relu: true },
      { x: 320, n: 8, label: "out", size: "512" },
    ];
    const cell = 7;
    const gap = 1;
    const w = 26;
    const mid = 152;
    const colTop = (c) => mid - (c.n * (cell + gap)) / 2;
    const colBottom = (c) => colTop(c) + c.n * (cell + gap) - gap;
    // A fixed pattern of switched-off features, just for the picture.
    const offPattern = (i) => (i * 7 + 3) % 5 < 2;
    let s = `<svg viewBox="0 0 360 305" class="l2p4-svg l2p4-shape" role="img" aria-label="512 numbers expand to 2048, pass through ReLU, and shrink back to 512">`;
    const funnel = (a, b, label, lit) => {
      const ax = a.x + w / 2;
      const bx = b.x - w / 2;
      s += `<polygon class="l2p4-funnel${lit ? " is-lit" : ""}" points="${ax},${colTop(a)} ${bx},${colTop(b)} ${bx},${colBottom(b)} ${ax},${colBottom(a)}"/>`;
      s += `<text class="l2p4-flabel${lit ? " is-lit" : ""}" x="${(ax + bx) / 2}" y="${mid + 4}" text-anchor="middle">${label}</text>`;
    };
    funnel(cols[0], cols[1], "W1", stage >= 1);
    s += `<line class="l2p4-link${stage >= 2 ? " is-lit" : ""}" x1="${cols[1].x + w / 2}" y1="${mid}" x2="${cols[2].x - w / 2}" y2="${mid}"/>`;
    funnel(cols[2], cols[3], "W2", stage >= 3);
    cols.forEach((c, ci) => {
      const lit = stage >= ci;
      s += `<g class="l2p4-col${lit ? " is-lit" : ""}${ci === stage ? " is-now" : ""}">`;
      for (let i = 0; i < c.n; i++) {
        const off = c.relu && offPattern(i);
        s += `<rect class="l2p4-cell${off ? " is-off" : ""}" x="${c.x - w / 2}" y="${colTop(c) + i * (cell + gap)}" width="${w}" height="${cell}" rx="1.5"/>`;
      }
      s += `<text class="l2p4-clabel" x="${c.x}" y="12" text-anchor="middle">${c.label}</text>`;
      s += `<text class="l2p4-csize" x="${c.x}" y="298" text-anchor="middle">${c.size}</text>`;
      s += "</g>";
    });
    s += "</svg>";
    return s;
  }
  function initShape() {
    const fig = document.getElementById("l2p4-shape");
    if (!fig) return;
    const stageEl = fig.querySelector(".ix-stage");
    const readout = fig.querySelector(".ix-readout");
    const play = fig.querySelector("[data-play]");
    let timer = 0;
    function show(stage) {
      stageEl.innerHTML = drawShape(stage);
      readout.innerHTML = SHAPE_TEXT[stage];
      setChips(fig, "data-stage", stage);
    }
    fig.querySelectorAll(".chip[data-stage]").forEach((c) =>
      c.addEventListener("click", () => {
        clearTimeout(timer);
        show(Number(c.getAttribute("data-stage")));
      })
    );
    play.addEventListener("click", () => {
      clearTimeout(timer);
      if (REDUCE) return show(3);
      let k = 0;
      const tick = () => {
        show(k);
        if (k++ < 3) timer = setTimeout(tick, 1400);
      };
      tick();
    });
    show(3);
  }

  /* ---------- 4.2 position-wise ---------- */
  function drawPos(mode, tokens, outs) {
    const xs = [60, 180, 300];
    let s = `<svg viewBox="0 0 360 250" class="l2p4-svg" role="img" aria-label="${
      mode === "attn" ? "Attention: every token is connected to every other token" : "FFN: each token goes through its own copy of the same FFN, with no links between tokens"
    }">`;
    s += `<defs><marker id="l2p4-ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" class="l2p4-ahead"/></marker></defs>`;
    tokens.forEach((t, i) => {
      s += `<text class="l2p4-tok" x="${xs[i]}" y="222" text-anchor="middle">token ${i + 1}</text>`;
      s += `<text class="l2p4-vec${i === 1 ? " is-edit" : ""}" x="${xs[i]}" y="240" text-anchor="middle">${vstr(t)}</text>`;
    });
    if (mode === "attn") {
      s += `<g class="l2p4-box is-attn"><rect x="20" y="110" width="320" height="40" rx="8"/><text x="180" y="134" text-anchor="middle">Attention: everyone shares notes</text></g>`;
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          s += `<path class="l2p4-mix" d="M${xs[i]} 205 C ${xs[i]} 175, ${xs[j]} 180, ${xs[j]} 154" marker-end="url(#l2p4-ah)"/>`;
        }
        s += `<text class="l2p4-vec is-faint" x="${xs[i]}" y="60" text-anchor="middle">a blend of all 3</text>`;
        s += `<line class="l2p4-up" x1="${xs[i]}" y1="108" x2="${xs[i]}" y2="72" marker-end="url(#l2p4-ah)"/>`;
      }
    } else {
      tokens.forEach((t, i) => {
        s += `<line class="l2p4-up" x1="${xs[i]}" y1="205" x2="${xs[i]}" y2="156" marker-end="url(#l2p4-ah)"/>`;
        s += `<g class="l2p4-box is-ffn"><rect x="${xs[i] - 52}" y="104" width="104" height="48" rx="8"/><text x="${xs[i]}" y="125" text-anchor="middle">FFN</text><text class="l2p4-sub" x="${xs[i]}" y="141" text-anchor="middle">same W1, W2</text></g>`;
        s += `<line class="l2p4-up" x1="${xs[i]}" y1="102" x2="${xs[i]}" y2="72" marker-end="url(#l2p4-ah)"/>`;
        s += `<text class="l2p4-vec is-out${i === 1 ? " is-edit" : ""}" x="${xs[i]}" y="60" text-anchor="middle">${vstr(outs[i])}</text>`;
      });
      s += `<text class="l2p4-tok" x="180" y="30" text-anchor="middle">outputs, one per token</text>`;
    }
    s += "</svg>";
    return s;
  }
  function initPos() {
    const fig = document.getElementById("l2p4-pos");
    if (!fig) return;
    const vIn = fig.querySelector("#l2p4-pos-v");
    const stage = fig.querySelector(".ix-stage");
    const readout = fig.querySelector(".ix-readout");
    let mode = "ffn";
    function update() {
      const v = Number(vIn.value);
      fig.querySelector("output[for='l2p4-pos-v']").textContent = num(v);
      const tokens = [[1, 2], [1, v], [2, 1]];
      const outs = tokens.map((t) => M.ffn(t, W1, W2).out);
      stage.innerHTML = drawPos(mode, tokens, outs);
      setChips(fig, "data-mode", mode);
      readout.innerHTML =
        mode === "ffn"
          ? `Token 2 goes in as <span class="math">${vstr(tokens[1])}</span> and comes out as <span class="math">${vstr(outs[1])}</span>. ` +
            `Move the slider: tokens 1 and 3 stay at <span class="math">${vstr(outs[0])}</span> and <span class="math">${vstr(outs[2])}</span>, ` +
            "because inside the FFN no token ever sees another. It is each person back at their desk, thinking over what they heard."
          : "In attention, every token reads every other token, so changing token 2 would change what all three receive. " +
            "This is the team meeting where everyone shares notes. It mixes information <b>between</b> tokens.";
    }
    fig.querySelectorAll(".chip[data-mode]").forEach((c) =>
      c.addEventListener("click", () => {
        mode = c.getAttribute("data-mode");
        update();
      })
    );
    vIn.addEventListener("input", update);
    update();
  }

  /* ---------- 4.4 parameter share ---------- */
  const D_OPTIONS = [128, 256, 512, 768, 1024, 2048, 4096, 8192];
  function initParams() {
    const fig = document.getElementById("l2p4-params");
    if (!fig) return;
    const dIn = fig.querySelector("#l2p4-dmodel");
    const stage = fig.querySelector(".ix-stage");
    const readout = fig.querySelector(".ix-readout");
    function update() {
      const d = D_OPTIONS[Number(dIn.value)];
      const dff = 4 * d;
      const a = attnParams(d);
      const f = ffnParams(d, dff);
      const share = f / (a + f);
      fig.querySelector("output[for='l2p4-dmodel']").textContent = d + " (d_ff = " + dff + ")";
      stage.innerHTML =
        `<div class="l2p4-pbar" role="img" aria-label="Attention ${M.fmtInt(a)} weights, FFN ${M.fmtInt(f)} weights">` +
        `<span class="l2p4-pseg is-attn" style="flex-grow:${a}"><span>attention</span></span>` +
        `<span class="l2p4-pseg is-ffn" style="flex-grow:${f}"><span>FFN</span></span></div>` +
        `<dl class="l2p4-pnums"><div><dt>attention (W_Q, W_K, W_V, W_O)</dt><dd class="math">4 × ${d} × ${d} = ${M.fmtInt(a)}</dd></div>` +
        `<div><dt>FFN (W1 and W2)</dt><dd class="math">2 × ${d} × ${dff} = ${M.fmtInt(f)}</dd></div></dl>`;
      readout.innerHTML =
        `The FFN has <b>${num(f / a)} times</b> as many weights as attention, so it holds <b>${M.fmt(share * 100, 1)}%</b> of this layer: about two thirds. ` +
        "Slide d_model anywhere and the ratio never moves, because 2 × d × 4d = 8d² and 4 × d × d = 4d².";
    }
    dIn.addEventListener("input", update);
    update();
  }

  /* ---------- 4.6 activation curves ---------- */
  const CURVES = [
    { name: "ReLU", fn: relu1, cls: "is-relu" },
    { name: "GELU", fn: gelu, cls: "is-gelu" },
    { name: "Swish", fn: swish, cls: "is-swish" },
  ];
  function drawCurves(at) {
    const W = 340;
    const H = 240;
    const sx = (x) => 30 + ((x + 4) / 8) * (W - 44);
    const sy = (y) => 16 + ((3.2 - y) / 4.4) * (H - 44);
    let s = `<svg viewBox="0 0 ${W} ${H}" class="l2p4-svg" role="img" aria-label="ReLU, GELU and Swish curves from x = −4 to 4">`;
    for (let g = -1; g <= 3; g++) s += `<line class="l2p4-grid" x1="${sx(-4)}" y1="${sy(g)}" x2="${sx(4)}" y2="${sy(g)}"/><text class="l2p4-tick" x="${sx(-4) - 6}" y="${sy(g) + 4}" text-anchor="end">${num(g)}</text>`;
    for (let g = -4; g <= 4; g += 2) s += `<text class="l2p4-tick" x="${sx(g)}" y="${sy(-1.2) + 14}" text-anchor="middle">${num(g)}</text>`;
    s += `<line class="l2p4-axis" x1="${sx(-4)}" y1="${sy(0)}" x2="${sx(4)}" y2="${sy(0)}"/><line class="l2p4-axis" x1="${sx(0)}" y1="${sy(-1.2)}" x2="${sx(0)}" y2="${sy(3.2)}"/>`;
    CURVES.forEach((c) => {
      const pts = [];
      for (let i = 0; i <= 160; i++) {
        const x = -4 + (i / 160) * 7.2; // stop at 3.2 so the curves stay inside the plot
        pts.push(sx(x).toFixed(1) + "," + sy(c.fn(x)).toFixed(1));
      }
      s += `<polyline class="l2p4-curve ${c.cls}" points="${pts.join(" ")}"/>`;
    });
    s += `<line class="l2p4-probe" x1="${sx(at)}" y1="${sy(-1.2)}" x2="${sx(at)}" y2="${sy(3.2)}"/>`;
    CURVES.forEach((c) => {
      const y = c.fn(at);
      if (y <= 3.2) s += `<circle class="l2p4-dot ${c.cls}" cx="${sx(at)}" cy="${sy(y)}" r="4.5"/>`;
    });
    s += "</svg>";
    return s;
  }
  function initCurves() {
    const fig = document.getElementById("l2p4-curves");
    if (!fig) return;
    const xIn = fig.querySelector("#l2p4-x");
    const stage = fig.querySelector(".ix-stage");
    const readout = fig.querySelector(".ix-readout");
    function update() {
      const x = Number(xIn.value);
      fig.querySelector("output[for='l2p4-x']").textContent = M.fmt(x, 1);
      stage.innerHTML = drawCurves(x);
      readout.innerHTML =
        CURVES.map((c) => `<span class="math">${c.name}(${M.fmt(x, 1)}) = ${M.fmt(c.fn(x), 3)}</span>`).join(" · ") +
        "<br>" +
        (x < 0
          ? "For a small negative input ReLU gives exactly 0, while GELU and Swish let a little through: a gentle dip instead of a hard cut."
          : "For positive inputs all three behave much alike, and for large values they almost match.");
    }
    xIn.addEventListener("input", update);
    update();
  }

  /* ---------- 4.6 the SwiGLU gate ---------- */
  const CONTENT = 2;
  function drawKnob(level) {
    // The needle turns from −135° (gate shut) towards +135° (gate at 4).
    const clamped = Math.max(-0.3, Math.min(4, level));
    const angle = -135 + (clamped / 4) * 270;
    const a = ((angle - 90) * Math.PI) / 180;
    const cx = 70;
    const cy = 70;
    const r = 50;
    const arc = (from, to) => {
      const p = (deg) => {
        const t = ((deg - 90) * Math.PI) / 180;
        return (cx + r * Math.cos(t)).toFixed(1) + " " + (cy + r * Math.sin(t)).toFixed(1);
      };
      return `M${p(from)} A${r} ${r} 0 ${to - from > 180 ? 1 : 0} 1 ${p(to)}`;
    };
    let s = `<svg viewBox="0 0 140 140" class="l2p4-svg l2p4-knob" role="img" aria-label="Volume knob turned to ${M.fmt(level, 3)}">`;
    s += `<path class="l2p4-ktrack" d="${arc(-135, 135)}"/>`;
    if (angle > -135) s += `<path class="l2p4-kfill" d="${arc(-135, angle)}"/>`;
    s += `<circle class="l2p4-kbody" cx="${cx}" cy="${cy}" r="34"/>`;
    s += `<line class="l2p4-kneedle" x1="${cx}" y1="${cy}" x2="${(cx + 30 * Math.cos(a)).toFixed(1)}" y2="${(cy + 30 * Math.sin(a)).toFixed(1)}"/>`;
    s += `<text class="l2p4-ktext" x="${cx}" y="128" text-anchor="middle">gate ${M.fmt(level, 2)}</text>`;
    s += "</svg>";
    return s;
  }
  function initGate() {
    const fig = document.getElementById("l2p4-gate");
    if (!fig) return;
    const gIn = fig.querySelector("#l2p4-g");
    const knob = fig.querySelector(".l2p4-knobwrap");
    const readout = fig.querySelector(".ix-readout");
    const meters = fig.querySelector(".l2p4-meters");
    function update() {
      const g = Number(gIn.value);
      fig.querySelector("output[for='l2p4-g']").textContent = M.fmt(g, 1);
      const knobLevel = swish(g);
      const out = knobLevel * CONTENT;
      const reluOut = relu1(g) > 0 ? CONTENT : 0;
      knob.innerHTML = drawKnob(knobLevel);
      const bar = (label, v, cls) =>
        `<div class="l2p4-meter ${cls}"><span class="l2p4-mlabel">${label}</span><span class="l2p4-mtrack"><span class="l2p4-mfill" style="width:${Math.max(0, Math.min(1, v / 8)) * 100}%"></span></span><span class="l2p4-mval math">${M.fmt(v, 3)}</span></div>`;
      meters.innerHTML =
        bar("ReLU switch passes", reluOut, "is-relu") + bar("SwiGLU knob passes", out, "is-swiglu");
      readout.innerHTML =
        `Content x·V for this feature = <span class="math">${num(CONTENT)}</span>. Gate input x·W = <span class="math">${M.fmt(g, 1)}</span>, ` +
        `so Swish(${M.fmt(g, 1)}) = <span class="math">${M.fmt(knobLevel, 3)}</span> and the feature passes ` +
        `<span class="math">${M.fmt(knobLevel, 3)} × ${num(CONTENT)} = ${M.fmt(out, 3)}</span>. ` +
        (g > 0 ? "A switch would pass it all or nothing; the knob chooses how much." : "A switch would be fully off here; the knob still lets a trace through.");
    }
    gIn.addEventListener("input", update);
    update();
    const budget = fig.querySelector("[data-budget]");
    if (budget) {
      const d = 512;
      const dff = (8 / 3) * d;
      budget.innerHTML =
        `With d_model = ${d}: a plain FFN uses <span class="math">2 × ${d} × ${4 * d} = ${M.fmtInt(ffnParams(d, 4 * d))}</span> weights. ` +
        `SwiGLU has three matrices, so d_ff shrinks to <span class="math">(8/3) × ${d} ≈ ${M.fmt(dff, 2)}</span>, and ` +
        `<span class="math">3 × ${d} × ${M.fmt(dff, 2)} = ${M.fmtInt(swigluParams(d, dff))}</span>: the same budget.`;
    }
  }

  /* ---------- 4.6 Mixture of Experts router ---------- */
  const SENTENCE = ["The", "cat", "sat", "on", "the", "warm", "mat"];
  const EXPERTS = 8;
  const TOPK = 2;
  function initMoe() {
    const fig = document.getElementById("l2p4-moe");
    if (!fig) return;
    const chips = fig.querySelector(".l2p4-tokens");
    const grid = fig.querySelector(".l2p4-experts");
    const readout = fig.querySelector(".ix-readout");
    SENTENCE.forEach((t, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "chip";
      b.textContent = t;
      b.setAttribute("data-tok", String(i));
      b.addEventListener("click", () => show(i));
      chips.append(b);
    });
    function show(i) {
      setChips(fig, "data-tok", i);
      const r = route(SENTENCE[i] + "@" + i, EXPERTS, TOPK);
      const max = Math.max.apply(null, r.probs);
      grid.innerHTML = r.probs
        .map((p, e) => {
          const chosen = r.top.includes(e);
          return (
            `<li class="l2p4-expert${chosen ? " is-chosen" : ""}"><span class="l2p4-ename">expert ${e + 1}</span>` +
            `<span class="l2p4-etrack"><span class="l2p4-efill" style="width:${(p / max) * 100}%"></span></span>` +
            `<span class="l2p4-eval math">${M.fmt(p, 2)}</span><span class="l2p4-estate">${chosen ? "runs" : "idle"}</span></li>`
          );
        })
        .join("");
      const per = ffnParams(512, 2048);
      const total = EXPERTS * per;
      const active = TOPK * per;
      readout.innerHTML =
        `The router sends <b>"${SENTENCE[i]}"</b> to experts <b>${r.top.map((e) => e + 1).join(" and ")}</b>, its top ${TOPK} scores. ` +
        `If each expert is the FFN from 4.4 (${M.fmtInt(per)} weights), this layer holds <span class="math">${EXPERTS} × ${M.fmtInt(per)} = ${M.fmtInt(total)}</span> weights, ` +
        `but this token only runs through <span class="math">${TOPK} × ${M.fmtInt(per)} = ${M.fmtInt(active)}</span> of them, ` +
        `<b>${M.fmt((active / total) * 100, 0)}%</b>. Lots of knowledge, small compute per token. Click another token: it can pick different experts.`;
    }
    show(1);
  }

  ready(() => {
    initShape();
    initPos();
    initParams();
    initCurves();
    initGate();
    initMoe();
  });
})(typeof window !== "undefined" ? window : globalThis);
