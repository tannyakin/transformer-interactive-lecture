// ============================================================
// Lecture 2, Part 3: Add & Norm.
// Walkthroughs: residual connection, layer norm, RMSNorm.
// Interactives: the block diagram (post-norm / pre-norm),
// the gradient highway and the residual stream.
// Every number is computed from the input, never typed.
// ============================================================
(function (root) {
  "use strict";

  const L = (root.Lecture = root.Lecture || {});
  const M = L.math;
  const defs = (L.defs = L.defs || {});
  const MINUS = "−";

  /* ---------------- formatting helpers ---------------- */
  // Fewest decimals (up to 3) that show every value exactly.
  function dpFor(xs) {
    for (let dp = 0; dp <= 3; dp++) {
      if (xs.every((v) => Math.abs(M.round(v, dp) - M.round(v, 3)) < 1e-9)) return dp;
    }
    return 3;
  }
  const num = (v) => M.fmt(v, dpFor([v]));
  const vstr = (xs, dp) => {
    const d = dp === undefined ? dpFor(xs) : dp;
    return "[" + xs.map((v) => M.fmt(v, d)).join(", ") + "]";
  };
  // Four significant figures, the way the notes write 2.236, 22.36 and 5.477.
  const sig4dp = (v) => Math.max(0, 3 - Math.floor(Math.log10(Math.abs(v) || 1)));
  const sig4 = (v) => M.fmt(v, sig4dp(v));
  const signed = (v) => (v < 0 ? MINUS + " " + num(Math.abs(v)) : "+ " + num(v));

  // Two significant figures, like 0.5^10 = 0.00098. Very small values use powers of ten.
  function fmtSmall(v) {
    if (v >= 1) return num(v);
    if (v === 0) return "0";
    const p = Number(v.toPrecision(2));
    if (p >= 1e-4) return String(p);
    const parts = p.toExponential(1).split("e");
    const e = Number(parts[1]);
    return parts[0] + " × 10<sup>" + (e < 0 ? MINUS + Math.abs(e) : e) + "</sup>";
  }

  const gradPlain = (deriv, layers) => Math.pow(deriv, layers);
  // Along the shortcut, every layer multiplies the signal by the "1" in 1 + F'(x).
  const gradShortcut = (layers) => Math.pow(1, layers);

  /* ---------------- the block diagram as data ---------------- */
  // y runs top to bottom in SVG space; the block is drawn bottom to top.
  function blockModel(mode) {
    if (mode === "pre") {
      return {
        mode,
        formula: "output = x + F( LayerNorm(x) )",
        hw: [[520, 371], [349, 161], [139, 91], [59, 34]],
        adds: [360, 150],
        norms: [{ y: 75, h: 32, onHighway: true, final: true }],
        branches: [
          { split: 505, join: 360, boxes: [{ y: 470, h: 32, label: "LayerNorm", kind: "norm" }, { y: 405, h: 40, label: "Multi-Head Attention", kind: "attn" }] },
          { split: 330, join: 150, boxes: [{ y: 284, h: 32, label: "LayerNorm", kind: "norm" }, { y: 220, h: 40, label: "Feedforward Network", kind: "ffn" }] },
        ],
      };
    }
    return {
      mode: "post",
      formula: "output = LayerNorm( x + F(x) )",
      hw: [[520, 371], [349, 322], [290, 161], [139, 112], [80, 34]],
      adds: [360, 150],
      norms: [{ y: 306, h: 32, onHighway: true, final: false }, { y: 96, h: 32, onHighway: true, final: false }],
      branches: [
        { split: 490, join: 360, boxes: [{ y: 430, h: 40, label: "Multi-Head Attention", kind: "attn" }] },
        { split: 260, join: 150, boxes: [{ y: 210, h: 40, label: "Feedforward Network", kind: "ffn" }] },
      ],
    };
  }
  // How many norms sit on the shortcut inside the blocks (the final norm is outside every block).
  const normsOnHighway = (mode) => blockModel(mode).norms.filter((n) => n.onHighway && !n.final).length;

  L.l2p3 = { gradPlain, gradShortcut, fmtSmall, blockModel, normsOnHighway, sig4 };

  /* ---------------- 3.2 Residual connection ---------------- */
  defs["l2-residual"] = {
    id: "l2-residual",
    title: "A residual connection, by hand",
    input: { x: [2, 1], fx: [0.3, -0.2] },
    setup:
      "One token's vector is x = [2, 1]. The attention layer looks at the context and produces " +
      "F(x) = [0.3, " + MINUS + "0.2]. There are two ways to use that result: replace x with it, or add it on top of x. " +
      "Watch what each choice does to the token.",
    setupBlocks: (input) => [
      { type: "vector", key: "x", label: "x", values: input.x, cols: ["dim 1", "dim 2"] },
      { type: "vector", key: "fx", label: "F(x)", values: input.fx },
    ],
    panel: (el) => residualPanel(el),
    build(input) {
      const { x, fx } = input;
      const out = M.add(x, fx);
      const slip = x.map((v, i) => v + Math.abs(fx[i]));
      const choices = [vstr(fx), vstr(out), vstr(slip)];
      const xRow = { type: "vector", label: "x", values: x, cols: ["dim 1", "dim 2"] };
      const fRow = { type: "vector", label: "F(x)", values: fx };
      const withoutRow = { type: "vector", key: "without", label: "no residual", values: fx.slice(), tone: "muted" };
      const withRow = { type: "vector", key: "with", label: "x + F(x)", values: out, tone: "result" };
      return {
        steps: [
          {
            title: "Option 1: let the layer replace x",
            say:
              "Without a residual, the layer's output simply is F(x). The next layer receives " + vstr(fx) +
              " and nothing else. The token's original information, " + vstr(x) + ", is gone.",
            blocks: [
              Object.assign({}, xRow, { tone: "muted" }),
              Object.assign({}, withoutRow, { op: "=", tone: undefined, fresh: true }),
              { type: "note", tone: "warn", html: "output = F(x) = " + vstr(fx) + ". The original " + vstr(x) + " is lost." },
            ],
          },
          {
            title: "Option 2: add F(x) on top of x",
            say:
              "With a residual connection, x takes a free shortcut around the layer and the layer's result is added to it, " +
              "one dimension at a time.",
            predict: {
              ask: "With the shortcut, what comes out of the layer?",
              choices,
              answer: 1,
              why:
                "Add position by position: " + num(x[0]) + " " + signed(fx[0]) + " = " + num(out[0]) + " and " +
                num(x[1]) + " " + signed(fx[1]) + " = " + num(out[1]) + ". Watch the sign on the second number.",
              hint: "Add x and F(x) one position at a time, and keep the minus sign on the second value.",
            },
            blocks: [
              xRow,
              Object.assign({}, fRow, { op: "+" }),
              Object.assign({}, withRow, { op: "=", fresh: true }),
              { type: "lines", lines: [{ t: "[" + x.map((v, i) => num(v) + " " + signed(fx[i])).join(", ") + "] = " + vstr(out), hl: true }] },
            ],
          },
          {
            title: "Put the two options side by side",
            say:
              "With the residual, the token is still close to where it started, " + vstr(x) +
              ". It keeps its identity and gains a small, context-based adjustment.",
            blocks: [xRow, withoutRow, withRow],
          },
          {
            title: "What the layer actually had to learn",
            say:
              "The shortcut carried " + vstr(x) + " through for free. The layer only had to learn the change, " + vstr(fx) +
              ", not rebuild the whole vector from scratch. A small correction is a much easier thing to learn.",
            blocks: [
              {
                type: "lines",
                lines: [
                  { t: "shortcut carries   x    = " + vstr(x) },
                  { t: "layer learns       F(x) = " + vstr(fx), hl: true },
                  { t: "output             x + F(x) = " + vstr(out) },
                ],
              },
              withRow,
            ],
          },
        ],
        takeaway:
          "output = x + F(x) means a layer never has to reproduce its input. It only writes a correction on top. " +
          "The token keeps its identity, and a deep stack of layers becomes a stack of small, safe edits.",
      };
    },
  };

  /* ---------------- 3.3 Layer norm ---------------- */
  const LN_X = [2, 4, 6, 8];
  const listSum = (xs) => xs.map(num).join(" + ");

  defs["l2-layernorm"] = {
    id: "l2-layernorm",
    title: "Layer norm, by hand",
    input: { x: LN_X, gamma: 1, beta: 0 },
    twist: {
      label: "Multiply the input by 10",
      offLabel: "Back to [2, 4, 6, 8]",
      input: { x: LN_X.map((v) => v * 10), factor: 10, base: LN_X },
    },
    setup:
      "One token's vector has d = 4 values. Layer norm will rescale it so the values have mean 0 and spread 1, " +
      "then hand the result to γ and β. Keep a calculator handy and check each line.",
    setupBlocks: (input) => [{ type: "vector", key: "x", label: "x", values: input.x, cols: ["x₁", "x₂", "x₃", "x₄"] }],
    build(input) {
      const { x, gamma, beta, factor, base } = input;
      const d = x.length;
      const ln = M.layerNorm(x); // ε = 0.00001, as in the notes
      const sum = x.reduce((a, b) => a + b, 0);
      const squares = ln.centered.map((c) => c * c);
      const sumSq = squares.reduce((a, b) => a + b, 0);
      const y = ln.normalized.map((v) => gamma * v + beta);
      const before = base ? M.layerNorm(base).normalized : null;
      const sdp = sig4dp(ln.std);

      const xRow = { type: "vector", label: "x", values: x, cols: ["x₁", "x₂", "x₃", "x₄"] };
      const cRow = { type: "vector", key: "centered", label: "x − μ", values: ln.centered };
      const nRow = { type: "vector", key: "normalized", label: "x̂", values: ln.normalized, dp: 3, tone: "result" };

      const spreadChoices = [ln.std, ln.variance, sumSq].sort((a, b) => a - b);
      const spreadStr = (v) => (v === ln.std ? sig4(v) : num(v));

      const normalizePredict = before
        ? {
            ask: "The input is now 10 times bigger. What will the normalized values be?",
            choices: [
              "10 times smaller: " + vstr(before.map((v) => v / factor), 4),
              "Exactly the same as before: " + vstr(before, 3),
              "10 times bigger: " + vstr(before.map((v) => v * factor), 2),
            ],
            answer: 1,
            why:
              "The mean grew 10 times and so did σ. Dividing a 10 times bigger deviation by a 10 times bigger σ cancels the 10 exactly.",
            hint: "Both the top (x − μ) and the bottom (σ) of the division grew by the same factor. What happens to the ratio?",
          }
        : undefined;

      const steps = [
        {
          title: "Find the mean, μ",
          say: "Add the " + d + " values and divide by " + d + ". This is the centre the values will be shifted around.",
          blocks: [
            xRow,
            { type: "lines", fresh: true, lines: [{ t: "μ = (" + listSum(x) + ") / " + d + " = " + num(sum) + " / " + d + " = " + num(ln.mean), hl: true }] },
            { type: "vector", key: "mean", label: "μ", values: [ln.mean], fresh: true },
          ],
        },
        {
          title: "Subtract the mean from every value",
          say:
            "Now the values sit around zero: two below the mean and two above, in the same symmetric pattern as before.",
          blocks: [
            xRow,
            {
              type: "lines",
              fresh: true,
              lines: [{ t: "[" + x.map((v) => num(v) + " − " + num(ln.mean)).join(", ") + "] = " + vstr(ln.centered), hl: true }],
            },
            Object.assign({}, cRow, { fresh: true }),
          ],
        },
        {
          title: "Measure the spread: variance and σ",
          say:
            "Square each centred value, average the squares to get the variance σ², then take the square root to get σ. " +
            "The tiny ε = 0.00001 under the root only guards against dividing by zero; it does not change these digits.",
          predict: {
            ask: "What is the variance, σ²?",
            choices: spreadChoices.map(spreadStr),
            answer: spreadChoices.indexOf(ln.variance),
            why:
              "The squares add up to " + num(sumSq) + ", and the variance is their average: " + num(sumSq) + " / " + d + " = " +
              num(ln.variance) + ". Its square root, " + sig4(ln.std) + ", is σ, the next line.",
            hint: "The variance is the average of the squared deviations, not their sum and not its square root.",
          },
          blocks: [
            cRow,
            { type: "vector", key: "squares", label: "(x − μ)²", values: squares, fresh: true },
            {
              type: "lines",
              lines: [
                { t: "σ² = (" + listSum(squares) + ") / " + d + " = " + num(sumSq) + " / " + d + " = " + num(ln.variance) },
                { t: "σ  = √" + num(ln.variance) + " = " + sig4(ln.std), hl: true },
              ],
            },
            { type: "vector", key: "variance", label: "σ²", values: [ln.variance] },
            { type: "vector", key: "std", label: "σ", values: [ln.std], dp: sdp },
          ],
        },
        {
          title: "Divide every centred value by σ",
          say:
            "Each value becomes its distance from the mean, measured in units of σ. This is x̂: mean 0, spread 1." +
            (before ? " Compare it with the result before the input was multiplied by 10." : ""),
          predict: normalizePredict,
          blocks: [
            cRow,
            {
              type: "lines",
              fresh: true,
              lines: ln.centered.map((c, i) => ({ t: num(c) + " / " + sig4(ln.std) + " = " + M.fmt(ln.normalized[i], 3) })),
            },
            Object.assign({}, nRow, { fresh: true }),
          ].concat(before ? [{ type: "vector", key: "before", label: "before × " + factor, values: before, dp: 3, tone: "muted" }] : []),
        },
        {
          title: "Scale and shift with γ and β",
          say:
            "At the start of training γ = " + num(gamma) + " and β = " + num(beta) + ", so y = x̂. As training goes on, " +
            "the model can learn different γ and β for each dimension and partly undo the normalization if that helps.",
          blocks: [
            nRow,
            { type: "lines", lines: [{ t: "y = γ · x̂ + β = " + num(gamma) + " · x̂ + " + num(beta), hl: true }] },
            { type: "vector", key: "y", label: "y", values: y, dp: 3, tone: "result", fresh: true },
          ],
        },
      ];
      if (!normalizePredict) delete steps[3].predict;

      return {
        steps,
        takeaway: before
          ? "Layer norm does not care about the overall size of the vector, only about the pattern of its values. " +
            "That is called scale invariance, and it is precisely why it stabilises training: no matter how large things drift, " +
            "every layer receives inputs of a predictable size."
          : "Whatever values the token arrived with, it leaves with mean 0 and spread 1: " + vstr(y, 3) +
            ". Now press the twist and multiply the input by 10. Before you look, guess what happens to this answer.",
      };
    },
  };

  /* ---------------- 3.4 RMSNorm ---------------- */
  defs["l2-rmsnorm"] = {
    id: "l2-rmsnorm",
    title: "RMSNorm, by hand",
    input: { x: LN_X },
    twist: {
      label: "Try x × 10 yourself",
      offLabel: "Back to [2, 4, 6, 8]",
      input: { x: LN_X.map((v) => v * 10), factor: 10, base: LN_X },
    },
    setup:
      "Same vector as the layer norm example, x = [2, 4, 6, 8]. RMSNorm asks: do we really need to subtract the mean? " +
      "It skips the mean and the β shift, and divides by one number, the root mean square.",
    setupBlocks: (input) => [{ type: "vector", key: "x", label: "x", values: input.x, cols: ["x₁", "x₂", "x₃", "x₄"] }],
    build(input) {
      const { x, factor, base } = input;
      const d = x.length;
      const r = M.rmsNorm(x);
      const sumSq = r.squares.reduce((a, b) => a + b, 0);
      const lnOut = M.layerNorm(x).normalized;
      const before = base ? M.rmsNorm(base).normalized : null;
      const allPositive = r.normalized.every((v) => v > 0);

      const xRow = { type: "vector", label: "x", values: x, cols: ["x₁", "x₂", "x₃", "x₄"] };
      const sqRow = { type: "vector", key: "squares", label: "x²", values: r.squares };
      const outRow = { type: "vector", key: "normalized", label: "x / RMS", values: r.normalized, dp: 3, tone: "result" };

      return {
        steps: [
          {
            title: "Square every value",
            say: "No mean, no centring. RMSNorm goes straight to the squares of the raw values.",
            blocks: [
              xRow,
              { type: "lines", fresh: true, lines: x.map((v, i) => ({ t: num(v) + "² = " + num(r.squares[i]) })) },
              Object.assign({}, sqRow, { fresh: true }),
            ],
          },
          {
            title: "Average the squares",
            say: "Add the squares and divide by d = " + d + ". This is the mean of squares, the M in RMS.",
            blocks: [
              sqRow,
              { type: "lines", lines: [{ t: "mean of squares = " + num(sumSq) + " / " + d + " = " + num(r.meanSquare), hl: true }] },
              { type: "vector", key: "meanSquare", label: "mean of x²", values: [r.meanSquare], fresh: true },
            ],
          },
          {
            title: "Take the square root: the RMS",
            say:
              "The root of the mean of the squares, the root mean square. In a real model a tiny ε sits under the root to " +
              "avoid dividing by zero, just like in layer norm.",
            blocks: [
              { type: "lines", lines: [{ t: "RMS = √" + num(r.meanSquare) + " = " + sig4(r.rms), hl: true }] },
              { type: "vector", key: "rms", label: "RMS", values: [r.rms], dp: sig4dp(r.rms), fresh: true },
            ],
          },
          {
            title: "Divide every value by the RMS",
            say:
              "Each value is divided by the same number, " + sig4(r.rms) + ". With γ = 1 at the start of training, this is the output." +
              (before ? " Compare it with the result before multiplying by " + factor + "." : ""),
            predict: {
              ask: "Layer norm's output was centred on zero. What about RMSNorm's?",
              choices: ["Centred on zero, like layer norm", "All positive, not centred on zero"],
              answer: allPositive ? 1 : 0,
              why:
                "Nothing was subtracted, so every positive input stays positive after dividing by a positive RMS. " +
                "The values are brought to a predictable scale, but they are not centred.",
              hint: "Which step of layer norm moved values below zero? Does RMSNorm have that step?",
            },
            blocks: [
              xRow,
              { type: "lines", fresh: true, lines: x.map((v, i) => ({ t: num(v) + " / " + sig4(r.rms) + " = " + M.fmt(r.normalized[i], 3) })) },
              Object.assign({}, outRow, { fresh: true }),
            ].concat(before ? [{ type: "vector", key: "before", label: "before × " + factor, values: before, dp: 3, tone: "muted" }] : []),
          },
          {
            title: "Side by side with layer norm",
            say:
              "Layer norm needed a pass for the mean and another for the variance. RMSNorm needed one pass over the data. " +
              "At the scale of large models that saving adds up.",
            blocks: [
              outRow,
              { type: "vector", key: "layernorm", label: "layer norm", values: lnOut, dp: 3, tone: "muted" },
            ],
          },
        ],
        takeaway: before
          ? "Multiplying the input by " + factor + " multiplied the RMS by " + factor + " too, so the output did not move: " +
            vstr(r.normalized, 3) + ". RMSNorm is scale invariant, just like layer norm, and cheaper."
          : "RMSNorm gives " + vstr(r.normalized, 3) + ": not centred like layer norm's output, but at a predictable scale, " +
            "and one pass cheaper. Llama, Mistral, Gemma, Qwen and T5 all use RMSNorm or something very close to it.",
      };
    },
  };

  if (typeof module !== "undefined" && module.exports) module.exports = defs;
  if (typeof document === "undefined") return;

  /* ============================================================
     Browser side
     ============================================================ */
  const REDUCE = root.matchMedia && root.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- residual walkthrough panel: a 2D picture ---------- */
  function residualPanel(el) {
    el.classList.add("l2p3-rpanel");
    const W = 320;
    const H = 230;
    const sx = (v) => 34 + (v + 0.5) * 80;
    const sy = (v) => 196 - (v + 0.8) * 72;
    return {
      render(index, step, input) {
        const { x, fx } = input;
        const out = M.add(x, fx);
        const last = 5;
        const showWithout = index === 1 || index === 3 || index === last;
        const showWith = index >= 2;
        const lostX = index === 1;
        const pt = (v, cls, label, dx, dy, anchor) =>
          `<circle class="${cls}" cx="${sx(v[0])}" cy="${sy(v[1])}" r="5"/>` +
          `<text class="l2p3-rlabel" x="${sx(v[0]) + (dx || 0)}" y="${sy(v[1]) + (dy || 0)}" text-anchor="${anchor || "start"}">${label}</text>`;
        let s = `<svg viewBox="0 0 ${W} ${H}" class="l2p3-svg" role="img" aria-label="The token x, the layer output F(x), and the results with and without a residual, drawn as points on a plane">`;
        s += `<defs><marker id="l2p3-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" class="l2p3-ahead"/></marker></defs>`;
        s += `<line class="l2p3-axis" x1="${sx(-0.5)}" y1="${sy(0)}" x2="${sx(2.9)}" y2="${sy(0)}"/>`;
        s += `<line class="l2p3-axis" x1="${sx(0)}" y1="${sy(-0.8)}" x2="${sx(0)}" y2="${sy(1.6)}"/>`;
        s += `<text class="l2p3-rlabel is-faint" x="${sx(2.9) - 40}" y="${sy(0) + 16}">dim 1</text>`;
        s += `<text class="l2p3-rlabel is-faint" x="${sx(0) + 6}" y="${sy(1.6) + 10}">dim 2</text>`;
        // F(x) as a small arrow from the origin: the correction on its own.
        s += `<line class="l2p3-rarrow is-fx" x1="${sx(0)}" y1="${sy(0)}" x2="${sx(fx[0])}" y2="${sy(fx[1])}" marker-end="url(#l2p3-ah)"/>`;
        s += pt(x, "l2p3-rpt is-x" + (lostX ? " is-lost" : ""), "x = " + vstr(x) + (lostX ? " (lost)" : ""), -8, -12, "end");
        if (showWithout) s += pt(fx, "l2p3-rpt is-without", "no residual " + vstr(fx), 8, 20);
        else s += `<text class="l2p3-rlabel is-faint" x="${sx(fx[0]) + 8}" y="${sy(fx[1]) + 20}">F(x)</text>`;
        if (showWith) {
          s += `<line class="l2p3-rarrow is-with" x1="${sx(x[0])}" y1="${sy(x[1])}" x2="${sx(out[0])}" y2="${sy(out[1])}" marker-end="url(#l2p3-ah)"/>`;
          s += pt(out, "l2p3-rpt is-with", "x + F(x) = " + vstr(out), 4, 26, "end");
        }
        s += "</svg>";
        el.innerHTML = s;
      },
    };
  }

  /* ---------- helpers ---------- */
  function ready(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else fn();
  }
  function setChips(fig, attr, value) {
    fig.querySelectorAll(".chip[" + attr + "]").forEach((c) => c.setAttribute("aria-pressed", String(c.getAttribute(attr) === value)));
  }

  /* ---------- 3.1 / 3.5 block diagram ---------- */
  const MX = 90;
  const BX = 250;
  function boxSvg(cx, cy, w, h, label, cls) {
    return (
      `<g class="l2p3-box ${cls}"><rect x="${cx - w / 2}" y="${cy - h / 2}" width="${w}" height="${h}" rx="8"/>` +
      `<text x="${cx}" y="${cy + 4}" text-anchor="middle">${label}</text></g>`
    );
  }
  // The diagram appears more than once on the page, so marker ids get a per-drawing suffix.
  let svgSeq = 0;
  function drawBlock(mode) {
    const m = blockModel(mode);
    const uid = mode + "-" + ++svgSeq;
    let s =
      `<svg viewBox="0 0 350 545" class="l2p3-svg l2p3-block${REDUCE ? "" : " is-entering"}" role="img" aria-label="${
        mode === "pre" ? "Pre-norm block: the residual path runs straight from x to the top" : "Post-norm block: the residual path passes through a LayerNorm after each Add"
      }">`;
    s += `<defs><marker id="l2p3-hwah-${uid}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0L10 5L0 10z" class="l2p3-hwhead"/></marker>` +
      `<marker id="l2p3-brah-${uid}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" class="l2p3-brhead"/></marker></defs>`;
    // Branches first so the highway sits on top.
    m.branches.forEach((b) => {
      const bottom = b.boxes[0].y + b.boxes[0].h / 2;
      s += `<path class="l2p3-br" d="M${MX} ${b.split} H${BX} V${bottom}" marker-end="url(#l2p3-brah-${uid})"/>`;
      for (let i = 1; i < b.boxes.length; i++) {
        const prevTop = b.boxes[i - 1].y - b.boxes[i - 1].h / 2;
        const nextBottom = b.boxes[i].y + b.boxes[i].h / 2;
        s += `<path class="l2p3-br" d="M${BX} ${prevTop} V${nextBottom}" marker-end="url(#l2p3-brah-${uid})"/>`;
      }
      const top = b.boxes[b.boxes.length - 1];
      s += `<path class="l2p3-br" d="M${BX} ${top.y - top.h / 2} V${b.join} H${MX + 12}" marker-end="url(#l2p3-brah-${uid})"/>`;
      b.boxes.forEach((bx) => (s += boxSvg(BX, bx.y, 164, bx.h, bx.label, "is-" + bx.kind)));
    });
    m.hw.forEach((seg, i) => {
      const end = i === m.hw.length - 1;
      s += `<line class="l2p3-hw" x1="${MX}" y1="${seg[0]}" x2="${MX}" y2="${seg[1]}"${end ? ` marker-end="url(#l2p3-hwah-${uid})"` : ""}/>`;
    });
    m.adds.forEach((y) => {
      s += `<g class="l2p3-add"><circle cx="${MX}" cy="${y}" r="11"/><path d="M${MX - 5} ${y}H${MX + 5}M${MX} ${y - 5}V${y + 5}"/></g>`;
      s += `<text class="l2p3-tag" x="${MX - 18}" y="${y + 4}" text-anchor="end">Add</text>`;
    });
    m.norms.forEach((n) => {
      s += boxSvg(MX, n.y, 120, n.h, "LayerNorm", "is-norm" + (n.onHighway && !n.final ? " is-hit" : "") + (n.final ? " is-final" : ""));
      if (n.final) {
        s += `<text class="l2p3-tag" x="160" y="${n.y - 2}">final norm, once,</text><text class="l2p3-tag" x="160" y="${n.y + 13}">at the very top of the stack</text>`;
      } else {
        s += `<text class="l2p3-tag is-warn" x="158" y="${n.y + 4}">norm on the highway</text>`;
      }
    });
    s += `<text class="l2p3-io" x="${MX}" y="538" text-anchor="middle">x</text>`;
    s += `<text class="l2p3-io" x="${MX}" y="22" text-anchor="middle">output</text>`;
    s += "</svg>";
    return s;
  }
  const BLOCK_READOUT = {
    post:
      "<b>Post-norm, the 2017 design:</b> <span class=\"math\">output = LayerNorm(x + F(x))</span>. Follow the teal highway from x " +
      "to the output. After each Add it runs into a LayerNorm, so the shortcut is rescaled twice in every block, " +
      "and again in the next block, and the next.",
    pre:
      "<b>Pre-norm, used since GPT-2:</b> <span class=\"math\">output = x + F(LayerNorm(x))</span>. Now the teal highway runs straight " +
      "from x to the top without touching a single norm inside the block. Only the branches are normalized, plus one final norm at the very top.",
  };
  function initBlockFigures() {
    document.querySelectorAll("[data-l2p3-block]").forEach((fig) => {
      const stage = fig.querySelector(".ix-stage");
      const readout = fig.querySelector(".ix-readout");
      function show(mode) {
        stage.innerHTML = drawBlock(mode);
        readout.innerHTML = BLOCK_READOUT[mode];
        setChips(fig, "data-mode", mode);
      }
      fig.querySelectorAll(".chip[data-mode]").forEach((c) => c.addEventListener("click", () => show(c.getAttribute("data-mode"))));
      show(fig.getAttribute("data-l2p3-block") || "post");
    });
  }

  /* ---------- 3.2 gradient highway ---------- */
  function drawHighway(n, deriv) {
    const x0 = 44;
    const x1 = 330;
    const slot = (x1 - x0) / n;
    const bw = Math.max(3, slot * 0.7);
    const chart = (top, values, cls, title) => {
      const h = 80;
      const base = top + h;
      let s = `<text class="l2p3-ctitle" x="${x0}" y="${top - 10}">${title}</text>`;
      s += `<line class="l2p3-axis" x1="${x0}" y1="${base}" x2="${x1}" y2="${base}"/>`;
      s += `<line class="l2p3-grid" x1="${x0}" y1="${top}" x2="${x1}" y2="${top}"/>`;
      s += `<text class="l2p3-tick" x="${x0 - 6}" y="${top + 4}" text-anchor="end">1</text>`;
      s += `<text class="l2p3-tick" x="${x0 - 6}" y="${base + 4}" text-anchor="end">0</text>`;
      values.forEach((v, i) => {
        const bh = Math.max(v * h, 0);
        const bx = x0 + i * slot + (slot - bw) / 2;
        s += `<rect class="${cls}" x="${bx}" y="${base - bh}" width="${bw}" height="${bh}" rx="1.5"/>`;
      });
      const lastV = values[values.length - 1];
      const lx = x0 + (n - 1) * slot + slot / 2;
      s += `<text class="l2p3-val" x="${Math.min(lx, x1 - 4)}" y="${base - Math.max(lastV * h, 0) - 6}" text-anchor="end">${fmtSmall(lastV).replace(/<\/?sup>/g, "")}</text>`;
      return s;
    };
    const plain = Array.from({ length: n }, (_, i) => gradPlain(deriv, i + 1));
    const shortcut = Array.from({ length: n }, (_, i) => gradShortcut(i + 1));
    let s = `<svg viewBox="0 0 340 280" class="l2p3-svg" role="img" aria-label="Signal reaching each layer: it shrinks layer by layer in the plain stack and stays at 1 along the shortcut">`;
    s += chart(30, plain, "l2p3-bar is-plain", "Plain stack: × F′(x) at every layer");
    s += chart(150, shortcut, "l2p3-bar is-short", "With residuals: along the shortcut, × 1 at every layer");
    s += `<text class="l2p3-tick" x="${x0}" y="262">1 layer back</text>`;
    s += `<text class="l2p3-tick" x="${x1}" y="262" text-anchor="end">${n} layer${n === 1 ? "" : "s"} back (the first layer)</text>`;
    s += "</svg>";
    return s;
  }
  function initHighway() {
    const fig = document.getElementById("l2p3-highway");
    if (!fig) return;
    const nIn = fig.querySelector("#l2p3-layers");
    const dIn = fig.querySelector("#l2p3-deriv");
    const stage = fig.querySelector(".ix-stage");
    const readout = fig.querySelector(".ix-readout");
    function update() {
      const n = Number(nIn.value);
      const d = Number(dIn.value);
      fig.querySelector("output[for='l2p3-layers']").textContent = n;
      fig.querySelector("output[for='l2p3-deriv']").textContent = num(d);
      stage.innerHTML = drawHighway(n, d);
      const g = gradPlain(d, n);
      readout.innerHTML =
        `<b>Plain stack:</b> <span class="math">${num(d)}<sup>${n}</sup> = ${fmtSmall(g)}</span>. ` +
        (g < 0.01 ? "The first layer gets almost nothing to learn from. " : "Some signal still gets through, but add layers and watch it fade. ") +
        `<b>With residuals:</b> the path along the shortcut multiplies by the 1 in <span class="math">1 + F′(x)</span> at every layer, ` +
        `so it arrives as <span class="math">1<sup>${n}</sup> = ${num(gradShortcut(n))}</span>, however deep the stack.`;
    }
    nIn.addEventListener("input", update);
    dIn.addEventListener("input", update);
    update();
  }

  /* ---------- 3.2 residual stream ---------- */
  const STREAM = [
    { label: "Attention 1", tag: "A1", kind: "attn" },
    { label: "FFN 1", tag: "F1", kind: "ffn" },
    { label: "Attention 2", tag: "A2", kind: "attn" },
    { label: "FFN 2", tag: "F2", kind: "ffn" },
  ];
  function drawStream(k) {
    const SX = 170;
    const levels = [230, 170, 110, 50];
    const seg = 24;
    const right = 150;
    const pile = (count, y, fresh) => {
      let s = "";
      for (let i = 0; i < count; i++) {
        const kind = i === 0 ? "x" : STREAM[i - 1].kind;
        const tag = i === 0 ? "x" : STREAM[i - 1].tag;
        const x = right - (i + 1) * seg;
        s += `<g class="l2p3-seg is-${kind}${fresh && i === count - 1 ? " is-new" : ""}"><rect x="${x}" y="${y - 10}" width="${seg - 2}" height="20" rx="3"/>` +
          `<text x="${x + (seg - 2) / 2}" y="${y + 4}" text-anchor="middle">${tag}</text></g>`;
      }
      return s;
    };
    let s = `<svg viewBox="0 0 340 330" class="l2p3-svg" role="img" aria-label="The residual stream after ${k} sublayers: x plus ${k} updates, none overwritten">`;
    s += `<line class="l2p3-hw" x1="${SX}" y1="312" x2="${SX}" y2="18"/>`;
    s += `<text class="l2p3-io" x="${SX}" y="326" text-anchor="middle">token enters</text>`;
    s += pile(1, 290, false);
    STREAM.forEach((layer, i) => {
      const y = levels[i];
      const on = i < k;
      s += `<g class="l2p3-slayer${on ? " is-on" : ""}">`;
      s += `<path class="l2p3-br" d="M${SX} ${y + 16} H${210}"/>`;
      s += `<path class="l2p3-br" d="M${210} ${y - 12} H${SX + 9}"/>`;
      s += `<g class="l2p3-box is-${layer.kind}"><rect x="210" y="${y - 18}" width="116" height="36" rx="8"/><text x="268" y="${y + 4}" text-anchor="middle">${layer.label}</text></g>`;
      s += `<g class="l2p3-add"><circle cx="${SX}" cy="${y - 12}" r="8"/><path d="M${SX - 4} ${y - 12}H${SX + 4}M${SX} ${y - 16}V${y - 8}"/></g>`;
      if (i === 0) {
        s += `<text class="l2p3-tag" x="182" y="${y + 30}">reads</text><text class="l2p3-tag" x="182" y="${y - 22}">writes</text>`;
      }
      s += "</g>";
      if (on) s += pile(i + 2, y - 12, i === k - 1);
    });
    s += "</svg>";
    return s;
  }
  function initStream() {
    const fig = document.getElementById("l2p3-stream");
    if (!fig) return;
    const kIn = fig.querySelector("#l2p3-stream-k");
    const stage = fig.querySelector(".ix-stage");
    const readout = fig.querySelector(".ix-readout");
    function update() {
      const k = Number(kIn.value);
      fig.querySelector("output[for='l2p3-stream-k']").textContent = k;
      stage.innerHTML = drawStream(k);
      readout.innerHTML =
        k === 0
          ? "Only the token's own vector, <b>x</b>, is in the stream so far. Move the slider to run the sublayers."
          : `After ${k} sublayer${k === 1 ? "" : "s"} the stream holds x plus ${k} update${k === 1 ? "" : "s"} (` +
            STREAM.slice(0, k).map((s) => s.tag).join(", ") +
            "). Each sublayer read the stream and added its update back in. Nothing was overwritten: <b>x</b> is still there, and knowledge piles up.";
    }
    kIn.addEventListener("input", update);
    update();
  }

  ready(() => {
    initBlockFigures();
    initHighway();
    initStream();
  });
})(typeof window !== "undefined" ? window : globalThis);
