// ============================================================
// Lecture 2, Part 2 (sections 2.1 to 2.3): learned positions,
// sinusoidal encoding and the shift to relative position.
// Walkthrough numbers are computed from Lecture.math, so they
// always match the notes and the interactives agree with them.
// ============================================================
(function (root) {
  "use strict";

  const L = (root.Lecture = root.Lecture || {});
  const M = L.math;
  const defs = (L.defs = L.defs || {});
  const f = (x, dp) => M.fmt(x, dp === undefined ? 3 : dp);

  // Show exact integers plainly (0, 1) and everything else to 3 decimals, as the notes do.
  const g = (x) => (Math.abs(x - Math.round(x)) < 1e-12 ? M.fmt(x, 0) : f(x));
  const par = (x) => (M.round(x, 3) < 0 ? "(" + g(x) + ")" : g(x));
  const dotTerms = (a, b) => a.map((x, i) => `${par(x)} × ${par(b[i])}`).join(" + ");
  const dims = (d) => Array.from({ length: d }, (_, j) => "dim " + j);

  /* ---------------- shared helpers (also used by the interactives) ---------------- */
  const pe = (pos, d) => M.sinusoidalPE(pos, d);
  const denominators = (d) => Array.from({ length: d / 2 }, (_, i) => Math.pow(10000, (2 * i) / d));
  const angles = (pos, d) => denominators(d).map((den) => pos / den);

  // A learned table only has the rows it was trained with.
  const TABLE_ROWS = 512;
  function learnedRow(p, width) {
    if (!Number.isInteger(p) || p < 0 || p >= TABLE_ROWS) return null;
    return Array.from({ length: width || 8 }, (_, j) => {
      const s = Math.sin(p * 127.1 + j * 311.7) * 43758.5453;
      return M.round((s - Math.floor(s)) * 0.2 - 0.1, 2);
    });
  }

  // Relative position: the bias only ever sees n - m.
  const offsets = (m, count) => Array.from({ length: count }, (_, n) => n - m);

  L.l2p2a = { pe, denominators, angles, learnedRow, TABLE_ROWS, offsets };

  /* ---------------- 2.2 walkthrough: sinusoidal encoding with d = 4 ---------------- */
  defs["l2-sinusoidal"] = {
    id: "l2-sinusoidal",
    title: "Sinusoidal encoding with d = 4, by hand",
    input: { d: 4, word: "cat", emb: [0.2, 0.5, -0.1, 0.3], at: 1 },
    setup:
      "We build the position vectors for positions 0, 1 and 2 with a tiny embedding size, d = 4. " +
      "Four dimensions means two pairs, i = 0 and i = 1, so two clock hands. " +
      "Then we add one of them to the word “cat”. All angles are in radians.",
    panel: (el) => clockPanel(el),
    build(input) {
      const { d, word, emb, at } = input;
      const den = denominators(d);
      const P = [0, 1, 2].map((p) => pe(p, d));
      const sum = M.add(emb, pe(at, d));
      const cols = dims(d);
      const angleText = (p, i) => String(M.round(p / den[i], 3));
      const fillLines = (p) =>
        P[p].map((v, j) => {
          const i = Math.floor(j / 2);
          const fn = j % 2 === 0 ? "sin" : "cos";
          const inner = p === 0 ? `${fn}(0/${g(den[i])}) = ${fn}(0)` : `${fn}(${angleText(p, i)})`;
          return { t: `dim ${j} = ${inner} = ${g(v)}`, hl: i === 0 && p > 0 };
        });
      const change = (a, b, js) => js.reduce((s, j) => s + Math.abs(P[b][j] - P[a][j]), 0);
      const fastMoves = change(1, 2, [0, 1]);
      const slowMoves = change(1, 2, [2, 3]);
      const answer = slowMoves < fastMoves ? 1 : 0;
      const deg = Math.round(180 / Math.PI);
      const table = (upto, key) => ({
        type: "matrix",
        key,
        rowLabels: P.slice(0, upto + 1).map((_, p) => `PE(${p})`),
        colLabels: cols,
        rows: P.slice(0, upto + 1),
        dp: 3,
      });

      return {
        steps: [
          {
            title: "Work out how fast each hand spins",
            clock: [0],
            say:
              `Each pair i divides the position by 10000<sup>2i/d</sup> before taking sin and cos. ` +
              `Pair 0 divides by ${g(den[0])}, so its angle is the position itself: the fast hand. ` +
              `Pair 1 divides by ${g(den[1])}, so its angle grows ${g(den[1])} times more slowly: the slow hand.`,
            blocks: [
              {
                type: "lines",
                fresh: true,
                lines: den.map((v, i) => ({
                  t: `i = ${i}:  10000<sup>2·${i}/${d}</sup> = 10000<sup>${M.round((2 * i) / d, 3)}</sup> = ${g(v)}`,
                  hl: true,
                })),
              },
              { type: "vector", key: "denoms", label: "divide by", values: den, cols: den.map((_, i) => "pair " + i), fresh: true },
            ],
          },
          {
            title: "Fill in position 0",
            clock: [0],
            say:
              `At position 0 every angle is 0, and sin 0 = 0 while cos 0 = 1. So PE(0) = [${P[0].map(g).join(", ")}]. ` +
              "Both hands start in the same place.",
            blocks: [
              { type: "lines", fresh: true, lines: fillLines(0) },
              { type: "vector", key: "pe0", label: "PE(0)", values: P[0], cols, tone: "result", fresh: true },
            ],
          },
          {
            title: "Fill in position 1",
            clock: [0, 1],
            say:
              `The fast hand turned a whole radian, about ${deg} degrees, so dims 0 and 1 jumped to ${g(P[1][0])} and ${g(P[1][1])}. ` +
              `The slow hand turned only ${M.round(1 / den[1], 3)} of a radian, so dims 2 and 3 barely left 0 and 1.`,
            blocks: [
              { type: "lines", fresh: true, lines: fillLines(1) },
              { type: "vector", key: "pe1", label: "PE(1)", values: P[1], cols, tone: "result", hl: [0, 1], dp: 3, fresh: true },
            ],
          },
          {
            title: "Fill in position 2",
            clock: [0, 1, 2],
            predict: {
              ask: "Going from position 1 to position 2, which dimensions will hardly change?",
              choices: ["Dims 0 and 1, the fast hand", "Dims 2 and 3, the slow hand", "All four change by about the same amount"],
              answer,
              why:
                `The slow hand’s angle only grows by ${M.round(1 / den[1], 3)} per position, so its sin and cos barely move. ` +
                "The fast hand turns a whole radian every step.",
              hint: `Compare the two angles: pos / ${g(den[0])} and pos / ${g(den[1])}.`,
            },
            say:
              `PE(2) = [${P[2].map(g).join(", ")}]. Stack the three rows: the first two columns swing around, ` +
              `the last two have hardly moved from [0, 1]. The slow hand only becomes useful when positions are far apart.`,
            blocks: [
              { type: "lines", fresh: true, lines: fillLines(2) },
              { type: "vector", key: "pe2", label: "PE(2)", values: P[2], cols, tone: "result", hl: [0, 1], dp: 3, fresh: true },
              table(2, "peTable"),
            ],
          },
          {
            title: `Add it to the word “${word}”`,
            clock: [at],
            say:
              `“${word}” sits at position ${at}, so we add PE(${at}) to its embedding, entry by entry, ` +
              `and get [${sum.map((v) => f(v)).join(", ")}]. Now “${word}” at position ${at} looks different ` +
              `from “${word}” at position ${at + 1}, and the model can tell them apart.`,
            blocks: [
              { type: "vector", key: "emb", label: word, values: emb, cols },
              { type: "vector", label: `PE(${at})`, op: "+", values: pe(at, d), dp: 3 },
              { type: "vector", key: "catInput", label: "input", op: "=", values: sum, dp: 3, tone: "result", fresh: true },
            ],
          },
        ],
        takeaway:
          "The first two dimensions changed a lot from position to position: that is the fast hand. " +
          "The last two barely moved: the slow hand, which only becomes useful when positions are far apart, like position 0 versus position 300. " +
          "Adding the pattern to the word embedding stamps every word with where it sits, and no training was needed to make it.",
      };
    },
  };

  /* ---------------- 2.2 walkthrough: similarity between positions ---------------- */
  defs["l2-pe-similarity"] = {
    id: "l2-pe-similarity",
    title: "Building the similarity band, by hand",
    input: { d: 4 },
    setup:
      "The video’s second heatmap takes the dot product of every pair of position vectors. " +
      "You can build a small corner of it yourself with the d = 4 vectors from the last example. First we need one more: PE(3).",
    setupBlocks: (input) => [
      {
        type: "matrix",
        rowLabels: ["PE(0)", "PE(1)", "PE(2)"],
        colLabels: dims(input.d),
        rows: [0, 1, 2].map((p) => pe(p, input.d)),
        dp: 3,
      },
    ],
    build(input) {
      const { d } = input;
      const P = [0, 1, 2, 3].map((p) => pe(p, d));
      const S = P.map((a) => P.map((b) => M.dot(a, b)));
      const labels = P.map((_, p) => `PE(${p})`);
      const DOT = "·";
      const grid = (known, hl, key) => ({
        type: "matrix",
        key,
        rowLabels: labels,
        colLabels: labels,
        rows: S.map((row, r) => row.map((v, c) => (known.some(([a, b]) => (a === r && b === c) || (a === c && b === r)) ? v : DOT))),
        hl: hl.flatMap(([a, b]) => (a === b ? [[a, b]] : [[a, b], [b, a]])),
        dp: 3,
      });
      const line = (a, b, hl) => ({ t: `PE(${a}) · PE(${b}) = ${dotTerms(P[a], P[b])} = ${f(S[a][b])}`, hl });
      const d1 = S[0][1];
      const same = (x, y) => Math.abs(M.round(x, 3) - M.round(y, 3)) < 1e-9;
      const ans1 = same(S[1][2], d1) ? 0 : S[1][2] > d1 ? 1 : 2;
      const ans2 = S[0][2] > d1 ? 0 : same(S[0][2], d1) ? 1 : 2;
      const k1 = [[0, 1]];
      const k2 = k1.concat([[1, 2], [2, 3]]);
      const k3 = k2.concat([[0, 2]]);
      const all = [];
      for (let a = 0; a < 4; a++) for (let b = a; b < 4; b++) all.push([a, b]);

      return {
        steps: [
          {
            title: "Work out PE(3)",
            say:
              `PE(3) = [sin 3, cos 3, sin 0.03, cos 0.03] = [${P[3].map(g).join(", ")}]. ` +
              "The fast hand has now swung round to point almost straight down; the slow hand has still barely moved.",
            blocks: [
              {
                type: "lines",
                fresh: true,
                lines: ["sin 3", "cos 3", "sin 0.03", "cos 0.03"].map((t, j) => ({ t: `dim ${j} = ${t} = ${g(P[3][j])}` })),
              },
              { type: "vector", key: "pe3", label: "PE(3)", values: P[3], cols: dims(d), dp: 3, tone: "result", fresh: true },
            ],
          },
          {
            title: "Distance 1: PE(0) · PE(1)",
            say:
              `Multiply matching entries and add: the dot product is ${f(d1)}. ` +
              "This is how similar the patterns for two neighbouring positions are. It goes into the grid twice, because PE(0) · PE(1) is the same as PE(1) · PE(0).",
            blocks: [
              { type: "lines", fresh: true, lines: [line(0, 1, true)] },
              { type: "vector", key: "d01", label: "distance 1", values: [d1], dp: 3, tone: "result", fresh: true },
              grid(k1, k1, "sim1"),
            ],
          },
          {
            title: "Slide along: two more neighbours",
            predict: {
              ask: `PE(1) and PE(2) are also one step apart, just further along the sentence. What will PE(1) · PE(2) be?`,
              choices: [`Exactly ${f(d1)} again`, `More than ${f(d1)}`, `Less than ${f(d1)}`],
              answer: ans1,
              why:
                "Only the distance matters, not where the pair sits. Both pairs are one step apart, so the dot product is the same. " +
                "That is the angle-addition identity at work, which you will meet again in a moment.",
              hint: "What do the two pairs have in common? Look at how far apart they are.",
            },
            say:
              `PE(1) · PE(2) = ${f(S[1][2])} and PE(2) · PE(3) = ${f(S[2][3])}. ` +
              "Every pair at distance 1 gives exactly the same similarity, no matter where the pair sits.",
            blocks: [
              { type: "lines", fresh: true, lines: [line(1, 2, true), line(2, 3, true)] },
              { type: "vector", key: "dist1", label: "distance 1", values: [S[1][2], S[2][3]], cols: ["PE(1)·PE(2)", "PE(2)·PE(3)"], dp: 3, tone: "result", fresh: true },
              grid(k2, [[1, 2], [2, 3]], "sim2"),
            ],
          },
          {
            title: "Distance 2: PE(0) · PE(2)",
            predict: {
              ask: `PE(0) and PE(2) are two steps apart. Compared with the ${f(d1)} for neighbours, what do you expect?`,
              choices: ["Higher: more alike", "The same", "Lower: less alike"],
              answer: ans2,
              why:
                "Two steps apart, the fast hands point in more different directions, so their sin and cos agree less. " +
                "Positions further apart look less alike.",
              hint: "The dot product is biggest when two hands point the same way. Do the fast hands for 0 and 2 point the same way?",
            },
            say:
              `PE(0) · PE(2) = ${f(S[0][2])}, well below ${f(d1)}. Two steps apart, the patterns already look much less alike.`,
            blocks: [
              { type: "lines", fresh: true, lines: [line(0, 2, true)] },
              { type: "vector", key: "d02", label: "distance 2", values: [S[0][2]], dp: 3, tone: "result", fresh: true },
              grid(k3, [[0, 2]], "sim3"),
            ],
          },
          {
            title: "Distance 3, and the whole grid",
            say:
              `PE(0) · PE(3) = ${f(S[0][3])}. So the similarity drops as distance grows: ${f(d1)}, then ${f(S[0][2])}, then ${f(S[0][3])}. ` +
              `Fill the rest by the same rule and the diagonal is ${g(S[0][0])}, because each sin and cos pair squares to 1 and there are two pairs. ` +
              "Big on the diagonal, fading away from it: that is the bright band.",
            blocks: [
              { type: "lines", fresh: true, lines: [line(0, 3, true)] },
              { type: "vector", key: "d03", label: "distance 3", values: [S[0][3]], dp: 3, tone: "result", fresh: true },
              Object.assign(grid(all, [[0, 3]], "simMatrix"), { fresh: true }),
            ],
          },
        ],
        takeaway:
          `Every pair at distance 1 gives exactly the same similarity (${f(d1)}), no matter where the pair sits. ` +
          `And similarity drops as the distance grows: ${f(d1)}, then ${f(S[0][2])}, then ${f(S[0][3])}. ` +
          "That is the diagonal band in the video’s heatmap, built by hand. In real models with many more dimensions the drop is smoother, " +
          "and it does not bounce back up as quickly as it does in this tiny 4-dimension example.",
      };
    },
  };

  if (typeof module !== "undefined" && module.exports) module.exports = defs;
  if (typeof document === "undefined") return;

  /* ============================================================
     Browser only: drawing helpers and the interactives.
     ============================================================ */
  const REDUCE = root.matchMedia && root.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const SVGNS = "http://www.w3.org/2000/svg";

  function svg(tag, attrs, text) {
    const e = document.createElementNS(SVGNS, tag);
    Object.keys(attrs || {}).forEach((k) => e.setAttribute(k, attrs[k]));
    if (text != null) e.textContent = text;
    return e;
  }
  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }
  function pressOnly(group, btn) {
    group.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
  }
  // Resolve a CSS custom property to [r, g, b] by letting the browser compute it.
  function rgbOf(name) {
    const probe = document.createElement("span");
    probe.style.color = `var(${name})`;
    probe.style.display = "none";
    document.body.append(probe);
    const c = getComputedStyle(probe).color;
    probe.remove();
    const m = (c.match(/[\d.]+/g) || [0, 0, 0]).map(Number);
    // color(srgb r g b) reports 0 to 1 channels.
    return /^color\(/.test(c) ? [m[0] * 255, m[1] * 255, m[2] * 255] : [m[0], m[1], m[2]];
  }
  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));

  /* ---------- a clock dial (one sin/cos pair) ---------- */
  // Hand tip = (cos θ, sin θ): the pair's two dimensions are the tip's coordinates.
  function dial(size) {
    const r = size / 2 - 10;
    const c = size / 2;
    const s = svg("svg", { viewBox: `0 0 ${size} ${size}`, class: "l2p2a-dial", "aria-hidden": "true" });
    s.append(svg("circle", { cx: c, cy: c, r, class: "l2p2a-dial-face" }));
    s.append(svg("line", { x1: c - r, y1: c, x2: c + r, y2: c, class: "l2p2a-dial-axis" }));
    s.append(svg("line", { x1: c, y1: c - r, x2: c, y2: c + r, class: "l2p2a-dial-axis" }));
    const ghosts = svg("g", { class: "l2p2a-dial-ghosts" });
    const guides = svg("g", { class: "l2p2a-dial-guides" });
    const hand = svg("line", { x1: c, y1: c, x2: c + r, y2: c, class: "l2p2a-dial-hand" });
    const tip = svg("circle", { cx: c + r, cy: c, r: 4.5, class: "l2p2a-dial-tip" });
    s.append(ghosts, guides, hand, tip, svg("circle", { cx: c, cy: c, r: 3, class: "l2p2a-dial-hub" }));
    const at = (theta) => [c + r * Math.cos(theta), c - r * Math.sin(theta)];
    return {
      el: s,
      set(theta, previous) {
        const [x, y] = at(theta);
        hand.setAttribute("x2", x);
        hand.setAttribute("y2", y);
        tip.setAttribute("cx", x);
        tip.setAttribute("cy", y);
        guides.replaceChildren(
          svg("line", { x1: x, y1: y, x2: x, y2: c, class: "l2p2a-dial-guide" }),
          svg("line", { x1: x, y1: y, x2: c, y2: y, class: "l2p2a-dial-guide" })
        );
        ghosts.replaceChildren(
          ...(previous || []).map((t) => {
            const [gx, gy] = at(t);
            return svg("line", { x1: c, y1: c, x2: gx, y2: gy, class: "l2p2a-dial-ghost" });
          })
        );
      },
    };
  }

  /* ---------- walkthrough panel: the two hands for d = 4 ---------- */
  function clockPanel(host) {
    host.classList.add("l2p2a-panel");
    const hands = [0, 1].map((i) => {
      const box = el("div", "l2p2a-panel-hand");
      const d = dial(150);
      const cap = el("p", "l2p2a-panel-cap");
      box.append(d.el, cap);
      host.append(box);
      return { d, cap, i };
    });
    const note = el("p", "l2p2a-panel-note");
    host.append(note);
    return {
      render(index, step, input) {
        const list = step.kind === "takeaway" ? [0, 1, 2] : step.clock || [0];
        const now = list[list.length - 1];
        const den = denominators(input.d);
        hands.forEach(({ d, cap, i }) => {
          const theta = now / den[i];
          d.set(theta, list.slice(0, -1).map((p) => p / den[i]));
          cap.innerHTML =
            `<b>${i === 0 ? "Fast" : "Slow"} hand</b>, pair i = ${i}<br>` +
            `angle = ${now} / ${g(den[i])} = ${g(theta)} rad<br>` +
            `<span class="math">sin ${g(Math.sin(theta))}, cos ${g(Math.cos(theta))}</span>`;
        });
        note.textContent =
          list.length > 1
            ? `Faint hands show positions ${list.slice(0, -1).join(" and ")}. The bold hand is position ${now}.`
            : `Both hands at position ${now}.`;
      },
    };
  }

  /* ---------- 2.1 the 512 wall ---------- */
  function initWall(fig) {
    const range = fig.querySelector("#l2p2a-wall-pos");
    const out = fig.querySelector("#l2p2a-wall-out");
    const stage = fig.querySelector(".l2p2a-wall-strip");
    const rowEl = fig.querySelector(".l2p2a-wall-row");
    const readout = fig.querySelector(".ix-readout");
    const MAX = Number(range.max);
    const W = 700;
    const H = 64;
    const X = (p) => (p / MAX) * W;
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", "aria-hidden": "true" });
    const defsEl = svg("defs");
    const pat = svg("pattern", { id: "l2p2a-hatch", width: 10, height: 10, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
    pat.append(svg("line", { x1: 0, y1: 0, x2: 0, y2: 10, class: "l2p2a-hatch-line" }));
    defsEl.append(pat);
    s.append(defsEl);
    const rows = svg("g", { class: "l2p2a-wall-rows" });
    for (let p = 0; p < TABLE_ROWS; p += 2) {
      const v = learnedRow(p, 1)[0];
      rows.append(svg("rect", { x: X(p), y: 8, width: X(2) + 0.4, height: H - 16, style: `opacity:${(0.25 + Math.abs(v) * 6).toFixed(2)}` }));
    }
    s.append(rows);
    s.append(svg("rect", { x: X(TABLE_ROWS), y: 8, width: W - X(TABLE_ROWS), height: H - 16, fill: "url(#l2p2a-hatch)", class: "l2p2a-wall-void" }));
    s.append(svg("line", { x1: X(TABLE_ROWS), y1: 0, x2: X(TABLE_ROWS), y2: H, class: "l2p2a-wall-edge" }));
    const marker = svg("line", { x1: 0, y1: 0, x2: 0, y2: H, class: "l2p2a-wall-marker" });
    s.append(marker);
    stage.prepend(s);
    fig.querySelector(".l2p2a-wall-edge-label").style.left = (TABLE_ROWS / MAX) * 100 + "%";

    function update(p) {
      range.value = p;
      out.textContent = p;
      marker.setAttribute("x1", X(p + 0.5));
      marker.setAttribute("x2", X(p + 0.5));
      const row = learnedRow(p, 8);
      fig.classList.toggle("is-missing", !row);
      rowEl.replaceChildren(el("span", "l2p2a-wall-name", `P<sub>${p}</sub> =`));
      const cells = el("span", "l2p2a-wall-cells");
      (row || new Array(8).fill(null)).forEach((v) => cells.append(el("span", "l2p2a-wall-cell", v == null ? "?" : M.fmt(v, 2))));
      cells.append(el("span", "l2p2a-wall-more", "…"));
      rowEl.append(cells);
      readout.innerHTML = row
        ? `Position ${p} has its own row, P<sub>${p}</sub>, learned during training (the numbers here are made up). ` +
          `The input for a token there is word_embedding(token) + P<sub>${p}</sub>.`
        : `<b>There is no P<sub>${p}</sub>.</b> The table has rows P<sub>0</sub> to P<sub>${TABLE_ROWS - 1}</sub> and nothing else. ` +
          `The model has never seen position ${p} and has no vector to add. The only fix is to add rows and retrain on longer text.`;
    }
    range.addEventListener("input", () => update(Number(range.value)));
    fig.querySelectorAll("[data-jump]").forEach((b) => b.addEventListener("click", () => update(Number(b.dataset.jump))));
    update(Number(range.value));
  }

  /* ---------- 2.2 the clock with many hands ---------- */
  function initClocks(fig) {
    const range = fig.querySelector("#l2p2a-clock-pos");
    const numIn = fig.querySelector("#l2p2a-clock-num");
    const handsEl = fig.querySelector(".l2p2a-hands");
    const readout = fig.querySelector(".ix-readout");
    const playBtn = fig.querySelector(".l2p2a-play");
    const state = { pos: 0, d: 4, timer: 0 };
    let dials = [];

    function build() {
      handsEl.replaceChildren();
      dials = denominators(state.d).map((den, i) => {
        const box = el("div", "l2p2a-hand");
        const d = dial(112);
        const cap = el("p", "l2p2a-hand-cap");
        box.append(d.el, cap);
        handsEl.append(box);
        return { d, cap, den, i };
      });
      draw();
    }
    function draw() {
      const v = pe(state.pos, state.d);
      dials.forEach(({ d, cap, den, i }) => {
        const theta = state.pos / den;
        d.set(theta);
        cap.innerHTML = `i = ${i}, ÷ ${den >= 100 ? M.fmtInt(den) : g(den)}<br><span class="math">${f(v[2 * i], 2)}, ${f(v[2 * i + 1], 2)}</span>`;
      });
      readout.innerHTML =
        `PE(${M.fmtInt(state.pos)}) = <span class="math">[${v.map((x) => f(x)).join(", ")}]</span>. ` +
        (state.d === 4
          ? "The fast hand (i = 0) jumps a whole radian each step; the slow hand (i = 1) creeps."
          : `Hand i = 0 spins fastest and hand i = ${state.d / 2 - 1} slowest. Together they give every position its own reading.`);
    }
    function setPos(p, from) {
      state.pos = Math.max(0, Math.min(100000, Math.round(p) || 0));
      if (from !== "range") range.value = Math.min(state.pos, Number(range.max));
      if (from !== "num") numIn.value = state.pos;
      draw();
    }
    function stopPlay() {
      clearInterval(state.timer);
      state.timer = 0;
      playBtn.setAttribute("aria-pressed", "false");
      playBtn.textContent = "Play";
    }
    range.addEventListener("input", () => {
      stopPlay();
      setPos(Number(range.value), "range");
    });
    numIn.addEventListener("input", () => {
      stopPlay();
      setPos(Number(numIn.value), "num");
    });
    fig.querySelectorAll("[data-d]").forEach((b) =>
      b.addEventListener("click", () => {
        state.d = Number(b.dataset.d);
        pressOnly(b.parentElement, b);
        build();
      })
    );
    if (REDUCE) playBtn.hidden = true;
    playBtn.addEventListener("click", () => {
      if (state.timer) return stopPlay();
      playBtn.setAttribute("aria-pressed", "true");
      playBtn.textContent = "Pause";
      state.timer = setInterval(() => {
        if (state.pos >= Number(range.max)) return stopPlay();
        setPos(state.pos + 1);
      }, 160);
    });
    build();
  }

  /* ---------- 2.2 the two heatmaps ---------- */
  const DS = [4, 8, 16, 32, 64, 128];
  const N = 64;
  function initHeatmaps(fig) {
    const range = fig.querySelector("#l2p2a-heat-d");
    const out = fig.querySelector("#l2p2a-heat-d-out");
    const cv = fig.querySelector(".l2p2a-heat-values");
    const cs = fig.querySelector(".l2p2a-heat-sim");
    const readout = fig.querySelector(".ix-readout");
    let d = DS[Number(range.value)];
    let colors = null;

    function paint(canvas, w, h, cols, rows, valueAt) {
      const dpr = Math.min(2, root.devicePixelRatio || 1);
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const cw = w / cols;
      const ch = h / rows;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const v = Math.max(-1, Math.min(1, valueAt(r, c)));
          const rgb = v >= 0 ? mix(colors.mid, colors.pos, v) : mix(colors.mid, colors.neg, -v);
          ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
          ctx.fillRect(c * cw, r * ch, Math.ceil(cw + 0.5), Math.ceil(ch + 0.5));
        }
      }
    }
    function draw() {
      colors = { pos: rgbOf("--accent"), neg: rgbOf("--decoder"), mid: rgbOf("--surface-2") };
      const P = Array.from({ length: N }, (_, p) => pe(p, d));
      paint(cv, 256, 256, d, N, (r, c) => P[r][c]);
      const half = d / 2;
      paint(cs, 256, 256, N, N, (r, c) => M.dot(P[r], P[c]) / half);
      out.textContent = d;
      range.setAttribute("aria-valuetext", "d = " + d);
    }
    function point(canvas, e, kind) {
      const rect = canvas.getBoundingClientRect();
      const fx = (e.clientX - rect.left) / rect.width;
      const fy = (e.clientY - rect.top) / rect.height;
      if (fx < 0 || fx >= 1 || fy < 0 || fy >= 1) return;
      const row = Math.floor(fy * N);
      if (kind === "values") {
        const col = Math.floor(fx * d);
        const v = pe(row, d)[col];
        readout.innerHTML = `Position ${row}, dimension ${col} (pair i = ${Math.floor(col / 2)}, ${col % 2 ? "cos" : "sin"}): <b>${f(v)}</b>.`;
      } else {
        const col = Math.floor(fx * N);
        const v = M.dot(pe(row, d), pe(col, d));
        readout.innerHTML = `PE(${row}) · PE(${col}) = <b>${f(v)}</b>, distance ${Math.abs(row - col)}. The largest possible value is d / 2 = ${d / 2}.`;
      }
    }
    range.addEventListener("input", () => {
      d = DS[Number(range.value)];
      draw();
    });
    cv.addEventListener("pointermove", (e) => point(cv, e, "values"));
    cv.addEventListener("pointerdown", (e) => point(cv, e, "values"));
    cs.addEventListener("pointermove", (e) => point(cs, e, "sim"));
    cs.addEventListener("pointerdown", (e) => point(cs, e, "sim"));
    document.addEventListener("lecture:theme", () => requestAnimationFrame(draw));
    draw();
  }

  /* ---------- 2.2 the hidden gift: rotation ---------- */
  function initRotation(fig) {
    const posIn = fig.querySelector("#l2p2a-rot-pos");
    const kIn = fig.querySelector("#l2p2a-rot-k");
    const posOut = fig.querySelector("#l2p2a-rot-pos-out");
    const kOut = fig.querySelector("#l2p2a-rot-k-out");
    const stage = fig.querySelector(".l2p2a-rot-stage");
    const formula = fig.querySelector(".l2p2a-rot-formula");
    const readout = fig.querySelector(".ix-readout");
    const size = 220;
    const c = size / 2;
    const r = size / 2 - 16;
    const s = svg("svg", { viewBox: `0 0 ${size} ${size}`, class: "l2p2a-dial l2p2a-rot-dial", "aria-hidden": "true" });
    s.append(svg("circle", { cx: c, cy: c, r, class: "l2p2a-dial-face" }));
    s.append(svg("line", { x1: c - r, y1: c, x2: c + r, y2: c, class: "l2p2a-dial-axis" }));
    s.append(svg("line", { x1: c, y1: c - r, x2: c, y2: c + r, class: "l2p2a-dial-axis" }));
    const arc = svg("path", { class: "l2p2a-rot-arc" });
    const arcLabel = svg("text", { class: "l2p2a-rot-arc-label" });
    const handA = svg("line", { x1: c, y1: c, class: "l2p2a-dial-hand is-muted" });
    const handB = svg("line", { x1: c, y1: c, class: "l2p2a-dial-hand" });
    const tipA = svg("circle", { r: 4.5, class: "l2p2a-dial-tip is-muted" });
    const tipB = svg("circle", { r: 5, class: "l2p2a-dial-tip" });
    const labA = svg("text", { class: "l2p2a-rot-label is-muted" });
    const labB = svg("text", { class: "l2p2a-rot-label" });
    s.append(arc, arcLabel, handA, handB, tipA, tipB, labA, labB, svg("circle", { cx: c, cy: c, r: 3, class: "l2p2a-dial-hub" }));
    stage.append(s);
    const at = (t, rr) => [c + (rr || r) * Math.cos(t), c - (rr || r) * Math.sin(t)];

    function draw() {
      const pos = Number(posIn.value);
      const k = Number(kIn.value);
      posOut.textContent = pos;
      kOut.textContent = k;
      const [ax, ay] = at(pos);
      const [bx, by] = at(pos + k);
      handA.setAttribute("x2", ax);
      handA.setAttribute("y2", ay);
      tipA.setAttribute("cx", ax);
      tipA.setAttribute("cy", ay);
      handB.setAttribute("x2", bx);
      handB.setAttribute("y2", by);
      tipB.setAttribute("cx", bx);
      tipB.setAttribute("cy", by);
      const ar = r * 0.42;
      const [sx, sy] = at(pos, ar);
      const [ex, ey] = at(pos + k, ar);
      const sweep = k % (2 * Math.PI);
      arc.setAttribute("d", k === 0 ? "" : `M ${sx} ${sy} A ${ar} ${ar} 0 ${sweep > Math.PI ? 1 : 0} 0 ${ex} ${ey}`);
      const [lx, ly] = at(pos + k / 2, ar + 18);
      arcLabel.setAttribute("x", lx);
      arcLabel.setAttribute("y", ly + 4);
      arcLabel.textContent = k ? `k = ${k}` : "";
      const [pax, pay] = at(pos, r + 11);
      const [pbx, pby] = at(pos + k, r + 11);
      labA.setAttribute("x", pax);
      labA.setAttribute("y", pay + 4);
      labA.textContent = "pos";
      labB.setAttribute("x", pbx);
      labB.setAttribute("y", pby + 4);
      labB.textContent = k ? "pos+k" : "";

      const sp = Math.sin(pos), cp = Math.cos(pos), sk = Math.sin(k), ck = Math.cos(k);
      const sinSum = sp * ck + cp * sk;
      const cosSum = cp * ck - sp * sk;
      formula.textContent =
        `sin(${pos} + ${k}) = sin ${pos} · cos ${k} + cos ${pos} · sin ${k}\n` +
        `           = ${f(sp)} × ${f(ck)} + ${f(cp)} × ${f(sk)} = ${f(sinSum)}\n` +
        `cos(${pos} + ${k}) = cos ${pos} · cos ${k} − sin ${pos} · sin ${k}\n` +
        `           = ${f(cp)} × ${f(ck)} − ${f(sp)} × ${f(sk)} = ${f(cosSum)}`;
      readout.innerHTML =
        `Check: sin ${pos + k} = ${f(Math.sin(pos + k))} and cos ${pos + k} = ${f(Math.cos(pos + k))}, the fast pair of PE(${pos + k}). ` +
        `The only new numbers you needed were cos ${k} and sin ${k}: they depend on the distance k alone. ` +
        `Drag pos and the arc between the hands never changes size.`;
    }
    posIn.addEventListener("input", draw);
    kIn.addEventListener("input", draw);
    draw();
  }

  /* ---------- 2.3 a query with one bias per key ---------- */
  const TOKENS = ["A", "cute", "teddy bear", "is", "reading", "."];
  function initBias(fig) {
    const list = fig.querySelector(".l2p2a-bias-list");
    const readout = fig.querySelector(".ix-readout");
    let m = 2;
    function draw() {
      const offs = offsets(m, TOKENS.length);
      list.replaceChildren(
        ...TOKENS.map((t, n) => {
          const o = offs[n];
          const li = el("li", "l2p2a-bias-row" + (n === m ? " is-query" : ""));
          const b = el("button", "l2p2a-bias-token", t);
          b.type = "button";
          b.setAttribute("aria-pressed", String(n === m));
          b.setAttribute("aria-label", `Make “${t}” the query`);
          b.addEventListener("click", () => {
            m = n;
            draw();
            list.querySelectorAll("button")[n].focus();
          });
          const where = o === 0 ? "the query itself" : `${Math.abs(o)} ${o < 0 ? "before" : "after"}`;
          li.append(
            b,
            el("span", "l2p2a-bias-n", `n = ${n}`),
            el("span", "l2p2a-bias-where", where),
            el("span", "l2p2a-bias-b", `+ bias(${o === 0 ? "0" : M.fmt(o, 0)})`)
          );
          return li;
        })
      );
      const far = m === 0 ? TOKENS.length - 1 : 0;
      readout.innerHTML =
        `The query is “${TOKENS[m]}” at m = ${m}. Each key’s score gets one extra number that depends only on n − m. ` +
        `“${TOKENS[far]}” is ${Math.abs(far - m)} ${far < m ? "before" : "after"} it, so it gets bias(${M.fmt(far - m, 0)}): ` +
        "exactly what any key that far from any query would get. Pick a different query and every label shifts with it.";
    }
    draw();
  }

  function init() {
    const map = {
      "l2p2a-wall": initWall,
      "l2p2a-clocks": initClocks,
      "l2p2a-heatmaps": initHeatmaps,
      "l2p2a-rotation": initRotation,
      "l2p2a-bias": initBias,
    };
    Object.keys(map).forEach((id) => {
      const node = document.getElementById(id);
      if (node) map[id](node);
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})(typeof window !== "undefined" ? window : globalThis);
