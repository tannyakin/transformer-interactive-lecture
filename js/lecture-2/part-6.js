// ============================================================
// Lecture 2, Part 6: sparse attention.
// One walkthrough (counting pairs, 22 vs 64 and 51,200,000 vs
// 10,000,000,000) and three interactives: Longformer's pattern,
// a 16 x 16 pattern explorer, and the receptive field of stacked
// sliding windows. Every number is computed, never typed.
// ============================================================
(function (root) {
  "use strict";

  const L = (root.Lecture = root.Lecture || {});
  const M = L.math;
  const defs = (L.defs = L.defs || {});
  const n0 = (x) => M.fmtInt(x);
  const range = (n) => Array.from({ length: n }, (_, i) => i);

  /* ---------------- pure helpers (tested in Node) ---------------- */
  // Small seeded random numbers, so "random attention" is repeatable.
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const P6 = {
    // Each token sees itself and r neighbours on each side.
    windowMask: (n, r) => range(n).map((i) => range(n).map((j) => Math.abs(i - j) <= r)),
    // Longformer on the running example: a band of one neighbour each side plus a global [CLS].
    longformerMask(n, view) {
      return range(n).map((i) =>
        range(n).map((j) => {
          const band = Math.abs(i - j) <= 1;
          const glob = i === 0 || j === 0;
          return view === "local" ? band : view === "global" ? glob : band || glob;
        })
      );
    },
    rowCounts: (mask) => mask.map((row) => row.filter(Boolean).length),
    countPairs: (mask) => mask.reduce((s, row) => s + row.filter(Boolean).length, 0),
    // Which pattern computes each pair. 0 means the pair is skipped.
    // opts: { window, strided, global, random } booleans; fixed settings below.
    patternKinds(n, opts, seed) {
      const cfg = { radius: 1, stride: 4, globals: [0], perRow: 2 };
      const rand = rng(seed || 1);
      const picks = range(n).map((i) => {
        const set = new Set();
        while (set.size < cfg.perRow) {
          const j = Math.floor(rand() * n);
          if (j !== i) set.add(j);
        }
        return set;
      });
      return range(n).map((i) =>
        range(n).map((j) => {
          if (opts.global && (cfg.globals.includes(i) || cfg.globals.includes(j))) return "global";
          if (opts.window && Math.abs(i - j) <= cfg.radius) return "window";
          if (opts.strided && Math.abs(i - j) % cfg.stride === 0) return "strided";
          if (opts.random && picks[i].has(j)) return "random";
          return 0;
        })
      );
    },
    // After L layers with window w, information travels about L x w positions.
    reach: (layers, w) => layers * w,
  };
  L.l2p6 = P6;

  /* ---------------- 6.3 counting pairs ---------------- */
  const SQ = "■";
  const DOT = "·";

  defs["l2-window-pairs"] = {
    id: "l2-window-pairs",
    title: "Counting pairs: full attention vs a sliding window",
    input: { n: 8, radius: 1, bigN: 100000, bigW: 512 },
    setup:
      "Take n = 8 tokens. With full attention every token scores all 8. With a sliding window, each token " +
      "only scores itself and its immediate neighbours, one on each side. You pay for every pair you compute, " +
      "so let us count the pairs both ways.",
    setupBlocks: (input) => [
      { type: "lines", lines: [{ t: `tokens      n = ${input.n}` }, { t: `window      itself + ${input.radius} neighbour each side` }] },
    ],
    build(input) {
      const { n, radius, bigN, bigW } = input;
      const cols = range(n).map(String);
      const rowLabels = range(n).map((i) => "tok " + i);
      const full = n * n;
      const win = P6.windowMask(n, radius);
      const counts = P6.rowCounts(win);
      const total = P6.countPairs(win);
      const half = Math.ceil(n / 2);

      const gridUpTo = (rows) => win.map((row, i) => (i < rows ? row.map((on) => (on ? SQ : DOT)) : row.map(() => "")));
      const dotsOf = (rows) => {
        const out = [];
        win.forEach((row, i) => row.forEach((on, j) => i < rows && !on && out.push([i, j])));
        return out;
      };
      const newCells = (from, to) => {
        const out = [];
        win.forEach((row, i) => row.forEach((on, j) => i >= from && i < to && on && out.push([i, j])));
        return out;
      };

      const bigFull = bigN * bigN;
      const bigWin = bigN * bigW;
      const ratio = bigFull / bigWin;
      const n2 = 2 * bigN;
      const fullGrowth = (n2 * n2) / bigFull;
      const winGrowth = (n2 * bigW) / bigWin;
      const TOTAL_CHOICES = [2 * n, total, 3 * n, full].filter((v, i, a) => a.indexOf(v) === i);
      const GROWTH = [1, 2, 4];

      return {
        steps: [
          {
            title: "Full attention: every pair",
            say:
              `With full attention every token attends to all ${n}, so the grid is completely filled: ` +
              `${n} × ${n} = <b>${full}</b> pairs to compute.`,
            blocks: [
              { type: "matrix", key: "fullGrid", rowLabels, colLabels: cols, rows: range(n).map(() => cols.map(() => SQ)) },
              { type: "lines", key: "full", value: full, fresh: true, lines: [{ t: `${n} × ${n} = ${full} pairs`, hl: true }] },
            ],
          },
          {
            title: `Build the window, rows 0 to ${half - 1}`,
            say:
              "Now each token keeps only itself and one neighbour on each side. Token 0 has nobody on its left, " +
              `so it gets just <b>${counts[0]}</b>. Tokens 1, 2 and 3 each get the full <b>${counts[1]}</b>.`,
            blocks: [
              { type: "matrix", rowLabels, colLabels: cols, rows: gridUpTo(half), hl: newCells(0, half), muted: dotsOf(half), fresh: true },
              { type: "vector", key: "countsA", label: "pairs in row", values: counts.slice(0, half), cols: rowLabels.slice(0, half), fresh: true },
            ],
          },
          {
            title: `Rows ${half} to ${n - 1}`,
            say:
              `The band slides down the diagonal. Every middle token gets ${counts[1]}, and the last token, ` +
              `like the first, has only one neighbour, so it gets <b>${counts[n - 1]}</b>.`,
            blocks: [
              { type: "matrix", key: "windowGrid", rowLabels, colLabels: cols, rows: gridUpTo(n), hl: newCells(half, n), muted: dotsOf(n) },
              { type: "vector", key: "rowCounts", label: "pairs in row", values: counts, cols: rowLabels, hl: [0, n - 1], fresh: true },
            ],
          },
          {
            title: "Add up the rows",
            say: `${counts.join(" + ")} = <b>${total}</b>. So <b>${total} instead of ${full}</b>: the window computes about a third of the pairs.`,
            predict: {
              ask: "Most rows have 3 pairs. How many pairs does the whole window grid have?",
              choices: TOTAL_CHOICES.map(String),
              answer: TOTAL_CHOICES.indexOf(total),
              why: `${n - 2} middle rows of ${counts[1]} make ${(n - 2) * counts[1]}, and the two end rows add ${counts[0]} each: ${total}.`,
              hint: `${n} × 3 would be ${n * 3}, but look at the first and last rows again.`,
            },
            blocks: [
              { type: "vector", label: "pairs in row", values: counts, cols: rowLabels },
              {
                type: "lines",
                key: "windowTotal",
                value: total,
                fresh: true,
                lines: [{ t: `window: ${counts.join(" + ")} = ${total} pairs`, hl: true }, { t: `full:   ${n} × ${n} = ${full} pairs` }],
              },
            ],
          },
          {
            title: "Scale it up",
            say:
              `Now a real document: n = ${n0(bigN)} with a window of ${bigW}. Full attention needs ${n0(bigFull)} pairs, ` +
              `the window needs ${n0(bigWin)}. That is about <b>${n0(ratio)} times fewer</b>.`,
            blocks: [
              {
                type: "lines",
                key: "big",
                value: { full: bigFull, window: bigWin, ratio },
                fresh: true,
                lines: [
                  { t: `full:    ${n0(bigN)} × ${n0(bigN)} = ${n0(bigFull)} pairs` },
                  { t: `window:  ${n0(bigN)} × ${bigW}     =     ${n0(bigWin)} pairs` },
                  { t: `${n0(bigFull)} ÷ ${n0(bigWin)} ≈ ${n0(ratio)}`, hl: true },
                ],
              },
            ],
          },
          {
            title: "Double the length",
            say:
              `At ${n0(n2)} tokens full attention grows ${fullGrowth} times, to ${n0(n2 * n2)} pairs. ` +
              `The window grows only <b>${winGrowth} times</b>, to ${n0(n2 * bigW)}, because each token still sees just ${bigW} others. ` +
              "The cost now grows <b>linearly</b> with n.",
            predict: {
              ask: `Double the document to ${n0(n2)} tokens, keeping the window at ${bigW}. How much does the window's cost grow?`,
              choices: ["It stays the same", "2 times", "4 times"],
              answer: GROWTH.indexOf(winGrowth),
              why: `Each token still computes ${bigW} pairs, and there are twice as many tokens, so the total doubles. Only full attention quadruples.`,
              hint: "The window size per token does not change. Only the number of tokens does.",
            },
            blocks: [
              {
                type: "lines",
                key: "doubled",
                value: { fullGrowth, winGrowth },
                fresh: true,
                lines: [
                  { t: `full:    ${n0(n2)} × ${n0(n2)} = ${n0(n2 * n2)}   (${fullGrowth}×)` },
                  { t: `window:  ${n0(n2)} × ${bigW}     =    ${n0(n2 * bigW)}   (${winGrowth}×)`, hl: true },
                ],
              },
            ],
          },
        ],
        takeaway:
          `A window of one neighbour each side cut ${full} pairs to ${total}. At ${n0(bigN)} tokens a window of ${bigW} ` +
          `computes about ${n0(ratio)} times fewer pairs than full attention. And the cost now grows linearly: ` +
          "double the length, double the cost, not quadruple.",
      };
    },
  };

  if (typeof module !== "undefined" && module.exports) module.exports = defs;
  if (typeof document === "undefined") return;

  /* ============================================================
     Browser side
     ============================================================ */
  const NS = "http://www.w3.org/2000/svg";
  function svg(tag, attrs, text) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }
  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }

  /* ---------- 6.1 Longformer's pattern ---------- */
  function setupLongformer() {
    const fig = document.getElementById("l2p6-long");
    if (!fig) return;
    const ROWS = ["[CLS]", "A", "cute", "teddy bear", "...", "...", "...", "..."];
    const COLS = ["[CLS]", "A", "cute", "teddy", "...", "...", "...", "..."];
    const N = ROWS.length;
    const chips = Array.from(fig.querySelectorAll(".chip[data-view]"));
    const stage = fig.querySelector(".ix-stage");
    const readout = fig.querySelector(".ix-readout");
    const state = { view: "both", q: 3 };
    const name = (i) => (i === 0 ? "[CLS]" : i < 4 ? `token ${i} (“${ROWS[i]}”)` : `token ${i}`);

    function draw() {
      const mask = P6.longformerMask(N, state.view);
      const q = state.q;
      let far = -1;
      mask[q].forEach((on, j) => {
        if (!on && (far < 0 || Math.abs(q - j) > Math.abs(q - far))) far = j;
      });
      const hop = far >= 0 && mask[q][0] && mask[0][far] && q !== 0;
      const table = el("table", "l2p6-lgrid");
      const cap = el("caption", "sr-only", "Longformer attention pattern. Rows are queries, columns are keys.");
      const thead = el("thead");
      const hr = el("tr");
      hr.append(el("th", "l2p6-corner", '<span class="sr-only">query</span>'));
      COLS.forEach((c, j) => {
        const th = el("th", "l2p6-colh" + (j === far ? " is-target" : ""));
        th.scope = "col";
        th.innerHTML = `<span>${c}</span>`;
        hr.append(th);
      });
      thead.append(hr);
      const tbody = el("tbody");
      mask.forEach((row, i) => {
        const tr = el("tr", i === q ? "is-sel" : "");
        const th = el("th");
        th.scope = "row";
        const b = el("button", "l2p6-q", ROWS[i]);
        b.type = "button";
        b.setAttribute("aria-pressed", String(i === q));
        b.addEventListener("click", () => {
          state.q = i;
          draw();
          const again = stage.querySelectorAll(".l2p6-q")[i];
          if (again) again.focus();
        });
        th.append(b);
        tr.append(th);
        row.forEach((on, j) => {
          let cls = "l2p6-c" + (on ? (i === 0 || j === 0 ? " is-glob" : " is-band") : "");
          if (hop && ((i === q && j === 0) || (i === 0 && j === far))) cls += " is-hop";
          const td = el("td", cls, `<span class="sr-only">${on ? "sees" : "skips"}</span>`);
          tr.append(td);
        });
        tbody.append(tr);
      });
      table.append(cap, thead, tbody);
      stage.replaceChildren(table);

      const sees = mask[q].map((on, j) => (on ? j : -1)).filter((j) => j >= 0);
      let t = `<b>${P6.countPairs(mask)}</b> of ${N * N} pairs computed. `;
      t += `${name(q)[0].toUpperCase() + name(q).slice(1)} attends to ${sees.length} key${sees.length === 1 ? "" : "s"}. `;
      if (q === 0 && mask[0].every(Boolean)) t += "[CLS] is global: it sees every token.";
      else if (far < 0) t += "It already sees every token.";
      else if (hop) t += `It cannot see ${name(far)} directly, but it can in <b>two hops</b>: [CLS] reads ${name(far)}, then ${name(q)} reads [CLS] (the outlined cells).`;
      else t += `It cannot see ${name(far)}. Without the global token, information has to creep along the band one neighbour per layer.`;
      readout.innerHTML = t;
    }

    chips.forEach((c) =>
      c.addEventListener("click", () => {
        state.view = c.dataset.view;
        chips.forEach((x) => x.setAttribute("aria-pressed", String(x === c)));
        draw();
      })
    );
    draw();
  }

  /* ---------- 6.2 pattern explorer ---------- */
  function setupPatterns() {
    const fig = document.getElementById("l2p6-pat");
    if (!fig) return;
    const N = 16;
    const chips = Array.from(fig.querySelectorAll(".chip[data-pat]"));
    const shuffle = fig.querySelector("#l2p6-shuffle");
    const stage = fig.querySelector(".ix-stage");
    const readout = fig.querySelector(".ix-readout");
    const counts = {};
    fig.querySelectorAll("[data-count]").forEach((s) => (counts[s.dataset.count] = s));
    const opts = { window: true, strided: false, global: false, random: false };
    let seed = 7;
    const C = 18;
    const O = 4;
    const box = svg("svg", { viewBox: `0 0 ${N * C + 2 * O} ${N * C + 2 * O}`, class: "l2p6-pat-svg", role: "img" });
    stage.append(box);
    const NAMES = { window: "sliding window", strided: "strided", global: "global tokens", random: "random" };

    function draw() {
      const kinds = P6.patternKinds(N, opts, seed);
      box.replaceChildren();
      kinds.forEach((row, i) =>
        row.forEach((k, j) => {
          const x = O + j * C;
          const y = O + i * C;
          if (k === "random") {
            box.append(svg("rect", { x: x + 1, y: y + 1, width: C - 2, height: C - 2, rx: 2, class: "l2p6-off" }));
            box.append(svg("circle", { cx: x + C / 2, cy: y + C / 2, r: C / 2 - 4, class: "l2p6-k-random" }));
          } else if (k === "strided") {
            box.append(svg("rect", { x: x + 1, y: y + 1, width: C - 2, height: C - 2, rx: 2, class: "l2p6-off" }));
            box.append(svg("rect", { x: x + 4, y: y + 4, width: C - 8, height: C - 8, rx: 1, class: "l2p6-k-strided" }));
          } else {
            box.append(svg("rect", { x: x + 1, y: y + 1, width: C - 2, height: C - 2, rx: 2, class: k ? "l2p6-k-" + k : "l2p6-off" }));
          }
        })
      );
      const total = kinds.reduce((s, row) => s + row.filter(Boolean).length, 0);
      Object.keys(NAMES).forEach((p) => {
        if (!counts[p]) return;
        const only = P6.patternKinds(N, { [p]: true }, seed);
        counts[p].textContent = n0(only.reduce((s, row) => s + row.filter(Boolean).length, 0)) + " pairs alone";
      });
      const on = Object.keys(NAMES).filter((p) => opts[p]);
      box.setAttribute("aria-label", `A ${N} by ${N} attention grid with ${total} of ${N * N} pairs computed.`);
      readout.innerHTML = on.length
        ? `With ${on.map((p) => NAMES[p]).join(" + ")}: <b>${n0(total)}</b> of ${N * N} pairs computed, ` +
          `${M.fmt((100 * total) / (N * N), 0)}% of full attention, about ${M.fmt(total / N, 1)} keys per token.`
        : `Nothing switched on: 0 of ${N * N} pairs. Full attention would compute all ${N * N}.`;
      shuffle.disabled = !opts.random;
    }

    chips.forEach((c) =>
      c.addEventListener("click", () => {
        opts[c.dataset.pat] = !opts[c.dataset.pat];
        c.setAttribute("aria-pressed", String(opts[c.dataset.pat]));
        draw();
      })
    );
    shuffle.addEventListener("click", () => {
      seed += 1;
      draw();
    });
    draw();
  }

  /* ---------- 6.3 receptive field ---------- */
  function setupReach() {
    const fig = document.getElementById("l2p6-reach");
    if (!fig) return;
    const input = fig.querySelector("#l2p6-layers");
    const out = fig.querySelector("#l2p6-layers-out");
    const stage = fig.querySelector(".ix-stage");
    const readout = fig.querySelector(".ix-readout");
    const T = 15;
    const F = 7;
    const MAXL = Number(input.max);
    const X0 = 70;
    const DX = 19.5;
    const Y0 = 20;
    const DY = 30;
    const H = Y0 + MAXL * DY + 40;
    const box = svg("svg", { viewBox: `0 0 ${X0 + (T - 1) * DX + 14} ${H}`, class: "l2p6-reach-svg", role: "img" });
    stage.append(box);

    function draw() {
      const Lr = Number(input.value);
      out.textContent = Lr;
      box.replaceChildren();
      const x = (t) => X0 + t * DX;
      // Layer 0 (the input tokens) sits at the bottom, layer Lr at the top.
      const y = (l) => Y0 + (MAXL - l) * DY;
      const inCone = (t, l) => l <= Lr && Math.abs(t - F) <= Lr - l;
      for (let l = 1; l <= Lr; l++) {
        for (let t = 0; t < T; t++) {
          if (!inCone(t, l)) continue;
          for (let d = -1; d <= 1; d++) {
            const s = t + d;
            if (s < 0 || s >= T) continue;
            box.append(svg("line", { x1: x(t), y1: y(l), x2: x(s), y2: y(l - 1), class: "l2p6-edge" }));
          }
        }
      }
      for (let l = 0; l <= MAXL; l++) {
        box.append(svg("text", { x: 4, y: y(l) + 4, class: "l2p6-rlab" + (l > Lr ? " is-off" : "") }, l === 0 ? "input" : "layer " + l));
        for (let t = 0; t < T; t++) {
          const cls = l > Lr ? "l2p6-node is-off" : l === Lr && t === F ? "l2p6-node is-focus" : inCone(t, l) ? "l2p6-node is-on" : "l2p6-node";
          box.append(svg("circle", { cx: x(t), cy: y(l), r: l === Lr && t === F ? 7 : 5, class: cls }));
        }
      }
      for (let t = 0; t < T; t++) box.append(svg("text", { x: x(t), y: y(0) + 22, "text-anchor": "middle", class: "l2p6-rlab" + (inCone(t, 0) ? " is-reached" : "") }, String(t)));

      const lo = Math.max(0, F - Lr);
      const hi = Math.min(T - 1, F + Lr);
      const span = hi - lo + 1;
      box.setAttribute("aria-label", `After ${Lr} layers token ${F} has information from tokens ${lo} to ${hi}.`);
      readout.innerHTML =
        `After <b>${Lr}</b> layer${Lr === 1 ? "" : "s"}, token ${F} has gathered information from tokens ${lo} to ${hi}: ` +
        `<b>${span}</b> positions, even though each layer only looks one neighbour to each side. ` +
        `The reach grows by one window per layer, ${Lr} × 1 on each side. ` +
        `A CNN grows the same way: ${Lr} layer${Lr === 1 ? "" : "s"} of 3x3 filters see a ${span}x${span} patch.`;
    }
    input.addEventListener("input", draw);
    draw();
  }

  function init() {
    setupLongformer();
    setupPatterns();
    setupReach();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})(typeof window !== "undefined" ? window : globalThis);
