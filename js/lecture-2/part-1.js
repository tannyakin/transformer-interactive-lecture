// ============================================================
// Lecture 2, Part 1: why attention cannot see order.
// One set of toy query and key vectors drives both the shuffle
// test and the "dog bites man" walkthrough, so the two always
// agree. Every number shown is computed here, never typed.
// ============================================================
(function (root) {
  "use strict";

  const L = (root.Lecture = root.Lecture || {});
  const M = L.math;
  const defs = (L.defs = L.defs || {});

  // Fewest decimals (up to 3) that show x exactly, with a true minus sign.
  function num(x) {
    for (let dp = 0; dp <= 3; dp++) {
      if (Math.abs(M.round(x, dp) - M.round(x, 3)) < 1e-9) return M.fmt(x, dp);
    }
    return M.fmt(x, 3);
  }
  const par = (x) => (x < 0 ? "(" + num(x) + ")" : num(x));
  const vec = (v) => "[" + v.map(num).join(", ") + "]";
  const dotTerms = (a, b) => a.map((x, i) => `${par(x)} × ${par(b[i])}`).join(" + ");

  /* ---------------- toy model ---------------- */
  // Each word owns one query and one key. Nothing in them says where the word sits.
  const WORDS = {
    dog: { q: [1, 0.5], k: [0.5, 1] },
    bites: { q: [0, 1], k: [1, 0] },
    man: { q: [0.5, 1], k: [1, 1] },
  };
  // A tiny vector per position, used only when we switch position on.
  const POS = [
    [0, 0],
    [0.5, 0],
    [0, 0.5],
  ];
  const SENTENCES = {
    a: ["dog", "bites", "man"],
    b: ["man", "bites", "dog"],
  };

  function vectorsFor(words, withPos) {
    return words.map((w, p) => ({
      word: w,
      pos: p,
      q: withPos ? M.add(WORDS[w].q, POS[p]) : WORDS[w].q.slice(),
      k: withPos ? M.add(WORDS[w].k, POS[p]) : WORDS[w].k.slice(),
    }));
  }

  // rows = the word doing the looking (query), columns = the word looked at (key).
  function scoreGrid(words, withPos) {
    const v = vectorsFor(words, withPos);
    return v.map((a) => v.map((b) => M.dot(a.q, b.k)));
  }

  function pair(words, from, to, withPos) {
    const v = vectorsFor(words, withPos);
    const a = v.find((x) => x.word === from);
    const b = v.find((x) => x.word === to);
    return { q: a.q, k: b.k, score: M.dot(a.q, b.k), qPos: a.pos, kPos: b.pos };
  }

  const ordinal = (i) => ["1st", "2nd", "3rd"][i];
  const quoted = (words) => "“" + words.join(" ") + "”";

  L.l2p1 = { WORDS, POS, SENTENCES, vectorsFor, scoreGrid, pair, num };

  /* ---------------- walkthrough: dog bites man ---------------- */
  defs["l2-order"] = {
    id: "l2-order",
    title: "Dog bites man, by hand",
    input: { from: "dog", to: "man", a: SENTENCES.a, b: SENTENCES.b },
    setup:
      "Three words, each with a tiny query and a tiny key of just two numbers. Think of them as what " +
      "W<sub>Q</sub> and W<sub>K</sub> made from each word’s embedding. Notice what is missing: " +
      "nothing in them says where the word sits. We will score how much “dog” attends to “man” " +
      "in two sentences that mean very different things.",
    setupBlocks: () => [
      {
        type: "matrix",
        key: "toy",
        rowLabels: Object.keys(WORDS),
        colLabels: ["q₁", "q₂", "k₁", "k₂"],
        rows: Object.keys(WORDS).map((w) => WORDS[w].q.concat(WORDS[w].k)),
      },
    ],
    build(input) {
      const { from, to, a, b } = input;
      const pa = pair(a, from, to, false);
      const pb = pair(b, from, to, false);
      const gridA = scoreGrid(a, false);
      const gridB = scoreGrid(b, false);
      const qa = pair(a, from, to, true);
      const qb = pair(b, from, to, true);
      const same = Math.abs(pb.score - pa.score) < 1e-9;
      const answer = same ? 1 : pb.score > pa.score ? 0 : 2;
      const where = (words) =>
        words.map((w, i) => (w === from || w === to ? `<b>${w}</b>` : w) + ` (${ordinal(i)})`).join(" · ");
      const qRow = (p) => ({ type: "vector", label: `q<sub>${from}</sub>`, values: p.q });
      const kRow = (p) => ({ type: "vector", label: `k<sub>${to}</sub>`, values: p.k });
      const cellA = [a.indexOf(from), a.indexOf(to)];
      const cellB = [b.indexOf(from), b.indexOf(to)];

      return {
        steps: [
          {
            title: `Look up the vectors in ${quoted(a)}`,
            say:
              `“${from}” is the ${ordinal(pa.qPos)} word and “${to}” is the ${ordinal(pa.kPos)}. ` +
              `To score ${from} looking at ${to}, attention takes ${from}’s query and ${to}’s key. ` +
              "Look at what the lookup used: which word it is. Where the word sits never came into it.",
            blocks: [
              { type: "lines", fresh: true, lines: [{ t: where(a) }] },
              Object.assign(qRow(pa), { fresh: true, key: "qA" }),
              Object.assign(kRow(pa), { fresh: true, key: "kA" }),
            ],
          },
          {
            title: `Score ${from} → ${to}`,
            say:
              "A dot product multiplies matching entries and adds them up: " +
              `${dotTerms(pa.q, pa.k)} = ${num(pa.score)}. We skip the division by √d<sub>k</sub> = √2 here. ` +
              "It divides every score by the same number, so it can never make two equal scores different.",
            blocks: [
              qRow(pa),
              kRow(pa),
              { type: "lines", fresh: true, lines: [{ t: `q<sub>${from}</sub> · k<sub>${to}</sub> = ${dotTerms(pa.q, pa.k)} = ${num(pa.score)}`, hl: true }] },
              { type: "vector", key: "scoreA", label: "score", values: [pa.score], tone: "result", fresh: true },
            ],
          },
          {
            title: `Now read ${quoted(b)}`,
            say:
              `This time “${from}” is ${ordinal(pb.qPos)} and “${to}” is ${ordinal(pb.kPos)}. ` +
              `But ${from}’s query is still ${vec(pb.q)} and ${to}’s key is still ${vec(pb.k)}, ` +
              `so the score is ${num(pb.score)} again. Same words in, same numbers out.`,
            predict: {
              ask:
                `In ${quoted(b)}, “${from}” is now the ${ordinal(pb.qPos)} word and “${to}” the ${ordinal(pb.kPos)}. ` +
                `What will the score from ${from} to ${to} be?`,
              choices: [`Higher than ${num(pa.score)}`, `Exactly ${num(pa.score)}`, `Lower than ${num(pa.score)}`],
              answer,
              why:
                `Nothing in the score depends on where the words sit. ${from} brings the same query and ${to} the same key, ` +
                "so the dot product cannot change.",
              hint: "Which numbers go into the score? Did any of them change when the words swapped places?",
            },
            blocks: [
              { type: "lines", fresh: true, lines: [{ t: where(b) }] },
              qRow(pb),
              kRow(pb),
              { type: "lines", lines: [{ t: `q<sub>${from}</sub> · k<sub>${to}</sub> = ${dotTerms(pb.q, pb.k)} = ${num(pb.score)}`, hl: true }] },
              { type: "vector", key: "scoreB", label: "score", values: [pb.score], tone: "result", fresh: true },
            ],
          },
          {
            title: "Every score, in both sentences",
            say:
              "Here is the full grid of scores for each sentence. Rows are the word doing the looking, columns the word being looked at. " +
              `The same nine numbers appear in both. The ${from} row is identical; it just moved from the top to the bottom, ` +
              "and the columns moved the same way. That is all a reordering does to attention.",
            blocks: [
              { type: "lines", lines: [quoted(a)] },
              { type: "matrix", key: "gridA", rowLabels: a, colLabels: a, rows: gridA, hl: [cellA], fresh: true },
              { type: "lines", lines: [quoted(b)] },
              { type: "matrix", key: "gridB", rowLabels: b, colLabels: b, rows: gridB, hl: [cellB], fresh: true },
            ],
          },
          {
            title: "Give each position its own small vector",
            say:
              "Now add a small vector for each position to every query and key. " +
              `In ${quoted(a)}, ${from} sits at position ${qa.qPos} and ${to} at position ${qa.kPos}, and the score becomes ${num(qa.score)}. ` +
              `In ${quoted(b)} they swap places, and the score becomes ${num(qb.score)}. ` +
              "For the first time, the two sentences look different to attention. Making that work well is what Part 2 is about.",
            blocks: [
              {
                type: "matrix",
                key: "posVecs",
                rowLabels: POS.map((_, p) => "position " + p),
                colLabels: ["1st", "2nd"],
                rows: POS,
              },
              {
                type: "lines",
                fresh: true,
                lines: [
                  { t: `${quoted(a)}: q = ${vec(WORDS[from].q)} + ${vec(POS[qa.qPos])}, k = ${vec(WORDS[to].k)} + ${vec(POS[qa.kPos])}` },
                  { t: `  ${dotTerms(qa.q, qa.k)} = ${num(qa.score)}`, hl: true },
                  { t: `${quoted(b)}: q = ${vec(WORDS[from].q)} + ${vec(POS[qb.qPos])}, k = ${vec(WORDS[to].k)} + ${vec(POS[qb.kPos])}` },
                  { t: `  ${dotTerms(qb.q, qb.k)} = ${num(qb.score)}`, hl: true },
                ],
              },
              {
                type: "vector",
                key: "posScores",
                label: `${from} → ${to}`,
                values: [qa.score, qb.score],
                cols: [a.join(" "), b.join(" ")],
                tone: "result",
                fresh: true,
              },
            ],
          },
        ],
        takeaway:
          `Without position, attention scored ${from} → ${to} as ${num(pa.score)} in both sentences, and every other score only changed rows. ` +
          "Shuffle the words and the outputs get shuffled the same way, but nothing else changes: that is <b>permutation equivariance</b>. " +
          "Add anything that depends on position and the tie breaks. That is the whole job of positional encoding.",
      };
    },
  };

  if (typeof module !== "undefined" && module.exports) module.exports = defs;
  if (typeof document === "undefined") return;

  /* ============================================================
     Browser only: the interactives.
     ============================================================ */
  const REDUCE = root.matchMedia && root.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const SVGNS = "http://www.w3.org/2000/svg";

  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }
  function svg(tag, attrs, text) {
    const e = document.createElementNS(SVGNS, tag);
    Object.keys(attrs || {}).forEach((k) => e.setAttribute(k, attrs[k]));
    if (text != null) e.textContent = text;
    return e;
  }
  function pressOnly(group, btn) {
    group.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
  }

  /* ---------- the shuffle test ---------- */
  function initShuffle(fig) {
    const gridEl = fig.querySelector(".l2p1-grid");
    const readout = fig.querySelector(".ix-readout");
    const state = { s: "a", pos: false };
    const words = Object.keys(WORDS);
    const cells = {};
    const colHead = {};
    const rowHead = {};

    const corner = el("span", "l2p1-corner", '<span class="l2p1-corner-q">query</span><span class="l2p1-corner-k">key</span>');
    corner.style.gridRow = "1";
    corner.style.gridColumn = "1";
    gridEl.append(corner);
    words.forEach((w) => {
      colHead[w] = el("span", "l2p1-head is-col", w);
      rowHead[w] = el("span", "l2p1-head is-row", w);
      gridEl.append(colHead[w], rowHead[w]);
    });
    words.forEach((q) =>
      words.forEach((k) => {
        const c = el("span", "l2p1-cell");
        c.setAttribute("role", "cell");
        cells[q + "|" + k] = c;
        gridEl.append(c);
      })
    );
    const animated = () => Array.from(gridEl.children).filter((c) => c !== corner);

    function layout() {
      const order = SENTENCES[state.s];
      const grid = scoreGrid(order, state.pos);
      const before = new Map(animated().map((c) => [c, c.getBoundingClientRect()]));
      order.forEach((w, i) => {
        colHead[w].style.gridRow = "1";
        colHead[w].style.gridColumn = String(i + 2);
        rowHead[w].style.gridRow = String(i + 2);
        rowHead[w].style.gridColumn = "1";
      });
      order.forEach((q, r) =>
        order.forEach((k, c) => {
          const cell = cells[q + "|" + k];
          const v = grid[r][c];
          cell.style.gridRow = String(r + 2);
          cell.style.gridColumn = String(c + 2);
          cell.style.setProperty("--v", Math.round((v / 2.25) * 60));
          cell.textContent = num(v);
          const focus = q === "dog" && k === "man";
          cell.classList.toggle("is-focus", focus);
          cell.setAttribute("aria-label", `${q} looking at ${k}: ${num(v)}`);
        })
      );
      rowHead.dog.classList.add("is-focus");
      if (!REDUCE) {
        animated().forEach((c) => {
          const a = before.get(c);
          const b = c.getBoundingClientRect();
          const dx = a.left - b.left;
          const dy = a.top - b.top;
          if (!dx && !dy) return;
          c.style.transition = "none";
          c.style.transform = `translate(${dx}px, ${dy}px)`;
          requestAnimationFrame(() =>
            requestAnimationFrame(() => {
              c.style.transition = "";
              c.style.transform = "";
            })
          );
        });
      }

      const here = pair(order, "dog", "man", state.pos);
      const other = pair(SENTENCES[state.s === "a" ? "b" : "a"], "dog", "man", state.pos);
      const weights = M.softmax(grid[order.indexOf("dog")]).weights;
      const wText = order.map((w, i) => `${w} ${M.fmt(weights[i], 3)}`).join(", ");
      const hereName = quoted(order);
      const otherName = quoted(SENTENCES[state.s === "a" ? "b" : "a"]);
      if (!state.pos) {
        readout.innerHTML =
          `In ${hereName}, dog → man scores <b>${num(here.score)}</b>. In ${otherName} it scores <b>${num(other.score)}</b>. ` +
          `The dog row always holds the same three numbers, so its attention weights are the same too (${wText}). ` +
          "Switching sentences only moved rows and columns around.";
      } else {
        readout.innerHTML =
          `With position added, dog → man scores <b>${num(here.score)}</b> in ${hereName} but <b>${num(other.score)}</b> in ${otherName}. ` +
          `The dog row now reads differently in each sentence (here: ${wText}), so attention can finally tell the two apart.`;
      }
      fig.classList.toggle("is-pos", state.pos);
    }

    fig.querySelectorAll("[data-sentence]").forEach((b) =>
      b.addEventListener("click", () => {
        state.s = b.dataset.sentence;
        pressOnly(b.parentElement, b);
        layout();
      })
    );
    fig.querySelectorAll("[data-pos]").forEach((b) =>
      b.addEventListener("click", () => {
        state.pos = b.dataset.pos === "on";
        pressOnly(b.parentElement, b);
        layout();
      })
    );
    layout();
  }

  /* ---------- "direct links lose position" ---------- */
  const TEDDY = ["A", "cute", "teddy bear", "is", "reading", "."];
  function initLinks(fig) {
    const stage = fig.querySelector(".l2p1-links-stage");
    const readout = fig.querySelector(".ix-readout");
    const QUERY = 2;

    function draw(far) {
      const tokens = TEDDY.map((t, i) => ({ t, i }));
      if (far) tokens.splice(2, 0, { t: "… 49 other words …", gap: true });
      const CH = 10.5;
      const PAD = 14;
      const GAP = 14;
      let x = 12;
      tokens.forEach((tk) => {
        tk.w = tk.t.length * CH + PAD * 2;
        tk.x = x;
        tk.cx = x + tk.w / 2;
        x += tk.w + GAP;
      });
      const W = x;
      const H = 168;
      const base = 128;
      const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-labelledby": "l2p1-links-title" });
      s.style.minWidth = Math.min(W, 560) + "px";
      const q = tokens.find((tk) => tk.i === QUERY);
      tokens.forEach((tk) => {
        if (tk.gap || tk.i === QUERY) return;
        const span = Math.abs(tk.cx - q.cx);
        const lift = Math.min(98, 26 + span * 0.28);
        const midX = (tk.cx + q.cx) / 2;
        const d = `M ${q.cx} ${base - 22} Q ${midX} ${base - 22 - lift * 2} ${tk.cx} ${base - 22}`;
        const isCute = tk.t === "cute";
        s.append(svg("path", { d, class: "l2p1-link" + (isCute ? " is-cute" : "") }));
        if (isCute) {
          const lx = midX;
          const ly = base - 22 - lift - 6;
          s.append(svg("rect", { x: lx - 30, y: ly - 15, width: 60, height: 24, rx: 6, class: "l2p1-link-tag" }));
          s.append(svg("text", { x: lx, y: ly + 2, class: "l2p1-link-label" }, "q·k"));
        }
      });
      tokens.forEach((tk) => {
        const g = svg("g", { class: "l2p1-token" + (tk.gap ? " is-gap" : "") + (tk.i === QUERY ? " is-query" : "") + (tk.t === "cute" ? " is-cute" : "") });
        g.append(svg("rect", { x: tk.x, y: base - 20, width: tk.w, height: 36, rx: 8 }));
        g.append(svg("text", { x: tk.cx, y: base + 4 }, tk.t));
        s.append(g);
      });
      stage.replaceChildren(s);
      readout.innerHTML = far
        ? "“cute” is now 50 words away from “teddy bear”. The link between them is still one score, " +
          "q<sub>teddy bear</sub> · k<sub>cute</sub>, built from the same two vectors. It comes out exactly as before."
        : "“cute” sits right beside “teddy bear”. The link between them is one score, " +
          "q<sub>teddy bear</sub> · k<sub>cute</sub>. Now push “cute” 50 words away and watch what changes.";
    }

    fig.querySelectorAll("[data-far]").forEach((b) =>
      b.addEventListener("click", () => {
        pressOnly(b.parentElement, b);
        draw(b.dataset.far === "on");
      })
    );
    draw(false);
  }

  function init() {
    const shuffle = document.getElementById("l2p1-shuffle");
    if (shuffle) initShuffle(shuffle);
    const links = document.getElementById("l2p1-links");
    if (links) initLinks(links);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})(typeof window !== "undefined" ? window : globalThis);
