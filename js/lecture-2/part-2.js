// ============================================================
// Lecture 2, Part 2 (sections 2.4 to 2.7): T5 bias, ALiBi, RoPE.
// Every number is computed from the input, so a twist that
// changes one number always shows correct arithmetic.
// The interactives for these sections live at the bottom and
// only run in the browser.
// ============================================================
(function (root) {
  "use strict";

  const L = (root.Lecture = root.Lecture || {});
  const M = L.math;
  const defs = (L.defs = L.defs || {});
  const f = (x, dp) => M.fmt(x, dp === undefined ? 3 : dp);
  const keyCols = (n) => Array.from({ length: n }, (_, i) => "key " + i);
  const argmax = (xs) => xs.reduce((best, x, i) => (x > xs[best] ? i : best), 0);

  // Fewest decimals (up to 3) that still show the value the way the notes do.
  function g(x) {
    for (let dp = 0; dp < 3; dp++) {
      if (Math.abs(M.round(x, dp) - M.round(x, 3)) < 1e-9) return M.fmt(x, dp);
    }
    return M.fmt(x, 3);
  }
  // Bracket negatives when they sit inside a product: (−0.866) × (−0.5).
  const gp = (x) => (M.round(x, 3) < 0 ? "(" + g(x) + ")" : g(x));
  const vec = (xs) => "[" + xs.map(g).join(", ") + "]";

  /* ---------------- pure helpers for 2.4 to 2.6 (tested) ---------------- */

  // T5's bucket rule for keys on one side of the query (the version its decoder
  // uses). The first half of the buckets hold one distance each; the rest grow on
  // a log scale up to maxDistance; everything further shares the last bucket.
  function t5Bucket(distance, numBuckets, maxDistance) {
    const nb = numBuckets || 32;
    const md = maxDistance || 128;
    const exact = nb / 2;
    const d = Math.max(0, Math.floor(Math.abs(distance)));
    if (d < exact) return d;
    const b = exact + Math.floor((Math.log(d / exact) / Math.log(md / exact)) * (nb - exact));
    return Math.min(nb - 1, b);
  }

  // Smallest and largest distance that share a bucket (Infinity for the last one).
  function t5BucketRange(bucket, numBuckets, maxDistance) {
    const limit = (maxDistance || 128) * 4;
    let lo = -1;
    let hi = -1;
    for (let d = 0; d <= limit; d++) {
      if (t5Bucket(d, numBuckets, maxDistance) === bucket) {
        if (lo < 0) lo = d;
        hi = d;
      }
    }
    return { lo, hi: hi === limit ? Infinity : hi };
  }

  // ALiBi's head slopes: a geometric sequence. For 8 heads, 1/2 down to 1/256.
  const alibiSlopes = (heads) => Array.from({ length: heads }, (_, h) => Math.pow(2, (-8 * (h + 1)) / heads));

  // The speed of pair i: θ_i = base^(−2i/d), counting i from 0 as code does.
  const ropeTheta = (i, d, base) => Math.pow(base === undefined ? 10000 : base, (-2 * i) / d);

  // Score between two identical, evenly spread vectors at this distance, scaled so
  // distance 0 gives 1: the average of cos(distance × θ_i) over the d/2 pairs.
  function ropeDecay(distance, d, base) {
    const pairs = d / 2;
    let s = 0;
    for (let i = 0; i < pairs; i++) s += Math.cos(distance * ropeTheta(i, d, base));
    return s / pairs;
  }

  function findBlock(steps, key) {
    for (const s of steps) for (const b of s.blocks || []) if (b.key === key) return b;
    return null;
  }

  /* ---------------- 2.4 T5 relative bias ---------------- */
  defs["l2-t5"] = {
    id: "l2-t5",
    title: "T5 bias, by hand",
    // beta[b] is the number this head learned for bucket b. Distances 0 to 3 each have their own bucket.
    input: { raw: [2, 1, 3, 2], m: 3, beta: [1.0, 0.5, 0.0, -0.5] },
    setup:
      "A query at position 3 looks back at the keys in positions 0 to 3. Its raw scores, " +
      "q·k / √d<sub>k</sub>, are already worked out. This head has also learned one bias per bucket. " +
      "Small distances each have a bucket of their own, so we can read the biases straight from the table.",
    setupBlocks: (input) => [
      { type: "vector", key: "raw", label: "raw score", values: input.raw, cols: keyCols(input.raw.length) },
      {
        type: "vector",
        key: "beta",
        label: "learned β",
        values: input.beta,
        dp: 1,
        cols: input.beta.map((_, d) => "dist " + d),
        tone: "muted",
      },
    ],
    build(input) {
      const { raw, m, beta } = input;
      const keys = raw.map((_, n) => n);
      const cols = keyCols(raw.length);
      const distances = keys.map((n) => m - n);
      const buckets = distances.map((d) => t5Bucket(d));
      const bias = buckets.map((b) => beta[b]);
      const adjusted = M.add(raw, bias);
      const soft = M.softmax(adjusted);
      const plain = M.softmax(raw);
      const far = 0;
      const near = keys.length - 1;

      // Where does the key at distance 0 end up? (It started tied with the farthest key.)
      const order = keys.slice().sort((a, b) => soft.weights[b] - soft.weights[a]);
      const nearRank = order.indexOf(near);
      const stillTied = Math.abs(soft.weights[near] - soft.weights[far]) < 1e-9;
      const answer = stillTied ? 0 : nearRank === 0 ? 2 : 1;

      const rawRow = { type: "vector", label: "raw score", values: raw, cols };
      const distRow = { type: "vector", key: "distance", label: "distance", values: distances, tone: "muted" };
      const biasRow = { type: "vector", key: "bias", label: "bias", values: bias, dp: 1 };
      const adjRow = { type: "vector", key: "adjusted", label: "new score", values: adjusted, dp: 1, tone: "result" };

      return {
        steps: [
          {
            title: "Measure how far back each key is",
            say:
              `The distance is m − n, the query's position minus the key's. The query is at m = ${m}, ` +
              `so the distances run ${distances.join(", ")}. All of them are small, so each lands in a bucket with the same number.`,
            blocks: [
              Object.assign({}, rawRow, { tone: "muted" }),
              {
                type: "lines",
                fresh: true,
                lines: keys.map((n) => ({ t: `key ${n}:  ${m} − ${n} = ${distances[n]}   → bucket ${buckets[n]}` })),
              },
              Object.assign({}, distRow, { cols, fresh: true }),
            ],
          },
          {
            title: "Look up the learned bias for each bucket",
            say:
              "T5 does not calculate the bias. It looks it up in the small table this head learned during training. " +
              `Key ${far}, the farthest, gets ${g(bias[far])}; key ${near}, the query's own position, gets ${g(bias[near])}.`,
            blocks: [
              distRow,
              {
                type: "lines",
                fresh: true,
                lines: keys.map((n) => ({
                  t: `key ${n}:  β(distance ${distances[n]}) = ${f(bias[n], 1)}`,
                  hl: n === far || n === near,
                })),
              },
              Object.assign({}, biasRow, { cols, fresh: true, hl: [far, near] }),
            ],
          },
          {
            title: "Add the bias to each score",
            say:
              `Key ${near} was tied with key ${far} on a raw score of ${g(raw[near])}. ` +
              `The bias pulls them apart: key ${near} rises to ${g(adjusted[near])} while key ${far} falls to ${g(adjusted[far])}.`,
            blocks: [
              rawRow,
              Object.assign({}, biasRow, { op: "+", hl: keys }),
              Object.assign({}, adjRow, { op: "=", fresh: true, hl: [far, near] }),
            ],
          },
          {
            title: "Raise e to each new score",
            say:
              "Softmax turns each score into e to the power of that score, so bigger scores grow much faster. " +
              `Adding them up gives ${f(soft.sum)}.`,
            blocks: [
              adjRow,
              {
                type: "lines",
                fresh: true,
                lines: adjusted
                  .map((a, n) => ({ t: `e<sup>${f(a, 1)}</sup> = ${f(soft.exps[n])}` }))
                  .concat([{ t: `sum   = ${f(soft.sum)}`, hl: true }]),
              },
              { type: "vector", key: "exps", label: "eˣ", values: soft.exps, dp: 3, fresh: true },
              { type: "vector", key: "sum", label: "sum", values: [soft.sum], dp: 3, tone: "result" },
            ],
          },
          {
            title: "Divide by the total to get attention weights",
            say:
              `Each weight is its e<sup>x</sup> divided by ${f(soft.sum)}. Key ${near}, at distance 0, now takes ` +
              `${f(soft.weights[near])} of the attention, about a third, while key ${far} keeps only ${f(soft.weights[far])}.`,
            predict: {
              ask:
                `Without any bias, key ${near} (distance 0) was tied with key ${far} for second place. ` +
                "Where does it end up once the weights are worked out?",
              choices: ["Still tied with key " + far, "A clear second, with about a third of the attention", "First place, ahead of key 2"],
              answer,
              why:
                `Key ${near} gained ${g(bias[near])} and key ${far} lost ${g(Math.abs(bias[far]))}, so the tie breaks. ` +
                `Key 2 still has the biggest score, ${g(adjusted[2])}, so key ${near} settles in second with ${f(soft.weights[near])}.`,
              hint: "Look at the new scores. Softmax keeps their order, so rank the new scores first.",
            },
            blocks: [
              {
                type: "lines",
                lines: soft.exps.map((e, n) => ({ t: `key ${n}:  ${f(e)} / ${f(soft.sum)} = ${f(soft.weights[n])}`, hl: n === near })),
              },
              { type: "bars", key: "weights", labels: cols, values: soft.weights, hl: [near], fresh: true },
            ],
          },
          {
            title: "Compare with no bias at all",
            say:
              `Without the bias, key ${far} and key ${near} would each get ${f(plain.weights[far])}. ` +
              `With it, key ${far} drops to ${f(soft.weights[far])} and key ${near} climbs to ${f(soft.weights[near])}.`,
            blocks: [
              {
                type: "bars",
                key: "compare",
                label: "with T5 bias",
                ghostLabel: "without bias",
                labels: cols,
                values: soft.weights,
                ghost: plain.weights,
                hl: [far, near],
              },
            ],
          },
        ],
        takeaway:
          "This head learned to prefer nearby tokens. The token at distance 0 jumped from being tied for second place " +
          "to taking a third of the attention. Keep these four weights in mind: you are about to meet them again.",
      };
    },
  };

  /* ---------------- 2.5 ALiBi ---------------- */
  defs["l2-alibi"] = {
    id: "l2-alibi",
    title: "ALiBi, by hand",
    input: { raw: [2, 1, 3, 2], m: 3, mu: 0.5 },
    twist: { label: "Try a gentler slope, μ = 0.1", offLabel: "Back to μ = 0.5", input: { mu: 0.1 } },
    setup:
      "A query at position 3 looks back at the keys in positions 0 to 3. Its raw scores, " +
      "q·k / √d<sub>k</sub>, are already worked out. ALiBi will add one fixed penalty per key, " +
      "and the penalty grows the further back the key sits.",
    setupBlocks: (input) => [
      { type: "vector", key: "raw", label: "raw score", values: input.raw, cols: keyCols(input.raw.length) },
      { type: "vector", label: "distance", values: input.raw.map((_, n) => input.m - n), tone: "muted" },
    ],
    build(input) {
      const { raw, m, mu } = input;
      const keys = raw.map((_, n) => n);
      const bias = M.alibiBias(m, keys, mu);
      const adjusted = M.add(raw, bias);
      const soft = M.softmax(adjusted);
      const plain = M.softmax(raw);
      const win = argmax(soft.weights);
      const far = 0;
      const near = keys.length - 1;
      const cols = keyCols(raw.length);

      const rawRow = { type: "vector", label: "raw score", values: raw, cols };
      const biasRow = { type: "vector", key: "bias", label: "bias", values: bias };
      const adjRow = { type: "vector", key: "adjusted", label: "new score", values: adjusted, tone: "result" };

      return {
        steps: [
          {
            title: "Work out the bias for each key",
            say:
              `The rule is bias = μ × (n − m). The query sits at m = ${m}, so every key behind it gets a negative number, ` +
              "and the further back it is, the more negative. Nothing here is learned. It is plain arithmetic.",
            blocks: [
              {
                type: "lines",
                fresh: true,
                lines: keys.map((n) => ({
                  t: `key ${n}:  ${f(mu, 1)} × (${n} − ${m}) = ${f(bias[n], 1)}`,
                  hl: n === far,
                })),
              },
              Object.assign({}, biasRow, { cols, fresh: true, hl: [far] }),
            ],
          },
          {
            title: "Add the bias to each score",
            say:
              `The farthest key dropped by ${f(Math.abs(bias[far]), 1)}, while the key at the query's own position did not move at all. ` +
              "That is the penalty growing in a straight line with distance.",
            blocks: [
              rawRow,
              Object.assign({}, biasRow, { op: "+", hl: keys }),
              Object.assign({}, adjRow, { op: "=", fresh: true }),
            ],
          },
          {
            title: "Raise e to each new score",
            say:
              "Softmax starts by turning every score into e to the power of that score. Bigger scores grow much faster, " +
              `and then we add them all up: the total is ${f(soft.sum)}.`,
            blocks: [
              adjRow,
              {
                type: "lines",
                fresh: true,
                lines: adjusted
                  .map((a, n) => ({ t: `e<sup>${f(a, 1)}</sup> = ${f(soft.exps[n])}` }))
                  .concat([{ t: `sum   = ${f(soft.sum)}`, hl: true }]),
              },
              { type: "vector", key: "exps", label: "eˣ", values: soft.exps, dp: 3, fresh: true },
            ],
          },
          {
            title: "Divide by the total to get attention weights",
            say:
              `Each weight is its e<sup>x</sup> divided by ${f(soft.sum)}. The four weights add up to 1, ` +
              `and key ${win} takes the largest share: ${f(soft.weights[win])}.`,
            predict: {
              ask: "Before we divide: which key will get the most attention?",
              choices: cols.map((c) => c.replace("key", "Key")),
              answer: win,
              why:
                `Key ${win} had a strong raw score and sits only ${m - win} step${m - win === 1 ? "" : "s"} back, ` +
                "so its penalty is small. Softmax keeps the order of the scores, so the biggest new score wins.",
              hint: "Look at the new scores. Softmax never changes their order, so the biggest new score wins.",
            },
            blocks: [
              {
                type: "lines",
                lines: soft.exps.map((e, n) => ({ t: `key ${n}:  ${f(e)} / ${f(soft.sum)} = ${f(soft.weights[n])}`, hl: n === win })),
              },
              { type: "bars", key: "weights", labels: cols, values: soft.weights, hl: [win], fresh: true },
            ],
          },
          {
            title: "Compare with no ALiBi at all",
            say:
              `Without the penalty, key ${far} would get ${f(plain.weights[far])}. With it, ${f(soft.weights[far])}. ` +
              `The key at the query's own position goes the other way, from ${f(plain.weights[near])} up to ${f(soft.weights[near])}.`,
            blocks: [
              {
                type: "bars",
                key: "compare",
                label: "with ALiBi",
                ghostLabel: "without ALiBi",
                labels: cols,
                values: soft.weights,
                ghost: plain.weights,
                hl: [far, near],
              },
            ],
          },
        ],
        takeaway:
          mu >= 0.5
            ? "Nearby tokens gained attention and distant ones lost it, which matches how language usually works. " +
              "Because the penalty is a straight line, it behaves the same at distance 5,000 as at distance 50. " +
              "That is why a model trained on short inputs with ALiBi copes with much longer ones."
            : `With a slope of only ${f(mu, 1)}, the penalties are tiny and the weights stay close to the raw ones. ` +
              "A small slope makes a long-sighted head that can still look far back. Real models mix steep and gentle slopes across heads.",
      };
    },
  };

  // T5 and ALiBi land on the same weights because every T5 score is ALiBi's plus one.
  function shiftCompare() {
    const t5 = defs["l2-t5"];
    const al = defs["l2-alibi"];
    const t5Steps = t5.build(t5.input).steps;
    const alSteps = al.build(al.input).steps;
    const t5Adj = findBlock(t5Steps, "adjusted").values;
    const alAdj = findBlock(alSteps, "adjusted").values;
    return {
      t5Adj,
      alAdj,
      gaps: t5Adj.map((v, i) => v - alAdj[i]),
      t5W: findBlock(t5Steps, "weights").values,
      alW: findBlock(alSteps, "weights").values,
    };
  }

  /* ---------------- 2.6 RoPE ---------------- */
  const SVG_NS = "http://www.w3.org/2000/svg";
  function el(tag, attrs, parent, text) {
    const e = document.createElementNS(SVG_NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  const rad = (deg) => (deg * Math.PI) / 180;
  const polar = (cx, cy, r, deg) => [cx + r * Math.cos(rad(deg)), cy - r * Math.sin(rad(deg))];

  function drawArrow(g, cx, cy, r, deg, cls, head) {
    const [x, y] = polar(cx, cy, r, deg);
    const hs = head || 9;
    const [bx, by] = polar(cx, cy, r - hs * 1.4, deg);
    const [lx, ly] = polar(bx, by, hs * 0.6, deg + 90);
    const [rx, ry] = polar(bx, by, hs * 0.6, deg - 90);
    el("line", { x1: cx, y1: cy, x2: bx, y2: by, class: cls + " l2p2b-shaft" }, g);
    el("polygon", { points: `${x},${y} ${lx},${ly} ${rx},${ry}`, class: cls + " l2p2b-head" }, g);
  }

  // One dial: a unit circle, ticks for each position, the key arrow, the query
  // arrow and an arc for the angle between them. Angles grow anticlockwise.
  function drawDial(g, o) {
    const { cx, cy, r, m, n, theta } = o;
    el("circle", { cx, cy, r, class: "l2p2b-ring" }, g);
    el("line", { x1: cx - r - 6, y1: cy, x2: cx + r + 6, y2: cy, class: "l2p2b-axisline" }, g);
    el("line", { x1: cx, y1: cy - r - 6, x2: cx, y2: cy + r + 6, class: "l2p2b-axisline" }, g);
    const count = Math.round(360 / theta);
    for (let p = 0; p < count; p++) {
      const a = p * theta;
      const [x1, y1] = polar(cx, cy, r - 4, a);
      const [x2, y2] = polar(cx, cy, r + 4, a);
      el("line", { x1, y1, x2, y2, class: "l2p2b-tick" }, g);
      if (o.labels) {
        const [tx, ty] = polar(cx, cy, r + 17, a);
        el("text", { x: tx, y: ty + 4, class: "l2p2b-ticklabel", "text-anchor": "middle" }, g, String(p));
      }
    }
    const qa = m * theta;
    const ka = n * theta;
    const delta = qa - ka;
    if (delta !== 0) {
      const ar = r * 0.34;
      const [sx, sy] = polar(cx, cy, ar, ka);
      const [ex, ey] = polar(cx, cy, ar, qa);
      const large = Math.abs(delta) > 180 ? 1 : 0;
      const sweep = delta > 0 ? 0 : 1;
      el("path", { d: `M${sx},${sy} A${ar},${ar} 0 ${large} ${sweep} ${ex},${ey}`, class: "l2p2b-arc" }, g);
      if (o.labels) {
        const [lx, ly] = polar(cx, cy, ar + 16, ka + delta / 2);
        el("text", { x: lx, y: ly + 4, class: "l2p2b-arclabel", "text-anchor": "middle" }, g, Math.abs(delta) + "°");
      }
    }
    drawArrow(g, cx, cy, r, ka, "l2p2b-k", o.head);
    drawArrow(g, cx, cy, r, qa, "l2p2b-q", o.head);
    el("circle", { cx, cy, r: 3, class: "l2p2b-pivot" }, g);
    if (o.labels) {
      const [kx, ky] = polar(cx, cy, r * 0.72, ka - 9);
      const [qx, qy] = polar(cx, cy, r * 0.72, qa + 9);
      el("text", { x: kx, y: ky + 4, class: "l2p2b-klabel", "text-anchor": "middle" }, g, "k");
      el("text", { x: qx, y: qy + 4, class: "l2p2b-qlabel", "text-anchor": "middle" }, g, "q");
    }
  }

  function ropePanel(host) {
    host.classList.add("l2p2b-rope-panel");
    const svgEl = el("svg", { viewBox: "0 0 320 300", role: "img" }, host);
    const cap = document.createElement("p");
    cap.className = "l2p2b-panel-cap";
    host.appendChild(cap);
    return {
      render(index, step, input) {
        svgEl.replaceChildren();
        const th = input.theta;
        if (step.rope) {
          const { m, n, name } = step.rope;
          drawDial(el("g", {}, svgEl), { cx: 160, cy: 150, r: 108, m, n, theta: th, labels: true });
          svgEl.setAttribute(
            "aria-label",
            `Case ${name}: the query arrow is turned ${m * th} degrees, the key arrow ${n * th} degrees, ${Math.abs(m - n) * th} degrees apart.`
          );
          cap.innerHTML = `Case ${name}: q at position <b>${m}</b>, k at position <b>${n}</b>. The gap is ${Math.abs(m - n)} × ${th}° = ${Math.abs(m - n) * th}°.`;
        } else if (step.kind === "setup") {
          drawDial(el("g", {}, svgEl), { cx: 160, cy: 150, r: 108, m: 0, n: 0, theta: th, labels: true });
          svgEl.setAttribute("aria-label", "Both arrows point along the x axis before any rotation.");
          cap.innerHTML = `Before any turning, q and k both point along [1, 0]. The numbers round the edge are positions, ${th}° apart.`;
        } else {
          svgEl.setAttribute("viewBox", "0 0 320 150");
          input.cases.forEach((c, i) => {
            const cx = 55 + i * 105;
            drawDial(el("g", {}, svgEl), { cx, cy: 62, r: 44, m: c.m, n: c.n, theta: th, head: 6 });
            el("text", { x: cx, y: 128, class: "l2p2b-minilabel", "text-anchor": "middle" }, svgEl, `${c.name}: m = ${c.m}, n = ${c.n}`);
            el("text", { x: cx, y: 145, class: "l2p2b-minilabel is-soft", "text-anchor": "middle" }, svgEl, `gap ${Math.abs(c.m - c.n) * th}°`);
          });
          svgEl.setAttribute("aria-label", "The three cases side by side. A and B have the same gap; C has a wider one.");
          cap.innerHTML = "A and B sit in different places but open the same 30° gap. C opens 90°.";
          return;
        }
        svgEl.setAttribute("viewBox", "0 0 320 300");
      },
    };
  }

  defs["l2-rope"] = {
    id: "l2-rope",
    title: "RoPE only sees distance",
    input: {
      q: [1, 0],
      k: [1, 0],
      theta: 30,
      cases: [
        { name: "A", m: 2, n: 1 },
        { name: "B", m: 5, n: 4 },
        { name: "C", m: 4, n: 1 },
      ],
    },
    setup:
      "Take the simplest query and key there are: q = [1, 0] and k = [1, 0], two arrows pointing the same way. " +
      "RoPE turns each arrow by θ = 30° for every step of position. We will score three pairs of positions " +
      "and watch what the score really depends on.",
    setupBlocks: (input) => [
      { type: "vector", key: "q", label: "q", values: input.q },
      { type: "vector", key: "k", label: "k", values: input.k },
      {
        type: "lines",
        lines: [
          "x1_new = x1 · cos(α) − x2 · sin(α)",
          "x2_new = x1 · sin(α) + x2 · cos(α)",
          { t: "so rotating [1, 0] by α gives [cos α, sin α]", hl: true },
        ],
      },
    ],
    panel: ropePanel,
    build(input) {
      const th = input.theta;
      const res = input.cases.map((c) => {
        const qa = c.m * th;
        const ka = c.n * th;
        const qr = M.rotate(input.q, qa);
        const kr = M.rotate(input.k, ka);
        const prods = qr.map((v, i) => v * kr[i]);
        return Object.assign({}, c, { qa, ka, qr, kr, prods, score: M.dot(qr, kr), dist: c.m - c.n });
      });
      const [A, B, C] = res;
      const rotLines = (c) => [
        { t: `q turned by ${c.m} × ${th}° = ${c.qa}°  → ${vec(c.qr)}` },
        { t: `k turned by ${c.n} × ${th}° = ${c.ka}°  → ${vec(c.kr)}` },
      ];
      const dotLine = (c) =>
        `${gp(c.qr[0])} × ${gp(c.kr[0])} + ${gp(c.qr[1])} × ${gp(c.kr[1])} = ${g(c.prods[0])} + ${g(c.prods[1])} = ${g(c.score)}`;
      const rotBlocks = (c) => [
        { type: "vector", key: c.name + "-q", label: "q turned", values: c.qr, dp: 3 },
        { type: "vector", key: c.name + "-k", label: "k turned", values: c.kr, dp: 3 },
      ];
      const scoreBlock = (c) => ({ type: "vector", key: c.name + "-score", label: "score", values: [c.score], dp: 3, tone: "result", fresh: true });
      const scoreChoices = ["0.866", "0.5", "0", "−0.866"];
      const pick = (v) => scoreChoices.findIndex((s) => Math.abs(parseFloat(s.replace("−", "-")) - M.round(v, 3)) < 1e-9);

      return {
        steps: [
          {
            title: `Case A: turn the query and the key`,
            rope: A,
            say:
              `The query sits at position ${A.m}, so it turns ${A.qa}°. The key sits at position ${A.n}, so it turns ${A.ka}°. ` +
              "Because both started as [1, 0], each new arrow is just [cos, sin] of its angle.",
            blocks: [{ type: "lines", fresh: true, lines: rotLines(A) }].concat(rotBlocks(A)),
          },
          {
            title: "Case A: take the dot product",
            rope: A,
            say:
              `Multiply matching parts and add. Both halves come to ${g(A.prods[0])}, so the score is ${g(A.score)}, ` +
              `which is cos ${A.qa - A.ka}°: the cosine of the gap between the arrows.`,
            blocks: rotBlocks(A).concat([{ type: "lines", fresh: true, lines: [{ t: dotLine(A), hl: true }] }, scoreBlock(A)]),
          },
          {
            title: "Case B: move both tokens later in the sentence",
            rope: B,
            say:
              `Now the query is at ${B.m} and the key at ${B.n}. Both arrows swing much further round, to ${B.qa}° and ${B.ka}°, ` +
              `but the gap between them is still ${B.qa - B.ka}°. The score comes out at ${g(B.score)} again.`,
            predict: {
              ask: `The pair moves from positions (${A.m}, ${A.n}) to (${B.m}, ${B.n}). The distance is still ${B.dist}. What will the score be?`,
              choices: scoreChoices,
              answer: pick(B.score),
              why:
                `Both arrows turn by an extra ${(B.m - A.m) * th}°, so the angle between them does not change. ` +
                `Same gap, same cosine, same score: ${g(B.score)}.`,
              hint: "Work out how far apart the two arrows are, in degrees. Does moving both of them change that?",
            },
            blocks: [{ type: "lines", fresh: true, lines: rotLines(B).concat([{ t: dotLine(B), hl: true }]) }]
              .concat(rotBlocks(B))
              .concat([scoreBlock(B)]),
          },
          {
            title: "Case C: pull the tokens further apart",
            rope: C,
            say:
              `The query is at ${C.m} and the key at ${C.n}: a distance of ${C.dist}, so a gap of ${C.dist} × ${th}° = ${C.qa - C.ka}°. ` +
              `The two halves of the dot product cancel exactly, and the score is ${g(C.score)}.`,
            predict: {
              ask: `Distance ${C.dist} means a gap of ${C.dist} × ${th}° = ${C.qa - C.ka}°. What will the score be?`,
              choices: scoreChoices,
              answer: pick(C.score),
              why: `The arrows are at right angles, and cos ${C.qa - C.ka}° is ${g(C.score)}. Perpendicular arrows have a dot product of zero.`,
              hint: `The score is the cosine of the gap. What is cos ${C.qa - C.ka}°?`,
            },
            blocks: [{ type: "lines", fresh: true, lines: rotLines(C).concat([{ t: dotLine(C), hl: true }]) }]
              .concat(rotBlocks(C))
              .concat([scoreBlock(C)]),
          },
          {
            title: "Put the three cases side by side",
            ropeAll: true,
            say:
              `A and B sit in different places but share a distance of ${A.dist}, and they share a score of ${g(A.score)}. ` +
              `C is ${C.dist} apart and scores ${g(C.score)}. The positions themselves never mattered, only the gap.`,
            blocks: [
              {
                type: "matrix",
                key: "summary",
                colLabels: ["m", "n", "m − n", "gap", "score"],
                rowLabels: res.map((c) => "case " + c.name),
                // Positions are strings so only the scores get three decimals.
                rows: res.map((c) => [String(c.m), String(c.n), String(c.dist), c.dist * th + "°", M.round(c.score, 3)]),
                hl: [[0, 4], [1, 4]],
                dp: 3,
              },
              { type: "note", html: "score = cos(distance × θ)" },
            ],
          },
        ],
        takeaway:
          `The score is cos(distance × θ). Distance ${A.dist} gives cos ${A.dist * th}° = ${g(A.score)}, and distance ${C.dist} gives ` +
          `cos ${C.dist * th}° = ${g(C.score)}. It depends only on how far apart the tokens are, not where they sit in the sentence. ` +
          "That is exactly the property language needs.",
      };
    },
  };

  L.l2p2 = { t5Bucket, t5BucketRange, alibiSlopes, ropeTheta, ropeDecay, shiftCompare };

  if (typeof module !== "undefined" && module.exports) module.exports = defs;
  if (typeof document === "undefined") return;

  /* ============================================================
     Browser only: the interactives for 2.4 to 2.6.
     Colours come from CSS classes that read the theme tokens,
     so SVG redraws are never needed on a theme change.
     ============================================================ */
  const $ = (sel, ctx) => (ctx || document).querySelector(sel);
  const fmtInt = M.fmtInt;
  const FRAC = (h) => "1/" + Math.pow(2, h + 1);

  function pressChip(group, btn) {
    group.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", String(c === btn)));
  }

  function chip(label, data) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.setAttribute("aria-pressed", "false");
    b.innerHTML = label;
    Object.assign(b.dataset, data || {});
    return b;
  }

  /* ---------- 2.4 bucket staircase ---------- */
  function initBuckets(fig) {
    const inA = $("#l2p2b-bk-a", fig);
    const inB = $("#l2p2b-bk-b", fig);
    const outA = $("#l2p2b-bk-a-out", fig);
    const outB = $("#l2p2b-bk-b-out", fig);
    const presets = $(".l2p2b-bk-presets", fig);
    const readout = $(".ix-readout", fig);
    const svgEl = $(".l2p2b-bk-svg", fig);
    const MAXD = 300;
    const X0 = 40;
    const X1 = 390;
    const Y0 = 16;
    const Y1 = 196;
    const sx = (d) => X0 + (Math.sqrt(d) / Math.sqrt(MAXD)) * (X1 - X0);
    const sy = (b) => Y1 - (b / 31) * (Y1 - Y0);

    // Static layer: bands, axes, the staircase itself.
    const stat = el("g", {}, svgEl);
    el("rect", { x: sx(0), y: Y0 - 8, width: sx(16) - sx(0), height: Y1 - Y0 + 8, class: "l2p2b-band is-exact" }, stat);
    el("rect", { x: sx(113), y: Y0 - 8, width: sx(MAXD) - sx(113), height: Y1 - Y0 + 8, class: "l2p2b-band is-last" }, stat);
    el("line", { x1: X0, y1: Y1, x2: X1, y2: Y1, class: "l2p2b-axis" }, stat);
    el("line", { x1: X0, y1: Y0 - 8, x2: X0, y2: Y1, class: "l2p2b-axis" }, stat);
    [0, 4, 16, 32, 64, 128, 200, 300].forEach((d) => {
      el("line", { x1: sx(d), y1: Y1, x2: sx(d), y2: Y1 + 4, class: "l2p2b-axis" }, stat);
      el("text", { x: sx(d), y: Y1 + 16, "text-anchor": "middle", class: "l2p2b-ticklabel" }, stat, String(d));
    });
    [0, 8, 16, 24, 31].forEach((b) => {
      el("text", { x: X0 - 6, y: sy(b) + 4, "text-anchor": "end", class: "l2p2b-ticklabel" }, stat, String(b));
    });
    el("text", { x: (X0 + X1) / 2, y: Y1 + 32, "text-anchor": "middle", class: "l2p2b-axislabel" }, stat, "distance (stretched near 0)");
    el("text", { x: 12, y: (Y0 + Y1) / 2, "text-anchor": "middle", class: "l2p2b-axislabel", transform: `rotate(-90 12 ${(Y0 + Y1) / 2})` }, stat, "bucket");
    const steps = el("g", {}, stat);
    for (let b = 0; b < 32; b++) {
      const r = t5BucketRange(b);
      const hi = r.hi === Infinity ? MAXD : r.hi;
      el("line", { x1: sx(r.lo), y1: sy(b), x2: sx(Math.min(MAXD, hi + 1)), y2: sy(b), class: "l2p2b-step", "data-b": b }, steps);
    }
    const dyn = el("g", {}, svgEl);

    function describe(d) {
      const b = t5Bucket(d);
      const r = t5BucketRange(b);
      if (r.lo === r.hi) return { b, text: `distance <b>${d}</b> has bucket <b>${b}</b> all to itself` };
      if (r.hi === Infinity) return { b, text: `distance <b>${d}</b> lands in the last bucket, <b>${b}</b>, shared by every distance from ${r.lo} upwards` };
      return { b, text: `distance <b>${d}</b> lands in bucket <b>${b}</b>, shared by distances ${r.lo} to ${r.hi}` };
    }

    function update() {
      const a = +inA.value;
      const b = +inB.value;
      outA.textContent = a;
      outB.textContent = b;
      const da = describe(a);
      const db = describe(b);
      steps.querySelectorAll(".l2p2b-step").forEach((s) => {
        const k = +s.dataset.b;
        s.classList.toggle("is-a", k === da.b);
        s.classList.toggle("is-b", k === db.b && k !== da.b);
      });
      dyn.replaceChildren();
      [[a, "is-a", "A"], [b, "is-b", "B"]].forEach(([d, cls, name]) => {
        const x = sx(d);
        const y = sy(t5Bucket(d));
        el("line", { x1: x, y1: Y1, x2: x, y2: y, class: "l2p2b-marker " + cls }, dyn);
        el("circle", { cx: x, cy: y, r: 4.5, class: "l2p2b-dot " + cls }, dyn);
        el("text", { x: x + (name === "A" ? -7 : 7), y: y - 8, "text-anchor": name === "A" ? "end" : "start", class: "l2p2b-marklabel " + cls }, dyn, name);
      });
      const verdict =
        da.b === db.b
          ? "Same bucket, so this head gives both distances exactly the same bias. It cannot tell them apart."
          : "Different buckets, so this head can give the two distances different biases.";
      readout.innerHTML = `A: ${da.text}. B: ${db.text}. ${verdict}`;
      svgEl.setAttribute("aria-label", `Staircase of T5 buckets. Distance ${a} is in bucket ${da.b}; distance ${b} is in bucket ${db.b}.`);
    }

    [[1, 2], [90, 95], [150, 300]].forEach(([a, b]) => {
      const btn = chip(`${a} and ${b}`, { a, b });
      btn.addEventListener("click", () => {
        inA.value = a;
        inB.value = b;
        pressChip(presets, btn);
        update();
      });
      presets.appendChild(btn);
    });
    [inA, inB].forEach((i) =>
      i.addEventListener("input", () => {
        pressChip(presets, null);
        update();
      })
    );
    pressChip(presets, presets.querySelector(".chip"));
    update();
  }

  /* ---------- 2.5 ALiBi slope explorer ---------- */
  function initSlopes(fig) {
    const heads = alibiSlopes(8);
    const chipsEl = $(".l2p2b-sl-heads", fig);
    const inD = $("#l2p2b-sl-d", fig);
    const outD = $("#l2p2b-sl-out", fig);
    const svgEl = $(".l2p2b-sl-svg", fig);
    const keep = $(".l2p2b-keep", fig);
    const readout = $(".ix-readout", fig);
    const MAXD = 64;
    const MAXP = 8;
    const X0 = 44;
    const X1 = 388;
    const Y0 = 14;
    const Y1 = 188;
    const sx = (d) => X0 + (d / MAXD) * (X1 - X0);
    const sy = (p) => Y0 + (p / MAXP) * (Y1 - Y0); // p is the size of the penalty
    let sel = 0;

    const stat = el("g", {}, svgEl);
    [0, 2, 4, 6, 8].forEach((p) => {
      el("line", { x1: X0, y1: sy(p), x2: X1, y2: sy(p), class: "l2p2b-grid" }, stat);
      el("text", { x: X0 - 6, y: sy(p) + 4, "text-anchor": "end", class: "l2p2b-ticklabel" }, stat, p === 0 ? "0" : "−" + p);
    });
    [0, 16, 32, 48, 64].forEach((d) => {
      el("text", { x: sx(d), y: Y1 + 16, "text-anchor": "middle", class: "l2p2b-ticklabel" }, stat, String(d));
    });
    el("line", { x1: X0, y1: Y0, x2: X0, y2: Y1, class: "l2p2b-axis" }, stat);
    el("text", { x: (X0 + X1) / 2, y: Y1 + 32, "text-anchor": "middle", class: "l2p2b-axislabel" }, stat, "distance back from the query");
    el("text", { x: 12, y: (Y0 + Y1) / 2, "text-anchor": "middle", class: "l2p2b-axislabel", transform: `rotate(-90 12 ${(Y0 + Y1) / 2})` }, stat, "bias added");
    const dyn = el("g", {}, svgEl);

    heads.forEach((mu, h) => {
      const b = chip(`Head ${h + 1} <span class="l2p2b-frac">${FRAC(h)}</span>`, { h });
      b.setAttribute("aria-label", `Head ${h + 1}, slope ${FRAC(h)}`);
      b.addEventListener("click", () => {
        sel = h;
        pressChip(chipsEl, b);
        update();
      });
      chipsEl.appendChild(b);
    });

    // One row per head; rebuilt values only.
    const rows = heads.map((mu, h) => {
      const row = document.createElement("div");
      row.className = "l2p2b-keep-row";
      row.innerHTML =
        `<span class="l2p2b-keep-name">Head ${h + 1}</span>` +
        `<span class="l2p2b-keep-track"><span class="l2p2b-keep-fill"></span></span>` +
        `<span class="l2p2b-keep-val"></span>`;
      keep.appendChild(row);
      return row;
    });

    function update() {
      const d = +inD.value;
      outD.textContent = d;
      dyn.replaceChildren();
      heads.forEach((mu, h) => {
        if (h === sel) return;
        const dEnd = Math.min(MAXD, MAXP / mu);
        el("line", { x1: sx(0), y1: sy(0), x2: sx(dEnd), y2: sy(mu * dEnd), class: "l2p2b-slope" }, dyn);
      });
      const mu = heads[sel];
      const dEnd = Math.min(MAXD, MAXP / mu);
      el("line", { x1: sx(d), y1: Y0, x2: sx(d), y2: Y1, class: "l2p2b-marker is-a" }, dyn);
      el("line", { x1: sx(0), y1: sy(0), x2: sx(dEnd), y2: sy(mu * dEnd), class: "l2p2b-slope is-sel" }, dyn);
      const endX = sx(dEnd);
      el(
        "text",
        { x: Math.min(endX, X1 - 4), y: sy(mu * dEnd) + (mu * dEnd >= MAXP ? -6 : -8), "text-anchor": "end", class: "l2p2b-marklabel is-a" },
        dyn,
        `head ${sel + 1}`
      );
      if (mu * d <= MAXP) el("circle", { cx: sx(d), cy: sy(mu * d), r: 4.5, class: "l2p2b-dot is-a" }, dyn);

      rows.forEach((row, h) => {
        const kept = Math.exp(-heads[h] * d);
        row.classList.toggle("is-sel", h === sel);
        row.querySelector(".l2p2b-keep-fill").style.width = kept * 100 + "%";
        row.querySelector(".l2p2b-keep-val").textContent = f(kept);
      });
      const pen = mu * d;
      const last = heads[heads.length - 1];
      readout.innerHTML =
        `Head ${sel + 1} has slope ${FRAC(sel)}. At distance ${d} its bias is <b>${pen === 0 ? "0" : "−" + g(pen)}</b>, ` +
        `which shrinks that key's e<sup>x</sup> to <b>${f(Math.exp(-pen))}</b> of what it would have been. ` +
        `Head 8, the gentlest, keeps ${f(Math.exp(-last * d))} at the same distance.`;
      svgEl.setAttribute("aria-label", `Penalty against distance for eight heads. Head ${sel + 1} is highlighted; at distance ${d} its bias is ${pen === 0 ? "0" : "minus " + g(pen)}.`);
    }

    inD.addEventListener("input", update);
    pressChip(chipsEl, chipsEl.querySelector(".chip"));
    update();
  }

  /* ---------- 2.5 quick check: T5 and ALiBi agree ---------- */
  function initShiftCheck(box) {
    const c = shiftCompare();
    const list = (xs) => "[" + xs.map((x) => f(x, 1)).join(", ") + "]";
    const same = c.t5W.every((w, i) => Math.abs(w - c.alW[i]) < 1e-9);
    const gap = c.gaps[0];
    const choices = [
      "ALiBi's weights are flatter, because its scores are smaller",
      "They are exactly the same",
      "ALiBi gives the far key more attention",
    ];
    const answer = same ? 1 : 0;
    box.innerHTML =
      `<p class="l2p2b-check-ask"><b>Before you read on.</b> T5 turned the raw scores into ${list(c.t5Adj)}. ` +
      `ALiBi turned them into ${list(c.alAdj)}. How will the two sets of attention weights compare?</p>` +
      `<div class="l2p2b-check-choices"></div><p class="l2p2b-check-feedback" aria-live="polite"></p>`;
    const wrap = $(".l2p2b-check-choices", box);
    const fb = $(".l2p2b-check-feedback", box);
    choices.forEach((text, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "l2p2b-check-choice";
      b.textContent = text;
      b.addEventListener("click", () => {
        if (i === answer) {
          b.classList.add("is-right");
          wrap.querySelectorAll("button").forEach((x) => (x.disabled = true));
          fb.className = "l2p2b-check-feedback is-right";
          fb.innerHTML =
            `<b>Yes.</b> Every T5 score is exactly ${g(gap)} higher than its ALiBi partner. Adding the same number to every score ` +
            `multiplies every e<sup>x</sup> by the same amount, on the top and the bottom of the fraction, so it cancels. ` +
            `Both give ${"[" + c.alW.map((w) => f(w)).join(", ") + "]"}.`;
        } else {
          b.classList.add("is-wrong");
          b.disabled = true;
          fb.className = "l2p2b-check-feedback is-wrong";
          fb.innerHTML = "<b>Not quite.</b> Subtract each ALiBi score from its T5 partner. What do you notice about the gaps?";
        }
      });
      wrap.appendChild(b);
    });
  }

  /* ---------- 2.6 RoPE dial ---------- */
  function initDial(fig) {
    const TH = 30;
    const COUNT = 360 / TH;
    const inM = $("#l2p2b-dl-m", fig);
    const inN = $("#l2p2b-dl-n", fig);
    const outM = $("#l2p2b-dl-m-out", fig);
    const outN = $("#l2p2b-dl-n-out", fig);
    const later = $(".l2p2b-dl-later", fig);
    const earlier = $(".l2p2b-dl-earlier", fig);
    const svgEl = $(".l2p2b-dl-svg", fig);
    const readout = $(".ix-readout", fig);
    const logEl = $(".l2p2b-log", fig);
    const CX = 160;
    const CY = 150;
    const R = 108;
    const layer = el("g", {}, svgEl);
    const hK = el("circle", { r: 20, class: "l2p2b-handle", "data-arrow": "n" }, svgEl);
    const hQ = el("circle", { r: 20, class: "l2p2b-handle", "data-arrow": "m" }, svgEl);
    const tried = [];

    function place(h, pos) {
      const [x, y] = polar(CX, CY, R, pos * TH);
      h.setAttribute("cx", x);
      h.setAttribute("cy", y);
    }

    function update(record) {
      const m = +inM.value;
      const n = +inN.value;
      outM.textContent = m;
      outN.textContent = n;
      layer.replaceChildren();
      drawDial(layer, { cx: CX, cy: CY, r: R, m, n, theta: TH, labels: true });
      place(hK, n);
      place(hQ, m);
      const dist = m - n;
      const score = Math.cos(rad(dist * TH));
      readout.innerHTML =
        `q at position ${m} is turned ${m} × ${TH}° = ${m * TH}°. k at position ${n} is turned ${n * TH}°. ` +
        `The distance m − n is ${g(dist)}, so the gap is ${g(dist * TH)}° and the score is cos(${g(dist * TH)}°) = <b>${g(score)}</b>.`;
      svgEl.setAttribute("aria-label", `Query arrow at position ${m}, key arrow at position ${n}. Score ${g(score)}.`);
      later.disabled = m >= COUNT - 1 || n >= COUNT - 1;
      earlier.disabled = m <= 0 || n <= 0;
      if (record) {
        const last = tried[0];
        if (!last || last.m !== m || last.n !== n) tried.unshift({ m, n, dist, score });
        if (tried.length > 6) tried.pop();
      }
      logEl.replaceChildren(
        ...tried.map((t) => {
          const li = document.createElement("li");
          const match = Math.abs(Math.abs(t.dist) - Math.abs(dist)) < 1e-9 && tried.length > 1;
          li.className = match ? "is-same" : "";
          li.innerHTML = `<span>m = ${t.m}, n = ${t.n}</span><span>distance ${g(t.dist)}</span><span>score ${g(t.score)}</span>`;
          return li;
        })
      );
    }

    // Record a pair once the learner lets go, not on every pixel of a drag.
    [inM, inN].forEach((i) => {
      i.addEventListener("input", () => update(false));
      i.addEventListener("change", () => update(true));
    });
    later.addEventListener("click", () => {
      inM.value = +inM.value + 1;
      inN.value = +inN.value + 1;
      update(true);
    });
    earlier.addEventListener("click", () => {
      inM.value = +inM.value - 1;
      inN.value = +inN.value - 1;
      update(true);
    });

    let dragging = null;
    function posFromEvent(e) {
      const pt = svgEl.createSVGPoint();
      pt.x = e.clientX;
      pt.y = e.clientY;
      const p = pt.matrixTransform(svgEl.getScreenCTM().inverse());
      let deg = (Math.atan2(CY - p.y, p.x - CX) * 180) / Math.PI;
      if (deg < 0) deg += 360;
      return Math.round(deg / TH) % COUNT;
    }
    [hK, hQ].forEach((h) => {
      h.addEventListener("pointerdown", (e) => {
        dragging = h.dataset.arrow === "m" ? inM : inN;
        h.setPointerCapture(e.pointerId);
        e.preventDefault();
      });
      h.addEventListener("pointermove", (e) => {
        if (!dragging) return;
        const p = posFromEvent(e);
        if (p !== +dragging.value) {
          dragging.value = p;
          update(false);
        }
      });
      const end = () => {
        if (!dragging) return;
        dragging = null;
        update(true);
      };
      h.addEventListener("pointerup", end);
      h.addEventListener("pointercancel", end);
    });
    update(true);
  }

  /* ---------- 2.6 block-diagonal rotation matrix ---------- */
  function initBlocks(fig) {
    const D = 8;
    const grid = $(".l2p2b-bd-grid", fig);
    const mode = $(".l2p2b-bd-mode", fig);
    const inM = $("#l2p2b-bd-m", fig);
    const outM = $("#l2p2b-bd-m-out", fig);
    const readout = $(".ix-readout", fig);
    const thetas = Array.from({ length: D / 2 }, (_, i) => ropeTheta(i, D, 10000));
    let numbers = false;
    const SUB = ["₁", "₂", "₃", "₄"];

    const cells = [];
    for (let r = 0; r < D; r++) {
      for (let c = 0; c < D; c++) {
        const cell = document.createElement("span");
        const pr = Math.floor(r / 2);
        const pc = Math.floor(c / 2);
        const inBlock = pr === pc;
        cell.className = "l2p2b-bd-cell" + (inBlock ? " is-block is-pair" + (pr % 2) : " is-zero");
        if (inBlock) {
          cell.classList.add(r % 2 === 0 ? "is-top" : "is-bottom");
          cell.classList.add(c % 2 === 0 ? "is-left" : "is-right");
        }
        grid.appendChild(cell);
        cells.push({ cell, r, c, pr, inBlock });
      }
    }

    function update() {
      const m = +inM.value;
      outM.textContent = m;
      cells.forEach(({ cell, r, c, pr, inBlock }) => {
        if (!inBlock) {
          cell.textContent = "0";
          return;
        }
        const top = r % 2 === 0;
        const left = c % 2 === 0;
        const isCos = top === left;
        const sign = top && !left ? -1 : 1;
        if (numbers) {
          const a = m * thetas[pr];
          const v = isCos ? Math.cos(a) : sign * Math.sin(a);
          cell.textContent = M.fmt(v, 2);
        } else {
          cell.textContent = (sign < 0 ? "−" : "") + (isCos ? "c" : "s") + SUB[pr];
        }
      });
      readout.innerHTML = thetas
        .map((t, i) => `Pair ${i + 1}: θ = ${g(t)}, so position ${m} turns it by ${g(m * t)} radians.`)
        .join("<br>");
    }

    mode.querySelectorAll(".chip").forEach((b) =>
      b.addEventListener("click", () => {
        numbers = b.dataset.mode === "numbers";
        pressChip(mode, b);
        update();
      })
    );
    inM.addEventListener("input", update);
    update();
  }

  /* ---------- 2.6 long-term decay ---------- */
  function initDecay(fig) {
    const D = 128;
    const BASES = [10000, 500000];
    const baseChips = $(".l2p2b-dc-base", fig);
    const rangeChips = $(".l2p2b-dc-range", fig);
    const inD = $("#l2p2b-dc-d", fig);
    const outD = $("#l2p2b-dc-d-out", fig);
    const svgEl = $(".l2p2b-dc-svg", fig);
    const readout = $(".ix-readout", fig);
    const X0 = 44;
    const X1 = 388;
    const Y0 = 12;
    const Y1 = 188;
    const VMIN = -0.25;
    let base = BASES[0];
    let logScale = false;
    const cache = {};

    const sy = (v) => Y0 + ((1 - v) / (1 - VMIN)) * (Y1 - Y0);
    const sx = (dist) => (logScale ? X0 + (Math.log10(Math.max(1, dist)) / 5) * (X1 - X0) : X0 + (dist / 256) * (X1 - X0));
    const distFromSlider = () => (logScale ? Math.round(Math.pow(10, +inD.value / 100)) : +inD.value);

    function samples() {
      const k = (logScale ? "log" : "lin") + base;
      if (cache[k]) return cache[k];
      const pts = [];
      if (logScale) {
        let prev = -1;
        for (let t = 0; t <= 500; t++) {
          const dist = Math.round(Math.pow(10, t / 100));
          if (dist === prev) continue;
          prev = dist;
          pts.push([dist, ropeDecay(dist, D, base)]);
        }
      } else {
        for (let dist = 0; dist <= 256; dist++) pts.push([dist, ropeDecay(dist, D, base)]);
      }
      return (cache[k] = pts);
    }

    function curve(b, cls, parent) {
      const keep = base;
      base = b;
      const pts = samples();
      base = keep;
      el("polyline", { points: pts.map(([d, v]) => `${sx(d).toFixed(1)},${sy(v).toFixed(1)}`).join(" "), class: cls }, parent);
    }

    function update() {
      svgEl.replaceChildren();
      const g0 = el("g", {}, svgEl);
      [1, 0.5, 0].forEach((v) => {
        el("line", { x1: X0, y1: sy(v), x2: X1, y2: sy(v), class: v === 0 ? "l2p2b-axis" : "l2p2b-grid" }, g0);
        el("text", { x: X0 - 6, y: sy(v) + 4, "text-anchor": "end", class: "l2p2b-ticklabel" }, g0, String(v));
      });
      const ticks = logScale ? [[1, "1"], [10, "10"], [100, "100"], [1000, "1k"], [10000, "10k"], [100000, "100k"]] : [0, 64, 128, 192, 256].map((d) => [d, String(d)]);
      ticks.forEach(([d, t]) => el("text", { x: sx(d), y: Y1 + 16, "text-anchor": "middle", class: "l2p2b-ticklabel" }, g0, t));
      el("text", { x: (X0 + X1) / 2, y: Y1 + 32, "text-anchor": "middle", class: "l2p2b-axislabel" }, g0, logScale ? "distance |m − n| (log scale)" : "distance |m − n|");
      const other = BASES.find((b) => b !== base);
      curve(other, "l2p2b-curve is-ghost", svgEl);
      curve(base, "l2p2b-curve", svgEl);
      const dist = distFromSlider();
      outD.textContent = fmtInt(dist);
      const v = ropeDecay(dist, D, base);
      const vo = ropeDecay(dist, D, other);
      el("line", { x1: sx(dist), y1: Y0, x2: sx(dist), y2: Y1, class: "l2p2b-marker is-a" }, svgEl);
      el("circle", { cx: sx(dist), cy: sy(v), r: 4.5, class: "l2p2b-dot is-a" }, svgEl);
      readout.innerHTML =
        `At distance ${fmtInt(dist)}, base ${fmtInt(base)} gives <b>${f(v)}</b>, ` +
        `and base ${fmtInt(other)} gives ${f(vo)}. ` +
        (base > other
          ? "The bigger base spins every hand more slowly, so the pairs stay in step for longer and the score fades later."
          : "With the original base, the pairs fall out of step sooner, so the score fades faster.");
      svgEl.setAttribute("aria-label", `Average RoPE score against distance for base ${fmtInt(base)}. At distance ${fmtInt(dist)} it is ${f(v)}.`);
    }

    baseChips.querySelectorAll(".chip").forEach((b) =>
      b.addEventListener("click", () => {
        base = +b.dataset.base;
        pressChip(baseChips, b);
        update();
      })
    );
    rangeChips.querySelectorAll(".chip").forEach((b) =>
      b.addEventListener("click", () => {
        const wasLog = logScale;
        logScale = b.dataset.range === "log";
        pressChip(rangeChips, b);
        if (logScale !== wasLog) {
          const dist = distFromSliderFor(wasLog);
          if (logScale) {
            inD.min = 0;
            inD.max = 500;
            inD.value = Math.round(Math.log10(Math.max(1, dist)) * 100);
          } else {
            inD.min = 0;
            inD.max = 256;
            inD.value = Math.min(256, dist);
          }
        }
        update();
      })
    );
    function distFromSliderFor(wasLog) {
      return wasLog ? Math.round(Math.pow(10, +inD.value / 100)) : +inD.value;
    }
    inD.addEventListener("input", update);
    update();
  }

  function initAll() {
    const map = [
      ["#l2p2b-buckets", initBuckets],
      ["#l2p2b-slopes", initSlopes],
      ["#l2p2b-shift", initShiftCheck],
      ["#l2p2b-dial", initDial],
      ["#l2p2b-blockdiag", initBlocks],
      ["#l2p2b-decay", initDecay],
    ];
    map.forEach(([sel, init]) => {
      const node = document.querySelector(sel);
      if (!node) return;
      try {
        init(node);
      } catch (err) {
        console.warn("Could not start the interactive", sel, err);
      }
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initAll);
  else initAll();
})(typeof window !== "undefined" ? window : globalThis);
