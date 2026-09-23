// ============================================================
// Lecture 2, Part 2: positional encoding walkthroughs.
// Every number is computed from the input, so a twist that
// changes one number always shows correct arithmetic.
// ============================================================
(function (root) {
  "use strict";

  const L = (root.Lecture = root.Lecture || {});
  const M = L.math;
  const defs = (L.defs = L.defs || {});
  const f = (x, dp) => M.fmt(x, dp === undefined ? 3 : dp);
  const keyCols = (n) => Array.from({ length: n }, (_, i) => "key " + i);
  const argmax = (xs) => xs.reduce((best, x, i) => (x > xs[best] ? i : best), 0);

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

  if (typeof module !== "undefined" && module.exports) module.exports = defs;
})(typeof window !== "undefined" ? window : globalThis);
