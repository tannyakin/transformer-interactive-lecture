// ============================================================
// Pure maths helpers shared by every walkthrough. No DOM here,
// so Node can test each worked example against the notes.
// ============================================================
(function (root) {
  "use strict";

  function round(x, dp) {
    const d = dp === undefined ? 3 : dp;
    const f = Math.pow(10, d);
    // The tiny nudge stops values like 1.0005 rounding down because of binary floats.
    const r = Math.round((x + Math.sign(x) * Number.EPSILON) * f) / f;
    return Object.is(r, -0) ? 0 : r;
  }

  // Display a number the way a textbook would: true minus sign, never "-0.000".
  function fmt(x, dp) {
    if (x === -Infinity) return "−∞";
    if (x === Infinity) return "∞";
    const d = dp === undefined ? 3 : dp;
    const r = round(x, d);
    const s = Math.abs(r).toFixed(d);
    return r < 0 ? "−" + s : s;
  }

  function fmtInt(n) {
    return Math.round(n).toLocaleString("en-US");
  }

  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  const add = (a, b) => a.map((v, i) => v + b[i]);
  const scale = (a, k) => a.map((v) => v * k);
  const relu = (v) => v.map((x) => Math.max(0, x));

  // Row vector times matrix: result[j] = sum over i of x[i] * W[i][j].
  function vecMat(x, W) {
    return W[0].map((_, j) => x.reduce((s, xi, i) => s + xi * W[i][j], 0));
  }

  // Deliberately not shifted by the max: the notes show the raw e^x values.
  function softmax(scores) {
    const exps = scores.map((s) => Math.exp(s));
    const sum = exps.reduce((a, b) => a + b, 0);
    return { exps, sum, weights: exps.map((e) => e / sum) };
  }

  function sinusoidalPE(pos, d, base) {
    const b = base === undefined ? 10000 : base;
    const out = new Array(d);
    for (let i = 0; i < d / 2; i++) {
      const angle = pos / Math.pow(b, (2 * i) / d);
      out[2 * i] = Math.sin(angle);
      out[2 * i + 1] = Math.cos(angle);
    }
    return out;
  }

  function rotate(v, deg) {
    const a = (deg * Math.PI) / 180;
    return [v[0] * Math.cos(a) - v[1] * Math.sin(a), v[0] * Math.sin(a) + v[1] * Math.cos(a)];
  }

  const alibiBias = (m, keys, mu) => keys.map((n) => mu * (n - m));

  function layerNorm(x, eps) {
    const e = eps === undefined ? 1e-5 : eps;
    const mean = x.reduce((a, b) => a + b, 0) / x.length;
    const centered = x.map((v) => v - mean);
    const variance = centered.reduce((a, c) => a + c * c, 0) / x.length;
    const std = Math.sqrt(variance + e);
    return { mean, centered, variance, std, normalized: centered.map((c) => c / std) };
  }

  function rmsNorm(x, eps) {
    const e = eps === undefined ? 0 : eps;
    const squares = x.map((v) => v * v);
    const meanSquare = squares.reduce((a, b) => a + b, 0) / x.length;
    const rms = Math.sqrt(meanSquare + e);
    return { squares, meanSquare, rms, normalized: x.map((v) => v / rms) };
  }

  function ffn(x, W1, W2) {
    const hidden = vecMat(x, W1);
    const activated = relu(hidden);
    return { hidden, activated, out: vecMat(activated, W2) };
  }

  // A class the teacher gives zero probability contributes nothing.
  function klDivergence(p, q) {
    const terms = p.map((pi, i) => (pi === 0 ? 0 : pi * Math.log(pi / q[i])));
    return { terms, total: terms.reduce((a, b) => a + b, 0) };
  }

  function choose(n, k) {
    let r = 1;
    for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i;
    return Math.round(r);
  }

  // The leading 2 is because both keys and values are cached.
  function kvCacheBytes(o) {
    const tokens = o.tokens === undefined ? 1 : o.tokens;
    const bytes = o.bytes === undefined ? 2 : o.bytes;
    return 2 * o.layers * o.kvHeads * o.headDim * tokens * bytes;
  }

  const api = {
    round, fmt, fmtInt, dot, add, scale, relu, vecMat, softmax, sinusoidalPE, rotate,
    alibiBias, layerNorm, rmsNorm, ffn, klDivergence, choose, kvCacheBytes,
  };

  root.Lecture = root.Lecture || {};
  root.Lecture.math = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
