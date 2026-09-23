// ============================================================
// Lecture 2, Part 5: what attention costs.
// One walkthrough (the 20 GB score matrix) and three
// interactives: the growing n x n grid, the 6d crossover chart,
// and a FlashAttention tiling sketch. Every number shown is
// computed here from the inputs, never typed by hand.
// ============================================================
(function (root) {
  "use strict";

  const L = (root.Lecture = root.Lecture || {});
  const M = L.math;
  const defs = (L.defs = L.defs || {});
  const n0 = (x) => M.fmtInt(x);

  /* ---------------- pure helpers (tested in Node) ---------------- */
  const P5 = {
    // Every token scores every token, including itself.
    scores: (n) => n * n,
    // Q times K transpose, and scores times V: about 2 n^2 d operations each.
    attnOps: (n, d) => ({ qk: 2 * n * n * d, sv: 2 * n * n * d }),
    matrixBytes: (n, bytes) => n * n * bytes,
    // Per token, per layer.
    ffnPerToken: (d) => 24 * d * d,
    attnPerToken: (n, d) => 4 * n * d,
    // 4 n d = 24 d^2  gives  n = 6 d.
    crossover: (d) => 6 * d,
  };
  L.l2p5 = P5;

  /* ---------------- 5.2 the 20 GB score matrix ---------------- */
  const GROWTH_CHOICES = [2, 4, 8, 16];

  defs["l2-attn-memory"] = {
    id: "l2-attn-memory",
    title: "How much memory one score matrix needs",
    input: { n: 100000, bytes: 2 },
    setup:
      "Take a long document of n = 100,000 tokens. Attention builds an n × n matrix of scores, " +
      "and each score is stored in 16-bit precision, which is 2 bytes. " +
      "How much memory does that one matrix take, for one head in one layer?",
    setupBlocks: (input) => [
      {
        type: "lines",
        lines: [
          { t: `sequence length   n = ${n0(input.n)} tokens` },
          { t: `one score         16 bits = ${input.bytes} bytes` },
        ],
      },
    ],
    panel: (el) => memoryPanel(el),
    build(input) {
      const { n, bytes } = input;
      const scores = P5.scores(n);
      const total = P5.matrixBytes(n, bytes);
      const gb = total / 1e9;
      const n2 = 2 * n;
      const scores2 = P5.scores(n2);
      const growth = scores2 / scores;
      const gb2 = P5.matrixBytes(n2, bytes) / 1e9;
      const nSmall = n / 10;
      const scoresSmall = P5.scores(nSmall);
      const tenfold = scores / scoresSmall;

      const scoreLine = { t: `${n0(n)} × ${n0(n)} = ${n0(scores)} scores`, hl: true };
      const bytesLine = { t: `${n0(scores)} × ${bytes} bytes = ${n0(total)} bytes`, hl: true };

      return {
        steps: [
          {
            title: "Count the scores",
            say:
              `Every one of the ${n0(n)} tokens scores every token, itself included. That is ${n0(n)} rows ` +
              `by ${n0(n)} columns, so the matrix holds n² = <b>${n0(scores)}</b> numbers. Ten billion, from one document.`,
            blocks: [{ type: "lines", key: "scores", value: scores, fresh: true, lines: [{ t: "n × n = n²" }, scoreLine] }],
          },
          {
            title: "Give each score its 2 bytes",
            say:
              `Each score takes ${bytes} bytes in 16-bit precision, so multiply the count by ${bytes}. ` +
              `The matrix needs <b>${n0(total)}</b> bytes.`,
            blocks: [
              { type: "lines", lines: [Object.assign({}, scoreLine, { hl: false })] },
              { type: "lines", key: "bytes", value: total, fresh: true, lines: [bytesLine] },
            ],
          },
          {
            title: "Turn bytes into gigabytes",
            say:
              `A gigabyte is 1,000,000,000 bytes, so this is about <b>${n0(gb)} GB</b>. And that is for <b>one head</b> ` +
              "in <b>one layer</b>. A real model has dozens of heads and dozens of layers, so storing the full " +
              "attention matrix for a long document simply does not fit on a GPU.",
            blocks: [
              { type: "lines", lines: [Object.assign({}, bytesLine, { hl: false })] },
              {
                type: "lines",
                key: "gb",
                value: gb,
                fresh: true,
                lines: [{ t: `${n0(total)} bytes ≈ ${n0(gb)} GB`, hl: true }, { t: "per head, per layer" }],
              },
            ],
          },
          {
            title: "Double the length",
            say:
              `With ${n0(n2)} tokens the matrix is ${n0(n2)} × ${n0(n2)} = ${n0(scores2)} scores, ` +
              `<b>${growth} times</b> as many, and about ${n0(gb2)} GB. Both sides of the square doubled, so its area went up by 2 × 2.`,
            predict: {
              ask: `Double the length to ${n0(n2)} tokens. How much more work is the score matrix?`,
              choices: GROWTH_CHOICES.map((g) => g + " times"),
              answer: GROWTH_CHOICES.indexOf(growth),
              why: "The matrix is n by n. Doubling n doubles the rows and doubles the columns, so the number of scores goes up 2 × 2 = 4 times.",
              hint: "Picture the square. Doubling the length stretches it in both directions, not just one.",
            },
            blocks: [
              {
                type: "lines",
                key: "double",
                value: growth,
                fresh: true,
                lines: [
                  { t: `${n0(n)}²  = ${n0(scores)}` },
                  { t: `${n0(n2)}²  = ${n0(scores2)}` },
                  { t: `${n0(scores2)} ÷ ${n0(scores)} = ${growth}`, hl: true },
                ],
              },
            ],
          },
          {
            title: "Ten times the length",
            say:
              `Go back to the table. From ${n0(nSmall)} tokens to ${n0(n)} tokens is ten times the length, ` +
              `and the scores go from ${n0(scoresSmall)} to ${n0(scores)}: <b>${n0(tenfold)} times</b> the work.`,
            blocks: [
              {
                type: "lines",
                key: "tenfold",
                value: tenfold,
                fresh: true,
                lines: [
                  { t: `${n0(nSmall)}²  = ${n0(scoresSmall)}` },
                  { t: `${n0(n)}² = ${n0(scores)}` },
                  { t: `${n0(scores)} ÷ ${n0(scoresSmall)} = ${n0(tenfold)}`, hl: true },
                ],
              },
            ],
          },
        ],
        takeaway:
          `Double the length and attention costs ${growth} times as much. Ten times the length and it costs ` +
          `${n0(tenfold)} times as much. At ${n0(n)} tokens one head in one layer already needs about ${n0(gb)} GB ` +
          "just for its scores. This is why early transformers were limited to 512 or 1,024 tokens.",
      };
    },
  };

  if (typeof module !== "undefined" && module.exports) module.exports = defs;
  if (typeof document === "undefined") return;

  /* ============================================================
     Browser side
     ============================================================ */
  const REDUCE = root.matchMedia && root.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const NS = "http://www.w3.org/2000/svg";

  function svg(tag, attrs, text) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  /* ---------- walkthrough panel: the matrix as a square ---------- */
  function memoryPanel(el) {
    const box = svg("svg", { viewBox: "0 0 300 250", class: "l2p5-mem-svg", role: "img" });
    const cap = document.createElement("p");
    cap.className = "l2p5-mem-cap";
    el.append(box, cap);
    const S = 190;
    const X = 55;
    const Y = 14;

    function grid(k, label) {
      const c = S / k;
      for (let i = 0; i < k; i++) {
        for (let j = 0; j < k; j++) {
          box.append(
            svg("rect", {
              x: X + j * c + 1,
              y: Y + i * c + 1,
              width: Math.max(1, c - 2),
              height: Math.max(1, c - 2),
              rx: k > 4 ? 1 : 4,
              class: "l2p5-mem-sq" + (i === 0 && j === 0 ? " is-orig" : ""),
            })
          );
        }
      }
      box.append(svg("text", { x: X + S / 2, y: Y + S + 26, "text-anchor": "middle", class: "l2p5-mem-label" }, label));
    }

    return {
      render(index, step, input) {
        box.replaceChildren();
        const n = input.n;
        const gb = P5.matrixBytes(n, input.bytes) / 1e9;
        let text;
        if (index <= 3) {
          grid(1, `${n0(n)} × ${n0(n)}`);
          text = ["One square: the whole score matrix for one head in one layer.",
            `${n0(P5.scores(n))} scores fill it.`,
            `Each score is ${input.bytes} bytes.`,
            `About ${n0(gb)} GB inside this one square.`][index];
        } else if (index === 4) {
          grid(2, `${n0(2 * n)} × ${n0(2 * n)}`);
          text = "Twice the length: the old square fits inside four times.";
        } else {
          grid(10, `${n0(n)} tokens vs ${n0(n / 10)}`);
          text = `Ten times the length: the ${n0(n / 10)}-token square fits inside a hundred times.`;
        }
        box.setAttribute("aria-label", text);
        cap.textContent = text;
      },
    };
  }

  /* ---------- 5.1 the growing score matrix ---------- */
  function setupGrow() {
    const fig = document.getElementById("l2p5-grow");
    if (!fig) return;
    const range = fig.querySelector("#l2p5-n");
    const out = fig.querySelector("#l2p5-n-out");
    const stage = fig.querySelector(".ix-stage");
    const readout = fig.querySelector(".ix-readout");
    const dbl = fig.querySelector("#l2p5-double");
    const half = fig.querySelector("#l2p5-halve");
    const MAX = Number(range.max);
    const S = 300;
    const O = 22;
    const box = svg("svg", { viewBox: `0 0 ${S + O + 6} ${S + O + 6}`, class: "l2p5-grow-svg", role: "img" });
    stage.append(box);
    let prev = 0;

    function draw() {
      const n = Number(range.value);
      const c = S / MAX;
      const h = n % 2 === 0 ? n / 2 : 0;
      out.textContent = n;
      box.replaceChildren();
      box.append(svg("rect", { x: O, y: O, width: S, height: S, class: "l2p5-frame" }));
      box.append(svg("text", { x: O, y: 14, class: "l2p5-axis" }, "keys →"));
      box.append(svg("text", { x: 14, y: O, class: "l2p5-axis", transform: `rotate(90 14 ${O})` }, "queries →"));
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const fresh = !REDUCE && (i >= prev || j >= prev);
          box.append(
            svg("rect", {
              x: O + j * c + 1,
              y: O + i * c + 1,
              width: c - 2,
              height: c - 2,
              rx: 2,
              class: "l2p5-cell" + (i < h && j < h ? " is-half" : "") + (fresh ? " is-new" : ""),
            })
          );
        }
      }
      if (h) box.append(svg("rect", { x: O, y: O, width: h * c, height: h * c, class: "l2p5-halfbox" }));
      prev = n;
      box.setAttribute("aria-label", `A ${n} by ${n} grid of attention scores.`);
      let t = `<b>n = ${n}</b> tokens, so the score matrix is ${n} × ${n} = <b>${n0(P5.scores(n))}</b> scores, each one computed and stored.`;
      if (h) {
        t +=
          ` The outlined corner is the matrix for n = ${h}: ${n0(P5.scores(h))} scores. ` +
          `Twice the length gives ${P5.scores(n) / P5.scores(h)} times the work.`;
      } else {
        t += " Pick an even n to see the half-length matrix outlined inside it.";
      }
      readout.innerHTML = t;
      dbl.disabled = n * 2 > MAX;
      half.disabled = n < 2;
    }

    range.addEventListener("input", draw);
    dbl.addEventListener("click", () => {
      range.value = Math.min(MAX, Number(range.value) * 2);
      draw();
    });
    half.addEventListener("click", () => {
      range.value = Math.max(1, Math.floor(Number(range.value) / 2));
      draw();
    });
    draw();
  }

  /* ---------- 5.3 the 6d crossover ---------- */
  function setupCross() {
    const fig = document.getElementById("l2p5-cross");
    if (!fig) return;
    const dIn = fig.querySelector("#l2p5-d");
    const nIn = fig.querySelector("#l2p5-ctx");
    const dOut = fig.querySelector("#l2p5-d-out");
    const nOut = fig.querySelector("#l2p5-ctx-out");
    const stage = fig.querySelector(".ix-stage");
    const readout = fig.querySelector(".ix-readout");
    const NMAX = Number(nIn.max);
    const W = 360;
    const H = 230;
    const x0 = 14;
    const x1 = W - 10;
    const y0 = 22;
    const y1 = H - 34;
    const box = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "l2p5-cross-svg", role: "img" });
    stage.append(box);

    function draw() {
      const d = Number(dIn.value);
      const n = Number(nIn.value);
      dOut.textContent = n0(d);
      nOut.textContent = n0(n);
      const ffn = P5.ffnPerToken(d);
      const attnMax = P5.attnPerToken(NMAX, d);
      const yMax = Math.max(ffn, attnMax) * 1.12;
      const X = (v) => x0 + ((x1 - x0) * v) / NMAX;
      const Y = (v) => y1 - ((y1 - y0) * v) / yMax;
      const cross = P5.crossover(d);
      const attnN = P5.attnPerToken(n, d);

      box.replaceChildren();
      if (cross < NMAX) {
        box.append(svg("rect", { x: X(cross), y: y0, width: x1 - X(cross), height: y1 - y0, class: "l2p5-zone" }));
        box.append(svg("text", { x: x1 - 4, y: y0 + 14, "text-anchor": "end", class: "l2p5-zone-label" }, "attention dominates"));
      }
      box.append(svg("line", { x1: x0, y1: y1, x2: x1, y2: y1, class: "l2p5-axisline" }));
      [0, 32000, 64000, 96000, 128000].forEach((v) => {
        if (v > NMAX) return;
        box.append(svg("line", { x1: X(v), y1: y1, x2: X(v), y2: y1 + 4, class: "l2p5-axisline" }));
        box.append(svg("text", { x: X(v), y: y1 + 16, "text-anchor": v === 0 ? "start" : v === NMAX ? "end" : "middle", class: "l2p5-tick" }, v === 0 ? "0" : v / 1000 + "k"));
      });
      box.append(svg("text", { x: (x0 + x1) / 2, y: H - 4, "text-anchor": "middle", class: "l2p5-tick" }, "sequence length n"));

      box.append(svg("line", { x1: X(0), y1: Y(ffn), x2: X(NMAX), y2: Y(ffn), class: "l2p5-ffn" }));
      box.append(svg("line", { x1: X(0), y1: Y(0), x2: X(NMAX), y2: Y(attnMax), class: "l2p5-attn" }));
      box.append(svg("text", { x: x0 + 4, y: Y(ffn) - 6, class: "l2p5-lab-ffn" }, "projections + FFN: 24·d²"));
      box.append(svg("text", { x: x1 - 4, y: Math.max(y0 + 30, Y(attnMax) + 4) + 12, "text-anchor": "end", class: "l2p5-lab-attn" }, "attention: 4·n·d"));

      if (cross <= NMAX) {
        box.append(svg("line", { x1: X(cross), y1: y0, x2: X(cross), y2: y1, class: "l2p5-crossline" }));
        box.append(svg("circle", { cx: X(cross), cy: Y(ffn), r: 5, class: "l2p5-crossdot" }));
        const anchor = X(cross) > W / 2 ? "end" : "start";
        box.append(svg("text", { x: X(cross) + (anchor === "end" ? -6 : 6), y: y1 - 8, "text-anchor": anchor, class: "l2p5-crosslab" }, "n = 6d = " + n0(cross)));
      }
      box.append(svg("line", { x1: X(n), y1: y0, x2: X(n), y2: y1, class: "l2p5-nline" }));
      box.append(svg("circle", { cx: X(n), cy: Y(attnN), r: 4, class: "l2p5-ndot-attn" }));
      box.append(svg("rect", { x: X(n) - 4, y: Y(ffn) - 4, width: 8, height: 8, class: "l2p5-ndot-ffn" }));
      box.setAttribute("aria-label", `Cost per token against sequence length for d = ${n0(d)}. The lines cross at n = ${n0(cross)}.`);

      let t =
        `The lines cross where 4 · n · d = 24 · d², at n = 6 × ${n0(d)} = <b>${n0(cross)}</b> tokens` +
        (d === 4096 ? ", about 24,000." : ".") +
        ` At n = ${n0(n)}, attention costs about ${n0(attnN)} operations per token and the projections plus FFN about ${n0(ffn)}. `;
      if (attnN > ffn) t += `Attention is <b>${M.fmt(attnN / ffn, 1)} times</b> the rest.`;
      else if (attnN < ffn) t += `The rest is <b>${M.fmt(ffn / attnN, 1)} times</b> attention.`;
      else t += "They cost exactly the same.";
      readout.innerHTML = t;
    }

    dIn.addEventListener("input", draw);
    nIn.addEventListener("input", draw);
    draw();
  }

  /* ---------- 5.5 FlashAttention tiling ---------- */
  function setupFlash() {
    const fig = document.getElementById("l2p5-flash");
    if (!fig) return;
    const chips = Array.from(fig.querySelectorAll(".chip[data-mode]"));
    const stepBtn = fig.querySelector("#l2p5-flash-step");
    const runBtn = fig.querySelector("#l2p5-flash-run");
    const resetBtn = fig.querySelector("#l2p5-flash-reset");
    const stage = fig.querySelector(".ix-stage");
    const readout = fig.querySelector(".ix-readout");
    const T = 4;
    const TOTAL = T * T;
    const size = 44;
    const gap = 4;
    const gx = 22;
    const gy = 42;
    const sx = 250;
    const sy = 64;
    const box = svg("svg", { viewBox: "0 0 360 262", class: "l2p5-flash-svg", role: "img" });
    stage.append(box);
    const state = { mode: "standard", done: 0, timer: 0 };

    function stopRun() {
      clearInterval(state.timer);
      state.timer = 0;
      runBtn.textContent = "Run all tiles";
    }

    function draw() {
      const { mode, done } = state;
      const flash = mode === "flash";
      box.replaceChildren();
      box.append(svg("rect", { x: 6, y: 26, width: 216, height: 222, rx: 10, class: "l2p5-hbm" }));
      box.append(svg("text", { x: 12, y: 18, class: "l2p5-memlab" }, "HBM: large, slow"));
      box.append(svg("rect", { x: sx - 12, y: sy - 8, width: 116, height: 76, rx: 10, class: "l2p5-sram" }));
      box.append(svg("text", { x: sx - 12, y: sy - 16, class: "l2p5-memlab" }, "SRAM: small, fast"));

      for (let i = 0; i < T; i++) {
        for (let j = 0; j < T; j++) {
          const k = i * T + j;
          let cls = "l2p5-tile";
          if (k < done) cls += flash ? " is-passed" : " is-stored";
          if (k === done - 1) cls += " is-current";
          box.append(svg("rect", { x: gx + j * (size + gap), y: gy + i * (size + gap), width: size, height: size, rx: 4, class: cls }));
        }
      }
      box.append(svg("text", { x: gx, y: gy + T * (size + gap) + 6, class: "l2p5-tick" }, flash ? "n × n scores: never stored" : "n × n scores"));

      if (done > 0) {
        const k = done - 1;
        const tx = gx + (k % T) * (size + gap) + size / 2;
        const ty = gy + Math.floor(k / T) * (size + gap) + size / 2;
        box.append(svg("rect", { x: sx + 4, y: sy + 4, width: size, height: size, rx: 4, class: "l2p5-tile is-live" }));
        box.append(svg("text", { x: sx + 56, y: sy + 24, class: "l2p5-tick" }, "tile " + done));
        box.append(svg("text", { x: sx + 56, y: sy + 40, class: "l2p5-tick" }, "of " + TOTAL));
        box.append(svg("line", { x1: sx - 12, y1: sy + 26, x2: tx + 6, y2: ty, class: flash ? "l2p5-link is-quiet" : "l2p5-link" }));
      }

      // The output, n x d, is written in both methods: one block per finished row of tiles.
      const oy = 176;
      box.append(svg("text", { x: sx - 12, y: oy - 8, class: "l2p5-memlab" }, "output n × d"));
      const rowsDone = Math.floor(done / T);
      for (let r = 0; r < T; r++) {
        box.append(svg("rect", { x: sx - 12, y: oy + r * 16, width: 116, height: 12, rx: 3, class: "l2p5-out" + (r < rowsDone ? " is-done" : "") }));
      }

      const stored = flash ? 0 : done;
      box.setAttribute("aria-label", `${mode === "flash" ? "FlashAttention" : "Standard attention"}: ${done} of ${TOTAL} tiles computed, ${stored} stored in slow memory.`);
      readout.innerHTML =
        `<span class="l2p5-meter"><span>Score tiles computed</span><span class="l2p5-bar"><span style="width:${(100 * done) / TOTAL}%"></span></span><b>${done} of ${TOTAL}</b></span>` +
        `<span class="l2p5-meter is-hbm"><span>Score tiles written to HBM</span><span class="l2p5-bar"><span style="width:${(100 * stored) / TOTAL}%"></span></span><b>${stored} of ${TOTAL}</b></span>` +
        `<span class="l2p5-note">` +
        (flash
          ? "Each tile is computed in SRAM, folded into a running softmax, and thrown away. The full n × n matrix never reaches slow memory, so memory grows linearly in n."
          : "Each tile is computed, then written out to slow memory and read back later for softmax. By the end the whole n × n matrix sits in HBM.") +
        (done === TOTAL ? " <b>Either way, all " + TOTAL + " tiles were computed: the n² work is still there.</b>" : "") +
        `</span>`;
      stepBtn.disabled = done >= TOTAL;
    }

    function step() {
      if (state.done < TOTAL) state.done += 1;
      if (state.done >= TOTAL) stopRun();
      draw();
    }

    chips.forEach((c) =>
      c.addEventListener("click", () => {
        stopRun();
        state.mode = c.dataset.mode;
        state.done = 0;
        chips.forEach((x) => x.setAttribute("aria-pressed", String(x === c)));
        draw();
      })
    );
    stepBtn.addEventListener("click", () => {
      stopRun();
      step();
    });
    runBtn.addEventListener("click", () => {
      if (state.timer) return stopRun();
      if (state.done >= TOTAL) state.done = 0;
      if (REDUCE) {
        state.done = TOTAL;
        draw();
        return;
      }
      runBtn.textContent = "Pause";
      state.timer = setInterval(step, 420);
      step();
    });
    resetBtn.addEventListener("click", () => {
      stopRun();
      state.done = 0;
      draw();
    });
    draw();
  }

  function init() {
    setupGrow();
    setupCross();
    setupFlash();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})(typeof window !== "undefined" ? window : globalThis);
