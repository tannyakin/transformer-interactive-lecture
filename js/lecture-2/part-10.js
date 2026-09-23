// ============================================================
// Lecture 2, Part 10: after BERT (RoBERTa, DistilBERT and the
// rest of the family). Walkthroughs compute every number from
// their input, and the interactives below reuse the same maths.
// ============================================================
(function (root) {
  "use strict";

  const L = (root.Lecture = root.Lecture || {});
  const M = L.math;
  const defs = (L.defs = L.defs || {});
  const f = (x, dp) => M.fmt(x, dp === undefined ? 3 : dp);
  const n = (x) => M.fmtInt(x);
  const CLASSES = ["positive", "neutral", "negative"];

  /* ---------------- pure helpers (tested in Node) ---------------- */

  // Where the student sits at slider position t (0 = early, 1 = later).
  // A straight line between two distributions always still adds up to 1.
  function studentAt(early, late, t) {
    return early.map((e, i) => e + t * (late[i] - e));
  }

  // k distinct positions out of n, sorted, using the given random source.
  function drawMask(count, k, rand) {
    const r = rand || Math.random;
    const picked = new Set();
    while (picked.size < k) picked.add(Math.floor(r() * count));
    return Array.from(picked).sort((a, b) => a - b);
  }

  const maskKey = (positions) => positions.join(",");

  // The mask used at a given epoch. Static masking cycles through a fixed set of copies.
  function maskForEpoch(mode, epoch, copies, rand, count, k) {
    if (mode === "static") return copies[epoch % copies.length];
    return drawMask(count, k, rand);
  }

  L.l2p10 = { studentAt, drawMask, maskKey, maskForEpoch, CLASSES };

  /* ---------------- 10.1 RoBERTa: how much more text ---------------- */
  defs["l2-roberta-batches"] = {
    id: "l2-roberta-batches",
    title: "How much more text RoBERTa actually saw",
    input: {
      bert: { steps: 1000000, batch: 256 },
      roberta: { steps: 500000, batch: 8000 },
    },
    setup:
      "A step is one update of the weights. The batch size is how many sequences are used in that one step. " +
      "So steps times batch size is the total number of sequences the model read during training. " +
      "Let's work it out for both models.",
    setupBlocks: (input) => [
      {
        type: "matrix",
        key: "recipe",
        colLabels: ["steps", "batch size"],
        rowLabels: ["BERT", "RoBERTa"],
        rows: [
          [n(input.bert.steps), n(input.bert.batch)],
          [n(input.roberta.steps), n(input.roberta.batch)],
        ],
      },
    ],
    panel: robertaPanel,
    build(input) {
      const { bert, roberta } = input;
      const bertSeq = bert.steps * bert.batch;
      const robSeq = roberta.steps * roberta.batch;
      const ratio = robSeq / bertSeq;
      const stepRatio = roberta.steps / bert.steps;
      const batchRatio = roberta.batch / bert.batch;

      const bertLine = { t: `BERT:     ${n(bert.steps)} steps × ${n(bert.batch)} sequences   =   ${n(bertSeq)} sequences` };
      const robLine = { t: `RoBERTa:    ${n(roberta.steps)} steps × ${n(roberta.batch)} sequences = ${n(robSeq)} sequences` };
      const table = (hl) => ({
        type: "matrix",
        key: "totals",
        colLabels: ["steps", "batch size", "sequences"],
        rowLabels: ["BERT", "RoBERTa"],
        rows: [
          [n(bert.steps), n(bert.batch), n(bertSeq)],
          [n(roberta.steps), n(roberta.batch), n(robSeq)],
        ],
        hl,
        raw: { bertSeq, robSeq },
      });

      return {
        steps: [
          {
            title: "Count BERT's sequences",
            say:
              `BERT took ${n(bert.steps)} steps, and each step looked at ${n(bert.batch)} sequences. ` +
              `Multiply them and BERT read ${n(bertSeq)} sequences in total. That sounds like a lot. Hold on to it.`,
            blocks: [
              { type: "lines", fresh: true, lines: [Object.assign({ hl: true }, bertLine)] },
              { type: "vector", key: "bert", label: "BERT total", values: [n(bertSeq)], raw: bertSeq, tone: "result", fresh: true },
            ],
          },
          {
            title: "Count RoBERTa's sequences",
            say:
              `RoBERTa took only ${n(roberta.steps)} steps, but each one used ${n(roberta.batch)} sequences. ` +
              `That comes to ${n(robSeq)} sequences.`,
            blocks: [
              { type: "lines", lines: [bertLine, Object.assign({ hl: true }, robLine)] },
              { type: "vector", key: "roberta", label: "RoBERTa total", values: [n(robSeq)], raw: robSeq, tone: "result", fresh: true },
            ],
          },
          {
            title: "Divide one by the other",
            say:
              `${n(robSeq)} divided by ${n(bertSeq)} is ${f(ratio, 3)}, so about ${f(ratio, 1)}. ` +
              `RoBERTa read roughly ${f(ratio, 1)} times as many sequences as BERT.`,
            predict: {
              ask: "RoBERTa took half as many steps as BERT. How many times more sequences did it see?",
              choices: ["About half as many", "About 2 times", `About ${f(ratio, 1)} times`, `About ${f(batchRatio, 1)} times`],
              answer: 2,
              why:
                `The batch grew ${f(batchRatio, 2)} times but the number of steps halved, and ` +
                `${f(batchRatio, 2)} × ${f(stepRatio, 1)} = ${f(ratio, 3)}. Both changes count.`,
              hint: "Total sequences = steps × batch size. The batch got much bigger, but the steps were cut in half.",
            },
            blocks: [
              table([[0, 2], [1, 2]]),
              {
                type: "lines",
                fresh: true,
                lines: [{ t: `ratio: ${n(robSeq)} / ${n(bertSeq)} ≈ ${f(ratio, 1)}`, hl: true }],
              },
              { type: "vector", key: "ratio", label: "ratio", values: [M.round(ratio, 1)], raw: ratio, dp: 1, tone: "result", fresh: true },
            ],
          },
          {
            title: "Half the steps, about 15.6 times the sequences",
            say:
              `Put the two ratios side by side. Steps went down to ${f(stepRatio, 1)} of BERT's, ` +
              `while sequences went up about ${f(ratio, 1)} times. Fewer updates, but each update learned from far more text.`,
            blocks: [
              table([[0, 0], [1, 0], [0, 2], [1, 2]]),
              {
                type: "lines",
                lines: [
                  { t: `steps:      ${n(roberta.steps)} / ${n(bert.steps)} = ${f(stepRatio, 1)}   (half as many)` },
                  { t: `sequences:  ${n(robSeq)} / ${n(bertSeq)} ≈ ${f(ratio, 1)}   (about ${f(ratio, 1)} times more)`, hl: true },
                ],
              },
            ],
          },
          {
            title: "Why fewer, bigger steps can teach more",
            say:
              "Each step asks: which way should the weights move? A bigger batch averages that answer over more examples, " +
              "so the direction is smoother and more reliable. GPUs also process large batches very efficiently in parallel, " +
              "so fewer, bigger steps can mean more learning in the same wall-clock time.",
            blocks: [
              table([[1, 1]]),
              {
                type: "note",
                html:
                  "A small batch is like asking three friends for directions: you get a rough answer. " +
                  "A batch of " + n(roberta.batch) + " is like asking a whole crowd and taking the average.",
              },
            ],
          },
        ],
        takeaway:
          `RoBERTa took half as many steps but processed about ${f(ratio, 1)} times more sequences. ` +
          "Bigger batches give a smoother, more reliable estimate of which direction to update the weights, " +
          "and GPUs handle them very efficiently. So fewer, bigger steps can mean more learning in the same time.",
      };
    },
  };

  // Area picture: width is the number of steps, height is the batch size, so the area is the sequences.
  function robertaPanel(el) {
    const NS = "http://www.w3.org/2000/svg";
    const W = 320;
    const H = 230;
    const x0 = 44;
    const y0 = 24;
    const pw = 256;
    const ph = 160;
    el.innerHTML =
      `<svg class="l2p10-area" viewBox="0 0 ${W} ${H}" role="img" aria-label="Rectangles whose width is steps and height is batch size">` +
      `<text class="l2p10-axis-t" x="${x0}" y="14">batch size</text>` +
      `<line class="l2p10-axis" x1="${x0}" y1="${y0}" x2="${x0}" y2="${y0 + ph}"/>` +
      `<line class="l2p10-axis" x1="${x0}" y1="${y0 + ph}" x2="${x0 + pw}" y2="${y0 + ph}"/>` +
      `<text class="l2p10-tick" x="${x0 - 6}" y="${y0 + 4}" text-anchor="end">8,000</text>` +
      `<text class="l2p10-tick" x="${x0 - 6}" y="${y0 + ph}" text-anchor="end">0</text>` +
      `<text class="l2p10-tick" x="${x0 + pw / 2}" y="${y0 + ph + 16}" text-anchor="middle">500k</text>` +
      `<text class="l2p10-tick" x="${x0 + pw}" y="${y0 + ph + 16}" text-anchor="end">1M</text>` +
      `<text class="l2p10-axis-t" x="${x0 + pw}" y="${y0 + ph + 34}" text-anchor="end">steps</text>` +
      `<g data-g="rob"></g><g data-g="bert"></g><g data-g="note"></g>` +
      `</svg>` +
      `<p class="l2p10-area-cap">Width is the number of steps, height is the batch size. The area is how many sequences the model read.</p>`;
    const svg = el.querySelector("svg");
    const g = (k) => svg.querySelector(`[data-g="${k}"]`);
    function rect(parent, cls, x, y, w, h) {
      const r = document.createElementNS(NS, "rect");
      r.setAttribute("class", cls);
      r.setAttribute("x", x);
      r.setAttribute("y", y);
      r.setAttribute("width", w);
      r.setAttribute("height", h);
      parent.append(r);
    }
    function text(parent, cls, x, y, s, anchor) {
      const t = document.createElementNS(NS, "text");
      t.setAttribute("class", cls);
      t.setAttribute("x", x);
      t.setAttribute("y", y);
      if (anchor) t.setAttribute("text-anchor", anchor);
      t.textContent = s;
      parent.append(t);
    }
    return {
      render(index, step, input) {
        const { bert, roberta } = input;
        const maxSteps = Math.max(bert.steps, roberta.steps);
        const maxBatch = Math.max(bert.batch, roberta.batch);
        const wOf = (s) => (s / maxSteps) * pw;
        const hOf = (b) => Math.max(2, (b / maxBatch) * ph);
        const ratio = (roberta.steps * roberta.batch) / (bert.steps * bert.batch);
        ["rob", "bert", "note"].forEach((k) => g(k).replaceChildren());
        const last = 6;
        if (index >= 1) {
          const h = hOf(bert.batch);
          rect(g("bert"), "l2p10-r-bert", x0, y0 + ph - h, wOf(bert.steps), h);
          text(g("bert"), "l2p10-r-label", x0 + pw - 4, y0 + ph - h - 6, "BERT: " + n(bert.steps * bert.batch), "end");
        }
        if (index >= 2) {
          const h = hOf(roberta.batch);
          rect(g("rob"), "l2p10-r-rob", x0, y0 + ph - h, wOf(roberta.steps), h);
          text(g("rob"), "l2p10-r-label", x0 + 8, y0 + 22, "RoBERTa:");
          text(g("rob"), "l2p10-r-label", x0 + 8, y0 + 40, n(roberta.steps * roberta.batch));
        }
        if (index >= 3) {
          text(g("note"), "l2p10-r-big", x0 + wOf(roberta.steps) + 12, y0 + 64, "≈ " + f(ratio, 1) + "×");
          text(g("note"), "l2p10-r-small", x0 + wOf(roberta.steps) + 12, y0 + 84, "the area");
        }
        if (index >= 4 && index <= last) {
          text(g("note"), "l2p10-r-small", x0 + wOf(roberta.steps) / 2, y0 + ph + 16 - 30, "half the width", "middle");
        }
      },
    };
  }

  /* ---------------- 10.1 dynamic versus static masking ---------------- */
  defs["l2-dynamic-masking"] = {
    id: "l2-dynamic-masking",
    title: "Dynamic versus static masking",
    input: { epochs: 40, copies: 10, tokens: 20, rate: 0.15 },
    setup:
      "One sentence is fed to the model over 40 epochs, which means 40 passes through the training data. " +
      "Original BERT made 10 masked copies of the data in advance. RoBERTa draws a fresh mask every time. " +
      "How much variety does each approach really give?",
    build(input) {
      const { epochs, copies, tokens, rate } = input;
      const timesEach = epochs / copies;
      const masked = Math.round(tokens * rate);
      const top = [];
      for (let i = 0; i < masked; i++) top.push(tokens - i);
      const bottom = [];
      for (let i = masked; i >= 1; i--) bottom.push(i);
      const topProd = top.reduce((a, b) => a * b, 1);
      const botProd = bottom.reduce((a, b) => a * b, 1);
      const ways = M.choose(tokens, masked);

      const staticLine = { t: `BERT (static, ${copies} copies):   ${copies} different masks, each seen ${timesEach} times` };
      const dynLine = { t: `RoBERTa (dynamic):          up to ${epochs} different masks, each seen once` };
      const countLine = `C(${tokens}, ${masked}) = (${top.join(" × ")}) / (${bottom.join(" × ")}) = ${n(topProd)} / ${botProd} = ${n(ways)}`;

      return {
        steps: [
          {
            title: "Static masking: 10 copies, reused",
            say:
              `BERT masked the data once, during preprocessing. To get some variety it duplicated the data ${copies} times ` +
              `with ${copies} different masks. Spread over ${epochs} epochs, that is ${epochs} / ${copies} = ${timesEach}: ` +
              `the model sees each exact mask ${timesEach} times.`,
            blocks: [
              { type: "lines", fresh: true, lines: [{ t: `${epochs} epochs / ${copies} copies = ${timesEach} views of each mask`, hl: true }] },
              { type: "lines", lines: [Object.assign({ hl: true }, staticLine)] },
              { type: "vector", key: "static", label: "static", values: [copies, timesEach], cols: ["masks", "each seen"], tone: "muted" },
            ],
          },
          {
            title: "Dynamic masking: a fresh mask every time",
            say:
              "RoBERTa generates a new random mask every time the sentence is fed to the model. " +
              `Over ${epochs} epochs that is up to ${epochs} different masks, and it never practises on the same gaps twice.`,
            blocks: [
              { type: "lines", lines: [staticLine, Object.assign({ hl: true }, dynLine)] },
              { type: "vector", key: "dynamic", label: "dynamic", values: [epochs, 1], cols: ["masks", "each seen"], tone: "result", fresh: true },
            ],
          },
          {
            title: "How many tokens get masked?",
            say:
              `BERT and RoBERTa both mask ${f(rate * 100, 0)}% of the tokens. A ${tokens}-token sentence ` +
              `therefore has ${f(rate, 2)} × ${tokens} = ${masked} masked positions.`,
            blocks: [
              { type: "lines", fresh: true, lines: [{ t: `${f(rate, 2)} × ${tokens} tokens = ${masked} masked positions`, hl: true }] },
              { type: "vector", key: "masked", label: "masked", values: [masked], tone: "result" },
            ],
          },
          {
            title: "Count every possible mask",
            say:
              `There are ${tokens} choices for the first position, ${tokens - 1} for the second and ${tokens - 2} for the third, ` +
              `which gives ${n(topProd)}. But the same three positions picked in a different order are the same mask, ` +
              `and ${masked} positions can be ordered ${botProd} ways. So divide by ${botProd}: ${n(ways)} possible masks.`,
            predict: {
              ask: `How many different ${masked}-position masks can a ${tokens}-token sentence have?`,
              choices: [`${tokens * masked}`, n(ways), n(topProd), n(Math.pow(tokens, masked))],
              answer: 1,
              why:
                `${tokens} × ${tokens - 1} × ${tokens - 2} = ${n(topProd)} counts every ordering of the same three positions separately. ` +
                `Each set of 3 appears ${botProd} times in that count, so ${n(topProd)} / ${botProd} = ${n(ways)}.`,
              hint: "Order does not matter: masking positions 2, 5, 9 is the same mask as 9, 5, 2.",
            },
            blocks: [
              { type: "lines", fresh: true, lines: [{ t: countLine, hl: true }] },
              { type: "vector", key: "ways", label: "possible masks", values: [n(ways)], raw: ways, tone: "result", fresh: true },
            ],
          },
          {
            title: "Compare what each approach uses",
            say:
              `There are ${n(ways)} possible masks for that one sentence. Static masking only ever uses ${copies} of them. ` +
              `Dynamic masking keeps drawing new ones, up to ${epochs} over training, so the model practises filling in ` +
              "many more different gaps from exactly the same text.",
            blocks: [
              {
                type: "matrix",
                key: "compare",
                colLabels: ["masks used", "possible", "each seen"],
                rowLabels: ["static", "dynamic"],
                rows: [
                  [copies, n(ways), `${timesEach} times`],
                  [`up to ${epochs}`, n(ways), "once"],
                ],
                hl: [[1, 0]],
              },
            ],
          },
        ],
        takeaway:
          `There are ${n(ways)} possible masks for one ${tokens}-token sentence. Static masking only ever uses ${copies} of them. ` +
          "Dynamic masking keeps drawing new ones, so the model practises filling in many more different gaps from the same text, " +
          "and that variety costs nothing extra.",
      };
    },
  };

  /* ---------------- 10.2 the distillation loss ---------------- */
  defs["l2-kl-distill"] = {
    id: "l2-kl-distill",
    title: "The distillation loss, by hand",
    input: {
      teacher: [0.25, 0.7, 0.05],
      early: [0.4, 0.5, 0.1],
      late: [0.27, 0.68, 0.05],
    },
    setup:
      "The teacher reads “The movie was fine.” and gives positive, neutral and negative the probabilities below. " +
      "Early in training, the student says something rather different. The KL divergence puts one number on how different.",
    setupBlocks: (input) => [
      { type: "vector", key: "teacher", label: "teacher ŷ_T", values: input.teacher, cols: CLASSES, dp: 2 },
      { type: "vector", key: "early", label: "student ŷ_S", values: input.early, dp: 2 },
    ],
    panel: klPanel,
    build(input) {
      const { teacher, early, late } = input;
      const ratios = teacher.map((t, i) => t / early[i]);
      const logs = ratios.map((r) => Math.log(r));
      const kl = M.klDivergence(teacher, early);
      const terms4 = kl.terms.map((x) => M.round(x, 4));
      const roundedSum = terms4.reduce((a, b) => a + b, 0);
      const later = M.klDivergence(teacher, late);
      const laterLogs = teacher.map((t, i) => Math.log(t / late[i]));
      const laterTerms4 = later.terms.map((x) => M.round(x, 4));

      const tRow = { type: "vector", label: "teacher ŷ_T", values: teacher, cols: CLASSES, dp: 2 };
      const sRow = { type: "vector", label: "student ŷ_S", values: early, dp: 2 };
      const pad = (s, w) => (s + " ".repeat(w)).slice(0, w);
      const signed = (x, dp) => (x < 0 ? "" : " ") + f(x, dp);
      const termLine = (i, T, S, lg, term) =>
        `${pad(CLASSES[i] + ":", 10)}${f(T[i], 2)} × log(${f(T[i], 2)} / ${f(S[i], 2)}) = ${f(T[i], 2)} × (${signed(lg, 3)}) = ${signed(term, 4)}`;

      return {
        steps: [
          {
            title: "Compare each class with a ratio",
            say:
              "For each class, divide the teacher's probability by the student's. A ratio of 1 would mean they agree exactly. " +
              `Here neutral is ${f(ratios[1], 3)}, because the teacher is more sure than the student. ` +
              `Positive and negative are below 1, because the student gives them too much.`,
            blocks: [
              tRow,
              sRow,
              {
                type: "lines",
                fresh: true,
                lines: CLASSES.map((c, i) => ({ t: `${pad(c + ":", 10)}${f(teacher[i], 2)} / ${f(early[i], 2)} = ${f(ratios[i], 3)}` })),
              },
              { type: "vector", key: "ratios", label: "ratio", values: ratios, dp: 3, fresh: true },
            ],
          },
          {
            title: "Take the natural log of each ratio",
            say:
              "The log turns agreement into zero: log(1) = 0. A ratio above 1 gives a positive number, a ratio below 1 a negative one. " +
              `So neutral gives ${f(logs[1], 3)} and the other two are negative.`,
            blocks: [
              { type: "vector", label: "ratio", values: ratios, cols: CLASSES, dp: 3 },
              { type: "vector", key: "logs", label: "log(ratio)", values: logs, dp: 3, hl: [1], fresh: true },
            ],
          },
          {
            title: "Weight each log by the teacher's probability",
            say:
              "Multiply each log by the teacher's probability, so the classes the teacher cares about most count the most. " +
              `Neutral, at ${f(teacher[1], 2)}, dominates. Negative, at only ${f(teacher[2], 2)}, barely registers.`,
            blocks: [
              {
                type: "lines",
                fresh: true,
                lines: CLASSES.map((c, i) => ({ t: termLine(i, teacher, early, logs[i], kl.terms[i]), hl: i === 1 })),
              },
              { type: "vector", key: "terms", label: "terms", values: terms4, cols: CLASSES, dp: 4, raw: kl.terms, fresh: true },
            ],
          },
          {
            title: "Add the terms: the KL divergence",
            say:
              `Adding the three rounded terms gives ${f(roundedSum, 4)}. ` +
              `Without rounding the terms first you get ${f(kl.total, 4)}, so do not worry if your calculator says that. ` +
              "Two terms were negative, yet the total is positive. That is always true of KL.",
            predict: {
              ask: "Two of the three terms are negative. Can the total KL ever come out below zero?",
              choices: ["Yes, if the negative terms are big enough", "No, KL is always 0 or more"],
              answer: 1,
              why:
                "Both distributions add up to 1, so where the student gives too much to one class it must give too little elsewhere. " +
                "The weighted log terms always balance out to 0 or more, and exactly 0 only when the two distributions match.",
              hint: "Think about what happens when student and teacher match: every ratio is 1. Can anything be smaller than a perfect match?",
            },
            blocks: [
              { type: "vector", label: "terms", values: terms4, cols: CLASSES, dp: 4 },
              {
                type: "lines",
                fresh: true,
                lines: [
                  { t: `KL = ${f(terms4[0], 4)} + ${f(terms4[1], 4)} ${terms4[2] < 0 ? "−" : "+"} ${f(Math.abs(terms4[2]), 4)} = ${f(roundedSum, 4)}`, hl: true },
                  { t: `(unrounded terms: ${f(kl.total, 4)})` },
                ],
              },
              { type: "vector", key: "kl-early", label: "KL", values: [roundedSum], raw: kl.total, dp: 4, tone: "result", fresh: true },
            ],
          },
          {
            title: "Later in training",
            say:
              `After more training the student says [${late.map((x) => f(x, 2)).join(", ")}], very close to the teacher. ` +
              `Work it through the same way and KL falls from ${f(roundedSum, 4)} to about ${f(later.total, 4)}.`,
            predict: {
              ask: `The student moves to [${late.map((x) => f(x, 2)).join(", ")}]. What happens to KL?`,
              choices: ["It goes up", "It drops towards 0", "It stays about the same"],
              answer: later.total < kl.total ? 1 : 0,
              why:
                "Every ratio is now close to 1, so every log is close to 0, and so is every term. " +
                "The closer the student gets to the teacher, the closer KL gets to zero.",
              hint: "Compare the new student with the teacher, class by class. How far is each ratio from 1?",
            },
            blocks: [
              tRow,
              { type: "vector", label: "student ŷ_S", values: late, dp: 2, hl: [0, 1] },
              {
                type: "lines",
                fresh: true,
                lines: CLASSES.map((c, i) => ({ t: termLine(i, teacher, late, laterLogs[i], later.terms[i]) })).concat([
                  { t: `KL ≈ ${f(later.total, 4)}`, hl: true },
                ]),
              },
              { type: "vector", key: "kl-late", label: "KL", values: [later.total], raw: laterTerms4, dp: 4, tone: "result", fresh: true },
            ],
          },
        ],
        takeaway:
          `As the student's answers get closer to the teacher's, the loss falls towards zero (${f(roundedSum, 4)} down to ${f(later.total, 4)}). ` +
          "Individual terms can be negative, but the total never is. Training pushes this number down, which pulls the student's whole " +
          "distribution, including the small “dark knowledge” probabilities, towards the teacher's.",
      };
    },
  };

  // Teacher and student side by side for each class.
  function klPanel(el) {
    return {
      render(index, step, input) {
        const late = index >= 5;
        const student = late ? input.late : input.early;
        const kl = M.klDivergence(input.teacher, student);
        const showKl = index >= 4;
        el.innerHTML = barsSvg(input.teacher, student, late ? "student, later" : "student, early", index === 3 ? [1] : []) +
          `<p class="l2p10-kl-read">${
            !showKl
              ? "KL is worked out step by step"
              : late
              ? "KL ≈ <b>" + f(kl.total, 4) + "</b>"
              : "KL = <b>" + f(kl.terms.map((x) => M.round(x, 4)).reduce((a, b) => a + b, 0), 4) + "</b> from the rounded terms (" + f(kl.total, 4) + " unrounded)"
          }</p>`;
      },
    };
  }

  // Grouped bars: teacher filled, student outlined, one pair per class. Returns SVG markup.
  function barsSvg(teacher, student, studentLabel, hl) {
    const W = 320;
    const H = 210;
    const base = 170;
    const top = 40;
    const maxV = 0.8;
    const groupW = 92;
    const barW = 30;
    const x0 = 22;
    const y = (v) => base - (v / maxV) * (base - top);
    let s = `<svg class="l2p10-kbars" viewBox="0 0 ${W} ${H}" role="img" aria-label="Teacher and ${studentLabel} probabilities for positive, neutral and negative">`;
    s += `<rect class="l2p10-k-teacher" x="22" y="8" width="14" height="10" rx="2"/><text class="l2p10-k-key" x="42" y="17">teacher</text>`;
    s += `<rect class="l2p10-k-student" x="120" y="8" width="14" height="10" rx="2"/><text class="l2p10-k-key" x="140" y="17">${studentLabel}</text>`;
    s += `<line class="l2p10-axis" x1="${x0 - 8}" y1="${base}" x2="${W - 8}" y2="${base}"/>`;
    CLASSES.forEach((c, i) => {
      const gx = x0 + i * (groupW + 10);
      const on = hl && hl.includes(i);
      const tx = gx + 8;
      const sx = gx + 8 + barW + 6;
      s += `<rect class="l2p10-k-teacher${on ? " is-hl" : ""}" x="${tx}" y="${y(teacher[i])}" width="${barW}" height="${base - y(teacher[i])}" rx="3"/>`;
      s += `<rect class="l2p10-k-student${on ? " is-hl" : ""}" x="${sx}" y="${y(student[i])}" width="${barW}" height="${Math.max(0.5, base - y(student[i]))}" rx="3"/>`;
      s += `<text class="l2p10-k-val" x="${tx + barW / 2}" y="${y(teacher[i]) - 5}" text-anchor="middle">${f(teacher[i], 2)}</text>`;
      s += `<text class="l2p10-k-val is-s" x="${sx + barW / 2}" y="${y(student[i]) - 5}" text-anchor="middle">${f(student[i], 2)}</text>`;
      s += `<text class="l2p10-k-cls" x="${gx + 8 + barW + 3}" y="${base + 18}" text-anchor="middle">${c}</text>`;
    });
    return s + "</svg>";
  }
  L.l2p10.barsSvg = barsSvg;

  if (typeof module !== "undefined" && module.exports) module.exports = defs;
  if (typeof document === "undefined") return;

  /* ---------------- browser-only interactives ---------------- */
  const REDUCE = root.matchMedia && root.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function setPressed(group, btn) {
    group.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
  }

  // Re-roll the mask on a 20-token sentence, static or dynamic.
  function initMask(fig) {
    const words = fig.dataset.sentence.split(" ");
    const EPOCHS = 40;
    const COPIES = 10;
    const K = Math.round(words.length * 0.15);
    const TOTAL = M.choose(words.length, K);
    const copies = [];
    const seenCopy = new Set();
    while (copies.length < COPIES) {
      const m = drawMask(words.length, K);
      if (!seenCopy.has(maskKey(m))) {
        seenCopy.add(maskKey(m));
        copies.push(m);
      }
    }
    const tokensEl = fig.querySelector(".l2p10-tokens");
    const stripEl = fig.querySelector(".l2p10-epochs");
    const readout = fig.querySelector(".ix-readout");
    const modeGroup = fig.querySelector(".chips");
    const rollBtn = fig.querySelector('[data-act="roll"]');
    const allBtn = fig.querySelector('[data-act="all"]');
    const resetBtn = fig.querySelector('[data-act="reset"]');
    let mode = "dynamic";
    let history = [];

    tokensEl.replaceChildren(
      ...words.map((w, i) => {
        const li = document.createElement("li");
        li.className = "l2p10-tok";
        li.innerHTML = `<span class="l2p10-tok-w"></span><span class="l2p10-tok-i">${i + 1}</span>`;
        li.firstChild.textContent = w;
        return li;
      })
    );
    stripEl.replaceChildren(
      ...Array.from({ length: EPOCHS }, (_, i) => {
        const s = document.createElement("span");
        s.className = "l2p10-ep";
        s.title = "Epoch " + (i + 1);
        return s;
      })
    );

    function render() {
      const cur = history[history.length - 1];
      const on = new Set(cur ? cur.mask : []);
      Array.from(tokensEl.children).forEach((li, i) => {
        const masked = on.has(i);
        li.classList.toggle("is-masked", masked);
        li.firstChild.textContent = masked ? "[MASK]" : words[i];
        li.setAttribute("aria-label", masked ? `position ${i + 1}, masked (${words[i]})` : `position ${i + 1}, ${words[i]}`);
      });
      Array.from(stripEl.children).forEach((s, i) => {
        const h = history[i];
        s.className = "l2p10-ep" + (h ? (h.fresh ? " is-new" : " is-repeat") : "") + (i === history.length - 1 ? " is-cur" : "");
      });
      const distinct = new Set(history.map((h) => h.key)).size;
      rollBtn.disabled = history.length >= EPOCHS;
      allBtn.disabled = history.length >= EPOCHS;
      if (!cur) {
        readout.innerHTML = `No epochs yet. Press <b>Next epoch</b> to feed the sentence in once. ${K} of its ${words.length} tokens get masked, and there are <b>${n(TOTAL)}</b> ways to choose them.`;
        return;
      }
      let s = `Epoch <b>${history.length}</b> of ${EPOCHS}. Masked positions: <b>${cur.mask.map((p) => p + 1).join(", ")}</b>. ` +
        `Distinct masks seen so far: <b>${distinct}</b> of ${n(TOTAL)} possible.`;
      if (mode === "static") {
        s += history.length > COPIES
          ? ` BERT made only ${COPIES} copies, so from epoch ${COPIES + 1} on every mask is a repeat. By epoch ${EPOCHS}, each has come round ${EPOCHS / COPIES} times.`
          : ` BERT made ${COPIES} masked copies in advance, so this count can never go past ${COPIES}.`;
      } else {
        s += " Each epoch draws a fresh mask, so almost every one is new.";
      }
      readout.innerHTML = s;
    }

    function roll() {
      if (history.length >= EPOCHS) return;
      const mask = maskForEpoch(mode, history.length, copies, Math.random, words.length, K);
      const key = maskKey(mask);
      const fresh = !history.some((h) => h.key === key);
      history.push({ mask, key, fresh });
    }

    rollBtn.addEventListener("click", () => {
      roll();
      render();
    });
    allBtn.addEventListener("click", () => {
      while (history.length < EPOCHS) roll();
      render();
    });
    resetBtn.addEventListener("click", () => {
      history = [];
      render();
    });
    modeGroup.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-mode]");
      if (!b) return;
      mode = b.dataset.mode;
      setPressed(modeGroup, b);
      history = [];
      render();
    });
    fig.querySelector(".ix-controls").hidden = false;
    render();
  }

  // Hard one-hot label versus the teacher's soft label for "The movie was fine."
  function initSoft(fig) {
    const hard = [0, 1, 0];
    const soft = JSON.parse(fig.dataset.soft);
    const group = fig.querySelector(".chips");
    const rows = fig.querySelectorAll(".l2p10-soft-row");
    const readout = fig.querySelector(".ix-readout");
    const text = {
      hard:
        "The hard label says one thing only: <b>neutral</b>. It says nothing about whether this sentence leans positive or negative. " +
        "To a hard label, “fine” and “terrible” are just as wrong as each other.",
      soft:
        `The teacher still picks <b>neutral</b> (${f(soft[1], 2)}), but it also gives positive ${f(soft[0], 2)} and negative only ${f(soft[2], 2)}. ` +
        "So this sentence is closer to positive than to negative. That extra information in the small probabilities is Hinton's “dark knowledge”.",
    };
    function show(mode) {
      const vals = mode === "hard" ? hard : soft;
      rows.forEach((row, i) => {
        row.querySelector(".l2p10-soft-fill").style.width = vals[i] * 100 + "%";
        row.querySelector(".l2p10-soft-val").textContent = mode === "hard" ? String(vals[i]) : f(vals[i], 2);
      });
      fig.dataset.mode = mode;
      readout.innerHTML = text[mode];
    }
    group.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-mode]");
      if (!b) return;
      setPressed(group, b);
      show(b.dataset.mode);
    });
    fig.querySelector(".ix-controls").hidden = false;
    show("soft");
  }

  // Slide the student from its early answer to its later one and watch KL fall.
  function initSlider(fig) {
    const def = defs["l2-kl-distill"];
    const { teacher, early, late } = def.input;
    const range = fig.querySelector("input[type=range]");
    const out = fig.querySelector("output");
    const stage = fig.querySelector(".l2p10-slide-bars");
    const termsEl = fig.querySelector(".l2p10-slide-terms");
    const curve = fig.querySelector(".l2p10-curve");
    const readout = fig.querySelector(".ix-readout");
    const N = 50;
    const pts = Array.from({ length: N + 1 }, (_, i) => M.klDivergence(teacher, studentAt(early, late, i / N)).total);
    const kMax = pts[0];
    const CW = 300;
    const CH = 110;
    const cx = (t) => 34 + t * (CW - 50);
    const cy = (k) => 14 + (1 - k / kMax) * (CH - 40);
    const path = pts.map((k, i) => (i ? "L" : "M") + cx(i / N).toFixed(1) + " " + cy(k).toFixed(1)).join(" ");
    curve.innerHTML =
      `<svg viewBox="0 0 ${CW} ${CH}" role="img" aria-label="KL divergence falling as the student moves from early to later">` +
      `<line class="l2p10-axis" x1="34" y1="${cy(0)}" x2="${cx(1)}" y2="${cy(0)}"/>` +
      `<line class="l2p10-axis" x1="34" y1="14" x2="34" y2="${cy(0)}"/>` +
      `<text class="l2p10-tick" x="30" y="${cy(kMax) + 4}" text-anchor="end">${f(kMax, 2)}</text>` +
      `<text class="l2p10-tick" x="30" y="${cy(0) + 4}" text-anchor="end">0</text>` +
      `<text class="l2p10-tick" x="34" y="${CH - 6}">early</text>` +
      `<text class="l2p10-tick" x="${cx(1)}" y="${CH - 6}" text-anchor="end">later</text>` +
      `<path class="l2p10-curve-line" d="${path}"/>` +
      `<circle class="l2p10-curve-dot" r="5" cx="0" cy="0"/>` +
      `</svg>`;
    const dot = curve.querySelector("circle");

    function update() {
      const t = Number(range.value) / 100;
      const s = studentAt(early, late, t);
      const kl = M.klDivergence(teacher, s);
      out.textContent = s.map((x) => f(x, 3)).join(", ");
      stage.innerHTML = barsSvg(teacher, s, "student", []);
      termsEl.innerHTML = CLASSES.map(
        (c, i) => `<span class="l2p10-term"><span>${c}</span><b>${f(kl.terms[i], 4)}</b></span>`
      ).join("") + `<span class="l2p10-term is-total"><span>KL</span><b>${f(kl.total, 4)}</b></span>`;
      dot.setAttribute("cx", cx(t));
      dot.setAttribute("cy", cy(kl.total));
      let note = "";
      if (t === 0) note = ` That is the early student from the worked example. The notes add the rounded terms and get 0.0833; unrounded it is ${f(kl.total, 4)}.`;
      else if (t === 1) note = " That is the later student from the worked example: KL is almost zero.";
      readout.innerHTML = `Student = [${s.map((x) => f(x, 3)).join(", ")}]. KL = <b>${f(kl.total, 4)}</b>.${note}`;
    }
    range.addEventListener("input", update);
    update();
  }

  function init() {
    const mask = document.getElementById("l2p10-mask");
    if (mask) initMask(mask);
    const soft = document.getElementById("l2p10-soft");
    if (soft) initSoft(soft);
    const slide = document.getElementById("l2p10-slider");
    if (slide) initSlider(slide);
    if (REDUCE) document.documentElement.classList.add("l2p10-reduced");
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})(typeof window !== "undefined" ? window : globalThis);
