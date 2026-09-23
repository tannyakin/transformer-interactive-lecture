// ============================================================
// Lecture 2, Part 9: BERT.
// Three walkthroughs (where the 110M parameters live, how many
// tokens MLM touches, the sentiment pipeline) plus the part's
// interactives: the two "bank" sentences, the size picker, the
// input builder, the masking sampler, the NSP pair builder and
// the fine-tuning heads. Every number is computed, never typed.
// ============================================================
(function (root) {
  "use strict";

  const L = (root.Lecture = root.Lecture || {});
  const M = L.math;
  const defs = (L.defs = L.defs || {});
  const n = (x) => M.fmtInt(x);
  const mil = (x, dp) => M.fmt(x / 1e6, dp);
  const argmax = (xs) => xs.reduce((best, x, i) => (x > xs[best] ? i : best), 0);

  /* ---------------------------------------------------------
     Pure data and helpers (tested in Node)
  --------------------------------------------------------- */

  // Small seeded random number generator so the first roll of every
  // sampler, and the MLM panel, look the same on every visit.
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

  function shuffle(xs, rand) {
    const out = xs.slice();
    for (let i = out.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [out[i], out[j]] = [out[j], out[i]];
    }
    return out;
  }

  // Section 9.4 of the notes.
  const SIZES = [
    { name: "BERT-Tiny", L: 2, H: 128, A: 2, params: 4e6, original: false },
    { name: "BERT-Mini", L: 4, H: 256, A: 4, params: 11e6, original: false },
    { name: "BERT-Small", L: 4, H: 512, A: 8, params: 30e6, original: false },
    { name: "BERT-Medium", L: 8, H: 512, A: 8, params: 42e6, original: false },
    { name: "BERT-Base", L: 12, H: 768, A: 12, params: 110e6, original: true },
    { name: "BERT-Large", L: 24, H: 1024, A: 16, params: 340e6, original: true },
  ];

  // Section 9.2: which words can each reading use when it reaches "bank"?
  const BANK = [
    { words: ["I", "went", "to", "the", "bank", "to", "deposit", "my", "paycheck", "."], focus: 4, clues: [6, 8], sense: "a bank for money" },
    { words: ["I", "sat", "on", "the", "bank", "of", "the", "river", "."], focus: 4, clues: [7], sense: "the side of a river" },
  ];

  function bankView(sentence, mode) {
    const states = sentence.words.map((_, i) => {
      if (i === sentence.focus) return "focus";
      const visible = mode === "both" || i < sentence.focus;
      if (!visible) return "hidden";
      return sentence.clues.includes(i) ? "clue" : "seen";
    });
    return { states, resolved: states.includes("clue") };
  }

  // Section 9.5: the input example.
  const INPUT = {
    tokens: ["[CLS]", "my", "dog", "is", "cute", "[SEP]", "he", "likes", "playing", "[SEP]"],
  };
  INPUT.segments = INPUT.tokens.map((_, i) => (i <= INPUT.tokens.indexOf("[SEP]") ? "A" : "B"));
  INPUT.positions = INPUT.tokens.map((_, i) => i);

  // Section 9.6: the MLM recipe applied to a real sentence.
  const MLM = { choose: 0.15, mask: 0.8, random: 0.1, keep: 0.1 };
  const MASK_SENTENCE =
    "the students walked across the campus to the library , where they spent the whole afternoon reading " +
    "about how a model like bert learns the meaning of a word from the words on both sides of it .";
  const RANDOM_WORDS = ["banana", "seven", "river", "quickly", "blue", "yam", "music", "table", "happy", "drive", "cloud", "paper"];

  function sampleMask(tokens, rand, recipe) {
    const r = recipe || MLM;
    const count = Math.round(r.choose * tokens.length);
    const chosen = new Set(shuffle(tokens.map((_, i) => i), rand).slice(0, count));
    return tokens.map((orig, i) => {
      if (!chosen.has(i)) return { orig, shown: orig, kind: "plain" };
      const u = rand();
      if (u < r.mask) return { orig, shown: "[MASK]", kind: "mask" };
      if (u < r.mask + r.random) {
        const pool = RANDOM_WORDS.filter((w) => w !== orig);
        return { orig, shown: pool[Math.floor(rand() * pool.length)], kind: "random" };
      }
      return { orig, shown: orig, kind: "keep" };
    });
  }

  // Section 9.7: NSP pairs. The first pair and "penguins cannot fly" are the notes' examples.
  const NSP_TEXT = [
    ["the man went to the store", "he bought milk"],
    ["the rain started at noon", "we waited under the bridge"],
    ["she opened her laptop", "the lecture notes were already there"],
  ];
  const NSP_RANDOM = ["penguins cannot fly", "the train leaves at six", "rice needs a lot of water"];

  function nspDraw(rand) {
    const i = Math.floor(rand() * NSP_TEXT.length);
    const [a, next] = NSP_TEXT[i];
    if (rand() < 0.5) return { a, b: next, label: "IsNext" };
    const pool = NSP_RANDOM.concat(NSP_TEXT.filter((_, j) => j !== i).map((p) => p[1]));
    return { a, b: pool[Math.floor(rand() * pool.length)], label: "NotNext" };
  }

  function pairTokens(a, b) {
    const A = ["[CLS]"].concat(a.split(" "), ["[SEP]"]);
    const B = b.split(" ").concat(["[SEP]"]);
    return { tokens: A.concat(B), segments: A.map(() => "A").concat(B.map(() => "B")) };
  }

  // Section 9.8: the video's sentiment pipeline, as pure steps.
  function sentimentPipeline(text, maxLen) {
    const lower = text.toLowerCase();
    const tokens = lower.match(/[a-z0-9]+|[^\sa-z0-9]/g) || [];
    const framed = ["[CLS]"].concat(tokens, ["[SEP]"]);
    const pad = Math.max(0, maxLen - framed.length);
    const padded = framed.concat(Array.from({ length: pad }, () => "[PAD]"));
    const positions = padded.map((_, i) => i);
    const segments = padded.map((t) => (t === "[PAD]" ? "B" : "A"));
    return { lower, tokens, framed, pad, padded, positions, segments };
  }

  // Section 9.8: what each fine-tuning head reads from BERT's outputs.
  const HEADS = [
    {
      id: "seq",
      name: "Sequence classification",
      tokens: ["[CLS]", "this", "teddy", "bear", "is", "so", "cute", "!", "[SEP]"],
      use: [0],
      tags: {},
      out: "[CLS] vector (768) → linear layer (768 × 2) → softmax → [P(positive), P(negative)]",
      say: "One label for the whole input, such as sentiment, spam or topic. Only the final [CLS] vector is read.",
    },
    {
      id: "tok",
      name: "Token classification",
      tokens: ["[CLS]", "ada", "lives", "in", "lagos", "[SEP]"],
      use: [1, 2, 3, 4],
      tags: { 1: "person", 2: "nothing", 3: "nothing", 4: "location" },
      out: "every token's vector → the same linear layer → one label per token",
      say: "One label per token. For named entity recognition the labels are person, location, organization or nothing.",
    },
    {
      id: "qa",
      name: "Question answering",
      tokens: ["[CLS]", "who", "trained", "bert", "?", "[SEP]", "bert", "was", "trained", "by", "google", "researchers", "[SEP]"],
      use: [6, 7, 8, 9, 10, 11],
      tags: { 10: "start", 11: "end" },
      out: "each passage token · start vector → start score;  · end vector → end score",
      say: "The input is [CLS] question [SEP] passage [SEP]. Every passage token gets a start score and an end score, and the answer is the span with the best pair.",
    },
    {
      id: "pair",
      name: "Sentence pairs",
      tokens: ["[CLS]", "how", "old", "are", "you", "?", "[SEP]", "what", "is", "your", "age", "?", "[SEP]"],
      use: [0],
      tags: {},
      out: "[CLS] vector → linear layer → duplicate or not duplicate",
      say: "Put both sentences in and classify from [CLS]: does B follow from A, or are these two questions the same?",
    },
    {
      id: "sbert",
      name: "Sentence-BERT embeddings",
      tokens: ["[CLS]", "the", "cat", "sat", "on", "the", "mat", "[SEP]"],
      use: [1, 2, 3, 4, 5, 6],
      tags: {},
      out: "outputs → one sentence vector → compare with other sentences by similarity",
      say: "Fine-tuned so that similar sentences land on similar vectors. This is the foundation of semantic search and the retrieval step in many RAG systems.",
    },
  ];

  const P9 = (L.l2p9 = { rng, shuffle, SIZES, BANK, bankView, INPUT, MLM, MASK_SENTENCE, RANDOM_WORDS, sampleMask, NSP_TEXT, NSP_RANDOM, nspDraw, pairTokens, sentimentPipeline, HEADS });

  /* ---------------------------------------------------------
     9.4 Worked example: where BERT-Base's 110M parameters live
  --------------------------------------------------------- */
  defs["l2-bert-params"] = {
    id: "l2-bert-params",
    title: "Where BERT-Base's 110M parameters live",
    input: { H: 768, ffn: 3072, layers: 12, vocab: 30522, positions: 512, segments: 2 },
    setup:
      "BERT-Base is described by H = 768 and L = 12, its FFN expands to 3,072, and its WordPiece vocabulary " +
      "has 30,522 entries. Using what you learned in Part 4, you will count its weights one matrix at a time " +
      "and see where the famous 110M comes from.",
    setupBlocks: (i) => [
      {
        type: "lines",
        lines: [
          `H (hidden size)     = ${n(i.H)}`,
          `FFN size (4 × H)   = ${n(i.ffn)}`,
          `L (layers)          = ${n(i.layers)}`,
          `vocabulary          = ${n(i.vocab)}`,
          `positions, segments = ${n(i.positions)}, ${n(i.segments)}`,
        ],
      },
    ],
    build(i) {
      const attn = 4 * i.H * i.H;
      const ffn = 2 * i.H * i.ffn;
      const layer = attn + ffn;
      const stack = i.layers * layer;
      const tok = i.vocab * i.H;
      const pos = i.positions * i.H;
      const seg = i.segments * i.H;
      const posSeg = pos + seg;
      const total = stack + tok + posSeg;
      const pieces = [i.layers * attn, i.layers * ffn, tok, posSeg];
      const pieceNames = [`Attention, all ${i.layers} layers`, `FFN, all ${i.layers} layers`, "Token embedding table", "Positions and segments"];
      const biggest = argmax(pieces);
      const ffnRatio = ffn / attn;

      const attnLine = { t: `attention  4 × ${n(i.H)} × ${n(i.H)}   = ${n(attn)}` };
      const ffnLine = { t: `FFN        2 × ${n(i.H)} × ${n(i.ffn)} = ${n(ffn)}` };
      const layerLine = { t: `per layer  ${n(attn)} + ${n(ffn)} = ${n(layer)} ≈ ${mil(layer, 2)} million` };

      return {
        steps: [
          {
            title: "Count one layer's attention weights",
            say:
              `Attention has four square matrices, W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> and W<sub>O</sub>. Each one maps ${n(i.H)} numbers to ${n(i.H)} numbers, ` +
              `so each holds ${n(i.H)} × ${n(i.H)} = ${n(i.H * i.H)} weights. Four of them make ${n(attn)}.`,
            blocks: [
              {
                type: "lines",
                key: "attn",
                value: attn,
                fresh: true,
                lines: ["W_Q", "W_K", "W_V", "W_O"]
                  .map((w) => ({ t: `${w}  ${n(i.H)} × ${n(i.H)} = ${n(i.H * i.H)}` }))
                  .concat([{ t: attnLine.t, hl: true }]),
              },
            ],
          },
          {
            title: "Add the FFN, and total the layer",
            say:
              `The FFN has two matrices: W1 grows each token from ${n(i.H)} to ${n(i.ffn)} numbers and W2 shrinks it back. ` +
              `That is ${n(ffn)} weights, ${M.fmt(ffnRatio, 0)} times the attention. One whole layer holds ${n(layer)}, about ${mil(layer, 2)} million.`,
            predict: {
              ask: `Attention uses ${n(attn)} weights per layer. How does the FFN compare?`,
              choices: ["Fewer than attention", "About the same", "About twice as many"],
              answer: ffnRatio > 1.5 ? 2 : ffnRatio > 0.75 ? 1 : 0,
              why:
                `Each FFN matrix is ${n(i.H)} × ${n(i.ffn)}, four times bigger than one attention matrix. ` +
                "Two of those beat four of the smaller ones, two to one. The FFN is where most of a layer lives.",
              hint: `Compare sizes: one FFN matrix is ${n(i.H)} × ${n(i.ffn)}, one attention matrix is ${n(i.H)} × ${n(i.H)}.`,
            },
            blocks: [
              { type: "lines", lines: [attnLine, Object.assign({ hl: true }, ffnLine), layerLine], key: "ffn", value: ffn, fresh: true },
              {
                type: "bars",
                key: "layerShare",
                labels: ["attention", "FFN"],
                values: [attn / layer, ffn / layer],
                hl: [1],
                dp: 2,
                fresh: true,
              },
            ],
          },
          {
            title: `Stack ${i.layers} layers`,
            say:
              `Every encoder layer has its own copy of these weights, so ${n(i.layers)} layers hold ${n(i.layers)} × ${mil(layer, 2)}M, ` +
              `about ${mil(stack, 0)} million. That is the whole encoder stack.`,
            blocks: [
              { type: "lines", lines: [attnLine, ffnLine, layerLine] },
              {
                type: "lines",
                key: "stack",
                value: stack,
                fresh: true,
                lines: [{ t: `${n(i.layers)} layers  ${n(i.layers)} × ${n(layer)} = ${n(stack)} ≈ ${mil(stack, 0)} million`, hl: true }],
              },
            ],
          },
          {
            title: "Add the token embedding table",
            say:
              `The token embedding is the "gigantic lookup table": one row of ${n(i.H)} numbers for each of the ${n(i.vocab)} WordPiece tokens. ` +
              `That is ${n(tok)} weights, about ${mil(tok, 1)} million, before a single layer has run.`,
            blocks: [
              { type: "lines", lines: [{ t: `${n(i.layers)} layers  ≈ ${mil(stack, 0)} million` }] },
              {
                type: "lines",
                key: "tok",
                value: tok,
                fresh: true,
                lines: [{ t: `tokens     ${n(i.vocab)} × ${n(i.H)} = ${n(tok)} ≈ ${mil(tok, 1)} million`, hl: true }],
              },
            ],
          },
          {
            title: "Add positions and segments",
            say:
              `Learned positions need one row per position, ${n(i.positions)} of them, and segments need one row each for A and B. ` +
              `Together they add only ${n(posSeg)} weights, about ${mil(posSeg, 1)} million.`,
            blocks: [
              {
                type: "lines",
                key: "posSeg",
                value: posSeg,
                fresh: true,
                lines: [
                  { t: `positions  ${n(i.positions)} × ${n(i.H)} = ${n(pos)}` },
                  { t: `segments   ${n(i.segments)} × ${n(i.H)} = ${n(seg)}` },
                  { t: `together   ${n(pos)} + ${n(seg)} = ${n(posSeg)} ≈ ${mil(posSeg, 1)} million`, hl: true },
                ],
              },
            ],
          },
          {
            title: "Add it all up",
            say:
              `${n(stack)} + ${n(tok)} + ${n(posSeg)} = ${n(total)}, about ${mil(total, 0)} million. ` +
              `The biggest single piece is the ${pieceNames[biggest].toLowerCase()}, at ${mil(pieces[biggest], 1)} million.`,
            predict: {
              ask: "Before the total: which piece of BERT-Base holds the most parameters?",
              choices: pieceNames,
              answer: biggest,
              why:
                `The FFN weights across all ${i.layers} layers come to ${mil(pieces[1], 1)} million, more than attention (${mil(pieces[0], 1)} million) ` +
                `and more than the token table (${mil(tok, 1)} million). The embedding table is big, but it is only one table.`,
              hint: "Each layer's FFN was twice its attention. Now multiply by the number of layers and compare with the token table.",
            },
            blocks: [
              {
                type: "lines",
                key: "total",
                value: total,
                fresh: true,
                lines: [
                  { t: `layers     ${n(stack)}` },
                  { t: `tokens     ${n(tok)}` },
                  { t: `pos + seg  ${n(posSeg)}` },
                  { t: `total      ${n(total)} ≈ ${mil(total, 0)} million`, hl: true },
                ],
              },
              {
                type: "bars",
                key: "share",
                label: "share of the total",
                labels: pieceNames.map((p, k) => `${p.replace(`, all ${i.layers} layers`, "").replace(" table", "")} ${mil(pieces[k], 1)}M`),
                values: pieces.map((p) => p / total),
                hl: [biggest],
                dp: 2,
                fresh: true,
              },
            ],
          },
        ],
        takeaway:
          `Add the small bias and layer norm parameters and you land on the famous 110M. The FFN holds two thirds of every layer, ` +
          `and the embedding table alone is about a fifth of the whole model (${mil(tok, 1)} of ${mil(total, 0)} million). ` +
          "Small by today's LLM standards, which is exactly why BERT is still practical.",
      };
    },
  };

  /* ---------------------------------------------------------
     9.6 Worked example: how many tokens get touched
  --------------------------------------------------------- */
  function mlmCounts(i) {
    const chosen = i.choose * i.tokens;
    const mask = i.mask * chosen;
    const random = i.random * chosen;
    const keep = i.keep * chosen;
    return { chosen, mask, random, keep, untouched: i.tokens - chosen };
  }

  // Which cell of the panel's grid plays which role. Fixed seed so it never jumps around.
  function mlmLayout(i) {
    const c = mlmCounts(i);
    const order = shuffle(Array.from({ length: i.tokens }, (_, k) => k), rng(9));
    const kind = new Array(i.tokens).fill("plain");
    order.slice(0, c.mask).forEach((k) => (kind[k] = "mask"));
    order.slice(c.mask, c.mask + c.random).forEach((k) => (kind[k] = "random"));
    order.slice(c.mask + c.random, c.chosen).forEach((k) => (kind[k] = "keep"));
    return kind;
  }
  P9.mlmCounts = mlmCounts;
  P9.mlmLayout = mlmLayout;

  const pct = (p) => M.fmt(p * 100, 0) + "%";

  defs["l2-mlm"] = {
    id: "l2-mlm",
    title: "MLM: how many tokens get touched",
    input: { tokens: 200, choose: 0.15, mask: 0.8, random: 0.1, keep: 0.1 },
    setup:
      "Take a sentence of 200 tokens and run BERT's masking recipe on it: pick 15% at random, then split those " +
      "80 / 10 / 10 between [MASK], a random word and no change at all. The grid shows all 200 tokens.",
    setupBlocks: (i) => [
      { type: "lines", lines: [`tokens      ${n(i.tokens)}`, `choose      ${pct(i.choose)}`, `of those    ${pct(i.mask)} [MASK], ${pct(i.random)} random, ${pct(i.keep)} unchanged`] },
    ],
    panel: (el) => mlmPanel(el),
    build(i) {
      const c = mlmCounts(i);
      const chosenLine = { t: `chosen:      ${pct(i.choose)} of ${n(i.tokens)} = ${n(c.chosen)} tokens` };
      const maskLine = { t: `-> [MASK]:   ${pct(i.mask)} of ${n(c.chosen)}  = ${n(c.mask)} tokens` };
      const randLine = { t: `-> random:   ${pct(i.random)} of ${n(c.chosen)}  =  ${n(c.random)} tokens` };
      const keepLine = { t: `-> unchanged: ${pct(i.keep)} of ${n(c.chosen)} =  ${n(c.keep)} tokens` };
      const maskChoices = [c.chosen, c.mask, Math.round(c.chosen / 2), c.random];
      const predictChoices = [c.mask, c.mask + c.random, c.chosen, i.tokens];

      return {
        steps: [
          {
            title: "Pick 15% of the tokens",
            stage: 1,
            say:
              `15% of ${n(i.tokens)} is ${n(c.chosen)}. These ${n(c.chosen)} positions are the only ones BERT will be asked about. ` +
              `The other ${n(c.untouched)} just sit there as context.`,
            blocks: [{ type: "lines", key: "chosen", value: c.chosen, fresh: true, lines: [Object.assign({ hl: true }, chosenLine)] }],
          },
          {
            title: "Most of them become [MASK]",
            stage: 2,
            say: `80% of the ${n(c.chosen)} chosen tokens are hidden behind [MASK]: ${n(c.mask)} tokens. Not all ${n(c.chosen)}.`,
            predict: {
              ask: `Of the ${n(c.chosen)} chosen tokens, how many actually show up as [MASK]?`,
              choices: maskChoices.map((x) => n(x)),
              answer: maskChoices.indexOf(c.mask),
              why: `Only 80% of the chosen ones: 0.8 × ${n(c.chosen)} = ${n(c.mask)}. The rest are disguised in two other ways.`,
              hint: "The recipe has a second split after choosing. Only 80% of the chosen tokens get the [MASK] treatment.",
            },
            blocks: [{ type: "lines", key: "mask", value: c.mask, fresh: true, lines: [chosenLine, Object.assign({ hl: true }, maskLine)] }],
          },
          {
            title: "The rest: a random word, or no change",
            stage: 3,
            say:
              `10% of ${n(c.chosen)} is ${n(c.random)}, so ${n(c.random)} chosen tokens are swapped for a random word, and ` +
              `${n(c.keep)} are left exactly as they were. Those ${n(c.keep)} look perfectly normal in the input.`,
            blocks: [
              {
                type: "lines",
                key: "split",
                value: [c.mask, c.random, c.keep],
                fresh: true,
                lines: [chosenLine, maskLine, Object.assign({ hl: true }, randLine), Object.assign({ hl: true }, keepLine)],
              },
            ],
          },
          {
            title: "Predict every chosen position",
            stage: 4,
            say:
              `The model must predict the original token at all ${n(c.chosen)} chosen positions, even the ${n(c.keep)} that look normal. ` +
              "It is never told which ones those are.",
            predict: {
              ask: "At how many positions must the model predict the original token?",
              choices: predictChoices.map((x, k) => n(x) + ["  (only the [MASK] ones)", "  ([MASK] and random)", "  (every chosen token)", "  (every token)"][k]),
              answer: predictChoices.indexOf(c.chosen),
              why:
                `All ${n(c.chosen)}. The loss is computed at every chosen position, including the ${n(c.keep)} unchanged ones, ` +
                "so BERT can never relax on a word just because it looks real.",
              hint: "The recipe says the model predicts the original token at every chosen position, whatever happened to it.",
            },
            blocks: [
              { type: "lines", lines: [chosenLine, maskLine, randLine, keepLine] },
              { type: "lines", key: "predict", value: c.chosen, fresh: true, lines: [{ t: `must predict: ${n(c.mask)} + ${n(c.random)} + ${n(c.keep)} = ${n(c.chosen)} positions`, hl: true }] },
            ],
          },
          {
            title: "Look at the whole sentence",
            stage: 4,
            say:
              `Out of all ${n(i.tokens)} tokens, ${pct(c.mask / i.tokens)} show [MASK], only ${M.fmt((c.random / i.tokens) * 100, 1)}% are random words, ` +
              `and ${pct(c.untouched / i.tokens)} are never asked about. BERT learns from only ${pct(c.chosen / i.tokens)} of what it reads.`,
            blocks: [
              {
                type: "bars",
                key: "shares",
                label: "share of all tokens",
                labels: ["[MASK]", "random", "unchanged", "not chosen"],
                values: [c.mask, c.random, c.keep, c.untouched].map((x) => x / i.tokens),
                hl: [3],
                dp: 3,
                fresh: true,
              },
            ],
          },
        ],
        takeaway:
          `The model predicts all ${n(c.chosen)} chosen tokens, even the ${n(c.keep)} that look normal, so it has to build a good representation ` +
          `of every word. The price: ${pct(c.untouched / i.tokens)} of the sentence gives no training signal at all. A decoder learns from 100% of its tokens.`,
      };
    },
  };

  /* ---------------------------------------------------------
     9.8 Worked example: sentiment extraction, step by step
  --------------------------------------------------------- */
  defs["l2-bert-sentiment"] = {
    id: "l2-bert-sentiment",
    title: "Sentiment extraction, step by step",
    input: { text: "This teddy bear is SO CUTE!", maxLen: 15, sentiment: 1 },
    setup:
      "The lecture video follows one sentence all the way through BERT and out the other side as a sentiment label. " +
      "You will watch the same ten steps, grouped into seven, with an uncased model.",
    setupBlocks: (i) => [{ type: "lines", lines: [{ t: `"${i.text}"`, hl: true }] }],
    build(i) {
      const p = sentimentPipeline(i.text, i.maxLen);
      const len = p.padded.length;
      const cols = p.padded.map((_, k) => k);
      const specials = cols.filter((k) => p.padded[k] !== "[PAD]" && /^\[/.test(p.padded[k]));
      const pads = cols.filter((k) => p.padded[k] === "[PAD]");
      const tokRow = p.padded;
      const embRow = p.padded.map((t) => "E(" + t + ")");
      const nA = p.segments.filter((s) => s === "A").length;
      const nB = len - nA;
      const padChoices = [p.pad - 2, p.pad - 1, p.pad, p.pad + 1];
      const outChoices = ["[CLS]", "cute", "!", "[SEP]"];

      return {
        steps: [
          {
            title: "Lowercase, then split into tokens",
            say:
              `An uncased model first lowercases everything, so "SO CUTE" becomes "so cute". Then the tokenizer cuts it into ${p.tokens.length} tokens. ` +
              'Notice the "!" is a token of its own.',
            blocks: [
              { type: "lines", key: "lower", value: p.lower, lines: [{ t: `"${i.text}"` }, { t: `"${p.lower}"`, hl: true }] },
              { type: "matrix", key: "tokens", rows: [p.tokens], fresh: true },
            ],
          },
          {
            title: "Add [CLS] and [SEP], then pad to 15",
            say:
              `[CLS] goes at the start as a placeholder for the sentiment, and [SEP] at the end. That makes ${p.framed.length} tokens, ` +
              `so ${p.pad} <abbr data-term="[PAD]">[PAD]</abbr> tokens fill it up to ${i.maxLen}. Padding lets sentences of different lengths travel together in one batch.`,
            predict: {
              ask: `After adding [CLS] and [SEP], how many [PAD] tokens are needed to reach a length of ${i.maxLen}?`,
              choices: padChoices.map((x) => n(x)),
              answer: 2,
              why: `${p.tokens.length} words and symbols plus [CLS] and [SEP] make ${p.framed.length}, and ${i.maxLen} − ${p.framed.length} = ${p.pad}.`,
              hint: `Count the ${p.tokens.length} tokens from the last step, add the two special tokens, then subtract from ${i.maxLen}.`,
            },
            blocks: [
              {
                type: "matrix",
                key: "padded",
                value: p.pad,
                rows: [tokRow],
                colLabels: cols.map(String),
                hl: specials.map((k) => [0, k]),
                muted: pads.map((k) => [0, k]),
                fresh: true,
              },
            ],
          },
          {
            title: "Look up a token embedding for each",
            say:
              `Each of the ${len} tokens pulls its own row out of the gigantic lookup table. The ${p.pad} [PAD] tokens all pull the same row, ` +
              "because they are the same token.",
            blocks: [
              {
                type: "matrix",
                key: "embed",
                rows: [tokRow, embRow],
                rowLabels: ["token", "token emb"],
                muted: pads.map((k) => [1, k]),
                fresh: true,
              },
            ],
          },
          {
            title: `Add position embeddings, 0 to ${len - 1}`,
            say: `Position ${len - 1} is the last one. BERT learned one embedding per position, up to 512, and adds the right one to each token.`,
            blocks: [
              {
                type: "matrix",
                key: "positions",
                rows: [tokRow, embRow, p.positions],
                rowLabels: ["token", "token emb", "+ position"],
                hl: cols.map((k) => [2, k]),
                fresh: true,
              },
            ],
          },
          {
            title: "Add segment embeddings",
            say:
              `The slide labels the real sentence A (${nA} tokens, [CLS] and [SEP] included) and the padding B (${nB} tokens). ` +
              "In practice an attention mask switches the padding off, so no real token attends to it and its segment label ends up not mattering.",
            blocks: [
              {
                type: "matrix",
                key: "segments",
                value: [nA, nB],
                rows: [tokRow, embRow, p.positions, p.segments],
                rowLabels: ["token", "token emb", "+ position", "+ segment"],
                hl: cols.map((k) => [3, k]),
                fresh: true,
              },
            ],
          },
          {
            title: "Sum them and run the encoder",
            say:
              `The three embeddings are added into one position- and segment-aware vector per token, so ${len} vectors go in. ` +
              `The pretrained BERT encoder mixes them with bidirectional attention, and ${len} output vectors come out.`,
            blocks: [
              {
                type: "lines",
                key: "sum",
                value: len,
                fresh: true,
                lines: [0, 1, 2]
                  .map((k) => ({ t: `${p.padded[k].padEnd(6)} E(${p.padded[k]}) + P(${k}) + S(${p.segments[k]})` }))
                  .concat([{ t: "  ⋮" }, { t: `${len} input vectors → pretrained BERT encoder → ${len} output vectors`, hl: true }]),
              },
            ],
          },
          {
            title: "Read the answer from [CLS]",
            say:
              `The output vector at [CLS] goes through a small FFN trained for sentiment extraction. Out comes the sentiment: ${i.sentiment}, positive.`,
            predict: {
              ask: "Which output vector does the sentiment head read?",
              choices: outChoices,
              answer: outChoices.indexOf("[CLS]"),
              why:
                "The [CLS] position was put there as a placeholder for the whole input. After every layer of bidirectional attention, " +
                "its vector has seen every other token, so it can stand for the sentence.",
              hint: "Go back to step 2: which token was added at the start as a placeholder for the sentiment?",
            },
            blocks: [
              {
                type: "matrix",
                rows: [tokRow, tokRow.map((t) => "h(" + t + ")")],
                rowLabels: ["token", "output"],
                hl: [[1, 0]],
                muted: cols.slice(1).map((k) => [1, k]),
              },
              {
                type: "lines",
                key: "sentiment",
                value: i.sentiment,
                fresh: true,
                lines: [{ t: `h([CLS]) → small FFN (sentiment head) → ${i.sentiment}  (positive)`, hl: true }],
              },
            ],
          },
        ],
        takeaway:
          "During fine-tuning, only that last small FFN is new. Everything below it starts from BERT's pretrained weights, " +
          "which is why a little labelled data goes such a long way.",
      };
    },
  };

  if (typeof module !== "undefined" && module.exports) module.exports = { defs, P9 };
  if (typeof document === "undefined") return;

  /* =========================================================
     Browser side
  ========================================================= */
  const REDUCE = root.matchMedia && root.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const SVG = "http://www.w3.org/2000/svg";

  function h(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }
  function s(tag, attrs) {
    const e = document.createElementNS(SVG, tag);
    Object.keys(attrs || {}).forEach((k) => e.setAttribute(k, attrs[k]));
    return e;
  }
  const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  function chipGroup(box, onPick) {
    const chips = Array.from(box.querySelectorAll(".chip"));
    chips.forEach((c) =>
      c.addEventListener("click", () => {
        chips.forEach((x) => x.setAttribute("aria-pressed", String(x === c)));
        onPick(c);
      })
    );
    return chips;
  }

  /* ---------- MLM walkthrough panel: the 200 token grid ---------- */
  function mlmPanel(el) {
    const COLS = 20;
    const CELL = 15;
    let layoutFor = null;
    let kinds = null;
    el.classList.add("l2p9-mlm-panel");
    const svg = s("svg", { role: "img" });
    const caption = h("p", "l2p9-mlm-caption");
    const legend = h(
      "ul",
      "l2p9-legend",
      '<li><span class="l2p9-key is-mask"></span>[MASK]</li>' +
        '<li><span class="l2p9-key is-random"></span>random word</li>' +
        '<li><span class="l2p9-key is-keep"></span>unchanged</li>' +
        '<li><span class="l2p9-key is-plain"></span>not chosen</li>'
    );
    el.append(svg, caption, legend);
    return {
      render(index, step, input) {
        if (layoutFor !== input) {
          layoutFor = input;
          kinds = mlmLayout(input);
          const rows = Math.ceil(input.tokens / COLS);
          svg.setAttribute("viewBox", `0 0 ${COLS * CELL} ${rows * CELL}`);
          svg.replaceChildren();
          kinds.forEach((_, k) => {
            const g = s("g", { transform: `translate(${(k % COLS) * CELL} ${Math.floor(k / COLS) * CELL})` });
            g.append(s("rect", { x: 1.5, y: 1.5, width: CELL - 3, height: CELL - 3, rx: 2 }), s("line", { x1: 3.5, y1: CELL - 3.5, x2: CELL - 3.5, y2: 3.5 }));
            svg.append(g);
          });
        }
        const stage = step.kind === "setup" ? 0 : step.stage || 4;
        const c = mlmCounts(input);
        Array.from(svg.children).forEach((g, k) => {
          const kind = kinds[k];
          let cls = "l2p9-cell";
          if (kind !== "plain" && stage >= 1) cls += " is-chosen";
          if (kind === "mask" && stage >= 2) cls += " is-mask";
          if (kind === "random" && stage >= 3) cls += " is-random";
          if (kind === "keep" && stage >= 3) cls += " is-keep";
          if (kind !== "plain" && stage >= 4) cls += " is-target";
          g.setAttribute("class", cls);
        });
        const text = [
          `All ${n(input.tokens)} tokens, before anything happens.`,
          `${n(c.chosen)} of ${n(input.tokens)} tokens chosen (outlined).`,
          `${n(c.mask)} of the chosen tokens now show [MASK].`,
          `${n(c.mask)} [MASK], ${n(c.random)} random words, ${n(c.keep)} unchanged.`,
          `All ${n(c.chosen)} outlined tokens are prediction targets.`,
        ][stage];
        caption.textContent = text;
        svg.setAttribute("aria-label", "Grid of " + input.tokens + " tokens. " + text);
      },
    };
  }

  /* ---------- 9.2 the two "bank" sentences ---------- */
  function initBank(fig) {
    const stage = fig.querySelector("[data-l2p9-bank]");
    const readout = fig.querySelector(".ix-readout");
    let mode = "left";
    function render() {
      stage.replaceChildren();
      let resolvedCount = 0;
      BANK.forEach((sent) => {
        const v = bankView(sent, mode);
        if (v.resolved) resolvedCount++;
        const line = h("p", "l2p9-bank-line");
        sent.words.forEach((w, k) => {
          const st = v.states[k];
          const tok = h("span", "l2p9-word is-" + st, esc(w));
          if (st === "hidden") tok.append(h("span", "sr-only", " (not read yet)"));
          if (st === "clue") tok.append(h("span", "sr-only", " (clue)"));
          line.append(tok, document.createTextNode(" "));
        });
        const verdict = h(
          "p",
          "l2p9-bank-verdict" + (v.resolved ? " is-yes" : ""),
          v.resolved
            ? `Resolved: <b>${sent.sense}</b>, thanks to "${sent.clues.map((k) => sent.words[k]).join('" and "')}".`
            : "Not resolved: at “bank” it has only seen “" + sent.words.slice(0, sent.focus).join(" ") + "”."
        );
        const row = h("div", "l2p9-bank-row");
        row.append(line, verdict);
        stage.append(row);
      });
      readout.innerHTML =
        mode === "left"
          ? "Reading left to right, the model reaches “bank” with only the four words before it, and they are the same shape in both sentences. " +
            "<b>It cannot tell the two meanings apart.</b> The clue words have not been read yet."
          : `Reading both sides at once, the clue words after “bank” are in view. <b>${resolvedCount} of ${BANK.length} sentences are resolved.</b> ` +
            "This is what BERT's encoder sees in every layer.";
    }
    chipGroup(fig.querySelector(".chips"), (c) => {
      mode = c.dataset.mode;
      render();
    });
    render();
  }

  /* ---------- 9.4 the six BERT sizes ---------- */
  function initSizes(fig) {
    const list = fig.querySelector("[data-l2p9-sizes]");
    const tower = fig.querySelector("[data-l2p9-tower]");
    const readout = fig.querySelector(".ix-readout");
    const maxP = Math.max(...SIZES.map((m) => m.params));
    const maxH = Math.max(...SIZES.map((m) => m.H));
    const maxL = Math.max(...SIZES.map((m) => m.L));
    const W = 320;
    const LAYER = 8;
    const GAP = 2;
    const TOP = 6;
    const HGT = TOP + maxL * (LAYER + GAP) + 22;
    let current = SIZES.findIndex((m) => m.name === "BERT-Base");
    const buttons = SIZES.map((m, k) => {
      const b = h("button", "l2p9-size");
      b.type = "button";
      b.setAttribute("aria-pressed", "false");
      b.innerHTML =
        `<span class="l2p9-size-name">${m.name}</span>` +
        `<span class="l2p9-size-lha"><span>L ${m.L}</span><span>H ${n(m.H)}</span><span>A ${m.A}</span></span>` +
        `<span class="l2p9-size-bar" aria-hidden="true"><span style="width:${((m.params / maxP) * 100).toFixed(2)}%"></span></span>` +
        `<span class="l2p9-size-p">${mil(m.params, 0)}M</span>`;
      b.addEventListener("click", () => select(k));
      list.append(b);
      return b;
    });

    function drawTower(m) {
      const scale = (W - 20) / maxH;
      const w = m.H * scale;
      const x0 = (W - w) / 2;
      const svg = s("svg", { viewBox: `0 0 ${W} ${HGT}`, role: "img", "aria-label": `${m.name}: ${m.L} layers, each ${m.H} wide, split into ${m.A} heads` });
      for (let l = 0; l < m.L; l++) {
        const y = HGT - 22 - (l + 1) * (LAYER + GAP);
        svg.append(s("rect", { class: "l2p9-layer", x: x0, y, width: w, height: LAYER, rx: 1.5 }));
        for (let a = 1; a < m.A; a++) {
          const x = x0 + (w * a) / m.A;
          svg.append(s("line", { class: "l2p9-head", x1: x, x2: x, y1: y + 1.5, y2: y + LAYER - 1.5 }));
        }
      }
      const label = s("text", { class: "l2p9-tower-label", x: W / 2, y: HGT - 6, "text-anchor": "middle" });
      label.textContent = `${m.L} layers · H = ${n(m.H)} wide · ${m.A} heads`;
      svg.append(label);
      tower.replaceChildren(svg);
    }

    function select(k) {
      current = k;
      const m = SIZES[k];
      buttons.forEach((b, j) => b.setAttribute("aria-pressed", String(j === k)));
      drawTower(m);
      readout.innerHTML =
        `<b>${m.name}</b> stacks <span class="math">L = ${m.L}</span> encoder layers. Each token is a vector of <span class="math">H = ${n(m.H)}</span> numbers, ` +
        `and <span class="math">A = ${m.A}</span> attention heads run side by side in every layer. The FFN inside each layer expands to ` +
        `<span class="math">4 × H = ${n(4 * m.H)}</span>. About <b>${mil(m.params, 0)}M</b> parameters, ` +
        `${M.fmt(m.params / SIZES[SIZES.length - 1].params * 100, 0)}% of BERT-Large. ` +
        (m.original ? "One of the two original models." : "One of the smaller models added later (Turc et al., 2019).");
    }
    select(current);
  }

  /* ---------- 9.5 the input builder ---------- */
  function initInput(fig) {
    const stage = fig.querySelector("[data-l2p9-input]");
    const readout = fig.querySelector(".ix-readout");
    const toggles = Array.from(fig.querySelectorAll("[data-layer]"));
    const on = { token: true, segment: false, position: false };
    let focus = 2;
    const cols = INPUT.tokens.map((t, k) => {
      const b = h("button", "l2p9-col");
      b.type = "button";
      b.addEventListener("click", () => {
        focus = k;
        render();
      });
      stage.append(b);
      return b;
    });
    const parts = (k) => {
      const out = [];
      if (on.token) out.push(`E(${INPUT.tokens[k]})`);
      if (on.segment) out.push(`E<sub>${INPUT.segments[k]}</sub>`);
      if (on.position) out.push(`E<sub>${INPUT.positions[k]}</sub>`);
      return out;
    };
    function render() {
      cols.forEach((b, k) => {
        const seg = INPUT.segments[k];
        b.setAttribute("aria-pressed", String(k === focus));
        b.setAttribute("aria-label", `Token ${k}: ${INPUT.tokens[k]}, segment ${seg}, position ${k}`);
        b.innerHTML =
          `<span class="l2p9-cell-t${/^\[/.test(INPUT.tokens[k]) ? " is-special" : ""}">${esc(INPUT.tokens[k])}</span>` +
          `<span class="l2p9-cell-e is-token${on.token ? "" : " is-off"}">E(${esc(INPUT.tokens[k])})</span>` +
          `<span class="l2p9-cell-e is-seg is-${seg}${on.segment ? "" : " is-off"}"><span class="l2p9-op">+</span>${seg}</span>` +
          `<span class="l2p9-cell-e is-pos${on.position ? "" : " is-off"}"><span class="l2p9-op">+</span>${INPUT.positions[k]}</span>` +
          `<span class="l2p9-cell-sum">= <b>${parts(k).length}</b> part${parts(k).length === 1 ? "" : "s"}</span>`;
      });
      const k = focus;
      const list = parts(k);
      const missing = [];
      if (!on.segment) missing.push("segment");
      if (!on.position) missing.push("position");
      readout.innerHTML =
        `Token ${k}, “${esc(INPUT.tokens[k])}”: input = ${list.join(" + ")}. ` +
        (missing.length
          ? `Without the ${missing.join(" and ")} embedding, BERT cannot tell ${missing.includes("position") ? "where this token sits" : ""}${missing.length === 2 ? " or " : ""}${missing.includes("segment") ? "which sentence it belongs to" : ""}.`
          : `One vector that knows the word, that it sits in segment ${INPUT.segments[k]}, and that it is at position ${k}.`);
    }
    toggles.forEach((t) =>
      t.addEventListener("click", () => {
        const layer = t.dataset.layer;
        on[layer] = !on[layer];
        t.setAttribute("aria-pressed", String(on[layer]));
        render();
      })
    );
    render();
  }

  /* ---------- 9.6 the masking sampler ---------- */
  function initMask(fig) {
    const stage = fig.querySelector("[data-l2p9-mask]");
    const readout = fig.querySelector(".ix-readout");
    const tally = fig.querySelector("[data-l2p9-tally]");
    const tokens = MASK_SENTENCE.split(" ");
    const totals = { mask: 0, random: 0, keep: 0, rolls: 0 };
    let rand = rng(2018);
    const label = { mask: "shown as [MASK]", random: "swapped for a random word", keep: "left unchanged" };
    function roll() {
      const out = sampleMask(tokens, rand);
      rand = Math.random;
      const c = { mask: 0, random: 0, keep: 0 };
      stage.replaceChildren();
      out.forEach((t) => {
        const cell = h("span", "l2p9-mtok is-" + t.kind);
        cell.append(h("span", "l2p9-mtok-shown", esc(t.shown)));
        if (t.kind !== "plain") {
          c[t.kind]++;
          cell.append(h("span", "l2p9-mtok-target", "predict: " + esc(t.orig)));
          cell.append(h("span", "sr-only", ", " + label[t.kind]));
        }
        stage.append(cell, document.createTextNode(" "));
      });
      const chosen = c.mask + c.random + c.keep;
      ["mask", "random", "keep"].forEach((k) => (totals[k] += c[k]));
      totals.rolls++;
      readout.innerHTML =
        `This roll chose <b>${chosen} of ${tokens.length}</b> tokens (15%): ${c.mask} [MASK], ${c.random} random, ${c.keep} unchanged. ` +
        `The model must predict all <b>${chosen}</b>, and the other ${tokens.length - chosen} give it nothing to learn from.`;
      const all = totals.mask + totals.random + totals.keep;
      tally.textContent =
        `After ${totals.rolls} roll${totals.rolls === 1 ? "" : "s"}: ` +
        `${M.fmt((totals.mask / all) * 100, 0)}% [MASK], ${M.fmt((totals.random / all) * 100, 0)}% random, ${M.fmt((totals.keep / all) * 100, 0)}% unchanged` +
        (totals.rolls < 20 ? ". Keep rolling and watch it settle toward 80 / 10 / 10." : ".");
    }
    fig.querySelector("[data-l2p9-reroll]").addEventListener("click", roll);
    fig.querySelector("[data-l2p9-reroll-many]").addEventListener("click", () => {
      for (let k = 0; k < 49; k++) {
        const out = sampleMask(tokens, Math.random);
        out.forEach((t) => {
          if (t.kind !== "plain") totals[t.kind]++;
        });
        totals.rolls++;
      }
      roll();
    });
    roll();
  }

  /* ---------- 9.7 the NSP pair builder ---------- */
  function initNsp(fig) {
    const stage = fig.querySelector("[data-l2p9-nsp]");
    const readout = fig.querySelector(".ix-readout");
    const chips = chipGroup(fig.querySelector(".chips"), (c) => show({ a: NSP_TEXT[0][0], b: c.dataset.b, label: c.dataset.b === NSP_TEXT[0][1] ? "IsNext" : "NotNext" }, false));
    function show(pair, drawn) {
      const { tokens, segments } = pairTokens(pair.a, pair.b);
      stage.replaceChildren();
      const seq = h("div", "l2p9-nsp-seq");
      tokens.forEach((t, k) => {
        const cell = h("span", "l2p9-nsp-tok is-" + segments[k]);
        cell.append(h("span", "l2p9-nsp-word" + (/^\[/.test(t) ? " is-special" : ""), esc(t)), h("span", "l2p9-nsp-seg", segments[k]));
        if (k === 0) cell.append(h("span", "l2p9-nsp-label is-" + pair.label.toLowerCase(), pair.label));
        seq.append(cell);
      });
      stage.append(seq);
      const truth = NSP_TEXT.find((p) => p[0] === pair.a)[1];
      readout.innerHTML =
        (drawn ? "Drawn from the text: " : "") +
        (pair.label === "IsNext"
          ? `“${esc(pair.b)}” really did follow “${esc(pair.a)}”, so the label is <b>IsNext</b>.`
          : `“${esc(pair.b)}” came from somewhere else (the real next sentence was “${esc(truth)}”), so the label is <b>NotNext</b>.`) +
        " The model predicts it from the [CLS] vector, and nobody had to label anything by hand.";
    }
    fig.querySelector("[data-l2p9-draw]").addEventListener("click", () => {
      chips.forEach((c) => c.setAttribute("aria-pressed", "false"));
      show(nspDraw(Math.random), true);
    });
    show({ a: NSP_TEXT[0][0], b: NSP_TEXT[0][1], label: "IsNext" }, false);
  }

  /* ---------- 9.8 the fine-tuning heads ---------- */
  function initHeads(fig) {
    const stage = fig.querySelector("[data-l2p9-heads]");
    const readout = fig.querySelector(".ix-readout");
    function show(id) {
      const head = HEADS.find((x) => x.id === id);
      stage.replaceChildren();
      const row = h("div", "l2p9-heads-row");
      head.tokens.forEach((t, k) => {
        const used = head.use.includes(k);
        const col = h("div", "l2p9-hcol" + (used ? " is-used" : ""));
        col.append(
          h("span", "l2p9-htag" + (head.tags[k] ? " has-tag is-" + head.tags[k] : ""), head.tags[k] ? esc(head.tags[k]) : "&nbsp;"),
          h("span", "l2p9-hout", used ? "▲" : ""),
          h("span", "l2p9-htok" + (/^\[/.test(t) ? " is-special" : ""), esc(t))
        );
        if (used) col.append(h("span", "sr-only", " (read by the head)"));
        row.append(col);
      });
      const enc = h("div", "l2p9-henc", "pretrained BERT encoder");
      const body = h("div", "l2p9-heads-body");
      body.append(row, enc);
      stage.append(h("p", "l2p9-hflow math", esc(head.out)), body);
      readout.innerHTML = `<b>${head.name}.</b> ${esc(head.say)}`;
    }
    chipGroup(fig.querySelector(".chips"), (c) => show(c.dataset.head));
    show("seq");
  }

  function init() {
    const run = (sel, fn) => document.querySelectorAll(sel).forEach((el) => fn(el));
    run("#l2p9-bank", initBank);
    run("#l2p9-sizes", initSizes);
    run("#l2p9-input", initInput);
    run("#l2p9-mask", initMask);
    run("#l2p9-nsp", initNsp);
    run("#l2p9-heads", initHeads);
    if (REDUCE) document.documentElement.classList.add("l2p9-reduce");
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})(typeof window !== "undefined" ? window : globalThis);
