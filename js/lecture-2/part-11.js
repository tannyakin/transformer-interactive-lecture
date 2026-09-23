// ============================================================
// Lecture 2, Part 11: the bridge to modern LLMs. One walkthrough
// follows a single token through a model like Llama 3, and every
// step links back to the part of this lecture that taught it.
// ============================================================
(function (root) {
  "use strict";

  const L = (root.Lecture = root.Lecture || {});
  const defs = (L.defs = L.defs || {});

  const link = (href, text) => `<a href="${href}">${text}</a>`;

  // The notes' recipe, one row per line. `stage` says which walkthrough step lights it up.
  const RECIPE = [
    { id: "1", text: "Tokenize the text into token IDs.", stage: 1 },
    { id: "2", text: "Look up each token's embedding vector. (No position added here.)", stage: 1 },
    { id: "3a", text: "RMSNorm the stream.", stage: 2, layer: true, href: "#p3", part: "Part 3.4, pre-norm 3.5" },
    { id: "3b", text: "Compute Q, K, V. Q has many heads, K and V fewer.", stage: 2, layer: true, href: "#p7", part: "Part 7, GQA" },
    { id: "3c", text: "Rotate Q and K with RoPE by their positions.", stage: 3, layer: true, href: "#p2-6", part: "Part 2.6" },
    { id: "3d", text: "Attention with the causal mask; past K and V come from the KV cache during generation.", stage: 3, layer: true, href: "#p8", part: "Part 8.1, Part 7.2" },
    { id: "3e", text: "Add the result back into the stream.", stage: 4, layer: true, href: "#p3", part: "Part 3.2, residual" },
    { id: "3f", text: "RMSNorm the stream again.", stage: 4, layer: true, href: "#p3", part: "Part 3.4" },
    { id: "3g", text: "SwiGLU feedforward network (or an MoE router).", stage: 4, layer: true, href: "#p4", part: "Part 4.6" },
    { id: "3h", text: "Add the result back into the stream.", stage: 4, layer: true, href: "#p3", part: "Part 3.2" },
    { id: "4", text: "Final RMSNorm.", stage: 5 },
    { id: "5", text: "Linear layer to a score for every word in the vocabulary.", stage: 5 },
    { id: "6", text: "Softmax to probabilities, pick the next token.", stage: 5 },
    { id: "7", text: "Append it, and repeat from step 3 for the new token only.", stage: 6 },
  ];

  const recipeLines = (stage) =>
    RECIPE.filter((r) => r.stage === stage).map((r) => ({ t: `${(r.id + ".").padEnd(4, " ")}${r.text}`, hl: true }));

  defs["l2-llama-token"] = {
    id: "l2-llama-token",
    title: "Following one token through a modern LLM",
    input: { layers: "N" },
    setup:
      "Here is what happens to a token in a model like Llama 3, using every idea from Lectures 1 and 2. " +
      "Each step names the part of this lecture that taught it, so you can jump back if a piece feels shaky.",
    panel: llamaPanel,
    build() {
      return {
        steps: [
          {
            title: "Text in: tokenize and embed",
            say:
              "The text is split into tokens, and each token ID picks out its row of the embedding table. " +
              "Notice what is missing: no position vector is added here, unlike the 2017 design.",
            blocks: [
              { type: "lines", key: "s1", fresh: true, lines: recipeLines(1) },
              {
                type: "note",
                html: `In 2017 a sinusoidal vector was added right here (${link("#p2-2", "Part 2.2")}). Modern models put position somewhere else. Keep that in mind.`,
              },
            ],
          },
          {
            title: "Normalize first, then make Q, K and V",
            say:
              "Each of the N layers starts by normalizing the stream with RMSNorm. That is pre-norm: normalize before the sublayer, not after. " +
              "Then the layer makes queries, keys and values, with many query heads but fewer key and value heads. That is GQA.",
            blocks: [
              { type: "lines", key: "s2", fresh: true, lines: recipeLines(2) },
              {
                type: "note",
                html: `RMSNorm and pre-norm: ${link("#p3", "Part 3")}. Sharing K and V heads to shrink the cache: ${link("#p7", "Part 7")}.`,
              },
            ],
          },
          {
            title: "Put position in, then attend",
            say:
              "Position finally enters here: Q and K are rotated by RoPE according to where their tokens sit, " +
              "so the attention score depends on the distance between them. Then attention runs with the causal mask, " +
              "and during generation the keys and values of earlier tokens come straight from the KV cache.",
            predict: {
              ask: "No position was added to the embedding. So where does the model learn the order of the tokens?",
              choices: ["It doesn't, order is lost", "Inside every layer, by rotating Q and K", "Only once, at the very last layer"],
              answer: 1,
              why:
                "RoPE rotates the queries and keys in every layer, by an angle set by each token's position. " +
                "The dot product then depends on how far apart two tokens are, which is exactly what attention needs.",
              hint: "Think back to RoPE in Part 2.6. What did it act on, and where?",
            },
            blocks: [
              { type: "lines", key: "s3", fresh: true, lines: recipeLines(3) },
              {
                type: "note",
                html: `RoPE: ${link("#p2-6", "Part 2.6")}. The causal mask: ${link("#p8", "Part 8")}. The KV cache: ${link("#p7", "Part 7")}.`,
              },
            ],
          },
          {
            title: "Add back, normalize, feedforward, add back",
            say:
              "The attention output is added back into the stream, the residual connection. The stream is normalized again, " +
              "passed through a SwiGLU feedforward network (or an MoE router that picks a few expert networks), " +
              "and added back once more. That is one full layer, and the model repeats it N times.",
            blocks: [
              { type: "lines", key: "s4", fresh: true, lines: recipeLines(4) },
              {
                type: "note",
                html: `Residual connections: ${link("#p3", "Part 3")}. SwiGLU and Mixture of Experts: ${link("#p4", "Part 4")}.`,
              },
            ],
          },
          {
            title: "Out: a score for every word, then softmax",
            say:
              "After the last layer comes one final RMSNorm. Then the output side, which Lecture 1 did not cover: " +
              "the last vector is multiplied by a big matrix, d_model × vocabulary size (often 100,000+ words), " +
              "to get one score per word. Softmax turns those scores into probabilities, and the model picks the next token.",
            blocks: [
              { type: "lines", key: "s5", fresh: true, lines: recipeLines(5) },
              { type: "note", html: "Softmax here is the same softmax you used for attention weights, just over the whole vocabulary instead of a handful of keys." },
            ],
          },
          {
            title: "Append the token and go again",
            say:
              "The chosen token is appended to the text, and the model runs again from step 3, but for the new token only. " +
              "Every earlier token's keys and values are already sitting in the KV cache, so nothing is recomputed.",
            predict: {
              ask: "To write the next token, what does the model need to push through the layers?",
              choices: ["The whole sequence again, from scratch", "Only the new token; past K and V come from the cache"],
              answer: 1,
              why:
                "Because of the causal mask, earlier tokens never look at later ones, so their keys and values never change. " +
                "The model stores them once in the KV cache and only computes the new token.",
              hint: "Can an earlier token's keys and values change when a later token arrives? Remember the causal mask.",
            },
            blocks: [
              { type: "lines", key: "s6", fresh: true, lines: recipeLines(6) },
              { type: "note", html: `This is why the KV cache, and shrinking it with GQA, matters so much: ${link("#p7", "Part 7")}.` },
            ],
          },
        ],
        takeaway:
          "Every line of that recipe is something you have now seen, worked out by hand. Almost every piece has been upgraded since 2017, " +
          "but the skeleton is the same: attention, residual connections, normalization and feedforward layers, stacked N times.",
      };
    },
  };

  // The recipe as a flow, with the lines for the current step lit up.
  function llamaPanel(el) {
    const row = (r) =>
      `<li class="l2p11-flow-row" data-stage="${r.stage}"><span class="l2p11-flow-id">${r.id}</span>` +
      `<span class="l2p11-flow-t">${r.text}</span></li>`;
    el.innerHTML =
      `<div class="l2p11-flow" aria-hidden="true">` +
      `<ol class="l2p11-flow-list">${RECIPE.filter((r) => !r.layer && r.stage <= 1).map(row).join("")}</ol>` +
      `<div class="l2p11-flow-layer"><span class="l2p11-flow-rep">repeat for each of the N layers</span>` +
      `<ol class="l2p11-flow-list">${RECIPE.filter((r) => r.layer).map(row).join("")}</ol></div>` +
      `<ol class="l2p11-flow-list">${RECIPE.filter((r) => !r.layer && r.stage >= 5).map(row).join("")}</ol>` +
      `</div>`;
    const rows = el.querySelectorAll(".l2p11-flow-row");
    return {
      render(index, step) {
        const takeaway = step && step.kind === "takeaway";
        rows.forEach((r) => {
          const s = Number(r.dataset.stage);
          r.classList.toggle("is-on", !takeaway && s === index);
          r.classList.toggle("is-done", takeaway || (index > 0 && s < index));
        });
      },
    };
  }

  L.l2p11 = { RECIPE };

  if (typeof module !== "undefined" && module.exports) module.exports = defs;
  if (typeof document === "undefined") return;

  /* ---------------- 2017 versus today toggle ---------------- */
  const SUMMARY = {
    then: "The 2017 Transformer: an encoder-decoder built for translation, with sinusoidal positions, post-norm LayerNorm, a ReLU FFN and full multi-head attention over 512 tokens.",
    now: "A modern LLM like Llama 3: decoder-only, RoPE in every layer, pre-norm RMSNorm, SwiGLU, GQA, and 128,000+ tokens of context, trained on about 15 trillion tokens.",
    both: "Side by side, almost every row has changed. Look at the skeleton, though: attention, residuals, normalization, feedforward, stacked N times. That part held up.",
  };

  function initThenNow(fig) {
    const group = fig.querySelector(".chips");
    const readout = fig.querySelector(".ix-readout");
    const labels = fig.querySelectorAll("[data-then]");
    function show(mode) {
      fig.dataset.mode = mode;
      labels.forEach((el) => {
        el.textContent = mode === "then" ? el.dataset.then : mode === "now" ? el.dataset.now : el.dataset.then + " → " + el.dataset.now;
      });
      readout.textContent = SUMMARY[mode];
    }
    group.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-mode]");
      if (!b) return;
      group.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      show(b.dataset.mode);
    });
    fig.querySelector(".ix-controls").hidden = false;
    show("then");
  }

  function init() {
    const fig = document.getElementById("l2p11-then-now");
    if (fig) initThenNow(fig);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})(typeof window !== "undefined" ? window : globalThis);
