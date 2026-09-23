// ============================================================
// Lecture 2, Part 7: MHA, MQA and GQA.
// One walkthrough (the KV cache for a Llama 2 7B sized model)
// and two interactives: token-by-token generation with the
// cache growing, and the G slider that morphs MHA into GQA and
// MQA. Every number is computed, never typed.
// ============================================================
(function (root) {
  "use strict";

  const L = (root.Lecture = root.Lecture || {});
  const M = L.math;
  const defs = (L.defs = L.defs || {});
  const MB = 1024 * 1024;
  const GB = 1024 * MB;
  const int = (n) => M.fmtInt(n);
  const plain = (x) => String(M.round(x, 3));

  // Sizes the way the notes write them: "0.5 MB", "64 MB", "2 GB".
  function fmtBytes(b) {
    const show = (x) => M.round(x, 1).toLocaleString("en-US", { maximumFractionDigits: 1 });
    if (b >= GB) return show(b / GB) + " GB";
    if (b >= MB / 2) return show(b / MB) + " MB";
    return show(b / 1024) + " KB";
  }

  /* ---------------- the model used all through Part 7 ---------------- */
  const LLAMA = { layers: 32, heads: 32, headDim: 128, bytes: 2, tokens: 4096 };
  const H = LLAMA.heads;
  const G_OPTIONS = [1, 2, 4, 8, 16, 32];

  function attnName(G, h) {
    if (G === 1) return { short: "MQA", long: "Multi-Query Attention" };
    if (G === h) return { short: "MHA", long: "Multi-Head Attention" };
    return { short: "GQA", long: "Grouped-Query Attention" };
  }
  const groupOf = (q, G, h) => Math.floor(q / (h / G));
  const cacheForG = (G) =>
    M.kvCacheBytes({ layers: LLAMA.layers, kvHeads: G, headDim: LLAMA.headDim, tokens: LLAMA.tokens, bytes: LLAMA.bytes });

  /* ---------------- generation: what the cache saves ---------------- */
  const GEN_TOKENS = ["The", "cat", "sat", "on", "the", "mat", "."];
  const GEN_PROMPT = 2;
  // After generating up to n tokens in total: key/value pairs computed in all
  // the generation steps, with the cache (one per step) and without (all of them, every step).
  function genTotals(n, prompt) {
    let withCache = 0;
    let without = 0;
    for (let t = prompt + 1; t <= n; t++) {
      withCache += 1;
      without += t;
    }
    return { withCache, without };
  }

  /* ---------------- 7.2 KV cache worked example ---------------- */
  defs["l2-kv-cache"] = {
    id: "l2-kv-cache",
    title: "The KV cache of a 7B-sized model, by hand",
    input: { layers: 32, kvHeads: 32, headDim: 128, bytes: 2, tokens: 4096, users: 20 },
    setup:
      "Take a model like Llama 2 7B. It has 32 layers and 32 heads, each head works in 128 dimensions, " +
      "and every number is stored in 16 bits, which is 2 bytes. With plain multi-head attention every head keeps its own K and V, " +
      "so there are 32 KV heads. How much memory does the cache need?",
    setupBlocks: (input) => [
      { type: "note", html: "KV cache size = 2 × layers × kv_heads × head_dim × tokens × bytes_per_number" },
      {
        type: "lines",
        lines: [
          `layers = ${input.layers}`,
          `kv_heads = ${input.kvHeads}`,
          `head_dim = ${input.headDim}`,
          `bytes_per_number = ${input.bytes}  (16-bit numbers)`,
        ],
      },
    ],
    build(input) {
      const { layers, kvHeads, headDim, bytes, tokens, users } = input;
      const perToken = M.kvCacheBytes({ layers, kvHeads, headDim, bytes, tokens: 1 });
      const factors = [2, layers, kvHeads, headDim, bytes];
      const running = factors.reduce((acc, f) => acc.concat([(acc.length ? acc[acc.length - 1] : 1) * f]), []);
      const perTokenMB = perToken / MB;
      const totalBytes = perToken * tokens;
      const totalMB = perTokenMB * tokens;
      const totalGB = totalBytes / GB;
      const usersGB = totalGB * users;
      const gqaBytes = M.kvCacheBytes({ layers, kvHeads: 8, headDim, bytes, tokens });

      const productLines = [
        `2 × ${layers} = ${int(running[1])}`,
        `${int(running[1])} × ${kvHeads} = ${int(running[2])}`,
        `${int(running[2])} × ${headDim} = ${int(running[3])}`,
        { t: `${int(running[3])} × ${bytes} bytes = ${int(running[4])} bytes`, hl: true },
      ];
      const perTokenRow = { type: "lines", key: "perToken", value: perToken, lines: productLines };
      const mbRow = {
        type: "lines",
        key: "perTokenMB",
        value: perTokenMB,
        lines: [`1 MB = 1,024 × 1,024 = ${int(MB)} bytes`, { t: `${int(perToken)} / ${int(MB)} = ${plain(perTokenMB)} MB per token`, hl: true }],
      };
      const convRow = {
        type: "lines",
        key: "conversation",
        value: totalGB,
        bytes: totalBytes,
        lines: [
          `${plain(perTokenMB)} MB × ${int(tokens)} tokens = ${int(totalMB)} MB`,
          { t: `${int(totalMB)} MB / 1,024 = ${plain(totalGB)} GB`, hl: true },
        ],
      };
      const choiceBytes = [totalBytes / 1024, totalBytes / 8, totalBytes, totalBytes * 10];

      return {
        steps: [
          {
            title: "Multiply it out for one token",
            say:
              "Set tokens to 1 and the formula is one long product. The 2 is there because every token stores both a key and a value. " +
              `Multiply left to right and a single token needs ${int(perToken)} bytes.`,
            blocks: [Object.assign({}, perTokenRow, { fresh: true })],
          },
          {
            title: "Turn bytes into megabytes",
            say:
              `A megabyte is 1,024 × 1,024 = ${int(MB)} bytes, so ${int(perToken)} bytes is exactly ${plain(perTokenMB)} MB. ` +
              "Half a megabyte sounds small. But it is for one token, and a conversation runs to thousands of them.",
            blocks: [perTokenRow, Object.assign({}, mbRow, { fresh: true })],
          },
          {
            title: `Scale up to a ${int(tokens)}-token conversation`,
            say:
              `Every token adds its ${plain(perTokenMB)} MB, and none of it can be thrown away while the conversation goes on. ` +
              `${plain(perTokenMB)} × ${int(tokens)} = ${int(totalMB)} MB, which is ${plain(totalGB)} GB.`,
            predict: {
              ask: `The cache grows by ${plain(perTokenMB)} MB with every token. How big is it after ${int(tokens)} tokens?`,
              choices: choiceBytes.map((b) => "About " + fmtBytes(b)),
              answer: 2,
              why:
                `${plain(perTokenMB)} × ${int(tokens)} = ${int(totalMB)} MB, and ${int(totalMB)} MB / 1,024 = ${plain(totalGB)} GB. ` +
                "Half a megabyte per token adds up fast.",
              hint: `Multiply ${plain(perTokenMB)} MB by ${int(tokens)} first, then remember that 1,024 MB make 1 GB.`,
            },
            blocks: [mbRow, Object.assign({}, convRow, { fresh: true })],
          },
          {
            title: `Now serve ${users} users at once`,
            say:
              `That ${plain(totalGB)} GB is for one conversation. A server answering ${users} people at the same time keeps ${users} caches, ` +
              `so the cache alone needs ${plain(usersGB)} GB. At long context lengths the KV cache can grow larger than the model's own weights.`,
            blocks: [
              convRow,
              {
                type: "lines",
                key: "users",
                value: usersGB,
                fresh: true,
                lines: [{ t: `${plain(totalGB)} GB × ${users} users = ${plain(usersGB)} GB`, hl: true }],
              },
            ],
          },
          {
            title: "Why it slows generation down too",
            say:
              "Memory is not the only cost. To write each new token, attention has to read every cached key and value from memory. " +
              `Near the end of this conversation that is ${plain(totalGB)} GB of reading for every single token written. This is the inference bottleneck.`,
            blocks: [
              {
                type: "note",
                fresh: true,
                html:
                  `Per token: ${int(perToken)} bytes (${plain(perTokenMB)} MB). One ${int(tokens)}-token conversation: ${plain(totalGB)} GB. ` +
                  `${users} users: ${plain(usersGB)} GB. All of it read again for each new token.`,
              },
            ],
          },
        ],
        takeaway:
          "The KV cache grows in a straight line with the number of tokens and the number of users, and it multiplies straight through by the number of KV heads. " +
          `That last factor is the one MQA and GQA go after. Keep 32 query heads but only 8 KV heads, and the ${plain(totalGB)} GB becomes ${fmtBytes(gqaBytes)}.`,
      };
    },
  };

  L.l2p7 = { fmtBytes, attnName, groupOf, cacheForG, genTotals, G_OPTIONS, GEN_TOKENS, GEN_PROMPT, LLAMA };

  if (typeof module !== "undefined" && module.exports) module.exports = defs;
  if (typeof document === "undefined") return;

  /* =========================================================
     Browser side: the two interactives
     ========================================================= */
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const perTokenBytes = M.kvCacheBytes({ layers: LLAMA.layers, kvHeads: LLAMA.heads, headDim: LLAMA.headDim, bytes: LLAMA.bytes, tokens: 1 });

  function initGen(fig) {
    const gridEl = fig.querySelector(".l2p7-gen-grid");
    const out = fig.querySelector(".ix-readout");
    const nextBtn = fig.querySelector('[data-act="next"]');
    const resetBtn = fig.querySelector('[data-act="reset"]');
    const chips = Array.from(fig.querySelectorAll("[data-mode]"));
    const words = GEN_TOKENS;
    const state = { n: GEN_PROMPT, cache: true };
    gridEl.style.setProperty("--slots", words.length);

    function cell(cls, html) {
      return `<span class="l2p7-cell ${cls}">${html}</span>`;
    }

    function render() {
      const n = state.n;
      const stepped = n > GEN_PROMPT;
      const newest = n - 1;
      let html = '<span class="l2p7-rowlabel">token</span>';
      words.forEach((w, i) => {
        const cls = i >= n ? "is-future" : i === newest && stepped ? "is-word is-newest" : "is-word";
        html += cell(cls, i >= n ? "?" : esc(w));
      });
      ["K", "V"].forEach((kind) => {
        html += `<span class="l2p7-rowlabel">${kind}</span>`;
        words.forEach((_, i) => {
          if (i >= n) return (html += cell("is-empty", ""));
          const label = `${kind}<sub>${i}</sub>`;
          if (stepped && i === newest) html += cell("is-new", label);
          else if (stepped && !state.cache) html += cell("is-redo", label + '<span class="l2p7-glyph" aria-hidden="true">↻</span>');
          else html += cell("is-cached", label);
        });
      });
      html += '<span class="l2p7-rowlabel">Q</span>';
      words.forEach((_, i) => {
        html += stepped && i === newest ? cell("is-query", `Q<sub>${i}</sub>`) : cell("is-blank", "");
      });
      gridEl.innerHTML = html;

      const t = genTotals(n, GEN_PROMPT);
      const mem = `The cache now holds ${n} tokens. For a Llama 2 7B sized model that is ${n} × ${int(perTokenBytes)} bytes = <b>${fmtBytes(n * perTokenBytes)}</b>.`;
      let text;
      if (!stepped) {
        text =
          `The prompt "${esc(words.slice(0, GEN_PROMPT).join(" "))}" has been read, and its keys and values are already stored. ` +
          "Press <b>Write the next token</b> to generate one word at a time.";
      } else {
        const w = esc(words[newest]);
        text = `To write "${w}", its query compares against <b>${n} keys</b>: the ${n - 1} before it and its own. `;
        text += state.cache
          ? `With the cache, only <b>1</b> new key and value pair is computed. The other ${n - 1} are read straight from memory. `
          : `Without a cache, all <b>${n}</b> keys and values are worked out again, even though ${n - 1} of them are exactly the same as last step. `;
        text += `Pairs computed so far: <b>${t.withCache}</b> with the cache, <b>${t.without}</b> without.`;
        if (n === words.length) text += " The sentence is finished, and the gap keeps widening the longer the text gets.";
      }
      out.innerHTML = text + (state.cache ? " " + mem : "");
      nextBtn.disabled = n >= words.length;
      chips.forEach((c) => c.setAttribute("aria-pressed", String((c.dataset.mode === "cache") === state.cache)));
    }

    nextBtn.addEventListener("click", () => {
      if (state.n < words.length) state.n += 1;
      render();
    });
    resetBtn.addEventListener("click", () => {
      state.n = GEN_PROMPT;
      render();
      nextBtn.focus();
    });
    chips.forEach((c) =>
      c.addEventListener("click", () => {
        state.cache = c.dataset.mode === "cache";
        render();
      })
    );
    render();
  }

  function headsSVG(G) {
    const h = H;
    const W = 640;
    const x0 = 10;
    const pitch = (W - 2 * x0) / h;
    const per = h / G;
    const qw = pitch - 5;
    const kw = Math.min(per * pitch - 6, 64);
    const name = attnName(G, h);
    const label =
      `${h} query heads sharing ${G} key and value head${G === 1 ? "" : "s"}, ` +
      `${per} query head${per === 1 ? "" : "s"} per group. This is ${name.long}.`;
    let s = `<svg viewBox="0 0 ${W} 178" role="img" aria-label="${label}">`;
    if (G < h) {
      for (let g = 0; g < G; g++) {
        s += `<rect class="l2p7-band${g % 2 ? " is-alt" : ""}" x="${x0 + g * per * pitch + 1}" y="4" width="${per * pitch - 2}" height="170" rx="8"/>`;
      }
    }
    for (let q = 0; q < h; q++) {
      const cx = x0 + q * pitch + pitch / 2;
      const g = groupOf(q, G, h);
      const kx = x0 + (g * per + per / 2) * pitch;
      s += `<line class="l2p7-link" x1="${cx}" y1="46" x2="${kx}" y2="100"/>`;
    }
    for (let q = 0; q < h; q++) {
      s += `<rect class="l2p7-q" x="${x0 + q * pitch + 2.5}" y="18" width="${qw}" height="28" rx="3"/>`;
    }
    for (let g = 0; g < G; g++) {
      const kx = x0 + (g * per + per / 2) * pitch;
      s += `<rect class="l2p7-k" x="${kx - kw / 2}" y="100" width="${kw}" height="28" rx="3"/>`;
      s += `<rect class="l2p7-v" x="${kx - kw / 2}" y="134" width="${kw}" height="28" rx="3"/>`;
      if (kw >= 30) {
        s += `<text class="l2p7-svgtext" x="${kx}" y="119" text-anchor="middle">K</text>`;
        s += `<text class="l2p7-svgtext" x="${kx}" y="153" text-anchor="middle">V</text>`;
      }
    }
    return s + "</svg>";
  }

  function initGroups(fig) {
    const slider = fig.querySelector("input[type=range]");
    const output = fig.querySelector("output");
    const stage = fig.querySelector(".ix-stage");
    const out = fig.querySelector(".ix-readout");
    const meter = fig.querySelector(".l2p7-meter-fill");
    const meterText = fig.querySelector(".l2p7-meter-text");
    const full = cacheForG(H);

    function render() {
      const G = G_OPTIONS[Number(slider.value)];
      const name = attnName(G, H);
      const per = H / G;
      output.textContent = "G = " + G;
      slider.setAttribute("aria-valuetext", `G = ${G}, ${name.long}`);
      stage.innerHTML = headsSVG(G);
      let what;
      if (G === 1) what = `one group: all ${H} query heads share one K and one V.`;
      else if (G === H) what = `every query head is its own group, with its own K and V.`;
      else what = `${G} groups, each with its own K and V, shared by ${H} / ${G} = ${per} query heads.`;
      out.innerHTML = `<b>G = ${G}: ${name.long} (<abbr data-term="${name.short}">${name.short}</abbr>).</b> ${what}`;
      const bytes = cacheForG(G);
      meter.style.setProperty("--w", (bytes / full) * 100 + "%");
      meterText.innerHTML =
        `KV cache for the ${int(LLAMA.tokens)}-token example: <b>${fmtBytes(bytes)}</b>` +
        (G === H ? ", the full MHA size." : `, ${H / G} times smaller than MHA's ${fmtBytes(full)}.`);
    }
    slider.addEventListener("input", render);
    render();
  }

  function init() {
    const gen = document.getElementById("l2p7-generate");
    if (gen) initGen(gen);
    const groups = document.getElementById("l2p7-groups");
    if (groups) initGroups(groups);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})(typeof window !== "undefined" ? window : globalThis);
