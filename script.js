// ============================================================
// The Transformer: interactive lecture notebook
// Vanilla JS, no dependencies.
// ============================================================
(function () {
  "use strict";

  const svgNS = "http://www.w3.org/2000/svg";
  const REDUCE = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  /* ---------------------------------------------------------
     Theme toggle
  --------------------------------------------------------- */
  const themeBtn = document.getElementById("theme-toggle");
  const root = document.documentElement;

  function applyStoredTheme() {
    const saved = localStorage.getItem("transformer-theme");
    if (saved === "light" || saved === "dark") {
      root.setAttribute("data-theme", saved);
      themeBtn.setAttribute("aria-pressed", saved === "light" ? "true" : "false");
    }
  }
  applyStoredTheme();

  themeBtn.addEventListener("click", () => {
    const isLight = root.getAttribute("data-theme") === "light";
    const next = isLight ? "dark" : "light";
    root.setAttribute("data-theme", next);
    localStorage.setItem("transformer-theme", next);
    themeBtn.setAttribute("aria-pressed", next === "light" ? "true" : "false");
    if (typeof redrawPE === "function") redrawPE();
  });

  /* ---------------------------------------------------------
     Scroll progress + active nav pill
  --------------------------------------------------------- */
  const progressFill = document.getElementById("progress-fill");
  function updateProgress() {
    const h = document.documentElement;
    const scrolled = h.scrollTop || document.body.scrollTop;
    const max = h.scrollHeight - h.clientHeight;
    progressFill.style.width = (max > 0 ? (scrolled / max) * 100 : 0) + "%";
  }
  document.addEventListener("scroll", updateProgress, { passive: true });
  updateProgress();

  const pills = Array.from(document.querySelectorAll(".pill-nav a"));
  const sections = Array.from(document.querySelectorAll("main > section[id]"));
  if ("IntersectionObserver" in window && sections.length) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const id = entry.target.id;
            pills.forEach((p) => p.classList.toggle("active", p.dataset.sec === id));
          }
        });
      },
      { rootMargin: "-40% 0px -55% 0px", threshold: 0 }
    );
    sections.forEach((s) => io.observe(s));

    function clearPillsAboveFirstSection() {
      if (window.scrollY < sections[0].offsetTop - 120) {
        pills.forEach((p) => p.classList.remove("active"));
      }
    }
    document.addEventListener("scroll", clearPillsAboveFirstSection, { passive: true });
    clearPillsAboveFirstSection();
  }

  /* ---------------------------------------------------------
     Shared: build a clickable/animatable row of word chips in an SVG
  --------------------------------------------------------- */
  function buildWordRow(svg, words, opts) {
    const o = Object.assign(
      { clickable: false, rectClass: "hero-word-rect", textClass: "hero-word-text", gap: 10, padX: 14, height: 34, charW: 7.6, topMargin: 46 },
      opts
    );
    svg.innerHTML = "";
    const items = [];
    let x = 8;
    words.forEach((w) => {
      const width = Math.max(36, w.length * o.charW + o.padX * 2);
      items.push({ word: w, x, width });
      x += width + o.gap;
    });
    const totalWidth = x - o.gap + 8;
    const rowY = o.topMargin;
    svg.setAttribute("viewBox", `0 0 ${totalWidth} ${rowY + o.height + 14}`);

    const rects = [];
    const centers = [];
    items.forEach((it, i) => {
      const rect = document.createElementNS(svgNS, "rect");
      rect.setAttribute("x", it.x);
      rect.setAttribute("y", rowY);
      rect.setAttribute("width", it.width);
      rect.setAttribute("height", o.height);
      rect.setAttribute("rx", 8);
      rect.setAttribute("class", o.rectClass);
      rect.dataset.idx = String(i);
      svg.appendChild(rect);

      const text = document.createElementNS(svgNS, "text");
      text.setAttribute("x", it.x + it.width / 2);
      text.setAttribute("y", rowY + o.height / 2 + 4.5);
      text.setAttribute("text-anchor", "middle");
      text.setAttribute("class", o.textClass);
      text.textContent = it.word;
      svg.appendChild(text);

      rects.push(rect);
      centers.push({ x: it.x + it.width / 2, top: rowY, mid: rowY + o.height / 2 });

      if (o.clickable) {
        rect.style.cursor = "pointer";
        rect.addEventListener("click", () => {
          svg.dispatchEvent(new CustomEvent("wordclick", { detail: { idx: i } }));
        });
      }
    });
    return { rects, centers, totalWidth };
  }

  function drawAttentionLines(svg, centers, qi, weights, lineClass) {
    Array.from(svg.querySelectorAll(".gen-line")).forEach((el) => el.remove());
    const q = centers[qi];
    weights.forEach((w, j) => {
      if (j === qi || w < 0.025) return;
      const t = centers[j];
      const midx = (q.x + t.x) / 2;
      const dist = Math.abs(q.x - t.x);
      const curveH = Math.min(q.top, t.top) - 16 - dist * 0.05;
      const path = document.createElementNS(svgNS, "path");
      path.setAttribute("d", `M${q.x},${q.top} Q${midx},${curveH} ${t.x},${t.top}`);
      path.setAttribute("class", `${lineClass} gen-line`);
      path.style.strokeWidth = (1 + w * 8.5).toFixed(2);
      path.style.strokeOpacity = Math.min(1, 0.14 + w * 1.05).toFixed(2);
      svg.appendChild(path);
    });
  }

  function computeWeights(n, qi, overrides) {
    const w = new Array(n).fill(0).map((_, i) => {
      const d = Math.abs(i - qi);
      return Math.exp(-(d * d) / (2 * 1.5 * 1.5));
    });
    if (overrides) {
      Object.keys(overrides).forEach((k) => {
        w[+k] += overrides[k];
      });
    }
    const sum = w.reduce((a, b) => a + b, 0);
    return w.map((x) => x / sum);
  }

  /* ---------------------------------------------------------
     Example sentence data (shared by hero + anatomy demo)
  --------------------------------------------------------- */
  const SENTENCES = [
    {
      label: "the tired animal",
      words: "the animal didn't cross the street because it was too tired".split(" "),
      overrides: { 7: { 1: 2.6 }, 10: { 1: 1.3, 7: 0.9 } },
      flagship: 7,
    },
    {
      label: "the poured water",
      words: "she poured water from the pitcher until it was full".split(" "),
      overrides: { 7: { 5: 2.4, 2: 0.7 } },
      flagship: 7,
    },
    {
      label: "the trophy & suitcase",
      words: "the trophy doesn't fit in the suitcase because it is too big".split(" "),
      overrides: { 8: { 1: 2.5 } },
      flagship: 8,
    },
  ];

  /* ---------------------------------------------------------
     HERO visualization
  --------------------------------------------------------- */
  const heroSvg = document.getElementById("hero-sentence");
  const heroTag = document.getElementById("hero-sentence-tag");
  let heroRow = null;
  let heroSentenceIdx = -1;

  function heroRender(sentenceIdx, qi) {
    const s = SENTENCES[sentenceIdx];
    if (sentenceIdx !== heroSentenceIdx) {
      heroRow = buildWordRow(heroSvg, s.words, {
        clickable: false,
        rectClass: "hero-word-rect",
        textClass: "hero-word-text",
        charW: 8.2,
        height: 36,
        topMargin: 56,
      });
      heroSentenceIdx = sentenceIdx;
    }
    heroRow.rects.forEach((r, i) => r.classList.toggle("is-focus", i === qi));
    const weights = computeWeights(s.words.length, qi, s.overrides[qi]);
    drawAttentionLines(heroSvg, heroRow.centers, qi, weights, "hero-line");
    heroTag.textContent = `example ${sentenceIdx + 1} / ${SENTENCES.length}`;
  }

  const heroCycle = [
    { s: 0, q: 7 },
    { s: 0, q: 10 },
    { s: 1, q: 7 },
    { s: 2, q: 8 },
  ];
  let heroCycleIdx = 0;
  let heroInterval = null;
  heroRender(heroCycle[0].s, heroCycle[0].q);

  function startHeroCycle() {
    if (heroInterval || REDUCE) return;
    heroCycleIdx = 0;
    heroRender(heroCycle[0].s, heroCycle[0].q);
    heroInterval = setInterval(() => {
      heroCycleIdx = (heroCycleIdx + 1) % heroCycle.length;
      const step = heroCycle[heroCycleIdx];
      heroRender(step.s, step.q);
    }, 3200);
  }
  function stopHeroCycle() {
    clearInterval(heroInterval);
    heroInterval = null;
  }

  /* ---------------------------------------------------------
     WHY ATTENTION: RNN vs parallel comparison
  --------------------------------------------------------- */
  const N_CMP = 5;
  const CMP_X = [34, 92, 150, 208, 266];
  const CMP_Y = 75;
  const CMP_R = 15;

  function buildCompare(svg, nodeClass, edgeClass, mode) {
    svg.innerHTML = "";
    if (mode === "chain") {
      for (let i = 0; i < N_CMP - 1; i++) {
        const line = document.createElementNS(svgNS, "line");
        line.setAttribute("x1", CMP_X[i] + CMP_R);
        line.setAttribute("y1", CMP_Y);
        line.setAttribute("x2", CMP_X[i + 1] - CMP_R);
        line.setAttribute("y2", CMP_Y);
        line.setAttribute("class", edgeClass);
        line.dataset.edge = String(i);
        svg.appendChild(line);
      }
    } else {
      for (let i = 0; i < N_CMP; i++) {
        for (let j = i + 1; j < N_CMP; j++) {
          const dist = CMP_X[j] - CMP_X[i];
          const curveY = CMP_Y - 20 - dist * 0.12 * ((i + j) % 2 === 0 ? 1 : 0.55);
          const path = document.createElementNS(svgNS, "path");
          const midx = (CMP_X[i] + CMP_X[j]) / 2;
          path.setAttribute("d", `M${CMP_X[i]},${CMP_Y - CMP_R} Q${midx},${curveY} ${CMP_X[j]},${CMP_Y - CMP_R}`);
          path.setAttribute("class", edgeClass);
          path.dataset.edge = `${i}-${j}`;
          svg.appendChild(path);
        }
      }
    }
    const nodes = [];
    for (let i = 0; i < N_CMP; i++) {
      const c = document.createElementNS(svgNS, "circle");
      c.setAttribute("cx", CMP_X[i]);
      c.setAttribute("cy", CMP_Y);
      c.setAttribute("r", CMP_R);
      c.setAttribute("class", nodeClass);
      svg.appendChild(c);
      const t = document.createElementNS(svgNS, "text");
      t.setAttribute("x", CMP_X[i]);
      t.setAttribute("y", CMP_Y + 34);
      t.setAttribute("text-anchor", "middle");
      t.setAttribute("class", "diagram-node-label");
      t.textContent = "w" + (i + 1);
      svg.appendChild(t);
      nodes.push(c);
    }
    return nodes;
  }

  const rnnSvg = document.getElementById("rnn-svg");
  const attnCmpSvg = document.getElementById("attn-cmp-svg");
  const rnnNodes = buildCompare(rnnSvg, "rnn-node", "rnn-edge", "chain");
  const attnNodes = buildCompare(attnCmpSvg, "attn-node", "attn-edge", "mesh");

  let rnnRunning = false;
  async function playRnnDemo() {
    if (rnnRunning) return;
    rnnRunning = true;
    const btn = document.getElementById("rnn-replay");
    btn.disabled = true;
    rnnNodes.forEach((n) => n.classList.remove("on"));
    const start = performance.now();
    for (let i = 0; i < rnnNodes.length; i++) {
      await new Promise((r) => setTimeout(r, REDUCE ? 0 : 450));
      rnnNodes[i].classList.add("on");
    }
    const elapsed = ((performance.now() - start) / 1000).toFixed(2);
    document.getElementById("rnn-timer").textContent = `${elapsed}s · sequential`;
    btn.disabled = false;
    rnnRunning = false;
  }
  document.getElementById("rnn-replay").addEventListener("click", playRnnDemo);

  let attnRunning = false;
  function playAttnDemo() {
    if (attnRunning) return;
    attnRunning = true;
    const btn = document.getElementById("attn-replay");
    btn.disabled = true;
    attnNodes.forEach((n) => n.classList.remove("on"));
    Array.from(attnCmpSvg.querySelectorAll(".attn-edge")).forEach((e) => e.classList.remove("on"));
    const start = performance.now();
    attnNodes.forEach((n) => n.classList.add("on"));
    const edges = Array.from(attnCmpSvg.querySelectorAll(".attn-edge"));
    edges.forEach((e, i) => {
      setTimeout(() => e.classList.add("on"), REDUCE ? 0 : i * 10);
    });
    setTimeout(() => {
      const elapsed = ((performance.now() - start) / 1000).toFixed(2);
      document.getElementById("attn-timer").textContent = `${elapsed}s · parallel`;
      btn.disabled = false;
      attnRunning = false;
    }, REDUCE ? 0 : edges.length * 10 + 40);
  }
  document.getElementById("attn-replay").addEventListener("click", playAttnDemo);

  /* ---------------------------------------------------------
     ANATOMY: interactive QKV demo
  --------------------------------------------------------- */
  const qkvSvg = document.getElementById("qkv-sentence");
  const qkvBars = document.getElementById("qkv-bars");
  const qkvHint = document.getElementById("qkv-hint");
  const qkvTabs = Array.from(document.querySelectorAll("#qkv-tabs .tab"));
  let qkvSentenceIdx = 0;
  let qkvRow = null;

  function renderQkvSentence(sentenceIdx) {
    const s = SENTENCES[sentenceIdx];
    qkvRow = buildWordRow(qkvSvg, s.words, {
      clickable: true,
      rectClass: "qkv-word-rect",
      textClass: "qkv-word-text",
      charW: 8,
      height: 34,
      topMargin: 44,
    });
    selectQuery(s.flagship);
  }
  qkvSvg.addEventListener("wordclick", (e) => selectQuery(e.detail.idx));

  function selectQuery(qi) {
    const s = SENTENCES[qkvSentenceIdx];
    qkvRow.rects.forEach((r, i) => r.classList.toggle("is-query", i === qi));
    const weights = computeWeights(s.words.length, qi, s.overrides[qi]);
    drawAttentionLines(qkvSvg, qkvRow.centers, qi, weights, "qkv-line");

    const order = weights.map((w, i) => ({ w, i })).sort((a, b) => b.w - a.w);
    qkvBars.innerHTML = "";
    order.forEach(({ w, i }) => {
      const row = document.createElement("div");
      row.className = "weight-row";
      const label = document.createElement("span");
      label.textContent = s.words[i] + (i === qi ? " (self)" : "");
      const track = document.createElement("span");
      track.className = "weight-track";
      const fill = document.createElement("span");
      fill.className = "weight-fill";
      track.appendChild(fill);
      const val = document.createElement("span");
      val.className = "weight-val";
      val.textContent = Math.round(w * 100) + "%";
      row.appendChild(label);
      row.appendChild(track);
      row.appendChild(val);
      qkvBars.appendChild(row);
      requestAnimationFrame(() => {
        fill.style.width = Math.round(w * 100) + "%";
      });
    });

    const top = order.find((o) => o.i !== qi) || order[0];
    qkvHint.innerHTML = `Query: <b class="mono">&ldquo;${s.words[qi]}&rdquo;</b>. The strongest match is <b class="mono">&ldquo;${s.words[top.i]}&rdquo;</b> at ${Math.round(top.w * 100)}%. This is how the model decides what a word like &ldquo;${s.words[qi]}&rdquo; is really pointing to.`;
  }

  qkvTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      qkvTabs.forEach((t) => {
        t.classList.remove("active");
        t.setAttribute("aria-selected", "false");
      });
      tab.classList.add("active");
      tab.setAttribute("aria-selected", "true");
      qkvSentenceIdx = +tab.dataset.idx;
      renderQkvSentence(qkvSentenceIdx);
    });
  });
  renderQkvSentence(0);

  /* ---------------------------------------------------------
     ARCHITECTURE: run the flow
  --------------------------------------------------------- */
  const archRunBtn = document.getElementById("arch-run");
  const archResetBtn = document.getElementById("arch-reset");
  const archStatus = document.getElementById("arch-status");
  const archPulse = document.getElementById("arch-pulse");
  const ALL_ARCH_NODES = [
    "node-enc-embed", "node-enc-attn", "node-enc-addnorm1", "node-enc-ffn", "node-enc-addnorm2",
    "node-dec-embed", "node-dec-mask", "node-dec-addnorm1", "node-dec-cross", "node-dec-addnorm2",
    "node-dec-ffn", "node-dec-addnorm3", "node-dec-linear", "node-dec-softmax",
  ];

  const ARCH_PATH = [
    { id: "node-enc-embed", x: 260, y: 77.5, status: "1 · The encoder turns each token into a vector and adds positional encoding." },
    { id: "node-enc-pe", x: 260, y: 127 },
    { id: "node-enc-attn", x: 260, y: 182.5 },
    { id: "node-enc-addnorm1", x: 260, y: 234.5 },
    { id: "node-enc-ffn", x: 260, y: 286.5 },
    { id: "node-enc-addnorm2", x: 260, y: 338.5 },
    { id: "node-bridge-1", x: 260, y: 371 },
    { id: "node-bridge-2", x: 470, y: 371, status: "2 · That output becomes K and V for every decoder layer's cross-attention." },
    { id: "node-bridge-3", x: 470, y: 287 },
    { id: "node-dec-cross", x: 680, y: 286.5 },
    { id: "node-dec-embed", x: 680, y: 77.5, status: "3 · Meanwhile the decoder embeds the tokens it has generated so far." },
    { id: "node-dec-pe", x: 680, y: 127 },
    { id: "node-dec-mask", x: 680, y: 182.5, status: "4 · Masked self-attention: each position sees only itself and earlier positions." },
    { id: "node-dec-addnorm1", x: 680, y: 234.5 },
    { id: "node-dec-cross", x: 680, y: 286.5, status: "5 · Cross-attention: the decoder's query meets the encoder's K, V." },
    { id: "node-dec-addnorm2", x: 680, y: 338.5 },
    { id: "node-dec-ffn", x: 680, y: 390.5 },
    { id: "node-dec-addnorm3", x: 680, y: 442.5 },
    { id: "node-dec-linear", x: 680, y: 495, status: "6 · Linear projects to vocabulary size, then softmax turns scores into probabilities." },
    { id: "node-dec-softmax", x: 680, y: 547 },
  ];

  function animatePulse(el, points, hopDuration, onArrive) {
    return new Promise((resolve) => {
      let i = 0;
      el.classList.add("show");
      el.setAttribute("cx", points[0].x);
      el.setAttribute("cy", points[0].y);
      if (onArrive) onArrive(points[0], 0);
      function hop() {
        if (i >= points.length - 1) {
          resolve();
          return;
        }
        const a = points[i], b = points[i + 1];
        const start = performance.now();
        const dur = REDUCE ? 1 : hopDuration;
        function frame(now) {
          const p = Math.min(1, (now - start) / dur);
          const e = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
          el.setAttribute("cx", a.x + (b.x - a.x) * e);
          el.setAttribute("cy", a.y + (b.y - a.y) * e);
          if (p < 1) requestAnimationFrame(frame);
          else {
            i++;
            if (onArrive) onArrive(points[i], i);
            hop();
          }
        }
        requestAnimationFrame(frame);
      }
      hop();
    });
  }

  let archRunning = false;
  async function runArchitectureFlow() {
    if (archRunning) return;
    archRunning = true;
    archRunBtn.disabled = true;
    ALL_ARCH_NODES.forEach((id) => document.getElementById(id).classList.remove("is-active"));
    archStatus.textContent = "Running…";
    await animatePulse(archPulse, ARCH_PATH, 420, (pt) => {
      const real = document.getElementById(pt.id);
      if (real) real.classList.add("is-active");
      if (pt.status) archStatus.innerHTML = "<b>" + pt.status + "</b>";
    });
    archStatus.innerHTML = "<b>Done.</b> The whole lit path is one token's round trip through the network.";
    archPulse.classList.remove("show");
    archRunBtn.disabled = false;
    archRunning = false;
  }
  archRunBtn.addEventListener("click", runArchitectureFlow);

  function resetArchitectureFlow() {
    archRunning = false;
    archRunBtn.disabled = false;
    ALL_ARCH_NODES.forEach((id) => document.getElementById(id).classList.remove("is-active"));
    archPulse.classList.remove("show");
    archStatus.textContent = "Press run to watch data move through the network.";
  }
  archResetBtn.addEventListener("click", resetArchitectureFlow);

  /* ---------------------------------------------------------
     INSIDE ATTENTION: step player
  --------------------------------------------------------- */
  const SDPA_STEPS = [
    { left: ["sdpa-mm1"], right: ["mh-lin-q", "mh-lin-k", "mh-lin-v"], status: "Compare every query to every key using MatMul(Q, Kᵀ), computed once per head on its own slice of the vectors." },
    { left: ["sdpa-scale"], right: [], status: "Divide by √d_k so the scores stay in a stable range as vectors get longer." },
    { left: ["sdpa-mask"], right: [], status: "Decoder self-attention only: hide every position after the current one." },
    { left: ["sdpa-softmax"], right: ["mh-stack"], status: "Turn scores into weights that sum to 1 using softmax, independently inside each of the h heads." },
    { left: ["sdpa-mm2"], right: ["mh-concat", "mh-linout"], status: "Blend the values by those weights, then concatenate all h heads and mix with one more linear layer." },
  ];
  const sdpaDotsEl = document.getElementById("sdpa-dots");
  SDPA_STEPS.forEach(() => {
    const d = document.createElement("div");
    d.className = "d";
    sdpaDotsEl.appendChild(d);
  });
  const sdpaDots = Array.from(sdpaDotsEl.children);
  const sdpaStatus = document.getElementById("sdpa-status");
  let sdpaStep = -1;
  let sdpaTimer = null;

  function sdpaAllIds() {
    const ids = new Set();
    SDPA_STEPS.forEach((s) => s.left.concat(s.right).forEach((id) => ids.add(id)));
    return Array.from(ids);
  }
  function sdpaRender() {
    sdpaAllIds().forEach((id) => document.getElementById(id).classList.remove("is-active"));
    sdpaDots.forEach((d, i) => d.classList.toggle("on", i <= sdpaStep));
    if (sdpaStep < 0) {
      sdpaStatus.textContent = "Step through scaled dot-product attention, then see it happen h=8 times in parallel on the right.";
      return;
    }
    for (let i = 0; i <= sdpaStep; i++) {
      SDPA_STEPS[i].left.concat(SDPA_STEPS[i].right).forEach((id) => document.getElementById(id).classList.add("is-active"));
    }
    sdpaStatus.innerHTML = `<b>Step ${sdpaStep + 1}/${SDPA_STEPS.length}.</b> ${SDPA_STEPS[sdpaStep].status}`;
  }
  function sdpaGoto(n) {
    sdpaStep = Math.max(-1, Math.min(SDPA_STEPS.length - 1, n));
    sdpaRender();
  }
  document.getElementById("sdpa-next").addEventListener("click", () => sdpaGoto(sdpaStep + 1));
  document.getElementById("sdpa-prev").addEventListener("click", () => sdpaGoto(sdpaStep - 1));
  function resetSdpaDemo() {
    clearInterval(sdpaTimer);
    sdpaTimer = null;
    sdpaGoto(-1);
  }
  document.getElementById("sdpa-reset").addEventListener("click", resetSdpaDemo);
  function playSdpaDemo() {
    if (sdpaTimer) return;
    if (sdpaStep >= SDPA_STEPS.length - 1) sdpaStep = -1;
    const btn = document.getElementById("sdpa-play");
    btn.disabled = true;
    sdpaTimer = setInterval(() => {
      sdpaGoto(sdpaStep + 1);
      if (sdpaStep >= SDPA_STEPS.length - 1) {
        clearInterval(sdpaTimer);
        sdpaTimer = null;
        btn.disabled = false;
      }
    }, REDUCE ? 10 : 1300);
  }
  document.getElementById("sdpa-play").addEventListener("click", playSdpaDemo);
  sdpaRender();

  /* ---------------------------------------------------------
     POSITIONAL ENCODING visualizer
  --------------------------------------------------------- */
  const peCanvas = document.getElementById("pe-canvas");
  const peCtx = peCanvas.getContext("2d");
  const peIInput = document.getElementById("pe-i");
  const peDInput = document.getElementById("pe-d");
  const peIVal = document.getElementById("pe-i-val");
  const peDVal = document.getElementById("pe-d-val");
  const peTokensRow = document.getElementById("pe-tokens");
  const peSimStatus = document.getElementById("pe-sim-status");
  const peHeatmap = document.getElementById("pe-heatmap");
  const peHeatmapA = document.getElementById("pe-heatmap-a");
  const peHeatmapB = document.getElementById("pe-heatmap-b");
  const peHeatmapLabelA = document.getElementById("pe-heatmap-label-a");
  const peHeatmapLabelB = document.getElementById("pe-heatmap-label-b");

  const PE_TOKEN_COUNT = 16;
  const PE_N_POS = 80;
  let peSelected = [];

  function peFullVector(pos, d) {
    const vec = new Array(d);
    for (let k = 0; k < d; k++) {
      const i = Math.floor(k / 2);
      const freq = 1 / Math.pow(10000, (2 * i) / d);
      vec[k] = k % 2 === 0 ? Math.sin(pos * freq) : Math.cos(pos * freq);
    }
    return vec;
  }

  function peCosineSimilarity(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let k = 0; k < a.length; k++) {
      dot += a[k] * b[k];
      na += a[k] * a[k];
      nb += b[k] * b[k];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
  }

  function renderPeTokens() {
    peTokensRow.innerHTML = "";
    for (let pos = 0; pos < PE_TOKEN_COUNT; pos++) {
      const btn = document.createElement("button");
      btn.className = "pe-token";
      btn.type = "button";
      btn.textContent = String(pos);
      btn.setAttribute("aria-label", "Token position " + pos);
      const slot = peSelected.indexOf(pos);
      if (slot === 0) btn.classList.add("is-a");
      if (slot === 1) btn.classList.add("is-b");
      btn.addEventListener("click", () => {
        if (peSelected.length >= 2) peSelected = [];
        peSelected.push(pos);
        renderPeTokens();
        updatePeComparison();
        redrawPE();
      });
      peTokensRow.appendChild(btn);
    }
  }

  function peCellColor(v) {
    const t = Math.max(-1, Math.min(1, v));
    return { color: t < 0 ? cssVar("--encoder") : cssVar("--decoder"), alpha: (Math.abs(t) * 0.85 + 0.15).toFixed(2) };
  }

  function renderPeHeatmapRow(container, vec) {
    container.innerHTML = "";
    vec.forEach((v) => {
      const cell = document.createElement("span");
      const c = peCellColor(v);
      cell.style.background = c.color;
      cell.style.opacity = c.alpha;
      container.appendChild(cell);
    });
  }

  function updatePeComparison() {
    if (peSelected.length < 2) {
      peHeatmap.hidden = true;
      peSimStatus.textContent =
        peSelected.length === 0
          ? "Click a token above, then a second one, to compare them."
          : "Now click a second token to compare it against token " + peSelected[0] + ".";
      return;
    }
    const d = +peDInput.value;
    const [a, b] = peSelected;
    const vecA = peFullVector(a, d);
    const vecB = peFullVector(b, d);
    const sim = peCosineSimilarity(vecA, vecB);
    const simLabel = sim > 0.9 ? "almost identical" : sim > 0.6 ? "fairly similar" : sim > 0.3 ? "somewhat different" : "clearly different";
    const distanceNote =
      Math.abs(a - b) <= 2
        ? "They sit close together, so the model can still treat them as neighbors."
        : "They sit far apart, so their encodings point in noticeably different directions.";
    peSimStatus.innerHTML = `Token <b>${a}</b> and token <b>${b}</b>: cosine similarity <b>${sim.toFixed(2)}</b>, ${simLabel}. ${distanceNote}`;
    peHeatmap.hidden = false;
    peHeatmapLabelA.textContent = "token " + a;
    peHeatmapLabelB.textContent = "token " + b;
    renderPeHeatmapRow(peHeatmapA, vecA);
    renderPeHeatmapRow(peHeatmapB, vecB);
  }

  function fitCanvas() {
    const rect = peCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    peCanvas.width = Math.max(300, rect.width) * dpr;
    peCanvas.height = 240 * dpr;
    peCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function redrawPE() {
    const i = +peIInput.value;
    const d = +peDInput.value;
    const w = peCanvas.getBoundingClientRect().width || 860;
    const h = 240;
    peCtx.clearRect(0, 0, w, h);

    const border = cssVar("--border");
    const inkFaint = cssVar("--ink-faint");
    const encoder = cssVar("--encoder");
    const decoder = cssVar("--decoder");
    const accent = cssVar("--accent");
    const attn = cssVar("--attn");

    peCtx.strokeStyle = border;
    peCtx.lineWidth = 1;
    peCtx.beginPath();
    peCtx.moveTo(0, h / 2);
    peCtx.lineTo(w, h / 2);
    peCtx.stroke();

    const freq = 1 / Math.pow(10000, (2 * i) / d);
    const marginX = 6;
    const plotW = w - marginX * 2;
    const amp = h / 2 - 34;

    function xFor(pos) {
      return marginX + (pos / PE_N_POS) * plotW;
    }
    function yFor(pos, fn) {
      return h / 2 - fn(pos * freq) * amp;
    }
    function plot(fn, color) {
      peCtx.strokeStyle = color;
      peCtx.lineWidth = 2.2;
      peCtx.lineJoin = "round";
      peCtx.beginPath();
      const step = 0.15;
      for (let pos = 0; pos <= PE_N_POS + 1e-6; pos += step) {
        const x = xFor(pos);
        const y = yFor(pos, fn);
        if (pos === 0) peCtx.moveTo(x, y);
        else peCtx.lineTo(x, y);
      }
      peCtx.stroke();
    }
    plot(Math.sin, encoder);
    plot(Math.cos, decoder);

    // faint markers for every token position, so the wave clearly represents discrete tokens
    for (let pos = 0; pos < PE_TOKEN_COUNT; pos++) {
      peCtx.fillStyle = inkFaint;
      peCtx.globalAlpha = 0.5;
      peCtx.beginPath();
      peCtx.arc(xFor(pos), yFor(pos, Math.sin), 2.6, 0, Math.PI * 2);
      peCtx.fill();
      peCtx.globalAlpha = 1;
    }

    // highlighted guides for the two selected tokens being compared
    peSelected.forEach((pos, idx) => {
      const color = idx === 0 ? accent : attn;
      const x = xFor(pos);
      peCtx.strokeStyle = color;
      peCtx.globalAlpha = 0.45;
      peCtx.lineWidth = 1.4;
      peCtx.setLineDash([3, 3]);
      peCtx.beginPath();
      peCtx.moveTo(x, 12);
      peCtx.lineTo(x, h - 12);
      peCtx.stroke();
      peCtx.setLineDash([]);
      peCtx.globalAlpha = 1;

      [Math.sin, Math.cos].forEach((fn) => {
        peCtx.fillStyle = color;
        peCtx.beginPath();
        peCtx.arc(x, yFor(pos, fn), 5, 0, Math.PI * 2);
        peCtx.fill();
      });

      peCtx.font = "700 11px 'JetBrains Mono', monospace";
      peCtx.fillStyle = color;
      peCtx.fillText(String(pos), x - 4, 24 + idx * 14);
    });

    function labelChip(text, x, y, color) {
      peCtx.font = "600 11.5px 'JetBrains Mono', monospace";
      const pad = 5;
      const metrics = peCtx.measureText(text);
      peCtx.fillStyle = cssVar("--surface");
      peCtx.globalAlpha = 0.92;
      peCtx.fillRect(x - pad, y - 12, metrics.width + pad * 2, 17);
      peCtx.globalAlpha = 1;
      peCtx.fillStyle = color;
      peCtx.fillText(text, x, y);
    }
    labelChip("sin (even dimension)", marginX + 4, 18, encoder);
    labelChip("cos (odd dimension)", marginX + 4, 36, decoder);
    labelChip("position 0 to " + PE_N_POS + ", tokens 0 to " + (PE_TOKEN_COUNT - 1) + " marked", w - 256, h - 12, inkFaint);

    peIVal.textContent = "i = " + i;
    peDVal.textContent = "d_model = " + d;
  }

  peIInput.addEventListener("input", redrawPE);
  peDInput.addEventListener("input", () => {
    const d = +peDInput.value;
    const maxI = Math.max(0, Math.floor(d / 2) - 1);
    peIInput.max = String(maxI);
    if (+peIInput.value > maxI) peIInput.value = String(maxI);
    redrawPE();
    updatePeComparison();
  });
  window.addEventListener("resize", () => {
    fitCanvas();
    redrawPE();
  });
  renderPeTokens();
  fitCanvas();
  redrawPE();

  /* ---------------------------------------------------------
     Scroll-triggered demos: each section's animation plays
     automatically only while that section is in view, and
     replays fresh every time it scrolls back into view.
  --------------------------------------------------------- */
  if (!REDUCE && "IntersectionObserver" in window) {
    const heroSection = document.querySelector(".hero");
    new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) startHeroCycle();
          else stopHeroCycle();
        });
      },
      { threshold: 0.4 }
    ).observe(heroSection);

    new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          rnnNodes.forEach((n) => n.classList.remove("on"));
          attnNodes.forEach((n) => n.classList.remove("on"));
          Array.from(attnCmpSvg.querySelectorAll(".attn-edge")).forEach((e) => e.classList.remove("on"));
          playRnnDemo();
          playAttnDemo();
        });
      },
      { threshold: 0.6 }
    ).observe(document.getElementById("why"));

    new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          resetArchitectureFlow();
          runArchitectureFlow();
        });
      },
      { threshold: 0.5 }
    ).observe(document.getElementById("architecture"));

    new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          resetSdpaDemo();
          playSdpaDemo();
        });
      },
      { threshold: 0.5 }
    ).observe(document.getElementById("inside"));
  }

  /* ---------------------------------------------------------
     GLOSSARY filter
  --------------------------------------------------------- */
  const glossSearch = document.getElementById("glossary-search");
  const glossItems = Array.from(document.querySelectorAll(".gloss-item"));
  const glossEmpty = document.getElementById("gloss-empty");
  glossSearch.addEventListener("input", () => {
    const q = glossSearch.value.trim().toLowerCase();
    let visible = 0;
    glossItems.forEach((item) => {
      const text = item.textContent.toLowerCase();
      const match = !q || text.includes(q);
      item.hidden = !match;
      if (match) visible++;
    });
    glossEmpty.classList.toggle("show", visible === 0);
  });
})();
