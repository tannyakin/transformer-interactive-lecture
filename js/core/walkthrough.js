// ============================================================
// Walkthrough engine: turns a worked example definition into a
// player that autoplays at reading pace, can be stepped by hand,
// and pauses to ask the learner to predict at key moments.
//
// A definition (registered on Lecture.defs by lecture files):
//   { id, title, setup, input, build(input) -> { steps, takeaway },
//     setupBlocks?(input), twist?: { label, offLabel?, input },
//     panel?(el) -> { render(index, step, input) } }
// A step: { title, say, blocks, predict?: { ask, choices, answer, why, hint } }
// A predict on step k is asked before step k is revealed.
// ============================================================
(function (root) {
  "use strict";

  const L = (root.Lecture = root.Lecture || {});
  L.defs = L.defs || {};

  const MIN_DWELL = 4000;
  const BASE_DWELL = 2500;
  const PER_WORD = 60;
  const SPEEDS = [0.75, 1, 1.5];
  const VISIBLE = 0.25; // below this a player counts as scrolled away
  const AUTOPLAY_AT = 0.6;

  function stripTags(s) {
    return String(s || "").replace(/<[^>]*>/g, " ");
  }

  function dwellMs(text, speed) {
    const words = stripTags(text).trim().split(/\s+/).filter(Boolean).length;
    return Math.max(MIN_DWELL, BASE_DWELL + PER_WORD * words) / (speed || 1);
  }

  function assemble(def, input) {
    const built = def.build(input);
    const steps = built.steps.map((s) => Object.assign({ kind: "step" }, s));
    const last = steps[steps.length - 1];
    return [
      {
        kind: "setup",
        title: def.setupTitle || "The setup",
        say: def.setup,
        blocks: def.setupBlocks ? def.setupBlocks(input) : [],
      },
      ...steps,
      { kind: "takeaway", title: "What did the math just say?", say: built.takeaway, blocks: last ? last.blocks : [] },
    ];
  }

  function pickActive(players) {
    const visible = players.filter((p) => p.ratio >= VISIBLE);
    if (!visible.length) return null;
    const touched = visible.filter((p) => p.touchedAt > 0);
    const pool = touched.length ? touched : visible;
    const key = touched.length ? "touchedAt" : "ratio";
    return pool.reduce((best, p) => (p[key] > best[key] ? p : best));
  }

  const pure = { dwellMs, assemble, pickActive };
  if (typeof module !== "undefined" && module.exports) module.exports = pure;
  if (typeof document === "undefined") return;

  /* ---------------------------------------------------------
     Browser side
  --------------------------------------------------------- */
  const M = L.math;
  const store = L.store;
  const REDUCE = root.matchMedia && root.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const players = [];

  const ICON = {
    back: '<svg viewBox="0 0 24 24" aria-hidden="true"><polyline points="15 6 9 12 15 18"/></svg>',
    next: '<svg viewBox="0 0 24 24" aria-hidden="true"><polyline points="9 6 15 12 9 18"/></svg>',
    play: '<svg viewBox="0 0 24 24" aria-hidden="true" class="fill"><path d="M8 5v14l11-7z"/></svg>',
    pause: '<svg viewBox="0 0 24 24" aria-hidden="true" class="fill"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>',
    restart: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 11A8 8 0 1 0 18.5 16"/><polyline points="20 4 20 11 13 11"/></svg>',
    check: '<svg viewBox="0 0 24 24" aria-hidden="true"><polyline points="5 12.5 10 17.5 19 7"/></svg>',
  };

  function h(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }

  function button(cls, label, html) {
    const b = h("button", cls, html);
    b.type = "button";
    b.setAttribute("aria-label", label);
    return b;
  }

  const lectureId = () => document.body.dataset.lecture || "site";

  // Fewest decimals (up to 3) that show every value in the block exactly as the notes do.
  function autoDp(values) {
    const nums = values.filter((v) => typeof v === "number" && isFinite(v));
    for (let dp = 0; dp <= 3; dp++) {
      if (nums.every((v) => Math.abs(M.round(v, dp) - M.round(v, 3)) < 1e-9)) return dp;
    }
    return 3;
  }

  function cellText(v, dp) {
    return typeof v === "number" ? M.fmt(v, dp) : String(v);
  }

  function countUp(span, to, dp) {
    if (REDUCE || !isFinite(to) || to === 0) return;
    const start = performance.now();
    const dur = 520;
    function frame(now) {
      const t = Math.min(1, (now - start) / dur);
      span.textContent = M.fmt(to * (1 - Math.pow(1 - t, 3)), dp);
      if (t < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  /* ---------- block renderers ---------- */
  const RENDER = {
    vector(b) {
      const dp = b.dp != null ? b.dp : autoDp(b.values);
      const hl = b.hl || [];
      const wrap = h("div", "wt-vec" + (b.tone ? " tone-" + b.tone : "") + (b.fresh ? " is-fresh" : ""));
      if (b.cols) {
        const cols = h("div", "wt-vec-row wt-vec-cols");
        cols.append(h("span", "wt-op"), h("span", "wt-label"));
        const cells = h("span", "wt-cells");
        b.cols.forEach((c) => cells.append(h("span", "wt-cell", c)));
        cols.append(cells);
        wrap.append(cols);
      }
      const row = h("div", "wt-vec-row");
      row.append(h("span", "wt-op", b.op || ""), h("span", "wt-label", b.label || ""));
      const cells = h("span", "wt-cells wt-bracket");
      b.values.forEach((v, i) => {
        const c = h("span", "wt-cell" + (hl.includes(i) ? " is-hl" : ""));
        c.textContent = cellText(v, dp);
        cells.append(c);
        if (b.fresh && typeof v === "number") countUp(c, v, dp);
      });
      row.append(cells);
      wrap.append(row);
      return wrap;
    },
    matrix(b) {
      const flat = b.rows.flat();
      const dp = b.dp != null ? b.dp : autoDp(flat);
      const key = (r, c) => r + ":" + c;
      const hl = new Set((b.hl || []).map(([r, c]) => key(r, c)));
      const muted = new Set((b.muted || []).map(([r, c]) => key(r, c)));
      const cols = b.rows[0].length;
      const grid = h("div", "wt-matrix" + (b.fresh ? " is-fresh" : ""));
      grid.style.setProperty("--cols", cols);
      grid.style.setProperty("--has-rows", b.rowLabels ? 1 : 0);
      if (b.colLabels) {
        if (b.rowLabels) grid.append(h("span", "wt-mlabel"));
        b.colLabels.forEach((c) => grid.append(h("span", "wt-mlabel is-col", c)));
      }
      b.rows.forEach((row, r) => {
        if (b.rowLabels) grid.append(h("span", "wt-mlabel is-row", b.rowLabels[r]));
        row.forEach((v, c) => {
          const cls = "wt-mcell" + (hl.has(key(r, c)) ? " is-hl" : "") + (muted.has(key(r, c)) ? " is-muted" : "");
          const cell = h("span", cls);
          cell.textContent = cellText(v, dp);
          grid.append(cell);
        });
      });
      return grid;
    },
    lines(b) {
      const box = h("div", "wt-lines" + (b.fresh ? " is-fresh" : ""));
      b.lines.forEach((ln) => {
        const o = typeof ln === "string" ? { t: ln } : ln;
        box.append(h("div", "wt-line" + (o.hl ? " is-hl" : ""), o.t));
      });
      return box;
    },
    bars(b) {
      const hl = b.hl || [];
      const dp = b.dp != null ? b.dp : 3;
      const box = h("div", "wt-bars" + (b.fresh ? " is-fresh" : ""));
      if (b.ghost && b.ghostLabel) {
        box.append(h("p", "wt-bars-key", `<span class="k-now"></span>${b.label || "now"} <span class="k-ghost"></span>${b.ghostLabel}`));
      }
      b.values.forEach((v, i) => {
        const row = h("div", "wt-bar" + (hl.includes(i) ? " is-hl" : ""));
        row.append(h("span", "wt-bar-label", b.labels[i]));
        const track = h("span", "wt-bar-track");
        if (b.ghost) {
          const g = h("span", "wt-bar-ghost");
          g.style.width = b.ghost[i] * 100 + "%";
          track.append(g);
        }
        const fill = h("span", "wt-bar-fill");
        fill.style.setProperty("--w", v * 100 + "%");
        track.append(fill);
        row.append(track, h("span", "wt-bar-val", M.fmt(v, dp)));
        box.append(row);
      });
      return box;
    },
    note(b) {
      return h("div", "wt-note" + (b.tone ? " tone-" + b.tone : "") + (b.fresh ? " is-fresh" : ""), b.html);
    },
  };

  /* ---------- a single player ---------- */
  function mount(el, def) {
    const lec = lectureId();
    const p = {
      def,
      root: null,
      input: def.input,
      twistOn: false,
      steps: [],
      index: 0,
      playing: false,
      userPaused: false,
      waiting: null, // the open predict question, if any
      elapsed: 0,
      lastTick: null,
      raf: 0,
      ratio: 0,
      touchedAt: 0,
      finished: store.isWalkDone(lec, def.id),
      answered: new Set(),
      speed: SPEEDS.includes(store.getPref("speed", 1)) ? store.getPref("speed", 1) : 1,
    };

    // ----- DOM -----
    const rootEl = h("div", "wt" + (def.panel ? " has-panel" : "") + (REDUCE ? " is-reduced" : ""));
    rootEl.tabIndex = 0;
    rootEl.setAttribute("role", "region");
    rootEl.setAttribute("aria-label", "Worked example: " + stripTags(def.title));
    p.root = rootEl;

    const head = h("div", "wt-head");
    const titles = h("div", "wt-titles");
    titles.append(h("p", "wt-kicker", "Worked example"), h("p", "wt-title", def.title));
    const doneBadge = h("span", "wt-done", ICON.check + "<span>Finished</span>");
    head.append(titles, doneBadge);
    let twistBtn = null;
    if (def.twist) {
      twistBtn = h("button", "wt-twist", def.twist.label);
      twistBtn.type = "button";
      twistBtn.setAttribute("aria-pressed", "false");
      head.append(twistBtn);
    }

    const body = h("div", "wt-body");
    let panel = null;
    if (def.panel) {
      const panelEl = h("div", "wt-panel");
      body.append(panelEl);
      panel = def.panel(panelEl);
    }
    const stage = h("div", "wt-stage");
    const stepLabel = h("p", "wt-step-label");
    const blocksEl = h("div", "wt-blocks");
    const sayEl = h("p", "wt-say");
    sayEl.setAttribute("aria-live", "polite");
    const predictEl = h("div", "wt-predict");
    predictEl.hidden = true;
    stage.append(stepLabel, blocksEl, sayEl, predictEl);
    body.append(stage);

    const controls = h("div", "wt-controls");
    const backBtn = button("wt-btn wt-back", "Previous step", ICON.back + "<span>Back</span>");
    const playBtn = button("wt-btn wt-play", "Play", ICON.play);
    const nextBtn = button("wt-btn wt-next", "Next step", "<span>Next</span>" + ICON.next);
    const dotsEl = h("div", "wt-dots");
    const speedBtn = button("wt-btn wt-speed", "Playback speed", "");
    const restartBtn = button("wt-btn wt-restart", "Start this example again", ICON.restart);
    const primary = h("div", "wt-primary");
    primary.append(backBtn, playBtn, nextBtn);
    const secondary = h("div", "wt-secondary");
    secondary.append(speedBtn, restartBtn);
    controls.append(primary, dotsEl, secondary);

    rootEl.append(head, body, controls);
    el.replaceChildren(rootEl);

    // ----- rendering -----
    function buildSteps() {
      p.steps = assemble(def, p.input);
      dotsEl.replaceChildren();
      p.steps.forEach((s, i) => {
        const label = s.kind === "setup" ? "The setup" : s.kind === "takeaway" ? "The takeaway" : "Step " + i;
        const d = button("wt-dot", label, "");
        d.addEventListener("click", () => goto(i, true));
        dotsEl.append(d);
      });
    }

    function render() {
      const step = p.steps[p.index];
      const total = p.steps.length - 2;
      rootEl.classList.toggle("is-setup", step.kind === "setup");
      rootEl.classList.toggle("is-takeaway", step.kind === "takeaway");
      if (step.kind === "step") {
        stepLabel.innerHTML = `<span class="wt-count">Step ${p.index} of ${total}</span> ${step.title}`;
      } else {
        stepLabel.innerHTML = `<span class="wt-count">${step.kind === "setup" ? "Before we start" : "Takeaway"}</span> ${step.title}`;
      }
      blocksEl.replaceChildren(...(step.blocks || []).map((b) => {
        const make = RENDER[b.type];
        if (!make) throw new Error("Unknown walkthrough block type: " + b.type);
        // Only animate a block the first time it appears, not when revisiting the takeaway.
        return make(step.kind === "takeaway" ? Object.assign({}, b, { fresh: false }) : b);
      }));
      sayEl.innerHTML = step.say || "";
      Array.from(dotsEl.children).forEach((d, i) => {
        d.classList.toggle("is-on", i <= p.index);
        if (i === p.index) d.setAttribute("aria-current", "step");
        else d.removeAttribute("aria-current");
      });
      backBtn.disabled = p.index === 0;
      nextBtn.disabled = p.index === p.steps.length - 1;
      if (panel) panel.render(p.index, step, p.input);
      if (step.kind === "takeaway") {
        p.finished = true;
        rootEl.classList.add("is-done");
        store.markWalk(lec, def.id);
        stop(false);
      }
      setProgress(0);
    }

    function setProgress(f) {
      nextBtn.style.setProperty("--wt-progress", f);
    }

    function syncPlayButton() {
      const showPause = p.playing;
      playBtn.innerHTML = showPause ? ICON.pause : ICON.play;
      playBtn.setAttribute("aria-label", showPause ? "Pause" : p.index === p.steps.length - 1 ? "Play again" : "Play");
      rootEl.classList.toggle("is-playing", p.playing);
    }

    function syncSpeed() {
      speedBtn.textContent = p.speed + "x";
      speedBtn.setAttribute("aria-label", "Playback speed " + p.speed + "x. Change speed");
    }

    // ----- timing -----
    function tick(now) {
      if (!p.playing) return;
      if (p.lastTick == null) p.lastTick = now;
      p.elapsed += now - p.lastTick;
      p.lastTick = now;
      const step = p.steps[p.index];
      const d = dwellMs(step.say, p.speed);
      setProgress(Math.min(1, p.elapsed / d));
      if (p.elapsed >= d) {
        advance(true);
        if (!p.playing) return;
      }
      p.raf = requestAnimationFrame(tick);
    }

    function play() {
      if (REDUCE) return;
      if (p.waiting) return;
      if (p.index === p.steps.length - 1) {
        p.answered.clear();
        goto(0, false);
      }
      p.userPaused = false;
      if (p.playing) return;
      p.playing = true;
      p.lastTick = null;
      cancelAnimationFrame(p.raf);
      p.raf = requestAnimationFrame(tick);
      syncPlayButton();
    }

    // byUser: the learner chose to stop, so scrolling back must not restart it.
    function stop(byUser) {
      if (byUser) p.userPaused = true;
      p.playing = false;
      p.lastTick = null;
      cancelAnimationFrame(p.raf);
      syncPlayButton();
    }

    // ----- navigation -----
    function goto(i, byUser) {
      const target = Math.max(0, Math.min(p.steps.length - 1, i));
      closePredict();
      if (byUser) stop(true);
      p.index = target;
      p.elapsed = 0;
      render();
      syncPlayButton();
    }

    function advance(auto) {
      if (p.index >= p.steps.length - 1) return;
      const nextStep = p.steps[p.index + 1];
      if (nextStep.predict && !p.answered.has(p.index + 1)) {
        openPredict(p.index + 1, auto);
        return;
      }
      if (auto) {
        p.elapsed = 0;
        p.index += 1;
        render();
      } else {
        goto(p.index + 1, true);
      }
    }

    // ----- predict stops -----
    function openPredict(revealIndex, resumeAfter) {
      const q = p.steps[revealIndex].predict;
      const wasPlaying = p.playing;
      stop(false);
      p.waiting = { revealIndex, resume: resumeAfter && wasPlaying, timer: 0 };
      predictEl.replaceChildren();
      predictEl.append(h("p", "wt-predict-kicker", "Your turn: predict it"), h("p", "wt-predict-ask", q.ask));
      const choices = h("div", "wt-choices");
      const feedback = h("p", "wt-feedback");
      feedback.setAttribute("aria-live", "polite");
      const skip = h("button", "wt-skip", "Just show me");
      skip.type = "button";
      q.choices.forEach((c, i) => {
        const b = h("button", "wt-choice", c);
        b.type = "button";
        b.addEventListener("click", () => {
          if (i === q.answer) {
            b.classList.add("is-right");
            Array.from(choices.children).forEach((x) => (x.disabled = true));
            feedback.className = "wt-feedback is-right";
            feedback.innerHTML = "<b>Yes.</b> " + q.why;
            skip.textContent = "Show the step";
            if (p.waiting && p.waiting.resume) {
              p.waiting.timer = setTimeout(() => reveal(true), dwellMs(q.why, p.speed));
            }
          } else {
            b.classList.add("is-wrong");
            b.disabled = true;
            feedback.className = "wt-feedback is-wrong";
            feedback.innerHTML = "<b>Not quite.</b> " + (q.hint || "Have another look at the numbers above.");
          }
        });
        choices.append(b);
      });
      skip.addEventListener("click", () => reveal(p.waiting && p.waiting.resume));
      predictEl.append(choices, feedback, skip);
      predictEl.hidden = false;
      rootEl.classList.add("is-asking");
      syncPlayButton();
      const first = choices.querySelector("button");
      if (first && rootEl.contains(document.activeElement)) first.focus({ preventScroll: true });
    }

    function closePredict() {
      if (p.waiting) clearTimeout(p.waiting.timer);
      p.waiting = null;
      predictEl.hidden = true;
      predictEl.replaceChildren();
      rootEl.classList.remove("is-asking");
    }

    function reveal(resume) {
      if (!p.waiting) return;
      const i = p.waiting.revealIndex;
      p.answered.add(i);
      closePredict();
      p.index = i;
      p.elapsed = 0;
      render();
      if (resume && !p.userPaused) play();
      else syncPlayButton();
    }

    // ----- twist -----
    function setTwist(on) {
      const wasPlaying = p.playing;
      stop(false);
      closePredict();
      p.twistOn = on;
      p.input = on ? Object.assign({}, def.input, def.twist.input) : def.input;
      p.answered.clear();
      buildSteps();
      p.index = 1;
      p.elapsed = 0;
      render();
      twistBtn.textContent = on ? def.twist.offLabel || "Back to the original numbers" : def.twist.label;
      twistBtn.setAttribute("aria-pressed", String(on));
      rootEl.classList.toggle("is-twisted", on);
      if (wasPlaying) play();
      else syncPlayButton();
    }

    function restart() {
      p.answered.clear();
      goto(0, false);
      if (!REDUCE) play();
    }

    // ----- wiring -----
    backBtn.addEventListener("click", () => goto(p.index - 1, true));
    nextBtn.addEventListener("click", () => {
      stop(true);
      if (p.waiting) reveal(false);
      else advance(false);
    });
    playBtn.addEventListener("click", () => (p.playing ? stop(true) : play()));
    speedBtn.addEventListener("click", () => {
      p.speed = SPEEDS[(SPEEDS.indexOf(p.speed) + 1) % SPEEDS.length];
      store.setPref("speed", p.speed);
      syncSpeed();
    });
    restartBtn.addEventListener("click", restart);
    if (twistBtn) twistBtn.addEventListener("click", () => setTwist(!p.twistOn));
    const touch = () => (p.touchedAt = performance.now());
    rootEl.addEventListener("pointerdown", touch);
    rootEl.addEventListener("focusin", touch);

    buildSteps();
    render();
    syncSpeed();
    syncPlayButton();
    if (p.finished) rootEl.classList.add("is-done");

    Object.assign(p, {
      goto: (i) => goto(i, true),
      next: () => nextBtn.click(),
      back: () => goto(p.index - 1, true),
      play,
      pause: () => stop(true),
      restart,
      setTwist,
      onVisibility(ratio) {
        p.ratio = ratio;
        if (ratio >= AUTOPLAY_AT && !p.playing && !p.userPaused && !p.finished && !p.waiting && !document.hidden) play();
        else if (ratio < VISIBLE && p.playing) stop(false);
      },
    });
    players.push(p);
    if (observer) observer.observe(rootEl);
    return p;
  }

  /* ---------- shared observers ---------- */
  // A tall walkthrough on a phone can never be 60% on screen, so visibility
  // is measured against whichever is smaller: the element or the viewport.
  const observer =
    "IntersectionObserver" in root
      ? new IntersectionObserver(
          (entries) => {
            entries.forEach((e) => {
              const p = players.find((x) => x.root === e.target);
              if (!p) return;
              const vh = e.rootBounds ? e.rootBounds.height : root.innerHeight;
              const basis = Math.min(e.boundingClientRect.height, vh) || 1;
              p.onVisibility(e.isIntersecting ? Math.min(1, e.intersectionRect.height / basis) : 0);
            });
          },
          { threshold: Array.from({ length: 21 }, (_, i) => i / 20) }
        )
      : null;

  document.addEventListener("visibilitychange", () => {
    players.forEach((p) => {
      if (document.hidden) {
        if (p.playing) {
          p.pause();
          p.userPaused = false; // the tab was hidden, the learner did not ask to stop
        }
      } else {
        p.onVisibility(p.ratio);
      }
    });
  });

  document.addEventListener("keydown", (e) => {
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
    if (document.querySelector("dialog[open]")) return;
    const t = e.target;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    const p = pickActive(players);
    if (!p) return;
    if (e.key === "ArrowRight") { p.next(); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { p.back(); e.preventDefault(); }
    else if (e.key === " " && t === p.root) {
      if (p.playing) p.pause();
      else p.play();
      e.preventDefault();
    }
  });

  function mountAll() {
    document.querySelectorAll("[data-walk]").forEach((el) => {
      const def = L.defs[el.dataset.walk];
      if (!def) {
        el.innerHTML = '<p class="wt-missing">This worked example could not load. Try refreshing the page.</p>';
        console.warn("No walkthrough definition for", el.dataset.walk);
        return;
      }
      mount(el, def);
    });
  }

  L.Walkthrough = Object.assign({ mount, players }, pure);
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mountAll);
  else mountAll();
})(typeof window !== "undefined" ? window : globalThis);
