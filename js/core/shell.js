// ============================================================
// Page shell: theme, slim top bar, contents drawer, reading
// progress, finished-part ticks and "pick up where you left off".
// It reads the page's own <section class="part"> elements, so the
// drawer and the part counter can never drift from the content.
// ============================================================
(function (root) {
  "use strict";

  // Index of the last part whose top has reached the reading line, or -1 in the hero.
  function currentIndex(tops, line) {
    let idx = -1;
    tops.forEach((t, i) => {
      if (t <= line) idx = i;
    });
    return idx;
  }

  // "p2-1:Learned positions|p2-2:Sinusoidal" -> [{ id, title }, ...]
  function parseSubs(attr) {
    if (!attr) return [];
    return attr.split("|").filter(Boolean).map((pair) => {
      const cut = pair.indexOf(":");
      return { id: pair.slice(0, cut).trim(), title: pair.slice(cut + 1).trim() };
    });
  }

  const pure = { currentIndex, parseSubs };
  if (typeof module !== "undefined" && module.exports) module.exports = pure;
  if (typeof document === "undefined") return;

  const L = (root.Lecture = root.Lecture || {});
  const store = L.store;
  const lec = document.body.dataset.lecture || "site";
  const THEME_KEY = "transformer-theme";
  const READ_LINE = 0.3;

  const CHECK = '<svg viewBox="0 0 24 24" aria-hidden="true"><polyline points="5 12.5 10 17.5 19 7"/></svg>';

  const themeBtn = document.getElementById("theme-toggle");
  const partBtn = document.getElementById("part-button");
  const dialog = document.getElementById("contents");
  const progressFill = document.getElementById("progress-fill");
  const toast = document.getElementById("resume-toast");

  /* ---------- theme ---------- */
  function currentTheme() {
    return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
  }
  function syncThemeButton() {
    if (!themeBtn) return;
    const t = currentTheme();
    themeBtn.setAttribute("aria-pressed", String(t === "light"));
    themeBtn.setAttribute("aria-label", t === "light" ? "Switch to dark theme" : "Switch to light theme");
  }
  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      const next = currentTheme() === "light" ? "dark" : "light";
      document.documentElement.setAttribute("data-theme", next);
      try {
        localStorage.setItem(THEME_KEY, next);
      } catch (e) {
        // Theme still switches for this visit.
      }
      syncThemeButton();
      document.dispatchEvent(new CustomEvent("lecture:theme", { detail: next }));
    });
  }
  syncThemeButton();

  /* ---------- parts ---------- */
  const parts = Array.from(document.querySelectorAll("section.part")).map((el, index) => ({
    el,
    index,
    id: el.id,
    title: el.dataset.title || el.id,
    subs: parseSubs(el.dataset.sub),
  }));
  const total = parts.length;
  let cur = -2;

  const pbNum = partBtn && partBtn.querySelector(".pb-num");
  const pbTotal = partBtn && partBtn.querySelector(".pb-total");
  const pbTitle = partBtn && partBtn.querySelector(".pb-title");
  if (pbTotal) pbTotal.textContent = total;

  function renderPartButton() {
    if (!partBtn) return;
    const inPart = cur >= 0;
    partBtn.classList.toggle("in-part", inPart);
    if (pbNum) pbNum.textContent = inPart ? cur + 1 : "";
    if (pbTitle) pbTitle.textContent = inPart ? parts[cur].title : "Contents";
    partBtn.setAttribute(
      "aria-label",
      inPart ? `Part ${cur + 1} of ${total}, ${parts[cur].title}. Open contents` : "Open contents"
    );
  }

  /* ---------- contents drawer ---------- */
  const tocList = dialog && dialog.querySelector(".toc");
  const tocCount = dialog && dialog.querySelector(".toc-count");
  const tocResume = dialog && dialog.querySelector(".toc-resume");

  function buildToc() {
    if (!tocList) return;
    tocList.replaceChildren();
    parts.forEach((p) => {
      const li = document.createElement("li");
      li.className = "toc-item";
      li.dataset.id = p.id;
      const a = document.createElement("a");
      a.href = "#" + p.id;
      a.className = "toc-link";
      a.innerHTML = `<span class="toc-num">${p.index + 1}</span><span class="toc-title"></span><span class="toc-state">${CHECK}<span class="sr-only">finished</span></span>`;
      a.querySelector(".toc-title").textContent = p.title;
      li.append(a);
      if (p.subs.length) {
        const sub = document.createElement("ol");
        sub.className = "toc-sub";
        p.subs.forEach((s) => {
          const sli = document.createElement("li");
          const sa = document.createElement("a");
          sa.href = "#" + s.id;
          sa.textContent = s.title;
          sli.append(sa);
          sub.append(sli);
        });
        li.append(sub);
      }
      tocList.append(li);
    });
    syncToc();
  }

  function syncToc() {
    if (!tocList) return;
    let done = 0;
    parts.forEach((p) => {
      const li = tocList.querySelector(`[data-id="${p.id}"]`);
      const finished = store.isPartDone(lec, p.id);
      if (finished) done++;
      li.classList.toggle("is-done", finished);
      li.classList.toggle("is-current", p.index === cur);
      const link = li.querySelector(".toc-link");
      if (p.index === cur) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
    if (tocCount) tocCount.textContent = `${done} of ${total} parts finished`;
    if (tocResume) {
      const last = parts.find((p) => p.id === store.getLast(lec));
      tocResume.hidden = !last || last.index === 0 || last.index === cur;
      if (last) {
        tocResume.href = "#" + last.id;
        tocResume.querySelector(".toc-resume-where").textContent = `Part ${last.index + 1}, ${last.title}`;
      }
    }
  }

  if (dialog && partBtn) {
    partBtn.addEventListener("click", () => {
      syncToc();
      dialog.showModal();
      const target = tocList.querySelector(".is-current .toc-link") || tocList.querySelector(".toc-link");
      if (target) target.focus();
    });
    dialog.addEventListener("click", (e) => {
      // A click on the dialog element itself is a click on the backdrop.
      if (e.target === dialog || e.target.closest("a[href^='#']") || e.target.closest("[data-close]")) dialog.close();
    });
    dialog.addEventListener("close", () => partBtn.focus({ preventScroll: true }));
  }

  /* ---------- welcome back ---------- */
  function hideToast() {
    if (!toast || toast.hidden) return;
    toast.classList.remove("is-shown");
    setTimeout(() => (toast.hidden = true), 250);
  }
  function maybeShowToast() {
    if (!toast || location.hash) return;
    const last = parts.find((p) => p.id === store.getLast(lec));
    if (!last || last.index === 0 || root.scrollY > 200) return;
    const link = toast.querySelector(".toast-go");
    link.href = "#" + last.id;
    toast.querySelector(".toast-where").textContent = `Part ${last.index + 1}, ${last.title}`;
    toast.hidden = false;
    requestAnimationFrame(() => toast.classList.add("is-shown"));
    link.addEventListener("click", hideToast);
    toast.querySelector(".toast-close").addEventListener("click", hideToast);
  }

  /* ---------- scroll: current part + reading progress ---------- */
  let ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      ticking = false;
      const line = root.innerHeight * READ_LINE;
      const i = currentIndex(parts.map((p) => p.el.getBoundingClientRect().top), line);
      if (i !== cur) {
        cur = i;
        renderPartButton();
        syncToc();
        if (i >= 0) store.setLast(lec, parts[i].id);
      }
      if (progressFill) {
        const h = document.documentElement;
        const max = h.scrollHeight - h.clientHeight;
        progressFill.style.transform = `scaleX(${max > 0 ? Math.min(1, h.scrollTop / max) : 0})`;
      }
      if (root.scrollY > 600) hideToast();
    });
  }

  /* ---------- finished parts ---------- */
  if ("IntersectionObserver" in root) {
    const bridgeObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (!e.isIntersecting) return;
          const part = e.target.closest("section.part");
          if (part && !store.isPartDone(lec, part.id)) {
            store.markPart(lec, part.id);
            syncToc();
          }
        });
      },
      { threshold: 0.6 }
    );
    document.querySelectorAll(".part-bridge").forEach((b) => bridgeObserver.observe(b));
  }

  // Remember where the learner is before anything scrolls, so the toast reflects the last visit.
  maybeShowToast();
  buildToc();
  document.addEventListener("scroll", onScroll, { passive: true });
  root.addEventListener("resize", onScroll);
  onScroll();

  L.shell = {
    currentPart() {
      return cur >= 0 ? { id: parts[cur].id, index: cur, title: parts[cur].title } : null;
    },
  };
})(typeof window !== "undefined" ? window : globalThis);
