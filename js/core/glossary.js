// ============================================================
// Glossary: tooltips for <abbr data-term="..."> and the glossary
// list at the end of a lecture. Shared by every lecture page.
// The page loads its own data first (for example
// js/lecture-2/glossary-data.js), which sets Lecture.glossary
// to an array of { term, full, meaning }.
//
// Tooltips open on hover, keyboard focus and tap. One is open at
// a time, Escape closes it, and it is position: fixed so no
// scrolling box can clip it. It flips above or below the word to
// stay on screen and keeps 16px clear of the screen edges.
// ============================================================
(function (root) {
  "use strict";

  const norm = (s) => String(s == null ? "" : s).trim().toLowerCase();

  // Case-insensitive exact match on the term. Returns the entry or null.
  function findTerm(list, key) {
    const k = norm(key);
    if (!k || !Array.isArray(list)) return null;
    for (const item of list) if (norm(item.term) === k) return item;
    return null;
  }

  // Entries whose term, full name or meaning contains the query.
  // An empty query returns every entry.
  function filterTerms(list, query) {
    if (!Array.isArray(list)) return [];
    const q = norm(query);
    if (!q) return list.slice();
    return list.filter((item) =>
      [item.term, item.full, item.meaning].some((field) => norm(field).includes(q))
    );
  }

  // Sort key: ignore brackets so [CLS] sits with the Cs.
  function sortKey(term) {
    return norm(term).replace(/[^a-z0-9]/g, "");
  }
  function sortTerms(list) {
    return list.slice().sort((a, b) => sortKey(a.term).localeCompare(sortKey(b.term)) || a.term.localeCompare(b.term));
  }

  const pure = { findTerm, filterTerms, sortTerms };
  if (typeof module !== "undefined" && module.exports) module.exports = pure;
  if (typeof document === "undefined") return;

  const L = (root.Lecture = root.Lecture || {});
  const EDGE = 16; // keep this far from the screen edges
  const GAP = 8; // space between the word and the tooltip
  const TAP_GRACE = 400; // ms: a tap's focus-then-click should open, not open-then-close

  let list = [];
  let pop, popFull, popMeaning, descBox;
  let current = null;
  let openedAt = 0;
  let hideTimer = 0;
  let descCount = 0;
  const descIds = new Map(); // term (lowercase) -> id of its hidden description
  const warned = new Set();

  /* ---------- tooltip element ---------- */
  function buildPopover() {
    pop = document.createElement("div");
    pop.className = "gl-pop";
    pop.id = "gl-pop";
    pop.setAttribute("role", "tooltip");
    // The words are announced through each abbr's aria-describedby,
    // so the floating copy is hidden from screen readers to avoid a repeat.
    pop.setAttribute("aria-hidden", "true");
    popFull = document.createElement("strong");
    popFull.className = "gl-pop-full";
    popMeaning = document.createElement("span");
    popMeaning.className = "gl-pop-meaning";
    pop.append(popFull, popMeaning);
    pop.addEventListener("mouseenter", () => clearTimeout(hideTimer));
    pop.addEventListener("mouseleave", () => scheduleHide());
    document.body.append(pop);

    descBox = document.createElement("div");
    descBox.hidden = true;
    document.body.append(descBox);
  }

  function descriptionId(entry) {
    const k = norm(entry.term);
    if (descIds.has(k)) return descIds.get(k);
    const id = "gl-desc-" + ++descCount;
    const span = document.createElement("span");
    span.id = id;
    span.textContent = entry.full + ". " + entry.meaning;
    descBox.append(span);
    descIds.set(k, id);
    return id;
  }

  /* ---------- wiring each abbr ---------- */
  function enhance(scope) {
    const nodes = [];
    if (scope.matches && scope.matches("abbr[data-term]")) nodes.push(scope);
    if (scope.querySelectorAll) nodes.push(...scope.querySelectorAll("abbr[data-term]"));
    const unknown = [];
    nodes.forEach((abbr) => {
      if (abbr.dataset.glReady) return;
      const entry = findTerm(list, abbr.dataset.term);
      if (!entry) {
        const k = abbr.dataset.term;
        if (!warned.has(k)) {
          warned.add(k);
          unknown.push(k);
        }
        return;
      }
      abbr.dataset.glReady = "1";
      abbr.tabIndex = 0;
      // A native title would show a second, unstyled tooltip.
      abbr.removeAttribute("title");
      const ids = (abbr.getAttribute("aria-describedby") || "").split(/\s+/).filter(Boolean);
      const id = descriptionId(entry);
      if (!ids.includes(id)) ids.push(id);
      abbr.setAttribute("aria-describedby", ids.join(" "));
    });
    if (unknown.length) {
      console.warn("Glossary: no entry for " + unknown.map((t) => '"' + t + '"').join(", ") + ". Those abbreviations get no tooltip.");
    }
  }

  function abbrFrom(target) {
    const el = target && target.closest ? target.closest("abbr[data-term]") : null;
    return el && el.dataset.glReady ? el : null;
  }

  /* ---------- show, place, hide ---------- */
  function show(abbr) {
    clearTimeout(hideTimer);
    const entry = findTerm(list, abbr.dataset.term);
    if (!entry) return;
    if (current !== abbr) {
      if (current) current.removeAttribute("data-gl-open");
      current = abbr;
      openedAt = Date.now();
      popFull.textContent = entry.full;
      popMeaning.textContent = entry.meaning;
    }
    abbr.setAttribute("data-gl-open", "");
    pop.classList.add("is-open");
    place();
  }

  function hide() {
    clearTimeout(hideTimer);
    if (!current) return;
    current.removeAttribute("data-gl-open");
    current = null;
    pop.classList.remove("is-open");
  }

  function scheduleHide() {
    clearTimeout(hideTimer);
    hideTimer = setTimeout(() => {
      // Keep it open while the word still has keyboard focus.
      if (current && document.activeElement === current) return;
      hide();
    }, 120);
  }

  function place() {
    if (!current) return;
    // A word that wraps onto two lines: anchor to the first line.
    const rects = current.getClientRects();
    const r = rects.length ? rects[0] : current.getBoundingClientRect();
    const vw = document.documentElement.clientWidth || root.innerWidth;
    const vh = root.innerHeight;
    if (r.bottom < 0 || r.top > vh) {
      hide();
      return;
    }
    pop.style.maxWidth = Math.max(0, Math.min(320, vw - 2 * EDGE)) + "px";
    const w = pop.offsetWidth;
    const h = pop.offsetHeight;

    const centre = r.left + r.width / 2;
    let left = centre - w / 2;
    left = Math.max(EDGE, Math.min(left, vw - EDGE - w));

    const below = r.bottom + GAP;
    const above = r.top - GAP - h;
    let side = "below";
    let top = below;
    if (below + h > vh - EDGE && above >= EDGE) {
      side = "above";
      top = above;
    } else if (below + h > vh - EDGE) {
      // Not enough room either way: use the side with more space, clamped.
      if (r.top > vh - r.bottom) {
        side = "above";
        top = Math.max(EDGE, above);
      } else {
        top = Math.min(below, vh - EDGE - h);
      }
    }

    pop.dataset.side = side;
    pop.style.left = Math.round(left) + "px";
    pop.style.top = Math.round(top) + "px";
    const arrow = Math.max(14, Math.min(w - 14, centre - left));
    pop.style.setProperty("--gl-arrow-x", Math.round(arrow) + "px");
  }

  let placing = false;
  function onViewportChange() {
    if (!current || placing) return;
    placing = true;
    requestAnimationFrame(() => {
      placing = false;
      place();
    });
  }

  function bindEvents() {
    document.addEventListener("mouseover", (e) => {
      const abbr = abbrFrom(e.target);
      if (abbr) show(abbr);
    });
    document.addEventListener("mouseout", (e) => {
      const abbr = abbrFrom(e.target);
      if (abbr && abbr === current && !abbr.contains(e.relatedTarget) && !pop.contains(e.relatedTarget)) scheduleHide();
    });
    document.addEventListener("focusin", (e) => {
      const abbr = abbrFrom(e.target);
      if (abbr) show(abbr);
    });
    document.addEventListener("focusout", (e) => {
      const abbr = abbrFrom(e.target);
      if (abbr && abbr === current) hide();
    });
    document.addEventListener("click", (e) => {
      const abbr = abbrFrom(e.target);
      if (!abbr) return;
      // A tap focuses the word (which opens it) and then clicks it.
      // Only a later tap on an open word closes it.
      if (abbr === current && Date.now() - openedAt > TAP_GRACE) hide();
      else show(abbr);
    });
    document.addEventListener("keydown", (e) => {
      if (!current) return;
      if (e.key === "Escape") {
        hide();
        return;
      }
      const abbr = abbrFrom(e.target);
      if (abbr && (e.key === "Enter" || e.key === " ")) {
        e.preventDefault();
        if (abbr === current && pop.classList.contains("is-open")) hide();
        else show(abbr);
      }
    });
    document.addEventListener("pointerdown", (e) => {
      if (!current) return;
      if (abbrFrom(e.target) || pop.contains(e.target)) return;
      hide();
    });
    document.addEventListener("scroll", onViewportChange, { passive: true, capture: true });
    root.addEventListener("resize", onViewportChange);
    if (root.visualViewport) root.visualViewport.addEventListener("resize", onViewportChange);

    // Walkthroughs and interactives add text later; wire any new abbrs too.
    if ("MutationObserver" in root) {
      new MutationObserver((records) => {
        records.forEach((rec) => rec.addedNodes.forEach((n) => {
          if (n.nodeType === 1 && n !== pop) enhance(n);
        }));
        if (current && !current.isConnected) hide();
      }).observe(document.body, { childList: true, subtree: true });
    }
  }

  /* ---------- the glossary section ---------- */
  function renderList() {
    const dl = document.getElementById("glossary-list");
    if (!dl) return;
    const items = new Map();
    dl.replaceChildren();
    sortTerms(list).forEach((entry) => {
      const row = document.createElement("div");
      row.className = "gl-row";
      const dt = document.createElement("dt");
      dt.className = "gl-term";
      dt.textContent = entry.term;
      const dd = document.createElement("dd");
      dd.className = "gl-def";
      const full = document.createElement("span");
      full.className = "gl-full";
      full.textContent = entry.full;
      const meaning = document.createElement("span");
      meaning.className = "gl-meaning";
      meaning.textContent = entry.meaning;
      dd.append(full, meaning);
      row.append(dt, dd);
      dl.append(row);
      items.set(entry, row);
    });

    const search = document.getElementById("glossary-search");
    const empty = document.getElementById("glossary-empty");
    const count = document.getElementById("glossary-count");
    const total = list.length;

    function apply() {
      const q = search ? search.value : "";
      const hits = new Set(filterTerms(list, q));
      items.forEach((row, entry) => (row.hidden = !hits.has(entry)));
      if (empty) {
        empty.hidden = hits.size > 0;
        const what = empty.querySelector(".gl-empty-q");
        if (what) what.textContent = q.trim();
      }
      if (count) {
        count.textContent = !q.trim()
          ? `${total} abbreviations`
          : hits.size === 0
            ? "No matches"
            : `${hits.size} of ${total} abbreviations`;
      }
    }

    if (search) {
      search.addEventListener("input", apply);
      search.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && search.value) {
          search.value = "";
          apply();
        }
      });
    }
    apply();
  }

  function init() {
    list = Array.isArray(L.glossary) ? L.glossary : [];
    buildPopover();
    enhance(document.body);
    bindEvents();
    renderList();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();

  L.glossaryUI = { hide, enhance: (el) => enhance(el || document.body) };
})(typeof window !== "undefined" ? window : globalThis);
