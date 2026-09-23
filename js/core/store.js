// ============================================================
// Progress store: which parts and walkthroughs a student has
// finished, where they stopped, and small preferences.
// Storage can be missing, full, blocked or corrupted. None of
// that may break the page, so every access is guarded and the
// in-memory copy keeps working for the current visit.
// ============================================================
(function (root) {
  "use strict";

  const KEY = "unipod-lectures-v1";

  function fresh() {
    return { v: 1, prefs: {} };
  }

  function createStore(storage) {
    let data = load();

    function load() {
      if (!storage) return fresh();
      try {
        const raw = storage.getItem(KEY);
        const parsed = raw ? JSON.parse(raw) : null;
        return parsed && parsed.v === 1 ? parsed : fresh();
      } catch (e) {
        return fresh();
      }
    }

    function save() {
      if (!storage) return;
      try {
        storage.setItem(KEY, JSON.stringify(data));
      } catch (e) {
        // Quota or privacy mode: progress lives only for this visit.
      }
    }

    function lecture(lec) {
      if (!data[lec]) data[lec] = { parts: {}, walk: {}, last: null };
      return data[lec];
    }

    return {
      markPart(lec, part) { lecture(lec).parts[part] = true; save(); },
      isPartDone(lec, part) { return Boolean(data[lec] && data[lec].parts && data[lec].parts[part]); },
      markWalk(lec, id) { lecture(lec).walk[id] = true; save(); },
      isWalkDone(lec, id) { return Boolean(data[lec] && data[lec].walk && data[lec].walk[id]); },
      setLast(lec, part) {
        if (lecture(lec).last === part) return;
        lecture(lec).last = part;
        save();
      },
      getLast(lec) { return (data[lec] && data[lec].last) || null; },
      getPref(key, fallback) {
        const p = data.prefs || {};
        return Object.prototype.hasOwnProperty.call(p, key) ? p[key] : fallback;
      },
      setPref(key, value) { data.prefs = data.prefs || {}; data.prefs[key] = value; save(); },
    };
  }

  function browserStorage() {
    try {
      return root.localStorage || null;
    } catch (e) {
      return null;
    }
  }

  root.Lecture = root.Lecture || {};
  if (typeof window !== "undefined") root.Lecture.store = createStore(browserStorage());
  if (typeof module !== "undefined" && module.exports) module.exports = { createStore, KEY };
})(typeof window !== "undefined" ? window : globalThis);
