// The Lecture 2 glossary must cover every row of the notes' Part 12 table,
// and the lookup and search helpers must behave.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const G = require("../js/core/glossary.js");
const data = require("../js/lecture-2/glossary-data.js");

const NOTES = fileURLToPath(new URL("../../Lecture2_Notes_Positional_Encoding_to_RoBERTa.md", import.meta.url));
const EM_DASH = String.fromCharCode(0x2014);

// Rows of the markdown table under "## Part 12", as { term, full }.
function notesRows() {
  const text = readFileSync(NOTES, "utf8");
  const start = text.indexOf("## Part 12");
  assert.ok(start >= 0, "the notes have a Part 12");
  const end = text.indexOf("\n## ", start + 5);
  const section = text.slice(start, end < 0 ? undefined : end);
  return section
    .split(/\r?\n/)
    .filter((line) => line.startsWith("|"))
    .map((line) => line.split("|").slice(1, -1).map((c) => c.trim()))
    .filter((cells) => cells.length >= 3 && cells[0] !== "Abbreviation" && !/^-+$/.test(cells[0]))
    .map(([term, full]) => ({ term, full }));
}

test("the data registers on Lecture.glossary and module.exports", () => {
  assert.ok(Array.isArray(data));
  assert.equal(globalThis.Lecture.glossary, data);
});

test("every abbreviation in the notes' Part 12 table has an entry", { skip: !existsSync(NOTES) && "notes file not found" }, () => {
  const rows = notesRows();
  assert.ok(rows.length >= 50, `expected about 55 rows, parsed ${rows.length}`);
  const missing = rows.filter((r) => !data.some((d) => d.term === r.term)).map((r) => r.term);
  assert.deepEqual(missing, [], "add these terms, spelled exactly as in the notes");
  const wrongName = rows.filter((r) => {
    const d = data.find((x) => x.term === r.term);
    return d && d.full !== r.full;
  }).map((r) => r.term);
  assert.deepEqual(wrongName, [], "full names must match the notes");
  assert.equal(data.length, rows.length, "no extra entries beyond the notes");
});

test("no entry has an empty field and no term appears twice", () => {
  const seen = new Set();
  for (const d of data) {
    for (const f of ["term", "full", "meaning"]) {
      assert.equal(typeof d[f], "string", `${d.term}: ${f} is a string`);
      assert.ok(d[f].trim().length > 0, `${d.term}: ${f} is not empty`);
    }
    const k = d.term.toLowerCase();
    assert.ok(!seen.has(k), `duplicate term ${d.term}`);
    seen.add(k);
  }
});

test("no em dashes in the glossary data", () => {
  assert.ok(!JSON.stringify(data).includes(EM_DASH));
});

test("findTerm matches the term exactly, ignoring case and surrounding spaces", () => {
  assert.equal(G.findTerm(data, "GQA").full, "Grouped-Query Attention");
  assert.equal(G.findTerm(data, "gqa").term, "GQA");
  assert.equal(G.findTerm(data, "  kv CACHE ").term, "KV cache");
  assert.equal(G.findTerm(data, "[cls]").term, "[CLS]");
  assert.equal(G.findTerm(data, "L, H, A").full, "Layers, Hidden size, Attention heads");
  assert.equal(G.findTerm(data, "GQ"), null, "a prefix is not a match");
  assert.equal(G.findTerm(data, "Grouped-Query Attention"), null, "only the term is matched");
  assert.equal(G.findTerm(data, ""), null);
  assert.equal(G.findTerm(data, undefined), null);
  assert.equal(G.findTerm(null, "GQA"), null);
});

test("filterTerms searches term, full name and meaning, trimmed and case-insensitive", () => {
  const terms = (q) => G.filterTerms(data, q).map((d) => d.term);
  assert.equal(G.filterTerms(data, "").length, data.length, "empty query shows everything");
  assert.equal(G.filterTerms(data, "   ").length, data.length, "blank query shows everything");
  assert.notEqual(G.filterTerms(data, ""), data, "returns a copy, not the original array");
  assert.ok(terms("  rope ").includes("RoPE"), "matches the term, trimmed, any case");
  assert.deepEqual(terms("yarn"), terms("\tYARN\n"), "whitespace and case do not change results");
  assert.ok(terms("GROUPED-QUERY").includes("GQA"), "matches the full name");
  assert.ok(terms("distillation").includes("KL"), "matches the meaning");
  assert.deepEqual(terms("zzzz-nothing"), []);
  // Order follows the list.
  const hits = terms("bert");
  assert.deepEqual(hits, data.filter((d) => hits.includes(d.term)).map((d) => d.term));
});

test("sortTerms orders alphabetically and ignores brackets", () => {
  const sorted = G.sortTerms(data).map((d) => d.term);
  assert.equal(sorted.length, data.length);
  assert.ok(sorted.indexOf("[CLS]") > sorted.indexOf("C4") && sorted.indexOf("[CLS]") < sorted.indexOf("CNN"));
  assert.equal(sorted[0], "ALBERT");
});
