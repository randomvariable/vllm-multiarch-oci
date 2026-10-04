// One benchmark renderer, two hosts (the recipe page and the deployment flow), so
// these tests are what keep a page from showing a figure its recipe does not carry.
// They read the real recipe data rather than a synthetic shape: a fixture that
// happens to match a buggy reader proves nothing.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { parse } from "yaml";

import { benchmarkHtml } from "../src/benchmark.js";

const siteRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = resolve(siteRoot, "..");
const recipes = JSON.parse(readFileSync(join(siteRoot, "public", "recipes.json"), "utf8")).recipes;

test("every recipe with a benchmark renders through the shared function", () => {
  const withBenchmark = recipes.filter((recipe) => recipe.benchmark);
  assert.ok(withBenchmark.length > 0, "no recipe carries benchmark data");
  for (const recipe of withBenchmark) {
    const html = benchmarkHtml(recipe.benchmark);
    assert.ok(html.startsWith('<div class="benchmark-panel">'), `${recipe.meta.slug}: panel wrapper`);
    // Every measured cell in the data must appear in the markup, so a renderer
    // that silently drops a column cannot pass.
    for (const table of recipe.benchmark.tables ?? []) {
      for (const row of table.rows ?? []) {
        for (const cell of row.cells ?? []) {
          assert.ok(html.includes(`<b>${cell.text}</b>`), `${recipe.meta.slug}: lost cell ${cell.text}`);
        }
      }
    }
  }
});

test("a row without cells is refused rather than skipped", () => {
  assert.throws(
    () => benchmarkHtml({ tables: [{ id: "t", columns: ["a"], rows: [{ text: "a" }] }] }),
    /benchmark table t: every row needs a cells list/,
  );
});

test("cell units, notes and emphasis survive into the markup", () => {
  const html = benchmarkHtml({
    tables: [
      {
        id: "one",
        columns: ["Concurrency", "Decode"],
        rows: [{ cells: [{ text: "1" }, { text: "47.9", unit: "tok/s", primary: true, notes: ["Warm TTFT: 0.72 s"] }] }],
      },
    ],
  });
  assert.match(html, /<span role="cell" class="benchmark-primary"/);
  assert.match(html, /<b>47\.9<\/b>tok\/s/);
  assert.match(html, /<small>Warm TTFT: 0\.72 s<\/small>/);
});

test("recipe text cannot inject markup", () => {
  const html = benchmarkHtml({
    context: ["<img src=x onerror=alert(1)>"],
    tables: [{ id: "one", columns: ["a"], rows: [{ cells: [{ text: "</b><script>bad()</script>" }] }] }],
    notes: [{ title: "Method", body: "uses `--max-num-seqs` and <b>bold</b>" }],
  });
  assert.ok(!html.includes("<img"), "an injected image element reached the markup");
  assert.ok(!html.includes("<script>"), "an injected script reached the markup");
  assert.ok(html.includes("&lt;/b&gt;&lt;script&gt;"), "escaped text should still be readable");
  assert.match(html, /<code>--max-num-seqs<\/code>/, "a backticked flag should render as code");
  assert.ok(!html.includes("<b>bold</b>"), "markdown-style markup must not pass through");
});
