// Renders a recipe's `benchmark` block into markup.
//
// One implementation, two hosts: the recipe page embeds it server-side through
// src/components/BenchmarkPanel.astro, and the deployment flow's fourth step sets
// it as HTML in the browser. Two renderers over the same data is how a page comes
// to show a figure its recipe does not carry, which is the reason the tables were
// moved out of hand-written MDX in the first place.
//
// Every string is escaped before it reaches the document. The data comes from a
// recipe YAML in this repository, but the output is inserted with innerHTML by the
// flow, so an unescaped value would be markup.

function escapeText(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

// A note's body keeps the inline code spans the hand-written markup carried, so a
// flag named in a caveat still reads as a flag rather than as prose.
function inlineFormat(value) {
  return escapeText(value)
    .split("`")
    .map((part, index) => (index % 2 ? `<code>${part}</code>` : part))
    .join("");
}

function cellHtml(cell, index, columns) {
  const body = [
    `<b>${escapeText(cell.text)}</b>`,
    cell.unit ? escapeText(cell.unit) : "",
    ...(cell.notes ?? []).map((note) => `<small>${escapeText(note)}</small>`),
  ].join("");
  if (index === 0) return `<strong role="cell">${body}</strong>`;
  const primary = cell.primary ? ' class="benchmark-primary"' : "";
  return `<span role="cell"${primary} aria-label="${escapeText(columns[index] ?? "")}">${body}</span>`;
}

export function benchmarkHtml(benchmark) {
  if (!benchmark || typeof benchmark !== "object" || Array.isArray(benchmark)) {
    throw new TypeError("benchmark must be a mapping");
  }
  const parts = [];

  const context = benchmark.context ?? [];
  if (!Array.isArray(context)) throw new TypeError("benchmark.context must be a list");
  if (context.length) {
    parts.push(`<div class="benchmark-context">${context.map((chip) => `<span>${escapeText(chip)}</span>`).join("")}</div>`);
  }

  if (!Array.isArray(benchmark.tables)) throw new TypeError("benchmark.tables must be a list");
  for (const table of benchmark.tables) {
    const columns = table.columns ?? [];
    const head = columns.map((column) => `<span role="columnheader">${escapeText(column)}</span>`).join("");
    const rows = [];
    if (!Array.isArray(table.rows)) throw new TypeError(`benchmark table ${table.id ?? "?"}: rows must be a list`);
    for (const row of table.rows) {
      if (!Array.isArray(row?.cells)) {
        throw new TypeError(`benchmark table ${table.id ?? "?"}: every row needs a cells list`);
      }
      const degraded = row.degraded ? " benchmark-degraded" : "";
      rows.push(`<div class="benchmark-row${degraded}" role="row">${row.cells.map((cell, index) => cellHtml(cell, index, columns)).join("")}</div>`);
    }
    const label = table.aria_label ?? table.title ?? "benchmark results";
    parts.push(
      `<div class="benchmark-table" role="table" aria-label="${escapeText(label)}">` +
        `<div class="benchmark-row benchmark-header" role="row">${head}</div>${rows.join("")}</div>`,
    );
  }

  const notes = benchmark.notes ?? [];
  if (!Array.isArray(notes)) throw new TypeError("benchmark.notes must be a list");
  for (const note of notes) {
    const title = note.title ? `<strong>${escapeText(`${note.title}:`)}</strong> ` : "";
    parts.push(`<p class="benchmark-note">${title}${inlineFormat(note.body ?? "")}</p>`);
  }

  return `<div class="benchmark-panel">${parts.join("")}</div>`;
}
