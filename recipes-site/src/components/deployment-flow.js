// The deployment flow: choose a build, a model, its settings, then run it.
//
// Every value the page shows comes from a generated document, so the flow cannot
// disagree with the image: releases.json from the published GitHub releases,
// configs.json from the launcher's own resolver, options.json from the pinned
// upstream parameter documentation plus this repository's recipe docs, and
// recipes.json from recipes/. The URL carries the whole selection, which makes any
// configuration shareable and a reload deterministic.
//
// Manifest rendering is shared with the test lane: the page calls the same
// renderRecipe() the Node tests exercise, and the vLLM command tab runs the same
// resolver.py the container runs, in the browser through the self-hosted Pyodide
// tree. Nothing here reimplements a rule that already lives in the launcher.

import { benchmarkHtml } from "../benchmark.js";
import { profileRecipe, renderRecipe, requiredSiteFields } from "../render.js";
import { SITE_PARAMETERS } from "../data/site-parameters.js";
import { resolveInBrowser } from "./browser-resolver.js";

const NIGHTLY = "nightly";
const OWNED_SOURCE = /^(recipe|preset|cli):/;
// Sources whose values are a deliberate choice for this model, and so belong on the
// form rather than only in the full table.
const KEY_SOURCE = /^(recipe|preset|cli|model):/;

// One spelling for a value everywhere it is shown or compared. Objects such as
// limit-mm-per-prompt are JSON, which is also how the launcher accepts them, so an
// unedited field compares equal to its default instead of to "[object Object]".
export function displayValue(value) {
  if (value === undefined || value === null) return "";
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}
const TARGETS = [
  { id: "lws", label: "Kubernetes", language: "yaml" },
  { id: "docker", label: "Docker", language: "bash" },
  { id: "compose", label: "Compose", language: "yaml" },
  { id: "routing", label: "Model routing", language: "yaml" },
  { id: "vllm", label: "vLLM command", language: "bash" },
];
const COPY_LABEL = "Copy";

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function headRow(columns) {
  // append() returns undefined, so a header cannot be built by chaining it onto
  // the element that holds it.
  const row = element("tr");
  for (const label of columns) row.append(element("th", null, label));
  const head = element("thead");
  head.append(row);
  return head;
}


function button(className, label, onClick) {
  const node = element("button", className, label);
  node.type = "button";
  node.addEventListener("click", onClick);
  return node;
}

export function payload(state) {
  const query = new URLSearchParams();
  query.set("model", state.model);
  // An empty build means "the default", which is the newest release; nightly is a
  // deliberate choice and so is recorded in the URL like any other build.
  if (state.build) query.set("build", state.build);
  if (state.target && state.target !== "lws") query.set("target", state.target);
  for (const [key, item] of Object.entries(state.settings ?? {})) query.set(`x.${key}`, item);
  for (const [key, item] of Object.entries(state.environment ?? {})) query.set(`e.${key}`, item);
  for (const [key, item] of Object.entries(state.parameters ?? {})) query.set(`s.${key}`, item);
  if (state.explain) query.set("explain", "all");
  if (state.advanced) query.set("advanced", "1");
  return `${window.location.pathname}?${query.toString()}`;
}

export function readState(search) {
  const query = new URLSearchParams(search);
  const state = {
    build: query.get("build") || "",
    model: query.get("model") || "",
    target: query.get("target") || "lws",
    settings: Object.fromEntries([...query].filter(([k]) => k.startsWith("x.")).map(([k, v]) => [k.slice(2), v])),
    environment: Object.fromEntries([...query].filter(([k]) => k.startsWith("e.")).map(([k, v]) => [k.slice(2), v])),
    parameters: Object.fromEntries([...query].filter(([k]) => k.startsWith("s.")).map(([k, v]) => [k.slice(2), v])),
    explain: query.get("explain") === "all",
    advanced: query.get("advanced") === "1",
  };
  return state;
}

export function changesFor(record, settings = {}, environment = {}) {
  // A value equal to the resolved default is not a change: the launcher would
  // append it, the manifest would carry a flag the accepted recipe never had, and
  // "Your changes" would report an edit that did not happen.
  if (!record) return [];
  const listed = [];
  for (const [name, value] of Object.entries(settings)) {
    const entry = record.settings?.[name];
    if (!entry || displayValue(entry.value) !== String(value)) listed.push({ kind: "setting", name, value });
  }
  for (const [name, value] of Object.entries(environment)) {
    const entry = record.environment?.[name];
    if (!entry || displayValue(entry.value) !== String(value)) listed.push({ kind: "environment", name, value });
  }
  return listed;
}

async function loadDocument(base, name) {
  const response = await fetch(`${base}${name}`, { cache: "no-cache" });
  if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
  return response.json();
}

async function copy(text, trigger) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = document.createElement("textarea");
    area.value = text;
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  trigger.textContent = "Copied";
  setTimeout(() => {
    trigger.textContent = COPY_LABEL;
  }, 1500);
}

function download(name, text) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
  link.download = name;
  link.click();
  URL.revokeObjectURL(link.href);
}

export function coerceParameters(recipe, supplied = {}) {
  // A query string can only carry text, so a reader's node selector arrives as
  // JSON text and an integer arrives as digits. The renderer validates types and
  // refuses a string where it needs a mapping, which is the right check to keep --
  // so the conversion happens here, once, against the recipe's own parameter
  // definitions, and a malformed object is reported instead of passed through.
  const definitions = recipe?.deployment?.parameters ?? {};
  const out = {};
  for (const [name, value] of Object.entries(supplied)) {
    const definition = definitions[name];
    if (!definition || value === null || value === undefined || typeof value !== "string") {
      out[name] = value;
      continue;
    }
    if (definition.type === "stringMap") {
      let parsed;
      try {
        parsed = JSON.parse(value);
      } catch (error) {
        throw new TypeError(`${name} must be a JSON object (${error.message})`);
      }
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new TypeError(`${name} must be a JSON object, not ${Array.isArray(parsed) ? "an array" : typeof parsed}`);
      }
      out[name] = parsed;
    } else if (definition.type === "integer") {
      const number = Number(value);
      if (!Number.isSafeInteger(number)) throw new TypeError(`${name} must be an integer, got ${value}`);
      out[name] = number;
    } else {
      out[name] = value;
    }
  }
  return out;
}

export function browserSelection(recipe, record) {
  const selection = record?.selection ?? {};
  const base = { profile: selection.profile, hardware: selection.hardware, preset: selection.preset ?? null };
  if (!recipe) return base;
  return {
    ...base,
    recipe: recipe.meta.slug,
    options: recipe.launch?.options ?? {},
    environment: recipe.launch?.environment ?? {},
  };
}

class DeploymentFlow {
  constructor(root) {
    this.root = root;
    this.base = root.dataset.base ?? "/";
    this.state = readState(window.location.search);
    this.tabs = new Map();
    // The inferred deployment for the current selection, memoised by model: the site
    // fields, the fill-in list and every manifest tab read this one object.
    this.viewed = null;
    this.viewedFor = null;
    this.viewError = null;
  }

  async start() {
    this.#shell();
    try {
      const [releases, latest, recipes, configs, options] = await Promise.all(
        ["releases.json", "latest-image.json", "recipes.json", "configs.json", "options.json"].map(
          (name) => loadDocument(this.base, name),
        ),
      );
      Object.assign(this, { releases: releases.releases, latest, recipes: recipes.recipes, configs, options });
      // Every URL the page writes is built from this.state, so the model that was
      // picked by default has to live there too, or links come out as `model=`.
      this.model = this.state.model = this.#resolveModel();
      this.#render();
    } catch (error) {
      this.#failure(error);
    }
    window.addEventListener("popstate", async () => {
      this.state = readState(window.location.search);
      this.model = this.state.model = this.#resolveModel();
      await this.#render();
    });
  }

  #shell() {
    this.root.textContent = "";
    const steps = element("ol", "flow-steps");
    this.slots = {};
    for (const [id, title, summary] of [
      ["build", "Choose a build", "Release publishes a tag. Nightly tracks the newest build the pipeline published."],
      ["model", "Choose a model", "Recipes are validated on this cluster. Upstream profiles run on the same image but are not validated here."],
      ["configure", "Configure it", "Every value carries a reason. Changing one moves it into your changes."],
      ["run", "Run it", "The image is pinned by digest, so what you copy is the build this page describes."],
    ]) {
      const step = element("li", `flow-step flow-step-${id}`);
      const heading = element("h3", "flow-step-title", title);
      heading.id = `flow-${id}`;
      step.append(heading, element("p", "flow-step-summary", summary));
      const slot = element("div", `flow-slot flow-slot-${id}`);
      step.append(slot);
      this.slots[id] = slot;
      steps.append(step);
    }
    this.root.append(steps);
  }

  #failure(error) {
    const notice = element("p", "flow-error", `The deployment data did not load: ${error.message}`);
    notice.append(
      element("span", "flow-error-hint", "The flow reads generated documents; run the site data build before previewing."),
    );
    this.slots.build.replaceChildren(notice);
  }

  #resolveModel() {
    const known = (selection) => Object.hasOwn(this.configs, selection);
    if (this.state.model && known(this.state.model)) return this.state.model;
    const validated = this.recipes.map((recipe) => `recipe:${recipe.meta.slug}`);
    return validated.find(known) ?? Object.keys(this.configs).find(known) ?? "";
  }

  record() {
    return this.configs[this.model] ?? null;
  }

  recipe() {
    if (!this.model.startsWith("recipe:")) return null;
    const slug = this.model.slice("recipe:".length);
    return this.recipes.find((candidate) => candidate.meta.slug === slug) ?? null;
  }

  // The one deployment object every manifest surface reads: the shape the image
  // resolved for this selection, with the recipe's measured fields layered over it
  // when the selection is a recipe. The site fields, the fill-in guidance and the
  // renderer all read this same object, so a selection cannot render a manifest with
  // one set of parameters and show the operator a different one. An un-inferable
  // record is kept as an error to display, never as a silent blank.
  view() {
    const record = this.record();
    if (!record) return null;
    if (this.viewedFor !== this.model) {
      this.viewed = null;
      this.viewError = null;
      try {
        this.viewed = profileRecipe(record, SITE_PARAMETERS, this.recipe());
      } catch (error) {
        this.viewError = error;
      }
      this.viewedFor = this.model;
    }
    return this.viewed;
  }

  // The default is the newest tagged release: it has a changelog, a build page and
  // was cut deliberately, while nightly is whatever the pipeline last pushed.
  // Nightly is only the fallback when nothing has been released.
  effectiveBuild() {
    return this.state.build || this.releases[0]?.tag || NIGHTLY;
  }

  build() {
    const chosen = this.effectiveBuild();
    if (chosen !== NIGHTLY) {
      const release = this.releases.find((candidate) => candidate.tag === chosen);
      if (release) return { kind: "release", release };
    }
    return {
      kind: "nightly",
      release: {
        tag: this.latest.tag,
        digest: this.latest.reference.split("@").pop(),
        published_at: this.latest.resolved_at,
        highlights: [],
        url: null,
        vllm_commit: null,
        b12x_commit: null,
        lil_runtime_commit: null,
        builder_commit: null,
        nightlyOnly: true,
      },
    };
  }

  changes() {
    return changesFor(this.record(), this.state.settings, this.state.environment);
  }

  select(patch) {
    Object.assign(this.state, patch);
    if (patch.model) this.model = this.state.model = this.#resolveModel();
    window.history.replaceState({}, "", payload(this.state));
    this.#render();
  }

  async #render() {
    const container = this.root;
    container.classList.toggle("flow-unset", !this.model);
    this.#renderBuilds();
    this.#renderModels();
    this.#renderConfigure();
    await this.#renderRun();
  }

  // -- step 1 --------------------------------------------------------------

  #renderBuilds() {
    const slot = this.slots.build;
    slot.textContent = "";
    const channels = element("div", "choice-grid choice-grid-channels");
    const nightly = button("choice-card", "Nightly", () => this.select({ build: NIGHTLY }));
    nightly.setAttribute("aria-pressed", String(this.effectiveBuild() === NIGHTLY));
    nightly.append(element("p", "choice-note", "The newest build the pipeline published, refreshed hourly."));
    const release = button("choice-card", "Release", () => this.select({ build: "" }));
    release.setAttribute("aria-pressed", String(this.effectiveBuild() !== NIGHTLY));
    release.append(element("p", "choice-note", "A tag you cut deliberately, with its changelog and its own page."));
    channels.append(nightly, release);
    slot.append(channels);

    const chosen = this.build().release;
    const facts = element("div", "build-facts");
    const heading = element("p", "build-fact-title");
    heading.append(
      element("strong", null, chosen.tag),
      element("span", "build-fact-published", `published ${String(chosen.published_at).slice(0, 10)}`),
    );
    facts.append(heading);
    const digest = element("p", "build-digest");
    digest.append(element("span", "build-label", "Digest"), element("code", null, chosen.digest));
    facts.append(digest);
    const driver = element("p", "build-pins");
    for (const [label, value] of [
      ["vLLM", chosen.vllm_commit],
      ["B12X", chosen.b12x_commit],
      ["runtime data", chosen.lil_runtime_commit],
      ["builder", chosen.builder_commit],
    ]) {
      const item = element("span", "build-pin");
      item.append(element("span", "build-label", label), element("code", null, value ? value.slice(0, 12) : "nightly"));
      driver.append(item);
    }
    facts.append(driver);
    if (chosen.highlights.length) {
      const list = element("ul", "build-highlights");
      for (const entry of chosen.highlights.slice(0, 4)) list.append(element("li", null, entry));
      facts.append(list);
    }
    if (chosen.url) {
      const link = element("a", "build-changelog", "Full changelog");
      link.href = chosen.url;
      facts.append(link);
    }
    slot.append(facts);

    if (this.releases.length > 1) {
      const details = element("details", "build-history");
      const summary = element("summary", null, `${this.releases.length - 1} earlier builds`);
      const table = element("table", "build-history-table");
      const body = element("tbody");
      table.append(headRow(["Published", "Build", "Carries", "Use"]), body);
      for (const entry of this.releases) {
        const row = element("tr");
        row.append(element("td", null, String(entry.published_at).slice(0, 10)));
        const tag = element("td");
        tag.append(element("code", null, entry.tag));
        row.append(tag);
        const note = element("td", null, entry.highlights[0] ?? "");
        if (entry.highlights.length > 1) note.append(element("span", "build-more", ` +${entry.highlights.length - 1}`));
        row.append(note);
        const use = element("td");
        use.append(button("link-button", this.effectiveBuild() === entry.tag ? "Selected" : "Use", () => this.select({ build: entry.tag })));
        row.append(use);
        body.append(row);
      }
      details.append(summary, table);
      slot.append(details);
    }
  }

  // -- step 2 --------------------------------------------------------------

  #renderModels() {
    const slot = this.slots.model;
    slot.textContent = "";
    const groups = [
      ["Validated here", this.recipes.map((recipe) => ({ selection: `recipe:${recipe.meta.slug}`, recipe }))],
      [
        "Upstream, not validated on this image",
        Object.keys(this.configs)
          .filter((selection) => selection.startsWith("profile:"))
          .map((selection) => ({ selection, recipe: null })),
      ],
    ];
    for (const [title, entries] of groups) {
      if (!entries.length) continue;
      // 23 upstream selections would bury the three validated recipes, so they sit
      // behind a disclosure and open only when one of them is selected.
      const upstream = entries[0].recipe === null;
      const section = element(upstream ? "details" : "section", "choice-group");
      if (upstream) {
        section.open = this.model.startsWith("profile:");
        section.append(element("summary", "choice-group-title", `${title} (${entries.length})`));
      } else {
        section.append(element("h4", "choice-group-title", title));
      }
      const grid = element("div", upstream ? "choice-grid choice-grid-compact" : "choice-grid");
      for (const { selection, recipe } of entries) {
        const record = this.configs[selection];
        const label = recipe
          ? recipe.meta.title
          : [
              selection.slice("profile:".length).split("#")[0],
              record?.selection?.hardware,
              record?.selection?.preset,
            ]
              .filter(Boolean)
              .join(" · ");
        const card = button("choice-card", label, () => this.select({ model: selection }));
        card.setAttribute("aria-pressed", String(selection === this.model));
        if (selection !== this.model && recipe === null) card.classList.add("choice-card-secondary");
        const meta = element("p", "choice-note");
        // TP is the total width of the group, so it is stated as such beside the
        // node count rather than as a bare GPU figure a reader could read per node.
        meta.textContent = record
          ? `${record.topology.nodes} node${record.topology.nodes === 1 ? "" : "s"}, TP=${record.settings["tensor-parallel-size"]?.value ?? "?"}`
          : "this selection did not resolve";
        card.append(meta);
        if (recipe) {
          const link = element("a", "choice-link", "Why these settings");
          link.href = `${this.base}recipes/${recipe.meta.slug}/`;
          card.append(link);
        }
        grid.append(card);
      }
      section.append(grid);
      slot.append(section);
    }
  }

  // -- step 3 --------------------------------------------------------------

  #documented(name, kind) {
    const docs = kind === "environment" ? this.options.parameter_docs.environment : this.options.parameter_docs.options;
    const fromRecipe = this.recipe() ? this.options.recipe_docs[this.recipe().meta.slug]?.[name] : null;
    const upstream = docs?.[name];
    if (fromRecipe && upstream) return { ...upstream, ...fromRecipe };
    return fromRecipe ?? upstream ?? null;
  }

  #control(name, entry, kind, { primary }) {
    const documented = this.#documented(name, kind) ?? {};
    const row = element("div", primary ? "control" : "control control-secondary");
    const label = element("label", "control-label");
    const input = document.createElement("input");
    input.className = "control-input";
    input.type = entry?.value !== undefined && Number.isInteger(entry.value) ? "number" : "text";
    input.name = `${kind === "environment" ? "e" : "x"}.${name}`;
    input.value = this.state[kind][name] ?? displayValue(entry?.value);
    const prefix = `${name} `;
    label.append(element("span", "control-name", name));
    if (documented.summary) {
      const hint = element("span", "control-help", "?");
      hint.title = documented.summary;
      hint.setAttribute("role", "note");
      hint.setAttribute("aria-label", documented.summary);
      label.append(hint);
    }
    row.append(label, input);
    if (documented.why) {
      const why = element("details", "control-why");
      why.append(element("summary", null, "Why this value"));
      const body = element("p", null, documented.why);
      if (entry) body.append(element("span", "control-default", `default ${String(entry.value)}`));
      why.append(body);
      if (this.state.explain) why.open = true;
      row.append(why);
    }
    input.addEventListener("change", () => {
      const value = input.value;
      const store = this.state[kind];
      const unchanged = entry && displayValue(entry.value) === value;
      if (value === "" || unchanged) delete store[name];
      else store[name] = value;
      window.history.replaceState({}, "", payload(this.state));
      this.#renderRun();
    });
    return row;
  }

  #renderConfigure() {
    const slot = this.slots.configure;
    slot.textContent = "";
    const record = this.record();
    if (!record) return;
    const recipe = this.recipe();
    const summary = element("p", "configure-summary");
    summary.append(
      element("span", null, record.selection.profile ?? this.model),
      element("span", "configure-hardware", `hardware ${record.selection.hardware}`),
    );
    if (record.selection.preset) summary.append(element("span", "configure-preset", `preset ${record.selection.preset}`));
    slot.append(summary);

    const controls = element("div", "control-groups");
    const groups = new Map();
    const consider = (kind, name, entry) => {
      const source = entry?.source ?? "";
      const documented = this.#documented(name, kind);
      // The form shows the engine options this deployment chose: the recipe's, the
      // preset's or the model profile's. Common defaults, hardware tuning, derived
      // values and environment variables are still editable, in the full table
      // below; putting all 130 on the form is what made it unreadable.
      if (kind !== "settings" || !KEY_SOURCE.test(source)) return;
      const id = documented?.group ?? "other";
      if (!groups.has(id)) groups.set(id, []);
      groups.get(id).push(this.#control(name, entry, kind, { primary: OWNED_SOURCE.test(source) }));
    };
    for (const [name, entry] of Object.entries(record.settings)) consider("settings", name, entry);
    for (const [name, entry] of Object.entries(record.environment)) consider("environment", name, entry);

    const titles = this.options.parameter_docs.groups ?? {};
    const order = Object.keys(titles);
    const rank = (id) => {
      const index = order.indexOf(id);
      return index === -1 ? order.length : index;
    };
    // The comparator reads the tuple it was handed. `id` is the loop binding below,
    // which does not exist yet while sort() runs.
    for (const [id, rows] of [...groups.entries()].sort((left, right) => rank(left[0]) - rank(right[0]))) {
      const group = element("section", "control-group");
      group.append(element("h5", "control-group-title", titles[id]?.title ?? id.replace(/-/g, " ")));
      group.append(...rows);
      controls.append(group);
    }
    slot.append(controls);

    const toggles = element("div", "control-toggles");
    const explain = button("link-button", this.state.explain ? "Hide all explanations" : "Show all explanations", () => {
      this.state.explain = !this.state.explain;
      window.history.replaceState({}, "", payload(this.state));
      this.#renderConfigure();
    });
    toggles.append(explain);
    slot.append(toggles);

    const advanced = element("details", "control-advanced");
    advanced.append(element("summary", null, `Advanced: every parameter (${Object.keys(record.settings).length + Object.keys(record.environment).length})`));
    if (this.state.advanced) advanced.open = true;
    const table = element("table", "advanced-table");
    const body = element("tbody");
    table.append(headRow(["Parameter", "Value", "Source"]), body);
    const rows = [
      ...Object.entries(record.settings).map(([name, entry]) => ({ name, entry, kind: "settings" })),
      ...Object.entries(record.environment).map(([name, entry]) => ({ name, entry, kind: "environment" })),
    ].sort((left, right) => left.name.localeCompare(right.name));
    for (const { name, entry, kind } of rows) {
      const row = element("tr");
      row.append(element("td", "advanced-name", name));
      const cell = element("td");
      const input = document.createElement("input");
      input.className = "advanced-input";
      input.value = this.state[kind][name] ?? displayValue(entry.value);
      input.addEventListener("change", () => {
        const store = this.state[kind];
        if (input.value === "" || displayValue(entry.value) === input.value) delete store[name];
        else store[name] = input.value;
        window.history.replaceState({}, "", payload(this.state));
        this.#renderRun();
      });
      cell.append(input);
      row.append(cell);
      const source = element("td");
      source.append(element("span", `source-badge source-${String(entry.source).split(":")[0]}`, entry.source));
      row.append(source);
      body.append(row);
    }
    table.append(body);
    advanced.append(table);
    slot.append(advanced);
  }

  // -- step 4 --------------------------------------------------------------

  async #renderRun() {
    const slot = this.slots.run;
    slot.textContent = "";
    const record = this.record();
    if (!record) return;
    const changes = this.changes();
    const missing = this.#missingFields();

    const tabs = element("div", "output-tabs");
    tabs.setAttribute("role", "tablist");
    const panels = element("div", "output-panels");
    const bar = element("div", "output-actions");
    for (const target of TARGETS) {
      const tab = button("output-tab", target.label, () => show(target.id));
      tab.setAttribute("role", "tab");
      tabs.append(tab);
      const panel = element("div", "output-panel");
      panel.dataset.target = target.id;
      panels.append(panel);
      this.tabs.set(target.id, { tab, panel, target });
    }

    const show = async (id) => {
      for (const [name, entry] of this.tabs) entry.tab.setAttribute("aria-selected", String(name === id));
      for (const [name, entry] of this.tabs) entry.panel.hidden = name !== id;
      const entry = this.tabs.get(id);
      if (!entry.rendered) {
        entry.panel.replaceChildren(element("p", "output-pending", "Rendering…"));
        const result = await this.#output(id);
        entry.rendered = result;
        entry.panel.replaceChildren(...result.nodes);
      }
      bar.replaceChildren();
      const text = entry.rendered.text ?? "";
      if (text && !missing.length) {
        bar.append(
          button("output-copy", COPY_LABEL, (event) => copy(text, event.currentTarget)),
          button("output-download", `Download ${entry.rendered.file}`, () => download(entry.rendered.file, text)),
        );
      }
      const share = button("output-share", "Copy link", () => copy(payload({ ...this.state, target: id }), share));
      bar.append(share);
    };

    // Cluster-specific inputs are many and mostly prefilled, so they are folded away
    // unless one of them is required and still empty -- which is exactly when the
    // reader has to look at them before anything can be copied.
    const site = element("details", "output-site");
    site.open = missing.length > 0;
    site.append(
      element(
        "summary",
        "output-site-title",
        missing.length
          ? `Your cluster: ${missing.length} required field${missing.length === 1 ? "" : "s"} to fill in`
          : "Your cluster: the namespace, name, host paths and node placement this deployment asks for",
      ),
      this.#siteFields(),
    );
    slot.append(site);

    const listed = element("div", "output-changes");
    const count = element("p", "output-changes-title", `Your changes (${changes.length})`);
    listed.append(count);
    if (changes.length) {
      const list = element("ul", "output-changes-list");
      for (const change of changes) list.append(element("li", null, `${change.name} = ${change.value}`));
      listed.append(list);
      listed.append(
        element(
          "p",
          "output-changes-warning",
          "Edited values leave the configuration this recipe was measured with. The manifest is still valid; the numbers are not.",
        ),
      );
    } else {
      listed.append(element("p", "output-changes-none", "Every value is the accepted default for this selection."));
    }

    slot.append(listed);
    if (this.recipe()?.benchmark) {
      const panel = element("details", "flow-benchmark");
      panel.open = true;
      panel.append(element("summary", null, "Measured on this hardware"));
      // The same function the recipe page uses, so the configurator and the prose
      // cannot disagree about what was measured.
      const table = element("div", "flow-benchmark-table");
      table.innerHTML = benchmarkHtml(this.recipe().benchmark);
      panel.append(table);
      const method = element("p", "benchmark-method");
      method.append(
        element("span", null, "These numbers came from the workload described in "),
        Object.assign(element("a", null, "how to benchmark a deployment"), {
          href: `${this.base}guides/benchmark-deployment/`,
        }),
        element("span", null, ". A configuration you edited here has no measurement until you take one."),
      );
      panel.append(method);
      slot.append(panel);
    }
    slot.append(tabs, bar, panels);
    if (missing.length) {
      slot.append(
        element(
          "p",
          "output-blocked",
          `Fill in ${missing.map((field) => field.label).join(", ")} to copy or download. These are yours to supply, not the recipe's.`,
        ),
      );
    }
    await show(TARGETS.some((target) => target.id === this.state.target) ? this.state.target : "lws");
  }

  #siteFields() {
    const wrapper = element("div", "output-site-fields");
    const parameters = this.view()?.deployment?.parameters ?? {};
    for (const [name, definition] of Object.entries(parameters)) {
      const row = element("label", "site-field");
      row.append(element("span", "site-field-label", definition.label ?? name));
      const input = document.createElement("input");
      input.type = definition.type === "integer" ? "number" : "text";
      const supplied = this.state.parameters[name];
      // A stringMap is an object, so it is presented and accepted as JSON. Putting
      // the object straight into input.value would render "[object Object]" and
      // send that back as the reader's answer.
      const text = (value) =>
        value === undefined || value === null ? "" : definition.type === "stringMap" ? JSON.stringify(value) : String(value);
      input.value = supplied !== undefined ? supplied : text(definition.default);
      if (definition.type === "stringMap") input.setAttribute("aria-describedby", `${name}-format`);
      if (definition.required) input.required = true;
      input.addEventListener("change", () => {
        if (input.value === "") {
          delete this.state.parameters[name];
          input.removeAttribute("aria-invalid");
        } else if (definition.type === "stringMap") {
          try {
            const parsed = JSON.parse(input.value);
            if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("not an object");
            this.state.parameters[name] = parsed;
            input.removeAttribute("aria-invalid");
          } catch (error) {
            // Refuse rather than send a half-parsed selector to the renderer: a
            // wrong node selector schedules the pod onto the wrong machines.
            input.setAttribute("aria-invalid", "true");
            note.textContent = `Not JSON (${error.message}). Expected an object such as {"node-role.kubernetes.io/dgx": ""}.`;
            return;
          }
        } else {
          this.state.parameters[name] = input.value;
        }
        window.history.replaceState({}, "", payload(this.state));
        this.#renderRun();
      });
      row.append(input);
      const note = element("span", "site-field-help");
      note.textContent = definition.description ?? "";
      if (definition.type === "stringMap") note.id = `${name}-format`;
      row.append(note);
      wrapper.append(row);
    }
    return wrapper;
  }

  #missingFields() {
    // The renderer's own rule about what it dereferences for this deployment's
    // shape, so a field the output never reads is never demanded on the form: a
    // single-node pod is asked for no HCA, no GID index, no network attachment, no
    // RDMA resource and no topology label.
    const view = this.view();
    if (!view) return [];
    return requiredSiteFields(view, this.state.parameters);
  }

  async #output(target) {
    const record = this.record();
    const image = this.build().release.nightlyOnly
      ? this.latest.reference
      : `${this.latest.repository}@${this.build().release.digest}`;
    const nodes = [];
    if (target === "vllm") {
      const changes = this.changes();
      let argv = record.argv;
      let note = null;
      if (changes.length) {
        try {
          argv = await resolveInBrowser({
            base: this.base,
            // The browser tree carries the resolver and the policy data, not the
            // recipe files or the entry point, so the recipe's own layer has to come
            // from the record the site already publishes. Passing the bare selection
            // string would fall back to the unedited command for every recipe.
            selection: browserSelection(this.recipe(), record),
            settings: this.state.settings,
            environment: this.state.environment,
            context: record.resolution_context,
          });
        } catch (error) {
          note = `Showing the unedited command: the in-browser resolver could not run (${error.message}). Run the printed command on a host with the image for the exact result.`;
        }
      }
      const text = argv.map((part) => (/^[A-Za-z0-9_.:/-]+$/.test(part) ? part : `'${part.replaceAll("'", "'\"'\"'")}'`)).join(" ");
      nodes.push(element("pre", "output-code", text));
      if (note) nodes.push(element("p", "output-note", note));
      return { text, file: "vllm.txt", nodes };
    }
    const view = this.view();
    if (!view) {
      // The record itself could not be turned into a deployment shape; say which
      // part of it, rather than printing the raw command as if it were a manifest.
      const text = this.viewError?.message ?? "This selection did not resolve, so there is nothing to render.";
      nodes.push(element("p", "output-error", text));
      return { text: "", file: `${target}.txt`, nodes };
    }
    try {
      const rendered = renderRecipe(view, {
        parameters: coerceParameters(view, this.state.parameters),
        settings: this.state.settings,
        environment: this.state.environment,
        target,
        image,
      });
      const parts = rendered.files ?? [rendered];
      const text = parts.map((part) => part.body).join("\n");
      nodes.push(element("pre", "output-code", text));
      for (const step of rendered.steps ?? []) nodes.push(element("p", "output-step", step));
      if (!this.recipe()) {
        // Step 2 labels an upstream selection as unvalidated; the manifest it now
        // produces has to carry that label too, including the one thing the renderer
        // deliberately left out.
        nodes.push(
          element(
            "p",
            "output-note",
            "Upstream profile, not validated on this image. The shape comes from the configuration the image resolved: the launcher selection, the device count and the node paths only. No CPU, memory or storage request or limit is claimed, because none has been measured for this hardware -- the vLLM command tab carries the full resolved engine configuration.",
          ),
        );
      }
      return { text, file: parts[0]?.name ?? `${target}.txt`, nodes };
    } catch (error) {
      // A site field this target needs and the reader has not filled in is guidance,
      // not a fault: the panel stays empty and the "Fill in ..." line names it in
      // its own label. Anything else is a real defect and is shown as one.
      const field = /^([\w-]+) is required for(?: the [\w-]+ target)?/.exec(String(error?.message ?? ""));
      if (field && this.#missingFields().some((missing) => missing.name === field[1])) {
        return { text: "", file: `${target}.txt`, nodes: [] };
      }
      const text = `${error.message}`;
      nodes.push(element("p", "output-error", text));
      return { text: "", file: `${target}.txt`, nodes };
    }
  }
}

export async function mountDeploymentFlow(root) {
  const flow = new DeploymentFlow(root);
  await flow.start();
  return flow;
}
