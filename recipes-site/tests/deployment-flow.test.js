// The URL is the flow's state. A shared link is only trustworthy if the query
// string round-trips exactly, and a value the reader set back to its default must
// not be reported as a change, because the manifest would otherwise carry a flag
// the accepted recipe never had.

import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { parse } from "yaml";

// The module imports the renderer and the browser resolver at load time, so the
// globals those expect have to exist before the import resolves.
globalThis.window = { location: { pathname: "/vllm-multiarch-oci/", hash: "" } };
const { changesFor, coerceParameters, mountDeploymentFlow, payload, readState } = await import("../src/components/deployment-flow.js");

const record = {
  settings: {
    "max-num-seqs": { value: 8, source: "recipe:qwen38-flash-next-gb10-tp2" },
    "gpu-memory-utilization": { value: 0.74, source: "model:qwen38-flash-next" },
  },
  environment: {
    NCCL_NET: { value: "IB", source: "hardware:gb10-roce" },
    HF_HOME: { value: "/models", source: "recipe:qwen38-flash-next-gb10-tp2" },
  },
};

function search(state) {
  return new URL(payload(state), "https://example.test").search;
}

test("a state round-trips through the query string", () => {
  const state = {
    build: "v20261003.1",
    model: "recipe:qwen38-flash-next-gb10-tp2",
    target: "compose",
    // Values chosen for what breaks a naive encoding: an endpoint carries : * /,
    // a JSON object carries braces, quotes and commas, and one has a space.
    settings: { "max-num-seqs": "16", "default-chat-template-kwargs": '{"thinking":true}' },
    environment: { NCCL_NET: "Socket", CUDA_LAUNCH_BLOCKING: "1" },
    parameters: { hca: "rocep1s0f1,roceP2p1s0f1", topology_values: "server17 server19" },
    explain: true,
    advanced: true,
  };
  assert.deepEqual(readState(search(state)), state);
});

test("defaults are left out of the URL", () => {
  const url = new URL(
    payload({ build: "", model: "recipe:x", target: "lws", settings: {}, environment: {}, parameters: {} }),
    "https://example.test",
  );
  assert.equal(url.searchParams.get("build"), null, "the newest release is the default and is not written");
  assert.equal(url.searchParams.get("target"), null, "lws is the default target");
  assert.equal(url.searchParams.get("model"), "recipe:x");
  const read = readState(url.searchParams.toString());
  assert.equal(read.build, "", "an absent build reads back as the default, not as nightly");
  assert.deepEqual([read.settings, read.environment, read.parameters], [{}, {}, {}]);
  assert.equal(read.explain, false);
  assert.equal(read.advanced, false);
});

test("choosing nightly is recorded, because it is not the default", () => {
  // A shared link to a nightly configuration must reopen on nightly. If nightly were
  // omitted from the URL like the default, the link would silently switch the
  // reader to the newest release and a different image digest.
  const url = new URL(
    payload({ build: "nightly", model: "recipe:x", target: "lws", settings: {}, environment: {}, parameters: {} }),
    "https://example.test",
  );
  assert.equal(url.searchParams.get("build"), "nightly");
  assert.equal(readState(url.searchParams.toString()).build, "nightly");
});

test("a value set back to its default is not a change", () => {
  // The reader typed the same thing the recipe already had. Emitting `--flag
  // value` here would put a flag in the manifest that the accepted command does
  // not contain, and "Your changes" would report an edit that did not happen.
  assert.deepEqual(changesFor(record, { "max-num-seqs": "8" }, {}), []);
  assert.deepEqual(changesFor(record, {}, { NCCL_NET: "IB" }), []);
  assert.deepEqual(changesFor(record, {}, {}), []);
  // The number and its text spelling are one value, because a query string can
  // only ever carry text.
  assert.deepEqual(changesFor(record, { "gpu-memory-utilization": "0.74" }, {}), []);
});

test("a differing or new value is a change", () => {
  assert.deepEqual(changesFor(record, { "max-num-seqs": "16" }, {}), [
    { kind: "setting", name: "max-num-seqs", value: "16" },
  ]);
  assert.deepEqual(changesFor(record, {}, { OMP_NUM_THREADS: "2" }), [
    { kind: "environment", name: "OMP_NUM_THREADS", value: "2" },
  ]);
});

test("no record yields no changes instead of an error", () => {
  assert.deepEqual(changesFor(null, { "max-num-seqs": "16" }, { NCCL_NET: "Socket" }), []);
});

const recipe = {
  deployment: {
    parameters: {
      node_selector: { type: "stringMap", required: true },
      rdma_units: { type: "integer", required: true },
      namespace: { type: "string", required: true },
    },
  },
};

test("text from the URL becomes the type the renderer asks for", () => {
  // The renderer refuses a string where it needs a mapping, which is the right
  // check; the query string simply cannot carry the difference.
  const out = coerceParameters(recipe, {
    node_selector: '{"node-role.kubernetes.io/dgx":""}',
    rdma_units: "63",
    namespace: "openai",
  });
  assert.deepEqual(out, { node_selector: { "node-role.kubernetes.io/dgx": "" }, rdma_units: 63, namespace: "openai" });
});

test("a malformed selector is reported, never passed through", () => {
  assert.throws(() => coerceParameters(recipe, { node_selector: "not json" }), /node_selector must be a JSON object/);
  assert.throws(() => coerceParameters(recipe, { node_selector: "[1,2]" }), /not an array/);
  assert.throws(() => coerceParameters(recipe, { rdma_units: "many" }), /rdma_units must be an integer/);
});

// -- step 4, mounted ------------------------------------------------------------
//
// "Every selection renders a manifest" is behaviour of the component, not of the pure
// helpers above: it is `#output` that decides to render an inferred deployment, and
// `#missingFields` that decides whether an unanswered site field reads as guidance or
// as a renderer error. So the component is mounted here, against the smallest DOM that
// satisfies it, and driven from the fixtures the Pages build is gated on.
const siteRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const fixtureRoot = join(siteRoot, "tests/fixtures");
const repositoryRoot = resolve(siteRoot, "..");

function node(tag) {
  const element = {
    tagName: tag,
    children: [],
    attributes: new Map(),
    listeners: new Map(),
    dataset: {},
    className: "",
    // Assigning textContent empties a real element; the flow clears each slot that
    // way before re-rendering, so a stub that kept its children would leave the
    // previous render's panels in the tree and the test would read a stale node.
    get textContent() { return element._text ?? ""; },
    set textContent(value) { element.children = []; element._text = String(value); },
    innerHTML: "",
    hidden: false,
    open: false,
    id: "",
    type: "",
    value: "",
    name: "",
    href: "",
    required: false,
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    append(...items) { element.children.push(...items); },
    replaceChildren(...items) { element.children = [...items]; },
    remove() {},
    setAttribute(name, value) { element.attributes.set(name, String(value)); },
    removeAttribute(name) { element.attributes.delete(name); },
    getAttribute(name) { return element.attributes.get(name) ?? null; },
    addEventListener(type, handler) { element.listeners.set(type, handler); },
  };
  return element;
}

function walk(element, found = []) {
  for (const child of element.children ?? []) {
    found.push(child);
    walk(child, found);
  }
  return found;
}

function textOf(element) {
  return [element?.textContent ?? "", element?.innerHTML ?? "", ...(element?.children ?? []).map(textOf)].join("\n");
}

function recipeDocuments() {
  const root = join(repositoryRoot, "recipes");
  const out = [];
  for (const owner of readdirSync(root, { withFileTypes: true })) {
    if (!owner.isDirectory()) continue;
    const directory = join(root, owner.name);
    for (const file of readdirSync(directory)) {
      if (file.endsWith(".yaml")) out.push(parse(readFileSync(join(directory, file), "utf8")));
    }
  }
  return out;
}

async function mount() {
  const served = new Map([
    ["releases.json", JSON.parse(readFileSync(join(fixtureRoot, "releases.json"), "utf8"))],
    ["latest-image.json", JSON.parse(readFileSync(join(fixtureRoot, "latest-image.json"), "utf8"))],
    ["configs.json", JSON.parse(readFileSync(join(fixtureRoot, "configs.json"), "utf8"))],
    ["options.json", JSON.parse(readFileSync(join(fixtureRoot, "options.json"), "utf8"))],
    ["recipes.json", { recipes: recipeDocuments() }],
  ]);
  globalThis.window = {
    location: { pathname: "/vllm-multiarch-oci/", search: "", hash: "" },
    history: { replaceState() {} },
    addEventListener() {},
  };
  globalThis.document = { createElement: (tag) => node(tag), body: node("body"), execCommand() {} };
  globalThis.fetch = async (url) => {
    const body = served.get(String(url).split("/").pop());
    return { ok: body !== undefined, status: body === undefined ? 404 : 200, json: async () => body };
  };
  const root = node("div");
  root.dataset.base = "/";
  const flow = await mountDeploymentFlow(root);
  // `select()` fires the async render without awaiting it; let the microtasks drain.
  const settle = async () => {
    for (let round = 0; round < 12; round += 1) await new Promise((resolveRound) => setTimeout(resolveRound, 0));
  };
  return { flow, root, settle };
}

function runSlot(root) {
  return walk(root).find((element) => element.className === "flow-slot flow-slot-run");
}

function tab(slots, label) {
  return walk(slots).find((element) => element.className === "output-tab" && element.textContent === label);
}

function panel(slots, target) {
  return walk(slots).find((element) => element.dataset?.target === target);
}

function click(element) {
  element.listeners.get("click")?.({ currentTarget: element });
}

test("an upstream profile selection renders a manifest instead of a notice", async () => {
  const { flow, root, settle } = await mount();
  flow.select({ model: "profile:ds41-flash#gb10-roce" });
  await settle();
  const slot = runSlot(root);
  assert.ok(slot, "step 4 rendered");
  const kubernetes = tab(slot, "Kubernetes");
  assert.ok(kubernetes, "the Kubernetes tab exists for an upstream selection");
  click(kubernetes);
  await settle();
  const body = textOf(panel(slot, "lws"));
  assert.ok(!body.includes("no recipe in this repository"), "the old refusal is gone");
  assert.ok(body.includes("--profile") && body.includes("ds41-flash") && body.includes("gb10-roce"), `the manifest selects the upstream profile: ${body.slice(0, 200)}`);
  assert.ok(body.includes("--topology") && body.includes("single"), "the topology is the resolved one");
  assert.ok(body.includes("nvidia.com/gpu"), "the pod is admitted on its devices");
  assert.ok(!body.includes("cpu:"), "no unmeasured cpu request is claimed");
  assert.ok(!body.includes("memory:") && !body.includes("ephemeral-storage:"), "no unmeasured memory or storage is claimed");
  assert.ok(body.includes("not validated on this image"), "the upstream labelling survives into the output");
  assert.equal(walk(panel(slot, "lws")).filter((element) => element.className === "output-error").length, 0, "nothing errored");
});

test("an unanswered site field reads as guidance, never as the renderer's error", async () => {
  const { flow, root, settle } = await mount();
  flow.select({ model: "profile:glm53-flash#native" });
  await settle();
  const slot = runSlot(root);
  click(tab(slot, "Model routing"));
  await settle();
  const routed = panel(slot, "routing");
  assert.equal(textOf(routed).trim(), "", "the routing panel stays empty while its field is unanswered");
  const guidance = textOf(slot);
  assert.match(guidance, /Fill in /, "the panel is driven by the fill-in guidance");
  assert.match(guidance, /GatewayClass name/, "which names the field in its own label");
  // A tab whose fields are all answered still renders while that guidance stands.
  click(tab(slot, "Docker"));
  await settle();
  assert.match(textOf(panel(slot, "docker")), /vllm-image/, "the docker tab renders for the same selection");
});

test("a recipe selection keeps rendering from its own measured deployment", async () => {
  const { flow, root, settle } = await mount();
  flow.select({ model: "recipe:qwen38-27b" });
  await settle();
  const slot = runSlot(root);
  click(tab(slot, "Kubernetes"));
  await settle();
  const body = textOf(panel(slot, "lws"));
  assert.match(body, /--recipe/, "a recipe selection keeps its recipe selection");
  assert.ok(!body.includes("not validated on this image"), "the upstream note belongs to upstream selections only");
  assert.match(body, /kind: Deployment/, "the single-node shape is a Deployment");
  assert.equal(walk(panel(slot, "lws")).filter((element) => element.className === "output-error").length, 0);
});
