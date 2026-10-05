import assert from "node:assert/strict";
import test from "node:test";
import { resolveInBrowser } from "../src/components/browser-resolver.js";

// The browser resolver boots the image's own resolver under Pyodide. These tests run it
// in Node with an injected `fetch` stub and a fake Pyodide object, so the three things
// that must hold are proven without a browser: every file is verified against
// `manifest.json` before anything runs (a tampered byte rejects), a clean tree returns
// the resolver's argv, and a tree that was never published rejects naming the artefact
// the caller needs for its fallback.

const BASE = "/vllm-multiarch-oci/";
const ROOT = `${BASE}pyodide/`;

async function sha256Hex(text) {
  const bytes = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

// The file set `prepare-pyodide.mjs` publishes: the loader, its lock, the one wheel, the
// launcher modules the resolver imports, and the policy data it reads.
const REQUIRED_PATHS = [
  "pyodide.js",
  "pyodide-lock.json",
  "pyyaml-6.0.3-cp312-cp312-emscripten_20_19_wasm32.whl",
  "image_tools/__init__.py",
  "image_tools/launcher/__init__.py",
  "image_tools/launcher/resolver.py",
  "image_tools/launcher/probes.py",
  "image_tools/launcher/cache_runtime.py",
  "options.yaml",
  "parameter-docs.yaml",
  "schema.json",
  "platform-environment.json",
  "presets.yaml",
  "profiles/qwen38-flash-next.yaml",
  "hardware/gb10-roce.yaml",
];

// `contents` is a map of path -> bytes; the manifest's digests are the SHA-256 of those
// bytes, so a clean run verifies. `tamper` overrides the bytes returned for one path
// (while the manifest still records the original digest), which is the mismatch case.
async function buildTree({ contents = {}, tamper = null, manifest = true } = {}) {
  const bodies = new Map();
  for (const path of REQUIRED_PATHS) bodies.set(path, contents[path] ?? `// ${path} contents\n`);
  if (manifest === false) {
    return { fetchImpl: async (url) => ({ ok: false, status: 404 }), expected: {}, bodies };
  }
  const files = [];
  for (const [path, bytes] of bodies) files.push({ path, bytes: bytes.length, sha256: await sha256Hex(bytes), source: "test" });
  const tree = {
    schema_version: 1,
    pyodide: { version: "314.0.7" },
    packages: [{ name: "pyyaml", version: "6.0.3", file: "pyyaml-6.0.3-cp312-cp312-emscripten_20_19_wasm32.whl", sha256: files.find((f) => f.path.endsWith(".whl")).sha256 }],
    files,
  };
  const manifestText = JSON.stringify(tree);
  const digests = Object.fromEntries(files.map((entry) => [entry.path, entry.sha256]));
  const fetchImpl = async (url) => {
    if (url === `${ROOT}manifest.json`) return { ok: true, arrayBuffer: async () => new TextEncoder().encode(manifestText).buffer };
    const path = url.slice(ROOT.length);
    if (!bodies.has(path)) return { ok: false, status: 404 };
    const served = tamper && tamper.path === path ? tamper.bytes : bodies.get(path);
    return { ok: true, arrayBuffer: async () => new TextEncoder().encode(served).buffer };
  };
  return { fetchImpl, digests, tree };
}

function fakePyodide(argv) {
  const state = { written: [], loads: [], runs: [], globals: {}, indexUrls: [] };
  return {
    state,
    loadPyodide: async ({ indexUrl }) => {
      state.indexUrls.push(indexUrl);
      return {
        FS: {
          mkdirTree() {},
          analyzePath: () => ({ exists: true }),
          mkdir() {},
          writeFile: (path) => state.written.push(path),
        },
        loadPackage: async (name) => state.loads.push(name),
        globals: { set: (key, value) => { state.globals[key] = value; } },
        toPy: (value) => value,
        runPython: async (code) => { state.runs.push(code); return argv; },
      };
    },
  };
}

test("a clean pyodide tree returns the resolver's argv", async () => {
  const argv = ["/opt/venv/bin/python", "-m", "vllm.entrypoints.cli.main", "serve", "model", "--port", "8888"];
  const { fetchImpl } = await buildTree();
  const fake = fakePyodide(argv);
  const result = await resolveInBrowser({
    base: BASE,
    selection: "profile:qwen38-flash-next#gb10-roce",
    // What the published container answered for itself. A browser cannot import
    // vLLM, so these have to come from the record or the edited command is resolved
    // against the wrong machine.
    context: { source: "image", vllm_environment: ["VLLM_USE_V1"], b12x_mxfp8_moe: true, runtime_identity: "a".repeat(64) },
    settings: { "max-num-seqs": "16" },
    environment: { NCCL_NET: "Socket" },
    fetch: fetchImpl,
    loadPyodide: fake.loadPyodide,
  });

  assert.deepEqual(
    fake.state.loads,
    [`${ROOT}pyyaml-6.0.3-cp312-cp312-emscripten_20_19_wasm32.whl`],
    "the tree's own wheel is fetched by URL, not resolved as a package name",
  );
  // The boot used the self-hosted tree, loaded exactly the PyYAML wheel, staged the
  // verified launcher modules, and ran the resolver.
  assert.deepEqual(fake.state.indexUrls, [ROOT], "booted from the pyodide tree");
  assert.ok(fake.state.written.some((path) => path.endsWith("image_tools/launcher/resolver.py")));
  assert.equal(fake.state.runs.length, 1, "ran the resolver exactly once");
  assert.ok(fake.state.runs[0].includes("resolver.resolve"), "the program calls resolver.resolve");
  const call = JSON.parse(fake.state.globals.VLLM_IMAGE_RESOLVE_CALL);
  assert.equal(call.profile, "qwen38-flash-next");
  assert.equal(call.hardware, "gb10-roce");
  assert.deepEqual(call.argv, ["--max-num-seqs", "16"], "settings are the CLI layer");
  assert.deepEqual(call.env, { NCCL_NET: "Socket" }, "reader-set variables arrive as process environment");
  assert.deepEqual(call.cli_env, { NCCL_NET: "Socket" }, "and are reported against the same layer");
  // The keyword set is `launch.py`'s resolved_plan() and nothing more: one extra
  // layer here, such as a `config` the container never passes, means the browser
  // answered from a different set of inputs than the image would have.
  assert.deepEqual(
    Object.keys(call).sort(),
    ["argv", "cli_env", "env", "hardware", "preset", "profile", "recipe_layer", "root", "runtime_identity", "vllm_environment"].sort(),
  );
  assert.deepEqual(call.vllm_environment, ["VLLM_USE_V1"], "the container's answer is forwarded");
  assert.equal(call.b12x_mxfp8_moe, undefined);
  assert.equal(call.runtime_identity, "a".repeat(64), "the JIT namespace comes from the record");
  assert.deepEqual(call.env, { NCCL_NET: "Socket" }, "reader-set variables resolve as process environment");
});

test("a resolution without the published context rejects instead of guessing", async () => {
  // Without vllm_environment and runtime_identity the browser would answer as if no
  // vLLM were installed: both outputs are well-formed, and only one is what runs.
  const { fetchImpl } = await buildTree();
  const fake = fakePyodide(["unused"]);
  await assert.rejects(
    () => resolveInBrowser({ base: BASE, selection: "profile:qwen38-flash-next#gb10-roce", settings: {}, environment: {}, fetch: fetchImpl, loadPyodide: fake.loadPyodide }),
    /resolution_context/,
    "the rejection names the missing field",
  );
  assert.equal(fake.state.runs.length, 0, "nothing ran without a context");
});
test("a tampered file rejects before anything runs", async () => {
  const argv = ["should", "not", "run"];
  const { fetchImpl } = await buildTree({ tamper: { path: "image_tools/launcher/resolver.py", bytes: "import os; os.system('rm -rf /')\n" } });
  const fake = fakePyodide(argv);
  await assert.rejects(
    () => resolveInBrowser({ base: BASE, selection: "profile:qwen38-flash-next#gb10-roce", settings: {}, environment: {}, fetch: fetchImpl, loadPyodide: fake.loadPyodide }),
    /digest .* does not match the manifest/,
    "the SHA-256 mismatch rejects",
  );
  assert.equal(fake.state.runs.length, 0, "no Python ran against tampered bytes");
  assert.equal(fake.state.written.length, 0, "nothing was staged");
});

test("an absent pyodide tree rejects naming the missing artefact", async () => {
  const { fetchImpl } = await buildTree({ manifest: false });
  const fake = fakePyodide(["unused"]);
  await assert.rejects(
    () => resolveInBrowser({ base: BASE, selection: "profile:qwen38-flash-next#gb10-roce", settings: {}, environment: {}, fetch: fetchImpl, loadPyodide: fake.loadPyodide }),
    /manifest\.json/,
    "the rejection names manifest.json so the caller can fall back",
  );
  assert.equal(fake.state.runs.length, 0);
});

test("a recipe selection without a resolved profile rejects naming the recipe artefact", async () => {
  const { fetchImpl } = await buildTree();
  const fake = fakePyodide(["unused"]);
  await assert.rejects(
    () => resolveInBrowser({ base: BASE, selection: "recipe:qwen38-flash-next-gb10-tp2", settings: { "max-num-seqs": "16" }, environment: {}, fetch: fetchImpl, loadPyodide: fake.loadPyodide }),
    /recipe document for qwen38-flash-next-gb10-tp2/,
    "a recipe selection needs the resolved profile/hardware the tree does not publish",
  );
});

test("a missing required policy file rejects even when the manifest is otherwise valid", async () => {
  const tree = await buildTree();
  const stripped = tree.tree.files.filter((entry) => entry.path !== "options.yaml");
  const fetchImpl = async (url) => {
    if (url === `${ROOT}manifest.json`) return { ok: true, arrayBuffer: async () => new TextEncoder().encode(JSON.stringify({ ...tree.tree, files: stripped })).buffer };
    const path = url.slice(ROOT.length);
    if (path === "options.yaml") return { ok: false, status: 404 };
    return tree.fetchImpl(url);
  };
  const fake = fakePyodide(["unused"]);
  await assert.rejects(
    () => resolveInBrowser({ base: BASE, selection: "profile:qwen38-flash-next#gb10-roce", settings: {}, environment: {}, fetch: fetchImpl, loadPyodide: fake.loadPyodide }),
    /options\.yaml/,
    "the resolver cannot run without its policy data",
  );
});
