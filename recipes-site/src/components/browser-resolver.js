// Run the image's own resolver in the browser so the vLLM command the reader edits is
// the exact command the container would produce. Nothing here re-implements a rule: it
// boots the self-hosted Pyodide tree at `${base}pyodide/`, verifies every file against
// `manifest.json` before executing any of it, then calls `resolver.resolve()` — the same
// entry point `launch.py` calls, with the same keyword arguments and nothing else.
// The reader's `settings` are applied as the CLI layer (`--key value` in `argv`) and
// `environment` as the process environment, which is how a container receives them
// from its pod spec.

// The launcher modules the resolver imports, and the policy files the resolver reads.
// These must be present for a resolution to run; the manifest is the trust anchor for
// their bytes, and a mismatch rejects rather than executing unverified code.
const PYODIDE_MODULE = "pyodide.mjs";
const REQUIRED_LAUNCHER_FILES = [
  "image_tools/__init__.py",
  "image_tools/launcher/__init__.py",
  "image_tools/launcher/resolver.py",
  "image_tools/launcher/probes.py",
  "image_tools/launcher/cache_runtime.py",
];
const REQUIRED_POLICY_FILES = [
  "options.yaml",
  "parameter-docs.yaml",
  "schema.json",
  "platform-environment.json",
  "presets.yaml",
];

function toHex(buffer) {
  return Array.from(new Uint8Array(buffer), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return toHex(digest);
}

function asBytes(value) {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  throw new TypeError("resolved files must be bytes");
}

async function readResponse(response, path) {
  if (!response || response.ok !== true) {
    const status = response && response.status ? ` (HTTP ${response.status})` : "";
    throw new Error(`the pyodide tree has no ${path}${status}`);
  }
  return asBytes(await response.arrayBuffer());
}

// The manifest is the trust anchor. Its absence means the pull-request build, which does
// not publish a pyodide tree: reject naming the missing artefact so the caller can fall
// back to the unedited command instead of executing nothing.
async function loadManifest(root, fetchImpl) {
  const path = `${root}manifest.json`;
  let response;
  try {
    response = await fetchImpl(path, { cache: "no-store" });
  } catch (error) {
    throw new Error(`could not load the pyodide tree at ${root}: ${error.message}`);
  }
  if (!response || response.ok !== true) {
    throw new Error(`the pyodide tree at ${root} is not published (missing ${path})`);
  }
  const text = new TextDecoder().decode(await readResponse(response, path));
  let manifest;
  try {
    manifest = JSON.parse(text);
  } catch (error) {
    throw new Error(`${path}: manifest.json is not valid JSON: ${error.message}`);
  }
  if (!manifest || !Array.isArray(manifest.files)) {
    throw new Error(`${path}: manifest.json has no files array`);
  }
  return manifest;
}

async function fetchVerified(root, entry, fetchImpl) {
  const response = await fetchImpl(`${root}${entry.path}`, { cache: "no-store" });
  const bytes = await readResponse(response, entry.path);
  const digest = await sha256Hex(bytes);
  if (digest !== entry.sha256) {
    throw new Error(`${entry.path}: digest ${digest} does not match the manifest ${entry.sha256}`);
  }
  return bytes;
}

// Parse the selection into the arguments `launch.py` hands to `resolver.resolve()`. The
// caller passes the model string; a `profile:` selection resolves upstream policy
// directly, while a `recipe:` selection must find the recipe document in the tree so the
// recipe's option and environment layer is applied exactly as the container applies it.
function parseSelection(selection) {
  if (selection && typeof selection === "object") {
    const profile = selection.profile;
    const hardware = selection.hardware;
    if (typeof profile !== "string" || typeof hardware !== "string") {
      throw new TypeError("a selection object needs profile and hardware");
    }
    return {
      kind: selection.recipe ? "recipe" : "profile",
      name: selection.recipe ?? null,
      profile,
      hardware,
      preset: selection.preset ?? null,
      layer: selection.options || selection.environment ? { options: selection.options ?? {}, environment: selection.environment ?? {}, source: `recipe:${selection.recipe}` } : null,
    };
  }
  if (typeof selection !== "string" || !selection) throw new TypeError("selection must be a string or an object");
  const [kind, ...rest] = selection.split(":");
  if (kind === "recipe") return { kind: "recipe", name: rest.join(":"), profile: null, hardware: null, preset: null, layer: null };
  if (kind === "profile") {
    const [profile, hardware, preset] = rest.join(":").split("#");
    if (!profile || !hardware) throw new TypeError(`profile selection ${selection} is missing hardware`);
    return { kind: "profile", name: null, profile, hardware, preset: preset ?? null, layer: null };
  }
  throw new TypeError(`unrecognised selection ${selection}`);
}

// The reader's edits as the container receives them: `settings` become the native CLI
// argv (mirroring `launch.py` passing `request.native` to `resolve(argv=...)`, and the
// same `--key value` / `--key=json` spelling `render.js`'s changeArgs produces), and
// `environment` becomes the process environment the pod spec would have set.
function nativeArgv(settings) {
  const argv = [];
  for (const [name, value] of Object.entries(settings ?? {})) {
    if (value === undefined || value === null || String(value).trim() === "") continue;
    const text = String(value);
    if (name.includes(".") || /^[{[]/.test(text)) argv.push(`--${name}=${text}`);
    else argv.push(`--${name}`, text);
  }
  return argv;
}

function configEnvironment(environment) {
  const env = {};
  for (const [name, value] of Object.entries(environment ?? {})) {
    if (value === undefined || value === null || String(value).trim() === "") continue;
    env[name] = String(value);
  }
  return env;
}

// Boot Pyodide in a browser. In the Node test lane a fake loader is injected and
// this never runs.
//
// It is imported from its served URL rather than from a blob of verified bytes,
// because pyodide.mjs resolves its own siblings (`pyodide.asm.mjs`, the wasm and
// the lock file) relative to its own URL; a blob at an opaque origin would strand
// them. The digest is still checked, on the same origin, immediately before the
// import -- which is the same trust boundary the rest of the tree relies on: this
// defends against a corrupted or tampered publication, not against the site
// serving different bytes to the import than it served to the check.
async function importLoader(root, manifest, fetchImpl) {
  const entry = manifest.files.find((candidate) => candidate.path === PYODIDE_MODULE);
  if (!entry) throw new Error(`the pyodide tree at ${root} publishes no ${PYODIDE_MODULE}`);
  await fetchVerified(root, entry, fetchImpl);
  const module = await import(/* @vite-ignore */ `${root}${PYODIDE_MODULE}`);
  const loader = module.loadPyodide ?? module.default?.loadPyodide;
  if (typeof loader !== "function") throw new Error(`${PYODIDE_MODULE} did not export loadPyodide`);
  return loader;
}

async function stageFiles(pyodide, files, rootDir) {
  const fs = pyodide.FS;
  for (const [path, bytes] of files) {
    const target = `${rootDir}/${path}`;
    const parent = target.slice(0, target.lastIndexOf("/"));
    fs.mkdirTree?.(parent);
    if (fs.analyzePath?.(parent).exists !== true && fs.mkdirTree === undefined) {
      // Older pyodide FS without mkdirTree: create each segment.
      const segments = parent.split("/").filter(Boolean);
      let built = "";
      for (const segment of segments) {
        built += `/${segment}`;
        try { fs.mkdir(built); } catch { /* already exists */ }
      }
    }
    fs.writeFile(target, bytes);
  }
}

async function runResolver(pyodide, { rootDir, selection, settings, environment, context }) {
  // `vllm_environment` and `runtime_identity` come from the published record's
  // `resolution_context`, because they are what the container's installed packages
  // say and a browser cannot discover them: left to their defaults, a preset that
  // needs a vLLM capability resolves as if vLLM were absent, and the JIT cache paths
  // that appear in the command lose their namespace. Both answers are well-formed,
  // so the only way to catch the difference is to not introduce it.
  //
  // The keyword set here is `launch.py`'s `resolved_plan()` exactly, including the
  // absence of `config`: a container's reader-set variables arrive as process
  // environment, which `launch.py` hands over twice -- once to resolve from, once to
  // report origins against. Nothing is inherited from the browser's own environment.
  const call = {
    profile: selection.profile,
    hardware: selection.hardware,
    preset: selection.preset,
    recipe_layer: selection.layer,
    argv: nativeArgv(settings),
    env: { ...configEnvironment(environment) },
    cli_env: { ...configEnvironment(environment) },
    vllm_environment: context?.vllm_environment === null ? null : [...(context?.vllm_environment ?? [])],
    runtime_identity: context?.runtime_identity ?? null,
    root: rootDir,
  };
  pyodide.globals.set("VLLM_IMAGE_RESOLVE_CALL", JSON.stringify(call));
  const program = [
    "import json",
    "import sys",
    "from pathlib import Path",
    `sys.path.insert(0, ${JSON.stringify(rootDir)})`,
    "from image_tools.launcher import resolver",
    "_call = json.loads(VLLM_IMAGE_RESOLVE_CALL)",
    "_plan = resolver.resolve(",
    "    _call['profile'], _call['hardware'],",
    "    preset=_call['preset'],",
    "    recipe_layer=_call['recipe_layer'],",
    "    argv=_call['argv'],",
    "    env=_call['env'],",
    "    cli_env=_call['cli_env'],",
    "    vllm_environment=(None if _call['vllm_environment'] is None else frozenset(_call['vllm_environment'])),",
    "    runtime_identity=_call['runtime_identity'],",
    "    root=Path(_call['root']),",
    ")",
    "json.dumps(_plan.argv)",
  ].join("\n");
  const result = await pyodide.runPython(program);
  const argv = typeof result === "string" ? JSON.parse(result) : result;
  if (!Array.isArray(argv) || argv.some((part) => typeof part !== "string")) {
    throw new Error("the in-browser resolver did not return an argv array");
  }
  return argv;
}

export async function resolveInBrowser({ base, selection, settings, environment, context, fetch: fetchImpl = globalThis.fetch, loadPyodide: loadImpl } = {}) {
  if (typeof fetchImpl !== "function") throw new TypeError("resolveInBrowser needs a fetch implementation");
  const root = `${String(base ?? "/").replace(/\/+$/, "")}/pyodide/`;
  const manifest = await loadManifest(root, fetchImpl);

  const files = new Map();
  for (const entry of manifest.files) {
    files.set(entry.path, await fetchVerified(root, entry, fetchImpl));
  }
  for (const required of [...REQUIRED_LAUNCHER_FILES, ...REQUIRED_POLICY_FILES]) {
    if (!files.has(required)) throw new Error(`the pyodide tree at ${root} is missing ${required}`);
  }

  const parsed = parseSelection(selection);
  // A recipe selection applies the recipe's option and environment layer, which only
  // `launch.py`'s recipe loader can build — and launch.py is not part of the browser
  // tree (prepare-pyodide.mjs ships the resolver modules and policy data, not the
  // recipe files or the entry point). A profile selection resolves upstream policy
  // directly and needs none of it. So a recipe selection with no resolved profile in
  // the selection object rejects naming the missing artefact, and the caller falls back
  // to the record's command; a caller that has the resolved selection may pass it as an
  // object and resolution proceeds exactly as the container's.
  if (parsed.kind === "recipe" && (!parsed.profile || !parsed.hardware)) {
    throw new Error(`the pyodide tree at ${root} publishes no recipe document for ${parsed.name}; the resolved profile and hardware must accompany a recipe selection`);
  }
  if (!context || typeof context !== "object") {
    throw new Error(
      `resolveInBrowser needs the record's resolution_context; without it a browser with no vLLM installed would resolve ${parsed.kind === "recipe" ? parsed.name : "this selection"} differently from the container that serves it`,
    );
  }

  const loader = loadImpl ?? (await importLoader(root, manifest, fetchImpl));
  const pyodide = await loader({ indexUrl: root });
  const wheel = manifest.packages?.find((pkg) => pkg.name === "pyyaml")?.file ?? manifest.files.find((entry) => entry.path.endsWith("pyyaml"))?.path;
  if (!wheel) throw new Error(`the pyodide tree at ${root} publishes no PyYAML wheel`);
  // A bare file name is not enough: loadPackage treats it as a name to resolve
  // against the index and, for a wheel outside pyodide-lock.json, installs nothing.
  // The absolute URL is what makes the browser fetch this tree's own wheel.
  await pyodide.loadPackage(`${root}${wheel}`);

  const rootDir = "/vllm-image";
  await stageFiles(pyodide, files, rootDir);
  return runResolver(pyodide, { rootDir, selection: parsed, settings, environment, context });
}
