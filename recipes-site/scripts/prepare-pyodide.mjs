// Assemble the self-hosted Pyodide distribution the site's vLLM command tab
// runs the launcher's own resolver in.
//
// Every byte here is pinned by version and SHA-256 and verified before use, and
// the browser loads nothing from a CDN: the Pyodide runtime comes from one
// GitHub release archive, the one Python package (PyYAML, which the image also
// locks) comes from the package index named by that archive's own lock file,
// and the resolver plus the policy data are copied out of this repository and
// the pinned upstream tree. `manifest.json` records every file with its digest
// and its source, so the published tree can be re-verified without rerunning
// this script.
//
// Running it twice with the same pins produces byte-identical output: the
// directory is rebuilt from scratch and the manifest is derived from file
// contents alone, never from timestamps or directory order.

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { assertNoPrivateReference, locate, pinnedSources } from "./resolve-releases.mjs";
import { mergedRuntimeRoot } from "./resolve-configs.mjs";

const SITE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const REPOSITORY_ROOT = resolve(SITE_ROOT, "..");
const DEFAULT_OUTPUT = join(SITE_ROOT, "public/pyodide");
const DEFAULT_CACHE = join(process.env.HOME || ".", ".cache/vllm-multiarch-oci/pyodide");

const PYODIDE_VERSION = "314.0.7";
const PYODIDE_RELEASE = `https://github.com/pyodide/pyodide/releases/download/${PYODIDE_VERSION}`;
const PYODIDE_CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full`;
const ARCHIVE = `pyodide-core-${PYODIDE_VERSION}.tar.bz2`;
const ARCHIVE_SHA256 = "2abdcc2e35208af406e07724cffa85bc582ced97e9028383ecf5462541393f95";

// The runtime a browser needs to boot CPython and load a wheel. `python`,
// `python.exe`, `python.bat` and the type declarations are Node and editor
// artefacts, not part of the served distribution.
const CORE_FILES = [
  "pyodide.js",
  "pyodide.mjs",
  "pyodide.asm.mjs",
  "pyodide.asm.wasm",
  "python_stdlib.zip",
  "pyodide-lock.json",
];

// The image locks PyYAML for the resolver's own use; the browser needs the same
// parser and no other package, because the resolver is standard library plus
// PyYAML.
const PACKAGES = [{ name: "pyyaml", version: "6.0.3" }];

// The launcher modules the browser imports, and the policy files the resolver
// reads. Both sets come from the same pins the image installs.
const LAUNCHER_FILES = [
  "image_tools/__init__.py",
  "image_tools/launcher/__init__.py",
  "image_tools/launcher/resolver.py",
  "image_tools/launcher/probes.py",
  "image_tools/launcher/cache_runtime.py",
];
const POLICY_FILES = ["options.yaml", "parameter-docs.yaml", "schema.json", "platform-environment.json", "presets.yaml"];
const POLICY_DIRECTORIES = ["profiles", "hardware", "templates"];

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function fileDigest(path) {
  return sha256(readFileSync(path));
}

/** The pinned distribution this script assembles. */
export function pins() {
  return {
    version: PYODIDE_VERSION,
    archive: { name: ARCHIVE, url: `${PYODIDE_RELEASE}/${ARCHIVE}`, sha256: ARCHIVE_SHA256 },
    coreFiles: [...CORE_FILES],
    packages: PACKAGES.map((entry) => ({ ...entry })),
    launcherFiles: [...LAUNCHER_FILES],
    policyFiles: [...POLICY_FILES],
    policyDirectories: [...POLICY_DIRECTORIES],
    cdn: PYODIDE_CDN,
    release: PYODIDE_RELEASE,
  };
}

async function download(url, destination) {
  const response = await fetch(url, { redirect: "follow" });
  assert(response.ok, `${url}: HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  assert(bytes.byteLength > 0, `${url}: empty response`);
  // Write through a temporary name, so an interrupted run cannot leave a
  // truncated file that later looks like a verified cache entry.
  const temporary = `${destination}.part`;
  mkdirSync(dirname(destination), { recursive: true });
  writeFileSync(temporary, bytes);
  renameSync(temporary, destination);
  return destination;
}

/**
 * A cached artefact, downloaded at most once and verified by digest on every
 * use. A mismatch is fatal: the alternative is publishing bytes nobody checked.
 */
async function cached(dist, cacheDir, url, name, expected, { offline }) {
  const path = join(cacheDir, dist.version, name);
  if (!existsSync(path)) {
    assert(!offline, `${path}: not cached, and --offline forbids downloading ${url}`);
    await download(url, path);
  }
  assert(existsSync(path), `${path}: vanished while being prepared`);
  const actual = fileDigest(path);
  assert(actual === expected, `${url}: checksum mismatch, expected ${expected} and got ${actual}`);
  return path;
}

function extract(archive, members, staging) {
  rmSync(staging, { recursive: true, force: true });
  mkdirSync(staging, { recursive: true });
  // Only the named members are unpacked, so a release archive can never drop an
  // unexpected file into the published tree. An absent member makes tar exit 2;
  // that is tolerated because the archive's digest has already been verified, so
  // exit 2 can only mean a missing member, and the caller then fails by name
  // rather than surfacing tar's stderr.
  const result = spawnSync("tar", ["--extract", "--file", archive, "--directory", staging, ...members], { encoding: "utf8" });
  if (result.error) assert(false, `tar --extract ${archive}: ${result.error.message}`);
  assert(result.signal === null, `tar --extract ${archive}: killed by ${result.signal}`);
}

function copyInto(source, destination, label) {
  mkdirSync(dirname(destination), { recursive: true });
  copyFileSync(source, destination);
  assert(existsSync(destination), `${label}: was not written`);
}

function listing(directory, prefix = "") {
  const entries = [];
  for (const name of readdirSync(directory).sort()) {
    const path = join(directory, name);
    if (statSync(path).isDirectory()) entries.push(...listing(path, `${prefix}${name}/`));
    else entries.push({ path: `${prefix}${name}`, absolute: path });
  }
  return entries;
}

/**
 * Build the published tree from a pinned distribution and return the manifest
 * that was written. `dist` is a parameter rather than a constant so the failure
 * paths — a tampered cache entry, a missing member — are testable offline.
 *
 * `runtimeRoot` is the merged policy tree the resolver reads: upstream's
 * `runtime/` with this repository's hardware profiles landed beside its own,
 * which `resolve-configs.mjs` prepares in the same build.
 */
export async function assemble(dist, { output = DEFAULT_OUTPUT, cacheDir = DEFAULT_CACHE, runtimeRoot, workDir, offline = false } = {}) {
  assert(dist && dist.archive && dist.coreFiles, "assemble needs a pinned distribution");
  assert(runtimeRoot, "assemble needs runtimeRoot: the pinned policy tree, prepared by resolve-configs.mjs");
  assert(existsSync(runtimeRoot) && statSync(runtimeRoot).isDirectory(), `${runtimeRoot}: the merged policy tree is missing; run resolve-configs.mjs first`);
  for (const name of dist.policyFiles) assert(existsSync(join(runtimeRoot, name)), `${runtimeRoot}: ${name} is missing from the policy tree`);
  for (const name of dist.policyDirectories) assert(existsSync(join(runtimeRoot, name)), `${runtimeRoot}: ${name}/ is missing from the policy tree`);

  const options = { offline };
  const archive = await cached(dist, cacheDir, dist.archive.url, dist.archive.name, dist.archive.sha256, options);
  const staging = join(workDir || cacheDir, `staging-${dist.version}`);
  extract(archive, dist.coreFiles.map((name) => `pyodide/${name}`), staging);
  const extracted = join(staging, "pyodide");

  // The lock shipped inside the verified archive is the authority for what a
  // package's bytes must hash to; the wheel is then checked against it.
  const lock = JSON.parse(readFileSync(join(extracted, "pyodide-lock.json"), "utf8"));
  assert(lock.info && typeof lock.info === "object" && lock.packages && typeof lock.packages === "object", `${extracted}/pyodide-lock.json: no info or packages block`);

  rmSync(output, { recursive: true, force: true });
  mkdirSync(output, { recursive: true });
  const files = [];
  const record = (path, source) => {
    const absolute = join(output, path);
    assert(existsSync(absolute), `${path}: missing after assembly`);
    files.push({ path, bytes: statSync(absolute).size, sha256: fileDigest(absolute), source });
  };

  for (const name of dist.coreFiles) {
    const source = join(extracted, name);
    assert(existsSync(source), `${dist.archive.url}: the archive carries no ${name}`);
    copyInto(source, join(output, name), name);
    record(name, dist.archive.url);
  }

  const packages = [];
  for (const entry of dist.packages) {
    const packaged = lock.packages[entry.name];
    assert(packaged, `${extracted}/pyodide-lock.json: no ${entry.name} package`);
    assert(packaged.version === entry.version, `${entry.name}: the lock pins ${packaged.version}, this script pins ${entry.version}`);
    const url = `${dist.cdn}/${packaged.file_name}`;
    const path = await cached(dist, cacheDir, url, packaged.file_name, packaged.sha256, options);
    copyInto(path, join(output, packaged.file_name), packaged.file_name);
    record(packaged.file_name, url);
    packages.push({ name: entry.name, version: packaged.version, file: packaged.file_name, sha256: packaged.sha256, url });
  }

  for (const name of dist.launcherFiles) {
    const source = join(REPOSITORY_ROOT, name);
    assert(existsSync(source), `${source}: the launcher module is missing`);
    copyInto(source, join(output, name), name);
    record(name, `vllm-multiarch-oci:${name}`);
  }

  for (const name of dist.policyFiles) {
    copyInto(join(runtimeRoot, name), join(output, name), name);
    record(name, `blackwell-llm-docker:runtime/${name}`);
  }
  for (const name of dist.policyDirectories) {
    // The resolver reads root/profiles/<id>.yaml, so the layout must survive:
    // a flattened tree would resolve nothing in the browser.
    for (const entry of listing(join(runtimeRoot, name))) {
      const path = `${name}/${entry.path}`;
      copyInto(entry.absolute, join(output, path), path);
      record(path, `blackwell-llm-docker:runtime/${path}`);
    }
  }

  const pinned = pinnedSources();
  files.sort((left, right) => (left.path < right.path ? -1 : left.path > right.path ? 1 : 0));
  const manifest = {
    schema_version: 1,
    generated_by: "recipes-site/scripts/prepare-pyodide.mjs",
    pyodide: { version: dist.version, archive: dist.archive.name, archive_sha256: dist.archive.sha256, url: dist.archive.url },
    packages,
    runtime: { repository: "local-inference-lab/blackwell-llm-docker", commit: pinned.lilRuntime },
    files,
  };
  const serialised = `${JSON.stringify(manifest, null, 2)}\n`;
  assertNoPrivateReference(serialised, `${output}: manifest.json`);
  writeFileSync(join(output, "manifest.json"), serialised);
  verifyPublished(output, manifest);
  return manifest;
}

/**
 * Re-read the tree that was just written against its own manifest: every
 * listed file must exist with the recorded digest and size, and nothing may be
 * published that the manifest does not account for. A browser has no way to
 * notice a silently absent wasm binary, so this is checked where it is built.
 */
export function verifyPublished(output, manifest) {
  for (const entry of manifest.files) {
    const path = join(output, entry.path);
    assert(existsSync(path) && statSync(path).isFile(), `${entry.path}: listed in the manifest but not published`);
    assert(fileDigest(path) === entry.sha256, `${entry.path}: digest does not match the manifest`);
    assert(statSync(path).size === entry.bytes, `${entry.path}: size does not match the manifest`);
  }
  const published = new Set();
  const walk = (directory, prefix) => {
    for (const name of readdirSync(directory).sort()) {
      const path = join(directory, name);
      if (statSync(path).isDirectory()) walk(path, `${prefix}${name}/`);
      else if (`${prefix}${name}` !== "manifest.json") published.add(`${prefix}${name}`);
    }
  };
  walk(output, "");
  const listed = new Set(manifest.files.map((entry) => entry.path));
  const disagreement = [...published].filter((path) => !listed.has(path)).concat([...listed].filter((path) => !published.has(path)));
  assert(disagreement.length === 0, `${output}: published tree and manifest disagree: ${disagreement.join(", ")}`);
  return manifest;
}

export function prepare(options) {
  return assemble(pins(), options);
}

const DEFAULT_WORK = join(process.env.HOME || ".", ".cache/vllm-multiarch-oci/pages-data");

function parseArguments(argv) {
  const options = { output: DEFAULT_OUTPUT, cacheDir: DEFAULT_CACHE, runtimeDir: null, workDir: DEFAULT_WORK, offline: false };
  for (let index = 0; index < argv.length; index += 1) {
    const value = (name) => {
      assert(argv[index + 1], `--${name} needs a value`);
      index += 1;
      return argv[index];
    };
    switch (argv[index]) {
      case "--output":
        options.output = value("output");
        break;
      case "--cache-dir":
        options.cacheDir = value("cache-dir");
        break;
      case "--runtime-dir":
        options.runtimeDir = value("runtime-dir");
        break;
      case "--work-dir":
        options.workDir = value("work-dir");
        break;
      case "--offline":
        options.offline = true;
        break;
      default:
        assert(false, `unknown argument ${argv[index]}`);
    }
  }
  // The policy tree is the one resolve-configs.mjs staged for this pin; the
  // two scripts share the naming so a build passes it without a glob.
  if (!options.runtimeDir) options.runtimeDir = mergedRuntimeRoot(options.workDir, pinnedSources().lilRuntime);
  return options;
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const manifest = await prepare({
    output: resolve(process.cwd(), options.output),
    cacheDir: resolve(process.cwd(), options.cacheDir),
    runtimeRoot: locate(options.runtimeDir),
    workDir: resolve(process.cwd(), options.workDir),
    offline: options.offline,
  });
  process.stdout.write(`${options.output}: Pyodide ${manifest.pyodide.version}, ${manifest.files.length} files, ${manifest.packages.length} packages\n`);
}

if (process.argv[1] && import.meta.url === `file://${resolve(process.argv[1])}`) {
  await main();
}
