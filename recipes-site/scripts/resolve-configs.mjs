// Generate the resolved-configuration and option-documentation artefacts the
// deployment flow renders from, using this repository's own launcher resolver.
//
// There are exactly two sources of truth: the recipe YAML under
// `recipes/<owner>/` and the pinned upstream policy data named by
// `profiles/vllmb12x/profile.json`. Nothing here restates a default: every
// value comes out of `python3 -m image_tools.vllm_image launch --print-config`,
// the same command the image runs, so what the site shows is what the container
// resolves. `--print-config` is offline by contract; this script never downloads.
//
// The pull-request path passes `--fixture-dir`, which copies committed
// documents instead of fetching and resolving, exactly as `latest-image.json`
// does. Both paths run the same validators.

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parse as parseYaml } from "yaml";

import { assertNoPrivateReference, locate, pinnedSources } from "./resolve-releases.mjs";

const SITE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const REPOSITORY_ROOT = resolve(SITE_ROOT, "..");
const PUBLIC_ROOT = join(SITE_ROOT, "public");
const RECIPE_ROOT = join(REPOSITORY_ROOT, "recipes");

// The recipes the deployment flow offers as validated. A missing one is a build
// failure, not a skip: the flow cannot show a card for a recipe that vanished.
const REQUIRED_RECIPES = ["qwen38-flash-next-gb10-tp2", "deepseek-v4-flash-vision-gb10-tp2", "qwen38-27b"];

// The policy files the browser-side resolver needs. Everything else in the
// upstream tree is Python the image does not ship.
const POLICY_FILES = ["options.yaml", "parameter-docs.yaml", "schema.json", "platform-environment.json", "presets.yaml"];
const POLICY_DIRECTORIES = ["profiles", "hardware", "templates"];

// A documentation entry for a value this repository sets beyond upstream.
const DOC_FIELDS = ["group", "summary", "why"];
const SETTINGS_KEYS = ["schema_version", "status", "qualification", "profile", "hardware", "argv", "settings", "environment", "warnings", "cache_service", "preset", "recipe", "role", "selection", "topology", "resolver_version"];
const COMMIT = /^[0-9a-f]{40}$/;
const SELECTOR = /^(?:recipe:[a-z0-9][a-z0-9-]*|profile:[a-z0-9][a-z0-9-]*#[a-z0-9][a-z0-9-]*(?:#[a-z0-9][a-z0-9-]*)?)$/;

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function digest(text) {
  return createHash("sha256").update(text).digest("hex").slice(0, 12);
}

// ---------------------------------------------------------------- selection

function listYaml(directory) {
  assert(existsSync(directory), `${directory}: directory is missing`);
  return readdirSync(directory)
    .filter((name) => /\.ya?ml$/.test(name))
    .map((name) => name.replace(/\.ya?ml$/, ""))
    .sort();
}

/** Every recipe file under a directory tree, as `<name, path>` pairs. Recipes
    live one level down in an owner directory, so a name must be unique across the
    whole tree: two files claiming one recipe name would make `--recipe <name>`
    ambiguous, and the site must not pick a winner by directory order. */
function recipeFiles(directory) {
  assert(existsSync(directory), `${directory}: directory is missing`);
  const found = new Map();
  const walk = (current) => {
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      const path = join(current, entry.name);
      if (entry.isDirectory()) walk(path);
      else if (/\.ya?ml$/.test(entry.name)) {
        const name = entry.name.replace(/\.ya?ml$/, "");
        assert(!found.has(name), `${path}: duplicates recipe ${name} from ${found.get(name)}`);
        found.set(name, path);
      }
    }
  };
  walk(directory);
  return found;
}

function recipePath(name, files = recipeFiles(RECIPE_ROOT)) {
  const path = files.get(name);
  assert(path, `${RECIPE_ROOT}: recipe ${name}.yaml is missing`);
  return path;
}

function readYaml(path, source) {
  let data;
  try {
    data = parseYaml(readFileSync(path, "utf8"));
  } catch (error) {
    assert(false, `${source}: cannot be parsed: ${error.message}`);
  }
  assert(data && typeof data === "object" && !Array.isArray(data), `${source}: must be a mapping`);
  return data;
}

/** The selections to resolve: every recipe, profile x hardware, and preset. */
export function selections(runtimeRoot, recipeNames) {
  const profiles = listYaml(join(runtimeRoot, "profiles")).filter((name) => {
    const kind = readYaml(join(runtimeRoot, "profiles", `${name}.yaml`), `profile ${name}`).kind;
    assert(["model", "common"].includes(kind), `profile ${name}: unexpected kind ${kind}`);
    return kind === "model";
  });
  assert(profiles.length > 0, `${runtimeRoot}: upstream publishes no model profiles`);
  const hardware = listYaml(join(runtimeRoot, "hardware")).filter((name) => {
    const kind = readYaml(join(runtimeRoot, "hardware", `${name}.yaml`), `hardware ${name}`).kind;
    assert(kind === "hardware", `hardware ${name}: expected kind hardware, found ${kind}`);
    return true;
  });
  assert(hardware.length > 0, `${runtimeRoot}: no hardware profiles, not even this repository's own`);
  const presets = readYaml(join(runtimeRoot, "presets.yaml"), "presets.yaml").presets;
  assert(presets && typeof presets === "object" && !Array.isArray(presets), "presets.yaml: presets must be a mapping");

  const requested = [];
  for (const name of recipeNames) requested.push({ key: `recipe:${name}`, argv: ["--recipe", name], recipe: name });
  for (const profile of profiles) {
    for (const device of hardware) requested.push({ key: `profile:${profile}#${device}`, argv: ["--profile", profile, "--hardware", device] });
  }
  for (const [name, preset] of Object.entries(presets)) {
    assert(typeof preset.profile === "string" && preset.profile, `preset ${name}: profile is required`);
    assert(typeof preset.hardware === "string" && preset.hardware, `preset ${name}: hardware is required`);
    assert(profiles.includes(preset.profile), `preset ${name}: refers to unknown profile ${preset.profile}`);
    assert(hardware.includes(preset.hardware), `preset ${name}: refers to unknown hardware ${preset.hardware}`);
    requested.push({
      key: `profile:${preset.profile}#${preset.hardware}#${name}`,
      argv: ["--profile", preset.profile, "--hardware", preset.hardware, "--preset", name],
    });
  }
  const keys = requested.map((entry) => entry.key);
  assert(new Set(keys).size === keys.length, `selections must be unique: ${keys.filter((key, index) => keys.indexOf(key) !== index).join(", ")}`);
  return requested;
}

// ------------------------------------------------------------------ records

function validateValue(value, where) {
  assert(value === null || typeof value === "string" || typeof value === "number" || typeof value === "boolean" || Array.isArray(value) || (typeof value === "object" && value !== null), `${where}: unsupported value`);
}

export function validateRecord(record, key) {
  const source = `configs.json: ${key}`;
  assert(record && typeof record === "object" && !Array.isArray(record), `${source}: record must be an object`);
  assert(JSON.stringify(Object.keys(record).sort()) === JSON.stringify([...SETTINGS_KEYS].sort()), `${source}: keys must be exactly ${SETTINGS_KEYS.join(", ")}`);
  assert(record.schema_version === 1, `${source}: unexpected schema_version ${record.schema_version}`);
  assert(typeof record.status === "string" && record.status.trim(), `${source}: status is required`);
  assert(typeof record.qualification === "string" && record.qualification.trim(), `${source}: qualification is required`);
  assert(typeof record.profile === "string" && /^[a-z][a-z0-9-]*$/.test(record.profile), `${source}: profile must be a lower-case identifier: ${record.profile}`);
  assert(typeof record.hardware === "string" && /^[a-z][a-z0-9-]*$/.test(record.hardware), `${source}: hardware must be a lower-case identifier: ${record.hardware}`);
  assert(record.preset === null || (typeof record.preset === "string" && /^[a-z][a-z0-9-]*$/.test(record.preset)), `${source}: preset must be null or an identifier`);
  assert(record.recipe === null || (typeof record.recipe === "string" && /^[a-z][a-z0-9-]*$/.test(record.recipe)), `${source}: recipe must be null or an identifier`);
  assert(Array.isArray(record.argv) && record.argv.every((value) => typeof value === "string"), `${source}: argv must be a string array`);
  assert(record.argv.length > 0, `${source}: argv is empty`);
  for (const field of ["settings", "environment"]) {
    assert(record[field] && typeof record[field] === "object" && !Array.isArray(record[field]), `${source}: ${field} must be a mapping`);
    for (const [name, entry] of Object.entries(record[field])) {
      const where = `${source}: ${field}.${name}`;
      assert(entry && typeof entry === "object" && !Array.isArray(entry), `${where} must be a mapping of value and source`);
      assert(JSON.stringify(Object.keys(entry).sort()) === JSON.stringify(["source", "value"]), `${where} must carry exactly value and source`);
      assert(typeof entry.source === "string" && entry.source.trim(), `${where}.source must name the layer the value came from`);
      validateValue(entry.value, where);
    }
  }
  assert(Array.isArray(record.warnings) && record.warnings.every((value) => typeof value === "string"), `${source}: warnings must be a string array`);
  assert(record.cache_service === null || (typeof record.cache_service === "object"), `${source}: cache_service must be null or an object`);
  assert(["server", "cache", "proxy", "probe"].includes(record.role), `${source}: role must be one of server, cache, proxy, probe`);
  assert(record.selection && typeof record.selection === "object" && !Array.isArray(record.selection), `${source}: selection is required`);
  for (const field of ["recipe", "profile", "preset", "hardware"]) {
    assert(Object.hasOwn(record.selection, field), `${source}: selection lacks ${field}`);
  }
  assert(record.topology && typeof record.topology === "object" && !Array.isArray(record.topology), `${source}: topology is required`);
  assert(["single", "lws"].includes(record.topology.kind), `${source}: topology.kind must be single or lws`);
  assert(Number.isSafeInteger(record.topology.nodes) && record.topology.nodes > 0, `${source}: topology.nodes must be a positive integer`);
  assert(Number.isSafeInteger(record.topology.rank) && record.topology.rank >= 0, `${source}: topology.rank must be a non-negative integer`);
  assert(record.topology.leader === null || typeof record.topology.leader === "string", `${source}: topology.leader must be null or a string`);
  assert(Number.isSafeInteger(record.topology.probe_port) && record.topology.probe_port > 0, `${source}: topology.probe_port must be a port`);
  for (const field of ["lil_runtime", "builder"]) {
    assert(record.resolver_version && COMMIT.test(record.resolver_version[field]), `${source}: resolver_version.${field} must be a full commit OID`);
  }
  assertNoPrivateReference(JSON.stringify(record), source);
  return record;
}

export function validateConfigs(document, source, declarations) {
  assert(document && typeof document === "object" && !Array.isArray(document), `${source}: configs.json must be an object`);
  const keys = Object.keys(document);
  assert(keys.length > 0, `${source}: configs.json lists no selections`);
  for (const key of keys) {
    assert(SELECTOR.test(key), `${source}: ${key} is not a selection key (recipe:<name> or profile:<name>#<hardware>[#<preset>])`);
    validateRecord(document[key], key);
    // The key and the record must agree, or a stale cache reads as a live one.
    const record = document[key];
    const [kind, ...parts] = key.split("#");
    const [prefix, name] = kind.split(":");
    if (prefix === "recipe") {
      assert(record.recipe === name && record.selection.recipe === name, `${source}: ${key} carries recipe ${record.recipe}`);
      assert(record.topology.rank === 0, `${source}: ${key} must be the rank-zero configuration`);
      const declared = declarations ? declarations[name] : null;
      if (declared) {
        // A record that no longer matches the recipe file that produced it is a
        // stale cache, and step 3 would render defaults nobody owns.
        for (const field of ["profile", "hardware", "preset"]) {
          assert(record[field] === declared[field], `${source}: ${key} resolved ${field} ${record[field]}, recipe ${name} declares ${declared[field]}`);
        }
        assert(record.topology.kind === declared.kind, `${source}: ${key} resolved a ${record.topology.kind} topology, recipe ${name} declares ${declared.kind}`);
        assert(record.topology.nodes === declared.nodes, `${source}: ${key} resolved ${record.topology.nodes} nodes, recipe ${name} declares ${declared.nodes}`);
      }
    } else {
      assert(record.profile === name, `${source}: ${key} carries profile ${record.profile}`);
      assert(record.hardware === parts[0], `${source}: ${key} carries hardware ${record.hardware}`);
      assert(record.preset === (parts[1] ?? null), `${source}: ${key} carries preset ${record.preset}`);
      assert(record.selection.profile === name && record.selection.hardware === parts[0], `${source}: ${key} disagrees with its own selection`);
      assert(record.recipe === null && record.selection.recipe === null, `${source}: ${key} is not a recipe selection but carries recipe ${record.recipe}`);
    }
  }
  for (const name of REQUIRED_RECIPES) assert(Object.hasOwn(document, `recipe:${name}`), `${source}: no configuration for recipe ${name}`);
  return document;
}

export function validateOptions(document, source, recipeValues) {
  assert(document && typeof document === "object" && !Array.isArray(document), `${source}: options.json must be an object`);
  assert(
    JSON.stringify(Object.keys(document).sort()) === JSON.stringify(["options", "parameter_docs", "recipe_docs", "schema_version"].sort()),
    `${source}: options.json must carry exactly schema_version, options, parameter_docs, recipe_docs`,
  );
  assert(document.schema_version === 1, `${source}: unexpected schema_version ${document.schema_version}`);

  assert(document.options && typeof document.options === "object" && !Array.isArray(document.options), `${source}: options must be a mapping`);
  const names = Object.keys(document.options);
  assert(names.length > 0, `${source}: upstream options.yaml documents no options`);
  for (const name of names) {
    const spec = document.options[name];
    const where = `${source}: options.${name}`;
    assert(spec && typeof spec === "object" && !Array.isArray(spec), `${where} must be a mapping`);
    assert(typeof spec.type === "string" && spec.type.trim(), `${where} needs a type`);
    assert(Array.isArray(spec.env) && spec.env.every((value) => typeof value === "string" && value), `${where}.env must be a non-empty string array`);
  }

  assert(document.parameter_docs && typeof document.parameter_docs === "object", `${source}: parameter_docs must be a mapping`);
  const docs = document.parameter_docs;
  assert(docs.schema_version === 1, `${source}: parameter_docs has unexpected schema_version ${docs.schema_version}`);
  for (const field of ["groups", "options", "environment"]) {
    assert(docs[field] && typeof docs[field] === "object" && !Array.isArray(docs[field]), `${source}: parameter_docs.${field} must be a mapping`);
  }
  // Upstream carries `why` only where a value needs justification beyond its
  // summary, so it is optional in this tree and required in recipe docs, where
  // the whole point is explaining what this repository changed.
  for (const [field, entries, managed] of [["options", docs.options, true], ["environment", docs.environment, false]]) {
    for (const [name, entry] of Object.entries(entries)) {
      const where = `${source}: parameter_docs.${field}.${name}`;
      if (managed) assert(Object.hasOwn(document.options, name), `${where} documents an option upstream does not manage`);
      for (const key of ["group", "summary"]) assert(typeof entry[key] === "string" && entry[key].trim(), `${where} needs ${key}`);
      assert(entry.why === undefined || (typeof entry.why === "string" && entry.why.trim()), `${where}.why must be absent or a non-empty string`);
      assert(Object.hasOwn(docs.groups, entry.group), `${where}.group ${entry.group} is not a declared group`);
    }
  }
  for (const [name, group] of Object.entries(docs.groups)) {
    const where = `${source}: parameter_docs.groups.${name}`;
    assert(/^[a-z][a-z0-9-]*$/.test(name), `${where} is not a group id`);
    assert(group && typeof group.title === "string" && group.title.trim(), `${where} needs a title`);
  }

  assert(document.recipe_docs && typeof document.recipe_docs === "object" && !Array.isArray(document.recipe_docs), `${source}: recipe_docs must be a mapping`);
  for (const [recipe, entries] of Object.entries(document.recipe_docs)) {
    const where = `${source}: recipe_docs.${recipe}`;
    assert(/^[a-z][a-z0-9-]*$/.test(recipe), `${where} is not a recipe name`);
    assert(entries && typeof entries === "object" && !Array.isArray(entries), `${where} must be a mapping`);
    const set = recipeValues ? recipeValues[recipe] : null;
    if (set) {
      const unknown = Object.keys(entries).filter((name) => !set.includes(name)).sort();
      assert(unknown.length === 0, `${where} documents ${unknown.join(", ")}, which the recipe never sets`);
    }
    for (const [name, entry] of Object.entries(entries)) {
      const field = `${where}.${name}`;
      assert(!Object.hasOwn(docs.options, name), `${field} duplicates documentation upstream already carries`);
      assert(!Object.hasOwn(docs.environment, name), `${field} duplicates documentation upstream already carries`);
      for (const key of DOC_FIELDS) assert(typeof entry[key] === "string" && entry[key].trim(), `${field} needs ${key}`);
      assert(Object.hasOwn(docs.groups, entry.group), `${field}.group ${entry.group} is not a declared group`);
    }
    if (set) {
      const documented = new Set([...Object.keys(docs.options), ...Object.keys(docs.environment), ...Object.keys(entries)]);
      const missing = set.filter((name) => !documented.has(name)).sort();
      assert(missing.length === 0, `${where} sets ${missing.join(", ")} with no documentation in parameter-docs.yaml or docs`);
    }
  }
  for (const name of REQUIRED_RECIPES) assert(Object.hasOwn(document.recipe_docs, name), `${source}: recipe ${name} supplies no docs`);
  assertNoPrivateReference(JSON.stringify(document), source);
  return document;
}

// ------------------------------------------------------------- upstream tree

function run(command, args, options) {
  return execFileSync(command, args, { encoding: "utf8", maxBuffer: 64 * 1024 * 1024, ...options });
}

/** Where the merged policy tree for a pin is staged inside a work directory. */
export function mergedRuntimeRoot(workRoot, commit) {
  return join(workRoot, `merged-${digest(commit)}`, "runtime");
}

/**
 * Depth-1 checkout of the exact pinned commit, then the merged policy root:
 * upstream's `runtime/` with this repository's hardware profiles landed beside
 * its own, which is what the image installs and what the tests resolve against.
 */
export function prepareRuntime(workRoot, pinned) {
  const checkout = join(workRoot, `blackwell-llm-docker-${pinned.lilRuntime.slice(0, 12)}`);
  const upstream = join(checkout, "runtime");
  if (!existsSync(join(upstream, "schema.json"))) {
    mkdirSync(workRoot, { recursive: true });
    rmSync(checkout, { recursive: true, force: true });
    // Fetch the commit object itself, never a branch, so the tree cannot move
    // under the build between the fetch and the checkout.
    run("git", ["init", "--quiet", checkout], { cwd: workRoot });
    run("git", ["-C", checkout, "remote", "add", "origin", pinned.lilRemote], { cwd: workRoot });
    run("git", ["-C", checkout, "fetch", "--depth", "1", "--no-tags", "origin", pinned.lilRuntime], { cwd: workRoot, timeout: 600_000 });
    run("git", ["-C", checkout, "checkout", "--quiet", "--detach", "FETCH_HEAD"], { cwd: workRoot });
    assert(existsSync(join(upstream, "schema.json")), `${checkout}: the pinned commit carries no runtime/schema.json`);
    const head = run("git", ["-C", checkout, "rev-parse", "HEAD"], { cwd: workRoot }).trim();
    assert(head === pinned.lilRuntime, `${checkout}: fetched ${head}, expected the pinned ${pinned.lilRuntime}`);
  }
  const root = mergedRuntimeRoot(workRoot, pinned.lilRuntime);
  rmSync(dirname(root), { recursive: true, force: true });
  mkdirSync(root, { recursive: true });
  for (const name of POLICY_FILES) copyFileSync(join(upstream, name), join(root, name));
  for (const name of POLICY_DIRECTORIES) {
    const source = join(upstream, name);
    assert(existsSync(source), `${upstream}: the pinned commit has no ${name}/ directory`);
    // Recursive, so a policy file added in a subdirectory can never be
    // silently left out of the tree the resolver reads.
    const walk = (directory, prefix) => {
      mkdirSync(join(root, prefix), { recursive: true });
      for (const entry of readdirSync(directory).sort()) {
        const path = join(directory, entry);
        if (statSync(path).isDirectory()) walk(path, `${prefix}/${entry}`);
        else copyFileSync(path, join(root, `${prefix}/${entry}`));
      }
    };
    walk(source, name);
  }
  mkdirSync(join(root, "hardware"), { recursive: true });
  for (const name of listYaml(join(REPOSITORY_ROOT, "image_tools/data/hardware"))) {
    const path = join(REPOSITORY_ROOT, "image_tools/data/hardware", `${name}.yaml`);
    assert(!existsSync(join(root, "hardware", `${name}.yaml`)), `hardware ${name}: this repository would hide an upstream profile`);
    copyFileSync(path, join(root, "hardware", `${name}.yaml`));
  }
  return root;
}

// ------------------------------------------------------------------- driving

/**
 * The resolver's environment, built from nothing but this list.
 *
 * `--print-config` reports the origin of every value, and a variable the
 * process inherits is reported as `cli:environment` with its value. Inheriting
 * the operator's or the runner's shell would therefore publish workstation
 * state — internal service URLs, and anything a `SECRET`-shaped name does not
 * already redact — into a public artefact. The build is hermetic instead: the
 * child sees only what a deployment manifest would give it.
 */
function launchEnvironment(runtimeRoot, topology, groupSize) {
  const environment = {
    PATH: process.env.PATH,
    LANG: "C.UTF-8",
    PYTHONIOENCODING: "utf-8",
    PYTHONDONTWRITEBYTECODE: "1",
    PYTHONNOUSERSITE: "1",
    PYTHONPATH: REPOSITORY_ROOT,
    VLLM_IMAGE_DATA_ROOT: runtimeRoot,
    VLLM_IMAGE_RECIPE_ROOT: RECIPE_ROOT,
  };
  // One pod template branches on the rank the controller injects, so an LWS
  // recipe is printed as rank zero: the leader configuration the site renders.
  if (topology === "lws") {
    environment.LWS_WORKER_INDEX = "0";
    environment.LWS_GROUP_SIZE = String(groupSize);
    environment.LWS_LEADER_ADDRESS = "$(LWS_LEADER_ADDRESS)";
    environment.POD_IP = "192.0.2.10";
  }
  return environment;
}

function resolveOne(runtimeRoot, entry) {
  let recipe = null;
  let groupSize = 1;
  if (entry.recipe) {
    const data = readYaml(recipePath(entry.recipe), `recipe ${entry.recipe}`);
    assert(data.launch && typeof data.launch === "object", `recipe ${entry.recipe}: no launch section`);
    recipe = data.launch;
    groupSize = recipe.topology?.kind === "lws" ? recipe.topology.nodes : 1;
  }
  const environment = launchEnvironment(runtimeRoot, recipe?.topology?.kind ?? "single", groupSize);
  let stdout;
  try {
    stdout = run("python3", ["-m", "image_tools.vllm_image", "launch", ...entry.argv, "--print-config"], {
      cwd: REPOSITORY_ROOT,
      env: environment,
      stdio: ["ignore", "pipe", "pipe"],
      timeout: 300_000,
    });
  } catch (error) {
    const detail = String(error.stderr || error.stdout || error.message).trim().split("\n").pop();
    assert(false, `${entry.key}: the launcher refused the selection: ${detail}`);
  }
  let record;
  try {
    record = JSON.parse(stdout);
  } catch (error) {
    assert(false, `${entry.key}: the launcher did not print JSON: ${String(stdout).slice(0, 200)}`);
  }
  return record;
}

function builderCommit() {
  if (process.env.GITHUB_SHA && COMMIT.test(process.env.GITHUB_SHA)) return process.env.GITHUB_SHA;
  const head = run("git", ["rev-parse", "HEAD"], { cwd: REPOSITORY_ROOT }).trim();
  assert(COMMIT.test(head), `the build is not anchored to a commit: ${head}`);
  return head;
}

function parseArguments(argv) {
  const options = {
    configs: join(PUBLIC_ROOT, "configs.json"),
    optionsOutput: join(PUBLIC_ROOT, "options.json"),
    fixtureDir: null,
    runtimeDir: null,
    workDir: process.env.RUNNER_TEMP ? join(process.env.RUNNER_TEMP, "recipes-site-data") : join(process.env.HOME || ".", ".cache/vllm-multiarch-oci/pages-data"),
  };
  for (let index = 0; index < argv.length; index += 1) {
    const value = (name) => {
      assert(argv[index + 1], `--${name} needs a value`);
      index += 1;
      return argv[index];
    };
    switch (argv[index]) {
      case "--fixture-dir":
        options.fixtureDir = value("fixture-dir");
        break;
      case "--runtime-dir":
        options.runtimeDir = value("runtime-dir");
        break;
      case "--work-dir":
        options.workDir = value("work-dir");
        break;
      case "--configs":
        options.configs = value("configs");
        break;
      case "--options":
        options.optionsOutput = value("options");
        break;
      default:
        assert(false, `unknown argument ${argv[index]}`);
    }
  }
  return options;
}

function writeJson(path, value) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`);
}

/** The names of every recipe this repository ships. */
export function recipeNamesFromDisk() {
  const names = [...recipeFiles(RECIPE_ROOT).keys()].sort();
  for (const name of REQUIRED_RECIPES) recipePath(name);
  return names;
}

/**
 * What each recipe sets, and the documentation it supplies for the values
 * upstream does not document. Step 3 of the flow renders `summary` and `why`
 * verbatim, and the coverage rule below is what keeps an undocumented value
 * from reaching the site as an unexplained number.
 */
export function recipeInventory(names = recipeNamesFromDisk()) {
  const docs = {};
  const values = {};
  const declarations = {};
  for (const name of names) {
    const data = readYaml(recipePath(name), `recipe ${name}`);
    assert(data.launch && typeof data.launch === "object" && !Array.isArray(data.launch), `recipe ${name}: no launch section`);
    docs[name] = data.docs ?? {};
    values[name] = [...new Set([...Object.keys(data.launch.options ?? {}), ...Object.keys(data.launch.environment ?? {})])];
    declarations[name] = {
      profile: data.launch.profile,
      hardware: data.launch.hardware,
      preset: data.launch.preset ?? null,
      kind: data.launch.topology?.kind ?? "single",
      nodes: data.launch.topology?.nodes ?? 1,
    };
  }
  return { docs, values, declarations };
}

/** options.json: upstream's parameter documentation plus this repository's. */
export function policyDocument(runtimeRoot, names = recipeNamesFromDisk()) {
  const { docs, values } = recipeInventory(names);
  const document = {
    schema_version: 1,
    options: readYaml(join(runtimeRoot, "options.yaml"), "options.yaml"),
    parameter_docs: readYaml(join(runtimeRoot, "parameter-docs.yaml"), "parameter-docs.yaml"),
    recipe_docs: docs,
  };
  return validateOptions(document, `${runtimeRoot}: policy data`, values);
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const pinned = pinnedSources();

  if (options.fixtureDir) {
    // The pull-request path: committed documents, no network, no interpreter.
    const dir = locate(options.fixtureDir);
    const inventory = recipeInventory();
    for (const [name, target] of [
      ["configs.json", options.configs],
      ["options.json", options.optionsOutput],
    ]) {
      const source = join(dir, name);
      const document = JSON.parse(readFileSync(source, "utf8"));
      if (name === "options.json") validateOptions(document, source, inventory.values);
      else validateConfigs(document, source, inventory.declarations);
      writeJson(target, document);
    }
    process.stdout.write(`${options.configs}: fixtures from ${dir}\n`);
    return;
  }

  const runtimeRoot = options.runtimeDir ? locate(options.runtimeDir) : prepareRuntime(options.workDir, pinned);
  assert(existsSync(join(runtimeRoot, "options.yaml")), `${runtimeRoot}: no options.yaml, the policy tree is incomplete`);

  const recipeNames = recipeNamesFromDisk();
  const version = { lil_runtime: pinned.lilRuntime, builder: builderCommit() };
  const configs = {};
  for (const entry of selections(runtimeRoot, recipeNames)) {
    const record = resolveOne(runtimeRoot, entry);
    record.resolver_version = version;
    configs[entry.key] = validateRecord(record, entry.key);
  }
  validateConfigs(configs, "resolved selections", recipeInventory(recipeNames).declarations);
  writeJson(options.configs, configs);
  const document = policyDocument(runtimeRoot, recipeNames);
  writeJson(options.optionsOutput, document);
  process.stdout.write(
    `${options.configs}: ${Object.keys(configs).length} selections against ${pinned.lilRuntime.slice(0, 12)} (${Object.keys(document.recipe_docs).length} recipes, ${Object.values(document.recipe_docs).reduce((total, entries) => total + Object.keys(entries).length, 0)} documented recipe values)\n`,
  );
}

if (process.argv[1] && import.meta.url === `file://${resolve(process.argv[1])}`) {
  await main();
}
