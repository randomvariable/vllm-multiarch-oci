import { mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parse as parseYaml } from "yaml";

// The deployment-flow artefacts are produced by these two scripts, so the same
// validators gate them here: one rule, checked at generation and again at build.
import { assertNoPrivateReference, RELEASE_TAG, validateReleases } from "./resolve-releases.mjs";
import { recipeInventory, validateConfigs, validateOptions } from "./resolve-configs.mjs";
// The same shared field set the renderer and the flow read, so a recipe is gated on
// the definitions it will actually be shown -- not on a copy of them.
import { SITE_PARAMETERS, siteParameterDefinitions } from "../src/data/site-parameters.js";

const SITE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const REPOSITORY_ROOT = resolve(SITE_ROOT, "..");
const RECIPES_ROOT = join(REPOSITORY_ROOT, "recipes");
const PUBLIC_ROOT = join(SITE_ROOT, "public");
const RUNTIME_CONFIGURATION_SOURCE = join(REPOSITORY_ROOT, "docs/reference/vllmb12x-runtime-configuration.md");
const RUNTIME_CONFIGURATION_PAGE = join(SITE_ROOT, "src/content/docs/reference/vllmb12x-runtime-configuration.md");
const BUILDS_DIRECTORY = join(SITE_ROOT, "src/content/docs/builds");
const COMMIT = /^[0-9a-f]{40}$/;
const DIGEST_REFERENCE = /^[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?(?:\/[a-z0-9]+(?:[._-][a-z0-9]+)*)+@sha256:[0-9a-f]{64}$/;
const DNS_LABEL = /^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$/;
const IMMUTABLE_TAG = /^vllmb12x-[a-z0-9][a-z0-9-]*-[0-9a-f]{12}-[0-9a-f]{12}-[0-9]{8}-n[1-9][0-9]*$/;
// One file per deployment: the launcher's sections and the site's sections in
// one document, so a deployment cannot be described twice. `launch`, `docs` and
// `benchmark` are read by `vllm-image launch` and the recipe page; the rest is
// what the site renders.
const TOP_LEVEL_KEYS = ["meta", "model", "runtime", "launch", "docs", "deployment", "benchmark", "validation", "guide"];
// Mirrored from image_tools/launcher/launch.py. A recipe carrying a key the
// launcher refuses, or missing one it requires, would pass the site build and
// fail at container start; `model_sync` is on this list's absence rather than
// its presence because that block was the duplicate copy of `model` and
// `deployment`, and a recipe must not grow one back.
const LAUNCH_KEYS = ["profile", "hardware", "preset", "options", "environment", "topology", "probe_port"];
const TOPOLOGY_KEYS = ["kind", "nodes", "rendezvous_port", "kv_events", "replica_port_base"];
const KV_EVENT_KEYS = ["publisher", "endpoint", "replay_endpoint"];
// The checkpoint facts, which live in `deployment` exactly together or not at
// all: their presence is what distinguishes model sync from engine download.
const SYNC_DEPLOYMENT_KEYS = ["storage_root", "model_path", "storage_min_free_gib", "download_workers", "ignore_patterns", "required_files"];
const DOC_KEYS = ["group", "summary", "why"];
const TABLE_KEYS = ["id", "title", "aria_label", "columns", "rows"];
const CELL_KEYS = ["text"];
const CELL_OPTIONAL_KEYS = ["unit", "primary", "notes"];
const GUIDE_KEYS = ["hardware", "fixed_tuning", "networking", "safety"];
const VALIDATION_STATUSES = new Set(["verified", "unverified"]);
const RECIPE_NAME = /^[a-z][a-z0-9-]*$/;
const PARAMETER_TYPES = new Set(["string", "integer", "stringMap"]);
const REQUIRED_DEPENDENCY_FIELDS = ["id", "name", "version", "source", "owner", "depends_on", "readiness", "tested"];

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function recipeFiles(directory) {
  const paths = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) paths.push(...await recipeFiles(path));
    else if (entry.isFile() && /\.ya?ml$/.test(entry.name)) paths.push(path);
  }
  return paths.sort();
}

function mapping(value, where) {
  assert(value && typeof value === "object" && !Array.isArray(value), `${where}: must be a mapping`);
  return value;
}

function keysAre(value, required, where, optional = []) {
  const present = Object.keys(mapping(value, where));
  const missing = required.filter((key) => !present.includes(key));
  const extra = present.filter((key) => !required.includes(key) && !optional.includes(key));
  assert(missing.length === 0, `${where}: missing ${missing.join(", ")}`);
  assert(extra.length === 0, `${where}: unexpected ${extra.join(", ")}`);
  return value;
}

function nonEmptyString(value, where) {
  assert(typeof value === "string" && value.trim(), `${where}: must be a non-empty string`);
  return value;
}

function positiveInteger(value, where) {
  assert(Number.isSafeInteger(value) && value > 0, `${where}: must be a positive integer`);
  return value;
}

function stringList(value, where, { nonEmpty = false } = {}) {
  assert(
    Array.isArray(value) && (!nonEmpty || value.length > 0) && value.every((entry) => typeof entry === "string" && entry),
    `${where}: must be ${nonEmpty ? "a non-empty" : "a"} string array`,
  );
  return value;
}

function portOrNull(value, where) {
  assert(value === null || (Number.isSafeInteger(value) && value >= 1 && value <= 65535), `${where}: must be a port or null`);
  return value;
}

// `launch` is what the container reads, so its shape is the launcher's own: the
// exact key set of image_tools/launcher/launch.py, checked here rather than at
// pod start. `model_sync` is refused by being absent from the list, because that
// block restated `model` and `deployment` and is what this consolidation deleted.
function validateLaunch(launch, source) {
  const where = `${source}: launch`;
  keysAre(launch, LAUNCH_KEYS, where);
  nonEmptyString(launch.profile, `${where}.profile`);
  nonEmptyString(launch.hardware, `${where}.hardware`);
  mapping(launch.options, `${where}.options`);
  const environment = mapping(launch.environment, `${where}.environment`);
  for (const [name, value] of Object.entries(environment)) {
    assert(typeof value === "string", `${where}.environment.${name} must be a string`);
  }
  const topology = keysAre(launch.topology, TOPOLOGY_KEYS, `${where}.topology`);
  assert(["single", "lws"].includes(topology.kind), `${where}.topology.kind must be single or lws`);
  positiveInteger(topology.nodes, `${where}.topology.nodes`);
  if (topology.kind === "lws") assert(topology.nodes >= 2, `${where}: an lws topology needs at least two nodes`);
  for (const field of ["rendezvous_port", "replica_port_base"]) portOrNull(topology[field], `${where}.topology.${field}`);
  if (topology.kv_events !== null) {
    const events = keysAre(topology.kv_events, KV_EVENT_KEYS, `${where}.topology.kv_events`);
    for (const field of KV_EVENT_KEYS) nonEmptyString(events[field], `${where}.topology.kv_events.${field}`);
  }
  portOrNull(launch.probe_port, `${where}.probe_port`);
  return launch;
}

// A docs entry explains a value this recipe itself sets, beyond what upstream's
// parameter-docs.yaml already carries. The completeness half of the rule -- every
// value must be documented somewhere -- is resolve-configs.mjs's job, because only
// it has the upstream tree to compare against.
function validateDocs(docs, launch, source) {
  const set = new Set([...Object.keys(launch.options), ...Object.keys(launch.environment)]);
  for (const [name, entry] of Object.entries(mapping(docs, `${source}: docs`))) {
    assert(set.has(name), `${source}: docs.${name} documents a value the recipe never sets`);
    keysAre(entry, DOC_KEYS, `${source}: docs.${name}`);
    for (const field of DOC_KEYS) nonEmptyString(entry[field], `${source}: docs.${name}.${field}`);
  }
}

// The measured panel, in data rather than in markup so a page cannot show a
// figure the recipe does not carry. `primary`, `notes` and `degraded` are how the
// panel marked the emphasised column, the small suffix and the incomplete wave.
function validateBenchmark(benchmark, source) {
  const where = `${source}: benchmark`;
  keysAre(benchmark, ["context", "tables", "notes"], where, ["tool", "version"]);
  for (const field of ["tool", "version"]) {
    if (benchmark[field] !== undefined) nonEmptyString(benchmark[field], `${where}.${field}`);
  }
  stringList(benchmark.context, `${where}.context`, { nonEmpty: true });
  assert(Array.isArray(benchmark.tables) && benchmark.tables.length > 0, `${where}: at least one table is required`);
  const ids = new Set();
  for (const table of benchmark.tables) {
    keysAre(table, TABLE_KEYS, `${where}.table`);
    nonEmptyString(table.id, `${where}.table.id`);
    assert(!ids.has(table.id), `${where}: table id ${table.id} is declared twice`);
    ids.add(table.id);
    for (const field of ["title", "aria_label"]) nonEmptyString(table[field], `${where}.${table.id}.${field}`);
    const columns = stringList(table.columns, `${where}.${table.id}.columns`, { nonEmpty: true });
    assert(Array.isArray(table.rows) && table.rows.length > 0, `${where}.${table.id}: at least one row is required`);
    for (const row of table.rows) {
      keysAre(row, ["cells"], `${where}.${table.id}.row`, ["degraded"]);
      assert(row.degraded === undefined || typeof row.degraded === "boolean", `${where}.${table.id}.row.degraded must be a boolean`);
      assert(Array.isArray(row.cells) && row.cells.length === columns.length, `${where}.${table.id}: a row must carry exactly ${columns.length} cells`);
      for (const cell of row.cells) {
        keysAre(cell, CELL_KEYS, `${where}.${table.id}.cell`, CELL_OPTIONAL_KEYS);
        nonEmptyString(cell.text, `${where}.${table.id}.cell.text`);
        if (cell.unit !== undefined) nonEmptyString(cell.unit, `${where}.${table.id}.cell.unit`);
        if (cell.primary !== undefined) assert(typeof cell.primary === "boolean", `${where}.${table.id}.cell.primary must be a boolean`);
        if (cell.notes !== undefined) stringList(cell.notes, `${where}.${table.id}.cell.notes`, { nonEmpty: true });
      }
    }
  }
  assert(Array.isArray(benchmark.notes) && benchmark.notes.length > 0, `${where}: the method and its caveats belong in notes`);
  for (const note of benchmark.notes) {
    keysAre(note, ["body"], `${where}.note`, ["title"]);
    nonEmptyString(note.body, `${where}.note.body`);
    if (note.title !== undefined) nonEmptyString(note.title, `${where}.note.title`);
  }
}

// The checkpoint facts are stated once, in `model` and `deployment`, and their
// presence together is what distinguishes the two ways a container can get its
// weights. With a published root the launcher syncs it before exec; with none the
// engine resolves the repository id itself, and no storage field may pretend
// otherwise.
function validateDeploymentAndLaunch(recipe, launch, source) {
  const where = `${source}: deployment`;
  const deployment = recipe.deployment;
  positiveInteger(deployment.nodes, `${where}.nodes`);
  positiveInteger(deployment.tensor_parallel_size, `${where}.tensor_parallel_size`);
  assert(deployment.nodes === launch.topology.nodes, `${where}.nodes must be launch.topology.nodes (${launch.topology.nodes})`);
  const options = launch.options;
  if (options["tensor-parallel-size"] !== undefined) {
    assert(options["tensor-parallel-size"] === deployment.tensor_parallel_size, `${where}.tensor_parallel_size must be launch.options['tensor-parallel-size'] (${options["tensor-parallel-size"]})`);
  }
  nonEmptyString(options.model, `${source}: launch.options.model must name what the engine loads`);
  const sync = SYNC_DEPLOYMENT_KEYS.filter((field) => Object.hasOwn(deployment, field));
  assert(sync.length === 0 || sync.length === SYNC_DEPLOYMENT_KEYS.length, `${where}: the storage fields come together; ${sync.join(", ")} without the rest`);
  if (sync.length === 0) {
    assert(options.model === recipe.model.model_id, `${where}: an engine-download recipe serves model.model_id, not a published path`);
    return;
  }
  nonEmptyString(deployment.storage_root, `${where}.storage_root`);
  assert(deployment.model_path.startsWith("/"), `${where}.model_path must be absolute`);
  assert(
    deployment.model_path === `${deployment.storage_root}/${recipe.model.served_name}`,
    `${where}.model_path must be the storage root plus the served name (${deployment.storage_root}/${recipe.model.served_name})`,
  );
  positiveInteger(deployment.storage_min_free_gib, `${where}.storage_min_free_gib`);
  positiveInteger(deployment.download_workers, `${where}.download_workers`);
  stringList(deployment.ignore_patterns, `${where}.ignore_patterns`);
  stringList(deployment.required_files, `${where}.required_files`, { nonEmpty: true });
  assert(options.model === deployment.model_path, `${where}: the engine must load the snapshot the launcher publishes`);
}

// The manifest cluster shape the renderer reads from `deployment`. Every value here is
// what the live cluster carries and the renderer used to hardcode; requiring it in the
// recipe is what keeps the generated manifest free of a hand-written cluster literal.
// "Absent is meaningful": an omitted tolerations/capabilities/securityContext key renders
// as no key, so a recipe that declares one must declare it with content -- a no-op empty
// list or empty map is refused rather than emitted.
const PROBE_WINDOW_KEYS = ["periodSeconds", "timeoutSeconds", "failureThreshold"];
const RESOURCE_KEYS = ["nvidia.com/gpu", "rdma"];

function validateProbeWindow(window, where) {
  keysAre(window, PROBE_WINDOW_KEYS, where);
  for (const key of PROBE_WINDOW_KEYS) positiveInteger(window[key], `${where}.${key}`);
}

function validateResourceMap(map, where, { nonEmpty = false } = {}) {
  const value = mapping(map, where);
  const keys = Object.keys(value);
  if (nonEmpty) assert(keys.length > 0, `${where}: must name at least one resource`);
  for (const [name, quantity] of Object.entries(value)) {
    // The GPU and RDMA counts are the renderer's to inject from deployment.gpu and the
    // rdma parameters; a recipe that also wrote one into a map could make requests and
    // limits disagree. Extended resources are exactly the slashed names, so reject those
    // here: the remaining rules (cpu, memory, ephemeral-storage) are free to differ
    // between the two maps.
    assert(!name.includes("/"), `${where}.${name}: the GPU and RDMA resources come from deployment.gpu and deployment.rdma, not from a resource map`);
    assert(typeof quantity === "string" && quantity.trim(), `${where}.${name}: must be a non-empty resource quantity string`);
    assert(!RESOURCE_KEYS.includes(name), `${where}.${name}: reserved resource`);
  }
  return value;
}

function validateDeploymentCluster(recipe, launch, source) {
  const where = `${source}: deployment`;
  const deployment = recipe.deployment;
  const isLws = launch.topology.kind === "lws";

  positiveInteger(deployment.gpu, `${where}.gpu`);
  if (deployment.rdma !== undefined) assert(typeof deployment.rdma === "boolean", `${where}.rdma must be a boolean`);

  const resources = mapping(deployment.resources, `${where}.resources`);
  keysAre(resources, ["requests"], `${where}.resources`, ["limits"]);
  validateResourceMap(resources.requests, `${where}.resources.requests`, { nonEmpty: true });
  if (resources.limits !== undefined) validateResourceMap(resources.limits, `${where}.resources.limits`);

  if (deployment.nodeSelector !== undefined) {
    const selector = mapping(deployment.nodeSelector, `${where}.nodeSelector`);
    for (const [key, value] of Object.entries(selector)) assert(typeof value === "string", `${where}.nodeSelector.${key} must be a string`);
  }

  // A tolerations key that is present must be a real list; the empty-list form is a
  // no-op the scheduler ignores, so tell the author to omit the key instead.
  if (deployment.tolerations !== undefined) {
    assert(Array.isArray(deployment.tolerations) && deployment.tolerations.length > 0, `${where}.tolerations: omit the key to declare no tolerations; do not carry an empty list`);
    for (const [index, entry] of deployment.tolerations.entries()) {
      const toleration = mapping(entry, `${where}.tolerations[${index}]`);
      keysAre(toleration, ["key", "operator", "value", "effect"], `${where}.tolerations[${index}]`);
      for (const field of ["key", "operator", "value", "effect"]) nonEmptyString(toleration[field], `${where}.tolerations[${index}].${field}`);
    }
  }

  if (deployment.securityContext !== undefined) {
    const context = mapping(deployment.securityContext, `${where}.securityContext`);
    for (const field of ["runAsUser", "runAsGroup"]) {
      if (context[field] !== undefined) assert(Number.isSafeInteger(context[field]) && context[field] >= 0, `${where}.securityContext.${field} must be a non-negative integer`);
    }
    if (context.capabilities !== undefined) {
      const capabilities = mapping(context.capabilities, `${where}.securityContext.capabilities`);
      keysAre(capabilities, [], `${where}.securityContext.capabilities`, ["add", "drop"]);
      const add = capabilities.add ?? [];
      const drop = capabilities.drop ?? [];
      assert(add.length + drop.length > 0, `${where}.securityContext.capabilities: omit it to declare no capabilities; do not carry an empty object`);
      for (const field of ["add", "drop"]) if (capabilities[field] !== undefined) stringList(capabilities[field], `${where}.securityContext.capabilities.${field}`, { nonEmpty: true });
    }
  }

  if (deployment.podSecurityContext !== undefined) {
    const context = mapping(deployment.podSecurityContext, `${where}.podSecurityContext`);
    assert(Object.keys(context).length > 0, `${where}.podSecurityContext: omit the key to declare no pod security context`);
  }

  const volumes = mapping(deployment.volumes, `${where}.volumes`);
  keysAre(volumes, ["hostPathType"], `${where}.volumes`);
  nonEmptyString(volumes.hostPathType, `${where}.volumes.hostPathType`);

  if (isLws) {
    positiveInteger(deployment.groups, `${where}.groups`);
    if (deployment.restartPolicy !== undefined) nonEmptyString(deployment.restartPolicy, `${where}.restartPolicy`);
    const rollout = mapping(deployment.rollout, `${where}.rollout`);
    keysAre(rollout, ["type", "maxUnavailable", "maxSurge", "partition"], `${where}.rollout`);
    nonEmptyString(rollout.type, `${where}.rollout.type`);
    for (const field of ["maxUnavailable", "maxSurge", "partition"]) {
      assert(Number.isSafeInteger(rollout[field]) && rollout[field] >= 0, `${where}.rollout.${field} must be a non-negative integer`);
    }
    // An lws group must name its startup and readiness windows explicitly: the whole
    // point is that no probe budget is a renderer literal. A zero or missing field in a
    // declared window is refused by validateProbeWindow.
    const probes = mapping(deployment.probes, `${where}.probes`);
    keysAre(probes, ["startup", "readiness"], `${where}.probes`, ["liveness"]);
    validateProbeWindow(probes.startup, `${where}.probes.startup`);
    validateProbeWindow(probes.readiness, `${where}.probes.readiness`);
    if (probes.liveness !== undefined) {
      // A liveness window has to be MEASURED against the worst /livez response during a
      // long prefill on both ranks; an unmeasured one is the failure mode that killed
      // healthy groups, so it is only valid when the recipe says it was measured.
      assert(probes.liveness.measured === true, `${where}.probes.liveness: refuse an unmeasured liveness window -- omit the block until /livez is measured on both ranks`);
      const window = { ...probes.liveness };
      delete window.measured;
      validateProbeWindow(window, `${where}.probes.liveness`);
    }
  } else {
    // A single-node Deployment may rely on the accepted launcher-form defaults, but if
    // it names windows they must be complete.
    if (deployment.probes !== undefined) {
      const probes = mapping(deployment.probes, `${where}.probes`);
      keysAre(probes, [], `${where}.probes`, ["startup", "readiness", "liveness"]);
      for (const field of ["startup", "readiness"]) if (probes[field] !== undefined) validateProbeWindow(probes[field], `${where}.probes.${field}`);
      if (probes.liveness !== undefined) {
        assert(probes.liveness.measured === true, `${where}.probes.liveness: refuse an unmeasured liveness window -- omit the block until /livez is measured`);
        const window = { ...probes.liveness };
        delete window.measured;
        validateProbeWindow(window, `${where}.probes.liveness`);
      }
    }
  }
}

function validateRecipe(recipe, source) {
  assert(recipe && typeof recipe === "object" && !Array.isArray(recipe), `${source}: recipe must be a mapping`);
  assert(JSON.stringify(Object.keys(recipe)) === JSON.stringify(TOP_LEVEL_KEYS), `${source}: top-level keys must be exactly ${TOP_LEVEL_KEYS.join(", ")}`);
  // The route, the selection string `--recipe` takes and the file's stem are one
  // name: load_recipe resolves a name across recipes/** by stem, so a slug that
  // drifted from its file would publish a page for a recipe the launcher cannot
  // find and a command line that no longer matches the page.
  const name = basename(source).replace(/\.ya?ml$/, "");
  assert(RECIPE_NAME.test(name), `${source}: recipe name ${name} is not a lower-case identifier`);
  assert(typeof recipe.meta.title === "string" && typeof recipe.meta.description === "string", `${source}: meta title and description are required`);
  assert(recipe.meta.slug === name, `${source}: meta.slug must be ${name}, the file stem the launcher resolves`);
  assert(typeof recipe.model.model_id === "string" && COMMIT.test(recipe.model.revision) && typeof recipe.model.served_name === "string", `${source}: model id, full commit OID and served name are required`);
  assert(recipe.model.serves_native_messages === undefined || typeof recipe.model.serves_native_messages === "boolean", `${source}: model.serves_native_messages must be a boolean`);
  assert(Array.isArray(recipe.runtime.command) && recipe.runtime.command.every((value) => typeof value === "string"), `${source}: runtime.command must be a string array`);
  assert(Array.isArray(recipe.runtime.base_args) && recipe.runtime.base_args.every((value) => typeof value === "string"), `${source}: runtime.base_args must be a string array`);
  assert(recipe.runtime.base_env && typeof recipe.runtime.base_env === "object" && !Array.isArray(recipe.runtime.base_env), `${source}: runtime.base_env must be a mapping`);
  assert(Object.values(recipe.runtime.base_env).every((value) => typeof value === "string"), `${source}: runtime.base_env values must be strings`);
  assert(recipe.runtime.leader_args === undefined || (Array.isArray(recipe.runtime.leader_args) && recipe.runtime.leader_args.every((value) => typeof value === "string")), `${source}: runtime.leader_args must be a string array`);
  const leaderPorts = recipe.runtime.leader_ports ?? [];
  assert(Array.isArray(leaderPorts) && leaderPorts.every((entry) => typeof entry?.name === "string" && DNS_LABEL.test(entry.name) && entry.name.length <= 15 && Number.isSafeInteger(entry.containerPort) && entry.containerPort > 0 && entry.containerPort <= 65535), `${source}: runtime.leader_ports entries need a DNS-label name of at most 15 characters (IANA_SVC_NAME) and a valid containerPort`);
  if (recipe.runtime.leader_args) {
    // The KV-event topic is the only pod identity the router indexes blocks
    // under, so its address:port and model must be the ones the engine actually
    // serves. A mismatch is silent: every prefix match reads zero.
    const portIndex = recipe.runtime.base_args.indexOf("--port");
    const nameIndex = recipe.runtime.base_args.indexOf("--served-model-name");
    assert(portIndex >= 0 && nameIndex >= 0, `${source}: a publishing recipe must declare --port and --served-model-name`);
    const topic = `:${recipe.runtime.base_args[portIndex + 1]}@${recipe.runtime.base_args[nameIndex + 1]}`;
    assert(recipe.runtime.leader_args.some((value) => value.includes('"enable_kv_cache_events":true') && value.includes(topic)), `${source}: the KV-event topic must carry the serving port and the served model name (${topic})`);
    assert(leaderPorts.some((entry) => entry.name === "kv-events"), `${source}: a publishing recipe must declare the kv-events container port`);
  }
  const launch = validateLaunch(recipe.launch, source);
  validateDocs(recipe.docs, launch, source);
  validateBenchmark(recipe.benchmark, source);
  validateDeploymentAndLaunch(recipe, launch, source);
  validateDeploymentCluster(recipe, launch, source);
  assert(recipe.model.served_name === launch.options["served-model-name"], `${source}: model.served_name and launch.options['served-model-name'] must be one value`);
  if (launch.options.revision !== undefined) {
    // The engine may pin the snapshot it downloads, but it cannot pin a different
    // revision from the one the recipe names: one checkout, one identity.
    assert(String(launch.options.revision) === recipe.model.revision, `${source}: launch.options.revision must be model.revision`);
  }
  // The fields a deployment asks its operator for come from the shared set for its
  // shape; the recipe below it overrides values only. Validating the MERGED result is
  // what makes that safe: an override cannot drop a label, a type or an explanation,
  // and a name that exists in neither the shared set nor the recipe is a typo that
  // would silently never reach the form.
  if (recipe.deployment.parameters !== undefined && recipe.deployment.parameters !== null) {
    assert(typeof recipe.deployment.parameters === "object" && !Array.isArray(recipe.deployment.parameters), `${source}: deployment.parameters must be a mapping`);
  }
  const shape = launch.topology.kind === "lws" ? "multi" : "single";
  for (const name of Object.keys(recipe.deployment.parameters ?? {})) {
    assert(name in SITE_PARAMETERS[shape], `${source}: ${name} is not a ${shape}-node site parameter; a recipe overrides a field its own shape asks for and never invents one`);
  }
  for (const [name, parameter] of Object.entries(siteParameterDefinitions(recipe))) {
    assert(typeof parameter.label === "string" && PARAMETER_TYPES.has(parameter.type) && typeof parameter.required === "boolean" && Object.hasOwn(parameter, "default") && typeof parameter.description === "string", `${source}: malformed parameter ${name}`);
    if (parameter.type === "string") assert(typeof parameter.default === "string", `${source}: ${name} default must be a string`);
    if (parameter.type === "integer") assert(Number.isSafeInteger(parameter.default), `${source}: ${name} default must be an integer`);
    if (parameter.type === "stringMap") assert(parameter.default && typeof parameter.default === "object" && !Array.isArray(parameter.default) && Object.values(parameter.default).every((value) => typeof value === "string"), `${source}: ${name} default must be a string mapping`);
    if (parameter.help) {
      assert(typeof parameter.help.label === "string" && typeof parameter.help.href === "string" && !parameter.help.href.startsWith("/"), `${source}: ${name} help must have a label and site-relative href`);
    }
    if (parameter.suggestions) {
      assert(parameter.type === "string" && Array.isArray(parameter.suggestions) && parameter.suggestions.length > 0 && parameter.suggestions.every((value) => typeof value === "string" && value), `${source}: ${name} suggestions must be non-empty strings for a string parameter`);
    }
  }
  keysAre(recipe.guide, GUIDE_KEYS, `${source}: guide`);
  for (const field of GUIDE_KEYS) nonEmptyString(recipe.guide[field], `${source}: guide.${field}`);
  // A status is allowed to claim only what an image can be pointed at. `image` is
  // required and public exactly when something was verified, and must be absent
  // rather than empty or null otherwise, so a later edit cannot flip a status to
  // `verified` around a placeholder digest.
  mapping(recipe.validation, `${source}: validation`);
  for (const field of ["kubernetes", "docker"]) {
    assert(VALIDATION_STATUSES.has(recipe.validation[field]), `${source}: validation.${field} must be verified or unverified`);
  }
  const accepted = recipe.validation.kubernetes === "verified" || recipe.validation.docker === "verified";
  if (accepted) {
    nonEmptyString(recipe.validation.image, `${source}: a verified recipe must pin the image it was verified on`);
    assert(DIGEST_REFERENCE.test(recipe.validation.image) && !recipe.validation.image.includes("internal.randomvariable"), `${source}: validation image must be a public digest-qualified reference`);
  } else {
    assert(recipe.validation.image === undefined, `${source}: an unverified recipe must not name an image`);
  }
  nonEmptyString(recipe.validation.evidence, `${source}: validation.evidence must record what was accepted`);
  assertNoPrivateReference(JSON.stringify(recipe), source);
  return recipe;
}

function validateLatest(latest) {
  assert(latest && typeof latest === "object" && !Array.isArray(latest), "latest-image.json must be an object");
  assert(typeof latest.repository === "string" && !latest.repository.includes("@") && !latest.repository.includes("internal.randomvariable"), "latest image repository must be public and unqualified");
  assert(typeof latest.tag === "string" && IMMUTABLE_TAG.test(latest.tag), "latest image tag must follow the immutable vllmb12x publication contract");
  assert(typeof latest.reference === "string" && DIGEST_REFERENCE.test(latest.reference), "latest image reference must be digest-qualified");
  assert(latest.reference.startsWith(`${latest.repository}@sha256:`), "latest image reference must belong to repository");
  assert(typeof latest.resolved_at === "string" && !Number.isNaN(Date.parse(latest.resolved_at)), "latest image resolved_at must be an ISO timestamp");
  return latest;
}

function validateDependencies(data) {
  assert(data && typeof data === "object" && Array.isArray(data.dependencies), "platform-dependencies.yaml must contain dependencies");
  const ids = new Set();
  for (const dependency of data.dependencies) {
    for (const field of REQUIRED_DEPENDENCY_FIELDS) assert(Object.hasOwn(dependency, field), `dependency ${dependency.id || "<unknown>"} lacks ${field}`);
    assert(typeof dependency.id === "string" && !ids.has(dependency.id), `dependency id is missing or duplicated: ${dependency.id}`);
    ids.add(dependency.id);
    assert(typeof dependency.version === "string" && dependency.version.trim(), `dependency ${dependency.id} lacks a version pin`);
    assert(typeof dependency.source === "string" && /^https:\/\//.test(dependency.source), `dependency ${dependency.id} lacks an HTTPS source`);
    assert(typeof dependency.readiness === "string" && dependency.readiness.trim(), `dependency ${dependency.id} lacks a readiness command`);
    assert(Array.isArray(dependency.depends_on) && dependency.depends_on.every((value) => typeof value === "string"), `dependency ${dependency.id} has invalid edges`);
    assert(typeof dependency.tested === "boolean", `dependency ${dependency.id} must record tested status`);
  }
  for (const dependency of data.dependencies) for (const edge of dependency.depends_on) assert(ids.has(edge), `dependency ${dependency.id} references unknown dependency ${edge}`);
  return data;
}

// The runtime configuration inventory is generated from the pinned vLLM and
// B12X sources by scripts/vllmb12x-runtime-config.py, so the site publishes
// that file rather than keeping a second copy that can drift from the lock.
async function publishRuntimeConfiguration() {
  const source = await readFile(RUNTIME_CONFIGURATION_SOURCE, "utf8");
  const [heading, ...body] = source.split("\n");
  assert(heading === "# VLLMB12X Runtime Configuration", `${RUNTIME_CONFIGURATION_SOURCE}: unexpected heading ${heading}`);
  const frontmatter = [
    "---",
    "title: VLLMB12X runtime configuration",
    "description: Source-backed descriptions and values for runtime controls added or changed by the pinned local-inference-lab vLLM and B12X sources.",
    "---",
  ].join("\n");
  await writeFile(RUNTIME_CONFIGURATION_PAGE, `${frontmatter}\n${body.join("\n")}`);
}

// The three artefacts the deployment flow renders from. Each is produced by a
// script in this directory and re-checked here, so a hand-edited or stale file
// under public/ cannot reach the Pages build.
export async function publishDeploymentData(publicRoot = PUBLIC_ROOT) {
  const { values, declarations } = recipeInventory();
  const artefacts = [
    ["releases.json", (document, source) => validateReleases(document, source)],
    ["configs.json", (document, source) => validateConfigs(document, source, declarations)],
    ["options.json", (document, source) => validateOptions(document, source, values)],
  ];
  const documents = {};
  for (const [name, validate] of artefacts) {
    const source = join(publicRoot, name);
    const document = JSON.parse(await readFile(source, "utf8"));
    validate(document, source);
    assertNoPrivateReference(JSON.stringify(document), source);
    documents[name.replace(/\.json$/, "")] = document;
  }
  return documents;
}

function cell(value) {
  // Table cells are the only place a release's own text can break the document:
  // a pipe would end the row and a newline would end the table.
  return String(value ?? "not recorded").replace(/\|/g, "\\|").replace(/\s+/g, " ").trim();
}

function buildPage(release, repository) {
  const code = (value) => "`" + value + "`";
  const short = (value) => code(String(value).slice(0, 12));
  const pins = [
    ["Digest", code(release.digest)],
    ["Publication tag", code(release.publication_tag)],
    ["vLLM", short(release.vllm_commit)],
    ["B12X", short(release.b12x_commit)],
    ["Runtime data", short(release.lil_runtime_commit)],
    ["Builder", short(release.builder_commit)],
  ];
  const frontmatter = [
    "---",
    `title: "${release.tag} build"`,
    `description: "The ${release.tag} publication: its digest, the revisions it was built from and the changes it carries."`,
    "editUrl: false",
    "---",
  ].join("\n");
  const lines = [
    frontmatter,
    `Published ${release.published_at.slice(0, 10)}.`,
    "",
    "Every build is assembled from pinned revisions of `local-inference-lab/vLLM` and its",
    "dependencies. These are not generic upstream vLLM images, and the digest identifies",
    "one build rather than a moving tag.",
    "",
    "## What this build was made from",
    "",
    "| Field | Value |",
    "| --- | --- |",
    ...pins.map(([label, value]) => `| ${label} | ${value} |`),
    "",
    "A field that reads `not recorded` predates the pin that would fill it. Treat it as",
    "unknown rather than as matching the current source.",
    "",
    "## Carried by this build",
    "",
    ...(release.highlights.length
      ? release.highlights.map((entry) => `- ${cell(entry)}`)
      : ["No additions were recorded for this build."]),
    "",
    "## Included upstream changes",
    "",
    "The image is built from pinned fork revisions rather than from upstream branches, so",
    "this records which upstream changes the current lock carries.",
    "",
    ...(release.included_changes.length
      ? [
          "| Component | Change | Included as |",
          "| --- | --- | --- |",
          ...release.included_changes.map(
            (change) =>
              `| ${cell(change.component)} | ${change.url ? `[${cell(change.change)}](${change.url})` : cell(change.change)} | ${cell(change.included_as)} |`,
          ),
        ]
      : ["No upstream changes were recorded for this build."]),
    "",
    "## Run this build",
    "",
    "```bash",
    `docker pull ${repository}@${release.digest}`,
    "```",
    "",
    "The [deployment flow](../../) selects a build, a model and its settings, then renders",
    "the manifest, the Docker command and the Compose file. Opening it with this build",
    "keeps every other choice at its default.",
    "",
  ];
  return `${lines.join("\n")}\n`;
}

// One page per published build, generated from releases.json the same way the
// runtime configuration inventory is generated from its source: the site would
// otherwise keep a second copy of the release record that can drift from the
// GitHub data it came from. Markdown rather than MDX because a release body is
// operator-written text, and `{` or `<` in it would be parsed as an expression.
//
// Each release becomes builds/<route>/index.md, where the route is the tag with
// its dot replaced by a dash. Astro removes dots from a route segment entirely,
// so builds/v20261003.1/index.md would be served at /builds/v202610031/ and the
// URL would read as a different build. A dash survives, and the page still names
// the exact tag in its title, heading and pull command.
export function buildRoute(tag) {
  assert(RELEASE_TAG.test(tag), `${tag}: not a vYYYYMMDD.N build tag`);
  return tag.replace(/\./g, "-");
}

export async function publishBuildPages(document, repository, directory = BUILDS_DIRECTORY) {
  await mkdir(directory, { recursive: true });
  const written = new Set();
  for (const release of document.releases) {
    const route = buildRoute(release.tag);
    written.add(route);
    const target = join(directory, route);
    await mkdir(target, { recursive: true });
    await writeFile(join(target, "index.md"), buildPage(release, repository));
  }
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    if (entry.isDirectory() && !written.has(entry.name)) {
      await rm(join(directory, entry.name), { recursive: true });
    }
  }
}

async function main() {
  const recipes = [];
  for (const path of await recipeFiles(RECIPES_ROOT)) recipes.push(validateRecipe(parseYaml(await readFile(path, "utf8")), path));
  assert(recipes.length > 0, "no recipes were found");
  const slugs = recipes.map((recipe) => recipe.meta.slug);
  assert(new Set(slugs).size === slugs.length, "recipe slugs must be unique");
  const latest = validateLatest(JSON.parse(await readFile(join(PUBLIC_ROOT, "latest-image.json"), "utf8")));
  validateDependencies(parseYaml(await readFile(join(SITE_ROOT, "src/data/platform-dependencies.yaml"), "utf8")));
  const documents = await publishDeploymentData();
  await publishBuildPages(documents.releases, latest.repository);
  const document = { recipes };
  assertNoPrivateReference(JSON.stringify(document), "recipes.json");
  await writeFile(join(PUBLIC_ROOT, "recipes.json"), `${JSON.stringify(document, null, 2)}\n`);
}

// Importable by the tests without regenerating the site.
if (process.argv[1] && import.meta.url === `file://${resolve(process.argv[1])}`) {
  await main();
}
