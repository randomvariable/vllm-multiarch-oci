import { stringify as stringifyYaml } from "yaml";
import { SITE_PARAMETERS, siteParameterDefinitions } from "./data/site-parameters.js";

// The site renders the LAUNCHER form of a deployment: every container runs
// `/opt/venv/bin/vllm-image launch ...`, never a hand-written `vllm serve`. That
// keeps the manifest and the image agreeing by construction, because the same
// selection the container resolves from `launch.py` is the selection the manifest
// carries. Nothing here re-implements a resolved default: the launcher resolves
// it, so a value the reader did not edit is absent from the output. The rules that
// already live in `image_tools/launcher/launch.py` (argument order, native-argument
// handling, the cache-mode decision, the stop grace) are mirrored below with a
// comment naming the source, so a `pnpm test` can prove the two have not drifted.

const DIGEST_REFERENCE = /^[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?(?:\/[a-z0-9]+(?:[._-][a-z0-9]+)*)+@sha256:[0-9a-f]{64}$/;
const DNS_LABEL = /^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$/;
const DNS_SUBDOMAIN = /^[a-z0-9](?:[-a-z0-9.]*[a-z0-9])?$/;
const RESOURCE_NAME = /^(?:[a-z0-9](?:[-a-z0-9.]*[a-z0-9])?\/)?[A-Za-z0-9](?:[-A-Za-z0-9_.]*[A-Za-z0-9])?$/;
const SAFE_INTERFACE = /^[A-Za-z0-9_.:,=-]+$/;
const ENV_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;

// Two selections reach this module. A validated recipe carries `--recipe <name>`,
// because a bare `--profile/--hardware` would resolve WITHOUT that recipe's option
// and environment layer, which is a different deployment. An upstream profile
// selection carries `--profile/--hardware/--preset` exactly as `launch.py.parse()`
// takes them, because there is no recipe file for the launcher to read; its
// deployment shape is inferred from the resolved record (see `profileRecipe`).
const TARGET_FIELDS = {
  lws: [
    "namespace", "name", "model_storage_path", "jit_storage_path", "hf_secret", "hf_secret_key",
    "network_attachment", "node_selector", "topology_key", "topology_values",
    "rdma_resource", "rdma_units", "hca", "gid_index",
  ],
  docker: ["name", "model_storage_path", "jit_storage_path", "hca", "gid_index", "control_interface", "leader_ip", "worker_ip"],
  compose: ["name", "model_storage_path", "jit_storage_path", "hca", "gid_index", "control_interface"],
  routing: ["namespace", "name", "gateway_class"],
};
const TARGET_KEYS = Object.keys(TARGET_FIELDS);

// The site fields each manifest target dereferences unconditionally. The renderer
// refuses rather than emit `undefined` names or host paths, so a field is only ever
// demanded when the output actually reads it: `collective` fields belong to a
// deployment whose ranks form one group, `rdma` fields to one that opens the RoCE
// device. A single-node pod has neither, and asking its operator for a Multus
// attachment, a topology label, an HCA list and an RDMA resource -- four values
// that never reach its manifest -- would be a form full of noise.
const REQUIRED_PRESENT = {
  lws: ["namespace", "name", "model_storage_path", "jit_storage_path", "hf_secret", "hf_secret_key", "network_attachment", "topology_key", "topology_values", "rdma_resource", "rdma_units", "hca", "gid_index"],
  docker: ["name", "model_storage_path", "jit_storage_path", "hca", "gid_index", "control_interface"],
  compose: ["name", "model_storage_path", "jit_storage_path", "hca", "gid_index", "control_interface"],
  routing: ["namespace", "name", "gateway_class"],
};

// Each field's scope; a field with no entry is asked of every shape.
const FIELD_SCOPES = {
  network_attachment: "collective",
  topology_key: "collective",
  topology_values: "collective",
  control_interface: "collective",
  leader_ip: "collective",
  worker_ip: "collective",
  rdma_resource: "rdma",
  rdma_units: "rdma",
  hca: "rdma",
  gid_index: "rdma",
};

// The group's ranks form one engine: more than one pod, so the collective network,
// the topology label and the host interfaces all matter.
function isCollective(recipe) {
  return recipe?.launch?.topology?.kind === "lws";
}

function usesRdma(recipe) {
  return Boolean(recipe?.deployment?.rdma);
}

function fieldInScope(recipe, field) {
  const scope = FIELD_SCOPES[field] ?? "always";
  if (scope === "collective") return isCollective(recipe);
  if (scope === "rdma") return usesRdma(recipe);
  return true;
}

function scopedFields(recipe, fields) {
  return fields.filter((field) => fieldInScope(recipe, field));
}

// Ports the launcher falls back to when the recipe states none; `launch.py` defines
// DEFAULT_RENDEZVOUS_PORT = 25000 and DEFAULT_PROBE_PORT = 8890.
const DEFAULT_PORT = 8888;
const DEFAULT_RENDEZVOUS_PORT = 25000;
const DEFAULT_PROBE_PORT = 8890;

// The httpGet probe windows the accepted single-node Deployment falls back to when a
// recipe declares none. These are the launcher-form windows already carried for the
// accepted GB10 groups (see the TP=2 recipe `deployment.probes`); a recipe that names
// its own windows overrides them, so no cluster value is invented here -- it is the
// value the accepted manifests already used.
const DEFAULT_STARTUP_PROBE = { periodSeconds: 30, timeoutSeconds: 5, failureThreshold: 240 };
const DEFAULT_READINESS_PROBE = { periodSeconds: 15, timeoutSeconds: 5, failureThreshold: 6 };
// The engine stop budget the accepted manifests carry, plus the cache stop grace
// from `resolver.py`: STOP_GRACE_SECONDS = 10 and CHECKPOINT_SHUTDOWN_FLUSH_SECONDS
// = 30. The pod must outlive a graceful engine drain and a RAM-checkpoint flush.
const ENGINE_STOP_GRACE_SECONDS = 120;
const CACHE_STOP_GRACE_SECONDS = 40;

// A `--` argument is the reader's edit verbatim; the launcher resolves everything
// else. `launch.py.parse()` collects them as `request.native` and hands them to
// `policy.resolve(argv=native)`, so the shape here is a vLLM CLI fragment: a
// dotted or JSON-valued flag is one `--key=value` token (as the parity harness's
// EDITED_CASES exercises), a scalar is the pair `--key value`.
function changeArgs(settings) {
  const args = [];
  for (const [name, raw] of Object.entries(settings ?? {})) {
    if (!hasValue(raw)) continue;
    const value = String(raw);
    if (name.includes(".") || /^[{[]/.test(value)) args.push(`--${name}=${value}`);
    else args.push(`--${name}`, value);
  }
  return args;
}

function fail(message) {
  throw new TypeError(message);
}

function shellQuote(value) {
  return "'" + String(value).replaceAll("'", "'\"'\"'") + "'";
}

function shellCommand(argv) {
  return argv.map(shellQuote).join(" ");
}

function yamlDocuments(documents) {
  return documents.map((document) => stringifyYaml(document, { lineWidth: 0 })).join("---\n");
}

function hasValue(value, type) {
  if (type === "stringMap") return value && typeof value === "object" && !Array.isArray(value) && Object.keys(value).length > 0;
  return value !== undefined && value !== null && String(value).trim() !== "";
}

// The launcher form the manifest is built from. `launch.py.Recipe.from_dict` is the
// authority: it reads `launch` (its keys are exact: profile, hardware, preset,
// options, environment, topology, probe_port), `model`, and `deployment`, and a
// deployment is model-sync mode exactly when `deployment.model_path` is set.
function validateRecipe(recipe) {
  if (!recipe || typeof recipe !== "object" || Array.isArray(recipe)) fail("recipe must be an object");
  if (!recipe.meta || typeof recipe.meta !== "object") fail("recipe.meta is required");
  const launch = recipe.launch;
  if (!launch || typeof launch !== "object" || Array.isArray(launch)) fail("recipe.launch is required");
  for (const key of ["profile", "hardware"]) {
    if (typeof launch[key] !== "string" || !launch[key]) fail(`launch.${key} must be a non-empty string`);
  }
  if (launch.preset !== null && launch.preset !== undefined && typeof launch.preset !== "string") fail("launch.preset must be a string or null");
  for (const key of ["options", "environment"]) {
    if (!launch[key] || typeof launch[key] !== "object" || Array.isArray(launch[key])) fail(`launch.${key} must be a mapping`);
  }
  if (!Object.values(launch.environment).every((value) => typeof value === "string")) fail("launch.environment values must be strings");
  const topology = launch.topology;
  if (!topology || typeof topology !== "object" || Array.isArray(topology)) fail("launch.topology is required");
  if (!["single", "lws"].includes(topology.kind)) fail("launch.topology.kind must be single or lws");
  if (!Number.isSafeInteger(topology.nodes) || topology.nodes < 1) fail("launch.topology.nodes must be a positive integer");
  if (topology.kind === "lws" && topology.nodes < 2) fail("an lws topology needs at least two nodes");
  if (topology.kind === "single" && topology.nodes !== 1) fail("a single topology is one node");
  if (launch.probe_port !== undefined && (!Number.isSafeInteger(launch.probe_port) || launch.probe_port < 1 || launch.probe_port > 65535)) fail("launch.probe_port must be a port");
  if (!recipe.model || typeof recipe.model !== "object") fail("recipe.model is required");
  const served = servedModelName(recipe, {});
  if (!hasValue(served, "string")) fail("the served model name is required for routing and the model header");
}

// A supplied value, or the definition's own default, checked against the
// definition's type. The `required` check belongs to the caller, because required
// is per target: the same field is asked of the lws output and ignored by compose.
function parameterValue(name, definition, supplied) {
  if (!["string", "integer", "stringMap"].includes(definition.type)) fail(`invalid parameter type for ${name}`);
  const value = Object.hasOwn(supplied, name) ? supplied[name] : definition.default;
  if (definition.type === "integer" && hasValue(value, definition.type) && (!Number.isSafeInteger(Number(value)) || Number(value) < 0)) fail(`${name} must be a non-negative integer`);
  if (definition.type === "stringMap" && value !== undefined && (value === null || typeof value !== "object" || Array.isArray(value) || !Object.values(value).every((entry) => typeof entry === "string"))) fail(`${name} must be a string mapping`);
  return definition.type === "integer" && hasValue(value, definition.type) ? Number(value) : value;
}

function normalizedParameters(recipe, supplied, target) {
  if (!(target in TARGET_FIELDS)) fail(`unsupported render target: ${target}`);
  const definitions = siteParameterDefinitions(recipe);
  const values = {};
  for (const [name, definition] of Object.entries(definitions)) {
    const value = parameterValue(name, definition, supplied);
    if (scopedFields(recipe, TARGET_FIELDS[target]).includes(name) && definition.required && !hasValue(value, definition.type)) fail(`${name} is required for ${target}`);
    values[name] = value;
  }
  if (hasValue(values.name, "string") && !DNS_LABEL.test(values.name)) fail("name must be a DNS label");
  if (hasValue(values.namespace, "string") && !DNS_LABEL.test(values.namespace)) fail("namespace must be a DNS label");
  for (const field of ["hf_secret", "network_attachment", "gateway_class"]) {
    if (hasValue(values[field], "string") && !DNS_SUBDOMAIN.test(values[field])) fail(`${field} must be a DNS subdomain`);
  }
  if (hasValue(values.hf_secret_key, "string") && !/^[A-Za-z0-9._-]+$/.test(values.hf_secret_key)) fail("hf_secret_key is not a valid Secret data key");
  for (const path of [values.model_storage_path, values.jit_storage_path].filter(Boolean)) {
    if (!String(path).startsWith("/")) fail("storage paths must be absolute");
  }
  for (const field of ["rdma_resource", "topology_key"]) {
    if (hasValue(values[field], "string") && !RESOURCE_NAME.test(values[field])) fail(`${field} is not a valid Kubernetes resource or label name`);
  }
  for (const field of ["hca", "control_interface"]) {
    if (hasValue(values[field], "string") && !SAFE_INTERFACE.test(values[field])) fail(`${field} contains unsupported characters`);
  }
  // A manifest target dereferences these site fields directly (names, mounts, the
  // Secret reference, the resource names), so the output cannot be produced without
  // them. Their definitions come from the shared set for this deployment's shape
  // rather than from the recipe, which is why a recipe with `deployment.parameters:
  // {}` still renders: the fields exist by construction and only the defaults are
  // authored.
  for (const field of scopedFields(recipe, REQUIRED_PRESENT[target])) {
    if (!hasValue(values[field], "string")) fail(`${field} is required for the ${target} target but this deployment has no answer for it`);
  }
  return values;
}

// The fields this deployment asks the operator for and has no usable answer for
// yet. `deployment-flow.js` drives its "Fill in ..." guidance from here so the
// panel never shows a raw renderer error for a value the reader simply has not
// typed: one rule about what counts as required, in one place.
export function requiredSiteFields(recipe, supplied = {}) {
  const definitions = siteParameterDefinitions(recipe);
  const asked = new Set(TARGET_KEYS.flatMap((target) => scopedFields(recipe, TARGET_FIELDS[target])));
  const missing = [];
  for (const [name, definition] of Object.entries(definitions)) {
    if (!asked.has(name) || !definition.required) continue;
    let value;
    try {
      value = parameterValue(name, definition, supplied);
    } catch {
      // A malformed answer is an unanswered one; the renderer still reports what
      // is wrong with it when the field is filled in.
      missing.push({ name, label: definition.label ?? name });
      continue;
    }
    if (!hasValue(value, definition.type)) missing.push({ name, label: definition.label ?? name });
  }
  return missing;
}

function validateImageReference(imageReference) {
  if (typeof imageReference !== "string" || !DIGEST_REFERENCE.test(imageReference)) fail("image must be a digest-qualified reference");
  if (imageReference.includes("internal.randomvariable")) fail("private registry references are not allowed");
}

// -- launcher selection ---------------------------------------------------------

// The name `--recipe` resolves is `meta.slug`, which `build-data.mjs` keeps equal to
// the file stem the launcher searches for under `recipe_root()`
// (`launch.py.recipe_path` resolves `<name>.yaml` across `recipes/**`). A bare
// `--profile/--hardware` would resolve with no recipe layer, so a recipe selection must
// carry `--recipe <meta.slug>`.
function recipeName(recipe) {
  const name = recipe.meta?.slug;
  if (typeof name !== "string" || !/^[a-z][a-z0-9-]*$/.test(name)) fail("recipe.meta.slug must be a launcher recipe name");
  return name;
}

function selectionArgs(recipe) {
  // A recipe file exists for `--recipe` to load; an upstream profile selection has
  // no file, so it names the three identifiers `launch.py.parse()` takes for that
  // form -- and `--hardware` is required with `--profile` there, which is why the
  // inferred recipe keeps both on `launch` rather than reading them back out of the
  // selection string.
  if (typeof recipe.meta?.slug === "string") return ["--recipe", recipeName(recipe)];
  const args = ["--profile", recipe.launch.profile, "--hardware", recipe.launch.hardware];
  if (recipe.launch.preset) args.push("--preset", recipe.launch.preset);
  return args;
}

// -- inference from a resolved record -------------------------------------------
//
// The deployment shape is a consequence of what the launcher resolved, so the site
// reads it from the record the image produced rather than from a hand-written block
// per selection: node count and tensor-parallel width come from the resolution, the
// topology follows it (a group of pods is an lws, one pod is not), and the RDMA
// device is asked for only when a group actually spans hosts and needs the
// collective. A recipe overrides this -- its `deployment` is what was measured on
// live hardware, down to the device-plugin GPU count, which is not tp/nodes.
//
// What inference does NOT do is invent a number nobody measured. Upstream profile
// selections have never run on a cluster this site documents, so an inferred
// deployment carries no cpu, memory or ephemeral-storage request or limit: the
// container gets its devices and nothing else is claimed, and the flow says so in
// the run panel. The same rule retires a probe or cache CPU reservation for an
// inferred shape.
const INFERRED_SETTINGS = ["port", "tensor-parallel-size", "replicas", "cache-mode", "cache-l1-gib", "served-model-name"];

function settingValue(record, key) {
  const entry = record?.settings?.[key];
  const value = entry?.value;
  return value === undefined || value === null ? null : value;
}

function positiveInt(value, where) {
  const number = Number(value);
  if (!Number.isSafeInteger(number) || number < 1) fail(`${where} must be a positive integer, got ${value}`);
  return number;
}

// The resource name an upstream selection gets by default: its own three identifiers,
// spelled the way Kubernetes wants them. Derived from the selection, so it is a
// label the reader can overwrite in the field rather than a name this page made up.
function selectionName(profile, hardware, preset) {
  const name = [profile, hardware, preset]
    .filter(Boolean)
    .join("-")
    .toLowerCase()
    .replace(/[^a-z0-9-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 57)
    .replace(/-+$/g, "");
  if (!DNS_LABEL.test(name)) fail(`the selection ${[profile, hardware, preset].join("/")} does not spell a DNS label for a deployment name`);
  return name;
}

// The set this deployment's fields come from, carried on the built recipe so the
// renderer, the form and the fill-in guidance all read the same definitions the
// caller supplied. `siteParameterDefinitions()` falls back to the shipped set when a
// recipe -- a file on disk, or a hand-built one in a test -- names none. The merged
// definitions are written back onto the recipe: the object the flow builds its site
// fields from is the object the renderer validates, so the two cannot disagree about
// which fields a deployment has.
function withSiteParameters(recipe, siteParameters) {
  const stamped = { ...recipe, deployment: { ...recipe.deployment, siteParameters } };
  return { ...stamped, deployment: { ...stamped.deployment, parameters: siteParameterDefinitions(stamped) } };
}

export function profileRecipe(record, siteParameters = SITE_PARAMETERS, recipe = null) {
  if (!record || typeof record !== "object" || Array.isArray(record)) fail("a resolved configuration record is required to infer a deployment");
  const selection = record.selection ?? {};
  const profile = selection.profile ?? record.profile;
  const hardware = selection.hardware ?? record.hardware;
  const preset = selection.preset ?? record.preset ?? null;
  if (typeof profile !== "string" || !profile) fail("the record names no model profile to select");
  if (typeof hardware !== "string" || !hardware) fail("a --profile selection needs a hardware profile");
  if (preset !== null && typeof preset !== "string") fail("the record's preset must be a string or null");

  const nodes = positiveInt(record.topology?.nodes, "topology.nodes");
  const tensorParallel = settingValue(record, "tensor-parallel-size");
  if (tensorParallel === null) fail("the record resolved no tensor-parallel-size, so the group width is unknown");
  const width = positiveInt(tensorParallel, "settings['tensor-parallel-size']");
  if (width % nodes !== 0) {
    fail(`tensor-parallel-size ${width} does not divide across ${nodes} nodes; every rank of the group needs a whole GPU`);
  }
  const collective = nodes > 1;
  const served = settingValue(record, "served-model-name");

  const options = {};
  for (const key of INFERRED_SETTINGS) {
    const value = settingValue(record, key);
    if (value !== null) options[key] = value;
  }
  const inferred = {
    meta: {
      title: [profile, hardware, preset].filter(Boolean).join(" · "),
      description: "Inferred from the configuration the image resolved for this selection; not validated on this image.",
    },
    model: served === null ? {} : { served_name: String(served) },
    launch: {
      profile,
      hardware,
      preset,
      options,
      environment: {},
      topology: {
        kind: collective ? "lws" : "single",
        nodes,
        rendezvous_port: record.topology?.rendezvous_port ?? null,
      },
      probe_port: record.topology?.probe_port ?? null,
    },
    deployment: {
      nodes,
      tensor_parallel_size: width,
      // The device count per pod. A recipe replaces this with what its node's
      // device plugin actually reports -- the accepted GB10 groups ask for four
      // time-sliced devices per rank at TP=2 -- which is exactly why the number is
      // inferred only in the absence of a measured one.
      gpu: width / nodes,
      rdma: collective,
      // The image declares no USER and its JIT paths derive from HOME, so a pod
      // that set its own uid dies importing flashinfer; root is what the accepted
      // groups run as and the only setting this site can claim without a
      // measurement of its own.
      securityContext: { runAsUser: 0 },
      volumes: { hostPathType: "DirectoryOrCreate" },
      // The record, not the renderer, is what says whether this layout has a cache:
      // `resolver.py` attaches a cache service exactly when the effective
      // cache-mode is not vram, and the docker refusal has to follow the plan.
      cache_service: Boolean(record.cache_service),
      inferred: true,
      parameters: {},
    },
  };
  if (collective) {
    // One group of `nodes` pods is what the record describes. The rollout is the
    // smallest admissible one -- a group at a time, never a surge, no held
    // partition -- because anything wider is a claim about fleet capacity this
    // site has not measured.
    inferred.deployment.groups = 1;
    inferred.deployment.rollout = { type: "RollingUpdate", maxUnavailable: 1, maxSurge: 0, partition: 0 };
  }

  // The one site field the selection itself can answer is the resource name: an
  // upstream profile takes its profile, hardware and preset, a recipe takes its
  // slug. Anything the recipe declares for its parameters wins over that, so a
  // recipe only ever writes the values that are specific to it.
  const parameters = { ...recipe?.deployment?.parameters ?? {} };
  if (!hasValue(parameters.name?.default, "string")) {
    parameters.name = { ...parameters.name, default: recipe?.meta?.slug ?? selectionName(profile, hardware, preset) };
  }

  if (!recipe) return withSiteParameters({ ...inferred, deployment: { ...inferred.deployment, parameters } }, siteParameters);

  // A validated recipe is the inferred shape plus its measured overrides: every
  // `deployment` field it carries wins, its own `launch` layer and checkpoint facts
  // replace the inferred ones, and its parameter entries override only the values
  // that are specific to it.
  return withSiteParameters(
    {
      ...inferred,
      meta: recipe.meta ?? inferred.meta,
      model: { ...inferred.model, ...recipe.model ?? {} },
      launch: {
        ...inferred.launch,
        ...recipe.launch ?? {},
        options: { ...inferred.launch.options, ...recipe.launch?.options ?? {} },
        environment: { ...recipe.launch?.environment ?? {} },
        topology: { ...inferred.launch.topology, ...recipe.launch?.topology ?? {} },
      },
      deployment: { ...inferred.deployment, ...recipe.deployment ?? {}, parameters },
    },
    siteParameters,
  );
}

// The container path the node's model cache is mounted at is the recipe's
// `deployment.storage_root` (`launch.py.ModelSync` publishes there); engine-download
// recipes state no storage_root and the launcher's HF cache defaults to /models.
function modelMount(recipe, parameters) {
  return `${parameters.model_storage_path}:${recipe.deployment?.storage_root ?? "/models"}`;
}

// A model-sync deployment publishes the checkpoint before the engine loads it, which
// is exactly `launch.py.Recipe.from_dict`: sync mode iff `deployment.model_path` is
// set. Engine-download recipes (no published root) leave the flag off and let vLLM
// resolve the repository id itself.
function modelSync(recipe) {
  const deployment = recipe.deployment;
  return Boolean(deployment && typeof deployment === "object" && hasValue(deployment.model_path, "string"));
}

// The structural ports a manifest needs to name. These are the recipe's declared
// values with the reader's edit layered on top, mirroring the resolver's precedence
// (reader CLI > recipe layer > profile/common default); the container re-resolves
// them from the same `--recipe` selection, so a mismatch is impossible.
function effective(recipe, settings, key, fallback) {
  if (settings && hasValue(settings[key], "string")) return String(settings[key]);
  const option = recipe.launch.options[key];
  if (option !== undefined && option !== null) return String(option);
  return fallback;
}

function servingPort(recipe, settings) {
  return Number(effective(recipe, settings, "port", DEFAULT_PORT));
}

function probePort(recipe) {
  const port = recipe.launch.probe_port;
  return Number.isSafeInteger(port) ? port : DEFAULT_PROBE_PORT;
}

function rendezvousPort(recipe) {
  const port = recipe.launch.topology.rendezvous_port;
  return Number.isSafeInteger(port) ? port : DEFAULT_RENDEZVOUS_PORT;
}

function servedModelName(recipe, settings) {
  const fromSettings = settings && hasValue(settings["served-model-name"], "string") ? String(settings["served-model-name"]) : null;
  const fromOptions = hasValue(recipe.launch?.options?.["served-model-name"], "string") ? String(recipe.launch.options["served-model-name"]) : null;
  return fromSettings ?? fromOptions ?? (hasValue(recipe.model?.served_name, "string") ? String(recipe.model.served_name) : null);
}

// The cache decision mirrors `resolver.py.configure_cache`: the plan carries a
// `cache_service` exactly when the effective `cache-mode` is not `vram` (and no
// shipped recipe defaults `cache-mode`, so in practice it is the reader's edit).
// `vram` is the resolver's default when `cache-mode` is unset.
function effectiveCacheMode(recipe, settings) {
  return effective(recipe, settings, "cache-mode", "vram");
}

function hasCache(recipe, settings) {
  // The reader's own `cache-mode` edit has the last word, exactly as it does in the
  // container. Without one, an inferred upstream layout follows the resolver's
  // answer carried on the record (`record.cache_service`), and a recipe follows its
  // effective cache-mode -- which is what `resolver.py.configure_cache` decides.
  if (settings && hasValue(settings["cache-mode"], "string")) return String(settings["cache-mode"]) !== "vram";
  return recipe.deployment?.cache_service === true || effectiveCacheMode(recipe, settings) !== "vram";
}

// The cache arena and its memory come from the resolved L1, `cache-l1-gib`
// (`resolver.py`: shm_bytes = int(values["cache-l1-gib"] * 1024**3)).
function cacheL1GiB(recipe, settings) {
  const value = effective(recipe, settings, "cache-l1-gib", "");
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? Math.ceil(number) : null;
}

// The replica layout is what turns a single service into several behind a proxy;
// `resolver.py` gates the proxy role on `plan.values["replicas"] > 1`, so a reader
// edit is the only way one appears for these recipes.
function effectiveReplicas(recipe, settings) {
  const value = effective(recipe, settings, "replicas", "1");
  const number = Number(value);
  return Number.isSafeInteger(number) && number > 1 ? number : 1;
}

// -- shared manifest building blocks --------------------------------------------

// The launcher command for one role. The engine (no role) carries `--model-sync` and
// the reader's `--` edits; the probe and cache sidecars carry the same selection with
// `--role` appended and nothing after it — they return from `launch.run()` before the
// engine stages. `launch.py` builds the resolved argv, then appends `topology_args`
// and `--middleware`; the manifest never writes those out, so the site and container
// cannot disagree about them.
function launcherArgs(recipe, settings, { role = null, topology = true } = {}) {
  const args = ["launch", ...selectionArgs(recipe)];
  if (topology) args.push("--topology", recipe.launch.topology.kind);
  if (role) {
    args.push("--role", role);
    return args;
  }
  if (modelSync(recipe)) args.push("--model-sync");
  const changes = changeArgs(settings);
  if (changes.length) args.push("--", ...changes);
  else args.push("--");
  return args;
}

function environmentEdits(environment) {
  return Object.entries(environment ?? {})
    .filter(([name, value]) => ENV_NAME.test(name) && hasValue(value, "string"))
    .map(([name, value]) => ({ name, value: String(value) }));
}

function envList(environment) {
  return Object.entries(environment).map(([name, value]) => ({ name, value: String(value) }));
}

// The site-owned collective-network environment. The image's hardware profile carries
// NCCL transport policy; HCA, GID index and the socket interfaces are per-site facts
// the manifest must inject because no policy file knows the operator's fabric. The
// HF token is never inlined: it comes from a Secret the reader names in `parameters`.
function siteEnvironment(recipe, parameters, controlInterface) {
  const environment = {};
  if (usesRdma(recipe)) {
    environment.NCCL_IB_HCA = parameters.hca;
    environment.NCCL_IB_GID_INDEX = String(parameters.gid_index);
  }
  if (isCollective(recipe)) {
    environment.NCCL_SOCKET_IFNAME = controlInterface;
    environment.GLOO_SOCKET_IFNAME = controlInterface;
  }
  return environment;
}

function hfTokenEnv(parameters) {
  return { name: "HF_TOKEN", valueFrom: { secretKeyRef: { name: parameters.hf_secret, key: parameters.hf_secret_key } } };
}

// The rank identity the launcher reads from `os.environ` in `launch.py.read_rank`:
// the LWS controller stamps these labels onto every pod of a group, so the manifest
// sources them through the downward API rather than hard-coding a per-rank template.
// POD_IP names the pod network address the InferencePool advertises. A single-node
// pod is not in a group: it has no worker index to read and no leader to reach, and
// the accepted live pod for that shape carries none of these, so nothing is injected.
function rankDownwardEnv(recipe) {
  if (!isCollective(recipe)) return [];
  return [
    { name: "LWS_WORKER_INDEX", valueFrom: { fieldRef: { fieldPath: "metadata.labels['leaderworkerset.sigs.k8s.io/worker-index']" } } },
    { name: "LWS_GROUP_SIZE", valueFrom: { fieldRef: { fieldPath: "metadata.labels['leaderworkerset.sigs.k8s.io/group-size']" } } },
    { name: "LWS_LEADER_ADDRESS", valueFrom: { fieldRef: { fieldPath: "metadata.labels['leaderworkerset.sigs.k8s.io/leader-address']" } } },
    { name: "POD_IP", valueFrom: { fieldRef: { fieldPath: "status.podIP" } } },
    { name: "VLLM_HOST_IP", valueFrom: { fieldRef: { fieldPath: "status.podIP" } } },
  ];
}

// -- cluster shape read from `deployment` ---------------------------------------
//
// Every value below is the live cluster's, not the renderer's: the GPU count, the
// resource maps, the container and pod security context, the tolerations, the node
// placement and the hostPath type all come from the recipe's `deployment` block, so
// the generated manifest carries no hand-written cluster literal. A manifest target
// reads them with `fail()` when the recipe omitted one it needs, rather than defaulting
// to a value the operator never chose.
function deploymentSection(recipe) {
  const deployment = recipe.deployment;
  if (!deployment || typeof deployment !== "object" || Array.isArray(deployment)) fail("deployment is required for a manifest target");
  return deployment;
}

// The non-device resources (cpu, memory, ephemeral-storage) come straight from the
// recipe. A slashed extended-resource name is refused here because the GPU and RDMA
// entries are injected from the single sources below, not written into a map: this is
// what makes requests and limits agree by construction rather than by convention.
function resourceMap(map, where) {
  if (map === undefined || map === null) return {};
  if (typeof map !== "object" || Array.isArray(map)) fail(`${where} must be a mapping`);
  const out = {};
  for (const [name, value] of Object.entries(map)) {
    if (name.includes("/")) fail(`${where}.${name}: nvidia.com/gpu and the RDMA resource are set from deployment.gpu and deployment.rdma, not from a resource map`);
    if (value === undefined || value === null || String(value).trim() === "") fail(`${where}.${name} must carry a value`);
    out[name] = String(value);
  }
  return out;
}

// The scheduler admits a pod on the request and the container's device visibility
// comes from the limit, so those two MUST name the same GPU count -- a pod admitted
// for N GPUs and handed M devices is a silent, expensive failure. This is the one
// cross-map rule, and it is enforced structurally: `deployment.gpu` (and, for a
// RoCE recipe, `parameters.rdma_units`) is written into BOTH maps here, so a recipe
// cannot express a disagreement. Nothing else is required to match between them --
// the single-node recipe caps memory and storage below its request, for example.
function gpuAndRdmaResources(deployment, parameters, base) {
  const out = { ...base };
  if (!Number.isSafeInteger(deployment.gpu) || deployment.gpu < 1) fail("deployment.gpu must be a positive integer");
  out["nvidia.com/gpu"] = String(deployment.gpu);
  if (deployment.rdma) out[parameters.rdma_resource] = String(parameters.rdma_units);
  return out;
}

function containerSecurityContext(deployment) {
  const context = deployment.securityContext;
  if (context === undefined || context === null) return undefined;
  if (typeof context !== "object" || Array.isArray(context)) fail("deployment.securityContext must be a mapping");
  const out = {};
  for (const key of ["runAsUser", "runAsGroup", "runAsNonRoot"]) {
    if (context[key] !== undefined && context[key] !== null) out[key] = context[key];
  }
  const capabilities = context.capabilities;
  if (capabilities !== undefined && capabilities !== null) {
    if (typeof capabilities !== "object" || Array.isArray(capabilities)) fail("deployment.securityContext.capabilities must be a mapping");
    const add = capabilities.add ?? [];
    const drop = capabilities.drop ?? [];
    if (!add.length && !drop.length) fail("deployment.securityContext.capabilities is empty; omit it instead of declaring no capabilities");
    out.capabilities = {};
    if (add.length) out.capabilities.add = add;
    if (drop.length) out.capabilities.drop = drop;
  }
  return Object.keys(out).length ? out : undefined;
}

function httpGetProbe(window, path, port, where) {
  const source = window ?? {};
  const probe = { httpGet: { path, port } };
  for (const key of ["periodSeconds", "timeoutSeconds", "failureThreshold"]) {
    const value = source[key];
    if (!Number.isSafeInteger(value) || value < 1) fail(`${where}.${key} must be a positive integer`);
    probe[key] = value;
  }
  return probe;
}

function modelServerContainer(recipe, parameters, imageReference, settings, environment) {
  const port = servingPort(recipe, settings);
  const probe = probePort(recipe);
  const deployment = deploymentSection(recipe);
  const probes = deployment.probes ?? {};
  const env = [
    ...envList(siteEnvironment(recipe, parameters, "eth0")),
    hfTokenEnv(parameters),
    ...rankDownwardEnv(recipe),
    ...environmentEdits(environment),
  ];
  const securityContext = containerSecurityContext(deployment);
  const container = {
    name: "modelserver",
    image: imageReference,
    imagePullPolicy: "IfNotPresent",
    command: ["/opt/venv/bin/vllm-image"],
    args: launcherArgs(recipe, settings),
    env,
    ports: [
      { name: "modelserver", containerPort: port, protocol: "TCP" },
      ...(recipe.launch.topology.kind === "lws" ? [{ name: "rdzv", containerPort: rendezvousPort(recipe), protocol: "TCP" }] : []),
    ],
    resources: {
      requests: gpuAndRdmaResources(deployment, parameters, resourceMap(deployment.resources?.requests, "deployment.resources.requests")),
      limits: gpuAndRdmaResources(deployment, parameters, resourceMap(deployment.resources?.limits, "deployment.resources.limits")),
    },
    // startupProbe and readinessProbe are httpGet on the probe sidecar's /readyz, with
    // the windows the recipe declares (the accepted GB10 groups carry 30/5/240 and
    // 15/5/6). A recipe that declares none -- a single-node Deployment -- takes the
    // accepted launcher-form defaults, so nothing is invented here.
    startupProbe: httpGetProbe(probes.startup ?? DEFAULT_STARTUP_PROBE, "/readyz", probe, "deployment.probes.startup"),
    readinessProbe: httpGetProbe(probes.readiness ?? DEFAULT_READINESS_PROBE, "/readyz", probe, "deployment.probes.readiness"),
    volumeMounts: [
      { name: "shm", mountPath: "/dev/shm" },
      { name: "jit-cache", mountPath: "/cache" },
      // The model cache is the node's persistent /models: the launcher publishes the
      // fetched snapshot there in sync mode and the engine loads it back, and in
      // engine-download mode vLLM resolves the repository id into the same root. It is
      // writable because the same container runs the model-sync phase before exec.
      { name: "model-weights", mountPath: recipe.deployment?.storage_root ?? "/models" },
      { name: "launch-record", mountPath: "/run/vllm-image" },
    ],
  };
  if (securityContext) container.securityContext = securityContext;
  // A livenessProbe is emitted ONLY when the recipe declares one and marks its window
  // measured. The measurement is the worst /livez response during a 1M-token prefill on
  // both ranks; inventing a window is exactly what previously killed healthy groups, so
  // an unmeasured or absent liveness block leaves the key off entirely.
  if (probes.liveness) {
    if (probes.liveness.measured !== true) fail("deployment.probes.liveness declares a window but is not marked measured; omit it until /livez is measured on both ranks");
    container.livenessProbe = httpGetProbe(probes.liveness, "/livez", probe, "deployment.probes.liveness");
  }
  return container;
}

// A native sidecar is an initContainer whose restartPolicy is Always; the acceptance
// rule "no init containers other than native sidecars" means every entry in the pod's
// initContainers carries restartPolicy Always — the old blocking model-sync and
// rendezvous-wait init containers are gone because the launcher does both in-process.
function probeSidecar(recipe, settings) {
  const container = {
    name: "probe",
    image: null, // filled by the pod builder so every container shares the digest
    imagePullPolicy: "IfNotPresent",
    restartPolicy: "Always",
    command: ["/opt/venv/bin/vllm-image"],
    args: launcherArgs(recipe, {}, { role: "probe" }),
    ports: [{ name: "probe", containerPort: probePort(recipe), protocol: "TCP" }],
    volumeMounts: [
      { name: "shm", mountPath: "/dev/shm" },
      // The probe answers from the launcher record the engine writes before exec
      // (`probes.DEFAULT_RECORD_PATH` = /run/vllm-image/launch.json), so it shares that
      // emptyDir with the modelserver rather than reading a path no container populated.
      { name: "launch-record", mountPath: "/run/vllm-image" },
    ],
  };
  // The probe answers liveness for the engine, so it needs its own CPU and memory
  // reservation: sharing the engine's cgroup let a spin-heavy rank starve the probe
  // and kubelet kill a healthy group (the reason liveness left modelserver). Those
  // figures are measurements taken on the accepted GB10 groups. An inferred shape
  // has no measurement, so it gets no reservation at all rather than a borrowed one.
  if (recipe.deployment?.inferred !== true) {
    container.resources = { requests: { cpu: "2", memory: "256Mi" }, limits: { memory: "512Mi" } };
  }
  return container;
}

// The cache tier's reservation. Memory tracks the resolved L1 arena, which
// `resolver.py` sizes from `cache-l1-gib`; the CPU figure is the accepted GB10
// measurement, so an inferred shape asks for memory alone.
function cacheResources(recipe, l1) {
  const resources = {
    requests: { memory: l1 ? `${l1 + 4}Gi` : "8Gi" },
    limits: { memory: l1 ? `${l1 + 8}Gi` : "16Gi" },
  };
  if (recipe.deployment?.inferred !== true) resources.requests.cpu = "4";
  return resources;
}

function cacheSidecar(recipe, parameters, settings, environment) {
  const l1 = cacheL1GiB(recipe, settings);
  const cachePort = Number(effective(recipe, settings, "cache-port", DEFAULT_PORT + 1));
  const env = [
    ...envList(siteEnvironment(recipe, parameters, "eth0")),
    ...environmentEdits(environment),
  ];
  return {
    name: "cache",
    image: null,
    imagePullPolicy: "IfNotPresent",
    restartPolicy: "Always",
    command: ["/opt/venv/bin/vllm-image"],
    args: launcherArgs(recipe, {}, { role: "cache" }),
    env,
    // The cache owns no GPU: it is a RAM/LMCache tier process, and requesting a GPU
    // would take the engine's only device on a one-GPU node. Its memory follows the
    // resolved L1; the CPU reservation is a GB10 measurement, so an inferred shape
    // gets none.
    resources: cacheResources(recipe, l1),
    startupProbe: { httpGet: { path: "/healthcheck", port: cachePort }, periodSeconds: 10, timeoutSeconds: 5, failureThreshold: 60 },
    volumeMounts: [
      { name: "shm", mountPath: "/dev/shm" },
      { name: "jit-cache", mountPath: "/cache" },
    ],
  };
}

function workerPodSpec(recipe, parameters, imageReference, settings, environment, { single = false } = {}) {
  const cache = hasCache(recipe, settings);
  const deployment = deploymentSection(recipe);
  const initContainers = [probeSidecar(recipe, settings)];
  if (cache) initContainers.push(cacheSidecar(recipe, parameters, settings, environment));
  for (const container of initContainers) container.image = imageReference;
  // The hostPath type is a per-deployment choice the recipe makes (every accepted
  // manifest so far uses DirectoryOrCreate); the two host paths themselves come from
  // the site parameters, so no node path is invented here.
  const hostPathType = deployment.volumes?.hostPathType;
  if (typeof hostPathType !== "string" || !hostPathType) fail("deployment.volumes.hostPathType must name the hostPath type the accepted manifest uses");
  const volumes = [
    // The cache arena and the engine's shm_broadcast share one tmpfs: the launcher
    // opens the LMCache arena under /dev/shm (cache_runtime.SHM_ROOT) and the engine
    // reads it, so both mount the SAME volume rather than two private ones.
    { name: "shm", emptyDir: { medium: "Memory", sizeLimit: cache ? `${(cacheL1GiB(recipe, settings) ?? 64) + 8}Gi` : "64Gi" } },
    { name: "jit-cache", hostPath: { path: parameters.jit_storage_path, type: hostPathType } },
    { name: "model-weights", hostPath: { path: parameters.model_storage_path, type: hostPathType } },
    // Carries the launcher record both the engine and the probe read, and the cache
    // validates against; shared so the sidecars see what the engine published.
    { name: "launch-record", emptyDir: {} },
  ];
  const grace = ENGINE_STOP_GRACE_SECONDS + (cache ? CACHE_STOP_GRACE_SECONDS : 0);
  // Node placement is entirely recipe-driven: the accepted GB10 groups carry the
  // arm64 architecture and the dgx node role here, while a single-host recipe pins
  // only the site parameter (a hostname) and names no architecture at all. An empty
  // merged map renders as no nodeSelector key, never as `{}`.
  const nodeSelector = { ...(deployment.nodeSelector ?? {}), ...(parameters.node_selector ?? {}) };
  const tolerations = Array.isArray(deployment.tolerations) ? deployment.tolerations : [];
  if (deployment.tolerations !== undefined && deployment.tolerations !== null && !tolerations.length) fail("deployment.tolerations is an empty list; omit the key instead of declaring no tolerations");
  const podSecurityContext = deployment.podSecurityContext;
  if (podSecurityContext !== undefined && podSecurityContext !== null && !Object.keys(podSecurityContext).length) fail("deployment.podSecurityContext is empty; omit the key instead of declaring no pod security context");
  return {
    runtimeClassName: single ? undefined : "nvidia",
    enableServiceLinks: false,
    subdomain: parameters.name,
    shareProcessNamespace: true,
    ...(Object.keys(nodeSelector).length ? { nodeSelector } : {}),
    ...(single ? {} : {
      affinity: {
        nodeAffinity: {
          requiredDuringSchedulingIgnoredDuringExecution: {
            nodeSelectorTerms: [{
              matchExpressions: [{
                key: parameters.topology_key,
                operator: "In",
                values: String(parameters.topology_values).split(",").map((value) => value.trim()).filter(Boolean),
              }],
            }],
          },
        },
        podAntiAffinity: {
          requiredDuringSchedulingIgnoredDuringExecution: [{
            labelSelector: { matchLabels: { app: parameters.name } },
            topologyKey: "kubernetes.io/hostname",
          }],
        },
      },
    }),
    ...(tolerations.length ? { tolerations } : {}),
    terminationGracePeriodSeconds: grace,
    ...(podSecurityContext ? { securityContext: podSecurityContext } : {}),
    initContainers,
    containers: [modelServerContainer(recipe, parameters, imageReference, settings, environment)],
    volumes,
  };
}

function podMetadata(recipe, parameters) {
  const metadata = {
    labels: { app: parameters.name, "llm-d.ai/model": parameters.name, "llm-d.ai/engine-type": "vllm" },
  };
  // The Multus attachment is the group's secondary collective network; a single-node
  // pod opens none, and the accepted live pod for that shape carries no such
  // annotation, so the key is left off rather than emitted with an empty value.
  if (fieldInScope(recipe, "network_attachment") && hasValue(parameters.network_attachment, "string")) {
    metadata.annotations = { "k8s.v1.cni.cncf.io/networks": parameters.network_attachment };
  }
  return metadata;
}

// -- LWS ------------------------------------------------------------------------

function renderLws(recipe, parameters, imageReference, settings, environment) {
  const lws = recipe.launch.topology.kind === "lws";
  const nodes = recipe.launch.topology.nodes;
  const deployment = deploymentSection(recipe);
  const port = servingPort(recipe, settings);
  const service = {
    apiVersion: "v1",
    kind: "Service",
    metadata: { name: `${parameters.name}-serve`, namespace: parameters.namespace, labels: { app: parameters.name } },
    spec: {
      // Only rank zero serves the API (the others are headless in topology_args), so
      // a group's debug Service selects its worker-index-0 pods. A single-node pod is
      // not in a group and carries no such label -- that selector would match nothing
      // -- so the shape decides, exactly as the accepted live Service does.
      selector: lws ? { app: parameters.name, "leaderworkerset.sigs.k8s.io/worker-index": "0" } : { app: parameters.name },
      ports: [{ name: "modelserver", port, targetPort: "modelserver" }],
    },
  };
  // One workerTemplate, no leader/worker split: `launch.py` reads LWS_WORKER_INDEX and
  // decides leader vs headless per pod, so every pod of the group is identical. A
  // single-node recipe is not a group at all: it becomes one plain Deployment pod.
  const podSpec = workerPodSpec(recipe, parameters, imageReference, settings, environment, { single: !lws });
  const workload = lws
    ? (() => {
        // The group count, rollout window and restart policy are the live cluster's:
        // `deployment.groups` is how many TP groups the fleet runs, and `deployment.rollout`
        // is the accepted RollingUpdate window. The restart policy is a fixed group rule the
        // recipe may name (the accepted manifests all use RecreateGroupAfterStart).
        if (!Number.isSafeInteger(deployment.groups) || deployment.groups < 1) fail("deployment.groups must be a positive integer");
        const rollout = deployment.rollout;
        if (!rollout || typeof rollout !== "object" || Array.isArray(rollout)) fail("deployment.rollout is required for an lws deployment");
        for (const key of ["type", "maxUnavailable", "maxSurge", "partition"]) {
          if (rollout[key] === undefined || rollout[key] === null || rollout[key] === "") fail(`deployment.rollout.${key} is required for an lws deployment`);
        }
        for (const key of ["maxUnavailable", "maxSurge", "partition"]) {
          if (!Number.isSafeInteger(rollout[key]) || rollout[key] < 0) fail(`deployment.rollout.${key} must be a non-negative integer`);
        }
        return {
          apiVersion: "leaderworkerset.x-k8s.io/v1",
          kind: "LeaderWorkerSet",
          metadata: {
            name: parameters.name,
            namespace: parameters.namespace,
            labels: { app: parameters.name, "llm-d.ai/model": parameters.name },
            ...(parameters.topology_key ? { annotations: { "leaderworkerset.sigs.k8s.io/exclusive-topology": parameters.topology_key } } : {}),
          },
          spec: {
            replicas: deployment.groups,
            startupPolicy: "LeaderCreated",
            networkConfig: { subdomainPolicy: "Shared" },
            rolloutStrategy: { type: rollout.type, rollingUpdateConfiguration: { maxUnavailable: rollout.maxUnavailable, maxSurge: rollout.maxSurge, partition: rollout.partition } },
            leaderWorkerTemplate: { size: nodes, restartPolicy: deployment.restartPolicy ?? "RecreateGroupAfterStart", workerTemplate: { metadata: podMetadata(recipe, parameters), spec: podSpec } },
          },
        };
      })()
    : {
        apiVersion: "apps/v1",
        kind: "Deployment",
        metadata: { name: parameters.name, namespace: parameters.namespace, labels: { app: parameters.name, "llm-d.ai/model": parameters.name } },
        spec: {
          replicas: effectiveReplicas(recipe, settings),
          strategy: { type: "Recreate" },
          selector: { matchLabels: { app: parameters.name } },
          template: { metadata: podMetadata(recipe, parameters), spec: podSpec },
        },
      };

  const filename = "lws.yaml";
  const files = [{ name: filename, body: yamlDocuments([service, workload]) }];
  const steps = [
    shellCommand(["kubectl", "create", "namespace", parameters.namespace]),
    shellCommand(["kubectl", "-n", parameters.namespace, "create", "secret", "generic", parameters.hf_secret]) + " " + shellQuote(`--from-literal=${parameters.hf_secret_key}=`) + '"$HF_TOKEN"',
    shellCommand(["kubectl", "apply", "-f", filename]),
    shellCommand(["kubectl", "-n", parameters.namespace, "wait", "--for=condition=Ready", "pod", "-l", `app=${parameters.name}`, "--timeout=2h"]),
    shellCommand(["kubectl", "-n", parameters.namespace, "get", "pod", "-l", `app=${parameters.name}`, "-o", "wide"]),
    shellCommand(["kubectl", "-n", parameters.namespace, "port-forward", `service/${parameters.name}-serve`, `${port}:${port}`]),
  ];
  return { files, steps };
}

// -- Docker ---------------------------------------------------------------------

// Docker runs the image once per rank with the launcher arguments. A cache or a
// replica layout is refused here and sent to Compose, because the cache must join
// the model's network and IPC namespaces and replicas need the proxy sidecar — both
// are service-graph features Compose expresses and a bare `docker run` loop cannot.
function renderDocker(recipe, parameters, imageReference, settings, environment) {
  if (hasCache(recipe, settings)) {
    fail("this deployment selects an external cache, which a plain `docker run` cannot share with the engine; use the Compose tab");
  }
  const replicas = effectiveReplicas(recipe, settings);
  if (replicas > 1) {
    fail(`this deployment runs ${replicas} replicas behind a proxy, which a bare \`docker run\` cannot express; use the Compose tab`);
  }
  const lws = recipe.launch.topology.kind === "lws";
  const nodes = lws ? recipe.launch.topology.nodes : 1;
  const port = servingPort(recipe, settings);

  // One builder for both layouts, so the single-node run and the per-rank run can
  // not disagree about anything but the rank identity.
  const run = ({ name, rm = false, rank = null }) => {
    const args = ["docker", "run", rm ? "--rm" : "--detach", "--name", name, "--user", "0", "--network", "host", "--gpus", "all"];
    // The RoCE device and the page-locking capability belong to the collective. A
    // single-node run opens no HCA, and asking docker for a /dev/infiniband the host
    // does not have fails the run outright.
    if (usesRdma(recipe)) args.push("--device", "/dev/infiniband", "--cap-add", "IPC_LOCK");
    args.push(
      "--ulimit", "memlock=-1", "--shm-size", "64g", "--stop-timeout", "120",
      "-v", modelMount(recipe, parameters),
      "-v", `${parameters.jit_storage_path}:/cache`,
      "-e", "HF_TOKEN", "-e", "HF_XET_HIGH_PERFORMANCE=1",
    );
    for (const [variable, value] of Object.entries(siteEnvironment(recipe, parameters, parameters.control_interface))) args.push("-e", `${variable}=${value}`);
    for (const edit of environmentEdits(environment)) args.push("-e", `${edit.name}=${edit.value}`);
    if (rank !== null) {
      args.push("-e", `LWS_WORKER_INDEX=${rank}`, "-e", `LWS_GROUP_SIZE=${nodes}`, "-e", `LWS_LEADER_ADDRESS=${rank === 0 ? "127.0.0.1" : parameters.leader_ip}`);
    }
    args.push("--entrypoint", "/opt/venv/bin/vllm-image", imageReference, ...launcherArgs(recipe, settings));
    return args;
  };

  const lines = ["#!/bin/sh", "set -eu", ""];
  const containers = [];
  if (nodes === 1) {
    const once = `${parameters.name}-once`;
    containers.push(once);
    lines.push(shellCommand(run({ name: once, rm: true })), "");
  } else {
    for (let rank = 0; rank < nodes; rank += 1) {
      const container = `${parameters.name}-rank${rank}`;
      containers.push(container);
      const host = rank === 0 ? parameters.leader_ip : parameters.worker_ip;
      lines.push(`# rank ${rank} on ${host}`, shellCommand(run({ name: container, rank })), "");
    }
  }
  const files = [{ name: "serve.sh", body: lines.join("\n") }];
  const steps = [
    shellCommand(["sh", "serve.sh"]),
    shellCommand(["docker", "logs", "--follow", containers[0]]),
    shellCommand(["docker", "stop", "--time", "120", ...containers]),
  ];
  void port;
  return { files, steps };
}

// -- Compose --------------------------------------------------------------------

// Compose splits the same way Kubernetes does. A cache joins the model service in its
// network and IPC namespaces and shares the tmpfs; replicas each get their own GPU and
// the proxy service load-balances them.
function modelComposeService(recipe, parameters, imageReference, settings, environment, { name, rank, gpu } = {}) {
  const service = {
    image: imageReference,
    entrypoint: "/opt/venv/bin/vllm-image",
    command: launcherArgs(recipe, settings),
    ipc: "host",
    network_mode: "host",
    restart: "unless-stopped",
    cap_add: ["IPC_LOCK"],
    ulimits: { memlock: { soft: -1, hard: -1 } },
    shm_size: "64g",
    stop_grace_period: "120s",
    volumes: [
      modelMount(recipe, parameters),
      `${parameters.jit_storage_path}:/cache`,
      "shm:/dev/shm",
    ],
    environment: {
      ...siteEnvironment(recipe, parameters, parameters.control_interface),
      HF_TOKEN: "${HF_TOKEN}",
      ...Object.fromEntries(environmentEdits(environment).map((edit) => [edit.name, edit.value])),
    },
    deploy: {
      resources: { reservations: { devices: [{ driver: "nvidia", count: gpu ? "1" : "0", capabilities: ["gpu"] }] } },
    },
  };
  void name;
  void rank;
  return service;
}

function probeComposeService(recipe, parameters, imageReference, settings) {
  return {
    image: imageReference,
    entrypoint: "/opt/venv/bin/vllm-image",
    command: launcherArgs(recipe, {}, { role: "probe" }),
    // The probe shares the model's network and IPC namespaces so its httpGet checks
    // read the engine's own state, exactly like the pod's shared process namespace.
    network_mode: "service:model",
    ipc: "service:model",
    restart: "unless-stopped",
    environment: {
      ...siteEnvironment(recipe, parameters, parameters.control_interface),
      ...Object.fromEntries(environmentEdits({}).map((edit) => [edit.name, edit.value])),
    },
    volumes: ["shm:/dev/shm"],
    depends_on: ["model"],
    deploy: {
      resources: { reservations: { cpus: "2", memory: "256M" } },
    },
  };
}

function cacheComposeService(recipe, parameters, imageReference, settings, environment) {
  const l1 = cacheL1GiB(recipe, settings);
  return {
    image: imageReference,
    entrypoint: "/opt/venv/bin/vllm-image",
    command: launcherArgs(recipe, {}, { role: "cache" }),
    network_mode: "service:model",
    ipc: "service:model",
    restart: "unless-stopped",
    environment: {
      ...siteEnvironment(recipe, parameters, parameters.control_interface),
      ...Object.fromEntries(environmentEdits(environment).map((edit) => [edit.name, edit.value])),
    },
    // The cache and the engine share the same tmpfs arena and the /cache tier volume.
    volumes: ["shm:/dev/shm", `${parameters.jit_storage_path}:/cache`],
    deploy: {
      resources: { reservations: { memory: l1 ? `${l1}G` : "8G" } },
    },
  };
}

function proxyComposeService(recipe, parameters, imageReference, settings, replicas) {
  return {
    image: imageReference,
    entrypoint: "/opt/venv/bin/vllm-image",
    command: launcherArgs(recipe, {}, { role: "proxy" }),
    network_mode: "host",
    restart: "unless-stopped",
    ports: [`${servingPort(recipe, settings)}:${servingPort(recipe, settings)}`],
    depends_on: Array.from({ length: replicas }, (_, index) => `model-${index}`),
    deploy: {
      resources: { reservations: { cpus: "1", memory: "256M" } },
    },
  };
}

function renderCompose(recipe, parameters, imageReference, settings, environment) {
  const lws = recipe.launch.topology.kind === "lws";
  const nodes = lws ? recipe.launch.topology.nodes : 1;
  const replicas = effectiveReplicas(recipe, settings);
  const cache = hasCache(recipe, settings);
  const services = {};

  if (replicas > 1) {
    // Each replica is its own single-node service with one GPU; the proxy fans requests
    // across them on their per-replica ports (replica_port_base in launch.py).
    for (let index = 0; index < replicas; index += 1) {
      services[`model-${index}`] = modelComposeService(recipe, parameters, imageReference, settings, environment, { name: `model-${index}`, rank: 0, gpu: true });
    }
    services["model-proxy"] = proxyComposeService(recipe, parameters, imageReference, settings, replicas);
  } else if (lws && nodes > 1) {
    // A TP group spans ranks; compose can only host them together on one docker host,
    // so each rank is a service pinned to that host and the leader's rendezvous port is
    // published for the followers.
    for (let rank = 0; rank < nodes; rank += 1) {
      const service = modelComposeService(recipe, parameters, imageReference, settings, environment, { name: `model-rank${rank}`, rank, gpu: true });
      service.environment.LWS_WORKER_INDEX = String(rank);
      service.environment.LWS_GROUP_SIZE = String(nodes);
      service.environment.LWS_LEADER_ADDRESS = rank === 0 ? "127.0.0.1" : "${LWS_LEADER_ADDRESS}";
      services[`model-rank${rank}`] = service;
    }
    services["model"] = services["model-rank0"];
  } else {
    services["model"] = modelComposeService(recipe, parameters, imageReference, settings, environment, { name: "model", rank: 0, gpu: true });
  }

  // The probe sidecar follows the leader model service, mirroring the pod.
  services["probe"] = probeComposeService(recipe, parameters, imageReference, settings);
  if (cache) services["cache"] = cacheComposeService(recipe, parameters, imageReference, settings, environment);

  const document = {
    name: parameters.name,
    services,
    volumes: {
      // The cache arena is a shared tmpfs so the engine and the cache see one /dev/shm.
      shm: { driver_opts: { type: "tmpfs", o: "size=64g" } },
    },
  };
  const files = [{ name: "compose.yaml", body: stringifyYaml(document, { lineWidth: 0 }) }];
  const steps = [
    shellCommand(["HF_TOKEN=... docker compose", "-f", "compose.yaml", "up", "--detach"]),
    shellCommand(["docker", "compose", "-f", "compose.yaml", "ps"]),
    shellCommand(["docker", "compose", "-f", "compose.yaml", "logs", "-f", "model"]),
    shellCommand(["docker", "compose", "-f", "compose.yaml", "down", "--timeout", "120"]),
  ];
  return { files, steps };
}

// -- Routing (llm-d) ------------------------------------------------------------

// The routing bundle is unchanged in substance: it installs the same Endpoint Picker,
// InferencePool, Gateway and AI Gateway route. It now references the recipe's served
// model name and the recipe selection rather than any hand-written vLLM command — the
// engine command comes only from the launcher form the other tabs render.
function routingResources(recipe, parameters, servedModel) {
  const name = parameters.name;
  const namespace = parameters.namespace;
  const poolName = `${name}-pool`;
  const eppName = `${name}-epp`;
  const gatewayName = `${name}-gateway`;
  const routeName = `${name}-route`;
  const config = {
    apiVersion: "inference.networking.x-k8s.io/v1alpha1",
    kind: "EndpointPickerConfig",
    plugins: [
      { type: "metrics-data-source", parameters: { scheme: "http", path: "/metrics", insecureSkipVerify: true } },
      { type: "core-metrics-extractor", parameters: { defaultEngine: "vllm", engineConfigs: [{ name: "vllm", queuedRequestsSpec: "vllm:num_requests_waiting", runningRequestsSpec: "vllm:num_requests_running", kvUsageSpec: "vllm:kv_cache_usage_perc", loraSpec: "", cacheInfoSpec: "vllm:cache_config_info" }] } },
      { type: "approx-prefix-cache-producer" },
      { type: "queue-scorer" },
      { type: "kv-cache-utilization-scorer" },
      { type: "prefix-cache-scorer" },
    ],
    schedulingProfiles: [{ name: "default", plugins: [
      { pluginRef: "queue-scorer", weight: 2 },
      { pluginRef: "kv-cache-utilization-scorer", weight: 2 },
      { pluginRef: "prefix-cache-scorer", weight: 3 },
    ] }],
  };
  const serviceAccount = { apiVersion: "v1", kind: "ServiceAccount", metadata: { name: eppName, namespace } };
  const clusterRoleName = `${eppName}-${namespace}`;
  const clusterRole = {
    apiVersion: "rbac.authorization.k8s.io/v1", kind: "ClusterRole", metadata: { name: clusterRoleName }, rules: [
      { apiGroups: ["inference.networking.k8s.io"], resources: ["inferencepools"], verbs: ["get", "list", "watch"] },
      { apiGroups: ["inference.networking.x-k8s.io"], resources: ["inferencemodelrewrites", "inferenceobjectives"], verbs: ["get", "list", "watch"] },
      { apiGroups: [""], resources: ["pods", "nodes"], verbs: ["get", "list", "watch"] },
      { apiGroups: ["authentication.k8s.io"], resources: ["tokenreviews"], verbs: ["create"] },
      { apiGroups: ["authorization.k8s.io"], resources: ["subjectaccessreviews"], verbs: ["create"] },
      { nonResourceURLs: ["/metrics"], verbs: ["get"] },
    ],
  };
  const clusterRoleBinding = { apiVersion: "v1", kind: "ClusterRoleBinding", metadata: { name: clusterRoleName }, roleRef: { apiGroup: "rbac.authorization.k8s.io", kind: "ClusterRole", name: clusterRoleName }, subjects: [{ kind: "ServiceAccount", name: eppName, namespace }] };
  const configMap = { apiVersion: "v1", kind: "ConfigMap", metadata: { name: `${eppName}-config`, namespace }, data: { "config.yaml": stringifyYaml(config, { lineWidth: 0 }) } };
  const deployment = {
    apiVersion: "apps/v1", kind: "Deployment", metadata: { name: eppName, namespace, labels: { app: eppName } },
    spec: { replicas: 1, strategy: { type: "Recreate" }, selector: { matchLabels: { app: eppName } }, template: { metadata: { labels: { app: eppName } }, spec: {
      serviceAccountName: eppName,
      securityContext: { runAsNonRoot: true, seccompProfile: { type: "RuntimeDefault" } },
      containers: [{
        name: "epp", image: "registry.k8s.io/gateway-api-inference-extension/epp:v1.5.0",
        securityContext: { allowPrivilegeEscalation: false, capabilities: { drop: ["ALL"] }, readOnlyRootFilesystem: true, runAsNonRoot: true, runAsUser: 65532 },
        resources: { requests: { cpu: "4", memory: "8Gi" }, limits: { memory: "16Gi" } },
        args: [`--pool-name=${poolName}`, `--pool-namespace=${namespace}`, "--pool-group=inference.networking.k8s.io", "--metrics-endpoint-auth=false", "--config-file=/config/config.yaml", "-v=4"],
        ports: [{ name: "grpc", containerPort: 9002, protocol: "TCP" }, { name: "metrics", containerPort: 9090, protocol: "TCP" }, { name: "health", containerPort: 9003, protocol: "TCP" }],
        env: [{ name: "NAMESPACE", valueFrom: { fieldRef: { fieldPath: "metadata.namespace" } } }, { name: "POD_NAME", valueFrom: { fieldRef: { fieldPath: "metadata.name" } } }],
        volumeMounts: [{ name: "epp-config", mountPath: "/config", readOnly: true }],
        livenessProbe: { grpc: { port: 9003 }, initialDelaySeconds: 5, periodSeconds: 10 },
        readinessProbe: { grpc: { port: 9003 }, initialDelaySeconds: 5, periodSeconds: 10 },
      }],
      volumes: [{ name: "epp-config", configMap: { name: `${eppName}-config` } }],
    } } },
  };
  const service = { apiVersion: "v1", kind: "Service", metadata: { name: eppName, namespace, labels: { app: eppName } }, spec: { selector: { app: eppName }, ports: [{ name: "grpc", port: 9002, targetPort: "grpc", protocol: "TCP" }, { name: "metrics", port: 9090, targetPort: "metrics", protocol: "TCP" }] } };
  // The pool fronts the pods that serve: rank zero of each group for a group, every
  // pod of a single-node Deployment (its replicas are independent engines).
  const pool = { apiVersion: "inference.networking.k8s.io/v1", kind: "InferencePool", metadata: { name: poolName, namespace }, spec: { appProtocol: "http", selector: { matchLabels: isCollective(recipe) ? { app: name, "leaderworkerset.sigs.k8s.io/worker-index": "0" } : { app: name } }, targetPorts: [{ number: servingPort(recipe, {}) }], endpointPickerRef: { name: eppName, port: { number: 9002 }, failureMode: "FailClose" } } };
  const gateway = { apiVersion: "gateway.networking.k8s.io/v1", kind: "Gateway", metadata: { name: gatewayName, namespace }, spec: { gatewayClassName: parameters.gateway_class, listeners: [{ name: "http", protocol: "HTTP", port: 10080 }] } };
  const route = { apiVersion: "aigateway.envoyproxy.io/v1beta1", kind: "AIGatewayRoute", metadata: { name: routeName, namespace }, spec: { parentRefs: [{ name: gatewayName, namespace }], rules: [{ matches: [{ headers: [{ type: "Exact", name: "x-ai-eg-model", value: servedModel }] }], backendRefs: [{ group: "inference.networking.k8s.io", kind: "InferencePool", name: poolName }], timeouts: { request: "3600s" } }] } };
  const backendTimeouts = { apiVersion: "gateway.envoyproxy.io/v1alpha1", kind: "BackendTrafficPolicy", metadata: { name: `${name}-backend-timeouts`, namespace }, spec: { targetRefs: [{ group: "gateway.networking.k8s.io", kind: "HTTPRoute", name: routeName }], timeout: { http: { requestTimeout: "3600s" } } } };
  const nativeMessages = recipe.model.serves_native_messages
    ? { apiVersion: "gateway.networking.k8s.io/v1", kind: "HTTPRoute", metadata: { name: `${name}-native-messages`, namespace }, spec: { parentRefs: [{ name: gatewayName, sectionName: "http" }], rules: [{ matches: [{ path: { type: "PathPrefix", value: "/v1/messages" } }], backendRefs: [{ group: "inference.networking.k8s.io", kind: "InferencePool", name: poolName, port: servingPort(recipe, {}) }], timeouts: { request: "3600s" } }] } }
    : null;
  const nativeMessagesTimeouts = nativeMessages
    ? { apiVersion: "gateway.envoyproxy.io/v1alpha1", kind: "BackendTrafficPolicy", metadata: { name: `${name}-native-messages-timeouts`, namespace }, spec: { targetRefs: [{ group: "gateway.networking.k8s.io", kind: "HTTPRoute", name: `${name}-native-messages` }], timeout: { http: { requestTimeout: "3600s", streamIdleTimeout: "3600s" } } } }
    : null;
  const policy = { apiVersion: "gateway.envoyproxy.io/v1alpha1", kind: "ClientTrafficPolicy", metadata: { name: `${name}-buffer-limit`, namespace }, spec: { targetRefs: [{ group: "gateway.networking.k8s.io", kind: "Gateway", name: gatewayName }], connection: { bufferLimit: "50Mi" } } };
  return [serviceAccount, clusterRole, clusterRoleBinding, configMap, deployment, service, pool, gateway, route, nativeMessages, backendTimeouts, nativeMessagesTimeouts, policy].filter(Boolean);
}

function renderRouting(recipe, parameters) {
  const servedModel = servedModelName(recipe, {});
  const filename = "routing.yaml";
  const files = [{ name: filename, body: yamlDocuments(routingResources(recipe, parameters, servedModel)) }];
  const steps = [
    shellCommand(["kubectl", "apply", "-f", filename]),
    shellCommand(["kubectl", "-n", parameters.namespace, "rollout", "status", `deployment/${parameters.name}-epp`, "--timeout=5m"]),
    shellCommand(["kubectl", "-n", parameters.namespace, "get", "gateway", `${parameters.name}-gateway`, "-o", "yaml"]),
    shellCommand(["curl", "--fail-with-body", "http://127.0.0.1:10080/v1/chat/completions", "-H", `x-ai-eg-model: ${servedModel}`, "-H", "Content-Type: application/json", "--data", JSON.stringify({ model: servedModel, messages: [{ role: "user", content: "Reply with ready." }] })]),
  ];
  return { files, steps };
}

// -- entry ----------------------------------------------------------------------

export function renderRecipe(recipe, options = {}) {
  const { parameters = {}, settings = {}, environment = {}, target = "lws", image } = options;
  if (typeof parameters !== "object" || parameters === null || Array.isArray(parameters)) fail("parameters must be an object");
  if (typeof settings !== "object" || settings === null || Array.isArray(settings)) fail("settings must be an object");
  if (typeof environment !== "object" || environment === null || Array.isArray(environment)) fail("environment must be an object");
  if (!(target in TARGET_FIELDS)) fail(`unsupported render target: ${target}`);
  validateRecipe(recipe);
  const values = normalizedParameters(recipe, parameters, target);
  if (target === "routing") return renderRouting(recipe, values);
  validateImageReference(image);
  if (target === "lws") return renderLws(recipe, values, image, settings, environment);
  if (target === "docker") return renderDocker(recipe, values, image, settings, environment);
  return renderCompose(recipe, values, image, settings, environment);
}
