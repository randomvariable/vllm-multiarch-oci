import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { parseAllDocuments } from "yaml";
import { renderRecipe } from "../src/render.js";

// The renderer is a pure function over a launcher-form recipe document, so these
// tests build the document directly rather than reading the on-disk YAML — those files
// are being consolidated concurrently and the shape the launcher consumes is fixed by
// `image_tools/launcher/launch.py.Recipe.from_dict`, which the parity test below checks
// this module against. The shape is: `launch` (profile, hardware, preset, options,
// environment, topology, probe_port), `model` (model_id, served_name), and `deployment`
// (model_path + storage_root for a sync deployment, parameters for the site fields).

const siteRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const REPOSITORY_ROOT = resolve(siteRoot, "..");
const LAUNCH_PY = join(REPOSITORY_ROOT, "image_tools/launcher/launch.py");

const DIGEST = "sha256:" + "a".repeat(64);
const IMAGE = `ghcr.io/local-inference-lab/vllmb12x@${DIGEST}`;

function parametersMap() {
  const field = (type, value, extra = {}) => ({ type, required: true, default: value, ...extra });
  return {
    namespace: field("string", "vllm"),
    name: field("string", "qwen38-flash-next"),
    model_storage_path: field("string", "/var/lib/models"),
    jit_storage_path: field("string", "/var/cache/vllm/qwen38-flash-next"),
    hf_secret: field("string", "huggingface-token"),
    hf_secret_key: field("string", "HF_TOKEN"),
    network_attachment: field("string", "roce-net"),
    node_selector: field("stringMap", {}),
    topology_key: field("string", "node.example.com/roce-pair"),
    topology_values: field("string", "pair-a,pair-b"),
    rdma_resource: field("string", "rdma.example.com/roce"),
    rdma_units: field("integer", 1),
    hca: field("string", "mlx5_0,mlx5_1"),
    gid_index: field("integer", 3),
    control_interface: field("string", "eth0", { required: false }),
    leader_ip: field("string", "192.0.2.10", { required: false }),
    worker_ip: field("string", "192.0.2.11", { required: false }),
    gateway_class: field("string", "ai-gateway", { required: false }),
  };
}

// A two-node RoCE TP=2 recipe in launcher form: sync mode (deployment.model_path set),
// no external cache (the reader can add one with `cache-mode`).
function tp2Recipe(overrides = {}) {
  return {
    meta: { slug: "qwen38-flash-next-gb10-tp2", title: "t", description: "d" },
    model: { model_id: "local-inference-lab/Qwen3.8-Flash-Next-NVFP4", revision: "60215d26cf5e42c2db6128774032d57fc62678da", served_name: "qwen38-flash-next", serves_native_messages: true },
    launch: {
      profile: "qwen38-flash-next",
      hardware: "gb10-roce",
      preset: null,
      options: { port: 8888, "served-model-name": "qwen38-flash-next", "tensor-parallel-size": 2, "max-num-seqs": 8 },
      environment: { NCCL_NET: "IB", VLLM_USE_V2_MODEL_RUNNER: "1" },
      topology: { kind: "lws", nodes: 2, rendezvous_port: 25000, kv_events: null, replica_port_base: null },
      probe_port: 8890,
    },
    deployment: {
      model_path: "/models/qwen38-flash-next",
      storage_root: "/models",
      parameters: parametersMap(),
    },
    ...overrides,
  };
}

// An engine-download single-node recipe: no model_path, so no `--model-sync`, and no
// sync-only deployment facts at all — the renderer must tolerate their absence.
function singleRecipe() {
  const recipe = tp2Recipe();
  recipe.meta.slug = "qwen38-27b";
  recipe.launch.topology = { kind: "single", nodes: 1, rendezvous_port: null, kv_events: null, replica_port_base: null };
  recipe.launch.options = { port: 8888, "served-model-name": "qwen3.8-27b", model: "local-inference-lab/Qwen3.8-27B-NVFP4-QAD" };
  recipe.launch.environment = { HF_HOME: "/models" };
  recipe.deployment = { parameters: parametersMap() };
  return recipe;
}

function documents(body) {
  return parseAllDocuments(body).map((document) => document.toJS());
}

function onlyDoc(docs, kind) {
  const found = docs.filter((document) => document.kind === kind);
  assert.equal(found.length, 1, `expected exactly one ${kind}`);
  return found[0];
}

// The whole rendered output flattened to text, so "this string never appears anywhere"
// assertions cover every container, sidecar and comment the renderer emits.
function renderedText(result) {
  return result.files.map((file) => file.body).join("\n");
}

function podSpec(docs) {
  const lws = onlyDoc(docs, "LeaderWorkerSet");
  return lws.spec.leaderWorkerTemplate.workerTemplate.spec;
}

test("LWS renders the launcher form with one worker template and no hand-written command", () => {
  const result = renderRecipe(tp2Recipe(), { parameters: { node_selector: { gpu: "gb10" } }, settings: {}, environment: {}, target: "lws", image: IMAGE });
  assert.deepEqual(result.files.map((file) => file.name), ["lws.yaml"]);
  const docs = documents(result.files[0].body);
  const lws = onlyDoc(docs, "LeaderWorkerSet");
  assert.equal(lws.apiVersion, "leaderworkerset.x-k8s.io/v1");
  // One workerTemplate: no leader/worker split. The launcher decides rank from the
  // controller-injected LWS_WORKER_INDEX, so every pod of the group is identical.
  assert.ok(lws.spec.leaderWorkerTemplate.workerTemplate, "a workerTemplate is required");
  assert.equal(lws.spec.leaderWorkerTemplate.leaderTemplate, undefined, "no separate leaderTemplate");
  assert.equal(lws.spec.leaderWorkerTemplate.size, 2);

  const spec = podSpec(docs);
  assert.equal(spec.shareProcessNamespace, true);
  assert.deepEqual(spec.containers.map((container) => container.name), ["modelserver"]);

  const server = spec.containers[0];
  assert.deepEqual(server.command, ["/opt/venv/bin/vllm-image"]);
  assert.deepEqual(server.args, ["launch", "--recipe", "qwen38-flash-next-gb10-tp2", "--topology", "lws", "--model-sync", "--"]);

  // The engine, probe and cache containers all run the image binary; none of them may
  // shell out or use the removed exec health helper.
  const text = renderedText(result);
  assert.ok(!/sh -c/.test(text), "no shell wrapper");
  assert.ok(!/["']bash["']/.test(text) && !/command:\s*\[bash/.test(text), "no bash command");
  assert.ok(!text.includes("vllm-image health"), "no exec health probe");
  const blockingNames = podSpec(documents(result.files[0].body)).initContainers.map((container) => container.name);
  assert.ok(!blockingNames.includes("model-sync"), "no blocking model-sync init container");
  assert.ok(!blockingNames.includes("rendezvous-wait"), "no blocking rendezvous-wait init container");
});

test("every pod initContainer is a native sidecar with restartPolicy Always", () => {
  const result = renderRecipe(tp2Recipe(), { parameters: { node_selector: { gpu: "gb10" } }, settings: {}, environment: {}, target: "lws", image: IMAGE });
  const spec = podSpec(documents(result.files[0].body));
  assert.ok(spec.initContainers.length > 0, "the probe sidecar must be present");
  for (const container of spec.initContainers) {
    assert.equal(container.restartPolicy, "Always", `${container.name} must be a native sidecar`);
  }
  const probe = spec.initContainers.find((container) => container.name === "probe");
  assert.ok(probe, "a probe sidecar exists");
  assert.deepEqual(probe.command, ["/opt/venv/bin/vllm-image"]);
  assert.deepEqual(probe.args, ["launch", "--recipe", "qwen38-flash-next-gb10-tp2", "--topology", "lws", "--role", "probe"]);
  // The probe needs its own CPU and memory request: the cgroup isolation that lets a
  // spin-heavy rank not starve the liveness check. And it owns no GPU.
  assert.ok(probe.resources.requests.cpu, "probe requests cpu");
  assert.ok(probe.resources.requests.memory, "probe requests memory");
  assert.equal(probe.resources.requests["nvidia.com/gpu"], undefined, "probe requests no gpu");
});

test("probes are httpGet on the probe port with /readyz and /livez paths", () => {
  const spec = podSpec(documents(renderRecipe(tp2Recipe(), { parameters: { node_selector: { gpu: "gb10" } }, settings: {}, environment: {}, target: "lws", image: IMAGE }).files[0].body));
  const server = spec.containers[0];
  assert.deepEqual(server.startupProbe.httpGet, { path: "/readyz", port: 8890 });
  assert.deepEqual(server.readinessProbe.httpGet, { path: "/readyz", port: 8890 });
  assert.deepEqual(server.livenessProbe.httpGet, { path: "/livez", port: 8890 });
  assert.equal(server.startupProbe.exec, undefined);
  assert.equal(server.livenessProbe.exec, undefined);
});

test("the modelserver carries the GPU and RDMA requests and IPC_LOCK", () => {
  const spec = podSpec(documents(renderRecipe(tp2Recipe(), { parameters: { node_selector: { gpu: "gb10" } }, settings: {}, environment: {}, target: "lws", image: IMAGE }).files[0].body));
  const server = spec.containers[0];
  assert.equal(server.resources.requests["nvidia.com/gpu"], "1");
  assert.equal(server.resources.requests["rdma.example.com/roce"], "1");
  assert.deepEqual(server.securityContext.capabilities.add, ["IPC_LOCK"]);
});

test("the reader's settings become -- changes and defaults stay absent", () => {
  const plain = renderRecipe(tp2Recipe(), { parameters: { node_selector: { gpu: "gb10" } }, settings: {}, environment: {}, target: "lws", image: IMAGE });
  // Nothing the reader did not change may appear: the launcher resolves it.
  for (const flag of ["--max-num-seqs", "--tensor-parallel-size", "--gpu-memory-utilization", "--served-model-name", "--port"]) {
    assert.ok(!renderedText(plain).includes(flag), `${flag} must be absent when unchanged`);
  }

  const edited = renderRecipe(tp2Recipe(), { parameters: { node_selector: { gpu: "gb10" } }, settings: { "max-num-seqs": "16", "gpu-memory-utilization": "0.8" }, environment: {}, target: "lws", image: IMAGE });
  const server = podSpec(documents(edited.files[0].body)).containers[0];
  const dash = server.args.indexOf("--");
  assert.ok(dash > 0, "a -- separator is present");
  assert.deepEqual(server.args.slice(dash), ["--", "--max-num-seqs", "16", "--gpu-memory-utilization", "0.8"]);
  // A JSON-valued or dotted edit is one --key=value token.
  const json = renderRecipe(tp2Recipe(), { parameters: { node_selector: { gpu: "gb10" } }, settings: { "speculative-config": '{"method":"mtp"}' }, environment: {}, target: "lws", image: IMAGE });
  const jsonServer = podSpec(documents(json.files[0].body)).containers[0];
  assert.ok(jsonServer.args.includes('--speculative-config={"method":"mtp"}'), "a JSON edit is a single = token");
});

test("the cache sidecar appears only with a cache and shares /dev/shm and /cache with the engine", () => {
  const noCache = renderRecipe(tp2Recipe(), { parameters: { node_selector: { gpu: "gb10" } }, settings: {}, environment: {}, target: "lws", image: IMAGE });
  const plainSpec = podSpec(documents(noCache.files[0].body));
  assert.equal(plainSpec.initContainers.find((container) => container.name === "cache"), undefined, "no cache sidecar without a cache");

  const withCache = renderRecipe(tp2Recipe(), { parameters: { node_selector: { gpu: "gb10" } }, settings: { "cache-mode": "lmcache", "cache-l1-gib": "64" }, environment: {}, target: "lws", image: IMAGE });
  const spec = podSpec(documents(withCache.files[0].body));
  const cache = spec.initContainers.find((container) => container.name === "cache");
  assert.ok(cache, "the cache sidecar is present");
  assert.equal(cache.restartPolicy, "Always", "the cache is a native sidecar");
  assert.deepEqual(cache.args, ["launch", "--recipe", "qwen38-flash-next-gb10-tp2", "--topology", "lws", "--role", "cache"]);
  assert.deepEqual(cache.startupProbe.httpGet.path, "/healthcheck", "startupProbe on the cache /healthcheck");
  assert.equal(cache.resources.requests["nvidia.com/gpu"], undefined, "the cache owns no GPU");
  // Both the cache and the modelserver mount the SAME tmpfs at /dev/shm and share /cache.
  const shm = spec.volumes.find((volume) => volume.name === "shm");
  assert.deepEqual(shm.emptyDir.medium, "Memory", "the shared /dev/shm is a Memory emptyDir");
  const server = spec.containers[0];
  assert.ok(server.volumeMounts.some((mount) => mount.name === "shm" && mount.mountPath === "/dev/shm"), "the engine mounts the shared shm");
  assert.ok(cache.volumeMounts.some((mount) => mount.name === "shm" && mount.mountPath === "/dev/shm"), "the cache mounts the same shm");
  assert.ok(cache.volumeMounts.some((mount) => mount.name === "jit-cache" && mount.mountPath === "/cache"), "the cache shares /cache");
  assert.ok(server.volumeMounts.some((mount) => mount.name === "jit-cache" && mount.mountPath === "/cache"), "the engine shares /cache");
});

test("the pod drain covers the engine stop plus the cache stop grace", () => {
  const noCache = podSpec(documents(renderRecipe(tp2Recipe(), { parameters: { node_selector: { gpu: "gb10" } }, settings: {}, environment: {}, target: "lws", image: IMAGE }).files[0].body));
  const withCache = podSpec(documents(renderRecipe(tp2Recipe(), { parameters: { node_selector: { gpu: "gb10" } }, settings: { "cache-mode": "lmcache", "cache-l1-gib": "32" }, environment: {}, target: "lws", image: IMAGE }).files[0].body));
  assert.ok(noCache.terminationGracePeriodSeconds >= 120, "at least the engine stop budget");
  assert.ok(withCache.terminationGracePeriodSeconds > noCache.terminationGracePeriodSeconds, "the cache stop grace extends the budget");
});

test("an engine-download single-node recipe omits --model-sync and the group scheduling", () => {
  const result = renderRecipe(singleRecipe(), { parameters: { node_selector: { gpu: "gb10" } }, settings: {}, environment: {}, target: "lws", image: IMAGE });
  const docs = documents(result.files[0].body);
  const deployment = onlyDoc(docs, "Deployment");
  const server = deployment.spec.template.spec.containers[0];
  assert.ok(!server.args.includes("--model-sync"), "no model-sync without a published root");
  assert.ok(!server.args.includes("--topology", server.args.indexOf("--")), "no --topology lws");
  // sync-only deployment facts are absent from this recipe and must not be required.
  assert.equal(renderRecipe(singleRecipe(), { parameters: { node_selector: { gpu: "gb10" } }, settings: {}, environment: {}, target: "compose", image: IMAGE }).files[0].name, "compose.yaml");
});

test("the launcher arguments match launch.py's own ordering", () => {
  const server = podSpec(documents(renderRecipe(tp2Recipe(), { parameters: { node_selector: { gpu: "gb10" } }, settings: {}, environment: {}, target: "lws", image: IMAGE }).files[0].body)).containers[0];
  const launch = readFileSync(LAUNCH_PY, "utf8");
  // The container builds the resolved argv, then appends topology_args, then --middleware.
  const topologyAt = launch.indexOf("argv += topology_args(");
  const middlewareAt = launch.indexOf('argv += ["--middleware"');
  assert.ok(topologyAt >= 0, "launch.py appends topology_args after the resolved argv");
  assert.ok(middlewareAt > topologyAt, "launch.py appends --middleware after topology_args");
  // The site carries only the launcher options before --; the native args go after it,
  // exactly where launch.py feeds them to the resolver (request.native).
  const dash = server.args.indexOf("--");
  assert.ok(dash > 0);
  assert.deepEqual(server.args.slice(0, dash), ["launch", "--recipe", "qwen38-flash-next-gb10-tp2", "--topology", "lws", "--model-sync"]);
});

test("docker runs the launcher once per rank and refuses a cache or replica layout", () => {
  const docker = renderRecipe(tp2Recipe(), { parameters: { node_selector: { gpu: "gb10" } }, settings: {}, environment: {}, target: "docker", image: IMAGE });
  assert.deepEqual(docker.files.map((file) => file.name), ["serve.sh"]);
  const body = docker.files[0].body;
  assert.ok(body.includes("vllm-image"), "runs the launcher");
  assert.ok(!/\bbash\b/.test(body), "no bash");
  assert.match(body, /--recipe' 'qwen38-flash-next-gb10-tp2'/);
  assert.match(body, /--topology' 'lws'/);
  assert.match(body, /--model-sync/);

  assert.throws(() => renderRecipe(tp2Recipe(), { parameters: { node_selector: { gpu: "gb10" } }, settings: { "cache-mode": "lmcache" }, environment: {}, target: "docker", image: IMAGE }), /Compose/);
  assert.throws(() => renderRecipe(tp2Recipe(), { parameters: { node_selector: { gpu: "gb10" } }, settings: { replicas: "3" }, environment: {}, target: "docker", image: IMAGE }), /Compose/);

  // A plain single-rank run without a cache is a bare `docker run`, not a refusal.
  const single = renderRecipe(singleRecipe(), { parameters: { node_selector: { gpu: "gb10" } }, settings: {}, environment: {}, target: "docker", image: IMAGE });
  assert.match(single.files[0].body, /'docker' 'run' '--rm'/, "a plain docker run, not a refusal");
  assert.ok(!single.files[0].body.includes("Compose"), "the single-rank layout is not refused");
});

test("compose splits the cache into the model service and adds replicas behind a proxy", () => {
  const compose = renderRecipe(tp2Recipe(), { parameters: { node_selector: { gpu: "gb10" } }, settings: { "cache-mode": "lmcache", "cache-l1-gib": "64" }, environment: {}, target: "compose", image: IMAGE });
  assert.deepEqual(compose.files.map((file) => file.name), ["compose.yaml"]);
  const document = parseAllDocuments(compose.files[0].body)[0].toJS();
  assert.equal(document.services.probe.network_mode, "service:model");
  assert.equal(document.services.probe.ipc, "service:model");
  assert.equal(document.services.cache.network_mode, "service:model");
  assert.equal(document.services.cache.ipc, "service:model");
  assert.ok(document.services.cache.volumes.some((volume) => volume.startsWith("shm:")), "the cache shares the tmpfs volume");

  const replica = renderRecipe(singleRecipe(), { parameters: { node_selector: { gpu: "gb10" } }, settings: { replicas: "2" }, environment: {}, target: "compose", image: IMAGE });
  const replicaDoc = parseAllDocuments(replica.files[0].body)[0].toJS();
  const modelServices = Object.keys(replicaDoc.services).filter((name) => name.startsWith("model-"));
  assert.ok(modelServices.length >= 2, "one service per replica");
  assert.ok(Object.keys(replicaDoc.services).some((name) => name.includes("proxy")), "a proxy service fronts the replicas");
});

test("routing still renders the llm-d resources and references the served model name", () => {
  const routing = renderRecipe(tp2Recipe(), { parameters: { node_selector: { gpu: "gb10" } }, settings: {}, environment: {}, target: "routing" });
  assert.deepEqual(routing.files.map((file) => file.name), ["routing.yaml"]);
  const docs = documents(routing.files[0].body);
  const kinds = docs.map((document) => document.kind);
  for (const kind of ["InferencePool", "Gateway", "AIGatewayRoute", "ServiceAccount", "ClusterRole", "ClusterRoleBinding", "ConfigMap", "Deployment"]) {
    assert.ok(kinds.includes(kind), `routing keeps ${kind}`);
  }
  // The route matches on the recipe's served model name, not a hand-written command.
  const route = onlyDoc(docs, "AIGatewayRoute");
  assert.equal(route.spec.rules[0].matches[0].headers[0].value, "qwen38-flash-next");
  const text = renderedText(routing);
  assert.ok(!text.includes("vllm-image serve") && !text.includes("vllm serve"), "no hand-written vLLM command in the routing bundle");
});

test("invalid input throws TypeError before any output", () => {
  const base = { parameters: { node_selector: { gpu: "gb10" } }, settings: {}, environment: {}, target: "lws", image: IMAGE };
  assert.throws(() => renderRecipe(tp2Recipe(), { ...base, image: "ghcr.io/x/y:latest" }), TypeError, "a tag is refused");
  assert.throws(() => renderRecipe(tp2Recipe(), { ...base, image: "internal.randomvariable.co.uk/vllm@sha256:" + "a".repeat(64) }), TypeError, "a private registry reference is refused");
  assert.throws(() => renderRecipe(tp2Recipe(), { parameters: {}, settings: {}, environment: {}, target: "lws", image: IMAGE }), TypeError, "a missing required site field");
  assert.throws(() => renderRecipe({ meta: {}, launch: {}, model: {}, deployment: {} }, base), TypeError, "a malformed launch section");
  assert.throws(() => renderRecipe(tp2Recipe(), { ...base, target: "kubernetes" }), TypeError, "an unknown target");
  // qwen38-27b ships `deployment.parameters: {}`: no manifest target can render it, so
  // every one must refuse rather than emit undefined names and host paths.
  const bare = singleRecipe();
  bare.deployment = { parameters: {} };
  for (const target of ["lws", "docker", "compose", "routing"]) {
    assert.throws(() => renderRecipe(bare, { parameters: {}, settings: {}, environment: {}, target, image: IMAGE }), TypeError, `${target} refuses a recipe with no parameters`);
  }
});
