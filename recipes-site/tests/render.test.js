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
// (model_path + storage_root for a sync deployment, the cluster shape the renderer reads,
// and parameters for the site fields).

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

// The GB10 TP=2 cluster shape, read straight from the accepted live manifest: three
// groups, a no-surge rolling rollout, the dgx placement/taint, four GPUs plus the RDMA
// allocator on both maps, root with IPC_LOCK, pod group 0, DirectoryOrCreate hostPaths
// and the two-hour startup / 6x15s readiness httpGet windows. The renderer owns none of
// these as a literal; every value here is what the recipe must carry.
function gb10Cluster(overrides = {}) {
  return {
    groups: 3,
    restartPolicy: "RecreateGroupAfterStart",
    rollout: { type: "RollingUpdate", maxUnavailable: 1, maxSurge: 0, partition: 0 },
    nodeSelector: { "kubernetes.io/arch": "arm64", "node-role.kubernetes.io/dgx": "" },
    tolerations: [{ key: "dgx", operator: "Equal", value: "true", effect: "NoSchedule" }],
    gpu: 4,
    rdma: true,
    resources: {
      requests: { cpu: "8", memory: "96Gi", "ephemeral-storage": "128Gi" },
      limits: {},
    },
    securityContext: { runAsUser: 0, capabilities: { add: ["IPC_LOCK"] } },
    podSecurityContext: { fsGroup: 0 },
    volumes: { hostPathType: "DirectoryOrCreate" },
    probes: {
      startup: { periodSeconds: 30, timeoutSeconds: 5, failureThreshold: 240 },
      readiness: { periodSeconds: 15, timeoutSeconds: 5, failureThreshold: 6 },
    },
    ...overrides,
  };
}

// The single-node Deployment cluster shape (qwen38-27b): one GPU, no RDMA, requests and
// limits that legitimately differ on memory and storage, a hostname-only selector with NO
// architecture key, root plus runAsGroup 0 and NO capabilities, and no liveness.
function singleCluster(overrides = {}) {
  return {
    nodeSelector: {},
    gpu: 1,
    rdma: false,
    resources: {
      requests: { cpu: "8", memory: "48Gi", "ephemeral-storage": "4Gi" },
      limits: { memory: "72Gi", "ephemeral-storage": "8Gi" },
    },
    securityContext: { runAsUser: 0, runAsGroup: 0 },
    volumes: { hostPathType: "DirectoryOrCreate" },
    ...overrides,
  };
}

// A two-node RoCE TP=2 recipe in launcher form: sync mode (deployment.model_path set),
// no external cache (the reader can add one with `cache-mode`). The GPU count and RDMA
// units are kept at 1 so the pre-existing GPU/RDMA assertions still hold against the
// parametersMap defaults, while the cluster shape is otherwise the accepted GB10 one.
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
      ...gb10Cluster({
        gpu: 1,
        probes: {
          startup: { periodSeconds: 30, timeoutSeconds: 5, failureThreshold: 240 },
          readiness: { periodSeconds: 15, timeoutSeconds: 5, failureThreshold: 6 },
        },
      }),
    },
    ...overrides,
  };
}

// An engine-download single-node recipe: no model_path, so no `--model-sync`, and no
// sync-only deployment facts at all — the renderer must tolerate their absence.
function singleRecipe(overrides = {}) {
  return {
    meta: { slug: "qwen38-27b", title: "t", description: "d" },
    model: { model_id: "local-inference-lab/Qwen3.8-27B-NVFP4-QAD", revision: "f40a31cd813a6746067e7d6446ff2cb708dbb779", served_name: "qwen3.8-27b" },
    launch: {
      profile: "qwen38-flash-next",
      hardware: "gb10-roce",
      preset: null,
      options: { port: 8888, "served-model-name": "qwen3.8-27b", model: "local-inference-lab/Qwen3.8-27B-NVFP4-QAD" },
      environment: { HF_HOME: "/models" },
      topology: { kind: "single", nodes: 1, rendezvous_port: null, kv_events: null, replica_port_base: null },
      probe_port: 8890,
    },
    deployment: {
      parameters: parametersMap(),
      ...singleCluster(),
    },
    ...overrides,
  };
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

function renderLws(recipe, parameters = {}, target = "lws") {
  return renderRecipe(recipe, { parameters: { node_selector: { gpu: "gb10" }, ...parameters }, settings: {}, environment: {}, target, image: IMAGE });
}

test("LWS renders the launcher form with one worker template and no hand-written command", () => {
  const result = renderLws(tp2Recipe());
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
  const result = renderLws(tp2Recipe());
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

test("startup and readiness probes are httpGet on the recipe's declared /readyz window", () => {
  const spec = podSpec(documents(renderLws(tp2Recipe()).files[0].body));
  const server = spec.containers[0];
  assert.deepEqual(server.startupProbe.httpGet, { path: "/readyz", port: 8890 });
  assert.deepEqual(server.readinessProbe.httpGet, { path: "/readyz", port: 8890 });
  // The windows come from the recipe, not a renderer literal.
  assert.deepEqual(
    { periodSeconds: server.startupProbe.periodSeconds, timeoutSeconds: server.startupProbe.timeoutSeconds, failureThreshold: server.startupProbe.failureThreshold },
    { periodSeconds: 30, timeoutSeconds: 5, failureThreshold: 240 },
  );
  assert.deepEqual(
    { periodSeconds: server.readinessProbe.periodSeconds, timeoutSeconds: server.readinessProbe.timeoutSeconds, failureThreshold: server.readinessProbe.failureThreshold },
    { periodSeconds: 15, timeoutSeconds: 5, failureThreshold: 6 },
  );
  assert.equal(server.startupProbe.exec, undefined, "no exec probe");
  assert.equal(server.readinessProbe.exec, undefined, "no exec probe");
});

test("no livenessProbe is emitted unless the recipe declares a measured one", () => {
  // The accepted recipe carries no liveness block: its window has to be measured on both
  // ranks, and an invented one killed healthy groups, so absence is the correct output.
  const spec = podSpec(documents(renderLws(tp2Recipe()).files[0].body));
  assert.equal(spec.containers[0].livenessProbe, undefined, "no livenessProbe without a measured declaration");

  // A measured liveness window is honoured.
  const measured = tp2Recipe();
  measured.deployment.probes = { ...measured.deployment.probes, liveness: { measured: true, periodSeconds: 20, timeoutSeconds: 5, failureThreshold: 6 } };
  const measuredServer = podSpec(documents(renderLws(measured).files[0].body)).containers[0];
  assert.deepEqual(measuredServer.livenessProbe.httpGet, { path: "/livez", port: 8890 });
  assert.equal(measuredServer.livenessProbe.failureThreshold, 6);

  // A declared-but-unmeasured liveness window is refused before any output.
  const unmeasured = tp2Recipe();
  unmeasured.deployment.probes = { ...unmeasured.deployment.probes, liveness: { periodSeconds: 20, timeoutSeconds: 5, failureThreshold: 6 } };
  assert.throws(() => renderLws(unmeasured), /not marked measured|measured/);
});

test("the modelserver carries the GPU and RDMA requests and IPC_LOCK", () => {
  const spec = podSpec(documents(renderLws(tp2Recipe()).files[0].body));
  const server = spec.containers[0];
  assert.equal(server.resources.requests["nvidia.com/gpu"], "1");
  assert.equal(server.resources.requests["rdma.example.com/roce"], "1");
  assert.deepEqual(server.securityContext.capabilities.add, ["IPC_LOCK"]);
});

test("requests and limits carry the same GPU count from one recipe field", () => {
  const spec = podSpec(documents(renderLws(tp2Recipe()).files[0].body));
  const { requests, limits } = spec.containers[0].resources;
  assert.equal(requests["nvidia.com/gpu"], limits["nvidia.com/gpu"], "the GPU count cannot disagree between the maps");
  assert.equal(requests["rdma.example.com/roce"], limits["rdma.example.com/roce"], "the RDMA resource cannot disagree between the maps");
});

test("the rendered Qwen LWS equals the accepted GB10 live manifest on every contract field", () => {
  // Rendered against the live site parameters so the comparison is on parsed values,
  // not text -- the `1` versus `"1"` quantity spelling can never bite a value check.
  const recipe = tp2Recipe();
  recipe.deployment.gpu = 4;
  const result = renderRecipe(recipe, {
    parameters: {
      namespace: "openai", name: "qwen38-flash-next",
      model_storage_path: "/var/lib/vllm-models/qwen38-flash-next-lil-qad",
      jit_storage_path: "/var/lib/vllm-qwen38-flash-next-cache",
      hf_secret: "llm-d-hf-token", hf_secret_key: "HF_TOKEN",
      network_attachment: "dspark-roce", node_selector: { "node-role.kubernetes.io/dgx": "" },
      topology_key: "dspark.rv/roce-pair", topology_values: "a,b,c",
      rdma_resource: "rdma/dgx_roce", rdma_units: 63,
      hca: "rocep1s0f1,roceP2p1s0f1", gid_index: 3,
    },
    settings: {}, environment: {}, target: "lws", image: IMAGE,
  });
  const lws = onlyDoc(documents(result.files[0].body), "LeaderWorkerSet");
  const pod = lws.spec.leaderWorkerTemplate.workerTemplate.spec;
  const server = pod.containers[0];

  assert.equal(lws.spec.replicas, 3, "spec.replicas is the group count");
  assert.equal(lws.spec.leaderWorkerTemplate.size, 2, "the TP group is two nodes");
  assert.equal(lws.spec.leaderWorkerTemplate.restartPolicy, "RecreateGroupAfterStart");
  assert.deepEqual(lws.spec.rolloutStrategy, {
    type: "RollingUpdate",
    rollingUpdateConfiguration: { maxUnavailable: 1, maxSurge: 0, partition: 0 },
  }, "the four rollout fields the live group carries");
  assert.deepEqual(pod.nodeSelector, { "kubernetes.io/arch": "arm64", "node-role.kubernetes.io/dgx": "" });
  assert.deepEqual(pod.tolerations, [{ key: "dgx", operator: "Equal", value: "true", effect: "NoSchedule" }]);
  assert.deepEqual(pod.securityContext, { fsGroup: 0 }, "the pod group the weights run under");
  assert.deepEqual(server.resources.requests, {
    cpu: "8", memory: "96Gi", "ephemeral-storage": "128Gi", "nvidia.com/gpu": "4", "rdma/dgx_roce": "63",
  });
  assert.deepEqual(server.resources.limits, { "nvidia.com/gpu": "4", "rdma/dgx_roce": "63" });
  assert.deepEqual(server.securityContext, { runAsUser: 0, capabilities: { add: ["IPC_LOCK"] } });
  assert.deepEqual(server.startupProbe.httpGet, { path: "/readyz", port: 8890 });
  assert.deepEqual(server.readinessProbe.httpGet, { path: "/readyz", port: 8890 });
  assert.equal(server.livenessProbe, undefined, "no liveness window until /livez is measured");
  // The hostPaths are the live node paths, typed by the recipe.
  assert.deepEqual(pod.volumes.find((volume) => volume.name === "jit-cache").hostPath, { path: "/var/lib/vllm-qwen38-flash-next-cache", type: "DirectoryOrCreate" });
  assert.deepEqual(pod.volumes.find((volume) => volume.name === "model-weights").hostPath, { path: "/var/lib/vllm-models/qwen38-flash-next-lil-qad", type: "DirectoryOrCreate" });
  // And the GPU count that feeds both maps traces to one field.
  assert.equal(server.resources.requests["nvidia.com/gpu"], server.resources.limits["nvidia.com/gpu"]);
});

test("the single-node Deployment drops the group shape the live pod does not carry", () => {
  // qwen38-27b lives on one host: no arch key, no tolerations, no capabilities, and its
  // limits legitimately differ from its requests on memory and storage.
  const result = renderRecipe(singleRecipe(), {
    parameters: { node_selector: { "kubernetes.io/hostname": "server21" } },
    settings: {}, environment: {}, target: "lws", image: IMAGE,
  });
  const deployment = onlyDoc(documents(result.files[0].body), "Deployment");
  const pod = deployment.spec.template.spec;
  const server = pod.containers[0];
  assert.deepEqual(pod.nodeSelector, { "kubernetes.io/hostname": "server21" }, "no architecture key is injected");
  assert.equal(pod.tolerations, undefined, "no tolerations key when the recipe declares none");
  assert.equal(pod.securityContext, undefined, "no pod security context when the recipe declares none");
  assert.deepEqual(server.securityContext, { runAsUser: 0, runAsGroup: 0 }, "root, no IPC_LOCK");
  assert.deepEqual(server.resources.requests, { cpu: "8", memory: "48Gi", "ephemeral-storage": "4Gi", "nvidia.com/gpu": "1" });
  assert.deepEqual(server.resources.limits, { memory: "72Gi", "ephemeral-storage": "8Gi", "nvidia.com/gpu": "1" });
  assert.equal(server.resources.requests["nvidia.com/gpu"], server.resources.limits["nvidia.com/gpu"], "the GPU count still agrees");
  assert.equal(server.resources.requests["rdma/dgx_roce"], undefined, "no RDMA resource on a node that never opens the device");
  // No declared probes: the launcher-form single-node defaults apply, and there is no liveness.
  assert.equal(server.startupProbe.httpGet.path, "/readyz");
  assert.equal(server.livenessProbe, undefined);
});

test("the reader's settings become -- changes and defaults stay absent", () => {
  const plain = renderLws(tp2Recipe());
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
  const noCache = renderLws(tp2Recipe());
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
  const noCache = podSpec(documents(renderLws(tp2Recipe()).files[0].body));
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
  const server = podSpec(documents(renderLws(tp2Recipe()).files[0].body)).containers[0];
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

  // A manifest target reads its cluster shape from the recipe; a recipe that omits the
  // GPU count or the hostPath type cannot render, and must fail rather than emit an
  // undefined or invented value. (An absent resource map is tolerated: the pod simply
  // requests its devices, and inventing no cpu/memory default is the point.)
  const noGpu = tp2Recipe();
  delete noGpu.deployment.gpu;
  assert.throws(() => renderLws(noGpu), TypeError, "the GPU count must come from the recipe");
  const noHostPathType = tp2Recipe();
  delete noHostPathType.deployment.volumes;
  assert.throws(() => renderLws(noHostPathType), TypeError, "the hostPath type must come from the recipe");

  // qwen38-27b ships `deployment.parameters: {}`: no manifest target can render it, so
  // every one must refuse rather than emit undefined names and host paths.
  const bare = singleRecipe();
  bare.deployment = { parameters: {} };
  for (const target of ["lws", "docker", "compose", "routing"]) {
    assert.throws(() => renderRecipe(bare, { parameters: {}, settings: {}, environment: {}, target, image: IMAGE }), TypeError, `${target} refuses a recipe with no parameters`);
  }
});
