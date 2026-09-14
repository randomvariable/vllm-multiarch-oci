import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { parseAllDocuments } from "yaml";
import { renderRecipe } from "../src/render.js";

const siteRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const recipe = parseAllDocuments(readFileSync(resolve(siteRoot, "../recipes/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp.yaml"), "utf8"))[0].toJS();
const image = recipe.validation.image;
const parameters = {
  namespace: "public-models",
  name: "deepseek-vision",
  model_storage_path: "/srv/models/deepseek vision's snapshot",
  jit_storage_path: "/srv/cache/deepseek vision's jit",
  hf_secret: "huggingface-token",
  hf_secret_key: "HF_TOKEN",
  network_attachment: "roce-network",
  node_selector: { "node.example.com/accelerator": "gb10" },
  topology_key: "node.example.com/roce-pair",
  topology_values: "pair-a,pair-b",
  gpu_resource: "nvidia.com/gpu",
  gpu_units: 1,
  rdma_resource: "rdma.example.com/roce",
  rdma_units: 1,
  hca: "mlx5_0,mlx5_1",
  gid_index: 3,
  socket_interface: "eno1",
  gloo_interface: "eno1",
  leader_ip: "192.0.2.10",
  worker_ip: "192.0.2.11",
  gateway_class: "public-ai-gateway",
};

function documents(text) {
  return parseAllDocuments(text).map((document) => document.toJS());
}

function findDocument(values, kind) {
  return values.find((value) => value.kind === kind);
}

function modelContainers(lws) {
  const template = lws.spec.leaderWorkerTemplate;
  return [template.leaderTemplate.spec.containers[0], template.workerTemplate.spec.containers[0]];
}

function envMap(container) {
  return Object.fromEntries(container.env.filter((entry) => Object.hasOwn(entry, "value")).map((entry) => [entry.name, entry.value]));
}

test("LWS renders explicit rank templates from one fixed runtime definition", () => {
  const result = renderRecipe(recipe, parameters, image, "lws");
  const docs = documents(result.files["deepseek-vision-lws.yaml"]);
  const service = findDocument(docs, "Service");
  const lws = findDocument(docs, "LeaderWorkerSet");
  const [leader, worker] = modelContainers(lws);

  assert.deepEqual(service.spec.selector, { app: parameters.name, "leaderworkerset.sigs.k8s.io/worker-index": "0" });
  assert.equal(lws.spec.leaderWorkerTemplate.size, 2);
  assert.equal(lws.spec.startupPolicy, "LeaderCreated");
  assert.equal(lws.spec.networkConfig.subdomainPolicy, "Shared");
  assert.equal(lws.metadata.annotations["leaderworkerset.sigs.k8s.io/exclusive-topology"], parameters.topology_key);
  assert.equal(lws.spec.leaderWorkerTemplate.leaderTemplate.spec.runtimeClassName, "nvidia");
  assert.equal(lws.spec.leaderWorkerTemplate.workerTemplate.spec.runtimeClassName, "nvidia");

  assert.deepEqual(leader.command, recipe.runtime.command);
  assert.deepEqual(worker.command, recipe.runtime.command);
  assert.deepEqual(leader.args.slice(1, 1 + recipe.runtime.base_args.length), recipe.runtime.base_args);
  assert.deepEqual(worker.args.slice(1, 1 + recipe.runtime.base_args.length), recipe.runtime.base_args);
  assert.equal(leader.args[leader.args.indexOf("--node-rank") + 1], "0");
  assert.equal(worker.args[worker.args.indexOf("--node-rank") + 1], "1");
  assert.equal(leader.args[leader.args.indexOf("--master-addr") + 1], "127.0.0.1");
  assert.equal(worker.args[worker.args.indexOf("--master-addr") + 1], "$(LWS_LEADER_ADDRESS)");
  assert(!leader.args.includes("--headless"));
  assert(worker.args.includes("--headless"));
  assert(!leader.args.includes("--moe-backend"));
  assert.equal(leader.args[leader.args.indexOf("--attention-backend") + 1], "B12X");
  assert.equal(leader.args[leader.args.indexOf("--linear-backend") + 1], "b12x");
  assert.deepEqual(JSON.parse(leader.args[leader.args.indexOf("--compilation-config") + 1]).cudagraph_capture_sizes, [1, 2, 3, 4, 8, 12, 16]);

  const leaderEnv = envMap(leader);
  const workerEnv = envMap(worker);
  assert.deepEqual(leaderEnv, workerEnv);
  assert.deepEqual(Object.fromEntries(Object.entries(leaderEnv).filter(([key]) => Object.hasOwn(recipe.runtime.base_env, key))), recipe.runtime.base_env);
  assert.equal(leaderEnv.NCCL_IB_HCA, parameters.hca);
  assert.equal(leaderEnv.NCCL_SOCKET_IFNAME, parameters.socket_interface);
  assert.equal(leader.env.find((entry) => entry.name === "VLLM_HOST_IP").valueFrom.fieldRef.fieldPath, "status.podIP");

  const leaderInit = lws.spec.leaderWorkerTemplate.leaderTemplate.spec.initContainers;
  const workerInit = lws.spec.leaderWorkerTemplate.workerTemplate.spec.initContainers;
  assert.deepEqual(leaderInit.map((container) => container.name), ["model-sync"]);
  assert.deepEqual(workerInit.map((container) => container.name), ["model-sync", "rendezvous-wait"]);
  assert.equal(leaderInit[0].args[leaderInit[0].args.indexOf("--revision") + 1], recipe.model.revision);
  assert.deepEqual(leaderInit[0].args.filter((value, index, args) => args[index - 1] === "--ignore"), recipe.deployment.ignore_patterns);
  assert.deepEqual(workerInit[1].args, ["rendezvous-wait", "--rank-env", "LWS_WORKER_INDEX", "--leader-env", "LWS_LEADER_ADDRESS", "--port", "25000", "--dns-timeout", "1200s", "--connect-timeout", "300s"]);

  for (const pod of [lws.spec.leaderWorkerTemplate.leaderTemplate.spec, lws.spec.leaderWorkerTemplate.workerTemplate.spec]) {
    const server = pod.containers[0];
    assert.equal(server.image, image);
    assert.equal(server.livenessProbe, undefined);
    assert.equal(server.startupProbe.failureThreshold, 240);
    assert.equal(server.readinessProbe.failureThreshold, 6);
    assert.equal(server.resources.limits.memory, undefined);
    assert.equal(server.resources.requests[parameters.gpu_resource], "1");
    assert.equal(server.resources.requests[parameters.rdma_resource], "1");
    assert.equal(pod.terminationGracePeriodSeconds, 120);
    assert.deepEqual(pod.volumes.find((volume) => volume.name === "shm").emptyDir, { medium: "Memory", sizeLimit: "64Gi" });
  }

  const serialized = JSON.stringify(docs);
  assert(!serialized.includes("harbor.services.home.internal"));
  assert(!serialized.includes("openai"));
  assert(!serialized.includes("dspark.rv"));
  assert(!serialized.includes("rdma/dgx_roce"));
  for (const pod of [lws.spec.leaderWorkerTemplate.leaderTemplate.spec, lws.spec.leaderWorkerTemplate.workerTemplate.spec]) {
    assert.deepEqual(pod.containers[0].resources.limits, { [parameters.gpu_resource]: "1", [parameters.rdma_resource]: "1" });
  }
});

test("Docker scripts preserve literal paths, isolate token use, and pass shell syntax", () => {
  const result = renderRecipe(recipe, parameters, image, "docker");
  const leader = result.files["deepseek-vision-leader.sh"];
  const worker = result.files["deepseek-vision-worker.sh"];
  const directory = mkdtempSync(join(siteRoot, ".render-test-"));
  const leaderPath = join(directory, "leader.sh");
  const workerPath = join(directory, "worker.sh");
  writeFileSync(leaderPath, leader);
  writeFileSync(workerPath, worker);
  try {
    execFileSync("bash", ["-n", leaderPath]);
    execFileSync("bash", ["-n", workerPath]);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }

  for (const script of [leader, worker]) {
    assert.match(script, /^#!\/bin\/sh\nset -eu\n/);
    assert(script.includes("'--network' 'host'"));
    assert(script.includes("'--gpus' 'all'"));
    assert(script.includes("'--device' '/dev/infiniband'"));
    assert(script.includes("'--cap-add' 'IPC_LOCK'"));
    assert(script.includes("'--ulimit' 'memlock=-1'"));
    assert(script.includes("'--shm-size' '64g'"));
    assert(script.includes("'--stop-timeout' '120'"));
    assert(script.includes("'--detach' '--name'"));
    assert(!script.includes("'--privileged'"));
    assert(!script.includes("'--rm'"));
    assert(script.includes("deepseek vision'\"'\"'s snapshot"));
    assert.equal((script.match(/'-e' 'HF_TOKEN'/g) || []).length, 1);
    assert(!script.includes("HF_TOKEN="));
    assert(script.includes("'--entrypoint' '/opt/venv/bin/python'"));
    assert(script.includes(`${image}`));
  }
  assert(!leader.includes("rendezvous-wait"));
  assert(worker.includes("'rendezvous-wait' '--rank' '1' '--leader' '192.0.2.10'"));
  assert(worker.includes("'--headless'"));
  assert(leader.includes("'VLLM_HOST_IP=192.0.2.10'"));
  assert(worker.includes("'VLLM_HOST_IP=192.0.2.11'"));
  assert(result.steps.some((step) => step.text.includes("not grounds for force-killing")));
  assert(result.steps.some((step) => step.command.includes("docker' 'exec")));
  assert(result.steps.some((step) => step.command.includes("'stop' '--time' '120'")));
});

test("llm-d routing selects only rank zero and fails closed through EPP", () => {
  const result = renderRecipe(recipe, parameters, image, "llm-d-routing");
  const docs = documents(result.files["deepseek-vision-llm-d-routing.yaml"]);
  const pool = findDocument(docs, "InferencePool");
  const gateway = findDocument(docs, "Gateway");
  const route = findDocument(docs, "AIGatewayRoute");
  const policy = findDocument(docs, "ClientTrafficPolicy");
  const epp = findDocument(docs, "Deployment");
  const config = parseAllDocuments(findDocument(docs, "ConfigMap").data["config.yaml"])[0].toJS();

  assert.deepEqual(pool.spec.selector.matchLabels, { app: parameters.name, "leaderworkerset.sigs.k8s.io/worker-index": "0" });
  assert.deepEqual(pool.spec.endpointPickerRef, { name: "deepseek-vision-epp", port: { number: 9002 }, failureMode: "FailClose" });
  assert.equal(gateway.spec.gatewayClassName, parameters.gateway_class);
  assert.equal(gateway.spec.listeners[0].port, 10080);
  assert.equal(route.spec.rules[0].matches[0].headers[0].value, recipe.model.served_name);
  assert.deepEqual(route.spec.rules[0].backendRefs[0], { group: "inference.networking.k8s.io", kind: "InferencePool", name: "deepseek-vision-pool" });
  assert.equal(route.spec.rules[0].timeouts.request, "1800s");
  assert.equal(policy.spec.connection.bufferLimit, "50Mi");
  assert.equal(epp.spec.template.spec.containers[0].image, "registry.k8s.io/gateway-api-inference-extension/epp:v1.5.0");
  assert(epp.spec.template.spec.containers[0].args.includes("--metrics-endpoint-auth=false"));
  assert.deepEqual(config.schedulingProfiles[0].plugins, [
    { pluginRef: "queue-scorer", weight: 2 },
    { pluginRef: "kv-cache-utilization-scorer", weight: 2 },
    { pluginRef: "prefix-cache-scorer", weight: 3 },
  ]);
  assert(!JSON.stringify(docs).includes("internal.randomvariable"));
});

test("required target fields and digest-only images fail before output", () => {
  assert.throws(() => renderRecipe(recipe, { ...parameters, network_attachment: "" }, image, "lws"), /network_attachment is required/);
  assert.throws(() => renderRecipe(recipe, { ...parameters, leader_ip: "" }, image, "docker"), /leader_ip is required/);
  assert.throws(() => renderRecipe(recipe, { ...parameters, gateway_class: "" }, image, "llm-d-routing"), /gateway_class is required/);
  assert.throws(() => renderRecipe(recipe, parameters, "ghcr.io/randomvariable/vllm-b12x-multi:latest", "lws"), /digest-qualified/);
  assert.throws(() => renderRecipe(recipe, { ...parameters, namespace: "x; touch /tmp/leak" }, image, "lws"), /namespace must be a DNS label/);
  assert.throws(() => renderRecipe(recipe, { ...parameters, hca: "mlx5_0; touch /tmp/leak" }, image, "docker"), /hca contains unsupported characters/);
});
