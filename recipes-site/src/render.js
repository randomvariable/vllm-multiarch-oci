import { stringify as stringifyYaml } from "yaml";

const DIGEST_REFERENCE = /^[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?(?:\/[a-z0-9]+(?:[._-][a-z0-9]+)*)+@sha256:[0-9a-f]{64}$/;
const DNS_LABEL = /^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$/;
const DNS_SUBDOMAIN = /^[a-z0-9](?:[-a-z0-9.]*[a-z0-9])?$/;
const RESOURCE_NAME = /^(?:[a-z0-9](?:[-a-z0-9.]*[a-z0-9])?\/)?[A-Za-z0-9](?:[-A-Za-z0-9_.]*[A-Za-z0-9])?$/;
const SAFE_INTERFACE = /^[A-Za-z0-9_.:,=-]+$/;
const TARGET_FIELDS = {
  lws: [
    "namespace", "name", "model_storage_path", "jit_storage_path", "hf_secret", "hf_secret_key",
    "network_attachment", "node_selector", "topology_key", "topology_values", "gpu_resource", "gpu_units",
    "rdma_resource", "rdma_units", "hca", "gid_index", "socket_interface", "gloo_interface",
  ],
  docker: ["name", "model_storage_path", "jit_storage_path", "hca", "gid_index", "socket_interface", "gloo_interface", "leader_ip", "worker_ip"],
  "llm-d-routing": ["namespace", "name", "gateway_class"],
};

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

function validateRecipe(recipe) {
  if (!recipe || typeof recipe !== "object") fail("recipe must be an object");
  for (const key of ["meta", "model", "runtime", "deployment", "validation", "guide"]) {
    if (!recipe[key] || typeof recipe[key] !== "object") fail(`recipe.${key} is required`);
  }
  if (!Array.isArray(recipe.runtime.command) || !recipe.runtime.command.every((value) => typeof value === "string")) fail("runtime.command must be a string array");
  if (!Array.isArray(recipe.runtime.base_args) || !recipe.runtime.base_args.every((value) => typeof value === "string")) fail("runtime.base_args must be a string array");
  if (!recipe.runtime.base_env || typeof recipe.runtime.base_env !== "object" || Array.isArray(recipe.runtime.base_env)) fail("runtime.base_env must be a mapping");
  if (!Object.values(recipe.runtime.base_env).every((value) => typeof value === "string")) fail("runtime.base_env values must be strings");
  if (recipe.deployment.nodes !== 2 || recipe.deployment.tensor_parallel_size !== 2) fail("this renderer requires the fixed two-node TP=2 topology");
  if (!/^[0-9a-f]{40}$/.test(recipe.model.revision)) fail("model.revision must be a full lowercase commit OID");
  if (!DIGEST_REFERENCE.test(recipe.validation.image)) fail("validation.image must be digest-qualified");
}

function normalizedParameters(recipe, supplied, target) {
  if (!(target in TARGET_FIELDS)) fail(`unsupported render target: ${target}`);
  const definitions = recipe.deployment.parameters;
  if (!definitions || typeof definitions !== "object") fail("deployment.parameters must be a mapping");
  const values = {};
  for (const [name, definition] of Object.entries(definitions)) {
    if (!["string", "integer", "stringMap"].includes(definition.type)) fail(`invalid parameter type for ${name}`);
    const value = Object.hasOwn(supplied, name) ? supplied[name] : definition.default;
    if (TARGET_FIELDS[target].includes(name) && definition.required && !hasValue(value, definition.type)) fail(`${name} is required for ${target}`);
    if (definition.type === "integer" && hasValue(value, definition.type) && (!Number.isSafeInteger(Number(value)) || Number(value) < 0)) fail(`${name} must be a non-negative integer`);
    if (definition.type === "stringMap" && value !== undefined && (value === null || typeof value !== "object" || Array.isArray(value) || !Object.values(value).every((entry) => typeof entry === "string"))) fail(`${name} must be a string mapping`);
    values[name] = definition.type === "integer" && hasValue(value, definition.type) ? Number(value) : value;
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
  for (const field of ["gpu_resource", "rdma_resource", "topology_key"]) {
    if (hasValue(values[field], "string") && !RESOURCE_NAME.test(values[field])) fail(`${field} is not a valid Kubernetes resource or label name`);
  }
  for (const field of ["hca", "socket_interface", "gloo_interface"]) {
    if (hasValue(values[field], "string") && !SAFE_INTERFACE.test(values[field])) fail(`${field} contains unsupported characters`);
  }
  return values;
}

function validateImageReference(imageReference) {
  if (!DIGEST_REFERENCE.test(imageReference)) fail("imageReference must be digest-qualified");
  if (imageReference.includes("internal.randomvariable")) fail("private registry references are not allowed");
}

function modelSyncArgs(recipe) {
  const args = [
    "model-sync", "--repo", recipe.model.model_id, "--revision", recipe.model.revision,
    "--storage-root", "/models", "--publish", recipe.deployment.model_path,
    "--min-free-gib", String(recipe.deployment.storage_min_free_gib), "--workers", String(recipe.deployment.download_workers),
  ];
  for (const pattern of recipe.deployment.ignore_patterns) args.push("--ignore", pattern);
  args.push("--token-env", "HF_TOKEN");
  return args;
}

function commonRuntimeEnv(recipe, parameters) {
  return {
    ...recipe.runtime.base_env,
    NCCL_IB_HCA: parameters.hca,
    NCCL_IB_GID_INDEX: String(parameters.gid_index),
    NCCL_SOCKET_IFNAME: parameters.socket_interface,
    GLOO_SOCKET_IFNAME: parameters.gloo_interface,
  };
}

function servingArgs(recipe, rank, masterAddress) {
  return [
    recipe.deployment.model_path,
    ...recipe.runtime.base_args,
    "--nnodes", String(recipe.deployment.nodes),
    "--node-rank", String(rank),
    "--master-addr", masterAddress,
    "--tensor-parallel-size", String(recipe.deployment.tensor_parallel_size),
    ...(rank === 1 ? ["--headless"] : []),
  ];
}

function envList(environment) {
  return Object.entries(environment).map(([name, value]) => ({ name, value: String(value) }));
}

function modelSyncContainer(recipe, parameters, imageReference) {
  return {
    name: "model-sync",
    image: imageReference,
    imagePullPolicy: "IfNotPresent",
    securityContext: { runAsUser: 0 },
    command: ["/opt/venv/bin/vllm-image"],
    args: modelSyncArgs(recipe),
    env: [
      { name: "HF_XET_HIGH_PERFORMANCE", value: "1" },
      { name: "HF_TOKEN", valueFrom: { secretKeyRef: { name: parameters.hf_secret, key: parameters.hf_secret_key } } },
    ],
    resources: { requests: { cpu: "4", memory: "12Gi" }, limits: { memory: "32Gi" } },
    volumeMounts: [
      { name: "model-weights", mountPath: "/models" },
      { name: "jit-cache", mountPath: "/cache" },
    ],
  };
}

function rendezvousContainer(imageReference) {
  return {
    name: "rendezvous-wait",
    image: imageReference,
    imagePullPolicy: "IfNotPresent",
    command: ["/opt/venv/bin/vllm-image"],
    args: ["rendezvous-wait", "--rank-env", "LWS_WORKER_INDEX", "--leader-env", "LWS_LEADER_ADDRESS", "--port", "25000", "--dns-timeout", "1200s", "--connect-timeout", "300s"],
  };
}

function healthProbe(phase) {
  const startup = phase === "startup";
  return {
    exec: { command: [
      "/opt/venv/bin/vllm-image", "health", "--phase", phase, "--rank-env", "LWS_WORKER_INDEX",
      "--local-url", "http://127.0.0.1:8888/v1/models", "--leader-host-env", "LWS_LEADER_ADDRESS",
      "--leader-port", "8888", "--leader-path", "/v1/models", "--engine-parent-pid", "1", "--timeout", startup ? "8s" : "15s",
    ] },
    ...(startup ? { initialDelaySeconds: 60, periodSeconds: 30, timeoutSeconds: 15, failureThreshold: 240 } : { periodSeconds: 15, timeoutSeconds: 20, failureThreshold: 6 }),
  };
}

function podTemplate(recipe, parameters, imageReference, rank) {
  const applicationLabels = {
    app: parameters.name,
    "llm-d.ai/model": parameters.name,
    "llm-d.ai/engine-type": "vllm",
  };
  const environment = commonRuntimeEnv(recipe, parameters);
  environment.VLLM_HOST_IP = undefined;
  const modelserver = {
    name: "modelserver",
    image: imageReference,
    imagePullPolicy: "IfNotPresent",
    securityContext: { runAsUser: 0, capabilities: { add: ["IPC_LOCK", "SYS_PTRACE"] } },
    command: recipe.runtime.command,
    args: servingArgs(recipe, rank, rank === 0 ? "127.0.0.1" : "$(LWS_LEADER_ADDRESS)"),
    env: [
      ...envList(Object.fromEntries(Object.entries(environment).filter(([, value]) => value !== undefined))),
      { name: "VLLM_HOST_IP", valueFrom: { fieldRef: { fieldPath: "status.podIP" } } },
    ],
    ports: [
      { name: "modelserver", containerPort: 8888, protocol: "TCP" },
      { name: "rdzv", containerPort: 25000, protocol: "TCP" },
    ],
    resources: {
      requests: { cpu: "8", memory: "96Gi", "ephemeral-storage": "128Gi", [parameters.gpu_resource]: String(parameters.gpu_units), [parameters.rdma_resource]: String(parameters.rdma_units) },
      limits: { [parameters.gpu_resource]: String(parameters.gpu_units), [parameters.rdma_resource]: String(parameters.rdma_units) },
    },
    startupProbe: healthProbe("startup"),
    readinessProbe: healthProbe("readiness"),
    volumeMounts: [
      { name: "shm", mountPath: "/dev/shm" },
      { name: "jit-cache", mountPath: "/cache" },
      { name: "model-weights", mountPath: "/models", readOnly: true },
    ],
  };
  const initContainers = [modelSyncContainer(recipe, parameters, imageReference)];
  if (rank === 1) initContainers.push(rendezvousContainer(imageReference));
  const pod = {
    metadata: {
      labels: applicationLabels,
      annotations: { "k8s.v1.cni.cncf.io/networks": parameters.network_attachment },
    },
    spec: {
      runtimeClassName: "nvidia",
      enableServiceLinks: false,
      subdomain: parameters.name,
      nodeSelector: { "kubernetes.io/arch": "arm64", ...parameters.node_selector },
      affinity: {
        nodeAffinity: { requiredDuringSchedulingIgnoredDuringExecution: { nodeSelectorTerms: [{ matchExpressions: [{ key: parameters.topology_key, operator: "In", values: parameters.topology_values.split(",").map((value) => value.trim()).filter(Boolean) }] }] } },
        podAntiAffinity: { requiredDuringSchedulingIgnoredDuringExecution: [{ labelSelector: { matchLabels: { app: parameters.name } }, topologyKey: "kubernetes.io/hostname" }] },
      },
      terminationGracePeriodSeconds: 120,
      securityContext: { fsGroup: 0 },
      initContainers,
      containers: [modelserver],
      volumes: [
        { name: "shm", emptyDir: { medium: "Memory", sizeLimit: "64Gi" } },
        { name: "jit-cache", hostPath: { path: parameters.jit_storage_path, type: "DirectoryOrCreate" } },
        { name: "model-weights", hostPath: { path: parameters.model_storage_path, type: "DirectoryOrCreate" } },
      ],
    },
  };
  return pod;
}

function renderLws(recipe, parameters, imageReference) {
  const service = {
    apiVersion: "v1", kind: "Service",
    metadata: { name: `${parameters.name}-serve`, namespace: parameters.namespace, labels: { app: parameters.name } },
    spec: { selector: { app: parameters.name, "leaderworkerset.sigs.k8s.io/worker-index": "0" }, ports: [{ name: "modelserver", port: 8888, targetPort: "modelserver" }] },
  };
  const lws = {
    apiVersion: "leaderworkerset.x-k8s.io/v1", kind: "LeaderWorkerSet",
    metadata: { name: parameters.name, namespace: parameters.namespace, labels: { app: parameters.name, "llm-d.ai/model": parameters.name }, annotations: { "leaderworkerset.sigs.k8s.io/exclusive-topology": parameters.topology_key } },
    spec: {
      replicas: 1,
      startupPolicy: "LeaderCreated",
      networkConfig: { subdomainPolicy: "Shared" },
      rolloutStrategy: { type: "RollingUpdate", rollingUpdateConfiguration: { maxUnavailable: 1, maxSurge: 0 } },
      leaderWorkerTemplate: {
        size: 2,
        restartPolicy: "RecreateGroupAfterStart",
        leaderTemplate: podTemplate(recipe, parameters, imageReference, 0),
        workerTemplate: podTemplate(recipe, parameters, imageReference, 1),
      },
    },
  };
  const filename = `${parameters.name}-lws.yaml`;
  const files = { [filename]: yamlDocuments([service, lws]) };
  const steps = [
    { title: "Create the namespace", text: "Create the isolated model namespace.", command: shellCommand(["kubectl", "create", "namespace", parameters.namespace]) },
    { title: "Create the Hugging Face Secret", text: "Read HF_TOKEN only from this workstation environment; the site never accepts the token.", command: shellCommand(["kubectl", "-n", parameters.namespace, "create", "secret", "generic", parameters.hf_secret]) + " " + shellQuote(`--from-literal=${parameters.hf_secret_key}=`) + '"$HF_TOKEN"' },
    { title: "Apply the TP=2 group", text: "Apply the generated rank-zero Service and LeaderWorkerSet.", command: shellCommand(["kubectl", "apply", "-f", filename]) },
    { title: "Wait for both ranks", text: "Both rank pods must become Ready before sending traffic.", command: shellCommand(["kubectl", "-n", parameters.namespace, "wait", "--for=condition=Ready", "pod", "-l", `app=${parameters.name}`, "--timeout=2h"]) },
    { title: "Inspect group status", text: "Check the group and both rank pods.", command: shellCommand(["kubectl", "-n", parameters.namespace, "get", "leaderworkerset,pod", "-l", `app=${parameters.name}`, "-o", "wide"]) },
    { title: "Forward the debug Service", text: "This direct rank-zero Service bypasses llm-d routing.", command: shellCommand(["kubectl", "-n", parameters.namespace, "port-forward", `service/${parameters.name}-serve`, "8888:8888"]) },
    { title: "Check the image helper", text: "Verify rank-zero readiness using the helper shipped in the serving image.", command: shellCommand(["kubectl", "-n", parameters.namespace, "exec", `${parameters.name}-0`, "-c", "modelserver", "--", "/opt/venv/bin/vllm-image", "health", "--phase", "readiness", "--rank", "0", "--local-url", "http://127.0.0.1:8888/v1/models"]) },
    { title: "Send text, tool, and vision requests", text: "Use the OpenAI-compatible endpoint with model name from the recipe; include a real image input for the vision check.", command: shellCommand(["curl", "--fail-with-body", "http://127.0.0.1:8888/v1/chat/completions", "-H", "Content-Type: application/json", "--data", JSON.stringify({ model: recipe.model.served_name, messages: [{ role: "user", content: "Reply with ready." }] })]) },
  ];
  return { files, steps };
}

function dockerEnvironment(recipe, parameters, hostIp) {
  return { ...commonRuntimeEnv(recipe, parameters), VLLM_HOST_IP: hostIp };
}

function dockerModelSync(recipe, parameters, imageReference) {
  return [
    "docker", "run", "--user", "0", "--network", "host", "--gpus", "all",
    "--device", "/dev/infiniband", "--cap-add", "IPC_LOCK", "--ulimit", "memlock=-1", "--shm-size", "64g",
    "--stop-timeout", "120", "-e", "HF_TOKEN", "-e", "HF_XET_HIGH_PERFORMANCE=1",
    "-v", `${parameters.model_storage_path}:/models`, "-v", `${parameters.jit_storage_path}:/cache`,
    "--entrypoint", "/opt/venv/bin/vllm-image", imageReference, ...modelSyncArgs(recipe),
  ];
}

function dockerServe(recipe, parameters, imageReference, rank, hostIp, masterAddress) {
  const name = `${parameters.name}-rank${rank}`;
  const args = [
    "docker", "run", "--detach", "--name", name, "--user", "0", "--network", "host", "--gpus", "all",
    "--device", "/dev/infiniband", "--cap-add", "IPC_LOCK", "--ulimit", "memlock=-1", "--shm-size", "64g", "--stop-timeout", "120",
    "-v", `${parameters.model_storage_path}:/models:ro`, "-v", `${parameters.jit_storage_path}:/cache`,
  ];
  for (const [key, value] of Object.entries(dockerEnvironment(recipe, parameters, hostIp))) args.push("-e", `${key}=${value}`);
  args.push("--entrypoint", recipe.runtime.command[0], imageReference, ...recipe.runtime.command.slice(1), ...servingArgs(recipe, rank, masterAddress));
  return args;
}

function dockerScript(recipe, parameters, imageReference, rank) {
  const hostIp = rank === 0 ? parameters.leader_ip : parameters.worker_ip;
  const lines = ["#!/bin/sh", "set -eu", "", shellCommand(dockerModelSync(recipe, parameters, imageReference)), ""];
  if (rank === 1) {
    lines.push(shellCommand([
      "docker", "run", "--network", "host", "--entrypoint", "/opt/venv/bin/vllm-image", imageReference,
      "rendezvous-wait", "--rank", "1", "--leader", parameters.leader_ip, "--port", "25000", "--dns-timeout", "1200s", "--connect-timeout", "300s",
    ]), "");
  }
  lines.push(shellCommand(dockerServe(recipe, parameters, imageReference, rank, hostIp, rank === 0 ? "127.0.0.1" : parameters.leader_ip)), "");
  return lines.join("\n");
}

function renderDocker(recipe, parameters, imageReference) {
  const leaderFile = `${parameters.name}-leader.sh`;
  const workerFile = `${parameters.name}-worker.sh`;
  return {
    files: { [leaderFile]: dockerScript(recipe, parameters, imageReference, 0), [workerFile]: dockerScript(recipe, parameters, imageReference, 1) },
    steps: [
      { title: "Not yet runtime-validated", text: "These two-host Docker instructions preserve the verified engine configuration but have not completed a two-host runtime acceptance.", command: "" },
      { title: "Start rank zero", text: "Copy the rendered leader script to the leader host, make it executable, and run it without positional arguments.", command: shellCommand(["sh", leaderFile]) },
      { title: "Start rank one", text: "After rank zero opens port 25000, run the rendered worker script on the worker host.", command: shellCommand(["sh", workerFile]) },
      { title: "Follow logs", text: "Long kernel work is not grounds for force-killing a GB10 process.", command: shellCommand(["docker", "logs", "--follow", `${parameters.name}-rank0`]) },
      { title: "Check rank-zero health", text: "Run the image helper in the serving container's own process namespace.", command: shellCommand(["docker", "exec", `${parameters.name}-rank0`, "/opt/venv/bin/vllm-image", "health", "--phase", "readiness", "--rank", "0", "--local-url", "http://127.0.0.1:8888/v1/models"]) },
      { title: "Stop gracefully", text: "Stop each rank with the full shutdown budget; do not force-remove the containers.", command: shellCommand(["docker", "stop", "--time", "120", `${parameters.name}-rank1`, `${parameters.name}-rank0`]) },
    ],
  };
}

function routingResources(recipe, parameters) {
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
  const clusterRoleBinding = { apiVersion: "rbac.authorization.k8s.io/v1", kind: "ClusterRoleBinding", metadata: { name: clusterRoleName }, roleRef: { apiGroup: "rbac.authorization.k8s.io", kind: "ClusterRole", name: clusterRoleName }, subjects: [{ kind: "ServiceAccount", name: eppName, namespace }] };
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
        args: [`--pool-name=${poolName}`, `--pool-namespace=${namespace}`, "--pool-group=inference.networking.k8s.io", "--metrics-endpoint-auth=false", "--config-file=/config/config.yaml", "-v=3"],
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
  const pool = { apiVersion: "inference.networking.k8s.io/v1", kind: "InferencePool", metadata: { name: poolName, namespace }, spec: { appProtocol: "http", selector: { matchLabels: { app: name, "leaderworkerset.sigs.k8s.io/worker-index": "0" } }, targetPorts: [{ number: 8888 }], endpointPickerRef: { name: eppName, port: { number: 9002 }, failureMode: "FailClose" } } };
  const gateway = { apiVersion: "gateway.networking.k8s.io/v1", kind: "Gateway", metadata: { name: gatewayName, namespace }, spec: { gatewayClassName: parameters.gateway_class, listeners: [{ name: "http", protocol: "HTTP", port: 10080 }] } };
  const route = { apiVersion: "aigateway.envoyproxy.io/v1beta1", kind: "AIGatewayRoute", metadata: { name: routeName, namespace }, spec: { parentRefs: [{ name: gatewayName, namespace }], rules: [{ matches: [{ headers: [{ type: "Exact", name: "x-ai-eg-model", value: recipe.model.served_name }] }], backendRefs: [{ group: "inference.networking.k8s.io", kind: "InferencePool", name: poolName }], timeouts: { request: "1800s" } }] } };
  const policy = { apiVersion: "gateway.envoyproxy.io/v1alpha1", kind: "ClientTrafficPolicy", metadata: { name: `${name}-buffer-limit`, namespace }, spec: { targetRefs: [{ group: "gateway.networking.k8s.io", kind: "Gateway", name: gatewayName }], connection: { bufferLimit: "50Mi" } } };
  return [serviceAccount, clusterRole, clusterRoleBinding, configMap, deployment, service, pool, gateway, route, policy];
}

function renderRouting(recipe, parameters) {
  const filename = `${parameters.name}-llm-d-routing.yaml`;
  return {
    files: { [filename]: yamlDocuments(routingResources(recipe, parameters)) },
    steps: [
      { title: "Apply model routing", text: "Install only after the LWS, inference, Envoy Gateway, and Envoy AI Gateway controllers and CRDs are Ready.", command: shellCommand(["kubectl", "apply", "-f", filename]) },
      { title: "Wait for the Endpoint Picker", text: "The EPP's plaintext gRPC path must remain namespace-internal and network-isolated.", command: shellCommand(["kubectl", "-n", parameters.namespace, "rollout", "status", `deployment/${parameters.name}-epp`, "--timeout=5m"]) },
      { title: "Check Gateway acceptance", text: "Require both Accepted and Programmed conditions before testing requests.", command: shellCommand(["kubectl", "-n", parameters.namespace, "get", "gateway", `${parameters.name}-gateway`, "-o", "yaml"]) },
      { title: "Forward the internal Gateway", text: "Use the Gateway implementation Service selected by the controller; do not expose this unauthenticated endpoint publicly.", command: shellCommand(["kubectl", "-n", parameters.namespace, "get", "gateway", `${parameters.name}-gateway`]) },
      { title: "Send a routed request", text: "The model header selects this InferencePool. EPP selects an endpoint; the request body goes directly from Gateway to rank zero.", command: shellCommand(["curl", "--fail-with-body", "http://127.0.0.1:10080/v1/chat/completions", "-H", `x-ai-eg-model: ${recipe.model.served_name}`, "-H", "Content-Type: application/json", "--data", JSON.stringify({ model: recipe.model.served_name, messages: [{ role: "user", content: "Reply with ready." }] })]) },
    ],
  };
}

export function renderRecipe(recipe, parameters = {}, imageReference = recipe?.validation?.image, target = "lws") {
  validateRecipe(recipe);
  const values = normalizedParameters(recipe, parameters, target);
  if (target !== "llm-d-routing") validateImageReference(imageReference);
  if (target === "lws") return renderLws(recipe, values, imageReference);
  if (target === "docker") return renderDocker(recipe, values, imageReference);
  return renderRouting(recipe, values);
}
