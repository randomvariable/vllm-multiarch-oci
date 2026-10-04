// The site fields a manifest needs, declared once and keyed by the SHAPE of the
// deployment instead of being copied into every recipe.
//
// A deployment asks the operator for exactly the facts the renderer dereferences
// for the shape it has. A single-node pod opens no RoCE device and joins no
// collective, so asking it for an HCA list, a GID index, a Multus attachment, an
// RDMA allocator resource or a topology label would be five fields with no
// meaning; a multi-node group asks for all of them because the collective is what
// makes the group one machine. `render.js` filters by the same scope rules it uses
// when it builds the pod (`FIELD_SCOPES` there), so a field that never reaches the
// output is never asked for on the form either.
//
// A recipe does not re-declare these fields. It overrides the values that are
// specific to it -- the checkpoint's own namespace, resource name and the two node
// host paths, as the GitOps repository carries them -- through
// `deployment.parameters`, and the merge happens in `siteParameterDefinitions()`
// below. That is what lets a recipe with no parameter block at all
// (`qwen38-27b` shipped one) still render every manifest target: the shape's
// fields are present by construction, and only their defaults are authored.
//
// The field schema is the recipe schema: `label`, `type` (string | integer |
// stringMap), `required`, `default`, `targets` (documentation of which outputs
// read it), `description`, and the optional `help` / `suggestions`. `build-data.mjs`
// validates the MERGED definitions, so an override can never drop a label or an
// explanation out of the form.

const field = (label, type, value, description, extra = {}) => ({
  label,
  type,
  required: true,
  default: value,
  description,
  ...extra,
});

// The fields every deployment asks for, whatever its shape: the namespace and
// resource name the objects land in, the two node paths the weights and the
// compiler caches live on, the Hugging Face credential the engine pulls with, the
// operator's own node labels, and the GatewayClass the route binds to.
const COMMON = {
  namespace: field("Kubernetes namespace", "string", "vllm", "Namespace for the model and its routing resources.", {
    targets: ["lws", "llm-d-routing"],
    help: { label: "Namespace and Secret setup", href: "guides/deploy-tp2/#create-namespace-and-credential" },
  }),
  name: field(
    "Deployment name",
    "string",
    "",
    "DNS-compatible base name for resources and Docker containers. An upstream profile selection defaults it from its own profile, hardware and preset; a recipe names it.",
    {
      help: { label: "Deployment walkthrough", href: "guides/deploy-tp2/" },
    },
  ),
  model_storage_path: field(
    "Model storage path",
    "string",
    "/var/lib/models",
    "Absolute persistent host path on the GPU node for the model cache and, in sync mode, the published snapshot.",
    {
      targets: ["lws", "docker"],
      help: { label: "Storage layout", href: "guides/deploy-tp2/#storage-paths" },
    },
  ),
  jit_storage_path: field("JIT cache path", "string", "/var/cache/vllm", "Absolute persistent host path on the GPU node for compiler and kernel caches.", {
    targets: ["lws", "docker"],
    help: { label: "Storage layout", href: "guides/deploy-tp2/#storage-paths" },
  }),
  hf_secret: field("Hugging Face Secret", "string", "huggingface-token", "Kubernetes Secret created from the operator's local HF_TOKEN environment variable.", {
    targets: ["lws"],
    help: { label: "Create the Secret", href: "guides/deploy-tp2/#create-namespace-and-credential" },
  }),
  hf_secret_key: field(
    "Hugging Face Secret key",
    "string",
    "HF_TOKEN",
    "Data key in the Kubernetes Secret; the browser never accepts or stores the token itself.",
    {
      targets: ["lws"],
      help: { label: "Create the Secret", href: "guides/deploy-tp2/#create-namespace-and-credential" },
    },
  ),
  node_selector: field(
    "GPU node selector",
    "stringMap",
    {},
    "Operator-added node labels for the GPU node; a recipe's own `deployment.nodeSelector` (architecture, node role) layers underneath this.",
    {
      required: false,
      targets: ["lws"],
      help: { label: "Find node labels", href: "guides/deploy-tp2/#node-placement" },
    },
  ),
  gateway_class: field(
    "GatewayClass name",
    "string",
    "",
    "Routing-only accepted Envoy GatewayClass supplied by the cluster operator.",
    {
      targets: ["llm-d-routing"],
      help: { label: "Install or inspect GatewayClass", href: "guides/cluster-prerequisites/#install-envoy-gateway-183" },
    },
  ),
};

// What a group that spans nodes adds: the secondary collective network the ranks
// talk over, the exclusive-topology label that keeps a group inside one pair of
// nodes, the RDMA allocator resource, and the host-side interfaces a bare
// `docker run` on the collective needs. All of it is collective-only: none of it
// exists on a single-node pod, which is why it is absent from the shape above
// rather than declared and left blank.
const COLLECTIVE = {
  network_attachment: field(
    "RoCE network attachment",
    "string",
    "roce-net",
    "Multus NetworkAttachmentDefinition name for the secondary collective network.",
    {
      targets: ["lws"],
      help: { label: "Create and inspect the attachment", href: "guides/multus-rdma/#define-the-secondary-network" },
    },
  ),
  topology_key: field("Exclusive topology key", "string", "", "Site label that groups the nodes allowed to form this TP group.", {
    targets: ["lws"],
    help: { label: "Configure topology labels", href: "guides/deploy-tp2/#node-placement" },
  }),
  topology_values: field("Allowed topology values", "string", "", "Comma-separated values accepted for the exclusive topology key.", {
    targets: ["lws"],
    help: { label: "Configure topology labels", href: "guides/deploy-tp2/#node-placement" },
  }),
  rdma_resource: field(
    "RDMA extended resource",
    "string",
    "rdma.com/roce",
    "RDMA shared-device-plugin resource key for the selected host interfaces.",
    {
      targets: ["lws"],
      help: { label: "Configure the RDMA resource", href: "guides/multus-rdma/#advertise-one-rdma-unit-per-node" },
    },
  ),
  rdma_units: field(
    "RDMA allocator units",
    "integer",
    1,
    "Extended-resource allocator units per rank, not the number of HCAs or ports.",
    {
      targets: ["lws"],
      help: { label: "Inspect RDMA allocation", href: "guides/multus-rdma/#advertise-one-rdma-unit-per-node" },
    },
  ),
  hca: field("NCCL HCA list", "string", "rocep1s0f1,roceP2p1s0f1", "Comma-separated RDMA devices. The preset is the pair used by the accepted GB10 deployment; type over it if host enumeration differs.", {
    targets: ["lws", "docker"],
    suggestions: ["rocep1s0f1,roceP2p1s0f1"],
    help: { label: "Find HCA names", href: "guides/multus-rdma/#advertise-one-rdma-unit-per-node" },
  }),
  gid_index: field("RoCE GID index", "integer", 3, "GID index verified for the site's RoCE VLAN and address family.", {
    targets: ["lws", "docker"],
    help: { label: "Select the GID", href: "guides/multus-rdma/#identify-the-runtime-network-values" },
  }),
  control_interface: field(
    "Host control network interface",
    "string",
    "",
    "Host interface used by Gloo coordination and NCCL socket bootstrap across the group; Kubernetes uses the primary pod interface eth0.",
    {
      targets: ["docker"],
      help: { label: "Find host interfaces", href: "guides/deploy-tp2/#runtime-network-interfaces" },
    },
  ),
  leader_ip: field("Leader host IP", "string", "", "Docker-only rank-zero address reachable from the worker over the host network.", {
    targets: ["docker"],
    help: { label: "Docker host networking", href: "guides/deploy-tp2/#docker-host-values" },
  }),
  worker_ip: field("Worker host IP", "string", "", "Docker-only rank-one address used as that rank's VLLM_HOST_IP.", {
    targets: ["docker"],
    help: { label: "Docker host networking", href: "guides/deploy-tp2/#docker-host-values" },
  }),
};

export const SITE_PARAMETERS = {
  single: { ...COMMON },
  multi: { ...COMMON, ...COLLECTIVE },
};

// The definitions this deployment's shape asks the operator for: the shared set for
// the shape, with the recipe's `deployment.parameters` layered on top. A recipe
// override wins per property, so it can change a `default` alone or replace the
// whole definition -- and a recipe may add a field the shape does not carry, which
// the renderer will then read like any other.
export function siteParameterDefinitions(recipe) {
  const collective = recipe?.launch?.topology?.kind === "lws";
  const shared = recipe?.deployment?.siteParameters?.[collective ? "multi" : "single"] ?? SITE_PARAMETERS[collective ? "multi" : "single"];
  const declared = recipe?.deployment?.parameters;
  if (declared !== undefined && declared !== null && (typeof declared !== "object" || Array.isArray(declared))) {
    throw new TypeError("deployment.parameters must be a mapping");
  }
  const out = {};
  for (const name of [...Object.keys(shared), ...Object.keys(declared ?? {})]) {
    out[name] = { ...shared[name], ...(declared?.[name] ?? {}) };
  }
  return out;
}
