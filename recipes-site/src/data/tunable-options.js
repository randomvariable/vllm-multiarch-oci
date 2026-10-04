// Which resolved engine options an operator may change on the site.
//
// Most of what the resolver emits is a property of the model, not of a deployment:
// the tokenizer mode, the quantization scheme, the reasoning and tool-call parsers,
// the kernel backends and the hf-overrides that encode the checkpoint's rope are fixed
// by the weights. Editing them does not tune a deployment, it breaks the model, so
// the site shows them read-only. What remains is the short list of deployment
// decisions: how many GPUs, how much speculation, how many requests and how much
// context, how much memory to give the engine, the prefix cache, and what the server
// is called and listens on.
//
// An allowlist rather than a denylist on purpose: an option a future upstream profile
// adds is read-only until someone decides it is a deployment decision.

export const TUNABLE_GROUPS = [
  {
    id: "gpus",
    title: "GPUs and speculation",
    // The speculation method is fixed by the model: it is whatever drafter the
    // checkpoint ships (MTP heads, DSpark, an EAGLE draft). Only how far it drafts
    // is a deployment choice.
    options: ["tensor-parallel-size", "replicas", "draft-tokens"],
  },
  {
    id: "requests",
    title: "Requests and context",
    options: [
      "max-model-len",
      "max-num-seqs",
      "max-num-batched-tokens",
      "max-parallel-prefills",
      "prefill-policy",
      "prefill-compute-share",
      "prefill-compute-half-life",
    ],
  },
  {
    id: "memory",
    title: "Engine memory",
    options: ["gpu-memory-utilization", "max-cudagraph-capture-size"],
  },
  {
    id: "cache",
    title: "Prefix cache",
    options: ["enable-prefix-caching", "cache-mode"],
  },
  {
    id: "server",
    title: "Server",
    options: ["served-model-name", "port"],
  },
];

export const TUNABLE_OPTIONS = new Set(TUNABLE_GROUPS.flatMap((group) => group.options));

// The external KV cache tier. Every option the resolver names `cache-*` other than
// `cache-mode` configures LMCache or the native cache server, so they are only
// meaningful -- and only shown -- once a reader has turned that tier on.
export function isCacheTierOption(name) {
  return name.startsWith("cache-") && name !== "cache-mode";
}

// GB10 has one pool of unified memory shared by the CPU and the GPU. An LMCache RAM
// tier there is not extra capacity: every byte it holds is a byte the engine's own KV
// cache cannot use. So on that hardware the external cache is opt-in, and the site
// says why before the reader turns it on.
export function unifiedMemory(hardware) {
  return typeof hardware === "string" && hardware.startsWith("gb10");
}

export function isTunable(name, cacheEnabled) {
  if (TUNABLE_OPTIONS.has(name)) return true;
  return cacheEnabled && isCacheTierOption(name);
}
