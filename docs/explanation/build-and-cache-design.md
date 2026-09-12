# Understanding Build and Cache Boundaries

This repository builds the [local-inference-lab/vllm fork](https://github.com/local-inference-lab/vllm) for consumer Blackwell GPUs. The pinned profile selects the fork revision and its dependency closure. It is not a general-purpose build configuration for arbitrary upstream vLLM releases.

The build separates dependency outputs from the frequently changed vLLM installation. Local execution runs these actions on the build host. NativeLink is our optional CI backend for the same build graph. Public builds do not require access to it.

## From Sources to an Image

The [source repository rule](../../bazel/sources.bzl) checks out pinned commits, applies declared patches, and produces archives with normalized ordering, timestamps, and ownership. It writes the revision and patch hashes into `source.identity.json`.

The dependency flow is:

```text
Pinned sources + CUDA toolkit
  -> dependency wheels + NCCL -> stable runtime layer
  -> vLLM native extensions -> vLLM wheel -> vLLM overlay
CUDA base + apt + cuSPARSELt + stable runtime + vLLM overlay
  -> ARM64 OCI image
```

The [vLLM rule](../../bazel/vllm_wheel.bzl) separates native compilation from packaging. Successful native outputs can remain reusable when packaging fails. PyTorch produces one combined native-and-Python wheel.

## Two Caches

Bazel reuses a complete successful action when its declared inputs and command match. Its local disk cache lives at `.bazel-cache`. Deleting that directory discards those entries.

ccache reuses compiler results inside a rerun action. Completed compilations can survive a later action failure. The action drivers store objects at `/ccache/objects` and extracted toolchains at `/ccache/toolchains`. These paths require persistent writable storage. Container removal must not remove that storage. Explicit deletion or loss of the backing disk can destroy it.

Sandbox path normalization supports reuse across action directories. Cache availability is required rather than silently falling back to uncached compilation. Action-local statistics distinguish a particular compilation's reuse from cumulative cache activity.

## Layer Boundaries

The [stable runtime layer](../../image/BUILD.bazel) contains Python, NCCL, and dependency wheels. The upper layer contains only vLLM. Both contribute to `/opt/venv`, but the lower layer must not contain an older vLLM package.

A source edit can reuse most compiler results while still paying for environment setup, linking, packaging, and output assembly. Cache hits are not a guarantee of a particular build time. Finer native-action boundaries require implementation work rather than a documentation setting.

## Further Reading

- [Measure Incremental Builds](../how-to/measure-incremental-builds.md) (how-to)
- [Build Configuration](../reference/build-configuration.md) (reference)

---

**Last Updated:** September 2026
**Version Compatibility:** Checked-in `glm53-0906` build
