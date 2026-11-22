# Bazel Builder for local-inference-lab/vLLM on NVIDIA DGX Spark

This repository is the Bazel builder and deployment recipe site for [local-inference-lab/vLLM](https://github.com/local-inference-lab/vllm) on NVIDIA DGX Spark. It packages that fork and its source-built dependencies into an Open Container Initiative (OCI) image. It is not the vLLM fork itself, an upstream `vllm-project/vllm` image, or a general-purpose builder for arbitrary vLLM releases.

The local build path targets DGX Spark. Bazel compiles PyTorch and CUDA extensions from source, then assembles their wheels into layered Python runtimes.

The implemented image is a multiarchitecture index targeting Linux ARM64 and Linux x86-64, CUDA 13.4.1, and Python 3.12. CI builds both architectures and runs the image contract for each. The ARM64 manifest carries the production two-node deployment evidence; the x86-64 manifest is verified by its CI contract plus a single-GPU serving smoke on an RTX 5090 (see [Latest image publication](https://randomvariable.github.io/vllm-multiarch-oci/image-releases/)). Publication requires exactly the two manifests together: the publisher and the recipe resolver reject an index missing either architecture.

CI uses Bazel's `--lockfile_mode=error`. Commit the regenerated `MODULE.bazel.lock`
with dependency or module-extension changes. A stale lock must fail before the
image build, rather than mutate the checkout and fail the release cleanliness check.

## Included Upstream Changes

<!-- included-changes:start -->
The image is built from pinned fork revisions rather than from upstream branches, so this table records which upstream changes the current lock carries. Regenerate it with `scripts/vllmb12x-included-changes.py`.

| Component | Change | Included as |
| --- | --- | --- |
| vLLM `68e08dfdd38d` | [vllm-project/vllm#68445](https://github.com/vllm-project/vllm/pull/68445) Bounded shared-memory broadcast waits | Merged into the pinned revision (3 commits) |
| vLLM `68e08dfdd38d` | b12x GB10 integration ported from local-inference-lab karmic-kraken (no upstream pull request) | Merged into the pinned revision (1 commits) |
| vLLM `68e08dfdd38d` | DeepSeek V4.1 Engram tables served from checkpoint files (no upstream pull request) | Merged into the pinned revision (1 commits) |
| vLLM `68e08dfdd38d` | b12x call sites moved to local-inference-lab FlashInfer APIs (no upstream pull request) | Merged into the pinned revision (1 commits) |
| vLLM `68e08dfdd38d` | Qwen4Exp b12x modules import their config from transformers (no upstream pull request) | Merged into the pinned revision (1 commits) |
| vLLM `68e08dfdd38d` | Request-boundary recurrent checkpoints for Qwen3.8-Flash-Next (no upstream pull request) | Merged into the pinned revision (1 commits) |
| vLLM `68e08dfdd38d` | Rewrite flash_attn.cute imports to vllm.vllm_flash_attn.cute (no upstream pull request) | Applied at build time by `third_party/vllm_flash_attn_cute_namespace.patch` |
| FlashInfer `fbf53f8dc859` | GB10 DSV4.1 decode split policy and NVFP4 CSF scale defaults (no upstream pull request) | Merged into the pinned revision (2 commits) |
| FlashInfer `fbf53f8dc859` | C S3-FIFO host row cache for disk PLE/Engram tables (no upstream pull request) | Merged into the pinned revision (12 commits) |
<!-- included-changes:end -->

The retained NVFP4-CSF loader converts checkpoint scale streams to FlashInfer's
lossless `CsfBlockPlanes` representation before preparing expert weights.

For a Moonshine build that includes the native DeepSeek V4.1 MXFP4-CSF port,
point `MODEL` to the complete checkpoint directory containing `manifest.json`,
`build-contract.json`, `metadata/`, and `tensors/`. Set both format options
explicitly; `CHECKPOINT=csf` alone is not sufficient with older launcher policy data:

```bash
MODEL=/models/DeepSeek-V4.1-Flash-1017-abliterated-lossless-CSF \
TP=4 LOAD_FORMAT=mxfp4_csf QUANTIZATION=mxfp4_csf \
vllm-image launch --profile ds41-flash --hardware native --print-config
```

The dry run should show `--load-format mxfp4_csf`, `--quantization mxfp4_csf`,
and `--moe-backend b12x`. This checks argument selection, not GPU execution
or whether the full checkpoint fits the chosen topology.

## Getting Started

Start with [Building Locally on DGX Spark](docs/how-to/build-locally-on-spark.md). The current build requires GCC 15, compatible userspace, and persistent writable standard ccache storage. CI replaces this with its durable cache volume. Stock DGX OS is not automatically compatible. A full local Spark build has not yet been verified.

With the prerequisites satisfied, use the [Justfile helpers](docs/reference/build-configuration.md):

```bash
just analyze
just build
just test
just load
```

`analyze` resolves the graph without compiling. `build` produces the image, `test` checks its structure, and `load` imports it into Docker without starting a container. Cold source builds can take hours.

## Documentation

The public [deployment recipe site](https://randomvariable.github.io/vllm-multiarch-oci/) provides the tested DeepSeek configuration, generated Kubernetes and Docker instructions, and the guided cluster setup path.

| Need | Document |
| --- | --- |
| Learn the build graph | [Inspect Your First Build](docs/tutorials/first-build.md) |
| Prepare a Spark build environment | [Build Locally on DGX Spark](docs/how-to/build-locally-on-spark.md) |
| Build, test, and load | [Build and Test an Image](docs/how-to/build-and-test.md) |
| Measure a native edit | [Measure Incremental Builds](docs/how-to/measure-incremental-builds.md) |
| Use model, rendezvous, and health helpers | [Use the Kubernetes Image Helpers](docs/how-to/use-kubernetes-helpers.md) |
| Run a pod as a pinned non-root uid | [Run the Image as a Non-Root User](docs/how-to/run-as-non-root.md) |
| Look up targets and options | [Build Configuration](docs/reference/build-configuration.md) |
| Configure the Mooncake runtime addition | [Mooncake Transfer Engine](docs/reference/mooncake-transfer-engine.md) |
| Understand build boundaries | [Build and Cache Design](docs/explanation/build-and-cache-design.md) |

NativeLink is our optional CI execution backend, not a prerequisite for local builds. CI access and deployment configuration are private and are not distributed here.

## Licence

Repository code is licensed under [Apache License 2.0](LICENSE), consistent with existing SPDX headers. Upstream sources, Python packages, CUDA redistributables, and base-image contents retain their respective licences and redistribution requirements.
