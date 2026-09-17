# Bazel Builder for local-inference-lab/vLLM on NVIDIA DGX Spark

This repository is the Bazel builder and deployment recipe site for [local-inference-lab/vLLM](https://github.com/local-inference-lab/vllm) on NVIDIA DGX Spark. It packages that fork and its source-built dependencies into an Open Container Initiative (OCI) image. It is not the vLLM fork itself, an upstream `vllm-project/vllm` image, or a general-purpose builder for arbitrary vLLM releases.

The local build path targets DGX Spark. Bazel compiles PyTorch and CUDA extensions from source, then assembles their wheels into layered Python runtimes.

The implemented image targets Linux ARM64, CUDA 13.3.1, and Python 3.12. The x86-64 platform declaration is not a verified second image build.

## Getting Started

Start with [Building Locally on DGX Spark](docs/how-to/build-locally-on-spark.md). The current build requires GCC 15, compatible userspace, and persistent writable `/ccache` storage. Stock DGX OS is not automatically compatible. A full local Spark build has not yet been verified.

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
| Look up targets and options | [Build Configuration](docs/reference/build-configuration.md) |
| Understand build boundaries | [Build and Cache Design](docs/explanation/build-and-cache-design.md) |

NativeLink is our optional CI execution backend, not a prerequisite for local builds. CI access and deployment configuration are private and are not distributed here.

## Included Upstream Changes

<!-- included-changes:start -->
The image is built from pinned fork revisions rather than from upstream branches, so this table records which upstream changes the current lock carries. Regenerate it with `scripts/vllmb12x-included-changes.py`.

| Component | Change | Included as |
| --- | --- | --- |
| vLLM `bd22e0f25043` | [local-inference-lab/vllm#777](https://github.com/local-inference-lab/vllm/pull/777) fix(qwen): propagate MTP positional overrides | Merged into the pinned revision (3 commits) |
| vLLM `bd22e0f25043` | [local-inference-lab/vllm#779](https://github.com/local-inference-lab/vllm/pull/779) perf(qwen): shard TP4 HC prefill and coalesce recurrent checkpoints | Merged into the pinned revision (9 commits) |
| vLLM `bd22e0f25043` | [vllm-project/vllm#52917](https://github.com/vllm-project/vllm/pull/52917) Adaptive spin grace and bounded architectural waits for shm_broadcast | Applied at build time by `third_party/vllm_shm_broadcast_spin_grace.patch` |
| vLLM `bd22e0f25043` | Rewrite flash_attn.cute imports to vllm.vllm_flash_attn.cute (no upstream pull request) | Applied at build time by `third_party/vllm_flash_attn_cute_namespace.patch` |
| B12X `4401ce4fb982` | [local-inference-lab/b12x#384](https://github.com/local-inference-lab/b12x/pull/384) fix(preparation): retain prepared launchers and coordinate collectives | Merged into the pinned revision (10 commits) |
| B12X `4401ce4fb982` | [local-inference-lab/b12x#386](https://github.com/local-inference-lab/b12x/pull/386) feat(ple): export prepared internal prefill checkpoints | Merged into the pinned revision (2 commits) |
| B12X `4401ce4fb982` | [local-inference-lab/b12x#387](https://github.com/local-inference-lab/b12x/pull/387) perf(qsa): reuse representative keys across paired queries | Merged into the pinned revision (4 commits) |
<!-- included-changes:end -->

## Licence

Repository code is licensed under [Apache License 2.0](LICENSE), consistent with existing SPDX headers. Upstream sources, Python packages, CUDA redistributables, and base-image contents retain their respective licences and redistribution requirements.
