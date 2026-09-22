# Bazel Builder for local-inference-lab/vLLM on NVIDIA DGX Spark

This repository is the Bazel builder and deployment recipe site for [local-inference-lab/vLLM](https://github.com/local-inference-lab/vllm) on NVIDIA DGX Spark. It packages that fork and its source-built dependencies into an Open Container Initiative (OCI) image. It is not the vLLM fork itself, an upstream `vllm-project/vllm` image, or a general-purpose builder for arbitrary vLLM releases.

The local build path targets DGX Spark. Bazel compiles PyTorch and CUDA extensions from source, then assembles their wheels into layered Python runtimes.

The implemented image is a multiarchitecture index targeting Linux ARM64 and Linux x86-64, CUDA 13.4.1, and Python 3.12. CI builds both architectures and runs the image contract for each. The ARM64 manifest carries the production two-node deployment evidence; the x86-64 manifest is verified by its CI contract plus a single-GPU serving smoke on an RTX 5090 (see [Latest image publication](https://randomvariable.github.io/vllm-multiarch-oci/image-releases/)). Publication requires exactly the two manifests together: the publisher and the recipe resolver reject an index missing either architecture.

## Included Upstream Changes

<!-- included-changes:start -->
The image is built from pinned fork revisions rather than from upstream branches, so this table records which upstream changes the current lock carries. Regenerate it with `scripts/vllmb12x-included-changes.py`.

| Component | Change | Included as |
| --- | --- | --- |
| vLLM `85a78f57a0e4` | [local-inference-lab/vllm#52917](https://github.com/local-inference-lab/vllm/pull/52917) Adaptive shared-memory broadcast spin (applied at build time) | Applied at build time by `third_party/vllm_flash_attn_cute_namespace.patch` |
<!-- included-changes:end -->

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
