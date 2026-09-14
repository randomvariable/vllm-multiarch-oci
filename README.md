# Bazel Builder for Consumer Blackwell vLLM

This repository is a Bazel builder for [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm), targeting consumer Blackwell GPUs. It packages that fork and its source-built dependencies into an Open Container Initiative (OCI) image. It is not the vLLM fork itself or a general-purpose builder for arbitrary vLLM releases.

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

## Licence

Repository code is licensed under [Apache License 2.0](LICENSE), consistent with existing SPDX headers. Upstream sources, Python packages, CUDA redistributables, and base-image contents retain their respective licences and redistribution requirements.
