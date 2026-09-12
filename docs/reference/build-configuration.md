# Build Configuration

This reference describes the Bazel builder for [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm) on consumer Blackwell. The implemented image profile targets DGX Spark ARM64; other platform declarations do not establish verified image support.

## Targets

| Target | Output or Behaviour |
| --- | --- |
| `//image:glm53_0906` | ARM64 OCI image layout |
| `//image:glm53_0906_load` | Local image loader |
| `//image:runtime_venv` | Python, NCCL, runtime wheels, and non-vLLM source wheels |
| `//image:vllm_venv` | vLLM-only overlay tar |
| `//components:torch` | Combined native-and-Python PyTorch wheel |
| `//components:vllm` | vLLM wheel |
| `//components:vllm_compiler_cache_stats` | Action-local compiler-cache statistics |
| `//tests/image:glm53_0906_contract` | Docker-backed ARM64 metadata and file contract |

Sources: [image targets](../../image/BUILD.bazel), [components](../../components/BUILD.bazel), [image contract](../../tests/image/BUILD.bazel).

## Configuration Files

| File | Authority |
| --- | --- |
| [`.bazelversion`](../../.bazelversion) | Bazel 9.2.0 |
| [`MODULE.bazel`](../../MODULE.bazel) | Bzlmod dependencies, OCI base digest, source patch labels, repository extensions |
| [`profile.json`](../../profiles/glm53-0906/profile.json) | Source revisions, profile image metadata, CUDA architecture values |
| [`image/BUILD.bazel`](../../image/BUILD.bazel) | Effective image composition and tag |
| [`.bazelrc`](../../.bazelrc) | Public build options |
| `.bazelrc.user` | Ignored, operator-owned endpoints, credentials, and worker properties |
| [`platforms/BUILD.bazel`](../../platforms/BUILD.bazel) | ARM64 and x86-64 platform declarations |

## Justfile Helpers

All helpers ignore Bazel rc files, including private host configuration.

| Recipe | Behaviour |
| --- | --- |
| `just analyze` | Local image analysis without compilation |
| `just build [args...]` | Local ARM64 image build with optional Bazel arguments |
| `just test [args...]` | Local Docker-backed image contract |
| `just load [args...]` | Load the image into the local daemon |
| `just bazel <command> [args...]` | Raw Bazel command without rc files; no platform or execution flags added |

Build, test, and load select the ARM64 target and execution platform, local execution, `.bazel-cache`, and the strict action environment. Source actions require persistent writable `/ccache` independently of Bazel's disk cache.

## Optional CI Configuration

NativeLink is our optional CI backend. The public `remote-aarch64` configuration selects remote execution with no local fallback, minimal downloads, compression, and a 21600-second remote timeout. Endpoints, authentication, and execution properties must be supplied separately by CI. No private infrastructure is configured in the repository. Local Justfile recipes do not use this configuration.

## Runtime

| Property | Value |
| --- | --- |
| Image tag | `local/vllm:glm53-flash-nvfp4-head-0906` |
| Platform | `linux/arm64` |
| CUDA base | CUDA 13.3.1 cuDNN development image, Ubuntu 26.04, digest-pinned |
| Python | 3.12 |
| Entrypoint | `/opt/venv/bin/vllm` |
| Working directory | `/root` |
| CUDA home | `/usr/local/cuda` |
| NCCL library path | `/opt/nccl/lib` |
| Torch architecture | `12.1a` |
| FlashInfer architecture | `12.1f` |
| Disabled kernel | `MarlinFP8ScaledMMLinearKernel` |

Source: [image rule](../../bazel/vllm_image.bzl). The x86-64 platform declaration does not establish a supported x86-64 image. PyTorch uses one wheel, not a `torch_libtorch`/Python-only split.

## Further Reading

- [Build and Test an Image](../how-to/build-and-test.md)

---

**Last Updated:** September 2026
**Version Compatibility:** Checked-in Bazel 9.2.0 build and `glm53-0906` profile
