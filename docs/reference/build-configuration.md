# Build Configuration

This reference describes the Bazel builder for [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm) on consumer Blackwell. The implemented image profile targets DGX Spark ARM64; other platform declarations do not establish verified image support.

## Targets

| Target | Output or Behaviour |
| --- | --- |
| `//image:vllmb12x` | ARM64 OCI image layout |
| `//image:vllmb12x_load` | Local image loader |
| `//image:runtime_venv` | Python, NCCL, runtime wheels, and non-vLLM source wheels |
| `//image:vllm_venv` | vLLM-only overlay tar |
| `//components:torch` | Combined native-and-Python PyTorch wheel |
| `//components:vllm` | vLLM wheel |
| `//components:vllm_compiler_cache_stats` | Action-local compiler-cache statistics |
| `//tests/image:vllmb12x_contract` | Docker-backed ARM64 metadata and file contract |

Sources: [image targets](../../image/BUILD.bazel), [components](../../components/BUILD.bazel), [image contract](../../tests/image/BUILD.bazel).

## Configuration Files

| File | Authority |
| --- | --- |
| [`.bazelversion`](../../.bazelversion) | Bazel 9.2.0 |
| [`MODULE.bazel`](../../MODULE.bazel) | Bzlmod dependencies, OCI base digest, source patch labels, repository extensions |
| [`profile.json`](../../profiles/vllmb12x/profile.json) | Source revisions, profile image metadata, CUDA architecture values |
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
| `just refresh-vllmb12x` | Resolve the moving vLLM branch into the immutable profile lock |
| `just bazel <command> [args...]` | Raw Bazel command without rc files; no platform or execution flags added |

Build, test, and load select the ARM64 target and execution platform, local execution, `.bazel-cache`, and the strict action environment. Source actions require persistent writable `/ccache` independently of Bazel's disk cache.

## Nightly Publication

The public PAC definition at [`.tekton/vllmb12x-nightly.yaml`](../../.tekton/vllmb12x-nightly.yaml) first builds the locked image and runs the image contract on the ARM64 remote worker. The publisher then rebuilds `//image:vllmb12x` with `--remote_download_outputs=all`, acquires a short Kubernetes Lease, and pushes the materialised OCI layout with a host `crane` rather than `//image:vllmb12x_push`. `rules_oci` packages `crane` and `jq` as exec-platform runfiles, so `bazel run` under the remote ARM64 configuration resolves binaries for the remote executor instead of the pipeline pod. Bazel symlinks the base-image and apt blobs into its external repository directories, and `crane` rejects a layout blob that is a symlink, so the publisher copies the layout beside the output base with hard links and pushes that copy.

`scripts/publish-vllmb12x.py` gives each publication an immutable tag containing the UTC date, locked vLLM revision, builder revision, and the Lease transition number. The transition number makes repeated same-revision rebuilds distinct. The publisher releases the Lease by shortening it rather than deleting it, so a delayed run cannot remove a later holder's lock. The image also receives `latest` after its immutable tag succeeds.

The public pipeline names only PAC parameters. The private PAC Repository binds registry, authentication, remote-execution, storage, and Lease details.

## Optional CI Configuration

NativeLink is our optional CI backend. The public `remote-aarch64` configuration selects remote execution with no local fallback, minimal downloads, compression, and a 21600-second remote timeout. Endpoints, authentication, and execution properties must be supplied separately by CI. No private infrastructure is configured in the repository. Local Justfile recipes do not use this configuration.

## Runtime

| Property | Value |
| --- | --- |
| Image tag | `randomvariable/vllm-b12x-multi:<build version>` |
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
**Version Compatibility:** Checked-in Bazel 9.2.0 build and `vllmb12x` profile
