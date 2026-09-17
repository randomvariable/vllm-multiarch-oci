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
| [`vllmb12x-runtime-configuration.md`](vllmb12x-runtime-configuration.md) | Generated runtime configuration delta for the checked-in VLLMB12X source lock. |
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

The generated [VLLMB12X Runtime Configuration](vllmb12x-runtime-configuration.md) lists every B12X runtime environment reader and every changed local-inference-lab/vLLM environment, CLI, or `additional_config` control relative to its pinned stock-vLLM merge-base. Regenerate it with `scripts/vllmb12x-runtime-config.py` whenever this profile changes source commits or patches.

## Nightly Publication

The public PAC definition at [`.tekton/vllmb12x-nightly.yaml`](../../.tekton/vllmb12x-nightly.yaml) first builds the locked image and runs the image contract on the ARM64 remote worker. The publisher then rebuilds `//image:vllmb12x` with `--remote_download_outputs=all`, acquires a short Kubernetes Lease, and pushes the materialised OCI layout with a host `crane` rather than `//image:vllmb12x_push`. `rules_oci` packages `crane` and `jq` as exec-platform runfiles, so `bazel run` under the remote ARM64 configuration resolves binaries for the remote executor instead of the pipeline pod. Bazel symlinks the base-image and apt blobs into its external repository directories, and `crane` rejects a layout blob that is a symlink, so the publisher copies the layout beside the output base with hard links and pushes that copy.

`scripts/publish-vllmb12x.py` gives each publication an immutable tag with this form:

```text
vllmb12x-<source-branch>-<vllm-commit-12>-<builder-commit-12>-<UTC-date>-n<sequence>
```

For example, `refs/heads/dev/jovian-judgement` becomes `dev-jovian-judgement`. The profile stores the full source ref and full commit. The tag uses normalized branch text and 12-character commit prefixes for operators. The Lease transition number makes repeated builds of the same revisions distinct. The publisher releases the Lease by shortening it rather than deleting it. A delayed run cannot remove a later holder's lock. The image also receives `latest` after its immutable tag succeeds.

The OCI metadata identifies the built vLLM source through `org.opencontainers.image.source`, `org.opencontainers.image.revision`, and `org.opencontainers.image.version`. Image-specific labels retain the full source ref, vLLM revision, vLLM version, and builder repository. The embedded `/opt/vllmb12x/sources.lock.json` records the complete dependency lock.

The public pipeline names only PAC parameters. The private PAC Repository binds registry, authentication, remote-execution, storage, and Lease details.

## Pull-Request Builds

[`.tekton/vllmb12x-pull-request.yaml`](../../.tekton/vllmb12x-pull-request.yaml) builds and contract-tests the same image for a pull request against `main`, then pushes it to the internal registry only. It skips the same documentation-only paths as the nightly, and a new commit on the pull request cancels the run building the previous one.

`scripts/publish-vllmb12x.py --pull-request <number>` differs from a nightly publication in three ways, all of which follow from the tag being reachable only by name:

```text
pr-<number>-<vllm-commit-12>-<builder-commit-12>
```

The tag carries no sequence and no date, so re-running a pull-request build overwrites its own image instead of adding another. Nothing else points at the image, so the publisher takes no Lease. `latest` is never retagged, in either registry.

## Build Serialization

Both lanes build on one node, against one remote worker pool and one ccache volume. Running them at once halves the workers each build sees and evicts the other lane's cache entries, so `scripts/build-mutex.py` holds a Kubernetes Lease named by the `build_lease` parameter for the whole build, test, and publication sequence, and the other lane waits for it.

The Lease duration is short relative to a build and the holder renews it, so a cancelled or killed run blocks the other lane only until its Lease expires, not until someone cleans up. A single failed renewal is not treated as a lost Lease: ownership ends only when another holder appears, the Lease is deleted, or the last successful renewal could itself have expired. The publication Lease is separate and still covers only tag allocation and registry mutation.

## Releases

Publication is continuous: every accepted build receives an immutable tag, and nothing about that tag says the image is the one to run. A release is that separate statement, and `scripts/release-vllmb12x.py` makes it in three places at once:

```bash
scripts/release-vllmb12x.py \
  --reference ghcr.io/randomvariable/vllm-b12x-multi@sha256:<digest> \
  --publication-tag vllmb12x-<branch>-<vllm>-<builder>-<date>-n<sequence> \
  --dry-run
```

Without `--dry-run` it tags the image in the registry as `v<YYYYMMDD>.<N>`, tags the commit, and opens a GitHub release whose notes carry the pins, both image references, and the same change ledger the image advertises. The registry tag is written first: a git tag pointing at an image that failed to tag is worse than a registry tag with no release yet.

The command refuses rather than releases when the image does not belong to this tree. It compares the image's vLLM revision, source ref, and description label against the current lock and ledger, requires a clean working tree and a pushed `HEAD`, and requires the builder revision inside the publication tag to be `HEAD` — so a release cannot promise source that never produced the image.

Release tags are calendar, not semantic. This builder tracks moving upstream fork branches, so a version number would imply a compatibility promise it cannot keep.

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
