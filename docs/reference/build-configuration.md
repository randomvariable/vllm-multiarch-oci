# Build Configuration

This reference describes the Bazel builder for [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm) on consumer Blackwell. The profile builds one OCI index containing native ARM64 and x86-64 image manifests.

## Targets

| Target | Output or Behaviour |
| --- | --- |
| `//image:vllmb12x` | Multiarchitecture OCI image index |
| `//image:vllmb12x_image` | OCI image layout selected by the configured target platform |
| `//image:vllmb12x_load` | Local image loader |
| `//image:runtime_venv` | Python, NCCL, runtime wheels, and non-vLLM source wheels |
| `//image:vllm_venv` | vLLM-only overlay tar |
| `//components:torch` | Combined native-and-Python PyTorch wheel |
| `//components:vllm` | vLLM wheel |
| `//components:vllm_compiler_cache_stats` | Action-local compiler-cache statistics |
| `//tests/image:vllmb12x_contract` | Docker-backed metadata and file contract selected by platform |

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
| `just build-multiarch [args...]` | Local analysis or build of the ARM64 and x86-64 OCI index |
| `just test [args...]` | Local Docker-backed image contract |
| `just load [args...]` | Load the image into the local daemon |
| `just refresh-vllmb12x` | Resolve the moving vLLM branch into the immutable profile lock |
| `just bazel <command> [args...]` | Raw Bazel command without rc files; no platform or execution flags added |

Build, test, and load select the ARM64 target and execution platform, local execution, `.bazel-cache`, and the strict action environment. Local source actions use standard ccache storage at `$XDG_CACHE_HOME/ccache`, defaulting to `~/.cache/ccache`. CI overwrites `VLLMB12X_CACHE_ROOT` with its durable cache volume independently of Bazel's disk cache.

The generated [VLLMB12X Runtime Configuration](vllmb12x-runtime-configuration.md) lists every B12X runtime environment reader and every changed local-inference-lab/vLLM environment, CLI, or `additional_config` control relative to its pinned stock-vLLM merge-base. Regenerate it with `scripts/vllmb12x-runtime-config.py` whenever this profile changes source commits or patches.

## Nightly Publication

The public PAC definition at [`.tekton/vllmb12x-nightly.yaml`](../../.tekton/vllmb12x-nightly.yaml) builds the locked OCI index and runs the image contract independently for the ARM64 and x86-64 remote workers. The publisher then materializes `//image:vllmb12x`, acquires a short Kubernetes Lease, and pushes the OCI index with a host `crane` rather than `//image:vllmb12x_push`. `rules_oci` packages `crane` and `jq` as exec-platform runfiles, so `bazel run` under a remote configuration resolves binaries for the remote executor instead of the pipeline pod. Bazel symlinks the base-image and apt blobs into its external repository directories, and `crane` rejects a layout blob that is a symlink, so the publisher copies the layout beside the output base with hard links and pushes that copy.

`scripts/publish-vllmb12x.py` gives each publication an immutable tag with this form:

```text
vllmb12x-<source-branch>-<vllm-commit-12>-<builder-commit-12>-<UTC-date>-n<sequence>
```

For example, `refs/heads/dev/jovian-judgement` becomes `dev-jovian-judgement`. The profile stores the full source ref and full commit. The tag uses normalized branch text and 12-character commit prefixes for operators. The Lease transition number makes repeated builds of the same revisions distinct. The publisher releases the Lease by shortening it rather than deleting it. A delayed run cannot remove a later holder's lock. The image also receives `latest` after its immutable tag succeeds.

The OCI metadata identifies the built vLLM source through `org.opencontainers.image.source`, `org.opencontainers.image.revision`, and `org.opencontainers.image.version`. Image-specific labels retain the full source ref, vLLM revision, vLLM version, and builder repository. The embedded `/opt/vllmb12x/sources.lock.json` records the complete dependency lock.

The public pipeline names only PAC parameters. The private PAC Repository binds registry, authentication, remote-execution, storage, and Lease details.

All three lanes invoke the same committed scripts: `scripts/ci/bootstrap-arm64-tools.sh` installs the pinned toolchain, `scripts/ci/login-ghcr.sh` writes the crane configuration for the public registry, and `scripts/ci/vllmb12x-build-and-publish.sh` runs the tests, the image build, both contract lanes, and the publisher. A lane therefore differs only by its `LANE` environment value and its publisher arguments, which is what `scripts/ci/vllmb12x_lane_test.py` asserts against stub tools.

Every CI lane requests a 256 GiB CSI workspace. `HOME`, `TMPDIR`, and
Bazel's `output_user_root` stay on that volume, so source extraction and OCI
assembly do not fill the node root filesystem through Tekton's `/tekton/home`
`emptyDir`. This is per-run staging, not a build cache: deleting its PVC removes
the staged files. The shared ccache and remote action cache remain separate.
The checkout occupies `checkout/` and staging occupies its sibling
`.build-home/`. Bazel rejects its repository contents cache inside the checkout.

## Pull-Request Builds

[`.tekton/vllmb12x-pull-request.yaml`](../../.tekton/vllmb12x-pull-request.yaml) builds and contract-tests the same image for a pull request against `main`, then pushes it to the internal registry only. It skips the same documentation-only paths as the nightly, and a new commit on the pull request cancels the run building the previous one.

`scripts/publish-vllmb12x.py --pull-request <number>` differs from a nightly publication in three ways, all of which follow from the tag being reachable only by name:

```text
pr-<number>-<vllm-commit-12>-<builder-commit-12>
```

The tag carries no sequence and no date, so re-running a pull-request build overwrites its own image instead of adding another. Nothing else points at the image, so the publisher takes no Lease. `latest` is never retagged, in either registry.

## Build Serialization

All three lanes build on one node, against one remote worker pool and one ccache volume. Running them at once halves the workers each build sees and evicts the other lane's cache entries, so `scripts/build-mutex.py` holds a Kubernetes Lease named by the `build_lease` parameter for the whole build, test, and publication sequence, and the other lane waits for it.

The Lease duration is short relative to a build and the holder renews it, so a cancelled or killed run blocks the other lane only until its Lease expires, not until someone cleans up. A single failed renewal is not treated as a lost Lease: ownership ends only when another holder appears, the Lease is deleted, or the last successful renewal could itself have expired. The publication Lease is separate and still covers only tag allocation and registry mutation.

## Releases

A release is the deliberate statement that one image is the one to run, and pushing a tag is what makes it:

```bash
git tag --annotate v20261003.1 --message 'v20261003.1: main at <commit>'
git push origin v20261003.1
```

[`.tekton/vllmb12x-release.yaml`](../../.tekton/vllmb12x-release.yaml) matches a `push` whose ref is `refs/tags/v<YYYYMMDD>.<N>`, checks out exactly the tagged commit, and runs the nightly's sequence unchanged: `//scripts:all`, the multiarch image build, then the image contract on each architecture. Publication follows the green tests, and only then does `scripts/release-vllmb12x.py` copy the published image to the release tag and open the GitHub release whose notes carry the pins, both image references, and the change ledger the image advertises.

A failed build or contract lane therefore creates nothing. The git tag stays as the request, the registry gains no release tag, and no release is opened. That is the one property the order buys: the tag is intent, and only a green run turns it into a release.

Two checks keep a tag from naming a build that never happened:

- the publication tag's builder revision must equal the checked-out `HEAD`, and the image's vLLM revision, source ref, and description label must match the committed lock and ledger, so a release cannot promise source that never produced the image;
- `--expected-revision` binds the checkout to the commit the webhook resolved from the tag, so the run promotes what was tagged rather than whatever `main` has since become.

Push at most three tags at a time; GitHub suppresses tag webhook deliveries beyond that. Repeating a tag is safe: the release tag is re-pointed at the newest publication and the existing GitHub release is updated rather than duplicated.

Release tags are calendar, not semantic. This builder tracks moving upstream fork branches, so a version number would imply a compatibility promise it cannot keep.

The script also runs by hand against an image the nightly published, which is how a release is cut when the lane is unavailable. Omit `--tag-exists` and it allocates the next `v<YYYYMMDD>.<N>` itself, and it requires a clean working tree and a pushed `HEAD` on `origin/main`:

```bash
scripts/release-vllmb12x.py \
  --reference ghcr.io/randomvariable/vllm-b12x-multi@sha256:<digest> \
  --publication-tag vllmb12x-<branch>-<vllm>-<builder>-<date>-n<sequence> \
  --dry-run
```

## Image Version

The version baked into the wheel and advertised by the image composes three values:

```
<upstream vLLM base>+<local-inference-lab cycle>.<vLLM commit>.<B12X commit>.<source digest>
0.29.0+karmic.kraken.<vLLM commit>.<B12X commit>.<source digest>
```

The commits and the digest are per-lock, so the value above illustrates the shape rather than naming a build; the value for the current lock is in `profiles/vllmb12x/version.bzl`.

- **Base** is the upstream release this cycle tracks. The cycle branch merges upstream pull requests selectively, so tag ancestry does not prove which release it descends from. The value is a reviewed claim: `vllm_base_version` in `profiles/vllmb12x/profile.json`, set with `scripts/refresh-vllmb12x.py --vllm-base-version`.
- **Cycle** is the branch `source_ref` names, without its `cycle/` or `dev/` namespace, spelled as packaging spells a local version segment: `cycle/karmic-kraken` becomes `karmic.kraken`. Packaging rewrites `-` and `_` to `.` there, so building from the branch spelling would leave the wheel filename and distribution metadata disagreeing with the label.
- **Commits** are the vLLM and B12X revisions the manifest pins, each named by its first twelve hexadecimal digits, the same form the publication tag uses. Those two move independently, so naming both answers which B12X an image carries without unpacking the manifest.
- **Digest** is the first twelve hexadecimal digits of the SHA-256 over `source_ref` and every locked `commit` and `remote`, so it also moves when a source neither named commit covers — torch, NCCL, a CMake download — changes. Any reader recomputes it from the committed manifest:

```bash
python3 -c '
import hashlib, json, pathlib
m = json.loads(pathlib.Path("profiles/vllmb12x/profile.json").read_text())
c = json.dumps({"source_ref": m["source_ref"],
                "sources": {n: {"commit": s["commit"], "remote": s["remote"]}
                            for n, s in sorted(m["sources"].items())}},
               separators=(",", ":"), sort_keys=True)
print(hashlib.sha256(c.encode()).hexdigest()[:12])'
```

A calendar component would make two builds of identical inputs disagree, so the digest takes its place. `scripts/refresh-vllmb12x.py` writes `profiles/vllmb12x/version.bzl`, whose value sets `VLLM_VERSION_OVERRIDE` for the wheel build; the wheel filename, `vllm.__version__`, and the `org.opencontainers.image.version` label therefore carry one string. The local loader tag substitutes `-` for `+`, because a docker tag cannot contain `+`.

## Optional CI Configuration

NativeLink is our optional CI backend. The public `remote-aarch64` configuration selects remote execution with no local fallback, minimal downloads, compression, and a 21600-second remote timeout. Endpoints, authentication, and execution properties must be supplied separately by CI. No private infrastructure is configured in the repository. Local Justfile recipes do not use this configuration.

## Runtime

| Property | Value |
| --- | --- |
| Image tag | `randomvariable/vllm-b12x-multi:<build version>` |
| Platform | `linux/arm64` (SM12x family) and `linux/amd64` (SM120) |
| CUDA base | CUDA 13.4.1 cuDNN development image, Ubuntu 26.04, digest-pinned |
| Python | 3.12 |
| Entrypoint | `/opt/venv/bin/vllm` |
| Working directory | `/root` |
| CUDA home | `/usr/local/cuda` |
| NCCL library path | `/opt/nccl/lib` |
| Allocator | mimalloc preloaded from the architecture-specific Ubuntu library path |
| C/C++ toolchain | Pinned Ubuntu Resolute GCC 15 closure materialized by Bazel, with no worker `/usr` compiler or include paths |
| Torch architecture | `12.0+PTX`; PyTorch's parser does not support the CUDA SM12x family suffix, so the SM120 PTX lets SM121 drivers JIT device code |
| FlashInfer architecture | `12.0f` |
| Disabled kernel | `MarlinFP8ScaledMMLinearKernel` |
| Unusable control | `B12X_DENSE_ATOM_24=1` aborts preparation on SM120 with `std::get: wrong index for variant`, raised from `b12x.gemm.blockscaled._preparation:compile_packed` at engine init. Reported by the Qwen deployment that measured it; not re-measured on the revisions this profile currently pins. Treat it as unusable rather than slow and do not spend a restart on it. |
| Mooncake Transfer Engine | CUDA 13 distribution `0.3.13.post1`; see [Mooncake Transfer Engine](mooncake-transfer-engine.md) |
| In-container build prerequisites | `cmake`, `liburing-dev`, `libxxhash-dev`, `libhiredis-dev` and `libcurl4t64`. The first four are there because b12x and FlexKV compile parts of themselves inside the running container — b12x's PLE reader and RoCE proxy, FlexKV's C++ core through its `build.sh` — rather than only at image build time. `libcurl4t64` is different: the transfer engine loads `libcurl.so.4` at run time and the CUDA base ships no curl at all. Do not remove them as unused build tooling; FlexKV additionally vendors xxHash, so `libxxhash-dev` follows its documented prerequisites rather than its build. |

Source: [image rule](../../bazel/vllm_image.bzl). Each image uses a native architecture-specific PyTorch wheel, not a `torch_libtorch`/Python-only split.

Run `just toolchain-check` to compile the `@gawk` bootstrap dependency through the registered toolchain and verify that poisoned worker include paths do not enter C/C++ actions.

On an x86-64 build client, ARM64 Python build actions use the pinned `qemu-user-binfmt-hwe` package with the declared GCC sysroot. Native ARM64 workers run the same interpreter directly. Neither path uses host binfmt registration or `/lib/ld-linux-aarch64.so.1`.

## Further Reading

- [Build and Test an Image](../how-to/build-and-test.md)

---

**Last Updated:** September 2026
**Version Compatibility:** Checked-in Bazel 9.2.0 build and `vllmb12x` profile
