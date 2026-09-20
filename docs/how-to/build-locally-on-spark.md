# Build Locally on ARM64 Blackwell

Use this guide to run the ARM64 Blackwell SM12x build actions on a DGX Spark or an ARM64 RTX PRO 6000 Blackwell host instead of submitting them to NativeLink. No remote service or remote credentials are required.

**Important:** The local command configuration has been analysed, but a complete local ARM64 Blackwell build has not been verified. Bazel materializes the pinned GCC 15 closure for C/C++ actions. The build does not use the host compiler, headers, or libraries. The maintainers build through CI; this path is community-maintained, and reports or fixes are welcome.

## Prerequisites

Before you begin, verify that you have the following:

- A Linux ARM64 Blackwell host with the checkout on persistent storage.
- Bazel 9.2.0, Git, and Cargo.
- A userspace capable of running the pinned ARM64 toolchain. Bazel materializes GCC 15 and glibc 2.43 from the locked Ubuntu package closure.
- Persistent writable storage for the standard ccache directory, `$XDG_CACHE_HOME/ccache` or `~/.cache/ccache`, including room for a compiler cache configured up to 100 GB. Source trees, wheels, extracted toolchains, and Bazel outputs need additional space.
- Docker for the optional image contract and local image loading.

The build fetches the pinned CUDA toolkit and Python build dependencies. Do not upgrade a serving machine's compiler in place for this build. Use a compatible isolated ARM64 build environment with persistent checkout and cache directory if the host differs.

### Step 1: Check the Build Host

Check the host architecture and Cargo:

```bash
uname -m
cargo --version
```

Expect `aarch64` and successful command checks. The locked GCC archive provides the compiler, headers, binutils, libc, and linker inputs. The `120f` family target covers SM120 and SM121 devices. Do not inject host compiler paths through `CC`, `CXX`, `CPATH`, or `LIBRARY_PATH`.

### Step 2: Check Persistent Cache Access

Check the standard ccache directory without removing existing contents:

```bash
cache_root=${XDG_CACHE_HOME:-$HOME/.cache}/ccache
mkdir -p "$cache_root" && test -w "$cache_root"
```

Expect exit zero. ccache stores compiler objects at `$cache_root`. The actions store extracted toolchains at `${XDG_CACHE_HOME:-$HOME/.cache}/vllmb12x/toolchains`. CI sets `VLLMB12X_CACHE_ROOT` to its durable mount, which moves objects to `$VLLMB12X_CACHE_ROOT/objects` and toolchains to `$VLLMB12X_CACHE_ROOT/toolchains`. A container writable layer or temporary directory loses either cache when that environment is removed.

### Step 3: Analyse Without Private Configuration

Ignore all Bazel rc files, select the ARM64 platform explicitly, and force local execution:

```bash
bazel --ignore_all_rc_files build //image:vllmb12x --platforms=//platforms:blackwell_arm64_sm12x --extra_execution_platforms=//platforms:blackwell_arm64_sm12x --spawn_strategy=local --disk_cache=.bazel-cache --incompatible_strict_action_env --nobuild
```

Expect successful analysis with zero build actions. This bypasses `.bazelrc.user` and the NativeLink configuration entirely. Analysis alone does not verify host executables or their ABI compatibility.

### Step 4: Build on ARM64 Blackwell

Use the same configuration without `--nobuild`:

```bash
bazel --ignore_all_rc_files build //image:vllmb12x --platforms=//platforms:blackwell_arm64_sm12x --extra_execution_platforms=//platforms:blackwell_arm64_sm12x --spawn_strategy=local --disk_cache=.bazel-cache --incompatible_strict_action_env
```

On a compatible host, a successful build produces `bazel-bin/image/vllmb12x`. Allow hours for the first source build and at least six hours in any command supervisor. Local execution permits writes to persistent standard cache directories but does not provide sandbox isolation. The remote-worker timings are not local Spark performance predictions.

### Step 5: Test and Load Locally

Run the structure contract and then load the image using only local actions:

```bash
bazel --ignore_all_rc_files test //tests/image:vllmb12x_contract --platforms=//platforms:blackwell_arm64_sm12x --extra_execution_platforms=//platforms:blackwell_arm64_sm12x --spawn_strategy=local --disk_cache=.bazel-cache --incompatible_strict_action_env --test_output=errors
bazel --ignore_all_rc_files run //image:vllmb12x_load --platforms=//platforms:blackwell_arm64_sm12x --extra_execution_platforms=//platforms:blackwell_arm64_sm12x --spawn_strategy=local --disk_cache=.bazel-cache --incompatible_strict_action_env
```

Expect a passing structure contract and the local tag `randomvariable/vllm-b12x-multi:<build version>`. Neither command starts a model server. The contract does not exercise GPU inference.

Run `just toolchain-check` before an image build to compile and link the C++ probe through the pinned GCC 15 package closure. The check rejects worker compiler include paths.

## Related Practices

- [Build Configuration](../reference/build-configuration.md) (reference)
- [Build and Cache Design](../explanation/build-and-cache-design.md) (explanation)
- [Hermetic Compiler Toolchain](../../bazel/hermetic_cc_toolchain.bzl)
- [NCCL Action Declaration](../../bazel/nccl.bzl)
