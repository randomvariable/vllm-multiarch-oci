# Build Locally on DGX Spark

Use this guide to run the build actions on a DGX Spark instead of submitting them to NativeLink. No remote service or remote credentials are required.

**Important:** The local command configuration has been analysed, but a complete local Spark build has not been verified. The current actions contain worker-specific compiler paths. A stock DGX OS installation is not automatically a compatible build host.

## Prerequisites

Before you begin, verify that you have the following:

- A Linux ARM64 DGX Spark with the checkout on persistent storage.
- Bazel 9.2.0, Git, GNU tar, patch, sha256sum, Cargo, and host GCC/G++ 15.
- A userspace compatible with the pinned ARM64 build tools. The registered C++ toolchain describes glibc 2.43 and GCC 15, not the stock DGX OS toolchain.
- Persistent writable storage at `/ccache`, including room for a compiler cache configured up to 100 GB and extracted toolchains. Source trees, wheels, and Bazel outputs need additional space.
- Docker for the optional image contract and local image loading.

The build fetches the pinned CUDA toolkit and Python build dependencies. A host CUDA installation alone does not satisfy the host compiler requirements. Do not upgrade a serving machine's OS or compiler in place just to satisfy this guide. Use a compatible isolated ARM64 build environment with persistent checkout and `/ccache` mounts if the host differs.

### Step 1: Check the Host Compiler

Check the host architecture and the compiler paths referenced by the actions:

```bash
uname -m
gcc -dumpfullversion
g++ -dumpfullversion
test -x /usr/bin/aarch64-linux-gnu-gcc
test -d /usr/lib/gcc/aarch64-linux-gnu/15/include
test -d /usr/libexec/gcc/aarch64-linux-gnu/15
```

Expect `aarch64`, GCC/G++ version 15, and successful path checks. The last path comes from the NCCL action. If these paths do not exist, stop: selecting `--config=spark` does not install or relocate the compiler. Do not substitute arbitrary compiler symlinks.

### Step 2: Check Persistent Cache Access

Have the machine administrator provide a persistent directory or mount at `/ccache`, writable by the build account. Check access without removing existing contents:

```bash
test -d /ccache && test -w /ccache
```

Expect exit zero. The actions create `/ccache/objects` and `/ccache/toolchains`. Setting a shell `CCACHE_DIR` does not override these hard-coded paths. A container writable layer or temporary directory loses the cache when that environment is removed.

### Step 3: Analyse Without Private Configuration

Ignore all Bazel rc files, select the ARM64 platform explicitly, and force local execution:

```bash
bazel --ignore_all_rc_files build //image:glm53_0906 --platforms=//platforms:spark_arm64_sm121 --extra_execution_platforms=//platforms:spark_arm64_sm121 --spawn_strategy=local --disk_cache=.bazel-cache --incompatible_strict_action_env --nobuild
```

Expect successful analysis with zero build actions. This bypasses `.bazelrc.user` and the NativeLink configuration entirely. Analysis alone does not verify host executables or their ABI compatibility.

### Step 4: Build on the Spark

Use the same configuration without `--nobuild`:

```bash
bazel --ignore_all_rc_files build //image:glm53_0906 --platforms=//platforms:spark_arm64_sm121 --extra_execution_platforms=//platforms:spark_arm64_sm121 --spawn_strategy=local --disk_cache=.bazel-cache --incompatible_strict_action_env
```

On a compatible host, a successful build produces `bazel-bin/image/glm53_0906`. Allow hours for the first source build and at least six hours in any command supervisor. Local execution permits writes to persistent `/ccache` but does not provide sandbox isolation. The remote-worker timings are not local Spark performance predictions.

### Step 5: Test and Load Locally

Run the structure contract and then load the image using only local actions:

```bash
bazel --ignore_all_rc_files test //tests/image:glm53_0906_contract --platforms=//platforms:spark_arm64_sm121 --extra_execution_platforms=//platforms:spark_arm64_sm121 --spawn_strategy=local --disk_cache=.bazel-cache --incompatible_strict_action_env --test_output=errors
bazel --ignore_all_rc_files run //image:glm53_0906_load --platforms=//platforms:spark_arm64_sm121 --extra_execution_platforms=//platforms:spark_arm64_sm121 --spawn_strategy=local --disk_cache=.bazel-cache --incompatible_strict_action_env
```

Expect a passing structure contract and the local tag `local/vllm:glm53-flash-nvfp4-head-0906`. Neither command starts a model server. The contract does not exercise GPU inference.

## Related Practices

- [Build Configuration](../reference/build-configuration.md) (reference)
- [Build and Cache Design](../explanation/build-and-cache-design.md) (explanation)
- [Compiler Toolchain Paths](../../bazel/remote_cc_toolchain.bzl)
- [NCCL Action Declaration](../../bazel/nccl.bzl)
