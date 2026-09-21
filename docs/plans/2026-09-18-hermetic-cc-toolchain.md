# Hermetic ARM64 C/C++ toolchain plan

## Goal

Seal every C/C++ compile, archive, and link action in the VLLMB12X image build to the pinned Ubuntu Resolute GCC 15 package closure. The build must not resolve a compiler, binutils executable, system header, library, linker script, or dynamic loader from a local machine or a NativeLink worker image.

The immediate fault is the registered `//platforms:remote_linux_aarch64_cc_toolchain`. It declares `/usr/bin/aarch64-linux-gnu-*` executables and `/usr/...` builtin include directories in `bazel/remote_cc_toolchain.bzl`. Bazel therefore rejects an external repository action such as `@gawk` that discovers worker headers. Native CUDA actions already extract `//platforms:gcc_15_sysroot` and set `CC`/`CXX`, but that does not control normal Bazel `cc_*` actions.

## Current closure

- `MODULE.bazel:224` registers `//platforms:remote_linux_aarch64_cc_toolchain`.
- `platforms/BUILD.bazel:41` exposes the pinned `@gcc_15_sysroot//:flat` package closure.
- `MODULE.bazel.lock` pins Ubuntu Resolute packages for GCC 15, binutils, glibc headers and runtime libraries.
- `bazel/action_lib.py:169` creates per-action GCC wrappers with `--sysroot` for custom Python/CMake/NCCL actions.
- `bazel/remote_cc_toolchain.bzl:7` bypasses this closure for Bazel C/C++ actions with `/usr/bin/aarch64-linux-gnu-*` and `/usr/...` header paths.
- `@gawk` is a normal external `cc_binary`; it uses the registered cc toolchain and exposes the host-toolchain breach before image assembly.

## Required behavior

1. A clean local build and a remote ARM64 build use the exact same declared GCC 15 closure.
2. Bazel C/C++ actions use compiler, binutils, libc startup objects, headers, linker scripts, and built-in include directories only from that closure.
3. Custom CUDA, CMake, setuptools, Cargo, and NCCL actions continue to use the same closure. They must not retain a secondary host compiler path.
4. Any missing tool required by an action is a declared, pinned input. The action must fail when it is absent rather than fall back to `PATH` or `/usr`.
5. Reproducibility remains deterministic. Toolchain wrapper paths and generated sysroot layouts must not leak action-root paths into artifacts.

## Design

Create a repository rule that materializes a toolchain root from the existing pinned `@gcc_15_sysroot//:flat` artifact. It must:

- unpack the `flat` tar into a repository-owned directory;
- create executable wrappers for `gcc`, `g++`, `ar`, `as`, `ld`, `nm`, `objcopy`, `objdump`, `strip`, and `cpp` that invoke only paths inside that extracted root;
- pass `--sysroot` and `-B` paths inside the root to GCC and binutils;
- expose a filegroup containing the complete root plus wrappers, startup objects, linker scripts, and headers;
- generate tool paths and builtin include directories using repository-relative paths, never absolute `/usr` paths;
- register a `cc_toolchain` that consumes those declared files and targets `@platforms//cpu:aarch64` and Linux.

Use that toolchain for every local and remote ARM64 execution platform. Keep the existing action-level `configure_compiler_sysroot()` only as a temporary compatibility layer while custom build drivers are migrated to the same generated wrappers. Remove it once every driver receives the toolchain artifact directly.

Do not make the toolchain use the host executable `tar`, `ar`, or compiler during an action. Repository setup may use Bazel-provided repository-context archive extraction only. Toolchain bytes must come from `@gcc_15_sysroot//:flat`, itself locked by `MODULE.bazel.lock`.

## Changes

### `bazel/hermetic_cc_toolchain.bzl` (new)

Implement the repository rule and toolchain configuration rule.

- Input: the single `@gcc_15_sysroot//:flat` artifact.
- Output: extracted GCC 15 root, stable compiler/binutils wrappers, generated `BUILD.bazel`, and a toolchain-config target.
- Generate wrappers without an action-root dependency. Resolve their root from the wrapper location.
- Define all `cc_toolchain` file attributes with the complete pinned closure, not `//platforms:empty`.
- Provide compile and link features that consistently pass the sysroot, GCC libexec path, and deterministic path-remapping flags already required by repository reproducibility policy.
- Provide explicit `cxx_builtin_include_directories` rooted in the materialized toolchain, including GCC internal headers and target libc headers. Do not reference `/usr`.

### `MODULE.bazel`

- Instantiate the toolchain repository extension or repository rule using `@gcc_15_sysroot//:flat`.
- Replace registration of `//platforms:remote_linux_aarch64_cc_toolchain` with the generated hermetic ARM64 toolchain.
- Keep the package closure source pinned through the existing rules_distroless lock.

### `platforms/BUILD.bazel`

- Remove the host-path toolchain declaration and `empty` filegroup once the generated toolchain owns its files.
- Retain `gcc_15_sysroot` only if custom Python actions still require the archive. Otherwise make the generated toolchain the single public input.
- Preserve the existing ARM64 target and execution platform constraints.

### `bazel/remote_cc_toolchain.bzl`

- Delete after its only consumer is removed. It encodes worker paths and must not survive the cutover.

### `bazel/action_lib.py`

- Replace ad hoc wrapper creation in `configure_compiler_sysroot()` with the generated toolchain wrapper directory provided as an action input.
- Keep the action-local unpack only for runtime shared-library resolution where the build backend executes native code. Do not create another compiler implementation here.
- Add fail-closed validation that `CC`, `CXX`, linker tools, and every include/library path resolve under the declared toolchain root.

### `bazel/cuda_wheel.bzl`, `bazel/vllm_wheel.bzl`, `bazel/nccl.bzl`

- Add the hermetic toolchain bundle as a declared input and command argument to every custom native action.
- Replace `use_default_shell_env = True` with an explicit minimal environment. Preserve only required Bazel target variables and the declared tool paths.
- Include build tools that upstream invokes directly, beginning with GNU make, patch, CMake, Ninja, Python, Cargo, Rustc, CUDA, ccache, and pkgconf. Each must be an existing declared artifact or a newly pinned input.

### `bazel/cuda_wheel_action.py`, `bazel/vllm_extensions_action.py`, `bazel/nccl_action.py`, `bazel/vllm_package_action.py`

- Accept the hermetic toolchain root or archive explicitly.
- Construct `PATH`, `CC`, `CXX`, `AR`, `AS`, `LD`, `NM`, `OBJCOPY`, `OBJDUMP`, `STRIP`, `PKG_CONFIG`, `CMAKE_*_COMPILER`, `CMAKE_FIND_ROOT_PATH`, and `CMAKE_SYSROOT` exclusively from it.
- Clear host discovery variables such as `CPATH`, `C_INCLUDE_PATH`, `CPLUS_INCLUDE_PATH`, `LIBRARY_PATH`, `COMPILER_PATH`, `GCC_EXEC_PREFIX`, and `PKG_CONFIG_PATH` before running upstream tooling.
- Keep CUDA and Python tool inputs explicit and prepend only their declared locations to `PATH`.
- Make every direct invocation in `action_lib.py` use a declared path. In particular, replace bare `patch` and `make` with paths from pinned tool inputs.

### `bazel/*_test.py` or `bazel/hermetic_cc_toolchain_test.py` (new)

Add a focused hermeticity test that:

- compiles and links a representative `cc_binary` using the registered ARM64 toolchain;
- inspects the action command line or execution log and rejects `/usr/bin`, `/usr/include`, `/usr/lib/gcc`, and undeclared `PATH` tool resolution;
- verifies compiler, header, startup object, and libc library identities come from the locked toolchain bundle;
- runs under a deliberately poisoned `PATH`, `CPATH`, `C_INCLUDE_PATH`, `CPLUS_INCLUDE_PATH`, and `LIBRARY_PATH` to prove no host fallback;
- exercises an external C/C++ target such as `@gawk//:gawk`, which previously failed;
- runs a custom native action probe so Bazel C++ rules and Python-driven CMake/NCCL rules prove the same closure.

### `bazel/reproducibility_test.py` and its target wiring

- Extend the deterministic build lane to rebuild the hermetic toolchain consumer with distinct sandbox roots and compare outputs.
- Record the toolchain artifact digest and generated-wrapper invariants in the existing evidence output.

### `Justfile` and `docs/reference/build-configuration.md`

- Add a narrow `just toolchain-check` command that runs the hermetic C/C++ probe locally with the ARM64 platform, poisoned host discovery, and strict action environment.
- Make `just build` and `just test` select the sealed toolchain by default.
- Document the pinned compiler closure, no-host-fallback rule, how to run the probe, and the expected external `@gawk` proof.

## Execution order

1. Add the materialized hermetic toolchain repository and register it without changing native action drivers.
2. Add the direct Bazel `cc_binary` and external `@gawk` probes. Demonstrate that the original host-path failure is eliminated.
3. Thread the same toolchain bundle through CUDA, vLLM, and NCCL custom actions and replace default shell environments with explicit input-derived environments.
4. Remove the obsolete host-path toolchain and action-local compiler-wrapper implementation.
5. Add poisoned-environment, action-log, reproducibility, image-build, and Docker contract checks.

## Acceptance

- Given no system ARM64 cross compiler is installed, when `just toolchain-check` runs, then an external C/C++ target and a repository C/C++ target compile and link successfully.
- Given arbitrary host compiler and include environment variables, when the native build actions run, then neither commands nor dependency traces contain host `/usr` compiler/header/library paths.
- Given a clean local sandbox and a NativeLink ARM64 worker, when the same target builds, then both use the pinned GCC closure recorded in `MODULE.bazel.lock`.
- Given two distinct sandbox roots, when reproducibility verification rebuilds a native target, then artifact digests match.
- Given the sealed toolchain, when `just build` finishes, then image assembly completes and `just test` proves the image contract, including the mimalloc preload checks.

## Risks

- The present closure contains `g++` and transitive GCC/binutils/glibc packages. The implementation must verify it contains all compiler drivers, startup objects, and linker scripts needed by normal Bazel C++ rules. If a package is absent, add the exact package to the locked `gcc_15_sysroot` set rather than accepting a host fallback.
- `nvcc` invokes its host compiler and may reject wrapper paths or need explicit `-ccbin`. The custom action must set that path explicitly and test it.
- Host-build dependencies such as GNU make and patch are currently implicit through `PATH`. They require their own pinned, declared tool bundle. This is in scope because the requirement is to seal the entire build.
- A local ARM64 executor is required for final proof. Cross compilation from the x86 workstation does not prove the execution path.

## Verification

Run in order:

```sh
just toolchain-check
just build
just test
bazel --ignore_all_rc_files test //bazel:reproducibility_test --platforms=//platforms:spark_arm64_sm121 --spawn_strategy=local --incompatible_strict_action_env
```

Run the equivalent image target and contract test through the existing `remote-aarch64` configuration after local proof. Record the locked toolchain artifact identity and the external `@gawk` action result in build evidence.
