# SPDX-License-Identifier: Apache-2.0
"""Build an upstream Python CUDA extension as a content-addressed wheel action."""

load("//bazel:compiler_cache.bzl", "CCACHE_ATTR", "GCC_SYSROOT_ATTR", "MEMORY_PER_JOB_ATTR", "compile_jobs")
load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")

def _cuda_wheel_impl(ctx):
    output = ctx.actions.declare_file(ctx.attr.output)
    args = ctx.actions.args()
    args.add(ctx.file._driver.path)
    args.add("--work-dir", ctx.label.name + ".work")
    args.add("--python", ctx.file.python.path)
    args.add("--src-tar", ctx.file.src.path)
    args.add("--cuda-tar", ctx.file.cuda.path)
    if ctx.file.nccl:
        args.add("--nccl-tar", ctx.file.nccl.path)
    args.add("--build-script", ctx.file.build.path)
    args.add("--ccache-tar", ctx.file.compiler_cache.path)
    args.add("--gcc-sysroot-tar", ctx.file.compiler_sysroot.path)
    args.add("--target-cpu", ctx.attr.target_cpu)
    args.add("--max-jobs", str(compile_jobs(ctx.attr)))
    args.add("--output", output.path)
    rust_toolchain = None
    if ctx.file.cargo_vendor:
        rust_toolchain = ctx.toolchains["@rules_rust//rust:toolchain"]
        args.add("--rustc", rust_toolchain.rustc.path)
        args.add("--cargo", rust_toolchain.cargo.path)
        args.add("--cargo-vendor-tar", ctx.file.cargo_vendor.path)
    args.add_all(ctx.files.host_wheels + ctx.files.deps, before_each = "--host-wheel")

    inputs = [
        ctx.file.src,
        ctx.file.cuda,
        ctx.file.build,
        ctx.executable._python_launcher,
        ctx.file.compiler_cache,
        ctx.file.compiler_sysroot,
        ctx.file._driver,
        ctx.file._action_lib,
    ] + ctx.files.python_runtime + ctx.files.python_headers + ctx.files.host_wheels + ctx.files.deps + ctx.files.nccl
    if rust_toolchain:
        inputs += [ctx.file.cargo_vendor, rust_toolchain.rustc, rust_toolchain.cargo]
    ctx.actions.run(
        executable = ctx.executable._python_launcher,
        arguments = [args],
        # rustc dynamically loads compiler crates beside its executable. The
        # toolchain's complete closure must travel with the remote action.
        inputs = depset(
            direct = inputs,
            transitive = [rust_toolchain.all_files] if rust_toolchain else [],
        ),
        outputs = [output],
        mnemonic = "BuildCUDAWheel",
        progress_message = "Building %s from source" % ctx.label.name,
        execution_requirements = {
            "cpu": str(ctx.attr.cpu),
            "ISA": ctx.attr.target_cpu,
            "memory": str(ctx.attr.memory),
        },
        env = dict(ctx.attr.env, **_cache_env(ctx)),
        use_default_shell_env = False,
    )
    return [DefaultInfo(files = depset([output]))]

cuda_wheel = rule(
    implementation = _cuda_wheel_impl,
    attrs = {
        "src": attr.label(mandatory = True, allow_single_file = True),
        "cuda": attr.label(mandatory = True, allow_single_file = True),
        "python": attr.label(mandatory = True, allow_single_file = True),
        "_python_launcher": attr.label(
            default = Label("//platforms:action_python"),
            executable = True,
            cfg = "target",
        ),
        # The interpreter binary alone has an /install prefix and cannot
        # initialize remotely without its standard-library tree.
        "python_runtime": attr.label(
            default = Label("@python_3_12//:files"),
            allow_files = True,
        ),
        # The interpreter reports its in-tree INCLUDEPY. Make those headers
        # action inputs, otherwise C/C++ extension builds cannot resolve Python.h.
        "python_headers": attr.label(
            default = Label("@python_3_12//:includes"),
            allow_files = True,
        ),
        "build": attr.label(mandatory = True, allow_single_file = True),
        "compiler_cache": CCACHE_ATTR,
        "compiler_sysroot": GCC_SYSROOT_ATTR,
        "target_cpu": attr.string(mandatory = True, values = ["aarch64", "x86_64"]),
        "output": attr.string(mandatory = True),
        "host_wheels": attr.label_list(allow_files = True),
        "deps": attr.label_list(allow_files = True),
        "cargo_vendor": attr.label(allow_single_file = True),
        "env": attr.string_dict(),
        "_cache_root": attr.label(default = Label("//platforms:vllmb12x_cache_root")),
        "nccl": attr.label(allow_single_file = True),
        # Upper bound only. compile_jobs() reduces this to what the action's
        # CPU and memory reservation affords.
        "max_jobs": attr.int(default = 64),
        "cpu": attr.int(default = 20),
        "memory": attr.int(default = 32768),
        "memory_per_job": MEMORY_PER_JOB_ATTR,
        "_driver": attr.label(
            default = Label("//bazel:cuda_wheel_action.py"),
            allow_single_file = True,
        ),
        "_action_lib": attr.label(
            default = Label("//bazel:action_lib.py"),
            allow_single_file = True,
        ),
    },
    toolchains = [
        "@bazel_tools//tools/cpp:toolchain_type",
        "@rules_rust//rust:toolchain",
    ],
)


def _cache_env(ctx):
    root = ctx.attr._cache_root[BuildSettingInfo].value
    return {} if not root else {
        "CCACHE_DIR": root + "/objects",
        "VLLMB12X_CACHE_ROOT": root,
    }
