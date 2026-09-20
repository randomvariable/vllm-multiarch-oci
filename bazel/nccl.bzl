# SPDX-License-Identifier: Apache-2.0
"""Build the profile-pinned NCCL tree against the hermetic CUDA toolkit."""

load("//bazel:compiler_cache.bzl", "CCACHE_ATTR", "GCC_SYSROOT_ATTR")
load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")


def _nccl_lib_impl(ctx):
    output = ctx.actions.declare_file(ctx.attr.output)
    sdk = ctx.actions.declare_file(ctx.attr.sdk_output)
    arch = ctx.attr.cuda_arch
    args = ctx.actions.args()
    args.add(ctx.file._driver.path)
    args.add("--work-dir", ctx.label.name + ".work")
    args.add("--python", ctx.file.python.path)
    args.add("--src-tar", ctx.file.src.path)
    args.add("--cuda-tar", ctx.file.cuda.path)
    args.add("--ccache-tar", ctx.file.compiler_cache.path)
    args.add("--gcc-sysroot-tar", ctx.file.compiler_sysroot.path)
    args.add("--target-cpu", ctx.attr.target_cpu)
    args.add("--cuda-arch", arch)
    args.add("--commit", ctx.attr.commit)
    args.add("--jobs", ctx.attr.jobs)
    args.add("--lib-output", output.path)
    args.add("--sdk-output", sdk.path)

    ctx.actions.run(
        executable = ctx.executable._python_launcher,
        inputs = depset(
            direct = [
                ctx.file.src,
                ctx.file.cuda,
                ctx.file.compiler_cache,
                ctx.file.compiler_sysroot,
                ctx.executable._python_launcher,
                ctx.file._driver,
                ctx.file._action_lib,
            ] + ctx.files.python_runtime,
        ),
        outputs = [output, sdk],
        arguments = [args],
        mnemonic = "BuildNCCL",
        progress_message = "Building patched NCCL 2.30.4 for sm_%s" % arch,
        execution_requirements = {
            "cpu": "20",
            "ISA": ctx.attr.target_cpu,
            "memory": "32768",
        },
        env = _cache_env(ctx),
        use_default_shell_env = False,
    )
    return [
        DefaultInfo(files = depset([output])),
        OutputGroupInfo(sdk = depset([sdk])),
    ]


nccl_lib = rule(
    implementation = _nccl_lib_impl,
    attrs = {
        "src": attr.label(mandatory = True, allow_single_file = True),
        "cuda": attr.label(mandatory = True, allow_single_file = True),
        "compiler_cache": CCACHE_ATTR,
        "compiler_sysroot": GCC_SYSROOT_ATTR,
        "target_cpu": attr.string(mandatory = True, values = ["aarch64", "x86_64"]),
        "commit": attr.string(mandatory = True),
        "cuda_arch": attr.string(mandatory = True),
        "python": attr.label(mandatory = True, allow_single_file = True),
        "_python_launcher": attr.label(
            default = Label("//platforms:action_python"),
            executable = True,
            cfg = "target",
        ),
        # The generated NCCL source invokes python3 while writing version
        # metadata. Supply rules_python's relocatable runtime to the action.
        "python_runtime": attr.label(
            default = Label("@python_3_12//:files"),
            allow_files = True,
            cfg = "target",
        ),
        "jobs": attr.int(default = 4),
        "_cache_root": attr.label(default = Label("//platforms:vllmb12x_cache_root")),
        "output": attr.string(default = "libnccl.so.2.30.4"),
        "sdk_output": attr.string(default = "nccl-sdk.tar"),
        "_driver": attr.label(
            default = Label("//bazel:nccl_action.py"),
            allow_single_file = True,
        ),
        "_action_lib": attr.label(
            default = Label("//bazel:action_lib.py"),
            allow_single_file = True,
        ),
    },
    # The Python launcher and CUDA toolchain match the target architecture.
    # Requiring its C++ toolchain also makes Bazel select that architecture's
    # execution platform under the OCI index transition.
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
)


def _cache_env(ctx):
    root = ctx.attr._cache_root[BuildSettingInfo].value
    return {} if not root else {
        "CCACHE_DIR": root + "/objects",
        "VLLMB12X_CACHE_ROOT": root,
    }
