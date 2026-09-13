# SPDX-License-Identifier: Apache-2.0
"""Build the profile-pinned NCCL tree against the hermetic CUDA toolkit."""

load("//bazel:compiler_cache.bzl", "CCACHE_ATTR", "GCC_SYSROOT_ATTR")


def _nccl_lib_impl(ctx):
    output = ctx.actions.declare_file(ctx.attr.output)
    sdk = ctx.actions.declare_file(ctx.attr.sdk_output)
    arch = ctx.attr.cuda_arch
    compiler_arch = arch[:-1] if arch[-1] in "af" else arch
    args = ctx.actions.args()
    args.add(ctx.file._driver.path)
    args.add("--work-dir", ctx.label.name + ".work")
    args.add("--python", ctx.file.python.path)
    args.add("--src-tar", ctx.file.src.path)
    args.add("--cuda-tar", ctx.file.cuda.path)
    args.add("--ccache-tar", ctx.file.compiler_cache.path)
    args.add("--gcc-sysroot-tar", ctx.file.compiler_sysroot.path)
    args.add("--cuda-arch", compiler_arch)
    args.add("--commit", ctx.attr.commit)
    args.add("--jobs", ctx.attr.jobs)
    args.add("--lib-output", output.path)
    args.add("--sdk-output", sdk.path)

    ctx.actions.run(
        executable = ctx.file.python,
        inputs = depset(
            direct = [
                ctx.file.src,
                ctx.file.cuda,
                ctx.file.compiler_cache,
                ctx.file.compiler_sysroot,
                ctx.file.python,
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
            "memory": "32768",
        },
        use_default_shell_env = True,
    )
    return [
        DefaultInfo(files = depset([output])),
        OutputGroupInfo(sdk = depset([sdk])),
    ]


nccl_lib = rule(
    implementation = _nccl_lib_impl,
    exec_compatible_with = [
        "@platforms//cpu:aarch64",
        "@platforms//os:linux",
    ],
    attrs = {
        "src": attr.label(mandatory = True, allow_single_file = True),
        "cuda": attr.label(mandatory = True, allow_single_file = True),
        "compiler_cache": CCACHE_ATTR,
        "compiler_sysroot": GCC_SYSROOT_ATTR,
        "commit": attr.string(mandatory = True),
        "cuda_arch": attr.string(mandatory = True),
        "python": attr.label(mandatory = True, allow_single_file = True),
        # The generated NCCL source invokes python3 while writing version
        # metadata. Supply rules_python's relocatable runtime to the action.
        "python_runtime": attr.label(
            default = Label("@python_3_12//:files"),
            allow_files = True,
        ),
        "jobs": attr.int(default = 4),
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
)
