# SPDX-License-Identifier: Apache-2.0
"""Build an upstream Python CUDA extension as a content-addressed wheel action."""

load("//bazel:compiler_cache.bzl", "CCACHE_ATTR")

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
    args.add("--max-jobs", str(min(ctx.attr.max_jobs, 4)))
    args.add("--output", output.path)
    args.add_all(ctx.files.host_wheels + ctx.files.deps, before_each = "--host-wheel")

    inputs = [
        ctx.file.src,
        ctx.file.cuda,
        ctx.file.build,
        ctx.file.python,
        ctx.file.compiler_cache,
        ctx.file._driver,
        ctx.file._action_lib,
    ] + ctx.files.python_runtime + ctx.files.python_headers + ctx.files.host_wheels + ctx.files.deps + ctx.files.nccl
    ctx.actions.run(
        executable = ctx.file.python,
        arguments = [args],
        inputs = depset(direct = inputs),
        outputs = [output],
        mnemonic = "BuildCUDAWheel",
        progress_message = "Building %s from source" % ctx.label.name,
        execution_requirements = {
            "cpu": str(ctx.attr.cpu),
            "memory": str(ctx.attr.memory),
        },
        env = ctx.attr.env,
        # Upstream setup.py/CMake builds invoke the distribution compiler.
        use_default_shell_env = True,
    )
    return [DefaultInfo(files = depset([output]))]

cuda_wheel = rule(
    implementation = _cuda_wheel_impl,
    attrs = {
        "src": attr.label(mandatory = True, allow_single_file = True),
        "cuda": attr.label(mandatory = True, allow_single_file = True),
        "python": attr.label(mandatory = True, allow_single_file = True),
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
        "output": attr.string(mandatory = True),
        "host_wheels": attr.label_list(allow_files = True),
        "deps": attr.label_list(allow_files = True),
        "env": attr.string_dict(),
        "nccl": attr.label(allow_single_file = True),
        "max_jobs": attr.int(default = 4),
        "cpu": attr.int(default = 20),
        "memory": attr.int(default = 32768),
        "_driver": attr.label(
            default = Label("//bazel:cuda_wheel_action.py"),
            allow_single_file = True,
        ),
        "_action_lib": attr.label(
            default = Label("//bazel:action_lib.py"),
            allow_single_file = True,
        ),
    },
)
