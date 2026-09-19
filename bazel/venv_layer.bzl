# SPDX-License-Identifier: Apache-2.0
"""Create a self-contained Python runtime layer from locked wheels."""

def _venv_layer_impl(ctx):
    output = ctx.actions.declare_file(ctx.attr.output)
    args = ctx.actions.args()
    args.add(ctx.file._driver.path)
    args.add("--work-dir", output.path + ".work")
    args.add("--python", ctx.file.python.path)
    args.add("--nccl-tar", ctx.file.nccl.path)
    args.add("--python-launcher", ctx.file._python_launcher.path)
    args.add("--vllm-launcher", ctx.file._vllm_launcher.path)
    args.add("--image-helper", ctx.file._image_helper.path)
    args.add("--image-helper-launcher", ctx.file._image_helper_launcher.path)
    args.add("--output", output.path)
    if ctx.attr.include_python:
        args.add("--include-python")
    if ctx.attr.include_nccl:
        args.add("--include-nccl")
    if ctx.attr.include_launchers:
        args.add("--include-launchers")
    args.add_all(ctx.files.runtime_wheels + ctx.files.source_wheels, before_each = "--wheel")

    ctx.actions.run(
        executable = ctx.executable._action_python,
        inputs = depset(
            direct = [
                ctx.executable._action_python,
                ctx.file._action_lib,
                ctx.file._driver,
                ctx.file._python_launcher,
                ctx.file._vllm_launcher,
                ctx.file._image_helper,
                ctx.file._image_helper_launcher,
                ctx.file.nccl,
            ] + ctx.files.python_runtime + ctx.files.runtime_wheels + ctx.files.source_wheels,
        ),
        outputs = [output],
        arguments = [args],
        mnemonic = "AssembleVenvLayer",
        progress_message = "Assembling Python runtime layer %{label}",
        use_default_shell_env = False,
    )
    return [DefaultInfo(files = depset([output]))]

venv_layer = rule(
    implementation = _venv_layer_impl,
    attrs = {
        "python": attr.label(mandatory = True, allow_single_file = True),
        "_action_python": attr.label(
            default = Label("//platforms:action_python"),
            executable = True,
            cfg = "target",
        ),
        "python_runtime": attr.label(mandatory = True, allow_files = True),
        "nccl": attr.label(mandatory = True, allow_single_file = True),
        "runtime_wheels": attr.label_list(allow_files = True),
        "source_wheels": attr.label_list(allow_files = True),
        "output": attr.string(mandatory = True),
        "include_python": attr.bool(default = False),
        "include_nccl": attr.bool(default = False),
        "include_launchers": attr.bool(default = False),
        "_action_lib": attr.label(
            default = Label("//bazel:action_lib.py"),
            allow_single_file = True,
        ),
        "_driver": attr.label(
            default = Label("//bazel:venv_layer_action.py"),
            allow_single_file = True,
        ),
        "_python_launcher": attr.label(
            default = Label("//bazel:venv_python_launcher.sh"),
            allow_single_file = True,
        ),
        "_vllm_launcher": attr.label(
            default = Label("//bazel:venv_vllm_launcher.sh"),
            allow_single_file = True,
        ),
        "_image_helper": attr.label(
            default = Label("//image_tools:vllm_image.py"),
            allow_single_file = True,
        ),
        "_image_helper_launcher": attr.label(
            default = Label("//image_tools:vllm-image.sh"),
            allow_single_file = True,
        ),
    },
    # The selected action Python is architecture-native. Resolve the matching
    # C++ toolchain to constrain this packaging action to that same executor.
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
)
