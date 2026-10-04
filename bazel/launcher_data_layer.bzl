# SPDX-License-Identifier: Apache-2.0
"""Install the pinned upstream model-policy data and our recipes as one layer."""

def _launcher_data_layer_impl(ctx):
    output = ctx.actions.declare_file(ctx.attr.output)
    args = ctx.actions.args()
    args.add(ctx.file._driver.path)
    args.add("--work-dir", output.path + ".work")
    args.add("--upstream-source", ctx.file.upstream_source.path)
    args.add("--output", output.path)
    args.add_all(ctx.files.recipes, before_each = "--recipe")
    args.add_all(ctx.files.hardware, before_each = "--hardware")

    ctx.actions.run(
        executable = ctx.executable._python,
        inputs = depset(direct = [
            ctx.executable._python,
            ctx.file._action_lib,
            ctx.file._driver,
            ctx.file.upstream_source,
        ] + ctx.files.recipes + ctx.files.hardware + ctx.files._python_runtime),
        outputs = [output],
        arguments = [args],
        mnemonic = "AssembleLauncherDataLayer",
        progress_message = "Installing launcher policy data %{label}",
        use_default_shell_env = False,
    )
    return [DefaultInfo(files = depset([output]))]

launcher_data_layer = rule(
    implementation = _launcher_data_layer_impl,
    # The driver reads only the declared archives with the standard library, so
    # pinning it to the host interpreter keeps the action off the QEMU-emulated
    # ARM64 Python that a cross build would otherwise select.
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    attrs = {
        "upstream_source": attr.label(
            mandatory = True,
            allow_single_file = True,
            doc = "The pinned blackwell-llm-docker source archive.",
        ),
        "recipes": attr.label(
            mandatory = True,
            allow_files = True,
            doc = "Our recipes, installed below /opt/vllm-image/recipes.",
        ),
        "hardware": attr.label(
            mandatory = True,
            allow_files = True,
            doc = "Our hardware profiles, merged into the upstream hardware directory.",
        ),
        "output": attr.string(mandatory = True),
        "_python": attr.label(
            default = Label("//platforms:x86_64_python"),
            executable = True,
            cfg = "exec",
        ),
        "_python_runtime": attr.label(
            default = Label("@python_3_12//:files"),
            allow_files = True,
            cfg = "exec",
        ),
        "_action_lib": attr.label(
            default = Label("//bazel:action_lib.py"),
            allow_single_file = True,
        ),
        "_driver": attr.label(
            default = Label("//bazel:launcher_data_layer_action.py"),
            allow_single_file = True,
        ),
    },
)
