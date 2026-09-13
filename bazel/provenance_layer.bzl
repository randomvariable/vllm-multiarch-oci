# SPDX-License-Identifier: Apache-2.0
"""Record the resolved source lock inside the image."""

def _provenance_layer_impl(ctx):
    output = ctx.actions.declare_file(ctx.attr.output)
    args = ctx.actions.args()
    args.add(ctx.file._driver.path)
    args.add("--work-dir", output.path + ".work")
    args.add("--manifest", ctx.file.manifest.path)
    args.add("--build-version", ctx.attr.build_version)
    args.add("--output", output.path)
    identities = []
    for name, identity in ctx.attr.source_identities.items():
        files = identity.files.to_list()
        if len(files) != 1:
            fail("%s must provide exactly one identity file" % identity.label)
        identities.append(files[0])
        args.add("--source-identity", "%s=%s" % (name, files[0].path))

    ctx.actions.run(
        executable = ctx.file._python,
        inputs = depset(direct = [
            ctx.file._python,
            ctx.file._action_lib,
            ctx.file._driver,
            ctx.file.manifest,
        ] + identities),
        outputs = [output],
        arguments = [args],
        mnemonic = "AssembleProvenanceLayer",
        progress_message = "Recording resolved source lock %{label}",
        use_default_shell_env = True,
    )
    return [DefaultInfo(files = depset([output]))]

provenance_layer = rule(
    implementation = _provenance_layer_impl,
    attrs = {
        "manifest": attr.label(mandatory = True, allow_single_file = True),
        "build_version": attr.string(mandatory = True),
        "source_identities": attr.string_keyed_label_dict(
            mandatory = True,
            allow_files = True,
        ),
        "output": attr.string(mandatory = True),
        "_python": attr.label(
            default = Label("@python_3_12//:python3"),
            allow_single_file = True,
        ),
        "_action_lib": attr.label(
            default = Label("//bazel:action_lib.py"),
            allow_single_file = True,
        ),
        "_driver": attr.label(
            default = Label("//bazel:provenance_layer_action.py"),
            allow_single_file = True,
        ),
    },
)
