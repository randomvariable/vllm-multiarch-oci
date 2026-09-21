# SPDX-License-Identifier: Apache-2.0
"""Name the non-root accounts the image carries in /etc/passwd.

The base image ends its passwd file at the `ubuntu` account, so a pod that pins
its own runAsUser has no entry for that uid. This rule derives complete passwd
and group files from the pinned base layout and appends the declared accounts.
"""

def _accounts_layer_impl(ctx):
    output = ctx.actions.declare_file(ctx.attr.output)
    bases = ctx.files.base
    if len(bases) != 1:
        fail("%s must resolve to exactly one base layout" % ctx.attr.base.label)

    args = ctx.actions.args()
    args.add(ctx.file._driver.path)
    args.add("--work-dir", output.path + ".work")
    args.add("--base", bases[0].path)
    args.add("--output", output.path)
    for account in ctx.attr.accounts:
        args.add("--account", account)
    for group in ctx.attr.groups:
        args.add("--group", group)

    ctx.actions.run(
        executable = ctx.executable._python,
        inputs = depset(
            direct = [
                ctx.executable._python,
                ctx.file._action_lib,
                ctx.file._driver,
            ] + bases + ctx.files._python_runtime,
        ),
        outputs = [output],
        arguments = [args],
        mnemonic = "AssembleAccountsLayer",
        progress_message = "Naming non-root accounts for %{label}",
        use_default_shell_env = False,
    )
    return [DefaultInfo(files = depset([output]))]

accounts_layer = rule(
    implementation = _accounts_layer_impl,
    # The action only reads the base layout, so it runs on the host like the
    # other layer assemblers instead of on an ARM64 worker.
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    attrs = {
        "base": attr.label(mandatory = True, allow_files = True),
        "accounts": attr.string_list(
            mandatory = True,
            doc = "name:uid:gid:gecos:home:shell entries to append to /etc/passwd",
        ),
        "groups": attr.string_list(
            default = [],
            doc = "name:gid: entries to append to /etc/group",
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
            default = Label("//bazel:accounts_layer_action.py"),
            allow_single_file = True,
        ),
    },
)
