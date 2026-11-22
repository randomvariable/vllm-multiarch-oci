# SPDX-License-Identifier: Apache-2.0
"""Select the same platform transition as the OCI index for contract inputs."""

def _platform_impl(_, attr):
    return {"//command_line_option:platforms": str(attr.image_platform.label)}

_platform = transition(implementation = _platform_impl, inputs = [], outputs = ["//command_line_option:platforms"])

def _image_impl(ctx):
    return [DefaultInfo(files = ctx.attr.image[0][DefaultInfo].files)]

contract_image = rule(
    implementation = _image_impl,
    attrs = {
        "_allowlist_function_transition": attr.label(default = "@bazel_tools//tools/allowlists/function_transition_allowlist"),
        "image": attr.label(cfg = _platform, mandatory = True),
        "image_platform": attr.label(mandatory = True, providers = [platform_common.PlatformInfo]),
    },
)
