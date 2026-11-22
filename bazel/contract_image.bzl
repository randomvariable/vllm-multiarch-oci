# SPDX-License-Identifier: Apache-2.0
"""Select image and execution platforms independently for image contracts."""

load("@container_structure_test//bazel:container_structure_test.bzl", _cst = "lib")

_X86_64_EXEC = ["@platforms//cpu:x86_64", "@platforms//os:linux"]
_CST_TOOLCHAINS = [
    "@aspect_bazel_lib//lib:jq_toolchain_type",
    "@bazel_tools//tools/sh:toolchain_type",
    "@container_structure_test//bazel:structure_test_toolchain_type",
]

def _x86_contract_impl(ctx):
    # Bazel's implicit test group matches the invocation's target platform,
    # not the execution platform used to resolve the launcher's binaries.
    return [_cst.implementation(ctx), testing.ExecutionInfo(exec_group = "contract_runner")]

x86_contract_test = rule(
    implementation = _x86_contract_impl,
    attrs = _cst.attrs,
    test = True,
    exec_compatible_with = _X86_64_EXEC,
    exec_groups = {
        "contract_runner": exec_group(
            exec_compatible_with = _X86_64_EXEC,
            toolchains = _CST_TOOLCHAINS,
        ),
    },
    toolchains = _CST_TOOLCHAINS,
)

def _platform_impl(_, attr):
    # A label attr is a Label in a transition; str() matches rules_oci's index
    # transition exactly, so both reach the same configuration and outputs.
    return {"//command_line_option:platforms": str(attr.image_platform)}

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
