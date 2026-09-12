"""C++ toolchain configuration for the NativeLink ARM64 worker."""

load("@rules_cc//cc:cc_toolchain_config_lib.bzl", "tool_path")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/toolchains:cc_toolchain_config_info.bzl", "CcToolchainConfigInfo")

_TOOL_PATHS = {
    "ar": "/usr/bin/aarch64-linux-gnu-ar",
    "cpp": "/usr/bin/aarch64-linux-gnu-cpp",
    "gcc": "/usr/bin/aarch64-linux-gnu-gcc",
    "gcov": "/usr/bin/aarch64-linux-gnu-gcov",
    "ld": "/usr/bin/aarch64-linux-gnu-ld",
    "nm": "/usr/bin/aarch64-linux-gnu-nm",
    "objcopy": "/usr/bin/aarch64-linux-gnu-objcopy",
    "objdump": "/usr/bin/aarch64-linux-gnu-objdump",
    "strip": "/usr/bin/aarch64-linux-gnu-strip",
}

_BUILTIN_INCLUDE_DIRS = [
    "/usr/lib/gcc/aarch64-linux-gnu/15/include",
    "/usr/local/include",
    "/usr/include/aarch64-linux-gnu",
    "/usr/include",
]

def _remote_linux_aarch64_cc_toolchain_config_impl(ctx):
    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = "remote-linux-aarch64-gcc-15",
        host_system_name = "remote-linux-aarch64",
        target_system_name = "remote-linux-aarch64",
        target_cpu = "aarch64",
        target_libc = "glibc-2.43",
        compiler = "gcc",
        abi_version = "aarch64",
        abi_libc_version = "glibc-2.43",
        tool_paths = [
            tool_path(name = name, path = path)
            for name, path in _TOOL_PATHS.items()
        ],
        cxx_builtin_include_directories = _BUILTIN_INCLUDE_DIRS,
    )

remote_linux_aarch64_cc_toolchain_config = rule(
    implementation = _remote_linux_aarch64_cc_toolchain_config_impl,
    provides = [CcToolchainConfigInfo],
)
