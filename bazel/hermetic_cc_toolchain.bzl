"""A Bazel C++ toolchain assembled from the locked GCC package closure."""

load("@rules_cc//cc:cc_toolchain_config_lib.bzl", "env_entry", "env_set", "feature", "flag_group", "flag_set", "tool_path")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/toolchains:cc_toolchain_config_info.bzl", "CcToolchainConfigInfo")
load("@aspect_bazel_lib//lib:tar.bzl", "tar_lib")

_TAR_TOOLCHAIN_TYPE = tar_lib.toolchain_type
HermeticSysrootInfo = provider(fields = ["sysroot"])

_TOOLS = ["ar", "as", "cpp", "gcc", "ld", "nm", "objcopy", "objdump", "strip"]

_COMPILE_ACTIONS = [
    "c-compile",
    "c++-compile",
    "assemble",
    "preprocess-assemble",
    "linkstamp-compile",
    "c++-header-parsing",
    "c++-module-compile",
]

_LINK_ACTIONS = [
    "c++-link-executable",
    "c++-link-dynamic-library",
    "c++-link-nodeps-dynamic-library",
]


def _toolchain_files_impl(ctx):
    sysroot = ctx.actions.declare_directory("sysroot")
    wrapper_files = ctx.files.wrappers
    tar = ctx.toolchains[_TAR_TOOLCHAIN_TYPE].tarinfo.binary
    arguments = ctx.actions.args()
    arguments.add(tar.path)
    arguments.add(sysroot.path)
    arguments.add_all(ctx.files.archives)
    ctx.actions.run(
        executable = ctx.executable._extractor,
        arguments = [arguments],
        inputs = depset(ctx.files.archives + [tar, ctx.executable._extractor]),
        outputs = [sysroot],
        mnemonic = "MaterializeHermeticCCToolchain",
        progress_message = "Materializing hermetic GCC for %{label}",
    )

    return [
        DefaultInfo(files = depset([sysroot] + wrapper_files)),
        HermeticSysrootInfo(sysroot = sysroot),
    ]


hermetic_cc_toolchain_files = rule(
    implementation = _toolchain_files_impl,
    attrs = {
        "archives": attr.label(mandatory = True, allow_files = True),
        "wrappers": attr.label_list(mandatory = True, allow_files = True),
        "_extractor": attr.label(
            default = Label("//bazel:hermetic_cc_toolchain.sh"),
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),
    },
    toolchains = [_TAR_TOOLCHAIN_TYPE],
)


def _arm64_python_launcher_impl(ctx):
    launcher = ctx.actions.declare_file(ctx.attr.output)
    sysroot = ctx.attr.sysroot[HermeticSysrootInfo].sysroot
    qemu_root = ctx.attr.qemu_root[DefaultInfo].files.to_list()
    if len(qemu_root) != 1:
        fail("qemu_root must provide exactly one tree artifact")
    ctx.actions.expand_template(
        template = ctx.file._template,
        output = launcher,
        substitutions = {
            "{qemu_root}": qemu_root[0].path,
            "{sysroot}": sysroot.path,
            "{python}": ctx.file.python.path,
        },
        is_executable = True,
    )
    return [DefaultInfo(
        files = depset([launcher, ctx.file.python, sysroot, qemu_root[0]]),
        executable = launcher,
    )]


arm64_python_launcher = rule(
    implementation = _arm64_python_launcher_impl,
    executable = True,
    attrs = {
        "python": attr.label(mandatory = True, allow_single_file = True),
        "qemu_root": attr.label(mandatory = True, cfg = "exec"),
        "sysroot": attr.label(mandatory = True),
        "output": attr.string(mandatory = True),
        "_template": attr.label(
            default = Label("//bazel:arm64_python_launcher.sh.tpl"),
            allow_single_file = True,
        ),
    },
)

def _native_python_launcher_impl(ctx):
    launcher = ctx.actions.declare_file(ctx.attr.output)
    ctx.actions.expand_template(
        template = ctx.file._template,
        output = launcher,
        substitutions = {
            "{python}": ctx.file.python.path,
        },
        is_executable = True,
    )
    return [DefaultInfo(
        files = depset([launcher, ctx.file.python]),
        executable = launcher,
    )]


native_python_launcher = rule(
    implementation = _native_python_launcher_impl,
    executable = True,
    attrs = {
        "python": attr.label(mandatory = True, allow_single_file = True),
        "output": attr.string(mandatory = True),
        "_template": attr.label(
            default = Label("//bazel:native_python_launcher.sh.tpl"),
            allow_single_file = True,
        ),
    },
)


def _arm64_python_probe_impl(ctx):
    output = ctx.actions.declare_file(ctx.attr.output)
    ctx.actions.run(
        executable = ctx.executable.launcher,
        arguments = [
            "-c",
            ("from pathlib import Path; import subprocess, sys; " +
             "subprocess.check_call(['python3', '--version']); " +
             "Path(%r).write_text('ok\\n')") % output.path,
        ],
        inputs = [ctx.executable.child_launcher],
        outputs = [output],
        mnemonic = "ProbeArm64Python",
    )
    return [DefaultInfo(files = depset([output]))]


arm64_python_probe = rule(
    implementation = _arm64_python_probe_impl,
    attrs = {
        "launcher": attr.label(mandatory = True, executable = True, cfg = "target"),
        "child_launcher": attr.label(mandatory = True, executable = True, cfg = "target"),
        "output": attr.string(mandatory = True),
    },
)


def _materialize_tar_tree_impl(ctx):
    output = ctx.actions.declare_directory(ctx.attr.output)
    tar = ctx.toolchains[_TAR_TOOLCHAIN_TYPE].tarinfo.binary
    arguments = ctx.actions.args()
    arguments.add(tar.path)
    arguments.add(ctx.file.archive.path)
    arguments.add(output.path)
    ctx.actions.run(
        executable = ctx.executable._extractor,
        arguments = [arguments],
        inputs = depset([ctx.file.archive, tar, ctx.executable._extractor]),
        outputs = [output],
        mnemonic = "MaterializeTarTree",
        execution_requirements = {"ISA": "x86_64"},
    )
    return [DefaultInfo(files = depset([output]))]


materialize_tar_tree = rule(
    implementation = _materialize_tar_tree_impl,
    # These trees supply the x86-host QEMU launcher and toolchain wrappers.
    # The pinned tar tool resolves for the host, so dispatch extraction to the
    # native x86 worker instead of the ARM CUDA execution platform.
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    attrs = {
        "archive": attr.label(mandatory = True, allow_single_file = True, cfg = "exec"),
        "output": attr.string(mandatory = True),
        "_extractor": attr.label(
            default = Label("//bazel:materialize_tar_tree.sh"),
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),
    },
    toolchains = [_TAR_TOOLCHAIN_TYPE],
)


def _hermetic_cc_toolchain_config_impl(ctx):
    files = {file.basename: file for file in ctx.attr.files[DefaultInfo].files.to_list()}
    required = ["sysroot"] + list(_TOOLS) + ["g++"]
    missing = [name for name in required if name not in files]
    if missing:
        fail("hermetic toolchain files are missing: %s" % ", ".join(missing))

    sysroot = files["sysroot"].path

    tool_paths = [
        tool_path(name = name, path = name)
        for name in _TOOLS
    ] + [
        tool_path(name = "g++", path = "g++"),
        tool_path(name = "gcov", path = "gcc"),
        tool_path(name = "dwp", path = "gcc"),
        tool_path(name = "compat-ld", path = "ld"),
    ]

    sysroot_flags = feature(
        name = "hermetic_sysroot",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = _COMPILE_ACTIONS + _LINK_ACTIONS,
                # GCC normally resolves the symlinked execution root to the
                # worker-specific absolute path. Bazel's header checker owns
                # the declared logical path, so retain that spelling.
                flag_groups = [flag_group(flags = [
                    "--sysroot=" + sysroot,
                    "-no-canonical-prefixes",
                    "-nostdinc",
                    "-isystem", sysroot + "/usr/lib/gcc/" + ctx.attr.triple + "/15/include",
                    "-isystem", sysroot + "/usr/lib/gcc/" + ctx.attr.triple + "/15/include-fixed",
                    "-isystem", sysroot + "/usr/include/c++/15",
                    "-isystem", sysroot + "/usr/include/" + ctx.attr.triple + "/c++/15",
                    "-isystem", sysroot + "/usr/include/" + ctx.attr.triple,
                    "-isystem", sysroot + "/usr/include",
                ])],
            ),
        ],
    )

    tool_environment_entries = [
        env_entry(key = "HERMETIC_CC_SYSROOT", value = sysroot),
        env_entry(key = "HERMETIC_CC_TRIPLE", value = ctx.attr.triple),
    ]
    if ctx.attr.compiler_prefix:
        tool_environment_entries.append(
            env_entry(key = "HERMETIC_CC_PREFIX", value = ctx.attr.compiler_prefix),
        )
    tool_environment = feature(
        name = "hermetic_tool_environment",
        enabled = True,
        env_sets = [
            env_set(
                actions = _COMPILE_ACTIONS + _LINK_ACTIONS,
                env_entries = tool_environment_entries,
            ),
            env_set(
                actions = ["c++-link-static-library"],
                env_entries = tool_environment_entries,
            ),
        ],
    )
    cpu = ctx.attr.cpu
    triple = ctx.attr.triple
    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = "hermetic-linux-%s-gcc-15" % cpu,
        host_system_name = "hermetic-linux-%s" % cpu,
        target_system_name = "hermetic-linux-%s" % cpu,
        target_cpu = cpu,
        target_libc = "glibc-2.43",
        compiler = "gcc",
        abi_version = cpu,
        abi_libc_version = "glibc-2.43",
        tool_paths = tool_paths,
        features = [
            sysroot_flags,
            tool_environment,
            feature(name = "supports_pic", enabled = True),
        ],
        builtin_sysroot = sysroot,
        cxx_builtin_include_directories = [
            "%sysroot%/usr/lib/gcc/" + triple + "/15/include",
            "%sysroot%/usr/lib/gcc/" + triple + "/15/include-fixed",
            "%sysroot%/usr/include/c++/15",
            "%sysroot%/usr/include/" + triple + "/c++/15",
            "%sysroot%/usr/include/" + triple,
            "%sysroot%/usr/include",
        ],
    )


hermetic_linux_cc_toolchain_config = rule(
    implementation = _hermetic_cc_toolchain_config_impl,
    attrs = {
        "cpu": attr.string(mandatory = True, values = ["aarch64", "x86_64"]),
        "compiler_prefix": attr.string(mandatory = True),
        "files": attr.label(mandatory = True),
        "triple": attr.string(mandatory = True),
    },
    provides = [CcToolchainConfigInfo],
)
