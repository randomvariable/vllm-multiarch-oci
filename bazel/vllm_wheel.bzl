# SPDX-License-Identifier: Apache-2.0
"""Compile vLLM CUDA extensions separately from pure-Python wheel packaging."""

load("//bazel:compiler_cache.bzl", "GCC_SYSROOT_ATTR", "MEMORY_PER_JOB_ATTR", "compile_jobs")
load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")

def _vllm_wheel_impl(ctx):
    wheel = ctx.actions.declare_file(ctx.attr.output)
    extensions = ctx.actions.declare_file(ctx.label.name + ".extensions.tar")
    cache_stats = ctx.actions.declare_file(ctx.label.name + ".ccache.statslog")
    rust_toolchain = ctx.toolchains["@rules_rust//rust:toolchain"]
    wheel_inputs = ctx.files.host_wheels + ctx.files.deps
    cmake_source_files = []

    compile_args = ctx.actions.args()
    compile_args.add("--work-dir", ctx.label.name + ".compile")
    compile_args.add("--python", ctx.file.python.path)
    compile_args.add("--src-tar", ctx.file.src.path)
    compile_args.add("--cuda-tar", ctx.file.cuda.path)
    if ctx.file.nccl:
        compile_args.add("--nccl-tar", ctx.file.nccl.path)
    compile_args.add("--rustc", rust_toolchain.rustc.path)
    compile_args.add("--cargo", rust_toolchain.cargo.path)
    compile_args.add("--cargo-vendor-tar", ctx.file.cargo_vendor.path)
    compile_args.add("--ccache-tar", ctx.file.compiler_cache.path)
    compile_args.add("--gcc-sysroot-tar", ctx.file.compiler_sysroot.path)
    compile_args.add("--target-cpu", ctx.attr.target_cpu)
    compile_args.add("--max-jobs", compile_jobs(ctx.attr))
    compile_args.add("--cuda-architecture", ctx.attr.cuda_architecture)
    compile_args.add("--output", extensions.path)
    compile_args.add("--cache-stats-output", cache_stats.path)
    for name in sorted(ctx.attr.cmake_sources):
        source_files = ctx.attr.cmake_sources[name][DefaultInfo].files.to_list()
        if len(source_files) != 1:
            fail("cmake_sources[%s] must provide exactly one archive" % name)
        cmake_source_files.append(source_files[0])
        compile_args.add("--cmake-source", name + "=" + source_files[0].path)
    compile_args.add_all(
        [wheel.path for wheel in wheel_inputs],
        before_each = "--host-wheel",
    )

    compile_inputs = [
        ctx.file._compile_driver,
        ctx.file._action_lib,
        ctx.file.src,
        ctx.file.cuda,
        ctx.executable._python_launcher,
        ctx.file.compiler_cache,
        ctx.file.compiler_sysroot,
        ctx.file.cargo_vendor,
    ] + ctx.files.python_runtime + ctx.files.python_headers + wheel_inputs + ctx.files.nccl + cmake_source_files + [
        rust_toolchain.rustc,
        rust_toolchain.cargo,
    ]
    ctx.actions.run(
        executable = ctx.executable._python_launcher,
        arguments = [ctx.file._compile_driver.path, compile_args],
        inputs = depset(direct = compile_inputs, transitive = [rust_toolchain.all_files]),
        outputs = [extensions, cache_stats],
        mnemonic = "CompileVLLMExtensions",
        progress_message = "Compiling %s CUDA extensions" % ctx.label.name,
        execution_requirements = {
            "cpu": str(ctx.attr.cpu),
            "ISA": ctx.attr.target_cpu,
            "memory": str(ctx.attr.memory),
        },
        env = dict(ctx.attr.env, **_cache_env(ctx)),
        use_default_shell_env = False,
    )

    package_args = ctx.actions.args()
    package_args.add("--work-dir", ctx.label.name + ".package")
    package_args.add("--python", ctx.file.python.path)
    package_args.add("--src-tar", ctx.file.src.path)
    package_args.add("--extensions-tar", extensions.path)
    package_args.add("--cuda-tar", ctx.file.cuda.path)
    if ctx.file.nccl:
        package_args.add("--nccl-tar", ctx.file.nccl.path)
    package_args.add("--gcc-sysroot-tar", ctx.file.compiler_sysroot.path)
    package_args.add("--target-cpu", ctx.attr.target_cpu)
    package_args.add("--build-script", ctx.file.build.path)
    package_args.add("--output", wheel.path)
    package_args.add_all(
        [item.path for item in wheel_inputs],
        before_each = "--host-wheel",
    )

    package_inputs = [
        ctx.file._package_driver,
        ctx.file._action_lib,
        ctx.file.src,
        ctx.executable._python_launcher,
        ctx.file.cuda,
        ctx.file.compiler_sysroot,
        extensions,
        ctx.file.build,
    ] + ctx.files.python_runtime + wheel_inputs + ctx.files.nccl
    ctx.actions.run(
        executable = ctx.executable._python_launcher,
        arguments = [ctx.file._package_driver.path, package_args],
        inputs = depset(direct = package_inputs),
        outputs = [wheel],
        mnemonic = "PackageVLLMWheel",
        progress_message = "Packaging %s Python wheel" % ctx.label.name,
        execution_requirements = {
            "cpu": str(ctx.attr.cpu),
            "ISA": ctx.attr.target_cpu,
            "memory": str(ctx.attr.memory),
        },
        env = ctx.attr.env,
        use_default_shell_env = False,
    )
    return [
        DefaultInfo(files = depset([wheel])),
        OutputGroupInfo(
            native_extensions = depset([extensions]),
            compiler_cache_stats = depset([cache_stats]),
        ),
    ]

vllm_wheel = rule(
    implementation = _vllm_wheel_impl,
    attrs = {
        "src": attr.label(mandatory = True, allow_single_file = True),
        "cuda": attr.label(mandatory = True, allow_single_file = True),
        "python": attr.label(mandatory = True, allow_single_file = True),
        "_python_launcher": attr.label(
            default = Label("//platforms:action_python"),
            executable = True,
            cfg = "target",
        ),
        # Keep the relocatable interpreter's standard library in every remote
        # action. The binary's fixed /install prefix otherwise cannot start.
        "python_runtime": attr.label(
            default = Label("@python_3_12//:files"),
            allow_files = True,
        ),
        "python_headers": attr.label(
            default = Label("@python_3_12//:includes"),
            allow_files = True,
        ),
        "build": attr.label(mandatory = True, allow_single_file = True),
        "compiler_cache": attr.label(
            default = Label("//platforms:compiler_cache"),
            allow_single_file = True,
        ),
        "compiler_sysroot": GCC_SYSROOT_ATTR,
        "target_cpu": attr.string(mandatory = True, values = ["aarch64", "x86_64"]),
        "cargo_vendor": attr.label(mandatory = True, allow_single_file = True),
        "output": attr.string(mandatory = True),
        "host_wheels": attr.label_list(allow_files = True),
        "deps": attr.label_list(allow_files = True),
        # Exact source archives for vLLM's CMake FetchContent dependencies.
        # Each key is passed to the driver as its supported CMake override.
        "cmake_sources": attr.string_keyed_label_dict(allow_files = True),
        "env": attr.string_dict(),
        "_cache_root": attr.label(default = Label("//platforms:vllmb12x_cache_root")),
        "nccl": attr.label(allow_single_file = True),
        "cuda_architecture": attr.string(mandatory = True),
        # Upper bound only. compile_jobs() reduces this to what the action's
        # CPU and memory reservation affords.
        "max_jobs": attr.int(default = 64),
        "cpu": attr.int(default = 20),
        "memory": attr.int(default = 65536),
        "memory_per_job": MEMORY_PER_JOB_ATTR,
        "_compile_driver": attr.label(
            default = Label("//bazel:vllm_extensions_action.py"),
            allow_single_file = True,
        ),
        "_package_driver": attr.label(
            default = Label("//bazel:vllm_package_action.py"),
            allow_single_file = True,
        ),
        "_action_lib": attr.label(
            default = Label("//bazel:action_lib.py"),
            allow_single_file = True,
        ),
    },
    toolchains = [
        "@bazel_tools//tools/cpp:toolchain_type",
        "@rules_rust//rust:toolchain",
    ],
)


def _vllm_preflight_impl(ctx):
    report = ctx.actions.declare_file(ctx.attr.output)
    rust_toolchain = ctx.toolchains["@rules_rust//rust:toolchain"]
    wheel_inputs = ctx.files.host_wheels + ctx.files.deps
    cmake_source_files = []

    args = ctx.actions.args()
    args.add("--work-dir", ctx.label.name + ".work")
    args.add("--python", ctx.file.python.path)
    args.add("--src-tar", ctx.file.src.path)
    args.add("--source-identity", ctx.file.source_identity.path)
    args.add("--cuda-tar", ctx.file.cuda.path)
    args.add("--nccl-tar", ctx.file.nccl.path)
    args.add("--rustc", rust_toolchain.rustc.path)
    args.add("--cargo", rust_toolchain.cargo.path)
    args.add("--cargo-vendor-tar", ctx.file.cargo_vendor.path)
    args.add("--ccache-tar", ctx.file.compiler_cache.path)
    args.add("--gcc-sysroot-tar", ctx.file.compiler_sysroot.path)
    args.add("--target-cpu", ctx.attr.target_cpu)
    args.add("--cuda-architecture", ctx.attr.cuda_architecture)
    args.add("--output", report.path)
    for name in sorted(ctx.attr.cmake_sources):
        source_files = ctx.attr.cmake_sources[name][DefaultInfo].files.to_list()
        if len(source_files) != 1:
            fail("cmake_sources[%s] must provide exactly one archive" % name)
        cmake_source_files.append(source_files[0])
        args.add("--cmake-source", name + "=" + source_files[0].path)
    args.add_all([wheel.path for wheel in wheel_inputs], before_each = "--host-wheel")

    inputs = [
        ctx.file._driver,
        ctx.file._action_lib,
        ctx.file.src,
        ctx.file.source_identity,
        ctx.file.cuda,
        ctx.file.nccl,
        ctx.executable._python_launcher,
        ctx.file.compiler_cache,
        ctx.file.compiler_sysroot,
        ctx.file.cargo_vendor,
    ] + ctx.files.python_runtime + ctx.files.python_headers + wheel_inputs + cmake_source_files + [
        rust_toolchain.rustc,
        rust_toolchain.cargo,
    ]
    ctx.actions.run(
        executable = ctx.executable._python_launcher,
        arguments = [ctx.file._driver.path, args],
        inputs = depset(direct = inputs, transitive = [rust_toolchain.all_files]),
        outputs = [report],
        mnemonic = "PreflightVLLMExtensions",
        progress_message = "Preflighting %s native extension build" % ctx.label.name,
        execution_requirements = {
            "cpu": str(ctx.attr.cpu),
            "memory": str(ctx.attr.memory),
        },
        env = dict(ctx.attr.env, **_cache_env(ctx)),
        use_default_shell_env = False,
    )
    return [DefaultInfo(files = depset([report]))]


vllm_preflight = rule(
    implementation = _vllm_preflight_impl,
    attrs = {
        "src": attr.label(mandatory = True, allow_single_file = True),
        "source_identity": attr.label(mandatory = True, allow_single_file = True),
        "cuda": attr.label(mandatory = True, allow_single_file = True),
        "python": attr.label(mandatory = True, allow_single_file = True),
        "_python_launcher": attr.label(
            default = Label("//platforms:action_python"),
            executable = True,
            cfg = "target",
        ),
        "python_runtime": attr.label(
            default = Label("@python_3_12//:files"),
            allow_files = True,
        ),
        "python_headers": attr.label(
            default = Label("@python_3_12//:includes"),
            allow_files = True,
        ),
        "compiler_cache": attr.label(
            default = Label("//platforms:compiler_cache"),
            allow_single_file = True,
        ),
        "compiler_sysroot": GCC_SYSROOT_ATTR,
        "target_cpu": attr.string(mandatory = True, values = ["aarch64", "x86_64"]),
        "cargo_vendor": attr.label(mandatory = True, allow_single_file = True),
        "output": attr.string(mandatory = True),
        "host_wheels": attr.label_list(allow_files = True),
        "deps": attr.label_list(allow_files = True),
        "cmake_sources": attr.string_keyed_label_dict(allow_files = True),
        "env": attr.string_dict(),
        "_cache_root": attr.label(default = Label("//platforms:vllmb12x_cache_root")),
        "nccl": attr.label(mandatory = True, allow_single_file = True),
        "cuda_architecture": attr.string(mandatory = True),
        "cpu": attr.int(default = 4),
        "memory": attr.int(default = 8192),
        "_driver": attr.label(
            default = Label("//bazel:vllm_preflight.py"),
            allow_single_file = True,
        ),
        "_action_lib": attr.label(
            default = Label("//bazel:action_lib.py"),
            allow_single_file = True,
        ),
    },
    toolchains = [
        "@bazel_tools//tools/cpp:toolchain_type",
        "@rules_rust//rust:toolchain",
    ],
)


def _cache_env(ctx):
    root = ctx.attr._cache_root[BuildSettingInfo].value
    return {} if not root else {
        "CCACHE_DIR": root + "/objects",
        "VLLMB12X_CACHE_ROOT": root,
    }
