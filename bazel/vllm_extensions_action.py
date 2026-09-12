#!/usr/bin/env python3
"""Compile vLLM native extensions into a reproducible archive."""

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from action_lib import (  # noqa: E402
    configure_compiler_cache,
    configure_cargo_vendor,
    extract,
    extract_durable,
    extract_wheel_script,
    install_wheels,
    materialize_absolute_symlinks,
    materialize_vllm_cmake_sources,
    rewrite_sysconfig,
    run,
    work_root,
    write_tar,
)


def path_from_execroot(value: str) -> Path:
    return Path.cwd() / value


def ensure_cuda_lib64(cuda: Path) -> None:
    lib64 = cuda / "lib64"
    if not lib64.exists() and not lib64.is_symlink():
        lib64.symlink_to("lib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--src-tar", required=True)
    parser.add_argument("--cuda-tar", required=True)
    parser.add_argument("--nccl-tar")
    parser.add_argument("--rustc", required=True)
    parser.add_argument("--cargo", required=True)
    parser.add_argument("--cargo-vendor-tar", required=True)
    parser.add_argument("--ccache-tar", required=True)
    parser.add_argument("--max-jobs", required=True, type=int)
    parser.add_argument("--cuda-architecture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-stats-output", required=True)
    parser.add_argument("--host-wheel", action="append", default=[])
    parser.add_argument("--cmake-source", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work = work_root(args.work_dir)
    python = path_from_execroot(args.python)
    source = work / "src"
    nccl = work / "nccl"
    site = work / "site"
    extensions = work / "extensions"
    output = path_from_execroot(args.output)
    cache_stats_output = path_from_execroot(args.cache_stats_output)

    extract(path_from_execroot(args.src_tar), source)
    cmake_sources = dict(item.split("=", 1) for item in args.cmake_source)
    cmake_env, cmake_source_args = materialize_vllm_cmake_sources(
        {name: path_from_execroot(path) for name, path in cmake_sources.items()}, work
    )
    cuda = extract_durable(path_from_execroot(args.cuda_tar), "cuda")
    ensure_cuda_lib64(cuda)
    if args.nccl_tar:
        extract(path_from_execroot(args.nccl_tar), nccl)

    sysconfig_dir = rewrite_sysconfig(python, work / "python-sysconfig")
    install_wheels(python, site, [path_from_execroot(wheel) for wheel in args.host_wheel])
    for wheel in args.host_wheel:
        wheel_path = path_from_execroot(wheel)
        if wheel_path.name.startswith("ninja-"):
            extract_wheel_script(wheel_path, "ninja", site / "bin" / "ninja")
            break

    env = os.environ.copy()
    env.update(cmake_env)
    env.update(
        {
            "CUDA_HOME": str(cuda),
            "CUDA_PATH": str(cuda),
            "CC": "gcc",
            "CXX": "g++",
            "NCCL_ROOT": str(nccl),
            "NCCL_INCLUDE_DIR": str(nccl / "include"),
            "NCCL_LIB_DIR": str(nccl / "lib"),
            "LD_LIBRARY_PATH": os.pathsep.join(
                part for part in (str(nccl / "lib"), str(cuda / "lib"), env.get("LD_LIBRARY_PATH")) if part
            ),
            "MAX_JOBS": str(args.max_jobs),
            "CMAKE_BUILD_PARALLEL_LEVEL": str(args.max_jobs),
            "PYTHONPATH": os.pathsep.join(
                part for part in (str(sysconfig_dir), str(site), env.get("PYTHONPATH")) if part
            ),
            "RUSTC": str(path_from_execroot(args.rustc)),
        }
    )
    env["PATH"] = os.pathsep.join(
        part
        for part in (
            str(cuda / "bin"),
            str(site / "cmake" / "data" / "bin"),
            str(site / "ninja" / "data" / "bin"),
            str(site / "bin"),
            str(path_from_execroot(args.rustc).parent),
            str(path_from_execroot(args.cargo).parent),
            env.get("PATH"),
        )
        if part
    )
    env["CMAKE_ARGS"] = " ".join(
        part for part in (env.get("CMAKE_ARGS"), "-DPython3_EXECUTABLE=" + str(python)) if part
    )

    ccache_launcher = configure_compiler_cache(work, path_from_execroot(args.ccache_tar), env)
    # This action owns the log, so its counters cannot include another build.
    env["CCACHE_STATSLOG"] = str(work / "ccache.statslog")
    # The declared vendor archive makes Cargo's complete source set available
    # offline. Mutable registry and target state stay action-local.
    cargo_home = configure_cargo_vendor(path_from_execroot(args.cargo_vendor_tar), work)
    cargo_target = work / "cargo" / "target"
    cargo_home.mkdir(parents=True, exist_ok=True)
    cargo_target.mkdir(parents=True, exist_ok=True)
    env["CARGO_HOME"] = str(cargo_home)
    env["CARGO_TARGET_DIR"] = str(cargo_target)
    env["CARGO_NET_OFFLINE"] = "true"

    run([str(python), "tools/build_rust.py", "--release"], cwd=source, env=env)
    configure = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(work / "build"),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        "-DCMAKE_CUDA_ARCHITECTURES=" + args.cuda_architecture,
        "-DCMAKE_CUDA_COMPILER=" + str(cuda / "bin" / "nvcc"),
        "-DCMAKE_INSTALL_PREFIX=" + str(extensions),
        "-DVLLM_TARGET_DEVICE=cuda",
        "-DVLLM_PYTHON_EXECUTABLE=" + str(python),
        "-DVLLM_PYTHON_PATH=" + env["PYTHONPATH"],
        "-DPython3_EXECUTABLE=" + str(python),
        "-DPython_EXECUTABLE=" + str(python),
        "-DRust_COMPILER=" + env["RUSTC"],
        "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
        *cmake_source_args,
    ]
    if ccache_launcher:
        configure.extend(
            [
                "-DCMAKE_C_COMPILER_LAUNCHER=" + str(ccache_launcher),
                "-DCMAKE_CXX_COMPILER_LAUNCHER=" + str(ccache_launcher),
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=" + str(ccache_launcher),
            ]
        )
    run(configure, env=env)
    run(
        ["cmake", "--build", str(work / "build"), "--target", "install", "-j", str(args.max_jobs)],
        env=env,
    )

    extension_vllm = extensions / "vllm"
    extension_vllm.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "vllm" / "vllm-rs", extension_vllm)
    for rust_extension in (source / "vllm").glob("_rust_*.so"):
        shutil.copy2(rust_extension, extension_vllm)
    materialize_absolute_symlinks(extensions)
    write_tar(output, extensions)
    statslog = Path(env["CCACHE_STATSLOG"])
    if not statslog.is_file():
        raise RuntimeError("vLLM native compilation did not produce ccache statistics")
    shutil.copyfile(statslog, cache_stats_output)


if __name__ == "__main__":
    main()
