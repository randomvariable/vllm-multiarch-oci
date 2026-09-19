#!/usr/bin/env python3
"""Fail before vLLM's long native compile when its build inputs are invalid."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from action_lib import (  # noqa: E402
    configure_compiler_cache,
    configure_compiler_sysroot,
    configure_cargo_vendor,
    extract,
    extract_durable,
    extract_wheel_script,
    install_wheels,
    materialize_vllm_cmake_sources,
    rewrite_sysconfig,
    run,
    work_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--src-tar", required=True)
    parser.add_argument("--source-identity", required=True)
    parser.add_argument("--cuda-tar", required=True)
    parser.add_argument("--nccl-tar", required=True)
    parser.add_argument("--rustc", required=True)
    parser.add_argument("--cargo", required=True)
    parser.add_argument("--cargo-vendor-tar", required=True)
    parser.add_argument("--ccache-tar", required=True)
    parser.add_argument("--gcc-sysroot-tar", required=True)
    parser.add_argument("--target-cpu", required=True)
    parser.add_argument("--cuda-architecture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--host-wheel", action="append", default=[])
    parser.add_argument("--cmake-source", action="append", default=[])
    return parser.parse_args()


def path_from_execroot(value: str) -> Path:
    return Path.cwd() / value


def ensure_cuda_lib64(cuda: Path) -> None:
    lib64 = cuda / "lib64"
    if not lib64.exists() and not lib64.is_symlink():
        lib64.symlink_to("lib")


def main() -> None:
    args = parse_args()
    work = work_root(args.work_dir)
    source = work / "src"
    site = work / "site"
    nccl = work / "nccl"
    build = work / "build"
    output = path_from_execroot(args.output)
    python = path_from_execroot(args.python)

    extract(path_from_execroot(args.src_tar), source)
    cmake_sources = dict(item.split("=", 1) for item in args.cmake_source)
    cmake_env, cmake_source_args = materialize_vllm_cmake_sources(
        {name: path_from_execroot(path) for name, path in cmake_sources.items()}, work, source
    )
    cuda = extract_durable(path_from_execroot(args.cuda_tar), "cuda")
    ensure_cuda_lib64(cuda)
    extract(path_from_execroot(args.nccl_tar), nccl)
    sysconfig_dir = rewrite_sysconfig(python, work / "python-sysconfig")
    wheels = [path_from_execroot(wheel) for wheel in args.host_wheel]
    install_wheels(python, site, wheels)
    for wheel in wheels:
        if wheel.name.startswith("ninja-"):
            extract_wheel_script(wheel, "ninja", site / "bin" / "ninja")
            break

    env = os.environ.copy()
    env.update(cmake_env)
    env.update(
        {
            "CUDA_HOME": str(cuda),
            "CUDA_PATH": str(cuda),
            "CARGO_NET_OFFLINE": "true",
            "NCCL_ROOT": str(nccl),
            "NCCL_INCLUDE_DIR": str(nccl / "include"),
            "NCCL_LIB_DIR": str(nccl / "lib"),
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
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        part for part in (str(nccl / "lib"), str(cuda / "lib"), env.get("LD_LIBRARY_PATH")) if part
    )
    env["CMAKE_ARGS"] = " ".join(
        part for part in (env.get("CMAKE_ARGS"), "-DPython3_EXECUTABLE=" + str(python)) if part
    )
    configure_compiler_sysroot(
        path_from_execroot(args.gcc_sysroot_tar), work, env, target_cpu=args.target_cpu
    )
    launcher = configure_compiler_cache(
        work, path_from_execroot(args.ccache_tar), env, target_cpu=args.target_cpu
    )
    cargo_home = configure_cargo_vendor(path_from_execroot(args.cargo_vendor_tar), work)
    env["CARGO_HOME"] = str(cargo_home)

    include_paths = json.loads(
        subprocess.check_output(
            [
                python,
                "-c",
                "import json, sysconfig, torch; "
                "from torch.utils.cpp_extension import include_paths; "
                "print(json.dumps([sysconfig.get_path('include'), *include_paths()]))",
            ],
            env=env,
            text=True,
        )
    )
    includes = [flag for path in include_paths for flag in ("-I", path)]
    syntax_object = work / "spinloop.o"
    run(
        [
            launcher,
            "g++",
            "-std=c++20",
            "-fPIC",
            "-c",
            source / "csrc" / "spinloop.cpp",
            *includes,
            "-o",
            syntax_object,
        ],
        env=env,
    )

    nccl_probe = work / "nccl-probe.cpp"
    nccl_probe.write_text("#include <nccl.h>\nint main() { return ncclSuccess == 0 ? 0 : 1; }\n")
    run(
        [
            launcher,
            "g++",
            nccl_probe,
            "-I",
            nccl / "include",
            "-I",
            cuda / "include",
            "-L",
            nccl / "lib",
            "-L",
            cuda / "lib",
            "-lnccl",
            "-Wl,-rpath," + str(nccl / "lib"),
            "-Wl,-rpath," + str(cuda / "lib"),
            "-o",
            work / "nccl-probe",
        ],
        env=env,
    )
    run([str(work / "nccl-probe")], env=env)

    run(
        [
            path_from_execroot(args.cargo),
            "metadata",
            "--offline",
            "--locked",
            "--no-deps",
            "--manifest-path",
            source / "rust" / "src" / "cmd" / "Cargo.toml",
        ],
        cwd=source,
        env=env,
    )
    run(
        [
            "cmake",
            "-S",
            source,
            "-B",
            build,
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            "-DCMAKE_CUDA_ARCHITECTURES=" + args.cuda_architecture,
            "-DCMAKE_CUDA_COMPILER=" + str(cuda / "bin" / "nvcc"),
            "-DVLLM_TARGET_DEVICE=cuda",
            "-DVLLM_PYTHON_EXECUTABLE=" + str(python),
            "-DVLLM_PYTHON_PATH=" + env["PYTHONPATH"],
            "-DPython3_EXECUTABLE=" + str(python),
            "-DPython_EXECUTABLE=" + str(python),
            "-DRust_COMPILER=" + env["RUSTC"],
            "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
            *cmake_source_args,
            "-DCMAKE_C_COMPILER_LAUNCHER=" + str(launcher),
            "-DCMAKE_CXX_COMPILER_LAUNCHER=" + str(launcher),
            "-DCMAKE_CUDA_COMPILER_LAUNCHER=" + str(launcher),
        ],
        env=env,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "cuda_architecture": args.cuda_architecture,
                "native_source": "csrc/spinloop.cpp",
                "source_identity": json.loads(path_from_execroot(args.source_identity).read_text()),
            },
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
