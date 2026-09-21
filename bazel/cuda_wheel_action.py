#!/usr/bin/env python3
"""Build a CUDA Python wheel from Bazel-declared action inputs."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from action_lib import (
    configure_compiler_cache,
    configure_compiler_sysroot,
    configure_cargo_vendor,
    configure_reproducible_compilation,
    create_nvcc_wrapper,
    extract,
    extract_durable,
    extract_wheel_script,
    install_wheels,
    materialize_console_scripts,
    rewrite_sysconfig,
    run,
    single_wheel,
    target_abi,
    wheel_scripts,
    work_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--src-tar", required=True)
    parser.add_argument("--cuda-tar", required=True)
    parser.add_argument("--nccl-tar")
    parser.add_argument("--build-script", required=True)
    parser.add_argument("--ccache-tar", required=True)
    parser.add_argument("--gcc-sysroot-tar", required=True)
    parser.add_argument("--target-cpu", required=True)
    parser.add_argument("--max-jobs", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rustc")
    parser.add_argument("--cargo")
    parser.add_argument("--cargo-vendor-tar")
    parser.add_argument("--host-wheel", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work = work_root(args.work_dir)
    python = work_root(args.python)
    source_tar = work_root(args.src_tar)
    cuda_tar = work_root(args.cuda_tar)
    build_script = work_root(args.build_script)
    ccache_tar = work_root(args.ccache_tar)
    gcc_sysroot_tar = work_root(args.gcc_sysroot_tar)
    output = work_root(args.output)
    wheels = [work_root(path) for path in args.host_wheel]

    source = work / "src"
    nccl = work / "nccl"
    site = work / "site"
    wheels_dir = work / "wheels"
    home = work / "home"
    for directory in (source, nccl, site / "bin", wheels_dir, home):
        directory.mkdir(parents=True, exist_ok=True)

    extract(source_tar, source)
    cuda = extract_durable(cuda_tar, "cuda")
    ensure_cuda_lib64(cuda)
    if args.nccl_tar:
        extract(work_root(args.nccl_tar), nccl)

    install_wheels(python, site, wheels)
    materialize_console_scripts(python, site, site / "bin")
    for wheel in wheels:
        for script in wheel_scripts(wheel, site / "bin"):
            script.chmod(0o755)
        if wheel.name.startswith("ninja-"):
            extract_wheel_script(wheel, "ninja", site / "bin" / "ninja")

    sysconfig = rewrite_sysconfig(python, work / "python-sysconfig")
    env = os.environ.copy()
    # Bazel supplies target-specific variables in the action environment.
    # Preserve them after establishing the hermetic build defaults below.
    target_env = {
        name: env[name]
        for name in (
            "BUILD_CAFFE2",
            "BUILD_TEST",
            "CMAKE_CUDA_ARCHITECTURES",
            "PYTORCH_BUILD_NUMBER",
            "PYTORCH_BUILD_VERSION",
            "TORCH_CUDA_ARCH_LIST",
            "USE_CUDA",
            "USE_CUDNN",
            "USE_CUFILE",
            "USE_CUSPARSELT",
            "USE_DISTRIBUTED",
            "USE_NCCL",
            "USE_SYSTEM_NCCL",
        )
        if name in env
    }
    env.update(
        {
            "CUDA_HOME": str(cuda),
            "CUDA_PATH": str(cuda),
            "HOME": str(home),
            "NCCL_ROOT": str(nccl),
            "NCCL_INCLUDE_DIR": str(nccl / "include"),
            "NCCL_LIB_DIR": str(nccl / "lib"),
            "MAX_JOBS": str(args.max_jobs),
            "CMAKE_BUILD_PARALLEL_LEVEL": str(args.max_jobs),
            "PYTHONPATH": _prepend_path(env.get("PYTHONPATH"), sysconfig, site),
        }
    )
    env.update(target_env)
    env["PATH"] = _prepend_path(env.get("PATH"), cuda / "bin", site / "cmake/data/bin", site / "bin")
    env["LD_LIBRARY_PATH"] = _prepend_path(
        env.get("LD_LIBRARY_PATH"), nccl / "lib", cuda / "lib"
    )
    env["CMAKE_ARGS"] = _append_argument(
        env.get("CMAKE_ARGS"), "-DPython3_EXECUTABLE=" + str(python)
    )
    configure_compiler_sysroot(gcc_sysroot_tar, work, env, target_cpu=args.target_cpu)
    configure_compiler_cache(work, ccache_tar, env, target_cpu=args.target_cpu)
    reproducible = configure_reproducible_compilation(work, source, env)
    nvcc = create_nvcc_wrapper(work, cuda / "bin" / "nvcc")
    env["CMAKE_ARGS"] = _append_argument(
        env.get("CMAKE_ARGS"), "-DCMAKE_CUDA_COMPILER=" + str(nvcc)
    )
    env["NVCC_FLAGS"] = _append_argument(
        env.get("NVCC_FLAGS"), reproducible["cuda"]
    )
    if args.cargo_vendor_tar:
        if not args.rustc or not args.cargo:
            raise ValueError("Rust source wheel requires --rustc and --cargo")
        cargo_home = configure_cargo_vendor(work_root(args.cargo_vendor_tar), work)
        cargo_target = work / "cargo" / "target"
        cargo_home.mkdir(parents=True, exist_ok=True)
        cargo_target.mkdir(parents=True, exist_ok=True)
        env.update(
            {
                "CARGO_HOME": str(cargo_home),
                "CARGO_NET_OFFLINE": "true",
                "CARGO_TARGET_DIR": str(cargo_target),
                "RUSTC": str(work_root(args.rustc)),
            }
        )
        env["PATH"] = _prepend_path(
            env.get("PATH"), work_root(args.rustc).parent, work_root(args.cargo).parent
        )
        # Cargo uses the bare `cc` linker name by default. Point it at the
        # declared compiler wrapper rather than requiring a host C compiler.
        rust_target = target_abi(args.target_cpu)["rust_triple"].upper().replace("-", "_")
        env["CC_" + rust_target] = env["CC"]
        env["CXX_" + rust_target] = env["CXX"]
        env["CARGO_TARGET_" + rust_target + "_LINKER"] = env["CC"]

    run([python, build_script, python, wheels_dir], cwd=source, env=env)
    wheel = single_wheel(wheels_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(wheel, output)
    output.chmod(0o644)

def _prepend_path(existing: str | None, *items: Path) -> str:
    return ":".join([*(str(item) for item in items), *([existing] if existing else [])])


def _append_argument(existing: str | None, value: str) -> str:
    return " ".join(part for part in (existing, value) if part)


def ensure_cuda_lib64(cuda: Path) -> None:
    lib64 = cuda / "lib64"
    if not lib64.exists() and not lib64.is_symlink():
        lib64.symlink_to("lib")


if __name__ == "__main__":
    main()
