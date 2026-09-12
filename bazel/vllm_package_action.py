#!/usr/bin/env python3
"""Package vLLM from Bazel-built native extensions."""

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from action_lib import extract, install_wheels, run, single_wheel, work_root  # noqa: E402


def path_from_execroot(value: str) -> Path:
    return Path.cwd() / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--src-tar", required=True)
    parser.add_argument("--extensions-tar", required=True)
    parser.add_argument("--cuda-tar", required=True)
    parser.add_argument("--nccl-tar")
    parser.add_argument("--build-script", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--host-wheel", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work = work_root(args.work_dir)
    python = path_from_execroot(args.python)
    source = work / "src"
    extensions = work / "extensions"
    cuda = work / "cuda"
    nccl = work / "nccl"
    site = work / "site"
    wheels = work / "wheels"

    extract(path_from_execroot(args.src_tar), source)
    extract(path_from_execroot(args.extensions_tar), extensions)
    extract(path_from_execroot(args.cuda_tar), cuda)
    if args.nccl_tar:
        extract(path_from_execroot(args.nccl_tar), nccl)
    install_wheels(python, site, [path_from_execroot(wheel) for wheel in args.host_wheel])

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                part for part in (str(site), env.get("PYTHONPATH")) if part
            ),
            "LD_LIBRARY_PATH": os.pathsep.join(
                part for part in (str(nccl / "lib"), str(cuda / "lib"), env.get("LD_LIBRARY_PATH")) if part
            ),
            "VLLM_USE_PRECOMPILED": "1",
            "VLLM_PRECOMPILED_EXTENSION_DIR": str(extensions),
        }
    )
    shutil.copytree(extensions / "vllm", source / "vllm", dirs_exist_ok=True)
    wheels.mkdir(parents=True, exist_ok=True)
    run(
        [str(python), str(Path.cwd() / args.build_script), str(python), str(wheels)],
        cwd=source,
        env=env,
    )
    shutil.copy2(single_wheel(wheels), path_from_execroot(args.output))
    path_from_execroot(args.output).chmod(0o644)


if __name__ == "__main__":
    main()
