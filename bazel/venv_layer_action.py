#!/usr/bin/env python3
"""Assemble the self-contained Python runtime OCI layer."""

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from action_lib import extract, install_wheels, work_root, write_tar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--nccl-tar", required=True)
    parser.add_argument("--python-launcher", required=True)
    parser.add_argument("--vllm-launcher", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-python", action="store_true")
    parser.add_argument("--include-nccl", action="store_true")
    parser.add_argument("--include-launchers", action="store_true")
    parser.add_argument("--wheel", action="append", default=[])
    args = parser.parse_args()

    execroot = Path.cwd()
    work = work_root(args.work_dir)
    root = work / "root"
    python = execroot / args.python
    python_home = python.parent.parent
    site = root / "opt/venv/lib/python3.12/site-packages"

    site.mkdir(parents=True)
    if args.include_launchers:
        (root / "opt/venv/bin").mkdir(parents=True)
        (root / "root/.cache/vllm").mkdir(parents=True)
        (root / "root/.cache/flashinfer").mkdir(parents=True)

    if args.include_python:
        # Preserve the rules_python interpreter with its matching standard library.
        shutil.copytree(python_home, root / "opt/python", symlinks=False)
    install_wheels(python, site, [execroot / wheel for wheel in args.wheel])
    if args.include_nccl:
        extract(execroot / args.nccl_tar, root / "opt/nccl")

    if args.include_launchers:
        for source, destination in (
            (execroot / args.python_launcher, root / "opt/venv/bin/python"),
            (execroot / args.vllm_launcher, root / "opt/venv/bin/vllm"),
        ):
            shutil.copy2(source, destination)
            os.chmod(destination, 0o755)

    write_tar(execroot / args.output, root)


if __name__ == "__main__":
    main()
