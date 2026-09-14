#!/usr/bin/env python3
"""Assemble the self-contained Python runtime OCI layer."""

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from action_lib import extract, install_wheels, work_root, write_tar

# The launchers put the venv on PYTHONPATH, and a PYTHONPATH entry is an
# ordinary sys.path directory: CPython never processes the .pth files in it.
# Wheels that publish their real package root through a .pth therefore stay
# unimportable -- nvidia-cutlass-dsl ships dsl_packages that way, so
# `import cutlass` failed even though the files were installed. Importing
# this module during interpreter startup turns the directory into a site
# directory, which runs every .pth it contains.
_SITECUSTOMIZE = '''\
"""Make the PYTHONPATH-mounted venv a site directory so .pth files run."""

import site

site.addsitedir("/opt/venv/lib/python3.12/site-packages")
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--nccl-tar", required=True)
    parser.add_argument("--python-launcher", required=True)
    parser.add_argument("--vllm-launcher", required=True)
    parser.add_argument("--image-helper", required=True)
    parser.add_argument("--image-helper-launcher", required=True)
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
            (execroot / args.image_helper_launcher, root / "opt/venv/bin/vllm-image"),
        ):
            shutil.copy2(source, destination)
            os.chmod(destination, 0o755)
        (site / "sitecustomize.py").write_text(_SITECUSTOMIZE)
        helper_package = site / "image_tools"
        helper_package.mkdir()
        (helper_package / "__init__.py").write_text("")
        shutil.copy2(execroot / args.image_helper, helper_package / "vllm_image.py")

    write_tar(execroot / args.output, root)


if __name__ == "__main__":
    main()
