#!/usr/bin/env python3
"""Assemble the self-contained Python runtime OCI layer."""

import argparse
import importlib.metadata
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from action_lib import extract, install_wheels, wheel_scripts, work_root, write_tar

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

# The image entry point runs this, so it is the first process in a serving
# container. It resolves the venv the same way sitecustomize.py does: the
# interpreter is the hermetic one under /opt/python, and the wheel tree is
# mounted, not installed, so nothing imports a host Python.
_IMAGE_HELPER_CONSOLE = '''\
#!/opt/python/bin/python
"""Console entry point for the helpers shipped with the vLLM image."""

import site
import sys

site.addsitedir("/opt/venv/lib/python3.12/site-packages")

from image_tools.vllm_image import main

if __name__ == "__main__":
    sys.exit(main())
'''


_CONSOLE_SCRIPT = '''\
#!/opt/python/bin/python
"""Console entry point materialised by the image build."""

import os
import site
import sys

sys.path.insert(0, "/opt/venv/lib/python3.12/site-packages")
site.addsitedir("/opt/venv/lib/python3.12/site-packages")
for _name, _value in (
    ("LD_LIBRARY_PATH", "/opt/nccl/lib:/usr/local/cuda/lib64"),
    ("CUDA_HOME", "/usr/local/cuda"),
):
    _existing = os.environ.get(_name, "")
    os.environ[_name] = (
        _value + (":" + _existing if _existing else "")
        if _existing != _value and _value not in _existing.split(os.pathsep)
        else _existing
    )

from importlib.metadata import distribution

_entry = next(iter(distribution(%r).entry_points.select(group="console_scripts", name=%r)))

if __name__ == "__main__":
    sys.exit(_entry.load()())
'''

def materialize_console_scripts(site: Path, destination: Path) -> None:
    """Create pip console-script wrappers omitted by `pip install --target`.

    These are Python files, not shell wrappers. The launcher execs some of them
    directly -- ``lmcache server`` is the cache container's PID 1 -- and a shell
    in that position both owns the process the container must signal and hides
    the real exit status. The mount points the shell wrappers used to export are
    set here instead, before the entry point is imported.
    """
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        for entry_point in distribution.entry_points:
            if entry_point.group != "console_scripts":
                continue
            wrapper = destination / entry_point.name
            wrapper.write_text(
                _CONSOLE_SCRIPT
                % (distribution.metadata["Name"], entry_point.name)
            )
            wrapper.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--nccl-tar", required=True)
    parser.add_argument("--python-launcher", required=True)
    parser.add_argument("--vllm-launcher", required=True)
    parser.add_argument("--image-tools", action="append", default=[])
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
    wheels = [execroot / wheel for wheel in args.wheel]
    install_wheels(python, site, wheels)
    bin_dir = root / "opt/venv/bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = work / "wheel-scripts"
    for wheel in wheels:
        for script in wheel_scripts(wheel, scripts_dir):
            destination = bin_dir / script.name
            shutil.copy2(script, destination)
            os.chmod(destination, 0o755)
    materialize_console_scripts(site, bin_dir)
    if args.include_nccl:
        extract(execroot / args.nccl_tar, root / "opt/nccl")

    if args.include_launchers:
        for source, destination in (
            (execroot / args.python_launcher, root / "opt/venv/bin/python"),
            (execroot / args.vllm_launcher, root / "opt/venv/bin/vllm"),
        ):
            shutil.copy2(source, destination)
            os.chmod(destination, 0o755)
        (site / "sitecustomize.py").write_text(_SITECUSTOMIZE)
        helper_package = site / "image_tools"
        for entry in args.image_tools:
            source = execroot / entry
            relative = Path(entry).relative_to("image_tools")
            destination = helper_package / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        # The image entry point and the operator command both start here. A shell
        # wrapper would put /bin/sh between the container's PID 1 and the engine,
        # which the launcher deliberately avoids, so this is a Python script whose
        # own shebang names the interpreter.
        console = root / "opt/venv/bin/vllm-image"
        console.write_text(_IMAGE_HELPER_CONSOLE)
        console.chmod(0o755)

    write_tar(execroot / args.output, root)


if __name__ == "__main__":
    main()
