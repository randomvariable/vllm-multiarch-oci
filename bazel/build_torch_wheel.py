#!/usr/bin/env python3
"""Build the pinned PyTorch source tree with the supplied CUDA toolchain.

PyTorch 2.14 builds through scikit-build-core's PEP 517 backend; `setup.py
bdist_wheel` is refused. The build dependencies are already installed from
the locked host wheels, so build without isolation or dependency resolution.
"""

import pathlib
import subprocess
import sys

source = pathlib.Path.cwd()
wheel_dir = pathlib.Path(sys.argv[2])
subprocess.run(
    [
        sys.argv[1],
        "-m",
        "pip",
        "wheel",
        "--no-build-isolation",
        "--no-deps",
        "--no-index",
        "-v",
        "-w",
        wheel_dir,
        ".",
    ],
    cwd=source,
    check=True,
)
