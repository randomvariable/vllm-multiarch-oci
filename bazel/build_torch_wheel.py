#!/usr/bin/env python3
"""Build the pinned PyTorch source tree with the supplied CUDA toolchain."""

import pathlib
import subprocess
import sys

source = pathlib.Path.cwd()
wheel_dir = pathlib.Path(sys.argv[2])
subprocess.run(
    [
        sys.argv[1],
        "setup.py",
        "bdist_wheel",
        "-d",
        wheel_dir,
    ],
    cwd=source,
    check=True,
)
