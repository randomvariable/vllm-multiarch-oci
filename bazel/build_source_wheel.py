#!/usr/bin/env python3
"""Build one wheel from an upstream source tree without resolving dependencies."""

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
        "--no-deps",
        "--no-build-isolation",
        "--wheel-dir",
        wheel_dir,
        source,
    ],
    check=True,
)
