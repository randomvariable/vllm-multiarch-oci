#!/usr/bin/env python3
"""Build FlashInfer's profile-pinned AOT cache wheel from the source tree."""

import pathlib
import subprocess
import sys

source = pathlib.Path.cwd()
wheel_dir = pathlib.Path(sys.argv[2])
subproject = source / "flashinfer-jit-cache"
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
        subproject,
    ],
    check=True,
)
