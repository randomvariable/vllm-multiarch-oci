#!/usr/bin/env python3
"""Build one architecture-specific FlashInfer jit-cache provider wheel.

The provider subproject compiles the AOT kernel inventory for the single CUDA
architecture named by FLASHINFER_JIT_CACHE_PROVIDER_ARCH and emits a
flashinfer-jit-cache-<smNNN> wheel whose entry point registers it with the
flashinfer-jit-cache shim.
"""

import pathlib
import subprocess
import sys

source = pathlib.Path.cwd()
wheel_dir = pathlib.Path(sys.argv[2])
subproject = source / "flashinfer-jit-cache-provider"
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
