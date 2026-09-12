#!/usr/bin/env python3
"""Package FlashInfer without its upstream networked build-time installers."""

import pathlib
import subprocess
import sys

source = pathlib.Path.cwd()
wheel_dir = pathlib.Path(sys.argv[2])
backend = source / "build_backend.py"
contents = backend.read_text()
patched = contents.replace(
    "    _install_cuda_tile_compile_deps()\n    _build_nvep_if_enabled()\n",
    "    # Bazel supplies locked runtime wheels in the final image layer.\n",
    1,
)
if patched == contents:
    raise RuntimeError("pinned FlashInfer build hook changed")
backend.write_text(patched)
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
