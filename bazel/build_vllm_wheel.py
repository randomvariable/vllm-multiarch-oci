#!/usr/bin/env python3
"""Build vLLM's Rust frontend before packaging its CUDA Python wheel."""

import os
import pathlib
import subprocess
import sys

source = pathlib.Path.cwd()
wheel_dir = pathlib.Path(sys.argv[2])


def enable_local_precompiled_extensions() -> None:
    """Teach the pinned vLLM package build to consume Bazel's extension tree."""
    setup = source / "setup.py"
    contents = setup.read_text()
    patched_contents = contents.replace(
        "if USE_PRECOMPILED_RUST_FRONTEND and not is_metadata_only_build():",
        "if (USE_PRECOMPILED_RUST_FRONTEND and not is_metadata_only_build() "
        "and not os.getenv(\"VLLM_PRECOMPILED_EXTENSION_DIR\")):",
        1,
    )
    if patched_contents == contents:
        raise RuntimeError("pinned vLLM precompiled download hook changed")
    old = """    def run(self) -> None:
        return
"""
    new = """    def run(self) -> None:
        extension_dir = os.environ.get(\"VLLM_PRECOMPILED_EXTENSION_DIR\")
        if extension_dir:
            shutil.copytree(
                os.path.join(extension_dir, \"vllm\"),
                os.path.join(self.build_lib, \"vllm\"),
                dirs_exist_ok=True,
            )
        return
"""
    if old not in contents:
        raise RuntimeError("pinned vLLM precompiled build hook changed")
    setup.write_text(patched_contents.replace(old, new, 1))


if "VLLM_PRECOMPILED_EXTENSION_DIR" in os.environ:
    enable_local_precompiled_extensions()

if "VLLM_PRECOMPILED_EXTENSION_DIR" not in os.environ:
    subprocess.run(
        [sys.argv[1], "tools/build_rust.py", "--release"],
        cwd=source,
        check=True,
    )
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
    cwd=source,
    check=True,
)
