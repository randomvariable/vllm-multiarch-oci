#!/usr/bin/env python3
"""Build one architecture-specific FlashInfer jit-cache provider wheel.

The provider subproject compiles the AOT kernel inventory for the single CUDA
architecture named by FLASHINFER_JIT_CACHE_PROVIDER_ARCH and emits a
flashinfer-jit-cache-<smNNN> wheel whose entry point registers it with the
flashinfer-jit-cache shim.
"""

import os
import pathlib
import subprocess
import sys

source = pathlib.Path.cwd()
wheel_dir = pathlib.Path(sys.argv[2])
subproject = source / "flashinfer-jit-cache-provider"

# The AOT kernel inventory includes csrc/dcp_lse_reduce.cu, whose header pulls
# in <nccl_device.h>. flashinfer's JIT adds cutlass/spdlog/cccl/tvm_ffi but not
# NCCL to the nvcc include set, and it appends FLASHINFER_EXTRA_{CFLAGS,CUDAFLAGS}
# last. Point those at the NCCL SDK the Bazel action already extracted.
nccl_include = os.environ.get("NCCL_INCLUDE_DIR")
if nccl_include:
    flag = f"-isystem {nccl_include}"
    for name in ("FLASHINFER_EXTRA_CFLAGS", "FLASHINFER_EXTRA_CUDAFLAGS"):
        existing = os.environ.get(name, "")
        os.environ[name] = f"{existing} {flag}".strip()

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
