# SPDX-License-Identifier: Apache-2.0
"""Pinned ccache launcher input for CUDA build actions."""

# The launcher is pinned against the same Ubuntu snapshot as the runtime layer,
# so it is content-addressed rather than taken from the worker image.
CCACHE_ATTR = attr.label(
    default = Label("@ccache//:ccache.tar"),
    allow_single_file = True,
)

# GCC 15, binutils, libc headers, and libraries from the pinned Ubuntu snapshot.
# Native build actions extract this instead of relying on worker image paths.
GCC_SYSROOT_ATTR = attr.label(
    default = Label("//platforms:gcc_15_sysroot"),
    allow_single_file = True,
)

# Budget per concurrent nvcc or g++ invocation. The CUDA translation units in
# these wheels are the memory-hungry ones, so the job count is clamped to keep
# a fully parallel build inside the action's declared memory reservation.
_MEBIBYTES_PER_JOB = 4096

# The action shares its cgroup with the nativelink agent, the extracted
# toolchains, and the page cache the NFS-backed ccache reads through. A
# 64GiB action on a 64GiB worker left nothing for any of them: both workers
# were OOMKilled (exit 137) compiling FlashInfer's AOT kernels with 16 jobs.
_WORKER_RESERVE_MEBIBYTES = 8192

def compile_jobs(attr):
    """Return the compile parallelism an action's declared reservation affords.

    Actions reserve 20 CPUs on the remote worker. Compiling with a fixed four
    jobs left that reservation idle and made every cache-cold wheel build
    roughly five times longer than the hardware required.
    """
    budget = attr.memory - _WORKER_RESERVE_MEBIBYTES
    return max(1, min(attr.max_jobs, attr.cpu, budget // _MEBIBYTES_PER_JOB))
