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

# Default budget per concurrent nvcc or g++ invocation. A target whose
# translation units are larger overrides it with memory_per_job.
MEBIBYTES_PER_JOB = 4096

# The action shares its cgroup with the nativelink agent, the extracted
# toolchains, and the page cache the NFS-backed ccache reads through.
_WORKER_RESERVE_MEBIBYTES = 8192

MEMORY_PER_JOB_ATTR = attr.int(default = MEBIBYTES_PER_JOB)

def compile_jobs(attr):
    """Return the compile parallelism an action's declared reservation affords.

    Actions reserve 20 CPUs on the remote worker. Compiling with a fixed four
    jobs left that reservation idle and made every cache-cold wheel build
    roughly five times longer than the hardware required.

    The worker is OOMKilled rather than throttled when the sum of the
    concurrent compilers exceeds its 64GiB cgroup, so the budget is a hard
    divisor rather than a hint.
    """
    budget = attr.memory - _WORKER_RESERVE_MEBIBYTES
    per_job = attr.memory_per_job or MEBIBYTES_PER_JOB
    return max(1, min(attr.max_jobs, attr.cpu, budget // per_job))
