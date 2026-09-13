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
