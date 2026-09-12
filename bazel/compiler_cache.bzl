# SPDX-License-Identifier: Apache-2.0
"""Pinned ccache launcher input for CUDA build actions."""

# The launcher is pinned against the same Ubuntu snapshot as the runtime layer,
# so it is content-addressed rather than taken from the worker image.
CCACHE_ATTR = attr.label(
    default = Label("@ccache//:ccache.tar"),
    allow_single_file = True,
)
