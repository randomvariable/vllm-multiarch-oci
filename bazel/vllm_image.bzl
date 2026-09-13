# SPDX-License-Identifier: Apache-2.0
"""Compose a profile's locked runtime layers into an OCI image."""

load("@rules_oci//oci:defs.bzl", "oci_image", "oci_load")

def vllm_image(name, base, apt_layer, cuda_layer, venv_layers, repository, tag, vllm_version):
    """Create the image and local loader for one profile."""
    oci_image(
        name = name,
        base = base,
        # The pinned CUDA base and all native wheels are ARM64-only. The
        # x86 Blackwell platform is a declaration for future profiles, not a
        # valid configuration for this image.
        target_compatible_with = [
            "@platforms//cpu:aarch64",
            "@platforms//os:linux",
        ],
        tars = [
            apt_layer,
            cuda_layer,
        ] + venv_layers,
        env = {
            "CUDA_HOME": "/usr/local/cuda",
            "FLASHINFER_CUDA_ARCH_LIST": "12.1f",
            # The pinned CUDA image owns the CUDA, cuBLAS, and cuDNN ABI.
            "LD_LIBRARY_PATH": "/opt/nccl/lib:/opt/cusparselt/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
            "PATH": "/opt/venv/bin:$PATH",
            "TORCH_CUDA_ARCH_LIST": "12.1a",
            "VLLM_DISABLED_KERNELS": "MarlinFP8ScaledMMLinearKernel",
        },
        entrypoint = ["/opt/venv/bin/vllm"],
        labels = {
            "org.opencontainers.image.title": "vLLM GLM53 CUDA runtime",
            "org.opencontainers.image.version": vllm_version,
        },
        workdir = "/root",
    )

    oci_load(
        name = name + "_load",
        image = ":" + name,
        repo_tags = [repository + ":" + tag],
    )
