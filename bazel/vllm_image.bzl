# SPDX-License-Identifier: Apache-2.0
"""Compose the profile's locked runtime layers into an OCI image."""

load("@rules_oci//oci:defs.bzl", "oci_image", "oci_load")

SOURCE_REPOSITORY = "https://github.com/local-inference-lab/vllm"
BUILDER_REPOSITORY = "https://github.com/randomvariable/vllm-multiarch-oci"

def vllm_image(
        name,
        base,
        apt_layer,
        cuda_layer,
        provenance_layer,
        venv_layers,
        title,
        local_tag,
        vllm_version,
        vllm_source_ref,
        vllm_revision):
    """Create the image and the local loader.

    The release tag is allocated after the build, so it is not an input here.
    The image records both builder and upstream vLLM source identities. The
    registry tag adds the publication date and sequence after the build.
    """
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
            provenance_layer,
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
            "org.opencontainers.image.title": title,
            "org.opencontainers.image.source": SOURCE_REPOSITORY,
            "org.opencontainers.image.revision": vllm_revision,
            "org.opencontainers.image.version": vllm_version,
            "org.opencontainers.image.licenses": "Apache-2.0",
            "uk.co.randomvariable.vllmb12x.builder-source": BUILDER_REPOSITORY,
            "uk.co.randomvariable.vllmb12x.vllm-source-ref": vllm_source_ref,
            "uk.co.randomvariable.vllmb12x.vllm-version": vllm_version,
            "uk.co.randomvariable.vllmb12x.vllm-revision": vllm_revision,
        },
        workdir = "/root",
    )

    oci_load(
        name = name + "_load",
        image = ":" + name,
        repo_tags = ["local/" + title + ":" + local_tag],
    )
