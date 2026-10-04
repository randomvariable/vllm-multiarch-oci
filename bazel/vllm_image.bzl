# SPDX-License-Identifier: Apache-2.0
"""Compose the profile's locked runtime layers into an OCI image."""

load("@rules_oci//oci:defs.bzl", "oci_image", "oci_load")

SOURCE_REPOSITORY = "https://github.com/local-inference-lab/vllm"
BUILDER_REPOSITORY = "https://github.com/randomvariable/vllm-multiarch-oci"

# The launcher owns the whole start-up path: it publishes the checkpoint, resolves
# rank, waits for the leader, validates the cache, then execs vLLM, which becomes
# the container's PID 1. The image therefore enters Python, never a shell. An
# ``sh`` at the front owns the process the runtime must signal, hides the real
# exit status from the restart policy, and puts a fork in every probe.
ENTRYPOINT = [
    "/opt/python/bin/python",
    "-m",
    "image_tools.vllm_image",
    "entrypoint",
]

# The entrypoint names the interpreter directly, so the mounted wheel tree has to
# be on its import path from the image configuration. The venv layer installs
# sitecustomize.py in that directory, which is what makes the .pth files inside it
# run; see bazel/venv_layer_action.py.
ENTRYPOINT_ENV = {"PYTHONPATH": "/opt/venv/lib/python3.12/site-packages"}

def vllm_image(
        name,
        base,
        accounts_layer,
        apt_layer,
        cuda_layer,
        env,
        provenance_layer,
        venv_layers,
        title,
        description,
        local_tag,
        vllm_version,
        vllm_source_ref,
        vllm_revision,
        launcher_data_layer = None):
    """Create the image and the local loader.

    The release tag is allocated after the build, so it is not an input here.
    The image records both builder and upstream vLLM source identities. The
    registry tag adds the publication date and sequence after the build.
    """
    tars = [
        apt_layer,
        cuda_layer,
        provenance_layer,
    ] + venv_layers + [
        # Last among the runtime content, so the account files this layer owns
        # win over anything the base or an earlier layer put at the same path.
        accounts_layer,
    ]
    if launcher_data_layer:
        # Below the accounts layer and above the wheels: /opt/vllm-image is owned
        # by this layer alone, so nothing here can be shadowed by a wheel, and it
        # cannot shadow an account file.
        tars.insert(3, launcher_data_layer)

    oci_image(
        name = name,
        base = base,
        tars = tars,
        env = env,
        entrypoint = ENTRYPOINT,
        labels = {
            "org.opencontainers.image.title": title,
            # Registries show this on the package page. Without it they show
            # the CUDA base image's own description, which describes Ubuntu.
            "org.opencontainers.image.description": description,
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
