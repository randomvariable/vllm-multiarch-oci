#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Installs the toolchain a vllmb12x CI lane runs on. The Tekton step image is a
# plain Ubuntu base because gcr.io/bazel-public/bazel publishes amd64 only and
# every lane is pinned to the arm64 build node, so nothing here is preinstalled.
#
# Downloads are pinned by SHA-256 rather than by moving URL, and each tool is
# skipped when it is already present so re-running a step does not refetch.

set -euo pipefail

BAZELISK_VERSION="v1.29.0"
BAZELISK_SHA256="e20e8b0f4f240091b7a55bf17b9398bd4f40ee70ae0208dff95dd4c445fb4010"
GO_CONTAINERREGISTRY_VERSION="v0.22.1"
GO_CONTAINERREGISTRY_SHA256="898c0cff975f898a33e8c4580bdafb0e7c02c7faa33374e946762f97c4ab7110"
GH_VERSION="2.102.0"
# Verified against the published gh_2.102.0_checksums.txt and the downloaded
# arm64 tarball, the same way the crane and bazelisk pins were.
GH_SHA256="7862c86c72f43df3a2d93ddde6f473285b4e2af61b494849846827e513ef6484"
RUST_TOOLCHAIN="1.95.0"
RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}"
CARGO_HOME="${CARGO_HOME:-/opt/cargo}"
ARCH="${TARGETARCH:-arm64}"
DOWNLOAD_DIR="${TMPDIR:-${HOME:-/var/tmp}/ci-downloads}"
mkdir -p "$DOWNLOAD_DIR"

fetch() {
    # $1: destination, $2: url
    curl --fail --silent --show-error --location "$2" --output "$1"
}

apt-get update
apt-get install --no-install-recommends --yes \
    build-essential ca-certificates curl git patch python3 unzip zip
rm -rf /var/lib/apt/lists/*

if ! command -v bazel >/dev/null 2>&1; then
    fetch /usr/local/bin/bazel \
        "https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-${ARCH}"
    printf '%s  %s\n' "$BAZELISK_SHA256" /usr/local/bin/bazel | sha256sum --check --status
    chmod 0755 /usr/local/bin/bazel
fi

if ! command -v cargo >/dev/null 2>&1; then
    export RUSTUP_HOME CARGO_HOME
    fetch "${DOWNLOAD_DIR}/rustup-init" https://sh.rustup.rs
    bash "${DOWNLOAD_DIR}/rustup-init" -y --profile minimal --default-toolchain "$RUST_TOOLCHAIN"
fi
export RUSTUP_HOME CARGO_HOME PATH="${CARGO_HOME}/bin:${PATH}"

if ! command -v kubectl >/dev/null 2>&1; then
    kubectl_version="$(curl --fail --silent --show-error --location https://dl.k8s.io/release/stable.txt)"
    fetch /usr/local/bin/kubectl "https://dl.k8s.io/release/${kubectl_version}/bin/linux/${ARCH}/kubectl"
    chmod 0755 /usr/local/bin/kubectl
fi

if ! command -v crane >/dev/null 2>&1; then
    archive="${DOWNLOAD_DIR}/go-containerregistry_Linux_${ARCH^}.tar.gz"
    fetch "$archive" \
        "https://github.com/google/go-containerregistry/releases/download/${GO_CONTAINERREGISTRY_VERSION}/go-containerregistry_Linux_${ARCH^}.tar.gz"
    printf '%s  %s\n' "$GO_CONTAINERREGISTRY_SHA256" "$archive" | sha256sum --check --status
    tar --extract --gzip --file "$archive" --directory /usr/local/bin crane
    chmod 0755 /usr/local/bin/crane
fi

if ! command -v gh >/dev/null 2>&1; then
    # The release lane opens the GitHub release; the build lanes never call it.
    archive="${DOWNLOAD_DIR}/gh_${GH_VERSION}_linux_${ARCH}.tar.gz"
    fetch "$archive" \
        "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_${ARCH}.tar.gz"
    printf '%s  %s\n' "$GH_SHA256" "$archive" | sha256sum --check --status
    tar --extract --gzip --file "$archive" --strip-components=2 \
        --directory /usr/local/bin "gh_${GH_VERSION}_linux_${ARCH}/bin/gh"
    chmod 0755 /usr/local/bin/gh
fi

crane version
