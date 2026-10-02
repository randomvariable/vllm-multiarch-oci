#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Logs the public lane into GHCR with the publisher token that the PipelineRun
# mounts as a file. The pull-request lane never touches this: its credential is
# mounted already in Docker config form, so the step does no token handling.

set -euo pipefail

: "${GITHUB_TOKEN_FILE:?GITHUB_TOKEN_FILE must name the mounted publisher token}"
: "${DOCKER_CONFIG:?DOCKER_CONFIG must name the directory crane reads}"

mkdir -p "$DOCKER_CONFIG"
printf '{"auths":{"ghcr.io":{"auth":"%s"}}}\n' \
    "$(printf 'x-access-token:%s' "$(cat "$GITHUB_TOKEN_FILE")" | base64 -w0)" \
    >"$DOCKER_CONFIG/config.json"
