#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# The one build sequence every vllmb12x CI lane runs: check the Python tools,
# build the multiarch image, run the image contract on both architectures, then
# publish it. Lanes differ only in the tag the publisher allocates and where the
# image lands, so that difference is the LANE variable rather than a copy.
#
# The caller holds the build mutex: two lanes building at once halve the remote
# workers each sees and evict each other's ccache entries.

set -euo pipefail

: "${LANE:?LANE must be one of nightly, pull-request, release}"
: "${IMAGE_REPOSITORY:?IMAGE_REPOSITORY must be a registry repository without a tag}"
: "${REMOTE_CACHE:?REMOTE_CACHE must be the remote cache endpoint}"
: "${REMOTE_EXECUTOR:?REMOTE_EXECUTOR must be the remote execution endpoint}"
: "${RESULT:?RESULT must be the path the publisher writes its result JSON to}"

BAZEL="${BAZEL:-bazel}"
PYTHON="${PYTHON:-python3}"
VLLMB12X_CACHE_ROOT="${VLLMB12X_CACHE_ROOT:-/ccache}"
BUILD_REVISION="${BUILD_REVISION:-$(git rev-parse HEAD)}"

case "$LANE" in
    nightly | release)
        : "${PUBLISHER_LEASE:?PUBLISHER_LEASE must name the Lease that allocates the tag}"
        : "${PUBLISHER_NAMESPACE:?PUBLISHER_NAMESPACE must hold that Lease}"
        ;;
    pull-request)
        : "${PULL_REQUEST:?PULL_REQUEST must be the pull request number}"
        ;;
    *)
        echo "unknown lane: $LANE" >&2
        exit 2
        ;;
esac

cd "$(git rev-parse --show-toplevel)"

cat >.bazelrc.user <<EOF
startup --output_user_root=${HOME:?HOME must be the per-run staging directory}/bazel
build --remote_cache=${REMOTE_CACHE}
build --remote_timeout=21600
build --remote_cache_compression
build --remote_upload_local_results=true
build:remote-aarch64 --remote_executor=${REMOTE_EXECUTOR}
build:remote-x86_64 --remote_executor=${REMOTE_EXECUTOR}
build:remote-multiarch --remote_executor=${REMOTE_EXECUTOR}
# Bazel gives repository rules a sanitised environment, so a rule that shells out
# to a bootstrapped host tool has to be told where it lives. _pinned_source_repository
# runs cargo vendor to seal the llguidance sources, and cargo is not on the default
# PATH: without this the fetch dies as execvp(cargo): No such file or directory.
build --repo_env=PATH=${PATH}
EOF

# Local strategy: these are host Python tests, and the remote ARM64 worker
# image carries no python3 interpreter.
"$BAZEL" test //scripts:all --disk_cache= --test_output=errors

"$BAZEL" build //image:vllmb12x \
    --config=remote-multiarch \
    --disk_cache= \
    --//platforms:vllmb12x_cache_root="$VLLMB12X_CACHE_ROOT"

# The rule defaults to --test-report $XML_OUTPUT_FILE --output junit, which
# sends every per-test line to a file that dies with the pod. Clearing the
# report and forcing text keeps container-structure-test's === RUN / --- FAIL
# detail in test.log, where --test_output=errors surfaces it in the TaskRun.
# Every worker runs its own dockerd and exposes its socket at
# /var/run/docker/docker.sock, not at the docker default. A remote
# action's environment is built by Bazel, not inherited from the
# worker's container, so the path has to be passed explicitly.
for platform in aarch64 x86_64; do
    "$BAZEL" test //tests/image:vllmb12x_contract \
        --config="remote-${platform}" \
        --disk_cache= \
        --//platforms:vllmb12x_cache_root="$VLLMB12X_CACHE_ROOT" \
        --test_output=errors \
        --test_arg=--test-report= \
        --test_arg=--output \
        --test_arg=text \
        --test_env=DOCKER_HOST=unix:///var/run/docker/docker.sock
done

publish_arguments=(
    --repository "$IMAGE_REPOSITORY"
    --build-revision "$BUILD_REVISION"
    --bazel-arg=--config=remote-multiarch
    --bazel-arg=--//platforms:vllmb12x_cache_root="$VLLMB12X_CACHE_ROOT"
    --bazel-arg=--remote_cache="$REMOTE_CACHE"
    --bazel-arg=--disk_cache=
    --result-file "$RESULT"
)
case "$LANE" in
    pull-request) publish_arguments+=(--pull-request "$PULL_REQUEST") ;;
    *) publish_arguments+=(--lease-name "$PUBLISHER_LEASE" --namespace "$PUBLISHER_NAMESPACE") ;;
esac

"$PYTHON" scripts/publish-vllmb12x.py "${publish_arguments[@]}"
