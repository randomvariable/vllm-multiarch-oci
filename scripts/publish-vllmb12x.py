#!/usr/bin/env python3
"""Publish a locked vLLMB12X image under an immutable, allocated tag.

A Kubernetes Lease serializes the short registry mutation, not the multi-hour
Bazel build. The tag includes both the locked vLLM revision and the builder
revision, so later source-lock or builder changes cannot overwrite it.

A pull-request publication takes no Lease: its tag is derived from the two
revisions alone, it never moves `latest`, and it goes to the internal registry.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Final

from ci_lease import Lease, utc_now

ROOT: Final = Path(__file__).resolve().parents[1]
PROFILE: Final = ROOT / "profiles/vllmb12x/profile.json"
_COMMIT: Final = re.compile(r"[0-9a-f]{40}")
_DATE: Final = re.compile(r"[0-9]{8}")
_REF_PREFIX: Final = "refs/heads/"


def source_branch(profile: Path) -> str:
    ref = json.loads(profile.read_text()).get("source_ref")
    if isinstance(ref, str) and re.fullmatch(r"[0-9a-f]{40}", ref):
        # Immutable-revision pins carry the SHA itself; slug it directly.
        return ref[:12]
    if not isinstance(ref, str) or not ref.startswith(_REF_PREFIX):
        raise ValueError(f"{profile} does not contain a full branch ref")
    branch = ref.removeprefix(_REF_PREFIX)
    slug = re.sub(r"[^a-z0-9]+", "-", branch.lower()).strip("-")
    if not slug:
        raise ValueError(f"invalid vLLM source branch: {ref!r}")
    return slug


def allocate_tag(source_branch: str, source_revision: str, build_revision: str, date: str, sequence: int) -> str:
    """Create an immutable tag from the branch and revisions that define the build."""
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", source_branch):
        raise ValueError(f"invalid vLLM source branch slug: {source_branch!r}")
    if not _COMMIT.fullmatch(source_revision):
        raise ValueError(f"invalid vLLM source revision: {source_revision!r}")
    if not _COMMIT.fullmatch(build_revision):
        raise ValueError(f"invalid builder revision: {build_revision!r}")
    if not _DATE.fullmatch(date):
        raise ValueError(f"invalid UTC build date: {date!r}")
    if sequence < 1:
        raise ValueError(f"invalid publication sequence: {sequence!r}")
    return f"vllmb12x-{source_branch}-{source_revision[:12]}-{build_revision[:12]}-{date}-n{sequence}"


def pull_request_tag(pull_request: int, source_revision: str, build_revision: str) -> str:
    """Create a pull-request tag from the two revisions that define the build.

    No sequence and no date: the same pull-request commit pair always produces
    the same tag, so a re-run overwrites its own image rather than adding one.
    """
    if pull_request < 1:
        raise ValueError(f"invalid pull request number: {pull_request!r}")
    if not _COMMIT.fullmatch(source_revision):
        raise ValueError(f"invalid vLLM source revision: {source_revision!r}")
    if not _COMMIT.fullmatch(build_revision):
        raise ValueError(f"invalid builder revision: {build_revision!r}")
    return f"pr-{pull_request}-{source_revision[:12]}-{build_revision[:12]}"


def push(crane: str, image_layout: Path, repository: str, tag: str) -> str:
    """Push the built OCI layout and return its digest-qualified reference."""
    # Alongside the output base so the blob hard links stay on one filesystem.
    with tempfile.TemporaryDirectory(dir=image_layout.resolve().parent) as temporary_directory:
        image_refs = Path(temporary_directory) / "image-refs.txt"
        pushable = materialize_layout(image_layout, Path(temporary_directory) / "layout")
        subprocess.run(
            [crane, "push", "--image-refs", str(image_refs), str(pushable), f"{repository}:{tag}"],
            check=True,
        )
        return published_reference(image_refs)


def source_revision(profile: Path) -> str:
    revision = json.loads(profile.read_text())["sources"]["vllm"]["commit"]
    if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
        raise ValueError(f"{profile} does not contain a full vLLM commit")
    return revision


def published_reference(image_refs: Path) -> str:
    """Return the single digest-qualified reference recorded by crane."""
    references = [line.strip() for line in image_refs.read_text().splitlines() if line.strip()]
    if len(references) != 1 or "@sha256:" not in references[0]:
        raise RuntimeError(f"crane did not write one digest-qualified image reference: {references!r}")
    return references[0]


def materialize_layout(layout: Path, destination: Path) -> Path:
    """Copy an OCI layout with every blob resolved to a regular file.

    Bazel symlinks base-image and apt blobs into the external repository
    directories, and crane refuses a layout blob that is a symlink. Hard links
    avoid copying the bytes when the output base shares a filesystem with the
    destination.
    """
    for source in sorted(layout.rglob("*")):
        if source.is_dir() and not source.is_symlink():
            continue
        target = destination / source.relative_to(layout)
        target.parent.mkdir(parents=True, exist_ok=True)
        resolved = source.resolve()
        try:
            os.link(resolved, target)
        except OSError:
            shutil.copyfile(resolved, target)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, help="Registry repository without a tag")
    parser.add_argument("--build-revision", required=True, help="Full commit of this builder checkout")
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--lease-name", default="vllmb12x-publish")
    parser.add_argument("--namespace", default="ci")
    parser.add_argument("--lease-duration", type=int, default=60)
    parser.add_argument("--kubectl", default="kubectl")
    parser.add_argument("--bazel", default="bazel")
    parser.add_argument("--crane", default="crane")
    parser.add_argument(
        "--bazel-arg",
        action="append",
        default=[],
        help="Additional option passed before Bazel's build command",
    )
    parser.add_argument("--image-target", default="//image:vllmb12x")
    parser.add_argument("--date", help="UTC YYYYMMDD, intended for deterministic tests")
    parser.add_argument(
        "--pull-request",
        type=int,
        help="Publish a pull-request build: revision-derived tag, no Lease, no latest",
    )
    parser.add_argument("--result-file", type=Path)
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z0-9./_-]+", args.repository):
        parser.error("repository must not contain a registry tag or digest")
    revision = source_revision(args.profile)
    branch = source_branch(args.profile)
    # The build holds no lock: only the registry mutation below is serialized.
    subprocess.run(
        [
            args.bazel,
            "build",
            *args.bazel_arg,
            "--remote_download_outputs=all",
            args.image_target,
        ],
        check=True,
    )
    image_layout = ROOT / "bazel-bin/image/vllmb12x"
    if args.pull_request is not None:
        # A pull-request image is addressed by its own tag and nothing else
        # points at it, so there is no shared registry state to serialize.
        tag = pull_request_tag(args.pull_request, revision, args.build_revision)
        reference = push(args.crane, image_layout, args.repository, tag)
        report(args, tag, reference)
        return
    date = args.date or utc_now().strftime("%Y%m%d")
    lease = Lease(args.kubectl, args.namespace, args.lease_name, args.lease_duration)
    sequence = lease.acquire()
    tag = allocate_tag(branch, revision, args.build_revision, date, sequence)
    try:
        reference = push(args.crane, image_layout, args.repository, tag)
        # The immutable tag is safe after the lease expires. `latest` is not.
        lease.assert_held()
        subprocess.run(
            [
                args.crane,
                "tag",
                f"{args.repository}:{tag}",
                "latest",
            ],
            check=True,
        )
        lease.assert_held()
    finally:
        lease.release()
    report(args, tag, reference)


def report(args: argparse.Namespace, tag: str, reference: str) -> None:
    result = json.dumps({"repository": args.repository, "tag": tag, "reference": reference})
    if args.result_file is not None:
        args.result_file.write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
