#!/usr/bin/env python3
"""Tag a published image as a release of this builder.

Publication is continuous: every accepted build gets an immutable tag. A
release is the separate, deliberate statement that one of those images is the
one to use. This command makes that statement in three places at once, and
refuses when they would disagree: a registry tag, a git tag, and a GitHub
release whose notes carry the same change ledger the image advertises.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parents[1]
PROFILE: Final = ROOT / "profiles/vllmb12x/profile.json"
VERSION: Final = ROOT / "profiles/vllmb12x/version.bzl"
_REFERENCE: Final = re.compile(r"(?P<repository>[a-z0-9./_-]+)@(?P<digest>sha256:[0-9a-f]{64})")
_RELEASE_TAG: Final = re.compile(r"v(?P<date>[0-9]{8})\.(?P<sequence>[0-9]+)")
# vllmb12x-<branch>-<vllm 12>-<builder 12>-<UTC date>-n<sequence>
_PUBLICATION_TAG: Final = re.compile(
    r"vllmb12x-.+-(?P<source>[0-9a-f]{12})-(?P<builder>[0-9a-f]{12})-(?P<date>[0-9]{8})-n(?P<sequence>[0-9]+)"
)
_RELEASE_PLATFORMS: Final = frozenset({("linux", "arm64"), ("linux", "amd64")})


def ledger() -> Any:
    specification = importlib.util.spec_from_file_location(
        "vllmb12x_included_changes", Path(__file__).with_name("vllmb12x-included-changes.py")
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def run(*command: str, cwd: Path = ROOT) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{' '.join(command)} failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout


def next_release_tag(existing: list[str], date: str) -> str:
    """Return the next calendar release tag for the given UTC date.

    Calendar rather than semantic: this builder tracks moving upstream forks,
    so a version number would imply a compatibility promise it cannot keep.
    """
    sequences = [
        int(match.group("sequence"))
        for tag in existing
        if (match := _RELEASE_TAG.fullmatch(tag.strip())) and match.group("date") == date
    ]
    return f"v{date}.{max(sequences, default=0) + 1}"


def image_labels(reference: str, crane: str) -> dict[str, str]:
    index = json.loads(run(crane, "manifest", reference))
    manifests = index.get("manifests") if isinstance(index, dict) else None
    if not isinstance(manifests, list):
        raise RuntimeError(f"{reference} is not an OCI image index")
    children: dict[tuple[str, str], str] = {}
    for descriptor in manifests:
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("platform"), dict):
            raise RuntimeError(f"OCI index child lacks a platform for {reference}")
        platform = descriptor["platform"]
        key = (platform.get("os"), platform.get("architecture"))
        digest = descriptor.get("digest")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise RuntimeError(f"OCI index child lacks a digest for {reference}")
        if key in children:
            raise RuntimeError(f"OCI index has duplicate child platform {key[0]}/{key[1]}")
        children[key] = digest
    if frozenset(children) != _RELEASE_PLATFORMS:
        raise RuntimeError(
            f"OCI index platforms are {sorted(children)}, expected linux/arm64 and linux/amd64"
        )

    child_labels: list[dict[str, str]] = []
    repository = reference.split("@", 1)[0]
    for platform, digest in sorted(children.items()):
        configuration = json.loads(run(crane, "config", f"{repository}@{digest}"))
        labels = configuration.get("config", {}).get("Labels") if isinstance(configuration, dict) else None
        if not isinstance(labels, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in labels.items()):
            raise RuntimeError(f"OCI child {platform[0]}/{platform[1]} lacks string image labels")
        child_labels.append(labels)
    if child_labels[0] != child_labels[1]:
        raise RuntimeError("OCI child image labels disagree")
    return child_labels[0]


def check_release_candidate(reference: str, publication_tag: str, labels: dict[str, str], module: Any) -> None:
    """Fail unless the image was built from the current, committed tree."""
    published = _PUBLICATION_TAG.fullmatch(publication_tag)
    if published is None:
        raise RuntimeError(f"{publication_tag!r} is not a publication tag this builder allocated")
    lock = module.load_lock()
    manifest = json.loads(module.MANIFEST.read_text())
    expected = {
        "uk.co.randomvariable.vllmb12x.vllm-revision": lock["sources"]["vllm"]["commit"],
        "uk.co.randomvariable.vllmb12x.vllm-source-ref": lock["source_ref"],
        # The description carries the change ledger, so comparing it ties the
        # release to the documented provenance, not just to the same commits.
        "org.opencontainers.image.description": module.description_body(manifest, lock),
    }
    for key, value in expected.items():
        if labels.get(key) != value:
            raise RuntimeError(
                f"{reference} does not match this tree: {key} is {labels.get(key)!r}, expected {value!r}"
            )
    if run("git", "status", "--porcelain", "--untracked-files=no").strip():
        raise RuntimeError("the working tree has uncommitted changes")
    head = run("git", "rev-parse", "HEAD").strip()
    # The tag names the builder revision, so a later commit means the release
    # would promise source that never produced this image.
    if not head.startswith(published.group("builder")):
        raise RuntimeError(
            f"{reference} was built from {published.group('builder')}, but HEAD is {head[:12]}"
        )
    # A release names the default branch, not a branch that may be rebased or
    # deleted once its pull request closes.
    branches = {line.strip().lstrip("* ") for line in run("git", "branch", "--remotes", "--contains", "HEAD").splitlines()}
    if "origin/main" not in branches:
        raise RuntimeError("HEAD is not on origin/main, so the release tag would name a commit that may disappear")


def release_notes(reference: str, publication_tag: str, release_tag: str, module: Any) -> str:
    lock = module.load_lock()
    manifest = json.loads(module.MANIFEST.read_text())
    version = re.search(r'VLLM_BUILD_VERSION = "([^"]+)"', VERSION.read_text()).group(1)
    repository = reference.split("@", 1)[0]
    return "\n".join((
        f"Linux arm64 (sm_121a) and amd64 (sm_120) image index, vLLM `{version}`.",
        "",
        "```bash",
        f"docker pull {reference}",
        "```",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Digest | `{reference.split('@', 1)[1]}` |",
        f"| Release tag | `{repository}:{release_tag}` |",
        f"| Publication tag | `{repository}:{publication_tag}` |",
        f"| vLLM | `{lock['sources']['vllm']['commit']}` |",
        f"| B12X | `{lock['sources']['b12x']['commit']}` |",
        f"| Builder | `{run('git', 'rev-parse', 'HEAD').strip()}` |",
        "",
        "## Base vLLM image additions",
        "",
        "- Mooncake Transfer Engine CUDA 13 `0.3.13.post1` provides the `mooncake` Python package and its service and benchmark commands for deployments that select vLLM's `MooncakeStoreConnector`. It does not enable a connector or start Mooncake services.",
        "- See [Mooncake Transfer Engine](https://github.com/randomvariable/vllm-multiarch-oci/blob/main/docs/reference/mooncake-transfer-engine.md) for the installed surface and deployment boundary.",
        "",
        "## Included upstream changes",
        "",
        module.render_readme(manifest, lock),
    ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="Digest-qualified image published earlier")
    parser.add_argument("--publication-tag", required=True, help="The immutable tag that image was published under")
    parser.add_argument("--tag", help="Release tag, by default the next vYYYYMMDD.N")
    parser.add_argument("--crane", default="crane")
    parser.add_argument("--dry-run", action="store_true", help="Print the notes and the commands, change nothing")
    arguments = parser.parse_args(argv)

    match = _REFERENCE.fullmatch(arguments.reference)
    if match is None:
        parser.error("--reference must be a digest-qualified image reference")
    repository = match.group("repository")

    module = ledger()
    labels = image_labels(arguments.reference, arguments.crane)
    check_release_candidate(arguments.reference, arguments.publication_tag, labels, module)

    release_tag = arguments.tag or next_release_tag(
        run("git", "tag", "--list").splitlines(), dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d")
    )
    if _RELEASE_TAG.fullmatch(release_tag) is None:
        parser.error(f"release tag must look like v20260917.1, not {release_tag!r}")
    notes = release_notes(arguments.reference, arguments.publication_tag, release_tag, module)

    if arguments.dry_run:
        print(f"release tag: {release_tag}\nregistry tag: {repository}:{release_tag}\n")
        print(notes)
        return 0

    # Registry first: a git tag pointing at an image that failed to tag is
    # worse than a registry tag with no release yet.
    run(arguments.crane, "copy", arguments.reference, f"{repository}:{release_tag}")
    run("git", "tag", "--annotate", release_tag, "--message", f"{release_tag}: {arguments.reference}")
    run("git", "push", "origin", release_tag)
    run(
        "gh", "release", "create", release_tag,
        "--repo", "randomvariable/vllm-multiarch-oci",
        "--title", f"{release_tag} — vLLM {lock_version()}",
        "--notes", notes,
    )
    print(json.dumps({
        "release": release_tag,
        "registry_tag": f"{repository}:{release_tag}",
        "reference": arguments.reference,
    }))
    return 0


def lock_version() -> str:
    return re.search(r'VLLM_BUILD_VERSION = "([^"]+)"', VERSION.read_text()).group(1)


if __name__ == "__main__":
    raise SystemExit(main())
