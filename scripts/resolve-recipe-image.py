#!/usr/bin/env python3
"""Resolve the current public recipe image to a stable immutable reference."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Final

_DIGEST: Final = re.compile(r"sha256:[0-9a-f]{64}")
_IMMUTABLE_TAG: Final = re.compile(r"vllmb12x-[a-z0-9][a-z0-9._-]*-n([1-9][0-9]*)")
_REPOSITORY: Final = re.compile(
    r"ghcr\.io/[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+"
)


class ResolutionError(RuntimeError):
    """Raised when registry state cannot produce one stable recipe image."""


def _run_crane(crane: str, *arguments: str) -> str:
    try:
        result = subprocess.run(
            [crane, *arguments],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        raise ResolutionError(f"could not execute {crane!r}: {error}") from error
    if result.returncode:
        detail = result.stderr.strip()
        suffix = f": {detail}" if detail else ""
        raise ResolutionError(
            f"crane {arguments[0]} failed with exit status {result.returncode}{suffix}"
        )
    return result.stdout.strip()


def _digest(crane: str, reference: str) -> str:
    digest = _run_crane(crane, "digest", reference)
    if not _DIGEST.fullmatch(digest):
        raise ResolutionError(f"crane returned a malformed digest for {reference!r}")
    return digest


def _validate_platform(crane: str, repository: str, digest: str) -> None:
    reference = f"{repository}@{digest}"
    try:
        config = json.loads(_run_crane(crane, "config", reference))
    except json.JSONDecodeError as error:
        raise ResolutionError(f"crane returned malformed config JSON for {reference!r}") from error
    if not isinstance(config, dict):
        raise ResolutionError(f"crane returned a non-object config for {reference!r}")
    platform = (config.get("os"), config.get("architecture"))
    if platform != ("linux", "arm64"):
        os_name = config.get("os", "unknown")
        architecture = config.get("architecture", "unknown")
        raise ResolutionError(
            f"latest image is {os_name}/{architecture}, expected linux/arm64"
        )


def _immutable_tags(crane: str, repository: str) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []
    for tag in _run_crane(crane, "ls", repository).splitlines():
        tag = tag.strip()
        match = _IMMUTABLE_TAG.fullmatch(tag)
        if match is not None and len(tag) <= 128:
            candidates.append((int(match.group(1)), tag))
    if not candidates:
        raise ResolutionError("registry contains no immutable vllmb12x publication tags")
    return candidates


def _tag_for_digest(
    crane: str,
    repository: str,
    digest: str,
    candidates: list[tuple[int, str]],
) -> str:
    for _, tag in sorted(candidates, reverse=True):
        if _digest(crane, f"{repository}:{tag}") == digest:
            return tag
    raise ResolutionError("latest digest has no matching immutable vllmb12x tag")


def resolve(crane: str, repository: str) -> dict[str, str]:
    """Resolve latest, retrying the full lookup once if latest moves."""
    for attempt in range(2):
        latest_digest = _digest(crane, f"{repository}:latest")
        _validate_platform(crane, repository, latest_digest)
        tag = _tag_for_digest(
            crane,
            repository,
            latest_digest,
            _immutable_tags(crane, repository),
        )
        if _digest(crane, f"{repository}:latest") != latest_digest:
            if attempt == 0:
                continue
            raise ResolutionError("latest changed during both resolution attempts")

        resolved_at = (
            dt.datetime.now(tz=dt.timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        return {
            "repository": repository,
            "tag": tag,
            "reference": f"{repository}@{latest_digest}",
            "resolved_at": resolved_at,
        }
    raise AssertionError("resolution retry loop did not return or fail")


def _write_json_atomic(output: Path, value: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        os.replace(temporary, output)
        directory = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository",
        required=True,
        help="Public GHCR repository without tag or digest",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination latest-image JSON file",
    )
    parser.add_argument("--crane", default="crane", help=argparse.SUPPRESS)
    arguments = parser.parse_args()

    if _REPOSITORY.fullmatch(arguments.repository) is None:
        parser.error(
            "repository must be a lowercase public ghcr.io repository without tag or digest"
        )

    try:
        result = resolve(arguments.crane, arguments.repository)
        _write_json_atomic(arguments.output, result)
    except ResolutionError as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
