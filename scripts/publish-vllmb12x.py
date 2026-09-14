#!/usr/bin/env python3
"""Publish a locked vLLMB12X image under an immutable, allocated tag.

A Kubernetes Lease serializes the short registry mutation, not the multi-hour
Bazel build. The tag includes both the locked vLLM revision and the builder
revision, so later source-lock or builder changes cannot overwrite it.
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
import threading
import uuid
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parents[1]
PROFILE: Final = ROOT / "profiles/vllmb12x/profile.json"
_COMMIT: Final = re.compile(r"[0-9a-f]{40}")
_DATE: Final = re.compile(r"[0-9]{8}")
_REF_PREFIX: Final = "refs/heads/"


def utc_now() -> dt.datetime:
    return dt.datetime.now(tz=dt.timezone.utc)


def rfc3339(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def parse_rfc3339(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def is_expired(lease: dict[str, Any], now: dt.datetime) -> bool:
    """Return whether the Kubernetes Lease can be acquired safely."""
    spec = lease.get("spec", {})
    holder = spec.get("holderIdentity")
    if not holder:
        return True
    duration = spec.get("leaseDurationSeconds")
    renewed = spec.get("renewTime") or spec.get("acquireTime")
    if not isinstance(duration, int) or duration <= 0 or not isinstance(renewed, str):
        return True
    return now >= parse_rfc3339(renewed) + dt.timedelta(seconds=duration)


def source_branch(profile: Path) -> str:
    ref = json.loads(profile.read_text()).get("source_ref")
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


class Lease:
    """A Kubernetes Lease with optimistic-concurrency renewal.

    The Lease is released by shortening its duration instead of deleting it.
    A delayed process cannot delete a Lease acquired by a later publisher.
    """

    def __init__(
        self,
        kubectl: str,
        namespace: str,
        name: str,
        duration_seconds: int,
        holder: str | None = None,
    ) -> None:
        if duration_seconds < 30:
            raise ValueError("lease duration must be at least 30 seconds")
        self.kubectl = kubectl
        self.namespace = namespace
        self.name = name
        self.duration_seconds = duration_seconds
        self.holder = holder or f"vllmb12x-publisher-{uuid.uuid4()}"
        self._stop = threading.Event()
        self._lost: Exception | None = None
        self._renewer: threading.Thread | None = None
        self.sequence: int | None = None

    def _command(self, *args: str, input: str | None = None, check: bool = True) -> str:
        result = subprocess.run(
            [self.kubectl, "-n", self.namespace, *args],
            input=input,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if check and result.returncode:
            raise RuntimeError(f"{' '.join(result.args)} failed: {result.stderr.strip()}")
        return result.stdout

    def _get(self) -> dict[str, Any] | None:
        result = subprocess.run(
            [self.kubectl, "-n", self.namespace, "get", "lease", self.name, "-o", "json"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        if "not found" in result.stderr.lower():
            return None
        raise RuntimeError(f"{' '.join(result.args)} failed: {result.stderr.strip()}")

    def _write(self, lease: dict[str, Any], create: bool) -> dict[str, Any]:
        action = ("create", "-f", "-", "-o", "json") if create else ("replace", "-f", "-", "-o", "json")
        return json.loads(self._command(*action, input=json.dumps(lease)))

    def _record(
        self,
        lease: dict[str, Any],
        now: dt.datetime,
        duration: int,
        increment: bool = False,
    ) -> dict[str, Any]:
        spec = lease.setdefault("spec", {})
        if increment:
            spec["leaseTransitions"] = int(spec.get("leaseTransitions", 0)) + 1
        spec["holderIdentity"] = self.holder
        spec["leaseDurationSeconds"] = duration
        spec["renewTime"] = rfc3339(now)
        spec.setdefault("acquireTime", rfc3339(now))
        return lease

    def acquire(self) -> int:
        now = utc_now()
        current = self._get()
        if current is None:
            candidate = {
                "apiVersion": "coordination.k8s.io/v1",
                "kind": "Lease",
                "metadata": {"name": self.name, "namespace": self.namespace},
                "spec": {},
            }
            try:
                acquired = self._write(self._record(candidate, now, self.duration_seconds, increment=True), create=True)
            except RuntimeError as error:
                raise RuntimeError(f"publisher Lease {self.name!r} is already being acquired") from error
        else:
            if not is_expired(current, now):
                holder = current.get("spec", {}).get("holderIdentity", "unknown")
                raise RuntimeError(f"publisher Lease {self.name!r} is held by {holder!r}")
            try:
                acquired = self._write(self._record(current, now, self.duration_seconds, increment=True), create=False)
            except RuntimeError as error:
                raise RuntimeError(f"publisher Lease {self.name!r} changed during acquisition") from error
        self._renewer = threading.Thread(target=self._renew_loop, name="lease-renewer", daemon=True)
        self._renewer.start()
        self.sequence = acquired["spec"]["leaseTransitions"]
        return self.sequence

    def _renew_loop(self) -> None:
        interval = max(10, self.duration_seconds // 3)
        while not self._stop.wait(interval):
            try:
                current = self._get()
                if current is None or current.get("spec", {}).get("holderIdentity") != self.holder:
                    raise RuntimeError(f"publisher Lease {self.name!r} was lost")
                self._write(self._record(current, utc_now(), self.duration_seconds), create=False)
            except Exception as error:  # The publisher must not continue after a lost lock.
                self._lost = error
                self._stop.set()

    def assert_held(self) -> None:
        if self._lost is not None:
            raise RuntimeError("publisher Lease renewal failed") from self._lost

    def release(self) -> None:
        self._stop.set()
        if self._renewer is not None:
            self._renewer.join()
        if self._lost is not None:
            return
        current = self._get()
        if current is None or current.get("spec", {}).get("holderIdentity") != self.holder:
            return
        # Conditional replace preserves a later holder if our Lease expired.
        try:
            self._write(self._record(current, utc_now(), 1), create=False)
        except RuntimeError:
            pass


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
    date = args.date or utc_now().strftime("%Y%m%d")
    lease = Lease(args.kubectl, args.namespace, args.lease_name, args.lease_duration)
    sequence = lease.acquire()
    tag = allocate_tag(branch, revision, args.build_revision, date, sequence)
    reference: str
    try:
        # Alongside the output base so the blob hard links stay on one filesystem.
        with tempfile.TemporaryDirectory(dir=image_layout.resolve().parent) as temporary_directory:
            image_refs = Path(temporary_directory) / "image-refs.txt"
            pushable = materialize_layout(image_layout, Path(temporary_directory) / "layout")
            subprocess.run(
                [
                    args.crane,
                    "push",
                    "--image-refs",
                    str(image_refs),
                    str(pushable),
                    f"{args.repository}:{tag}",
                ],
                check=True,
            )
            reference = published_reference(image_refs)
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
    result = json.dumps({"repository": args.repository, "tag": tag, "reference": reference})
    if args.result_file is not None:
        args.result_file.write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
