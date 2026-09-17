#!/usr/bin/env python3
"""A Kubernetes Lease used as a mutex by the vLLMB12X CI lanes.

Two lanes share one build node, one remote worker pool, and one ccache volume,
and the publication lane additionally mutates registry tags. Both serialize
through this Lease rather than through repository-local state, because a
PipelineRun that is cancelled mid-build leaves nothing behind to clean up: the
Lease simply expires.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import threading
import time
import uuid
from typing import Any, Final

_MINIMUM_DURATION: Final = 30


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


class LeaseHeld(RuntimeError):
    """The Lease is held by a live holder."""

    def __init__(self, name: str, holder: str) -> None:
        super().__init__(f"Lease {name!r} is held by {holder!r}")
        self.holder = holder


class LeaseLost(RuntimeError):
    """The Lease is definitively gone: deleted, or taken by another holder."""


class Lease:
    """A Kubernetes Lease with optimistic-concurrency renewal.

    The Lease is released by shortening its duration instead of deleting it.
    A delayed process cannot delete a Lease acquired by a later holder.
    """

    def __init__(
        self,
        kubectl: str,
        namespace: str,
        name: str,
        duration_seconds: int,
        holder: str | None = None,
    ) -> None:
        if duration_seconds < _MINIMUM_DURATION:
            raise ValueError(f"lease duration must be at least {_MINIMUM_DURATION} seconds")
        self.kubectl = kubectl
        self.namespace = namespace
        self.name = name
        self.duration_seconds = duration_seconds
        self.holder = holder or f"vllmb12x-{uuid.uuid4()}"
        self._stop = threading.Event()
        self._lost: Exception | None = None
        self._renewer: threading.Thread | None = None
        self._held_until: dt.datetime | None = None
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
                raise RuntimeError(f"Lease {self.name!r} is already being acquired") from error
        else:
            if not is_expired(current, now):
                raise LeaseHeld(self.name, current.get("spec", {}).get("holderIdentity", "unknown"))
            try:
                acquired = self._write(self._record(current, now, self.duration_seconds, increment=True), create=False)
            except RuntimeError as error:
                raise RuntimeError(f"Lease {self.name!r} changed during acquisition") from error
        self._held_until = now + dt.timedelta(seconds=self.duration_seconds)
        self._renewer = threading.Thread(target=self._renew_loop, name="lease-renewer", daemon=True)
        self._renewer.start()
        self.sequence = acquired["spec"]["leaseTransitions"]
        return self.sequence

    def acquire_waiting(self, timeout_seconds: float, poll_seconds: float = 15.0) -> int:
        """Acquire the Lease, waiting for a live holder to finish or expire."""
        deadline = time.monotonic() + timeout_seconds
        reported = ""
        while True:
            try:
                return self.acquire()
            except LeaseHeld as held:
                if time.monotonic() >= deadline:
                    raise
                if held.holder != reported:
                    reported = held.holder
                    print(f"waiting for Lease {self.name!r} held by {held.holder!r}", flush=True)
            if self._stop.wait(poll_seconds):
                raise RuntimeError(f"interrupted while waiting for Lease {self.name!r}")

    def _renew_loop(self) -> None:
        # A single failed renewal is not proof of a lost Lease: the API server
        # may be briefly unavailable. Give up only once another holder appears
        # or the Lease we last wrote could actually have expired.
        interval = max(10, self.duration_seconds // 3)
        while not self._stop.wait(interval):
            try:
                current = self._get()
                if current is None:
                    raise LeaseLost(f"Lease {self.name!r} was deleted")
                if current.get("spec", {}).get("holderIdentity") != self.holder:
                    raise LeaseLost(f"Lease {self.name!r} was taken by another holder")
                self._write(self._record(current, utc_now(), self.duration_seconds), create=False)
                self._held_until = utc_now() + dt.timedelta(seconds=self.duration_seconds)
            except LeaseLost as error:
                self._lost = error
                self._stop.set()
            except Exception as error:
                if self._held_until is None or utc_now() >= self._held_until:
                    self._lost = error
                    self._stop.set()
                else:
                    print(f"Lease {self.name!r} renewal failed, retrying: {error}", flush=True)

    @property
    def lost(self) -> Exception | None:
        """The failure that ended this holder's ownership, if any."""
        return self._lost

    def assert_held(self) -> None:
        if self._lost is not None:
            raise RuntimeError(f"Lease {self.name!r} renewal failed") from self._lost

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
