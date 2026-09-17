#!/usr/bin/env python3
"""Run a command while holding the vLLMB12X build Lease.

The nightly lane and the pull-request lane build on the same node, against the
same remote worker pool and the same ccache volume. Running both at once does
not corrupt anything, but it halves the workers each build sees and evicts the
other lane's ccache entries, so they take turns instead.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
from typing import Final

from ci_lease import Lease, LeaseHeld

DEFAULT_LEASE_SECONDS: Final = 300


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Lease name serializing the build lanes")
    parser.add_argument("--namespace", default="ci")
    parser.add_argument("--holder", help="Lease holder identity, defaults to a fresh UUID")
    parser.add_argument("--kubectl", default="kubectl")
    parser.add_argument(
        "--lease-duration",
        type=int,
        default=DEFAULT_LEASE_SECONDS,
        help="Seconds a cancelled run can block the other lane before its Lease expires",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=8 * 60 * 60,
        help="Seconds to wait for the other lane to finish before failing",
    )
    parser.add_argument("--poll", type=float, default=30.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")

    lease = Lease(args.kubectl, args.namespace, args.name, args.lease_duration, args.holder)
    try:
        sequence = lease.acquire_waiting(args.wait, args.poll)
    except LeaseHeld as held:
        print(f"{args.name}: still held by {held.holder!r} after {args.wait:.0f}s", file=sys.stderr)
        return 1
    print(f"{args.name}: acquired as {lease.holder!r} (transition {sequence})", flush=True)

    child = subprocess.Popen(command)

    def forward(number: int, _frame: object) -> None:
        # Tekton cancels a TaskRun by signalling the step. Pass it on so the
        # build stops and the Lease is released instead of expiring.
        child.send_signal(number)

    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, forward)

    try:
        status = child.wait()
    finally:
        lease.release()
    if lease.lost is not None:
        # Another lane may have started building against the same workers.
        print(f"{args.name}: lease lost during the command: {lease.lost}", file=sys.stderr)
        return status or 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
