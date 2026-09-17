# SPDX-License-Identifier: Apache-2.0

import datetime as dt
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# The scripts import each other the way they do when run directly, so the
# package directory has to be importable under Bazel's runfiles root too.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ci_lease


def lease_object(holder: str, renewed: dt.datetime, duration: int = 300, transitions: int = 1) -> dict:
    return {
        "apiVersion": "coordination.k8s.io/v1",
        "kind": "Lease",
        "metadata": {"name": "vllmb12x-build", "namespace": "ci"},
        "spec": {
            "holderIdentity": holder,
            "leaseDurationSeconds": duration,
            "leaseTransitions": transitions,
            "renewTime": ci_lease.rfc3339(renewed),
        },
    }


class ExpiryTest(unittest.TestCase):
    def test_a_lapsed_holder_releases_the_lease(self):
        now = ci_lease.utc_now()
        live = lease_object("nightly", now - dt.timedelta(seconds=100))
        lapsed = lease_object("nightly", now - dt.timedelta(seconds=400))

        self.assertFalse(ci_lease.is_expired(live, now))
        self.assertTrue(ci_lease.is_expired(lapsed, now))
        self.assertTrue(ci_lease.is_expired({"spec": {}}, now))


class WaitingAcquireTest(unittest.TestCase):
    def test_the_second_lane_waits_for_the_first_to_finish(self):
        now = ci_lease.utc_now()
        held = lease_object("nightly", now)
        states = [held, held, None]
        lease = ci_lease.Lease("kubectl", "ci", "vllmb12x-build", 300, holder="pull-request")

        with (
            patch.object(lease, "_get", side_effect=lambda: states.pop(0)),
            patch.object(lease, "_write", side_effect=lambda body, create: body),
            patch.object(lease, "_renew_loop"),
        ):
            sequence = lease.acquire_waiting(timeout_seconds=30, poll_seconds=0)

        self.assertEqual(sequence, 1)
        self.assertEqual(states, [])

    def test_waiting_gives_up_and_names_the_holder(self):
        now = ci_lease.utc_now()
        lease = ci_lease.Lease("kubectl", "ci", "vllmb12x-build", 300, holder="pull-request")

        with patch.object(lease, "_get", return_value=lease_object("nightly", now)):
            with self.assertRaises(ci_lease.LeaseHeld) as held:
                lease.acquire_waiting(timeout_seconds=0, poll_seconds=0)

        self.assertEqual(held.exception.holder, "nightly")


class RenewalTest(unittest.TestCase):
    def test_a_transient_api_failure_does_not_abandon_a_live_lease(self):
        lease = ci_lease.Lease("kubectl", "ci", "vllmb12x-build", 300, holder="nightly")
        lease._held_until = ci_lease.utc_now() + dt.timedelta(seconds=300)
        lease._stop.set()  # Run the loop body exactly zero times, then drive it directly.

        with patch.object(lease, "_get", side_effect=RuntimeError("connection refused")):
            lease._stop.clear()
            with patch.object(lease._stop, "wait", side_effect=[False, True]):
                lease._renew_loop()

        self.assertIsNone(lease.lost)

    def test_another_holder_ends_ownership_immediately(self):
        lease = ci_lease.Lease("kubectl", "ci", "vllmb12x-build", 300, holder="nightly")
        lease._held_until = ci_lease.utc_now() + dt.timedelta(seconds=300)

        with patch.object(lease, "_get", return_value=lease_object("pull-request", ci_lease.utc_now())):
            with patch.object(lease._stop, "wait", side_effect=[False, True]):
                lease._renew_loop()

        self.assertIsInstance(lease.lost, ci_lease.LeaseLost)


if __name__ == "__main__":
    unittest.main()
