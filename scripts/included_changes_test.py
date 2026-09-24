# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "vllmb12x_included_changes", Path(__file__).with_name("vllmb12x-included-changes.py")
)
LEDGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEDGER)


class IncludedChangesTest(unittest.TestCase):
    def test_published_ledger_matches_the_lock_and_the_patch_series(self):
        # --check fails when the README table, the site page, or the image
        # description drifts from the ledger, when the recorded patch series
        # stops matching the patches Bazel applies, or when a component claims
        # a branch the lock does not select.
        self.assertEqual(LEDGER.main(["--check"]), 0)

    def test_validate_rejects_a_branch_the_lock_does_not_select(self):
        lock = LEDGER.load_lock()
        for name, key in sorted(LEDGER.BRANCH_CLAIMS.items()):
            self.assertEqual(lock[key], self.ledger_branch(name), name)
            manifest = {
                "components": [
                    {"name": name, "branch": "refs/heads/stale-line", "changes": []},
                ]
            }
            with self.assertRaises(RuntimeError) as raised:
                LEDGER.validate(manifest, lock)
            self.assertIn(f"records {name} branch", str(raised.exception))
            self.assertIn(lock[key], str(raised.exception))

    def ledger_branch(self, name: str) -> str:
        manifest = json.loads(Path(LEDGER.MANIFEST).read_text())
        return next(c["branch"] for c in manifest["components"] if c["name"] == name)


if __name__ == "__main__":
    unittest.main()
