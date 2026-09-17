# SPDX-License-Identifier: Apache-2.0

import importlib.util
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
        # stops matching the patches Bazel applies, or when the ledger claims
        # a vLLM branch the lock does not select.
        self.assertEqual(LEDGER.main(["--check"]), 0)


if __name__ == "__main__":
    unittest.main()
