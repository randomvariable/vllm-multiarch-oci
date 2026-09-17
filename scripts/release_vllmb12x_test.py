# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parent))
SPEC = importlib.util.spec_from_file_location(
    "release_vllmb12x", Path(__file__).with_name("release-vllmb12x.py")
)
RELEASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELEASE)


class ReleaseTagTest(unittest.TestCase):
    def test_sequence_continues_within_a_day_and_restarts_on_the_next(self):
        tags = ["v20260916.1", "v20260917.1", "v20260917.2", "not-a-release"]

        self.assertEqual(RELEASE.next_release_tag(tags, "20260917"), "v20260917.3")
        self.assertEqual(RELEASE.next_release_tag(tags, "20260918"), "v20260918.1")
        self.assertEqual(RELEASE.next_release_tag([], "20260918"), "v20260918.1")


class ReleaseNotesTest(unittest.TestCase):
    def test_includes_mooncake_base_image_addition(self):
        module = RELEASE.ledger()
        reference = "example/image@sha256:" + "0" * 64

        with patch.object(RELEASE, "run", return_value="0123456789ab" + "0" * 28 + "\n"):
            notes = RELEASE.release_notes(reference, "vllmb12x-dev-abcdefabcdef-0123456789ab-20260917-n1", "v20260917.1", module)

        self.assertIn("## Base vLLM image additions", notes)
        self.assertIn("Mooncake Transfer Engine CUDA 13 `0.3.13.post1`", notes)


class ReleaseCandidateTest(unittest.TestCase):
    def setUp(self):
        self.module = RELEASE.ledger()
        lock = self.module.load_lock()
        manifest = __import__("json").loads(self.module.MANIFEST.read_text())
        self.labels = {
            "uk.co.randomvariable.vllmb12x.vllm-revision": lock["sources"]["vllm"]["commit"],
            "uk.co.randomvariable.vllmb12x.vllm-source-ref": lock["source_ref"],
            "org.opencontainers.image.description": self.module.description_body(manifest, lock),
        }
        self.reference = "example/image@sha256:" + "0" * 64
        self.publication_tag = "vllmb12x-dev-branch-abcdefabcdef-0123456789ab-20260917-n57"

    def check(self, labels=None, tag=None):
        RELEASE.check_release_candidate(self.reference, tag or self.publication_tag, labels or self.labels, self.module)

    def test_an_image_from_another_lock_is_rejected(self):
        stale = dict(self.labels, **{"uk.co.randomvariable.vllmb12x.vllm-revision": "0" * 40})

        with self.assertRaisesRegex(RuntimeError, "does not match this tree"):
            self.check(labels=stale)

    def test_an_image_advertising_a_different_ledger_is_rejected(self):
        drifted = dict(self.labels, **{"org.opencontainers.image.description": "something else"})

        with self.assertRaisesRegex(RuntimeError, "does not match this tree"):
            self.check(labels=drifted)

    def test_an_image_built_from_an_earlier_commit_is_rejected(self):
        with patch.object(RELEASE, "run", side_effect=["", "f" * 40 + "\n"]):
            with self.assertRaisesRegex(RuntimeError, "was built from 0123456789ab, but HEAD is"):
                self.check()

    def test_a_dirty_tree_is_rejected(self):
        with patch.object(RELEASE, "run", side_effect=["scripts/release-vllmb12x.py\n"]):
            with self.assertRaisesRegex(RuntimeError, "uncommitted changes"):
                self.check()

    def test_a_head_outside_main_is_rejected(self):
        unmerged = ["", "0123456789ab" + "0" * 28 + "\n", "  origin/build/topic\n"]
        with patch.object(RELEASE, "run", side_effect=unmerged):
            with self.assertRaisesRegex(RuntimeError, "not on origin/main"):
                self.check()

    def test_a_head_on_main_is_accepted(self):
        merged = ["", "0123456789ab" + "0" * 28 + "\n", "  origin/build/topic\n  origin/main\n"]
        with patch.object(RELEASE, "run", side_effect=merged):
            self.check()

    def test_a_foreign_publication_tag_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "not a publication tag"):
            self.check(tag="latest")


if __name__ == "__main__":
    unittest.main()
