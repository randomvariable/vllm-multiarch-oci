# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location("publish_vllmb12x", Path(__file__).with_name("publish-vllmb12x.py"))
PUBLISH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISH)


class PublishTagTest(unittest.TestCase):
    def test_source_branch_and_tag_include_both_commits(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.json"
            profile.write_text(json.dumps({"source_ref": "refs/heads/dev/jovian-judgement"}))
            branch = PUBLISH.source_branch(profile)

        self.assertEqual(branch, "dev-jovian-judgement")
        self.assertEqual(
            PUBLISH.allocate_tag(branch, "a" * 40, "b" * 40, "20260914", 13),
            "vllmb12x-dev-jovian-judgement-aaaaaaaaaaaa-bbbbbbbbbbbb-20260914-n13",
        )

    def test_source_branch_rejects_non_branch_ref(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.json"
            profile.write_text(json.dumps({"source_ref": "refs/tags/v1"}))
            with self.assertRaisesRegex(ValueError, "full branch ref"):
                PUBLISH.source_branch(profile)


if __name__ == "__main__":
    unittest.main()
