# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


# publish-vllmb12x.py imports its sibling ci_lease module the way it does when
# run as a script, so the package directory has to be importable here too.
sys.path.insert(0, str(Path(__file__).resolve().parent))
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


class PullRequestTagTest(unittest.TestCase):
    def test_tag_is_derived_from_the_two_revisions(self):
        self.assertEqual(
            PUBLISH.pull_request_tag(26, "a" * 40, "b" * 40),
            "pr-26-aaaaaaaaaaaa-bbbbbbbbbbbb",
        )

    def test_tag_rejects_an_invalid_pull_request_or_revision(self):
        with self.assertRaisesRegex(ValueError, "pull request number"):
            PUBLISH.pull_request_tag(0, "a" * 40, "b" * 40)
        with self.assertRaisesRegex(ValueError, "builder revision"):
            PUBLISH.pull_request_tag(26, "a" * 40, "not-a-commit")


if __name__ == "__main__":
    unittest.main()
