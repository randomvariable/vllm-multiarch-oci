# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch


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

    def test_pull_commits_counts_the_head_against_its_merge_base(self):
        # A head branch cut from an older base reports that base's own commits
        # in the pull request's commit list, so the count must come from the
        # merge-base delta or a complete replay reads as a shortfall.
        calls: list[tuple] = []

        def fake_git(repository, *arguments):
            calls.append(arguments)
            if arguments[0] == "merge-base":
                return "m\n"
            if arguments[0] == "log":
                return "one\nsecond\n"
            return ""

        def fake_run(command, **_kwargs):
            revision = "h" if "head.sha" in command[-1] else "b"
            return subprocess.CompletedProcess(command, 0, revision + "\n", "")

        with patch.object(LEDGER, "git", fake_git), patch.object(
            LEDGER.subprocess, "run", fake_run
        ):
            listed = LEDGER.pull_commits(Path("."), "o/r#1")
        self.assertEqual(listed, ["one", "second"])
        self.assertIn(("merge-base", "b", "h"), calls)
        self.assertIn(("log", "--format=%s", "m..h"), calls)

    def test_pull_commits_measures_against_a_declared_base(self):
        # When upstream rewrites the branch a pull request was cut from, the
        # ledger names that cut point and no base is read from the pull request.
        calls: list[tuple] = []
        declared = "d" * 40

        def fake_git(repository, *arguments):
            calls.append(arguments)
            if arguments[0] == "merge-base":
                return declared + "\n"
            if arguments[0] == "log":
                return "only\n"
            return ""

        def fake_run(command, **_kwargs):
            return subprocess.CompletedProcess(command, 0, "h\n", "")

        with patch.object(LEDGER, "git", fake_git), patch.object(
            LEDGER.subprocess, "run", fake_run
        ):
            listed = LEDGER.pull_commits(Path("."), "o/r#1", declared)
        self.assertEqual(listed, ["only"])
        self.assertIn(("merge-base", declared, "h"), calls)
        self.assertNotIn("base.sha", " ".join(" ".join(c) for c in calls))


if __name__ == "__main__":
    unittest.main()
