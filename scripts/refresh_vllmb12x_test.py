# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, patch


SPEC = importlib.util.spec_from_file_location("refresh_vllmb12x", Path(__file__).with_name("refresh-vllmb12x.py"))
REFRESH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFRESH)


class RefreshVLLMB12XTest(unittest.TestCase):
    def test_dry_run_selects_both_fork_sources_without_writing_lock(self):
        initial = {
            "name": "vllmb12x",
            "source_ref": "refs/heads/dev/jovian-judgement",
            "sources": {
                "b12x": {"commit": "a" * 40, "remote": "https://example.invalid/b12x.git"},
                "vllm": {"commit": "b" * 40, "remote": "https://example.invalid/vllm.git"},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile.json"
            version = root / "version.bzl"
            profile.write_text(json.dumps(initial))
            version.write_text("unchanged\n")
            output = io.StringIO()
            args = [
                "refresh-vllmb12x.py",
                "--vllm-remote", "https://github.com/randomvariable/vllm.git",
                "--vllm-ref", "refs/heads/dev/rv-jovian-judgement",
                "--vllm-commit", "c" * 40,
                "--b12x-remote", "https://github.com/randomvariable/b12x.git",
                "--b12x-ref", "refs/heads/dev/rv-jovian-judgement",
                "--b12x-commit", "d" * 40,
                "--dry-run",
            ]
            resolved = ["c" * 40, "d" * 40, "c" * 40, "d" * 40]
            with (
                patch.object(REFRESH, "PROFILE", profile),
                patch.object(REFRESH, "VERSION", version),
                patch.object(REFRESH, "CMAKE_SOURCES", {}),
                patch.object(REFRESH, "resolve_ref", side_effect=resolved),
                patch.object(REFRESH, "checkout") as checkout,
                patch.object(sys, "argv", args),
                contextlib.redirect_stdout(output),
            ):
                REFRESH.main()

            emitted = json.loads(output.getvalue().split("\n# SPDX", 1)[0])
            self.assertEqual(emitted["source_ref"], "refs/heads/dev/rv-jovian-judgement")
            self.assertEqual(emitted["sources"]["vllm"], {
                "commit": "c" * 40,
                "remote": "https://github.com/randomvariable/vllm.git",
            })
            self.assertEqual(emitted["sources"]["b12x"], {
                "commit": "d" * 40,
                "remote": "https://github.com/randomvariable/b12x.git",
            })
            checkout.assert_called_once_with(
                "https://github.com/randomvariable/vllm.git", "c" * 40, ANY
            )
            self.assertEqual(json.loads(profile.read_text()), initial)
            self.assertEqual(version.read_text(), "unchanged\n")


class CanonicalSourceRefTest(unittest.TestCase):
    def test_cmake_source_inventory_covers_deepselect(self):
        self.assertEqual(
            REFRESH.CMAKE_SOURCES["vllm_cmake_deepselect"],
            (
                "cmake/external_projects/deepselect.cmake",
                "https://github.com/vllm-project/DeepSelect.git",
                "GIT_TAG",
            ),
        )

    def test_lock_records_a_branch_ref_or_a_commit(self):
        self.assertEqual(
            REFRESH.canonical_source_ref("dev/rv-jovian-judgement-profile-base"),
            "refs/heads/dev/rv-jovian-judgement-profile-base",
        )
        self.assertEqual(
            REFRESH.canonical_source_ref("refs/heads/dev/karmic-kraken"),
            "refs/heads/dev/karmic-kraken",
        )
        self.assertEqual(REFRESH.canonical_source_ref("c" * 40), "c" * 40)
        with self.assertRaisesRegex(RuntimeError, "branch or a commit"):
            REFRESH.canonical_source_ref("refs/tags/v1")


if __name__ == "__main__":
    unittest.main()
