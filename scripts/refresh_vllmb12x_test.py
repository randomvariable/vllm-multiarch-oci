# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib.util
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, patch

SPEC = importlib.util.spec_from_file_location("refresh_vllmb12x", Path(__file__).with_name("refresh-vllmb12x.py"))
REFRESH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFRESH)


FORK_PAIR = {
    "name": "vllmb12x",
    "b12x_ref": "refs/heads/integration/karmic-kraken-beta",
    "source_ref": "refs/heads/cycle/karmic-kraken",
    "vllm_base_version": "0.29.0",
    "sources": {
        "b12x": {"commit": "a" * 40, "remote": "https://example.invalid/b12x.git"},
        "vllm": {"commit": "b" * 40, "remote": "https://example.invalid/vllm.git"},
    },
}


def run_refresh(initial, *flags, resolved=("c" * 40, "d" * 40, "d" * 40, "c" * 40)):
    """Run the refresh in a temporary profile and return its emitted output."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        profile = root / "profile.json"
        version = root / "version.bzl"
        profile.write_text(json.dumps(initial))
        version.write_text("unchanged\n")
        output = io.StringIO()
        written = {}
        with (
            patch.object(REFRESH, "PROFILE", profile),
            patch.object(REFRESH, "VERSION", version),
            patch.object(REFRESH, "CMAKE_SOURCES", {}),
            patch.object(REFRESH, "resolve_ref", side_effect=list(resolved)),
            patch.object(REFRESH, "checkout") as checkout,
            patch.object(sys, "argv", ["refresh-vllmb12x.py", "--dry-run", *flags]),
            contextlib.redirect_stdout(output),
        ):
            REFRESH.main()
        written["output"] = output.getvalue()
        written["profile"] = json.loads(profile.read_text())
        written["version"] = version.read_text()
        written["checkout"] = checkout
        return written


class RefreshVLLMB12XTest(unittest.TestCase):
    def test_bare_refresh_resolves_the_pair_the_manifest_records(self):
        result = run_refresh(FORK_PAIR)

        emitted = json.loads(result["output"].split("\n# SPDX", 1)[0])
        self.assertEqual(emitted["source_ref"], "refs/heads/cycle/karmic-kraken")
        self.assertEqual(emitted["b12x_ref"], "refs/heads/integration/karmic-kraken-beta")
        self.assertEqual(emitted["sources"]["vllm"], {
            "commit": "c" * 40,
            "remote": "https://example.invalid/vllm.git",
        })
        self.assertEqual(emitted["sources"]["b12x"], {
            "commit": "d" * 40,
            "remote": "https://example.invalid/b12x.git",
        })
        result["checkout"].assert_called_once_with(
            "https://example.invalid/vllm.git", "c" * 40, ANY
        )
        # A dry run leaves the lock and the generated version module alone.
        self.assertEqual(result["profile"], FORK_PAIR)
        self.assertEqual(result["version"], "unchanged\n")

        # The version composes the declared base, the cycle the ref names, and a
        # digest the reader can recompute from the emitted manifest.
        build_version = re.search(
            r'VLLM_BUILD_VERSION = "([^"]+)"', result["output"]
        ).group(1)
        expected = REFRESH.sources_digest(emitted["source_ref"], emitted["sources"])[:12]
        self.assertEqual(build_version, f"0.29.0+karmic.kraken.{expected}")

    def test_refresh_refuses_to_move_a_source_without_the_flag(self):
        with self.assertRaisesRegex(RuntimeError, "tracked at .*pass --allow-source-change"):
            run_refresh(
                FORK_PAIR,
                "--vllm-remote", "https://github.com/local-inference-lab/vllm.git",
                "--vllm-ref", "refs/heads/dev/karmic-kraken",
            )
        with self.assertRaisesRegex(RuntimeError, "b12x is tracked at"):
            run_refresh(
                FORK_PAIR,
                "--b12x-remote", "https://github.com/local-inference-lab/b12x.git",
                "--b12x-ref", "refs/heads/master",
            )

    def test_source_change_is_recorded_when_the_flag_asks_for_it(self):
        result = run_refresh(
            FORK_PAIR,
            "--vllm-remote", "https://github.com/randomvariable/vllm.git",
            "--vllm-ref", "refs/heads/cycle/karmic-kraken",
            "--allow-source-change",
        )

        emitted = json.loads(result["output"].split("\n# SPDX", 1)[0])
        self.assertEqual(
            emitted["sources"]["vllm"]["remote"],
            "https://github.com/randomvariable/vllm.git",
        )
        self.assertEqual(emitted["source_ref"], "refs/heads/cycle/karmic-kraken")

    def test_missing_recorded_pair_is_named_rather_than_guessed(self):
        without_ref = json.loads(json.dumps(FORK_PAIR))
        del without_ref["b12x_ref"]
        with self.assertRaisesRegex(RuntimeError, "does not record b12x_ref"):
            run_refresh(without_ref)


class VersionCompositionTest(unittest.TestCase):
    def test_cycle_name_names_the_branch_not_its_namespace(self):
        self.assertEqual(REFRESH.cycle_name("refs/heads/cycle/karmic-kraken"), "karmic-kraken")
        self.assertEqual(REFRESH.cycle_name("refs/heads/dev/jovian-judgement"), "jovian-judgement")
        self.assertEqual(REFRESH.cycle_name("refs/heads/main"), "main")
        with self.assertRaisesRegex(RuntimeError, "must be a branch"):
            REFRESH.cycle_name("c" * 40)

    def test_base_version_prefers_the_requested_value(self):
        self.assertEqual(REFRESH.base_version("0.28.1", None), "0.28.1")
        self.assertEqual(REFRESH.base_version("0.28.1", "0.29.0"), "0.29.0")
        with self.assertRaisesRegex(RuntimeError, "needs vllm_base_version"):
            REFRESH.base_version(None, None)
        with self.assertRaisesRegex(RuntimeError, "must look like"):
            REFRESH.base_version(None, "release-candidate")

    def test_version_composes_base_cycle_and_digest_in_pep440_spelling(self):
        # Packaging rewrites "-" and "_" to "." in a local segment, so the
        # version is built from the spelling the wheel and metadata will carry.
        version = REFRESH.build_version("0.29.0", "karmic-kraken", "a" * 64)
        self.assertEqual(version, "0.29.0+karmic.kraken." + "a" * 12)
        self.assertRegex(version, r"^[0-9]+\.[0-9]+\.[0-9]+\+[a-z0-9.]+\.[0-9a-f]{12}$")
        self.assertEqual(
            REFRESH.build_version("0.29.0", "Cycle_Name", "a" * 64),
            "0.29.0+cycle.name." + "a" * 12,
        )

    def test_sources_digest_tracks_every_pin_and_ignores_order(self):
        sources = {
            "vllm": {"commit": "a" * 40, "remote": "https://example.invalid/vllm.git"},
            "b12x": {"commit": "b" * 40, "remote": "https://example.invalid/b12x.git"},
        }
        digest = REFRESH.sources_digest("refs/heads/cycle/karmic-kraken", sources)
        reordered = {"b12x": sources["b12x"], "vllm": sources["vllm"]}
        self.assertEqual(digest, REFRESH.sources_digest("refs/heads/cycle/karmic-kraken", reordered))

        moved = json.loads(json.dumps(sources))
        moved["vllm"]["commit"] = "c" * 40
        self.assertNotEqual(digest, REFRESH.sources_digest("refs/heads/cycle/karmic-kraken", moved))

        moved_ref = REFRESH.sources_digest("refs/heads/cycle/other-cycle", sources)
        self.assertNotEqual(digest, moved_ref)


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
