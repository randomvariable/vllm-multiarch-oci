# SPDX-License-Identifier: Apache-2.0

import csv
import importlib.util
import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

_ACTION_LIB = Path(__file__).with_name("action_lib.py")
if not _ACTION_LIB.is_file():
    _ACTION_LIB = Path(__file__).parents[1] / "bazel" / "action_lib.py"
SPEC = importlib.util.spec_from_file_location("action_lib", _ACTION_LIB)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to load action_lib")
ACTION_LIB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACTION_LIB)

# The launcher data layer builds its fixture tree in its own test module, which
# owns what the layer selects. It is loaded the same way action_lib is so the
# byte-identity checks below describe exactly that layer, not a second copy of it.
_LAYER_FIXTURES = Path(__file__).with_name("launcher_data_layer_test.py")
if not _LAYER_FIXTURES.is_file():
    _LAYER_FIXTURES = Path(__file__).parents[1] / "bazel" / "launcher_data_layer_test.py"
LAYER_SPEC = importlib.util.spec_from_file_location("launcher_data_layer_test", _LAYER_FIXTURES)
if LAYER_SPEC is None or LAYER_SPEC.loader is None:
    raise RuntimeError("unable to load launcher_data_layer_test")
LAYER = importlib.util.module_from_spec(LAYER_SPEC)
LAYER_SPEC.loader.exec_module(LAYER)


class ReproducibilityTests(unittest.TestCase):
    def test_write_tar_normalizes_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            (root / "nested").mkdir(mode=0o700)
            executable = root / "run.sh"
            executable.write_text("#!/bin/sh\n")
            executable.chmod(0o700)
            regular = root / "data"
            regular.write_text("data\n")
            regular.chmod(0o600)

            archive = Path(directory) / "layer.tar"
            ACTION_LIB.write_tar(archive, root)

            with tarfile.open(archive) as contents:
                modes = {entry.name: entry.mode for entry in contents}
            self.assertEqual(modes, {"nested": 0o700, "run.sh": 0o700, "data": 0o600})

    def test_normalize_local_wheel_metadata_repairs_record(self):
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory)
            metadata = site / "demo-1.0.dist-info"
            metadata.mkdir()
            direct_url = metadata / "direct_url.json"
            direct_url.write_text(json.dumps({"url": "file:///scratch/work/action/wheel.whl"}))
            record = metadata / "RECORD"
            with record.open("w", newline="") as contents:
                csv.writer(contents, lineterminator="\n").writerows(
                    [["demo-1.0.dist-info/direct_url.json", "sha256=abc", "3"], ["demo.py", "", ""]]
                )

            ACTION_LIB.normalize_local_wheel_metadata(site)

            self.assertFalse(direct_url.exists())
            self.assertEqual(record.read_text(), "demo.py,,\n")

    def test_compiler_flags_remap_paths_and_fix_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = {"CFLAGS": "-O2", "RUSTFLAGS": "-Copt-level=3"}
            values = ACTION_LIB.configure_reproducible_compilation(
                root / "work", root / "source", env
            )

            self.assertEqual(env["SOURCE_DATE_EPOCH"], "0")
            self.assertIn("--objdir-as-tempdir", values["cuda"])
            self.assertIn("--remap-path-prefix=", env["RUSTFLAGS"])

    def test_nvcc_wrapper_uses_source_relative_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nvcc = root / "nvcc"
            nvcc.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
            nvcc.chmod(0o755)
            work = root / "work"
            wrapper = ACTION_LIB.create_nvcc_wrapper(work, nvcc)
            source = work / "kernel.cu"
            source.parent.mkdir(exist_ok=True)
            source.write_text("__global__ void kernel() {}\n")

            first = subprocess.run(
                [wrapper, "-c", source, "-o", root / "one.o"],
                check=True,
                capture_output=True,
                text=True,
            )
            second = subprocess.run(
                [wrapper, "-c", source, "-o", root / "two.o"],
                check=True,
                capture_output=True,
                text=True,
            )
            first_seed = [
                line for line in first.stdout.splitlines() if line.startswith("--frandom-seed=")
            ]
            second_seed = [
                line for line in second.stdout.splitlines() if line.startswith("--frandom-seed=")
            ]
            self.assertEqual(first_seed, second_seed)


class LauncherDataLayerTarTests(unittest.TestCase):
    # A layer tar is a Bazel build artifact exactly like the wheel and provenance
    # outputs above, so its reproducibility is proven here: an image digest means
    # nothing unless the tar under it is a pure function of its declared inputs.
    # The selection, collision and completeness rules for this layer stay in
    # bazel/launcher_data_layer_test.py.

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.base = Path(self._temporary.name)
        self.recipes, self.hardware = LAYER.make_overlay(self.base)
        self.archive = LAYER.make_source_archive(self.base, LAYER.make_upstream(self.base))

    def tearDown(self):
        self._temporary.cleanup()

    def test_tar_is_byte_identical_across_two_runs(self):
        first = LAYER.run_action(self.base / "one", self.archive, self.recipes, self.hardware)
        second = LAYER.run_action(self.base / "two", self.archive, self.recipes, self.hardware)

        self.assertEqual(first.read_bytes(), second.read_bytes())
        with tarfile.open(first) as contents:
            self.assertEqual(
                {member.name for member in contents if member.isfile()},
                LAYER.expected_paths(),
            )

    def test_tar_does_not_depend_on_checkout_modes(self):
        # write_tar records the on-disk mode, so an operator's umask must not
        # reach the layer digest.
        reference = self.run_with_modes(0o600, 0o700, "strict")
        variant = self.run_with_modes(0o664, 0o775, "loose")

        self.assertEqual(reference.read_bytes(), variant.read_bytes())
        with tarfile.open(reference) as contents:
            members = {member.name: member for member in contents}
        recipe = members[f"{LAYER.ROOT}/recipes/qwen38-27b.yaml"]
        self.assertTrue(recipe.isfile())
        self.assertEqual(recipe.mode, 0o644)
        self.assertEqual(recipe.uid, 0)
        self.assertEqual(recipe.gid, 0)
        self.assertEqual(recipe.mtime, 0)
        directory = members[f"{LAYER.ROOT}/{LAYER.RUNTIME}/profiles"]
        self.assertTrue(directory.isdir())
        self.assertEqual(directory.mode, 0o755)

    def run_with_modes(self, file_mode: int, directory_mode: int, name: str) -> Path:
        # Both overlay trees: the hardware profiles live under //image_tools/data
        # and the recipes under //recipes, and either one's on-disk mode would
        # otherwise reach the layer digest.
        for directory in (LAYER.data_dir(self.base), LAYER.recipes_dir(self.base)):
            for path in directory.rglob("*"):
                path.chmod(directory_mode if path.is_dir() else file_mode)
        return LAYER.run_action(self.base / name, self.archive, self.recipes, self.hardware)


if __name__ == "__main__":
    unittest.main()
