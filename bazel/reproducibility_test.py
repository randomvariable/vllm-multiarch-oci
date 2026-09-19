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


if __name__ == "__main__":
    unittest.main()
