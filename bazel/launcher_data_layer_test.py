# SPDX-License-Identifier: Apache-2.0
"""Check what the launcher policy-data layer installs, and refuse a bad merge.

The action is driven here as a module against a fixture tree, exactly as Bazel
runs it from the command line, so a missing upstream file or a shadowed hardware
profile fails this test rather than a pod start-up. The two determinism cases for
the produced tar live in ``bazel/reproducibility_test.py``, which is where this
repository proves every build artifact; the fixture builders below are shared with
it.
"""

import importlib.util
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_ACTION = Path(__file__).with_name("launcher_data_layer_action.py")
if not _ACTION.is_file():
    _ACTION = Path(__file__).parents[1] / "bazel" / "launcher_data_layer_action.py"
SPEC = importlib.util.spec_from_file_location("launcher_data_layer_action", _ACTION)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to load launcher_data_layer_action")
ACTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACTION)

ROOT = ACTION.INSTALL_ROOT
RUNTIME = ACTION.RUNTIME

# The upstream data the launcher resolves a profile from.
TOP_LEVEL = (
    "options.yaml",
    "schema.json",
    "platform-environment.json",
    "presets.yaml",
    "parameter-docs.yaml",
)
PROFILES = (
    "common.yaml",
    "ds4-flash.yaml",
    "ds4-vision.yaml",
    "ds41-flash.yaml",
    "glm53-flash.yaml",
    "mimo26-flash.yaml",
    "qwen38-flash-next.yaml",
)
HARDWARE = ("native.yaml", "rtx-pro-6000-pcie.yaml")
TEMPLATES = ("glm53-flash.jinja", "ds4-vision.jinja")

# Upstream files the layer must never carry: its code, its tests, its generated
# compose files, and its validation notes.
EXCLUDED = (
    "__init__.py",
    "launcher.py",
    "supervisor.py",
    "replica_proxy.py",
    "requirements.txt",
    "README.md",
    "tests/__init__.py",
    "tests/test_launcher.py",
    "tests/x.yaml",
    "generated/parameters.md",
    "generated/qwen38-flash-next.compose.yaml",
    "validation/glm-spark-tp2-external-cache.md",
    "kimi-k3-qsrt/source-overlay/sitecustomize.py",
)

RECIPES = (
    "qwen38-flash-next-gb10-tp2.yaml",
    "deepseek-v4-flash-vision-gb10-tp2.yaml",
    "qwen38-27b.yaml",
)
OUR_HARDWARE = ("gb10-roce.yaml",)


def expected_paths() -> set[str]:
    """Return every path the fixture installs, relative to the layer root."""
    installed = {f"{ROOT}/{RUNTIME}/{name}" for name in TOP_LEVEL}
    installed |= {f"{ROOT}/{RUNTIME}/profiles/{name}" for name in PROFILES}
    installed |= {f"{ROOT}/{RUNTIME}/hardware/{name}" for name in HARDWARE + OUR_HARDWARE}
    installed |= {f"{ROOT}/{RUNTIME}/templates/{name}" for name in TEMPLATES}
    installed |= {f"{ROOT}/recipes/{name}" for name in RECIPES}
    return installed


def write(path: Path, text: str = "value: 1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def make_upstream(base: Path, *, hardware: tuple[str, ...] = HARDWARE) -> Path:
    """Materialise the extracted pinned source archive below *base*."""
    source = base / "source"
    runtime = source / RUNTIME
    for name in TOP_LEVEL:
        write(runtime / name)
    for name in PROFILES:
        write(runtime / "profiles" / name)
    for name in hardware:
        write(runtime / "hardware" / name)
    for name in TEMPLATES:
        write(runtime / "templates" / name)
    for name in EXCLUDED:
        write(runtime / name)
    return source


def data_dir(base: Path) -> Path:
    """Return the directory the fixture lays out like //image_tools/data."""
    return base / "image_tools" / "data"


def make_overlay(base: Path) -> tuple[list[Path], list[Path]]:
    """Materialise our recipe and hardware filegroups below *base*."""
    data = data_dir(base)
    recipes = [write(data / "recipes" / name) for name in RECIPES]
    hardware = [write(data / "hardware" / name) for name in OUR_HARDWARE]
    return recipes, hardware


def make_source_archive(base: Path, source: Path) -> Path:
    """Pack the extracted tree the way the source repository rule does."""
    archive = base / "source.tar"
    with tarfile.open(archive, "w") as output:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                output.add(path, arcname=str(path.relative_to(source)))
    return archive


def run_action(work: Path, archive: Path, recipes: list[Path], hardware: list[Path]) -> Path:
    """Invoke the driver the way the action does: paths from the execroot, cwd there."""
    work.mkdir(parents=True, exist_ok=True)
    output = work / "launcher-data.tar"
    argv = [
        str(_ACTION),
        "--work-dir", "launcher-data.tar.work",
        "--upstream-source", str(archive),
        "--output", str(output),
    ]
    for path in recipes:
        argv += ["--recipe", str(path)]
    for path in hardware:
        argv += ["--hardware", str(path)]

    cwd = Path.cwd()
    os.chdir(work)
    try:
        with patch.object(sys, "argv", argv):
            ACTION.main()
    finally:
        os.chdir(cwd)
    return output


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.base = Path(self._temporary.name)
        self.upstream = make_upstream(self.base)
        self.recipes, self.hardware = make_overlay(self.base)

    def tearDown(self):
        self._temporary.cleanup()

    def test_plan_installs_exactly_the_selected_paths(self):
        plan = ACTION.install_plan(self.upstream, self.recipes, self.hardware)

        self.assertEqual(set(plan), expected_paths())

        # Each path comes from the tree its name claims, not from a rename.
        self.assertEqual(
            plan[f"{ROOT}/{RUNTIME}/profiles/glm53-flash.yaml"],
            self.upstream / RUNTIME / "profiles" / "glm53-flash.yaml",
        )
        self.assertEqual(
            plan[f"{ROOT}/{RUNTIME}/hardware/gb10-roce.yaml"],
            self.base / "image_tools" / "data" / "hardware" / "gb10-roce.yaml",
        )
        self.assertEqual(
            plan[f"{ROOT}/recipes/qwen38-27b.yaml"],
            self.base / "image_tools" / "data" / "recipes" / "qwen38-27b.yaml",
        )

    def test_plan_never_installs_unselected_upstream_paths(self):
        plan = ACTION.install_plan(self.upstream, self.recipes, self.hardware)

        for name in EXCLUDED:
            self.assertNotIn(f"{ROOT}/{RUNTIME}/{name}", plan)
        self.assertEqual([path for path in plan if path.endswith(".py")], [])
        self.assertEqual([path for path in plan if "/tests/" in path], [])
        self.assertEqual([path for path in plan if "/generated/" in path], [])
        self.assertEqual([path for path in plan if "/validation/" in path], [])

    def test_collision_with_upstream_hardware_names_both_files(self):
        # gb10-roce exists upstream in this fixture, so the merged hardware
        # directory would have two candidates for one installed path.
        upstream_copy = write(self.upstream / RUNTIME / "hardware" / "gb10-roce.yaml")
        ours = self.base / "image_tools" / "data" / "hardware" / "gb10-roce.yaml"

        with self.assertRaises(SystemExit) as raised:
            ACTION.install_plan(self.upstream, self.recipes, [ours])

        message = str(raised.exception)
        self.assertIn(f"{ROOT}/{RUNTIME}/hardware/gb10-roce.yaml", message)
        self.assertIn(str(ours), message)
        self.assertIn(str(upstream_copy), message)

    def test_missing_required_profile_is_named(self):
        (self.upstream / RUNTIME / "profiles" / "ds41-flash.yaml").unlink()

        with self.assertRaises(SystemExit) as raised:
            ACTION.install_plan(self.upstream, self.recipes, self.hardware)
        self.assertIn(f"{RUNTIME}/profiles/ds41-flash.yaml", str(raised.exception))

    def test_missing_required_data_file_is_named(self):
        (self.upstream / RUNTIME / "parameter-docs.yaml").unlink()

        with self.assertRaises(SystemExit) as raised:
            ACTION.install_plan(self.upstream, self.recipes, self.hardware)
        self.assertIn(f"{RUNTIME}/parameter-docs.yaml", str(raised.exception))

    def test_empty_profiles_directory_fails(self):
        for path in (self.upstream / RUNTIME / "profiles").iterdir():
            path.unlink()

        with self.assertRaises(SystemExit) as raised:
            ACTION.install_plan(self.upstream, self.recipes, self.hardware)
        self.assertIn(f"{RUNTIME}/profiles", str(raised.exception))

    def test_source_without_runtime_directory_fails(self):
        with self.assertRaises(SystemExit) as raised:
            ACTION.install_plan(self.base / "absent", self.recipes, self.hardware)
        self.assertIn(RUNTIME, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
