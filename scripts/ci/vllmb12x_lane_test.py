# SPDX-License-Identifier: Apache-2.0
"""Run the shared CI lane script against stub tools.

Every lane of the vllmb12x build takes hours, so the dispatch logic that is new
here gets proved locally instead: the assertions are on what each lane actually
asks its tools for, in order, with the real environment contract of the script.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LANE_SCRIPT = ROOT / "scripts" / "ci" / "vllmb12x-build-and-publish.sh"
BUILDER_REVISION = "b" * 40
BASH = shutil.which("bash")
CALL = "\x01"
ARGUMENT = "\x01ARG "


def scratch() -> Path:
    """Keep test scratch on disk, never in the tmpfs at /tmp."""
    root = Path(os.environ.get("TEST_TMPDIR", Path.home() / ".cache" / "vllm-multiarch-oci-tests"))
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=root))


class LaneScriptTest(unittest.TestCase):
    def setUp(self):
        self.directory = scratch()
        self.checkpoint = self.directory / "checkpoint"
        self.checkpoint.mkdir()
        self.trace = self.directory / "trace"
        self.trace.write_text("")
        self.bin = self.directory / "bin"
        self.bin.mkdir()
        self.stub("bazel")
        self.stub("python3")
        self.stub("cargo")
        # git answers the two queries the lane script asks and records nothing:
        # the lane is defined by what it asks bazel and the publisher for.
        self.stub(
            "git",
            'if [ "${1-}" = rev-parse ] && [ "${2-}" = --show-toplevel ]; then\n'
            '    printf \'%s\\n\' "$CHECKPOINT"\n'
            "else\n"
            '    printf \'%s\\n\' "$BUILDER_REVISION"\n'
            "fi\n",
            log=False,
        )

    def stub(self, name: str, tail: str = "", log: bool = True) -> None:
        # Plain bash, not a Python script with an interpreter shebang: the sandbox
        # this test runs in does not reliably execute sys.executable from a stub.
        # The absolute path also survives the one test that restricts PATH.
        script = self.bin / name
        script.write_text(
            "#!/bin/bash\n"
            + (
                "for argument in \"$@\"; do printf '\\001ARG %s\\n' \"$argument\" >> \"$TRACE\"; done\n"
                "printf '\\001\\n' >> \"$TRACE\"\n"
                if log
                else ""
            )
            + tail
        )
        script.chmod(0o755)

    def run_lane(self, **overrides):
        environment = dict(
            os.environ,
            PATH=f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            TRACE=str(self.trace),
            CHECKPOINT=str(self.checkpoint),
            BUILDER_REVISION=BUILDER_REVISION,
            HOME=str(self.directory / "home"),
            IMAGE_REPOSITORY="registry.example/vllm-b12x-multi",
            REMOTE_CACHE="grpc://cache:50055",
            REMOTE_EXECUTOR="grpc://executor:50062",
            RESULT=str(self.directory / "result.json"),
        )
        environment.update(overrides)
        return subprocess.run(
            [BASH, str(LANE_SCRIPT)],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def commands(self) -> list[list[str]]:
        blocks: list[list[str]] = []
        current: list[str] = []
        for line in self.trace.read_text().splitlines():
            if line == CALL:
                blocks.append(current)
                current = []
            elif line.startswith(ARGUMENT):
                current.append(line[len(ARGUMENT):])
        return blocks

    def positions(self, prefix: list[str]) -> list[int]:
        """Every traced call whose arguments start with `prefix`, in order."""
        return [
            index
            for index, command in enumerate(self.commands())
            if command[: len(prefix)] == prefix
        ]

    def only(self, prefix: list[str]) -> int:
        """The one traced call whose arguments start with `prefix`."""
        positions = self.positions(prefix)
        self.assertEqual(len(positions), 1, f"expected exactly one call {prefix!r}, got {positions!r}")
        return positions[0]

    def publisher_command(self) -> list[str]:
        matches = [command for command in self.commands() if command[:1] == ["scripts/publish-vllmb12x.py"]]
        self.assertEqual(len(matches), 1, f"expected one publisher call, got {matches!r}")
        return matches[0]

    def test_publication_lane_builds_tests_and_allocates_a_tag_under_lease(self):
        result = self.run_lane(LANE="nightly", PUBLISHER_LEASE="vllmb12x-publish", PUBLISHER_NAMESPACE="ci")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands()
        # Selected by content, not by index: a step added to the lane has to
        # report itself as the call no assertion covers, not silently shift
        # these offsets apart.
        scripts_tests = self.only(["test", "//scripts:all"])
        image_build = self.only(["build", "//image:vllmb12x"])
        host_lanes = {
            directory: self.only(["-m", "unittest", "discover", "-s", directory, "-p", "*_test.py"])
            for directory in ("bazel", "image_tools")
        }
        publisher = self.only(["scripts/publish-vllmb12x.py"])
        contracts = self.positions(["test", "//tests/image:vllmb12x_contract"])
        self.assertEqual(len(contracts), 2, f"one contract run per architecture: {contracts!r}")
        lanes = {scripts_tests, image_build, publisher, *contracts, *host_lanes.values()}
        unclassified = sorted(set(range(len(commands))) - lanes)
        self.assertEqual(
            unclassified,
            [],
            f"the lane made calls no assertion covers: {[commands[i] for i in unclassified]}",
        )
        # Everything that can fail cheaply runs before the image build; the
        # per-architecture contract runs stay last but one.
        for test_lane in [scripts_tests, *host_lanes.values()]:
            self.assertLess(test_lane, image_build, "every test lane must precede the image build")
        self.assertLess(image_build, contracts[0], "the image contract needs the built image")
        self.assertLess(contracts[1], publisher, "the contract runs stay last but one")
        self.assertEqual(
            [commands[index][2] for index in contracts],
            ["--config=remote-aarch64", "--config=remote-x86_64"],
        )
        # The fetch of a cargo-vendored source failed in CI because repository
        # rules run with a sanitised PATH, so the lane must declare it.
        bazelrc = (self.checkpoint / ".bazelrc.user").read_text().splitlines()
        self.assertIn(f"startup --output_user_root={self.directory}/home/bazel", bazelrc)
        self.assertIn("build --remote_cache=grpc://cache:50055", bazelrc)
        self.assertIn("build:remote-multiarch --remote_executor=grpc://executor:50062", bazelrc)
        self.assertTrue(
            any(line.startswith("build --repo_env=PATH=") and str(self.bin) in line for line in bazelrc),
            f"repository rules must see the bootstrapped tools: {bazelrc!r}",
        )
        publisher = self.publisher_command()
        self.assertIn("--lease-name", publisher)
        self.assertIn("vllmb12x-publish", publisher)
        self.assertNotIn("--pull-request", publisher)

    def test_release_lane_follows_the_publication_path(self):
        result = self.run_lane(LANE="release", PUBLISHER_LEASE="vllmb12x-publish", PUBLISHER_NAMESPACE="ci")
        self.assertEqual(result.returncode, 0, result.stderr)
        publisher = self.publisher_command()
        self.assertIn("--lease-name", publisher)
        self.assertIn("registry.example/vllm-b12x-multi", publisher)
        position = publisher.index("--build-revision")
        self.assertEqual(publisher[position + 1], BUILDER_REVISION)

    def test_pull_request_lane_publishes_by_number_and_holds_no_lease(self):
        result = self.run_lane(LANE="pull-request", PULL_REQUEST="41")
        self.assertEqual(result.returncode, 0, result.stderr)
        publisher = self.publisher_command()
        self.assertIn("--pull-request", publisher)
        self.assertIn("41", publisher)
        self.assertNotIn("--lease-name", publisher)

    def test_unknown_lane_is_refused_before_anything_runs(self):
        result = self.run_lane(LANE="midnight")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown lane", result.stderr)
        self.assertEqual(self.commands(), [])

    def test_publication_lane_without_a_lease_is_refused(self):
        result = self.run_lane(LANE="release")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PUBLISHER_LEASE", result.stderr)

    def test_the_lane_refuses_without_the_bootstrapped_toolchain(self):
        # A step that runs the bootstrap in a subshell loses its PATH export.
        # That has to fail here, not inside a repository rule mid-build.
        stripped = self.directory / "bin-no-cargo"
        stripped.mkdir()
        for name in ("bazel", "python3", "git"):
            (stripped / name).write_text((self.bin / name).read_text())
            (stripped / name).chmod(0o755)
        result = self.run_lane(
            LANE="nightly", PUBLISHER_LEASE="vllmb12x-publish", PUBLISHER_NAMESPACE="ci", PATH=str(stripped)
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("cargo is not on PATH", result.stderr)
        self.assertEqual(self.commands(), [])


if __name__ == "__main__":
    unittest.main()
