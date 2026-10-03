# SPDX-License-Identifier: Apache-2.0
"""Run the shared CI lane script against stub tools.

Every lane of the vllmb12x build takes hours, so the dispatch logic that is new
here gets proved locally instead: the assertions are on what each lane actually
asks its tools for, in order, with the real environment contract of the script.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LANE_SCRIPT = ROOT / "scripts" / "ci" / "vllmb12x-build-and-publish.sh"
BUILDER_REVISION = "b" * 40
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
        script = self.bin / name
        script.write_text(
            "#!/usr/bin/env bash\n"
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
            ["bash", str(LANE_SCRIPT)],
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

    def publisher_command(self) -> list[str]:
        matches = [command for command in self.commands() if command[:1] == ["scripts/publish-vllmb12x.py"]]
        self.assertEqual(len(matches), 1, f"expected one publisher call, got {matches!r}")
        return matches[0]

    def test_publication_lane_builds_tests_and_allocates_a_tag_under_lease(self):
        result = self.run_lane(LANE="nightly", PUBLISHER_LEASE="vllmb12x-publish", PUBLISHER_NAMESPACE="ci")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands()
        self.assertEqual(commands[0][:2], ["test", "//scripts:all"])
        self.assertEqual(commands[1][:2], ["build", "//image:vllmb12x"])
        contracts = [command[:2] for command in commands[2:-1]]
        self.assertEqual(contracts, [["test", "//tests/image:vllmb12x_contract"]] * 2)
        self.assertEqual(
            [command[2] for command in commands[2:-1]],
            ["--config=remote-aarch64", "--config=remote-x86_64"],
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


if __name__ == "__main__":
    unittest.main()
