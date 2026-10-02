# SPDX-License-Identifier: Apache-2.0
"""Run the shared CI lane script against stub tools.

Every lane of the vllmb12x build takes hours, so the dispatch logic that is new
here gets proved locally instead: the assertions are on what each lane actually
asks its tools for, in order, with the real environment contract of the script.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LANE_SCRIPT = ROOT / "scripts" / "ci" / "vllmb12x-build-and-publish.sh"
BUILDER_REVISION = "b" * 40


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
        self.trace = self.directory / "trace.jsonl"
        self.bin = self.directory / "bin"
        self.bin.mkdir()
        self.stub("bazel", "log(sys.argv[1:])")
        self.stub("python3", "log(sys.argv[1:])")
        self.stub(
            "git",
            "if '--show-toplevel' in sys.argv:\n"
            "    print(os.environ['CHECKPOINT'])\n"
            "else:\n"
            "    print(os.environ['BUILDER_REVISION'])",
        )

    def stub(self, name: str, body: str) -> None:
        script = self.bin / name
        script.write_text(
            "#!" + sys.executable + "\n"
            "import json, os, sys\n"
            "def log(arguments):\n"
            "    with open(os.environ['TRACE'], 'a') as handle:\n"
            "        handle.write(json.dumps(arguments) + '\\n')\n"
            + body
            + "\n"
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
        if not self.trace.exists():
            return []
        return [json.loads(line) for line in self.trace.read_text().splitlines()]

    def publisher_command(self) -> list[str]:
        matches = [
            command
            for command in self.commands()
            if command[:1] == ["scripts/publish-vllmb12x.py"]
        ]
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
        self.run_lane(LANE="pull-request", PULL_REQUEST="41")
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
