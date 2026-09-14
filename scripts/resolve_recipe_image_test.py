# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("resolve-recipe-image.py")
REPOSITORY = "ghcr.io/randomvariable/vllm-b12x-multi"
DIGEST_A = f"sha256:{'a' * 64}"
DIGEST_B = f"sha256:{'b' * 64}"
OTHER_DIGEST = f"sha256:{'c' * 64}"
TAG_N9 = "vllmb12x-dev-jovian-judgement-aaaaaaaaaaaa-bbbbbbbbbbbb-20260914-n9"
TAG_N10 = "vllmb12x-dev-jovian-judgement-aaaaaaaaaaaa-bbbbbbbbbbbb-20260914-n10"


class ResolverTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.crane = self.directory / "crane"
        self.output = self.directory / "latest-image.json"
        self.state = self.directory / "state.json"
        self.crane.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import json
                import os
                import sys
                from pathlib import Path

                scenario = json.loads(os.environ["FAKE_CRANE_SCENARIO"])
                state_path = Path(os.environ["FAKE_CRANE_STATE"])
                state = json.loads(state_path.read_text()) if state_path.exists() else {"latest": 0}
                command, reference = sys.argv[1:3]

                failure = scenario.get("failure")
                if failure and command == failure["command"]:
                    print(failure["message"], file=sys.stderr)
                    sys.exit(failure.get("code", 1))

                if command == "ls":
                    print("\\n".join(scenario["tags"]))
                elif command == "config":
                    config = scenario.get(
                        "config", {"os": "linux", "architecture": "arm64"}
                    )
                    print(json.dumps(config))
                elif command == "digest" and reference.endswith(":latest"):
                    values = scenario["latest"]
                    index = min(state["latest"], len(values) - 1)
                    print(values[index])
                    state["latest"] += 1
                    state_path.write_text(json.dumps(state))
                elif command == "digest":
                    tag = reference.rsplit(":", 1)[1]
                    if tag not in scenario["digests"]:
                        print(f"unknown tag: {tag}", file=sys.stderr)
                        sys.exit(1)
                    print(scenario["digests"][tag])
                else:
                    print(f"unsupported call: {sys.argv[1:]}", file=sys.stderr)
                    sys.exit(2)
                """
            )
        )
        self.crane.chmod(0o755)

    def run_resolver(self, scenario):
        environment = os.environ.copy()
        environment["FAKE_CRANE_SCENARIO"] = json.dumps(scenario)
        environment["FAKE_CRANE_STATE"] = str(self.state)
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--repository",
                REPOSITORY,
                "--output",
                str(self.output),
                "--crane",
                str(self.crane),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )

    @staticmethod
    def scenario(*, latest=None, tags=None, digests=None, **extra):
        return {
            "latest": latest or [DIGEST_A, DIGEST_A],
            "tags": tags or [TAG_N9, TAG_N10, "latest", "unrelated"],
            "digests": digests or {TAG_N9: DIGEST_A, TAG_N10: DIGEST_A},
            **extra,
        }

    def test_selects_highest_numeric_sequence_for_latest_digest(self):
        result = self.run_resolver(self.scenario(tags=[TAG_N10, TAG_N9, "latest"]))

        self.assertEqual(result.returncode, 0, result.stderr)
        published = json.loads(self.output.read_text())
        self.assertEqual(published["tag"], TAG_N10)
        self.assertEqual(published["reference"], f"{REPOSITORY}@{DIGEST_A}")
        self.assertEqual(published["repository"], REPOSITORY)
        self.assertRegex(
            published["resolved_at"],
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
        )

    def test_retries_entire_resolution_when_latest_changes_once(self):
        result = self.run_resolver(
            self.scenario(
                latest=[DIGEST_A, DIGEST_B, DIGEST_B, DIGEST_B],
                digests={TAG_N9: DIGEST_A, TAG_N10: DIGEST_B},
            )
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        published = json.loads(self.output.read_text())
        self.assertEqual(published["tag"], TAG_N10)
        self.assertEqual(published["reference"], f"{REPOSITORY}@{DIGEST_B}")
        self.assertEqual(json.loads(self.state.read_text())["latest"], 4)

    def test_fails_without_replacing_output_when_latest_changes_twice(self):
        original = '{"preserved": true}\n'
        self.output.write_text(original)
        result = self.run_resolver(
            self.scenario(
                latest=[DIGEST_A, DIGEST_B, DIGEST_B, OTHER_DIGEST],
                digests={TAG_N9: DIGEST_A, TAG_N10: DIGEST_B},
            )
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("changed during both resolution attempts", result.stderr)
        self.assertEqual(self.output.read_text(), original)

    def test_registry_failure_does_not_publish_output(self):
        result = self.run_resolver(
            self.scenario(
                failure={"command": "ls", "message": "registry unavailable", "code": 7}
            )
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("registry unavailable", result.stderr)
        self.assertFalse(self.output.exists())

    def test_rejects_non_arm64_latest(self):
        result = self.run_resolver(
            self.scenario(config={"os": "linux", "architecture": "amd64"})
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected linux/arm64", result.stderr)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
