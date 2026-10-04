"""Where the pinned upstream policy tree lives, and what its absence means.

``image_tools/launcher_test.py``, ``launcher_data_test.py`` and
``launcher_parity_test.py`` all need the same three things: the checkout
``LIL_RUNTIME_CHECKOUT`` points at (the developer clone when it is unset), the
commit ``profiles/vllmb12x/profile.json`` pins, and a decision about what a
missing tree is. One module owns that, because a second literal commit would
go stale in silence -- which is exactly how the parity gate became a skip --
and because "skip" and "fail" have to mean the same thing in every lane.

The pin belongs to the profile alone: ``scripts/refresh-vllmb12x.py`` moves it
together with the vLLM and B12X pins, so the profile is the only file that can
say what the launcher is a port of.

``LIL_RUNTIME_REQUIRED`` is what turns a skip into a failure. The
``launcher-tests`` job of ``.github/workflows/recipes-pages.yaml`` sets it, and
that job is the only place the pinned tree is fetched: a gate that has only
ever passed by skipping never caught the drift it exists to catch. A
workstation run without the switch is the one case allowed to skip, with a
message naming the path and the commit to clone.

This is a test-support module: it is not in ``//image_tools:launcher_sources``,
so nothing here reaches the image.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import unittest
from pathlib import Path

IMAGE_TOOLS = Path(__file__).resolve().parent
REPOSITORY_ROOT = IMAGE_TOOLS.parent

DEFAULT_CHECKOUT = Path(
    "/home/naadir/go/src/github.com/local-inference-lab/blackwell-llm-docker"
)
UPSTREAM_CHECKOUT = Path(os.environ.get("LIL_RUNTIME_CHECKOUT") or DEFAULT_CHECKOUT)
UPSTREAM_RUNTIME = UPSTREAM_CHECKOUT / "runtime"

PROFILE = REPOSITORY_ROOT / "profiles" / "vllmb12x" / "profile.json"
LIL_RUNTIME = json.loads(PROFILE.read_text())["sources"]["lil_runtime"]
UPSTREAM_REMOTE = LIL_RUNTIME["remote"]
UPSTREAM_COMMIT = LIL_RUNTIME["commit"]

# Where the absence of the tree is a defect rather than a convenience.
REQUIRED = os.environ.get("LIL_RUNTIME_REQUIRED", "").strip().lower() in (
    "1",
    "true",
    "yes",
)


def _revision() -> str | None:
    """HEAD of the checkout, or None when it cannot be read."""
    result = subprocess.run(
        ["git", "-C", str(UPSTREAM_CHECKOUT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


class UpstreamTree:
    """The pinned tree, as one host-lane module needs it.

    ``required_files`` are the files under ``runtime/`` a module imports or
    reads directly: a tree that carries the commit but not the file the module
    is about to open is not usable, so presence is asked per module rather than
    globally.
    """

    def __init__(self, *required_files: str) -> None:
        self.present = all(
            (UPSTREAM_RUNTIME / name).is_file() for name in required_files
        )
        # Only a tree that is there is asked for its commit: an absent checkout
        # is reported by its own reason, not as a mismatch. A tree whose commit
        # cannot be read is unverified, and an unverified tree is what these
        # gates exist to rule out, so a missing ``git`` is a failure not a pass.
        self.head = _revision() if self.present else None
        self.mismatch = self.present and self.head != UPSTREAM_COMMIT
        self.available = self.present and not self.mismatch

    @property
    def skip_reason(self) -> str:
        return (
            f"pinned upstream checkout is absent: {UPSTREAM_CHECKOUT} @ "
            f"{UPSTREAM_COMMIT} (clone {UPSTREAM_REMOTE} at that commit, or "
            "point LIL_RUNTIME_CHECKOUT at it)"
        )

    @property
    def required_reason(self) -> str:
        return (
            f"{self.skip_reason}; LIL_RUNTIME_REQUIRED says this lane must have "
            "it, so an absent checkout is a broken gate and not a passing one"
        )

    @property
    def mismatch_reason(self) -> str:
        return (
            f"{UPSTREAM_CHECKOUT} is at "
            f"{self.head or 'an unreadable commit'}, not the pinned "
            f"{UPSTREAM_COMMIT} from profiles/vllmb12x/profile.json: comparing "
            "the launcher against a tree that is not the pin proves nothing"
        )

    def barrier(self) -> BaseException:
        """What the gate raises when the pinned tree is not usable.

        An ``AssertionError`` wherever the checkout is at the wrong commit or
        the lane declared it mandatory -- both mean the comparison silently
        stopped happening -- and a ``SkipTest`` only for the developer who
        genuinely has not cloned upstream yet.
        """
        if self.mismatch:
            return AssertionError(self.mismatch_reason)
        if REQUIRED:
            return AssertionError(self.required_reason)
        return unittest.SkipTest(self.skip_reason)

    def gate(self, item):
        """Gate one case, or a whole class of them, on the pinned checkout."""
        if self.available:
            return item
        if isinstance(item, type):

            def setUpClass(cls):
                raise self.barrier()

            item.setUpClass = classmethod(setUpClass)
            return item

        @functools.wraps(item)
        def wrapped(test_case, *args, **kwargs):
            # Not `self`: the first parameter is the TestCase instance, and it
            # would shadow the tree whose barrier this raises.
            raise self.barrier()

        return wrapped
