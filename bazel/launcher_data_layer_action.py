#!/usr/bin/env python3
"""Install the pinned upstream model-policy data and our recipes as one OCI layer.

The launcher resolves a profile through the upstream
``local-inference-lab/blackwell-llm-docker`` runtime data: ``options.yaml``,
``schema.json``, ``platform-environment.json``, ``presets.yaml``,
``parameter-docs.yaml``, the model profiles, the hardware profiles and the chat
templates. Only that data ships. None of upstream's Python enters the image, so a
launch runs our own resolver against a pinned tree and never imports upstream
code.

Our recipes and hardware profiles merge into the same directory. The two checks
below exist because a merge is where a silent breakage would otherwise appear:
an upstream rename would reach the image as a missing file and only fail when a
pod launches, and a file we own that shadows an upstream file of the same name
would leave the merged tree without an unambiguous provenance. Both fail here, at
build time.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).parent))

from action_lib import extract, work_root, write_tar

# The layer owns this directory outright: no wheel, package or base file writes
# below it, so its position in the image's layer order cannot be shadowed.
INSTALL_ROOT = "opt/vllm-image"
RUNTIME = "runtime"
RECIPES = "recipes"

# Every upstream file the resolver reads. Selection is an allowlist rather than a
# copy of the tree, so upstream's supervisor, its tests, its generated compose
# files and its requirements stay out of the image.
TOP_LEVEL_FILES = (
    "options.yaml",
    "schema.json",
    "platform-environment.json",
    "presets.yaml",
    "parameter-docs.yaml",
)
YAML_DIRECTORIES = ("profiles", "hardware")
OPEN_DIRECTORIES = ("templates",)

# A rename or removal upstream would otherwise deliver as a launch-time failure.
# ``profiles/common.yaml`` is the merge base rather than a model, so it is
# installed when present but a model profile is what the launcher is asked for.
REQUIRED_TOP_LEVEL_FILES = TOP_LEVEL_FILES
REQUIRED_PROFILES = (
    "glm53-flash.yaml",
    "qwen38-flash-next.yaml",
    "ds4-flash.yaml",
    "ds4-vision.yaml",
    "ds41-flash.yaml",
    "mimo26-flash.yaml",
)


def _relative(name: str) -> str:
    return f"{RUNTIME}/{name}"


def select_upstream_files(upstream: Path) -> dict[str, Path]:
    """Return the upstream policy files to install, keyed below ``runtime/``.

    *upstream* is the extracted pinned source archive, so the policy data sits at
    ``upstream/runtime``.
    """
    runtime = upstream / RUNTIME
    if not runtime.is_dir():
        raise SystemExit(
            f"pinned upstream source has no {RUNTIME}/ directory: {runtime}"
        )

    # Every key carries the `runtime/` prefix, so an installed path is the layer
    # root joined with the key and one directory cannot be confused with another.
    selected: dict[str, Path] = {}
    for name in TOP_LEVEL_FILES:
        path = runtime / name
        if path.is_file():
            selected[_relative(name)] = path
    for directory in YAML_DIRECTORIES:
        for path in sorted((runtime / directory).glob("*.yaml")):
            if path.is_file():
                selected[_relative(f"{directory}/{path.name}")] = path
    for directory in OPEN_DIRECTORIES:
        for path in sorted((runtime / directory).iterdir()):
            if path.is_file():
                selected[_relative(f"{directory}/{path.name}")] = path
    return selected


def verify_selection(selected: Mapping[str, Path]) -> None:
    """Fail when the pinned upstream tree no longer carries what the launcher needs."""
    missing = [name for name in REQUIRED_TOP_LEVEL_FILES if _relative(name) not in selected]
    if missing:
        raise SystemExit(
            f"pinned upstream source is missing required {RUNTIME}/ files: "
            + ", ".join(_relative(name) for name in missing)
        )
    profiles = sorted(
        name for name in selected if name.startswith(_relative("profiles") + "/")
    )
    if not profiles:
        raise SystemExit(
            f"pinned upstream source has no files under {_relative('profiles')}"
        )
    for name in REQUIRED_PROFILES:
        if _relative(f"profiles/{name}") not in selected:
            raise SystemExit(
                f"pinned upstream source is missing required profile "
                f"{_relative(f'profiles/{name}')}"
            )


def install_plan(
    upstream: Path,
    recipes: Sequence[Path],
    hardware: Sequence[Path],
) -> dict[str, Path]:
    """Map every installed layer path to the source file that provides it.

    Keys are relative to the layer root, so the upstream policy data lands under
    ``opt/vllm-image/runtime`` and ``opt/vllm-image/recipes``. Our hardware
    profiles merge into the upstream hardware directory, which is why a name that
    both sides carry is an error rather than an override: the merged tree has to
    keep one provable provenance per path.
    """
    selected = select_upstream_files(upstream)
    verify_selection(selected)

    plan = {
        f"{INSTALL_ROOT}/{name}": path for name, path in selected.items()
    }

    if not recipes:
        raise SystemExit("no launcher recipes to install; check //recipes:all_yaml_files")
    for path in recipes:
        if path.suffix != ".yaml":
            raise SystemExit(f"launcher recipe must be YAML: {path}")
        installed = f"{INSTALL_ROOT}/{RECIPES}/{path.name}"
        # The layer installs one recipe per name, so the name has to identify one
        # file. Two owners shipping the same stem would otherwise reach the image
        # as one silently winning, and `--recipe` resolves by name alone.
        if installed in plan:
            raise SystemExit(
                f"{installed} is declared twice: by {plan[installed]} and by {path}"
            )
        plan[installed] = path

    if not hardware:
        raise SystemExit(
            "no hardware profiles to install; check //image_tools/data:hardware"
        )
    for path in hardware:
        if path.suffix != ".yaml":
            raise SystemExit(f"hardware profile must be YAML: {path}")
        installed = f"{INSTALL_ROOT}/{_relative(f'hardware/{path.name}')}"
        if installed in plan:
            raise SystemExit(
                f"{installed} is provided twice: by our hardware profile {path} and "
                f"by the pinned upstream source {plan[installed]}"
            )
        plan[installed] = path

    return plan


def normalize_tree(root: Path) -> None:
    """Pin the modes ``write_tar`` records, which it takes from disk.

    ``write_tar`` normalises ownership and timestamps but keeps each file mode, so
    a checkout with a different umask would otherwise change the layer digest.
    This layer holds only configuration: no entry is executable, and a directory
    has to be traversable.
    """
    for path in sorted(root.rglob("*")):
        os.chmod(path, 0o755 if path.is_dir() else 0o644)


def build_layer(
    upstream_archive: Path,
    recipes: Sequence[Path],
    hardware: Sequence[Path],
    work: Path,
    output: Path,
) -> None:
    """Extract the pinned source, apply the plan, and write the layer tar."""
    upstream = work / "upstream"
    extract(upstream_archive, upstream)
    plan = install_plan(upstream, recipes, hardware)

    root = work / "root"
    for installed, source in sorted(plan.items()):
        destination = root / installed
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    normalize_tree(root)
    write_tar(output, root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--upstream-source", required=True)
    parser.add_argument("--recipe", action="append", default=[])
    parser.add_argument("--hardware", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    execroot = Path.cwd()
    build_layer(
        execroot / args.upstream_source,
        [execroot / path for path in args.recipe],
        [execroot / path for path in args.hardware],
        work_root(args.work_dir),
        execroot / args.output,
    )


if __name__ == "__main__":
    main()
