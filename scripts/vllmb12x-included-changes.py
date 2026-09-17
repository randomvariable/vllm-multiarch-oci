#!/usr/bin/env python3
"""Render the ledger of upstream changes the VLLMB12X lock carries.

The image is built from pinned fork revisions, so neither the pin nor the
patch series says which upstream pull requests reached the build. This command
renders that ledger from `profiles/vllmb12x/included-changes.json` into the
README and the recipe site, checks it against the lock and the patch series,
and can verify each entry against the fork history.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parents[1]
PROFILE: Final = ROOT / "profiles/vllmb12x/profile.json"
MANIFEST: Final = ROOT / "profiles/vllmb12x/included-changes.json"
MODULE: Final = ROOT / "MODULE.bazel"
README: Final = ROOT / "README.md"
SITE: Final = ROOT / "recipes-site/src/content/docs/image-releases/included-changes.mdx"
DESCRIPTION: Final = ROOT / "profiles/vllmb12x/description.bzl"
# MDX rejects HTML comments, so each file carries its own comment syntax.
MARKERS: Final = {
    README: ("<!-- included-changes:start -->", "<!-- included-changes:end -->"),
    SITE: ("{/* included-changes:start */}", "{/* included-changes:end */}"),
}


def pull_request_url(reference: str) -> str:
    repository, number = reference.split("#", 1)
    return f"https://github.com/{repository}/pull/{number}"


def load_lock() -> dict[str, Any]:
    profile = json.loads(PROFILE.read_text())
    return {"source_ref": profile["source_ref"], "sources": profile["sources"]}


def module_patches() -> set[str]:
    """Return the vLLM patch series Bazel applies, as repository paths."""
    text = MODULE.read_text()
    block = re.search(r"vllm_patches\s*=\s*\[(.*?)\]", text, re.S)
    if block is None:
        raise RuntimeError(f"{MODULE} declares no vllm_patches list")
    labels = re.findall(r'"//([^"]+):([^"]+)"', block.group(1))
    return {f"{package}/{name}" for package, name in labels}


def validate(manifest: dict[str, Any], lock: dict[str, Any]) -> None:
    """Fail when the ledger no longer describes what the build actually uses."""
    recorded: set[str] = set()
    for component in manifest["components"]:
        name = component["name"]
        if name not in lock["sources"]:
            raise RuntimeError(f"{MANIFEST} names component {name!r}, which the lock does not pin")
        # The lock records the selected ref for vLLM only, so that is the one
        # branch claim this ledger can contradict.
        if name == "vllm" and component["branch"] != lock["source_ref"]:
            raise RuntimeError(
                f"{MANIFEST} records vLLM branch {component['branch']!r}, but the lock selects {lock['source_ref']!r}"
            )
        for change in component["changes"]:
            if change["inclusion"] != "patch":
                continue
            patch = change["patch"]
            if not (ROOT / patch).is_file():
                raise RuntimeError(f"{MANIFEST} references missing patch {patch}")
            recorded.add(patch)
    applied = module_patches()
    if recorded != applied:
        raise RuntimeError(
            "the recorded patch series and the applied patch series differ: "
            f"recorded only {sorted(recorded - applied)}, applied only {sorted(applied - recorded)}"
        )


def describe(change: dict[str, Any]) -> str:
    if change["inclusion"] == "branch":
        return f"Merged into the pinned revision ({change['commits']} commits)"
    return f"Applied at build time by `{change['patch']}`"


def describe_plainly(change: dict[str, Any]) -> str:
    """Describe the inclusion without markup, for the registry description."""
    if change["inclusion"] == "branch":
        return "merged into the pinned revision"
    return f"applied at build time from {change['patch']}"


def render_readme(manifest: dict[str, Any], lock: dict[str, Any]) -> str:
    lines = [
        "The image is built from pinned fork revisions rather than from upstream branches, so this table records which upstream changes the current lock carries. Regenerate it with `scripts/vllmb12x-included-changes.py`.",
        "",
        "| Component | Change | Included as |",
        "| --- | --- | --- |",
    ]
    for component in manifest["components"]:
        pin = lock["sources"][component["name"]]["commit"][:12]
        for change in component["changes"]:
            reference = change["pull_request"]
            change_text = (
                f"[{reference}]({pull_request_url(reference)}) {change['title']}"
                if reference
                else f"{change['title']} (no upstream pull request)"
            )
            lines.append(f"| {component['title']} `{pin}` | {change_text} | {describe(change)} |")
    return "\n".join(lines)


def render_site(manifest: dict[str, Any], lock: dict[str, Any]) -> str:
    lines: list[str] = []
    for component in manifest["components"]:
        source = lock["sources"][component["name"]]
        lines += [
            f"## {component['title']}",
            "",
            f"Pinned at [`{source['commit'][:12]}`]({source['remote'].removesuffix('.git')}/commit/{source['commit']}), "
            f"branched from [`{component['upstream']}`](https://github.com/{component['upstream']}) "
            f"`{component['upstream_base'].removeprefix('refs/heads/')}`.",
            "",
        ]
        for change in component["changes"]:
            reference = change["pull_request"]
            heading = (
                f"- **[{reference}]({pull_request_url(reference)})** — {change['title']}"
                if reference
                else f"- **{change['title']}** — no upstream pull request"
            )
            lines.append(heading)
            lines.append(f"  - {describe(change)}.")
            if change.get("note"):
                lines.append(f"  - {change['note']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_description(manifest: dict[str, Any], lock: dict[str, Any]) -> str:
    """Render the image description label.

    Registries show this text as a single truncated line beside the package
    name, so it names the pins and the change numbers and leaves the ledger
    itself to the README that the package page renders below it. Without any
    value the page inherits the CUDA base image's own Ubuntu description.
    """
    pins = " + ".join(
        f"{component['title']} {lock['sources'][component['name']]['commit'][:12]}"
        for component in manifest["components"]
    )
    # Grouped by repository: the page truncates this line, so the numbers
    # should not be spent repeating a repository name.
    grouped: dict[str, list[str]] = {}
    for component in manifest["components"]:
        for change in component["changes"]:
            if not change["pull_request"]:
                continue
            repository, number = change["pull_request"].split("#", 1)
            grouped.setdefault(repository, []).append(f"#{number}")
    numbers = ", ".join(
        f"{repository}{' '.join(references)}" for repository, references in grouped.items()
    )
    body = (
        f"{pins} for NVIDIA DGX Spark (GB10, sm_121a), carrying {numbers}. "
        "Linux ARM64, CUDA 13.3.1, Python 3.12. "
        "Full change ledger in the README below."
    )
    if "\n" in body:
        # rules_oci writes labels as key=value lines, so a newline here becomes
        # a second, bogus label rather than a second line of this one.
        raise RuntimeError("the image description must be a single line")
    return "".join((
        "# SPDX-License-Identifier: Apache-2.0\n",
        "# Generated by scripts/vllmb12x-included-changes.py. Do not edit by hand.\n",
        '"""Registry description for the image this profile builds."""\n\n',
        "VLLM_IMAGE_DESCRIPTION = ",
        json.dumps(body),
        "\n",
    ))


def splice(path: Path, body: str) -> str:
    text = path.read_text()
    opening, closing = MARKERS[path]
    start = text.index(opening) + len(opening)
    end = text.index(closing)
    return text[:start] + "\n" + body.rstrip() + "\n" + text[end:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail instead of writing when a file is stale")
    parser.add_argument("--verify", action="store_true", help="Check every entry against the fork history on GitHub")
    parser.add_argument("--clone-root", type=Path, help="Directory holding component clones used by --verify")
    arguments = parser.parse_args(argv)

    manifest = json.loads(MANIFEST.read_text())
    lock = load_lock()
    validate(manifest, lock)

    if arguments.verify:
        return verify(manifest, lock, arguments.clone_root)

    stale: list[Path] = []
    rendered = (
        (README, splice(README, render_readme(manifest, lock))),
        (SITE, splice(SITE, render_site(manifest, lock))),
        (DESCRIPTION, render_description(manifest, lock)),
    )
    for path, updated in rendered:
        if path.exists() and updated == path.read_text():
            continue
        if arguments.check:
            stale.append(path)
            continue
        path.write_text(updated)
        print(f"updated {path.relative_to(ROOT)}")
    if stale:
        for path in stale:
            print(f"stale: {path.relative_to(ROOT)}", file=sys.stderr)
        print("run scripts/vllmb12x-included-changes.py to regenerate", file=sys.stderr)
        return 1
    return 0


def git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(["git", "-C", str(repository), *arguments], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout


def verify(manifest: dict[str, Any], lock: dict[str, Any], clone_root: Path | None) -> int:
    """Check that each branch entry's commits are present in the pinned revision.

    The fork integrates a pull request by replaying its commits, so the head
    SHA is not an ancestor of the pin. Subjects survive the replay, and the
    commit count catches a partially replayed pull request.
    """
    if clone_root is None:
        raise SystemExit("--verify needs --clone-root")
    failures = 0
    for component in manifest["components"]:
        source = lock["sources"][component["name"]]
        repository = clone_root / component["name"]
        if not (repository / ".git").is_dir():
            repository.mkdir(parents=True, exist_ok=True)
            git(repository.parent, "clone", "--filter=blob:none", "--no-checkout", source["remote"], str(repository))
        git(repository, "fetch", "--quiet", source["remote"], source["commit"])
        git(repository, "fetch", "--quiet", f"https://github.com/{component['upstream']}.git", component["upstream_base"])
        base = git(repository, "rev-parse", "FETCH_HEAD").strip()
        carried = set(git(repository, "log", "--format=%s", f"{base}..{source['commit']}").splitlines())
        for change in component["changes"]:
            reference = change["pull_request"]
            if change["inclusion"] != "branch" or not reference:
                continue
            owner, number = reference.split("#", 1)
            listed = json.loads(
                subprocess.run(
                    ["gh", "api", f"repos/{owner}/pulls/{number}/commits", "--jq",
                     "[.[] | .commit.message | split(\"\\n\")[0]]"],
                    text=True, capture_output=True, check=True,
                ).stdout
            )
            missing = [subject for subject in listed if subject not in carried]
            if missing or len(listed) != change["commits"]:
                failures += 1
                print(f"{reference}: {len(listed) - len(missing)}/{len(listed)} commits present, "
                      f"ledger records {change['commits']}", file=sys.stderr)
                for subject in missing:
                    print(f"  missing: {subject}", file=sys.stderr)
            else:
                print(f"{reference}: {len(listed)} commits present in {source['commit'][:12]}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
