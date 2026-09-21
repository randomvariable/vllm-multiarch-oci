#!/usr/bin/env python3
"""Refresh the immutable source lock for the VLLMB12X build profile.

The vLLM branch is moving input only to this command. Bazel consumes the
resulting full commit SHAs and never resolves a branch itself.

The generated version composes three things: the upstream vLLM base version
declared in the manifest, the local-inference-lab cycle the source ref names,
and a digest of the locked source set. The digest stands in for a build date,
so rebuilding identical inputs reproduces the same version, wheel filename and
image label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parents[1]
PROFILE: Final = ROOT / "profiles/vllmb12x/profile.json"
VERSION: Final = ROOT / "profiles/vllmb12x/version.bzl"

# These are every CMake download that the CUDA profile replaces through an
# explicit *_SRC_DIR. A new CMake source declaration must be classified here
# before it can enter the build lock.
CMAKE_SOURCES: Final = {
    "vllm_cmake_cutlass": ("CMakeLists.txt", "https://github.com/nvidia/cutlass.git", "CUTLASS_REVISION"),
    "vllm_cmake_deepgemm": ("cmake/external_projects/deepgemm.cmake", "https://github.com/deepseek-ai/DeepGEMM.git", "_DEEPGEMM_UPSTREAM_TAG"),
    "vllm_cmake_deepselect": ("cmake/external_projects/deepselect.cmake", "https://github.com/vllm-project/DeepSelect.git", "GIT_TAG"),
    "vllm_cmake_qutlass": ("cmake/external_projects/qutlass.cmake", "https://github.com/IST-DASLab/qutlass.git", "_QUTLASS_UPSTREAM_TAG"),
    "vllm_cmake_triton": ("cmake/external_projects/triton_kernels.cmake", "https://github.com/triton-lang/triton.git", "TRITON_KERNELS_TAG"),
    "vllm_cmake_msa": ("cmake/external_projects/fmha_sm100.cmake", "https://github.com/vllm-project/MSA.git", "GIT_TAG"),
    "vllm_cmake_flashmla": ("cmake/external_projects/flashmla.cmake", "https://github.com/vllm-project/FlashMLA.git", "GIT_TAG"),
    "vllm_cmake_flashkda": ("cmake/external_projects/flashkda.cmake", "https://github.com/vllm-project/FlashKDA.git", "GIT_TAG"),
    "vllm_cmake_tml_fa4": ("cmake/external_projects/tml_fa4.cmake", "https://github.com/vllm-project/tml-fa4.git", "GIT_TAG"),
    "vllm_cmake_flash_attn": ("cmake/external_projects/vllm_flash_attn.cmake", "https://github.com/vllm-project/flash-attention.git", "GIT_TAG"),
}


def run(*command: str, cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(
            f"{' '.join(command)} failed with {result.returncode}: {result.stderr.strip()}"
        )
    return result.stdout


def resolve_ref(remote: str, ref: str) -> str:
    lines = [line.split() for line in run("git", "ls-remote", remote, ref).splitlines()]
    commits = {line[0] for line in lines if len(line) == 2 and re.fullmatch(r"[0-9a-f]{40}", line[0])}
    if len(commits) == 1:
        return commits.pop()
    if commits or not re.fullmatch(r"[0-9a-f]{40}", ref):
        raise RuntimeError(f"{remote} {ref} resolved to {len(commits)} commits, expected one")

    # Git's upload-pack does not advertise arbitrary detached commits through
    # ls-remote. Fetch a full-SHA object into a temporary bare repository so a
    # stale or force-pruned pin fails before we replace the manifest.
    with tempfile.TemporaryDirectory(prefix="verify-vllmb12x-pin-") as temporary:
        repository = Path(temporary) / "repository"
        run("git", "init", "--bare", str(repository))
        run("git", "-C", str(repository), "fetch", "--depth=1", remote, ref)
        resolved = run("git", "-C", str(repository), "rev-parse", "FETCH_HEAD").strip()
    if resolved != ref:
        raise RuntimeError(f"{remote} returned {resolved} for pinned commit {ref}")
    return resolved


def checkout(remote: str, commit: str, destination: Path) -> None:
    run("git", "clone", "--filter=blob:none", "--no-checkout", remote, str(destination))
    run("git", "-C", str(destination), "fetch", "--depth=1", "origin", commit)
    run("git", "-C", str(destination), "checkout", "--detach", commit)


def cmake_value(source: Path, variable: str) -> str:
    text = source.read_text()
    match = re.search(rf'set\s*\(\s*{re.escape(variable)}\s+"?([^\s\)"]+)', text)
    if match:
        return match.group(1)
    # A literal GIT_TAG is unambiguous in these single-source CMake files.
    matches = re.findall(r'GIT_TAG\s+"?([^\s)"]+)', text)
    if variable == "GIT_TAG" and len(matches) == 1:
        return matches[0]
    raise RuntimeError(f"cannot extract {variable} from {source}")


def sources_digest(source_ref: str, sources: dict[str, dict[str, str]]) -> str:
    """Return the digest of every source the image is built from.

    The image is assembled from many pinned repositories, so no single commit
    identifies it. Hashing each locked remote and commit makes the version
    change exactly when a pin changes, and lets anyone recompute the value from
    the committed manifest.
    """
    canonical = json.dumps(
        {
            "source_ref": source_ref,
            "sources": {
                name: {"commit": source["commit"], "remote": source["remote"]}
                for name, source in sorted(sources.items())
            },
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def cycle_name(source_ref: str) -> str:
    """Return the local-inference-lab cycle a source ref names."""
    if re.fullmatch(r"[0-9a-f]{40}", source_ref):
        raise RuntimeError(
            f"a version names the cycle, so the vLLM ref must be a branch: {source_ref!r}"
        )
    branch = source_ref.removeprefix("refs/heads/")
    for prefix in ("cycle/", "dev/"):
        if branch.startswith(prefix):
            return branch[len(prefix):]
    return branch


def base_version(declared: str | None, requested: str | None) -> str:
    """Return the upstream vLLM base version, preferring the requested value.

    The cycle branch merges upstream pull requests selectively, so tag ancestry
    does not prove which release it descends from. The base is therefore a
    reviewed claim recorded in the manifest, not a value inferred from tags.
    """
    value = requested or declared
    if not value:
        raise RuntimeError(
            "profile.json needs vllm_base_version; set it with --vllm-base-version"
        )
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[-.][0-9A-Za-z.-]+)?", value):
        raise RuntimeError(f"vLLM base version must look like 0.29.0: {value!r}")
    return value


def build_version(base: str, cycle: str, digest: str) -> str:
    """Compose the PEP 440 distribution version recorded in the image.

    Packaging normalises "-" and "_" to "." inside a local version segment, so
    a version built from the branch spelling would reach the wheel filename and
    the distribution metadata differently from the image label. Building from
    the normalised spelling keeps one string everywhere; the branch itself is
    still named by VLLM_SOURCE_REF.
    """
    return f"{base}+{re.sub(r'[-_]+', '.', cycle).lower()}.{digest[:12]}"


def canonical_source_ref(ref: str) -> str:
    """Return the lock form of a selected ref: a full branch ref or a commit.

    The image label and the publication tag both read this value, and both
    accept only those two forms, so a shorthand branch name is expanded here
    rather than propagated into generated files.
    """
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        return ref
    branch = ref.removeprefix("refs/heads/")
    if branch.startswith("refs/") or not re.fullmatch(r"[A-Za-z0-9._][A-Za-z0-9._/-]*", branch):
        raise RuntimeError(f"source ref must name a branch or a commit: {ref!r}")
    return f"refs/heads/{branch}"


def tracked_pair(manifest: dict[str, Any], name: str) -> tuple[str, str]:
    """Return the remote and ref the manifest records for one source.

    The manifest, not this script, decides which lineage a pin follows: the
    remote is recorded per source, and the ref is recorded as `source_ref` for
    vLLM and `b12x_ref` for B12X. Script defaults cannot hold these, because a
    default that drifts from the profile resolves another lineage and rewrites
    the pin silently.
    """
    key = "source_ref" if name == "vllm" else f"{name}_ref"
    if key not in manifest or "remote" not in manifest["sources"][name]:
        raise RuntimeError(
            f"profile.json does not record {key} and the {name} remote; "
            "pass the remote and ref explicitly"
        )
    return manifest["sources"][name]["remote"], canonical_source_ref(manifest[key])


def select_pair(
    name: str,
    manifest: dict[str, Any],
    remote: str | None,
    ref: str | None,
    allow_change: bool,
) -> tuple[str, str]:
    """Return the remote and ref to resolve, refusing an unrequested move.

    Resolving a pair other than the recorded one rewrites the pin onto another
    lineage, which is how a bare refresh would walk a pin backwards after the
    fork pair changed. The move is therefore an explicit act.
    """
    recorded_remote, recorded_ref = tracked_pair(manifest, name)
    if remote is None and ref is None:
        return recorded_remote, recorded_ref
    selected_remote = remote or recorded_remote
    selected_ref = canonical_source_ref(ref or recorded_ref)
    if (selected_remote, selected_ref) != (recorded_remote, recorded_ref) and not allow_change:
        raise RuntimeError(
            f"{name} is tracked at {recorded_remote} {recorded_ref}, but this run would "
            f"resolve {selected_remote} {selected_ref}; pass --allow-source-change to move it"
        )
    return selected_remote, selected_ref


def version_module(version: str, source_ref: str, commit: str) -> str:
    return "\n".join((
        "# SPDX-License-Identifier: Apache-2.0",
        "# Generated by scripts/refresh-vllmb12x.py. Do not edit by hand.",
        '"""PEP 440 distribution version for the vLLM wheel this profile builds.',
        "",
        "Composed from the upstream vLLM base version declared in profile.json, the",
        "local-inference-lab cycle that VLLM_SOURCE_REF names, and the first twelve",
        "hexadecimal digits of the digest over the locked source set. The digest is a",
        "pure function of the manifest, so identical inputs rebuild the same version;",
        "recompute it with `scripts/refresh-vllmb12x.py --dry-run`.",
        '"""',
        "",
        f'VLLM_BUILD_VERSION = "{version}"',
        f'VLLM_SOURCE_REF = "{source_ref}"',
        f'VLLM_SOURCE_REVISION = "{commit}"',
        "",
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-remote")
    parser.add_argument("--vllm-ref")
    parser.add_argument("--vllm-commit")
    parser.add_argument("--vllm-base-version")
    parser.add_argument("--b12x-remote")
    parser.add_argument("--b12x-ref")
    parser.add_argument("--b12x-commit")
    parser.add_argument("--allow-source-change", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(PROFILE.read_text())
    with tempfile.TemporaryDirectory(prefix="refresh-vllmb12x-") as temporary:
        checkout_root = Path(temporary) / "vllm"
        vllm_remote, vllm_ref = select_pair(
            "vllm", manifest, args.vllm_remote, args.vllm_ref, args.allow_source_change
        )
        b12x_remote, b12x_ref = select_pair(
            "b12x", manifest, args.b12x_remote, args.b12x_ref, args.allow_source_change
        )
        source_ref = canonical_source_ref(vllm_ref)
        vllm_commit = resolve_ref(vllm_remote, args.vllm_commit or vllm_ref)
        b12x_commit = resolve_ref(b12x_remote, args.b12x_commit or b12x_ref)
        checkout(vllm_remote, vllm_commit, checkout_root)

        updated = json.loads(json.dumps(manifest))
        updated["vllm_base_version"] = base_version(
            manifest.get("vllm_base_version"), args.vllm_base_version
        )
        updated["source_ref"] = source_ref
        updated["b12x_ref"] = canonical_source_ref(b12x_ref)
        updated["sources"]["vllm"]["remote"] = vllm_remote
        updated["sources"]["vllm"]["commit"] = vllm_commit
        updated["sources"]["b12x"]["remote"] = b12x_remote
        updated["sources"]["b12x"]["commit"] = b12x_commit
        for name, (relative_path, remote, variable) in CMAKE_SOURCES.items():
            source = checkout_root / relative_path
            if not source.is_file():
                raise RuntimeError(f"selected vLLM source removed required {relative_path}")
            ref = cmake_value(source, variable)
            updated["sources"][name]["remote"] = remote
            updated["sources"][name]["commit"] = resolve_ref(remote, ref)

        # Validate every declared source before any lock replacement. This
        # catches deleted commits and repository rewrites without touching the
        # current manifest or generated version module.
        for name, source in sorted(updated["sources"].items()):
            resolve_ref(source["remote"], source["commit"])

        content = json.dumps(updated, indent=2, sort_keys=True) + "\n"
        version = version_module(
            build_version(
                updated["vllm_base_version"],
                cycle_name(source_ref),
                sources_digest(source_ref, updated["sources"]),
            ),
            source_ref,
            vllm_commit,
        )
        if args.dry_run:
            print(content, end="")
            print(version, end="")
            return

        temporary_manifest = PROFILE.with_suffix(".json.tmp")
        temporary_version = VERSION.with_suffix(".bzl.tmp")
        temporary_manifest.write_text(content)
        temporary_version.write_text(version)
        temporary_manifest.replace(PROFILE)
        temporary_version.replace(VERSION)
        print(f"refreshed vLLMB12X to {vllm_commit} and B12X to {b12x_commit}")


if __name__ == "__main__":
    main()
