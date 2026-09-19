#!/usr/bin/env python3
"""Refresh the immutable source lock for the VLLMB12X build profile.

The vLLM branch is moving input only to this command. Bazel consumes the
resulting full commit SHAs and never resolves a branch itself.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
PROFILE: Final = ROOT / "profiles/vllmb12x/profile.json"
VERSION: Final = ROOT / "profiles/vllmb12x/version.bzl"
DEFAULT_VLLM_REMOTE: Final = "https://github.com/local-inference-lab/vllm.git"
DEFAULT_VLLM_REF: Final = "refs/heads/dev/karmic-kraken"
DEFAULT_B12X_REMOTE: Final = "https://github.com/local-inference-lab/b12x.git"
DEFAULT_B12X_REF: Final = "refs/heads/master"

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
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
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


def source_version(commit: str) -> str:
    # This fork publishes no release tags. Its own pinned CI synthesizes this
    # setuptools-scm public version for a detached branch checkout. The local
    # segment below carries the full selected immutable revision.
    del commit
    return "0.1.dev1"


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
        raise RuntimeError(f"vLLM source ref must name a branch or a commit: {ref!r}")
    return f"refs/heads/{branch}"


def version_module(public_version: str, source_ref: str, commit: str) -> str:
    build_version = f"{public_version}+vllmb12x.g{commit[:12]}"
    return "\n".join((
        "# SPDX-License-Identifier: Apache-2.0",
        "# Generated by scripts/refresh-vllmb12x.py. Do not edit by hand.",
        '"""PEP 440 distribution version for the vLLM wheel this profile builds."""',
        "",
        f'VLLM_BUILD_VERSION = "{build_version}"',
        f'VLLM_SOURCE_REF = "{source_ref}"',
        f'VLLM_SOURCE_REVISION = "{commit}"',
        "",
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-remote", default=DEFAULT_VLLM_REMOTE)
    parser.add_argument("--vllm-ref", default=DEFAULT_VLLM_REF)
    parser.add_argument("--vllm-commit")
    parser.add_argument("--b12x-remote", default=DEFAULT_B12X_REMOTE)
    parser.add_argument("--b12x-ref", default=DEFAULT_B12X_REF)
    parser.add_argument("--b12x-commit")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(PROFILE.read_text())
    with tempfile.TemporaryDirectory(prefix="refresh-vllmb12x-") as temporary:
        checkout_root = Path(temporary) / "vllm"
        source_ref = canonical_source_ref(args.vllm_ref)
        vllm_commit = resolve_ref(args.vllm_remote, args.vllm_commit or args.vllm_ref)
        b12x_commit = resolve_ref(args.b12x_remote, args.b12x_commit or args.b12x_ref)
        checkout(args.vllm_remote, vllm_commit, checkout_root)

        updated = json.loads(json.dumps(manifest))
        updated["source_ref"] = source_ref
        updated["sources"]["vllm"]["remote"] = args.vllm_remote
        updated["sources"]["vllm"]["commit"] = vllm_commit
        updated["sources"]["b12x"]["remote"] = args.b12x_remote
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
        version = version_module(source_version(vllm_commit), source_ref, vllm_commit)
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
