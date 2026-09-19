#!/usr/bin/env python3
"""Shared utilities for hermetic Python build actions."""

from __future__ import annotations

import fcntl
import csv
import hashlib
import importlib.metadata
import os
import shutil
import shlex
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Mapping, Sequence


_VLLM_CMAKE_SOURCE_NAMES = frozenset(
    {
        "cutlass",
        "deepgemm",
        "deepselect",
        "flash_attn",
        "flashkda",
        "flashmla",
        "msa",
        "qutlass",
        "tml_fa4",
        "triton",
    }
)


def work_root(value: str) -> Path:
    """Resolve an execroot-relative action path."""
    return Path.cwd() / value


def extract(archive: Path, destination: Path) -> None:
    """Extract a tar archive without permitting unsafe member types."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")


def materialize_vllm_cmake_sources(
    archives: Mapping[str, Path], work: Path, vllm_source: Path
) -> tuple[dict[str, str], list[str]]:
    """Extract the exact CMake FetchContent trees and return supported overrides."""
    names = set(archives)
    if names != _VLLM_CMAKE_SOURCE_NAMES:
        missing = sorted(_VLLM_CMAKE_SOURCE_NAMES - names)
        unexpected = sorted(names - _VLLM_CMAKE_SOURCE_NAMES)
        raise RuntimeError(f"invalid vLLM CMake sources: missing={missing}, unexpected={unexpected}")

    root = work / "cmake-sources"
    sources: dict[str, Path] = {}
    for name in sorted(archives):
        destination = root / name
        extract(archives[name], destination)
        sources[name] = destination

    apply_flashkda_checkpoint_patch(sources["flashkda"], vllm_source)

    triton_kernels = sources["triton"] / "python" / "triton_kernels" / "triton_kernels"
    if not triton_kernels.is_dir():
        raise RuntimeError(f"declared Triton source lacks triton_kernels: {triton_kernels}")

    msa_cutlass = sources["msa"] / "python" / "fmha_sm100" / "cutlass" / "include"
    if not msa_cutlass.is_dir():
        raise RuntimeError(f"declared MSA source lacks fmha_sm100 CUTLASS headers: {msa_cutlass}")

    env = {
        "DEEPGEMM_SRC_DIR": str(sources["deepgemm"]),
        "DEEPSELECT_SRC_DIR": str(sources["deepselect"]),
        "FLASH_KDA_SRC_DIR": str(sources["flashkda"]),
        "FLASH_MLA_SRC_DIR": str(sources["flashmla"]),
        "FMHA_SM100_SRC_DIR": str(sources["msa"]),
        "QUTLASS_SRC_DIR": str(sources["qutlass"]),
        "TML_FA4_SRC_DIR": str(sources["tml_fa4"]),
        "TRITON_KERNELS_SRC_DIR": str(triton_kernels),
        "VLLM_FLASH_ATTN_SRC_DIR": str(sources["flash_attn"]),
    }
    cmake_args = [
        "-DCUTLASS_INCLUDE_DIR=" + str(sources["cutlass"] / "include"),
        "-DCUTLASS_TOOLS_UTIL_INCLUDE_DIR=" + str(sources["cutlass"] / "tools" / "util" / "include"),
        "-DVLLM_CUTLASS_SRC_DIR=" + str(sources["cutlass"]),
    ]
    return env, cmake_args


def apply_flashkda_checkpoint_patch(flashkda: Path, vllm_source: Path) -> None:
    """Apply vLLM's packed-checkpoint patch that its FLASH_KDA_SRC_DIR branch skips.

    vLLM patches FlashKDA only when CMake fetches it. A caller-supplied source
    directory stays unpatched, and vLLM then requires the packed checkpoint API
    anyway. The patch ships inside the pinned vLLM source, so apply that exact
    file rather than a vendored copy that would drift from the pin.
    """
    patch_file = (
        vllm_source / "cmake" / "external_projects" / "patches" / "flashkda-packed-checkpoints.patch"
    )
    if not patch_file.is_file():
        raise RuntimeError(f"pinned vLLM source lacks the FlashKDA patch: {patch_file}")

    run(["patch", "--strip=1", "--input", str(patch_file)], cwd=flashkda)

    header = flashkda / "csrc" / "flash_kda.h"
    if "checkpoint_indptr" not in header.read_text():
        raise RuntimeError(f"patched FlashKDA source lacks the packed checkpoint API: {header}")


def extract_durable(archive: Path, name: str) -> Path:
    """Extract a content-addressed toolchain once on the shared cache volume."""
    with archive.open("rb") as contents:
        digest = hashlib.file_digest(contents, "sha256").hexdigest()
    root = toolchain_cache_dir()
    destination = root / f"{name}-{digest}"
    marker = destination / ".complete"
    root.mkdir(parents=True, exist_ok=True)

    with (root / f".{name}-{digest}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.is_file() and marker.read_text() == digest:
            return destination

        temporary = root / f".{name}-{digest}-{os.getpid()}"
        if temporary.exists():
            shutil.rmtree(temporary)
        try:
            extract(archive, temporary)
            (temporary / ".complete").write_text(digest)
            if destination.exists():
                raise RuntimeError(f"incomplete durable toolchain at {destination}")
            temporary.rename(destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    return destination


def _cache_override() -> Path | None:
    configured = os.environ.get("VLLMB12X_CACHE_ROOT")
    return Path(configured) if configured else None


def _xdg_cache_home() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))


def compiler_cache_dir() -> Path:
    """Use ccache's normal per-user location unless CI supplies a volume."""
    configured = _cache_override()
    return configured / "objects" if configured else _xdg_cache_home() / "ccache"


def toolchain_cache_dir() -> Path:
    """Keep extracted toolchains alongside the local cache, not inside ccache."""
    configured = _cache_override()
    return configured / "toolchains" if configured else _xdg_cache_home() / "vllmb12x" / "toolchains"


def configure_cargo_vendor(archive: Path, work: Path) -> Path:
    """Extract declared Cargo sources and make their replacement path local."""
    cargo_home = work / "cargo"
    extract(archive, cargo_home)
    config = cargo_home / "config.toml"
    vendor = cargo_home / "vendor"
    if not vendor.is_dir():
        raise RuntimeError(f"declared Cargo vendor archive lacks {vendor}")
    config.write_text(config.read_text().replace('directory = "vendor"', 'directory = "%s"' % vendor))
    return cargo_home


_TARGET_ABI = {
    "aarch64": {
        "gcc_triple": "aarch64-linux-gnu",
        "library_dir": "aarch64-linux-gnu",
        "loader": "ld-linux-aarch64.so.1",
        "rust_triple": "aarch64-unknown-linux-gnu",
    },
    "x86_64": {
        "gcc_triple": "x86_64-linux-gnu",
        "library_dir": "x86_64-linux-gnu",
        "loader": "ld-linux-x86-64.so.2",
        "rust_triple": "x86_64-unknown-linux-gnu",
    },
}


def target_abi(target_cpu: str) -> dict[str, str]:
    try:
        return _TARGET_ABI[target_cpu]
    except KeyError as error:
        raise ValueError(f"unsupported target CPU: {target_cpu!r}") from error


def configure_sysroot_runtime(
    archive: Path, work: Path, env: dict[str, str], *, target_cpu: str
) -> Path:
    """Extract the pinned sysroot and resolve ELF dependencies only from it.

    Actions that merely import the native wheels this build produces still need
    their shared libraries, such as the libibverbs that RDMA-enabled NCCL pulls
    into Torch. Serve them from the declared input, never the worker image.
    """
    root = work / "gcc-sysroot"
    extract(archive, root)
    libraries = root / "usr" / "lib" / target_abi(target_cpu)["library_dir"]
    if not libraries.is_dir():
        raise RuntimeError(f"compiler sysroot lacks shared libraries: {libraries}")
    env["LD_LIBRARY_PATH"] = _prepend_path(env.get("LD_LIBRARY_PATH"), libraries)
    return root


def configure_compiler_sysroot(
    archive: Path, work: Path, env: dict[str, str], *, target_cpu: str
) -> Path:
    """Extract the pinned GCC tree and select wrappers that never use host GCC."""
    for name in (
        "CPATH",
        "C_INCLUDE_PATH",
        "CPLUS_INCLUDE_PATH",
        "LIBRARY_PATH",
        "COMPILER_PATH",
        "GCC_EXEC_PREFIX",
    ):
        env.pop(name, None)
    # Build backends import native wheels while compiling dependent wheels, so
    # the same sysroot must satisfy both their loader and this compiler.
    abi = target_abi(target_cpu)
    root = configure_sysroot_runtime(archive, work, env, target_cpu=target_cpu)
    compiler_dir = root / "usr" / "bin"
    library_dir = root / "usr" / "lib" / abi["library_dir"]
    loader = root / "usr" / "lib" / abi["loader"]
    if not loader.is_file():
        loader = library_dir / abi["loader"]
    if not loader.is_file():
        raise RuntimeError(f"compiler sysroot lacks dynamic loader: {loader}")
    # GNU ld resolves its fixed interpreter path through --sysroot. The Debian
    # package closure stores the loader below an ABI library directory.
    for directory in (root / "lib", root / "lib64"):
        directory.mkdir(exist_ok=True)
        sysroot_loader = directory / abi["loader"]
        if not sysroot_loader.exists():
            sysroot_loader.symlink_to("../" + str(loader.relative_to(root)))
    for name in ("gcc", "g++"):
        compiler = compiler_dir / name
        if not compiler.is_file():
            raise RuntimeError(f"compiler sysroot lacks {compiler}")

    wrappers = work / "compiler-bin"
    wrappers.mkdir(parents=True, exist_ok=True)
    # These packages use alternatives-managed public command names. Native
    # actions have no host alternatives database, so expose declared targets.
    for public_name, target_name in (
        ("sh", "bash"),
        ("which", "which.gnu"),
        ("awk", "gawk"),
    ):
        target = compiler_dir / target_name
        public = compiler_dir / public_name
        if target.is_file() and not public.exists():
            public.symlink_to(target_name)
    for name in ("gcc", "g++"):
        # Compute the extraction root from $0 so wrapper bytes remain stable
        # across action roots and ccache can reuse compatible objects.
        wrapper = wrappers / name
        wrapper.write_text(
            "#!/bin/sh\n"
            "set -eu\n"
            'wrapper_dir=${0%/*}\n'
            'root=$(CDPATH= cd -- "$wrapper_dir/../gcc-sysroot" && pwd)\n'
            f'exec "$root/usr/bin/{name}" --sysroot="$root" '
            '-B"$root/usr/bin" '
            f'-B"$root/usr/lib/gcc/{abi["gcc_triple"]}/15" "$@"\n'
        )
        wrapper.chmod(0o755)

    # NCCL's makefiles use env(1) to locate bash. The pinned shell must remain
    # usable after PATH is sealed to the declared sysroot.
    bash = compiler_dir / "bash"
    if not bash.is_file():
        raise RuntimeError(f"compiler sysroot lacks {bash}")
    ldd = compiler_dir / "ldd"
    if not ldd.is_file():
        raise RuntimeError(f"compiler sysroot lacks {ldd}")

    env["CC"] = str(wrappers / "gcc")
    env["CXX"] = str(wrappers / "g++")
    # The build utilities in the declared package closure are the only tools
    # visible to native actions.  Return their directory so callers can invoke
    # a tool by its declared absolute path rather than relying on PATH lookup.
    # Keep the sysroot's standard shell path valid for tools that resolve it.
    sh = root / "bin" / "sh"
    sh.parent.mkdir(exist_ok=True)
    if not sh.exists():
        sh.symlink_to("../usr/bin/bash")
    env["PATH"] = _prepend_path(env.get("PATH"), wrappers, compiler_dir)
    return compiler_dir


def install_wheels(python: Path, destination: Path, wheels: Sequence[Path]) -> None:
    """Install build-only wheel dependencies into an action-local site directory."""
    if not wheels:
        return
    destination.mkdir(parents=True, exist_ok=True)
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-compile",
            "--target",
            destination,
            *wheels,
        ]
    )
    normalize_local_wheel_metadata(destination)


def normalize_local_wheel_metadata(site: Path) -> None:
    """Remove action-root direct URLs and repair their wheel RECORD entries."""
    for metadata in site.glob("*.dist-info"):
        direct_url = metadata / "direct_url.json"
        if not direct_url.is_file():
            continue
        direct_url.unlink()
        record = metadata / "RECORD"
        if not record.is_file():
            continue
        with record.open(newline="") as contents:
            rows = list(csv.reader(contents))
        kept = [
            row
            for row in rows
            if row and not row[0].endswith("/direct_url.json")
        ]
        with record.open("w", newline="") as contents:
            csv.writer(contents, lineterminator="\n").writerows(kept)


def configure_reproducible_compilation(
    work: Path, source: Path, env: dict[str, str]
) -> dict[str, str]:
    """Set deterministic compiler flags without retaining action-root paths."""
    env["SOURCE_DATE_EPOCH"] = "0"
    env["PYTHONHASHSEED"] = "0"
    mappings = (
        f"-ffile-prefix-map={work}=.",
        f"-ffile-prefix-map={source}=.",
        f"-fdebug-prefix-map={work}=.",
        f"-fdebug-prefix-map={source}=.",
    )
    for name in ("CFLAGS", "CXXFLAGS"):
        env[name] = _append_argument(env.get(name), " ".join(mappings))
    env["RUSTFLAGS"] = _append_argument(
        env.get("RUSTFLAGS"),
        f"--remap-path-prefix={work}=. --remap-path-prefix={source}=.",
    )
    return {"cuda": "--objdir-as-tempdir"}


def create_nvcc_wrapper(work: Path, nvcc: Path) -> Path:
    """Wrap nvcc with a source-relative seed that ignores output filenames."""
    work.mkdir(parents=True, exist_ok=True)
    wrapper = work / "nvcc-reproducible"
    wrapper.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "seed=unknown\n"
        "for arg in \"$@\"; do\n"
        "  case $arg in\n"
        "    *.cu|*.cuh) seed=${arg#\"$PWD\"/}; break ;;\n"
        "  esac\n"
        "done\n"
        "exec "
        + shlex.quote(str(nvcc))
        + " --frandom-seed=\"$seed\" \"$@\"\n"
    )
    wrapper.chmod(0o755)
    return wrapper


def materialize_console_scripts(python: Path, site: Path, destination: Path) -> None:
    """Create console entry points omitted by pip's --target installation."""
    destination.mkdir(parents=True, exist_ok=True)
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        for entry_point in distribution.entry_points:
            if entry_point.group != "console_scripts":
                continue
            wrapper = destination / entry_point.name
            wrapper.write_text(
                "#!/bin/sh\nexec "
                + shlex.quote(str(python))
                + " -c "
                + shlex.quote(
                    "import sys; from importlib.metadata import distribution; "
                    "entry = distribution(%r).entry_points.select("
                    "group='console_scripts', name=%r)[0]; sys.exit(entry.load()())"
                    % (distribution.metadata["Name"], entry_point.name)
                )
                + ' "$@"\n'
            )
            wrapper.chmod(0o755)


def wheel_scripts(wheel: Path, destination: Path) -> list[Path]:
    """Materialize a wheel's packaged scripts for venv installation.

    pip --target omits the .data/scripts tree, so console entry points such
    as ninja's binary never reach the venv bin directory on their own.
    """
    prefix = ".data/scripts/"
    scripts: list[Path] = []
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if prefix not in name or name.endswith("/"):
                continue
            target = destination / Path(name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
            scripts.append(target)
    return scripts


def extract_wheel_script(wheel: Path, script: str, destination: Path) -> Path:
    """Extract a wheel's packaged script, which pip --target omits."""
    suffix = "/scripts/" + script
    with zipfile.ZipFile(wheel) as archive:
        matches = [name for name in archive.namelist() if name.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one {script!r} script in {wheel}, found {len(matches)}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.read(matches[0]))
    destination.chmod(0o755)
    return destination


def rewrite_sysconfig(python: Path, destination: Path) -> Path:
    """Rewrite rules_python's fixed /install prefix for this action's interpreter."""
    name = subprocess.check_output(
        [python, "-c", "import sysconfig; print(sysconfig._get_sysconfigdata_name())"],
        text=True,
    ).strip()
    source = Path(
        subprocess.check_output(
            [
                python,
                "-c",
                "import importlib, sysconfig; "
                "print(importlib.import_module(sysconfig._get_sysconfigdata_name()).__file__)",
            ],
            text=True,
        ).strip()
    )
    destination.mkdir(parents=True, exist_ok=True)
    rewritten = destination / (name + ".py")
    python_home = python.parent.parent
    rewritten.write_text(source.read_text().replace("/install", str(python_home)))
    return destination


def configure_compiler_cache(
    work: Path, ccache_tar: Path, env: dict[str, str], *, target_cpu: str
) -> Path | None:
    """Configure and verify the required durable compiler cache."""
    tools = work / "tools"
    extract(ccache_tar, tools)

    cache_dir = compiler_cache_dir()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        probe = cache_dir / ".ccache-write-probe"
        probe.touch(exist_ok=True)
        probe.unlink()
    except OSError as error:
        raise RuntimeError(f"required durable ccache is unavailable at {cache_dir}: {error}") from error

    launcher = tools / "usr/bin/ccache"
    packaged_lib = tools / "usr" / "lib" / target_abi(target_cpu)["library_dir"]
    env["CCACHE_DIR"] = str(cache_dir)
    # Source paths are below the per-operation execroot. Strip it from keys.
    env["CCACHE_BASEDIR"] = str(Path.cwd())
    env["CCACHE_NOHASHDIR"] = "1"
    env["CCACHE_COMPILERCHECK"] = "content"
    # CUDA headers arrive from the declared toolkit tar at a fresh remote path.
    # The toolkit digest remains in every action key, so system-header hashing
    # would only prevent cross-sandbox reuse without improving correctness.
    env["CCACHE_SLOPPINESS"] = "include_file_ctime,include_file_mtime,time_macros,system_headers"
    env["CCACHE_MAXSIZE"] = "100G"
    env["LD_LIBRARY_PATH"] = _append_path(env.get("LD_LIBRARY_PATH"), packaged_lib)
    env["PATH"] = _prepend_path(env.get("PATH"), launcher.parent)
    env["CCACHE_LAUNCHER"] = str(launcher)
    subprocess.run([launcher, "--version"], env=env, check=True, stdout=subprocess.DEVNULL)
    for name in (
        "CMAKE_C_COMPILER_LAUNCHER",
        "CMAKE_CXX_COMPILER_LAUNCHER",
        "CMAKE_CUDA_COMPILER_LAUNCHER",
    ):
        env[name] = str(launcher)
    launchers = " ".join(
        "-D%s=%s" % (name, launcher)
        for name in (
            "CMAKE_C_COMPILER_LAUNCHER",
            "CMAKE_CXX_COMPILER_LAUNCHER",
            "CMAKE_CUDA_COMPILER_LAUNCHER",
        )
    )
    env["CMAKE_ARGS"] = _append_argument(env.get("CMAKE_ARGS"), launchers)
    env["FLASHINFER_CXX_LAUNCHER"] = str(launcher)
    env["FLASHINFER_NVCC_LAUNCHER"] = str(launcher)
    return launcher


def run(
    command: Sequence[str | Path],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    """Echo and run a checked subprocess without a shell."""
    argv = [str(part) for part in command]
    print("+ " + shlex.join(argv))
    subprocess.run(argv, cwd=cwd, env=env, check=True)


def single_wheel(directory: Path) -> Path:
    """Return the sole wheel produced by a build action."""
    wheels = sorted(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one wheel in {directory}, found {len(wheels)}")
    return wheels[0]


def materialize_absolute_symlinks(root: Path) -> None:
    """Replace absolute links so archives remain relocatable and safe to extract."""
    for path in sorted(root.rglob("*")):
        if not path.is_symlink():
            continue
        target = path.readlink()
        if not target.is_absolute():
            continue
        resolved = path.resolve(strict=True)
        path.unlink()
        if resolved.is_dir():
            shutil.copytree(resolved, path, symlinks=True)
        else:
            shutil.copy2(resolved, path)


def write_tar(archive: Path, source: Path) -> None:
    """Create a deterministic tar containing the contents of *source*."""
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as output:
        for path in sorted(source.rglob("*")):
            info = output.gettarinfo(str(path), arcname=str(path.relative_to(source)))
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            if info.isfile():
                with path.open("rb") as contents:
                    output.addfile(info, contents)
            else:
                output.addfile(info)


def _append_path(existing: str | None, item: Path) -> str:
    return ":".join(part for part in (existing, str(item)) if part)


def _prepend_path(existing: str | None, *items: Path) -> str:
    return ":".join(
        part for part in (*(str(item) for item in items), existing) if part
    )


def _append_argument(existing: str | None, value: str) -> str:
    return " ".join(part for part in (existing, value) if part)
