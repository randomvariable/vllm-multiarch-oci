#!/usr/bin/env python3
"""Verify the registered C++ toolchain has no worker-path fallback."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import importlib.util


ROOT = Path(__file__).resolve().parents[1]
BAZEL = "bazelisk"
PLATFORM = "//platforms:spark_arm64_sm121"
X86_PLATFORM = "//platforms:blackwell_x86_64_sm120"
PROBE = "//platforms:hermetic_cc_toolchain_probe"
ARM64_PYTHON_PROBE = "//platforms:arm64_python_probe"
_ACTION_LIB_SPEC = importlib.util.spec_from_file_location("action_lib", ROOT / "bazel/action_lib.py")
assert _ACTION_LIB_SPEC and _ACTION_LIB_SPEC.loader
ACTION_LIB = importlib.util.module_from_spec(_ACTION_LIB_SPEC)
_ACTION_LIB_SPEC.loader.exec_module(ACTION_LIB)


def bazel(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [BAZEL, "--ignore_all_rc_files", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


class HermeticCCToolchainTest(unittest.TestCase):
    def test_arm64_python_runs_with_declared_loader_and_emulator(self) -> None:
        result = bazel(
            "build",
            ARM64_PYTHON_PROBE,
            "--platforms=" + PLATFORM,
            "--extra_execution_platforms=//platforms:local_x86_64," + PLATFORM,
            "--spawn_strategy=local",
            "--disk_cache=.bazel-cache",
            "--incompatible_strict_action_env",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("qemu-aarch64-static", result.stderr)
        self.assertNotIn("/lib/ld-linux-aarch64.so.1", result.stderr)

    def test_probe_does_not_use_worker_include_paths(self) -> None:
        poisoned = {
            "CPATH": "/worker/include",
            "C_INCLUDE_PATH": "/worker/c/include",
            "CPLUS_INCLUDE_PATH": "/worker/cxx/include",
            "LIBRARY_PATH": "/worker/lib",
            "COMPILER_PATH": "/worker/compiler",
            "GCC_EXEC_PREFIX": "/worker/gcc",
        }
        result = bazel(
            "build",
            PROBE,
            "--platforms=" + X86_PLATFORM,
            "--extra_execution_platforms=" + X86_PLATFORM,
            "--spawn_strategy=local",
            "--disk_cache=.bazel-cache",
            "--incompatible_strict_action_env",
            *["--action_env=%s=%s" % item for item in poisoned.items()],
            env={**__import__("os").environ, **poisoned},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("/usr/aarch64-linux-gnu", result.stderr)
        self.assertNotIn("/usr/lib/gcc", result.stderr)
        probe = ROOT / "bazel-bin/platforms/hermetic_cc_toolchain_probe"
        elf = subprocess.run(["file", str(probe)], text=True, capture_output=True, check=True)
        self.assertIn("x86-64", elf.stdout)

    def test_native_wrapper_clears_host_compiler_discovery(self) -> None:
        env = {
            "CPATH": "/host/include",
            "C_INCLUDE_PATH": "/host/c/include",
            "CPLUS_INCLUDE_PATH": "/host/cxx/include",
            "LIBRARY_PATH": "/host/lib",
            "COMPILER_PATH": "/host/compiler",
            "GCC_EXEC_PREFIX": "/host/gcc",
        }
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "empty.tar"
            with __import__("tarfile").open(archive, "w"):
                pass
            with self.assertRaisesRegex(RuntimeError, "lacks shared libraries"):
                ACTION_LIB.configure_compiler_sysroot(
                    archive, Path(directory), env, target_cpu="aarch64"
                )
        self.assertEqual(env, {})

    def test_compiler_sysroot_exposes_declared_native_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            for relative in (
                "usr/bin/gcc",
                "usr/bin/g++",
                "usr/bin/bash",
                "usr/bin/gawk",
                "usr/bin/ldd",
                "usr/bin/which.gnu",
                "usr/bin/make",
                "usr/bin/patch",
                "usr/bin/pkgconf",
                "usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("#!/bin/sh\n")
            archive = Path(directory) / "sysroot.tar"
            with __import__("tarfile").open(archive, "w") as contents:
                contents.add(root, arcname="")
            env: dict[str, str] = {}
            compiler_bin = ACTION_LIB.configure_compiler_sysroot(
                archive, Path(directory) / "work", env, target_cpu="x86_64"
            )
            self.assertEqual(compiler_bin, Path(directory) / "work/gcc-sysroot/usr/bin")
            self.assertIn(str(compiler_bin), env["PATH"])
            self.assertIn("gcc-sysroot/usr/bin", env["PATH"])
            self.assertTrue((compiler_bin / "sh").is_symlink())
            self.assertTrue((compiler_bin / "which").is_symlink())
            self.assertTrue((compiler_bin / "awk").is_symlink())
            self.assertTrue((Path(directory) / "work/gcc-sysroot/bin/sh").is_symlink())

    def test_compiler_cache_uses_configured_durable_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "durable"
            archive = Path(directory) / "ccache.tar"
            tools = Path(directory) / "tools"
            launcher = tools / "usr/bin/ccache"
            with __import__("tarfile").open(archive, "w") as contents:
                for path in (launcher, tools / "usr/lib/aarch64-linux-gnu/libccache.so"):
                    path.parent.mkdir(parents = True, exist_ok = True)
                    path.write_text("#!/bin/sh\nexit 0\n")
                    contents.add(path, arcname = str(path.relative_to(tools)))
            env: dict[str, str] = {}
            with mock.patch.dict("os.environ", {"VLLMB12X_CACHE_ROOT": str(root)}, clear = False), mock.patch.object(ACTION_LIB.subprocess, "run"):
                ACTION_LIB.configure_compiler_cache(
                    Path(directory) / "work", archive, env, target_cpu="aarch64"
                )
            self.assertEqual(env["CCACHE_DIR"], str(root / "objects"))
            self.assertTrue((root / "objects").is_dir())

    def test_x86_cache_uses_x86_runtime_library_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "durable"
            archive = Path(directory) / "ccache.tar"
            tools = Path(directory) / "tools"
            launcher = tools / "usr/bin/ccache"
            library = tools / "usr/lib/x86_64-linux-gnu/libccache.so"
            with __import__("tarfile").open(archive, "w") as contents:
                for path in (launcher, library):
                    path.parent.mkdir(parents = True, exist_ok = True)
                    path.write_text("#!/bin/sh\nexit 0\n")
                    contents.add(path, arcname = str(path.relative_to(tools)))
            env: dict[str, str] = {}
            with mock.patch.dict("os.environ", {"VLLMB12X_CACHE_ROOT": str(root)}, clear = False), mock.patch.object(ACTION_LIB.subprocess, "run"):
                ACTION_LIB.configure_compiler_cache(
                    Path(directory) / "work", archive, env, target_cpu="x86_64"
                )
            self.assertIn("/usr/lib/x86_64-linux-gnu", env["LD_LIBRARY_PATH"])

    def test_local_cache_uses_standard_ccache_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_home = Path(directory) / "cache"
            with mock.patch.dict("os.environ", {"XDG_CACHE_HOME": str(cache_home)}, clear = True):
                self.assertEqual(ACTION_LIB.compiler_cache_dir(), cache_home / "ccache")
                self.assertEqual(
                    ACTION_LIB.toolchain_cache_dir(),
                    cache_home / "vllmb12x" / "toolchains",
                )


if __name__ == "__main__":
    unittest.main()
