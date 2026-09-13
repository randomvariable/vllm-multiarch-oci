#!/usr/bin/env python3
"""Build the profile-pinned NCCL tree against the hermetic CUDA toolkit."""

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from action_lib import (
    configure_compiler_cache,
    configure_compiler_sysroot,
    extract,
    extract_durable,
    run,
    work_root,
    write_tar,
)

_VERSION = b"NCCL version 2.30.4 compiled with CUDA 13.3"
def _action_path(value: str) -> Path:
    return Path.cwd() / value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--src-tar", required=True)
    parser.add_argument("--cuda-tar", required=True)
    parser.add_argument("--ccache-tar", required=True)
    parser.add_argument("--gcc-sysroot-tar", required=True)
    parser.add_argument("--cuda-arch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--jobs", required=True, type=int)
    parser.add_argument("--lib-output", required=True)
    parser.add_argument("--sdk-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    work = work_root(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    source = work / "src"
    extract(_action_path(args.src_tar), source)
    cuda = extract_durable(_action_path(args.cuda_tar), "cuda")

    env = os.environ.copy()
    python = _action_path(args.python)
    ccache = configure_compiler_cache(work, _action_path(args.ccache_tar), env)
    configure_compiler_sysroot(_action_path(args.gcc_sysroot_tar), work, env)
    env["PATH"] = os.pathsep.join((str(python.parent), env["PATH"]))
    env["PYTHON"] = str(python)
    run(
        [
            "make",
            "-C",
            "src/src",
            f"-j{args.jobs}",
            "lib",
            f"BUILDDIR={work / 'build'}",
            f"CUDA_HOME={cuda}",
            f"CUDA_LIB={cuda / 'lib'}",
            f"NVCC={ccache} {cuda / 'bin/nvcc'}",
            # nvcc invokes CXX with its own preprocessor flags. Keep ccache as
            # its launcher and pass the pinned compiler wrapper directly.
            "CXX=" + env["CXX"],
            "CXXSTD=-std=c++17",
            # GDAKI's DOCA wrappers resolve libmlx5 at runtime. Do not enable
            # NCCL's separate direct-mlx5 path, which needs an unshipped .so
            # linker symlink and is not part of this transport.
            "RDMA_CORE=1",
            # NCCL adds -libverbs to CXXFLAGS, before its object list. GNU ld
            # can discard that dependency under --as-needed, leaving GDAKI's
            # verbs symbols unresolved when Torch loads libnccl. Put it in the
            # link flags, which NCCL emits after all objects.
            "LDFLAGS=-libverbs",
            "NCCL_GIT_BRANCH=profile-pinned",
            f"NCCL_GIT_COMMIT_HASH={args.commit}",
            f"NVCC_GENCODE=-gencode=arch=compute_{args.cuda_arch},code=sm_{args.cuda_arch}",
        ],
        cwd=work,
        env=env,
    )

    library = work / "build/lib/libnccl.so.2.30.4"
    if not library.is_file():
        raise RuntimeError(f"NCCL build did not produce {library}")
    if _VERSION not in library.read_bytes():
        raise RuntimeError("NCCL library does not contain expected CUDA 13.3 version")

    lib_output = _action_path(args.lib_output)
    lib_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(library, lib_output)
    lib_output.chmod(0o755)

    sdk = work / "sdk"
    shutil.copytree(work / "build/include", sdk / "include")
    sdk_library = sdk / "lib/libnccl.so.2.30.4"
    sdk_library.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(library, sdk_library)
    sdk_library.chmod(0o755)
    (sdk / "lib/libnccl.so.2").symlink_to("libnccl.so.2.30.4")
    (sdk / "lib/libnccl.so").symlink_to("libnccl.so.2")
    write_tar(_action_path(args.sdk_output), sdk)


if __name__ == "__main__":
    main()
