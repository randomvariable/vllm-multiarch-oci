#!/usr/bin/env python3
"""Build ExLlamaV3 after removing its unreachable x86 CPU reduction path."""

import pathlib
import subprocess
import sys

source = pathlib.Path.cwd()
extension = source / "exllamav3" / "exllamav3_ext"

patches = {
    extension / "avx2_target.cpp": (
        'avx2_supported = __builtin_cpu_supports("avx2");',
        "avx2_supported = false;",
    ),
    extension / "avx512_target.cpp": (
        'avx512_supported = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");',
        "avx512_supported = false;",
    ),
}
for path, (old, new) in patches.items():
    text = path.read_text()
    if old not in text:
        raise RuntimeError(f"unexpected ExLlamaV3 source: {path}")
    path.write_text(text.replace(old, new, 1))

# Grace has no AVX. The extension guards this path with is_avx2_supported(),
# so retain the linker ABI but make accidental tensor-parallel use fail safely.
(extension / "parallel" / "all_reduce_cpu_avx2.cpp").write_text(
    """#include <cstdlib>
#include \"all_reduce_cpu_avx2.h\"

void enable_fast_fp() {}
void enable_fast_fp_avx2() {}
void perform_cpu_reduce(PGContext*, size_t, uint32_t, uint8_t*, size_t) { std::abort(); }
void perform_cpu_reduce_avx2(PGContext*, size_t, uint32_t, uint8_t*, size_t) { std::abort(); }
"""
)
(extension / "parallel" / "all_reduce_cpu_avx512.cpp").write_text(
    """#include <cstdlib>
#include \"all_reduce_cpu_avx512.h\"

void enable_fast_fp_avx512() {}
void bf16_add_inplace_avx512(uint16_t*, const uint16_t*, size_t) { std::abort(); }
void perform_cpu_reduce_avx512(PGContext*, size_t, uint32_t, uint8_t*, size_t) { std::abort(); }
"""
)

subprocess.run(
    [
        sys.argv[1],
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "--wheel-dir",
        sys.argv[2],
        source,
    ],
    check=True,
)
