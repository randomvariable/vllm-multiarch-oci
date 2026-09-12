#!/usr/bin/env python3
import os
import pathlib
import shlex
import socket
import subprocess
import sys

source = pathlib.Path.cwd()
wheel_dir = pathlib.Path(sys.argv[2])
ccache = os.environ["CCACHE_LAUNCHER"]
flags = shlex.split(os.environ.get("CCACHE_SMOKE_FLAGS", ""))
evidence_id = os.environ.get("CCACHE_SMOKE_EVIDENCE")
if evidence_id:
    evidence_dir = pathlib.Path("/ccache/evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CCACHE_LOGFILE"] = str(evidence_dir / ("cuda-wheel-smoke-" + evidence_id + ".log"))
subprocess.run([ccache, "g++", *flags, "-c", "host.cpp", "-o", "host.o"], cwd=source, check=True)
subprocess.run([ccache, "nvcc", *flags, "-c", "kernel.cu", "-o", "kernel.o"], cwd=source, check=True)
subprocess.run(["nvcc", "host.o", "kernel.o", "-o", "smoke"], cwd=source, check=True)
result = subprocess.run([source / "smoke"], cwd=source, check=True, capture_output=True, text=True)
expected = os.environ.get("CCACHE_SMOKE_EXPECTED", "11")
if result.stdout != "cuda-wheel-smoke: " + expected + "\n":
    raise RuntimeError(f"stale or incorrect compiler output: {result.stdout!r}")
if evidence_id:
    with (evidence_dir / ("cuda-wheel-smoke-" + evidence_id + ".stats")).open("w") as output:
        subprocess.run([ccache, "--show-stats", "--verbose"], check=True, stdout=output)
    (evidence_dir / ("cuda-wheel-smoke-" + evidence_id + ".worker")).write_text(
        socket.gethostname() + "\n"
    )
if os.environ.get("CCACHE_SMOKE_FAIL_AFTER_COMPILE") == "1":
    raise RuntimeError("deliberate failure after compiler output verification")
subprocess.run(
    [sys.argv[1], "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--wheel-dir", wheel_dir, source],
    check=True,
)
