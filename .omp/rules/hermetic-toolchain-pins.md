---
name: hermetic-toolchain-pins
description: "Keep CI Rust/Cargo bootstrap aligned with the repository's pinned toolchain."
condition: "--default-toolchain 1\\.89\\.0"
scope: "tool:edit"
---

Derive the CI Rust/Cargo version from the authoritative hermetic pins before editing. Here, `MODULE.bazel` and the pinned vLLM `rust-toolchain.toml` require `1.95.0`. Do not select the minimum version that fixes one observed vendor failure; the bootstrap toolchain must reproduce the declared build graph.