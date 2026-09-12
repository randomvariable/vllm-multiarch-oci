# Inspect Your First Build

In this tutorial, you locate the image and analyse its graph without compiling CUDA code.

## Prerequisites

Before you begin, verify that you have the following:

- A checkout on a compatible ARM64 build host, with a terminal at its root.
- Bazel 9.2.0 and Just installed.
- Git, GNU tar, patch, sha256sum, Cargo, and network access for repository preparation.
- The compiler prerequisites in [Building Locally on DGX Spark](../how-to/build-locally-on-spark.md).

### Step 1: List the Helpers

Display the available build tasks:

```bash
just --list
```

The list includes `analyze`, `build`, `test`, `load`, and `bazel`.

### Step 2: Locate the Image

Query the image target through the local Bazel helper:

```bash
just bazel query //image:glm53_0906
```

The result includes `//image:glm53_0906`.

### Step 3: Analyse the Image

Resolve the local build graph without compiling:

```bash
just analyze
```

Expect successful analysis and zero build actions. Repository preparation can still download dependencies and execute repository tools. This does not produce an image or validate the host compiler ABI.

Continue with [Build and Test an Image](../how-to/build-and-test.md).
