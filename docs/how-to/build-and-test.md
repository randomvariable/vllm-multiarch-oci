# Build and Test an Image

Build the image locally on a compatible DGX Spark, validate its structure, and load it into Docker.

## Prerequisites

Before you begin, verify that you have the following:

- The environment described in [Building Locally on DGX Spark](build-locally-on-spark.md).
- Just installed and a terminal at the checkout root.
- Docker for testing and loading.

### Step 1: Analyse the Graph

Check dependency resolution before compilation:

```bash
just analyze
```

Expect successful analysis and zero executed build actions.

### Step 2: Build the Image

Compile dependencies and assemble the image:

```bash
just build
```

On a compatible host, success produces `bazel-bin/image/vllmb12x`. Allow hours for a cold build. The command does not push or start the image.

### Step 3: Check the Structure

Run the Docker-backed image contract:

```bash
just test
```

Expect a passing test. It checks metadata and required files, not GPU inference or shared-library loading.

### Step 4: Load the Image

Import the image into Docker:

```bash
just load
```

The loader uses `randomvariable/vllm-b12x-multi:<build version>`. It does not start a container or model server.

## Related Practices

- [Build Configuration](../reference/build-configuration.md) (reference)
- [Measure Incremental Builds](measure-incremental-builds.md) (how-to)
