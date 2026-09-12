# Measure Incremental Builds

Use this procedure to distinguish a real native-source rebuild from an unchanged action-cache hit.

## Prerequisites

Before you begin, verify that you have the following:

- A successful [complete image build](build-and-test.md).
- A quiet local build host and enough time for the measurement.
- Persistent local `/ccache` storage.
- A semantic native-source patch against the pinned vLLM revision.

### Step 1: Declare the Edit

Place the patch under `third_party/`, export it in `third_party/BUILD.bazel`, and append its label to `vllm_patches` in `MODULE.bazel`. Retain the required spinloop patch. Change real native behaviour, not a comment or artificial cache nonce. Record the patch hash and the source revision.

### Step 2: Check the Source Identity

Materialize the patched source archive and confirm the declared identity:

```bash
just bazel build @@+profile_sources+glm53_0906_vllm//:source.tar
```

Inspect `external/+profile_sources+glm53_0906_vllm/source.identity.json` under the directory reported by `just bazel info output_base`. Confirm your patch hash and read the patched source there. The repository rule watches declared patch files, so editing a patch invalidates the source archive on the next command that needs it. Analysis-only commands do not fetch it; check identity through a command that materializes the archive.

### Step 3: Capture the Full Build

Choose an unused evidence prefix and create its parent directory before running this command. Replace `[prefix]` with that path:

```bash
/usr/bin/time -p just build --build_event_json_file=[prefix].bep.json --profile=[prefix].profile.gz
```

Expect exit zero. Retain the terminal output, wall time, build events, profile, source identity, and image digest. Use identical flags for comparison runs. Repository refresh is a separate setup cost: record it separately rather than silently including or excluding it between runs.

### Step 4: Attribute Compiler Reuse

Retrieve the compilation's declared cache-statistics output:

```bash
just bazel build //components:vllm_compiler_cache_stats --platforms=//platforms:spark_arm64_sm121 --extra_execution_platforms=//platforms:spark_arm64_sm121 --spawn_strategy=local --disk_cache=.bazel-cache --incompatible_strict_action_env
```

Use the output path reported by Bazel to read the action-local statistics. Associate it with the measured native action. A cumulative shared-cache counter does not establish that this build reused objects. Reject a timing if the source identity was stale or the native action never executed.

### Step 5: Compare a Second Edit

Apply a second comparable native edit and repeat Steps 2 through 4. Keep build options and host load comparable. Record whether the native action actually executed and whether it reused compiler results. An unchanged whole-action cache hit does not measure native recompilation.

### Step 6: Restore the Source

Remove only the experiment patch label, export, and file. Repeat the forced repository refresh and verify that the identity contains only the intended production patches. Retain evidence. Report all successful timings and distinguish the fastest observation from a proven lower bound.

## Related Practices

- [Build and Cache Design](../explanation/build-and-cache-design.md) (explanation)
