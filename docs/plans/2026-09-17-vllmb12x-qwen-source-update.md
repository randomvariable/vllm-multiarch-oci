# VLLMB12X Qwen Source Update Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `planning-review` to implement this plan task-by-task.

**Goal:** Advance the VLLMB12X Qwen source lock to the current `dev/jovian-judgement` and B12X `master` tips, then integrate the requested Qwen and adaptive-grace pull requests as immutable, reproducible source inputs.

**Architecture:** The build profile remains a lock of immutable commits. Create integration branches in the `randomvariable` forks because the project cannot push to `local-inference-lab`; each branch starts from the requested upstream tip and carries the requested PR commits in dependency order. Keep the upstream adaptive-grace change as a profile patch, regenerated against the resulting vLLM integration revision, because `profile_sources` applies that patch at source-archive creation and records its digest in `source.identity.json`.

**Tech Stack:** Git, GitHub PR refs, Bazel Bzlmod profile sources, Python source-lock refresh tooling, ARM64 image contract.

---

## Resolved Inputs

- Current `local-inference-lab/vllm` `dev/jovian-judgement`: `8e1f1e587f8d24faf606f334a1c4bdaaa6bd4368`.
- Current `local-inference-lab/b12x` `master`: `02407f653609e2c4c49db2efb5f77835d2e04e32`.
- B12X PR heads: #384 `45e03da14ba64b74ddd62cb27d8240f084c50eea`, #386 `8801c016c4294cb2aaf2b1b5aeae7c9efc04fb89`, #387 `3ee490bc198dee9c24fd96cfa88dcc3e93c201c2`.
- vLLM PR heads: #777 `beef20f158e3a4f917ce3607c73e13f8f994517d`, #779 `d320ffd57302b29da7cbebeac98648df1ac6e7ba`.
- Upstream adaptive-grace PR head: vLLM #52917 `d732758a8041db5172e4b39cbd6e7a76758c32b0`.

The requested B12X and vLLM PR heads are independent Git branches, not linear ancestors. Integrate #384, #386, then #387 on B12X, and #777 then #779 on vLLM. Resolve conflicts against the fresh tips rather than pinning a PR head that omits another requested change.

### Task 1: Create isolated integration branches

**Files:**
- Modify: no files until both integration branch tips exist.

**Step 1: Start from clean dedicated worktrees**

Create a clean worktree for this repository from `main` and independent temporary worktrees for `randomvariable/vllm` and `randomvariable/b12x`. Do not reuse this checkout while its reproducibility edits remain uncommitted. Name the vLLM integration branch `dev/rv-jovian-judgement` on the `randomvariable/vllm` fork. Use the same branch name on `randomvariable/b12x` unless its existing refs conflict; if they do, use `dev/rv-jovian-judgement-b12x` and record the exact ref in the profile.

**Step 2: Construct the B12X source branch**

Start at `local-inference-lab/b12x` `master` tip `02407f653...`. Cherry-pick the requested #384, #386, and #387 ranges in that order with `-x`; resolve only conflicts caused by their independent bases. Preserve each PR's tests and audit the final range for all three feature sets: preparation/collective fixes, PLE checkpoint export, and paired QSA scoring.

**Step 3: Construct the vLLM source branch**

Start at `local-inference-lab/vllm` `dev/jovian-judgement` tip `8e1f1e...`. Cherry-pick #777 before #779 with `-x`; resolve against the tipped source. Confirm the final tree contains MTP dictionary override propagation, HC prefill sharding, and recurrent checkpoint coalescing.

**Step 4: Port adaptive grace to the vLLM integration branch**

Apply vLLM #52917's final two-commit range to the integration branch. Reconcile conflicts with the fork's current `shm_broadcast`, env, test, and spinloop code. Do not retain the older fixed `VLLM_SHM_BROADCAST_BUSY_LOOP_S` implementation alongside the adaptive policy. Preserve the PR's runtime-only compile-factor exclusions, reader and writer policies, and architecture-gated WFET/WAITPKG behavior.

**Step 5: Prove source branch composition**

Run `git range-diff` from each upstream tip through the integration branch. Confirm each requested PR's semantic diff is present once, and record the immutable fork branch SHAs for the build profile.

**Step 6: Push the source branches**

Push the vLLM integration commit to `randomvariable/vllm` branch `dev/rv-jovian-judgement`, and the B12X integration commit to the matching `randomvariable/b12x` branch selected in Step 1. Do not open or modify upstream `local-inference-lab` PRs.

### Task 2: Make the profile lock refreshable

**Files:**
- Modify: `scripts/refresh-vllmb12x.py`
- Test: focused tests or a deterministic dry-run fixture for the refresh script, if the script's existing test surface supports one.

**Step 1: Define explicit source selection**

Extend the canonical refresh command so it accepts the vLLM and B12X remote/ref pair used by the profile, with defaults preserving the upstream `dev/jovian-judgement` to `master` pairing. Keep resolution to immutable full SHA values before writing any file.

**Step 2: Preserve profile-wide CMake synchronization**

Keep CMake dependency extraction from the selected vLLM checkout. Refresh every `vllm_cmake_*` source lock from that same checkout. Do not silently retain CMake dependency refs from the previous vLLM pin.

**Step 3: Add a deterministic dry-run regression**

Exercise remote/ref argument parsing and ensure the emitted profile changes both vLLM and B12X identities while leaving the checked-in profile untouched under `--dry-run`. Use fixture-level behavior only. Do not make a network-dependent unit test.

**Step 4: Verify source resolution failure safety**

Run the focused test and a live `--dry-run` against the two fork integration branches. Confirm that each declared source, including every refreshed CMake source, resolves before the command attempts a replacement.

### Task 3: Replace the adaptive-grace profile patch and pin integrated sources

**Files:**
- Modify: `third_party/vllm_shm_broadcast_spin_grace.patch`
- Modify: `third_party/BUILD.bazel`
- Modify: `profiles/vllmb12x/profile.json`
- Modify: `profiles/vllmb12x/version.bzl`
- Modify when vLLM changes package inputs: `profiles/vllmb12x/requirements.lock`, `profiles/vllmb12x/host-build-requirements.lock`
- Modify if the selected vLLM changes modules: `MODULE.bazel.lock`

**Step 1: Regenerate the profile patch**

Generate `third_party/vllm_shm_broadcast_spin_grace.patch` as the exact diff between the final vLLM integration commit before and after the adaptive-grace port. Require clean application with `patch --dry-run -p1` in a fresh checkout of the pinned vLLM source. Do not hand-edit patch context.

**Step 2: Record complete source identities**

Run the extended refresh command against the fork integration branches. Update `profile.json` and generated `version.bzl` with the resulting full source commits, remotes, ref text, and vLLM CMake dependency revisions. Retain the flash-attention namespace patch after confirming it applies to the new vLLM tree.

**Step 3: Refresh lockfiles only when inputs changed**

Compare the selected vLLM package requirements and Bazel module graph with the checked-in locks. Regenerate the two Python requirement locks and `MODULE.bazel.lock` only when the selected source changes their resolved input graph. Do not churn unchanged generated locks.

**Step 4: Verify repository-rule identity**

Run `just analyze` or its equivalent with the VLLMB12X configuration. Inspect the generated vLLM `source.identity.json` and confirm it records the final fork commit plus both declared patch digests. Confirm B12X records the final integration commit with no unintended source patch.

### Task 4: Publish the runtime configuration delta reference

**Files:**
- Add: `docs/reference/vllmb12x-runtime-configuration.md`
- Add: deterministic source-inventory generator and its focused test under `scripts/` and `tests/`, using the repository's existing conventions.
- Modify: `docs/reference/build-configuration.md`

**Step 1: Define the stock-vLLM comparison point**

Resolve and record the exact `vllm-project/vllm` merge-base from which the selected `local-inference-lab/vllm` source diverges. The document must name this immutable baseline and the final integrated vLLM and B12X commits. Do not compare moving branches or describe an option as fork-specific merely because the current upstream tip lacks it.

**Step 2: Generate the complete source inventory**

Build the reference from the final source trees, rather than selecting variables from a deployment manifest. Include every runtime control that the integration adds or changes relative to the declared stock-vLLM baseline:

- Local-inference-lab vLLM environment variables, CLI arguments, configuration fields, and `additional_config`/model configuration keys.
- Every B12X environment variable and B12X-specific configuration key consumed by the pinned B12X source.
- The semantic extension of an existing vLLM setting, such as #777 forwarding dictionary `--hf-overrides` into the MTP draft `ModelConfig`.
- Each runtime control added or changed by #777, #779, and the adaptive-grace port, attributed to the source commit or PR that introduced it.

The inventory generator must trace both registered environment accessors and direct environment reads, command-line parser/config declarations, and model `additional_config` consumers. Its `--check` mode must fail if the checked-in Markdown no longer matches the resolved source inventory. It must exclude unchanged stock-vLLM, CUDA, PyTorch, and NCCL controls. It must not infer a default from a recipe when the source uses no default.

**Step 3: Make each control operationally useful**

For every entry, document the exact name, surface, type or accepted values, source default or `unset`, effect, source location, provenance, and whether it is a supported serving control, an experimental tuning knob, a diagnostic, or a build-only setting. Document interactions and hard dependencies, including #779 coalescing's B12X #386 requirement. Distinguish the values the Qwen recipe currently sets from controls it does not set. Do not present an unset tuning or diagnostic control as a recommended deployment value.

**Step 4: Verify the reference**

Run the generator against fresh checkouts of the stock baseline, final vLLM integration commit, and final B12X integration commit. Run its focused test and `--check`. Link the completed reference from the build-configuration documentation and the VLLMB12X source-update notes.

### Task 5: Validate source-level behavior and artifact construction

**Files:**
- Test: vLLM upstream focused adaptive-grace test paths from #52917
- Test: B12X #384, #386, and #387 focused test paths
- Test: `//bazel:reproducibility_test`
- Test: `//tests/image:vllmb12x_contract`

**Step 1: Run focused vLLM tests**

Run the MTP override/Qwen tests introduced by #777, HC/coalescing tests from #779, and `tests/distributed/test_shm_broadcast.py` from #52917 against the final vLLM integration source.

**Step 2: Run focused B12X tests**

Run the collective-barrier test from #384, PLE checkpoint tests from #386, and QSA contract tests from #387 against the final B12X integration source. Preserve GPU-required tests as GPU checks. Do not claim a skipped test as passed.

**Step 3: Verify the reproducible-builder controls**

Run `bazel test //bazel:reproducibility_test --test_output=errors`. This covers profile-impacting source packaging and must pass before a source-lock update proceeds to the image build.

**Step 4: Build and run the image contract**

Use the existing ARM64 CI pipeline to build `//image:vllmb12x` and run `//tests/image:vllmb12x_contract`. The contract must pass before publication. The build cache remains durable at `/ccache/objects`; it survives worker-container replacement but not destruction of that backing volume.

**Step 5: Publish only after the contract succeeds**

Use `scripts/publish-vllmb12x.py` through the established nightly pipeline. Record the digest-qualified ARM64 image reference. Do not inspect every registry blob; a successful publication is the registry proof required here.

### Task 6: Review and submit

**Files:**
- Modify: all completed source-lock and patch files above.

**Step 1: Check the final diff**

Verify the diff contains only the source integration, refresh command support, source locks, patch replacement, configuration reference and inventory check, necessary generated locks, and focused tests.

**Step 2: Run static project diagnostics**

Run `aft_inspect` over the changed paths and resolve any introduced errors.

**Step 3: Commit and open the builder PR**

Commit the builder-profile changes on the dedicated branch, push it, and open one PR. Keep the source-fork commits referenced from the PR description and source-lock diff.

**Step 4: Hold deployment changes**

Do not repin an LWS deployment in this work. Deployment begins only after the image contract and publication produce a digest-qualified image.
