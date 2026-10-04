---
title: "v20260917.1 build"
description: "The v20260917.1 publication: its digest, the revisions it was built from and the changes it carries."
editUrl: false
---
Published 2026-09-17.

Every build is assembled from pinned revisions of `local-inference-lab/vLLM` and its
dependencies. These are not generic upstream vLLM images, and the digest identifies
one build rather than a moving tag.

## What this build was made from

| Field | Value |
| --- | --- |
| Digest | `sha256:cdf2602bb2e42da5b8f8cae8ae4bbe0842388cd6d8bf6ca2e6e3c27d7e27ad1e` |
| Publication tag | `vllmb12x-dev-rv-jovian-judgement-profile-base-bd22e0f25043-13527e984717-20260917-n59` |
| vLLM | `bd22e0f25043` |
| B12X | `4401ce4fb982` |
| Runtime data | `353efc679f63` |
| Builder | `13527e984717` |

A field that reads `not recorded` predates the pin that would fill it. Treat it as
unknown rather than as matching the current source.

## Carried by this build

No additions were recorded for this build.

## Included upstream changes

The image is built from pinned fork revisions rather than from upstream branches, so
this records which upstream changes the current lock carries.

| Component | Change | Included as |
| --- | --- | --- |
| vLLM bd22e0f25043 | [local-inference-lab/vllm#777 fix(qwen): propagate MTP positional overrides](https://github.com/local-inference-lab/vllm/pull/777) | Merged into the pinned revision (3 commits) |
| vLLM bd22e0f25043 | [local-inference-lab/vllm#779 perf(qwen): shard TP4 HC prefill and coalesce recurrent checkpoints](https://github.com/local-inference-lab/vllm/pull/779) | Merged into the pinned revision (9 commits) |
| vLLM bd22e0f25043 | [vllm-project/vllm#52917 Adaptive spin grace and bounded architectural waits for shm_broadcast](https://github.com/vllm-project/vllm/pull/52917) | Applied at build time by third_party/vllm_shm_broadcast_spin_grace.patch |
| vLLM bd22e0f25043 | Rewrite flash_attn.cute imports to vllm.vllm_flash_attn.cute (no upstream pull request) | Applied at build time by third_party/vllm_flash_attn_cute_namespace.patch |
| B12X 4401ce4fb982 | [local-inference-lab/b12x#384 fix(preparation): retain prepared launchers and coordinate collectives](https://github.com/local-inference-lab/b12x/pull/384) | Merged into the pinned revision (10 commits) |
| B12X 4401ce4fb982 | [local-inference-lab/b12x#386 feat(ple): export prepared internal prefill checkpoints](https://github.com/local-inference-lab/b12x/pull/386) | Merged into the pinned revision (2 commits) |
| B12X 4401ce4fb982 | [local-inference-lab/b12x#387 perf(qsa): reuse representative keys across paired queries](https://github.com/local-inference-lab/b12x/pull/387) | Merged into the pinned revision (4 commits) |

## Run this build

```bash
docker pull ghcr.io/randomvariable/vllm-b12x-multi@sha256:cdf2602bb2e42da5b8f8cae8ae4bbe0842388cd6d8bf6ca2e6e3c27d7e27ad1e
```

The [deployment flow](../../) selects a build, a model and its settings, then renders
the manifest, the Docker command and the Compose file. Opening it with this build
keeps every other choice at its default.

