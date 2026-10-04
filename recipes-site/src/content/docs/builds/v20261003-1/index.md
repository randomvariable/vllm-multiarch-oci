---
title: "v20261003.1 build"
description: "The v20261003.1 publication: its digest, the revisions it was built from and the changes it carries."
editUrl: false
---
Published 2026-10-03.

Every build is assembled from pinned revisions of `local-inference-lab/vLLM` and its
dependencies. These are not generic upstream vLLM images, and the digest identifies
one build rather than a moving tag.

## What this build was made from

| Field | Value |
| --- | --- |
| Digest | `sha256:17ef10a5323b6fa58aad4a96bf7237e1c1430b70dedce64f0a62f26bcc56570a` |
| Publication tag | `vllmb12x-cycle-karmic-kraken-28d82aa6479a-db2e3bec08e3-20261003-n74` |
| vLLM | `28d82aa6479a` |
| B12X | `a77b3f85e5e2` |
| Runtime data | `353efc679f63` |
| Builder | `db2e3bec08e3` |

A field that reads `not recorded` predates the pin that would fill it. Treat it as
unknown rather than as matching the current source.

## Carried by this build

- Mooncake Transfer Engine CUDA 13 0.3.13.post1 provides the mooncake Python package and its service and benchmark commands for deployments that select vLLM's MooncakeStoreConnector. It does not enable a connector or start Mooncake services.
- See Mooncake Transfer Engine for the installed surface and deployment boundary.

## Included upstream changes

The image is built from pinned fork revisions rather than from upstream branches, so
this records which upstream changes the current lock carries.

| Component | Change | Included as |
| --- | --- | --- |
| vLLM 28d82aa6479a | [local-inference-lab/vllm#800 Bounded shared-memory broadcast waits](https://github.com/local-inference-lab/vllm/pull/800) | Merged into the pinned revision (1 commits) |
| vLLM 28d82aa6479a | Rewrite flash_attn.cute imports to vllm.vllm_flash_attn.cute (no upstream pull request) | Applied at build time by third_party/vllm_flash_attn_cute_namespace.patch |

## Run this build

```bash
docker pull ghcr.io/randomvariable/vllm-b12x-multi@sha256:17ef10a5323b6fa58aad4a96bf7237e1c1430b70dedce64f0a62f26bcc56570a
```

The [deployment flow](../../) selects a build, a model and its settings, then renders
the manifest, the Docker command and the Compose file. Opening it with this build
keeps every other choice at its default.

