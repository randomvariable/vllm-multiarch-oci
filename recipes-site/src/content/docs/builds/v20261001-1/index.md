---
title: "v20261001.1 build"
description: "The v20261001.1 publication: its digest, the revisions it was built from and the changes it carries."
editUrl: false
---
Published 2026-10-01.

Every build is assembled from pinned revisions of `local-inference-lab/vLLM` and its
dependencies. These are not generic upstream vLLM images, and the digest identifies
one build rather than a moving tag.

## What this build was made from

| Field | Value |
| --- | --- |
| Digest | `sha256:43aaf6d1b5e0b96bff630f5b5211afccd1b4838c40a29864aaee3eb7dae6cbde` |
| Publication tag | `vllmb12x-dev-rv-mxfp8-mtp-0a6739845a32-607a9619a9c3-20261001-n70` |
| vLLM | `0a6739845a32` |
| B12X | `965e748f74fe` |
| Runtime data | `353efc679f63` |
| Builder | `607a9619a9c3` |

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
| vLLM 0a6739845a32 | Native MXFP8 MTP through the ModelOpt B12X backend (no upstream pull request) | Merged into the pinned revision (4 commits) |
| vLLM 0a6739845a32 | Partial port of vLLM PR 779: Qwen HC ownership and checkpoint coalescing (no upstream pull request) | Merged into the pinned revision (9 commits) |
| vLLM 0a6739845a32 | [local-inference-lab/vllm#800 Bounded shared-memory broadcast waits](https://github.com/local-inference-lab/vllm/pull/800) | Merged into the pinned revision (1 commits) |
| vLLM 0a6739845a32 | Rewrite flash_attn.cute imports to vllm.vllm_flash_attn.cute (no upstream pull request) | Applied at build time by third_party/vllm_flash_attn_cute_namespace.patch |
| B12X 965e748f74fe | Native block-scaled MXFP8 W8A8 MoE execution (no upstream pull request) | Merged into the pinned revision (3 commits) |
| B12X 965e748f74fe | PLE checkpoint exports, prepared launcher closures and collective entry barriers (no upstream pull request) | Merged into the pinned revision (11 commits) |

## Run this build

```bash
docker pull ghcr.io/randomvariable/vllm-b12x-multi@sha256:43aaf6d1b5e0b96bff630f5b5211afccd1b4838c40a29864aaee3eb7dae6cbde
```

The [deployment flow](../../) selects a build, a model and its settings, then renders
the manifest, the Docker command and the Compose file. Opening it with this build
keeps every other choice at its default.

