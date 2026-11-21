# VLLMB12X Source Update Notes

This file records the selected source lock. Source selection does not establish that an image has been built or deployed.

## Integration Branches

- vLLM target: `randomvariable/vllm:cycle/monstrous-moonshine` at `851f8046784c`. Its base is `vllm-project/vllm` `main` at `d1ad1088bc`, not the frozen `local-inference-lab/vllm:dev/karmic-kraken`. On top: the stacked shm-broadcast pull requests #68441, #68444 and #68445; one commit porting the fork's b12x GB10 integration onto main, with upstream #65233, #65240 and #67236 folded in; one commit serving DeepSeek V4.1 Engram tables from the checkpoint files; one commit moving the b12x call sites to local-inference-lab FlashInfer's APIs; a Qwen4Exp config import fix; and one commit porting request-boundary recurrent checkpoints.
- FlashInfer target: `randomvariable/flashinfer:cycle/monstrous-moonshine` at `aaf6e76176a9` — `local-inference-lab/flashinfer` `main` at `1d76585b1` plus the twelve row-cache commits and the two GB10 tips. LIL `main` contains flashinfer-ai up to 2026-11-05 and is 271 commits ahead of `v0.7.4` under `experimental/b12x`; `v0.7.4` has no b12x commit that LIL lacks. It carries `paged_decode`, the PCIe/RoCE collectives and the CSF readers.
- The two scale-swizzle and pin-memory compatibility patches are retired: the vLLM branch reads the swizzle from `b12x._lib.intrinsics.reference` itself.
- Do not push to `local-inference-lab` repositories. The source lock stores immutable commits, not these moving branch names.

## Historical beta-base selections

The previous cycle recorded these changes as supplied by its base. This historical list is not a fresh audit of the lunar base and does not request additional cherry-picks.

| Component | Reaching the base | Change |
| --- | --- | --- |
| vLLM | [#958](https://github.com/local-inference-lab/vllm/pull/958), merged as `85314135e0f9` | Native MXFP8 MTP draft experts through the ModelOpt B12X backend, with the per-rank intermediate at 32-aligned sizes. |
| B12X | [#453](https://github.com/local-inference-lab/b12x/pull/453), merged as `914921dad15d` | Native block-scaled MXFP8 W8A8 expert execution, including 32-aligned intermediate sizes. |
| B12X | [#384](https://github.com/local-inference-lab/b12x/pull/384), merged into the beta composition as `748fa9ea94f2` | Retained prepared launcher programs, one-shot RoCE dtype normalisation, collective preparation coordination and source-hashed preparation-memory extensions. |
| vLLM | [#777](https://github.com/local-inference-lab/vllm/pull/777), closed unmerged, whose added test `tests/config/test_speculative_draft_hf_overrides.py` is present in the base | Preserve dictionary `hf_overrides` when vLLM constructs the Qwen MTP draft model, carrying the target YaRN context geometry into the draft configuration. |

## Carried by the current lock

| Component | Change | Inclusion |
| --- | --- | --- |
| vLLM | [#68441](https://github.com/vllm-project/vllm/pull/68441), [#68444](https://github.com/vllm-project/vllm/pull/68444), [#68445](https://github.com/vllm-project/vllm/pull/68445): bounded shared-memory broadcast waits | Three commits replayed onto `main` `d1ad1088bc`. |
| vLLM | b12x GB10 integration from the LIL karmic-kraken fork: V2 runner, worker and engine-core preparation; b12x attention, noncausal and GDN backends; Qwen4Exp b12x QSA, PLE, hyperconnection and MTP; PCIe/RoCE all-reduce; row-cache prefetch bridge and Prometheus relay. Folds in [#65233](https://github.com/vllm-project/vllm/pull/65233), [#65240](https://github.com/vllm-project/vllm/pull/65240) and [#67236](https://github.com/vllm-project/vllm/pull/67236). | One commit `72161c0f43`. Dropped: DCP-replicated hooks, adaptive-verification cost profiling, the `b12x` load format, and the fork's `ParallelLMHead` quantization API. |
| vLLM | DeepSeek V4.1 Engram tables served from checkpoint files (`engram-config.disk_offload`). No upstream pull request. | One commit `0e8f5041a1`. |
| vLLM | b12x call sites moved to local-inference-lab FlashInfer APIs. No upstream pull request. | One commit `ccf4a4ef07`: MoE compressed scales use b12x online CSF compression instead of the removed `CsfScalePlanes` encoder; the scale swizzle comes from `b12x._lib.intrinsics.reference`. |
| vLLM | Qwen4Exp b12x modules import their config from `transformers`. No upstream pull request. | One commit `dbcc086e84`: fixes an import break after the fork's config shim was removed; MTP index sharing is read from the draft model config (default off). |
| vLLM | Request-boundary recurrent checkpoints for Qwen3.8-Flash-Next (`--recurrent-checkpoint-policy`). No upstream pull request. | One commit `851f804678`, ported from the LIL fork for the LMCache recurrent checkpoint connector. Out of scope: GLM-5-Next, Kimi-K3/KDA, DFlash replay, DCP, LoRA and the fork's prefill-policy scheduler. |
| FlashInfer | [#44](https://github.com/local-inference-lab/flashinfer/pull/44) C S3-FIFO host row cache for disk PLE/Engram tables | Twelve commits replayed onto `local-inference-lab/flashinfer:main` at `1d76585b1`. |
| FlashInfer | GB10 DSV4.1 decode split policy and NVFP4 CSF scale defaults. No upstream pull request. | Two commits replayed onto `main` `1d76585b1`. |
| LMCache | `third_party/lmcache_upstream_vllm_kv_api.patch` | Applied at build time. Adapts the pinned LMCache fork's vLLM integration to upstream's KV-cache group API and config hashing; both connectors refuse KV groups marked non-transferable. |
| LMCache, DeepGEMM, InstantTensor | `third_party/{lmcache,deepgemm,instanttensor}_cxx20.patch` | Applied at build time. torch 2.14 headers need C++20. |
| vLLM | `third_party/vllm_flash_attn_cute_namespace.patch` | Applied at build time. Packaging repair for this builder's symlink install path; carries no upstream change. |

Risk: the image launcher maps LIL fork-only options to upstream equivalents or drops them when the installed vLLM is upstream lineage. `qwen38-flash-next-gb10-tp2` explicitly sets LIL scheduler options and fails to resolve on this image.

## Retired selections, available for re-selection

These historical selections are not part of this source request. Re-evaluate their upstream status and dependencies before selecting them again.

| Component | Change | Where it survives |
| --- | --- | --- |
| vLLM | [#779](https://github.com/local-inference-lab/vllm/pull/779) partial port: Qwen HyperConnection prefill token-row ownership with deferred tensor-parallel reductions, and opt-in recurrent checkpoint coalescing. Nine commits. | `randomvariable/vllm:dev/rv-mxfp8-mtp` and `backup/cycle-karmic-kraken-before-reset`, both at `39e9ae0eabae` |
| vLLM | Six follow-on changes carried with that port: Qwen GDN layer-norm warmup sizing to the norm weight, prefill checkpoint blocks wired into the NVIDIA GDN decoder, AMD `Qwen4Exp` import and PLE guard restoration, an explicit deadline on preparation control reads, and two warmup-fixture alignments. | same |
| B12X | [#386](https://github.com/local-inference-lab/b12x/pull/386) export of prepared internal PLE prefill checkpoints: head commits `087b15b326a5` and `8801c016c429`, which supply the `export_checkpoint` API. | `randomvariable/b12x:feat/mxfp8-moe` and `backup/cycle-karmic-kraken-before-reset`, both at `efbc65547ff8` |
| B12X | [#387](https://github.com/local-inference-lab/b12x/pull/387) reuse of paged representative keys across paired four-head QSA queries, stacked on the #386 head branch `codex/qwen-ple-checkpoint-export`. Never carried by a published lock. | the open pull request branch |

Risk: the #779 coalescing path calls the #386 `export_checkpoint` API, so the two must be re-selected together.

## Runtime Configuration Reference

The source-lock update also publishes `docs/reference/vllmb12x-runtime-configuration.md`. It is a generated, checked inventory of every runtime environment variable and configuration option added or changed over the exact stock-vLLM merge-base by local-inference-lab/vLLM, B12X, and the selected pull requests. It separates supported serving controls from experimental tuning, diagnostics, and build-only controls, and it identifies the Qwen recipe's current settings without treating unset controls as recommendations.
