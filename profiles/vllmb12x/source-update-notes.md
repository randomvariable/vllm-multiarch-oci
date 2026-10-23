# VLLMB12X Source Update Notes

This file records the selected source lock. Source selection does not establish that an image has been built or deployed.

## Integration Branches

- vLLM target: `randomvariable/vllm:cycle/karmic-kraken-canonical` — canonical `local-inference-lab/vllm:dev/karmic-kraken` at `8244d1c44348` plus PR #800, rebased as `37f4cc4cd309`. Canonical karmic uses the CSF source/block API shipped by the selected FlashInfer; lunar and karmic-beta still call the removed `CsfScalePlanes` API.
- FlashInfer target: `randomvariable/flashinfer:cycle/lunar-leviathan` — `local-inference-lab/flashinfer:main` at `9c9e94ef9aff` plus all five commits from PR #44, rebased to `0a2c4387a`. B12X now lives under FlashInfer's experimental package.
- The scale-swizzle compatibility patch updates both the intrinsics module cache and getter to `b12x._lib.intrinsics.reference`.
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
| vLLM | [#800](https://github.com/local-inference-lab/vllm/pull/800) bounded shared-memory broadcast waits | One commit rebased onto canonical `dev/karmic-kraken`, published on `cycle/karmic-kraken-canonical`. |
| FlashInfer | [#44](https://github.com/local-inference-lab/flashinfer/pull/44) C S3-FIFO host row cache for disk PLE/Engram tables | Five commits rebased onto `main`, published on `cycle/lunar-leviathan`. |
| vLLM | `third_party/vllm_flash_attn_cute_namespace.patch` | Applied at build time. Packaging repair for this builder's symlink install path, which skips the import rewrite the upstream CMake copy performs; carries no upstream change. |

## Retired selections, available for re-selection

These historical selections are not part of the lunar cycle request. Re-evaluate their upstream status and dependencies before selecting them again.

| Component | Change | Where it survives |
| --- | --- | --- |
| vLLM | [#779](https://github.com/local-inference-lab/vllm/pull/779) partial port: Qwen HyperConnection prefill token-row ownership with deferred tensor-parallel reductions, and opt-in recurrent checkpoint coalescing. Nine commits. | `randomvariable/vllm:dev/rv-mxfp8-mtp` and `backup/cycle-karmic-kraken-before-reset`, both at `39e9ae0eabae` |
| vLLM | Six follow-on changes carried with that port: Qwen GDN layer-norm warmup sizing to the norm weight, prefill checkpoint blocks wired into the NVIDIA GDN decoder, AMD `Qwen4Exp` import and PLE guard restoration, an explicit deadline on preparation control reads, and two warmup-fixture alignments. | same |
| B12X | [#386](https://github.com/local-inference-lab/b12x/pull/386) export of prepared internal PLE prefill checkpoints: head commits `087b15b326a5` and `8801c016c429`, which supply the `export_checkpoint` API. | `randomvariable/b12x:feat/mxfp8-moe` and `backup/cycle-karmic-kraken-before-reset`, both at `efbc65547ff8` |
| B12X | [#387](https://github.com/local-inference-lab/b12x/pull/387) reuse of paged representative keys across paired four-head QSA queries, stacked on the #386 head branch `codex/qwen-ple-checkpoint-export`. Never carried by a published lock. | the open pull request branch |

Risk: the #779 coalescing path calls the #386 `export_checkpoint` API, so the two must be re-selected together.

## Runtime Configuration Reference

The source-lock update also publishes `docs/reference/vllmb12x-runtime-configuration.md`. It is a generated, checked inventory of every runtime environment variable and configuration option added or changed over the exact stock-vLLM merge-base by local-inference-lab/vLLM, B12X, and the selected pull requests. It separates supported serving controls from experimental tuning, diagnostics, and build-only controls, and it identifies the Qwen recipe's current settings without treating unset controls as recommendations.
