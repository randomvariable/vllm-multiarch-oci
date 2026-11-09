# VLLMB12X Source Update Notes

This file records the selected source lock. Source selection does not establish that an image has been built or deployed.

## Integration Branches

- vLLM target: `randomvariable/vllm:auto/karmic-kraken-20261109` at `3ab11fdec715` — canonical `local-inference-lab/vllm:dev/karmic-kraken` at `8244d1c44348`, the advisory disk row-cache prefetch commit `c237f8373701`, PR #800's adaptive-spin commit replayed as `ce5f4148d44a`, the seven b12x row-cache and prefetch metrics commits, then `3ab11fdec715`, which switches the DeepSeek V4.1 indexer to FlashInfer's fused MXFP4 plan/run API. The canonical karmic base is unchanged from the previous lock, so this bump moves FlashInfer and that one adaptation. Canonical karmic uses the CSF source/block API shipped by the selected FlashInfer; lunar and karmic-beta still call the removed `CsfScalePlanes` API.
- FlashInfer target: `randomvariable/flashinfer:auto/karmic-kraken-20261109` at `accdb1a229c7` — `local-inference-lab/flashinfer:main` at `ea74ea610ce8` plus all twelve commits from PR #44, replayed as `cd736c5e3..accdb1a229c7`. `git range-diff` against the previous lock is all-equal: the row cache, the caller-gated `DiskTable.prefetch`, the `ple_hash.host` / `engram.host` row-id predictors and the five prefetch-telemetry and shutdown-race commits carried over without a single change, and the files they touch are byte-identical to the previous pin. The new base is 2 408 commits of upstream work, of which two matter to this lock: the MXFP4 DSA indexer is now a single fused plan/run (`573e91e07` removed the standalone score/select API that vLLM called, which is what `3ab11fdec715` adapts), and `B12X_AUTOTUNE_EFFORT` collapsed to `off`/`default`/`audit` (`60344962c`), rejecting the old `low`/`medium`/`high`/`max` spellings and `B12X_AUTOTUNE_EXHAUSTIVE`, which no shipped recipe sets.
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
| vLLM | [#800](https://github.com/local-inference-lab/vllm/pull/800) bounded shared-memory broadcast waits | One commit replayed on top of the prefetch commit, published on `auto/karmic-kraken-20261109`. |
| vLLM | Advisory disk row-cache prefetch for Qwen4Exp PLE and DeepSeek V4.1 Engram. No upstream pull request. | One commit `c237f8373701` on `auto/karmic-kraken-20261109`, behind `VLLM_PLE_PREFETCH`. |
| vLLM | b12x row-cache and prefetch Prometheus metrics, served by the API process on `/metrics`. No upstream pull request. | Seven commits on `auto/karmic-kraken-20261109` (`59f5be0030..e306090442`); the caches are sampled on a daemon thread, never on the step path. |
| vLLM | DeepSeek V4.1 MXFP4 DSA indexer on the fused plan/run API. No upstream pull request. | One commit `3ab11fdec715` on `auto/karmic-kraken-20261109`, required by the FlashInfer base that dropped the standalone score/select API. |
| FlashInfer | [#44](https://github.com/local-inference-lab/flashinfer/pull/44) C S3-FIFO host row cache for disk PLE/Engram tables | Twelve commits rebased onto `main` at `ea74ea610ce8`, published on `auto/karmic-kraken-20261109`: the five row-cache commits, the caller-gated `DiskTable.prefetch` and the `ple_hash.host` / `engram.host` row-id predictors, then five commits of prefetch telemetry that count requests, rows queued and dropped, jobs, errors, and rows loaded or cancelled only as the worker really published or released them, and reject a prefetch whose cache closed mid-call instead of queueing it behind a stopped worker. |
| vLLM | `third_party/vllm_flash_attn_cute_namespace.patch` | Applied at build time. Packaging repair for this builder's symlink install path, which skips the import rewrite the upstream CMake copy performs; carries no upstream change. |

Risk: `VLLM_PLE_PREFETCH` defaults off, and the hook only warns once per process when the pinned b12x lacks `prefetch(ids)` or when the row cache carries no `B12X_ROW_CACHE_ROWS`/`B12X_ROW_CACHE_BYTES` size; no shipped recipe sets the switch yet. The row-cache and prefetch metrics are published whether or not prefetch is enabled, because the cache serves synchronous lookups either way, and the only extra cost on a step is a published-snapshot read measured at 0.6 us median.

## Ledger verification after an upstream rewrite

`scripts/vllmb12x-included-changes.py --verify` counts a pull request's own commits as the head's history against its merge-base with the base, because GitHub's commit list includes whatever base commits the head branch carries. Two entries need care after this bump:

- `local-inference-lab/flashinfer#44` was cut from `main` at `9c9e94ef9aff`, and upstream then rewrote `main`, so the head is 173 commits from the currently recorded base while only twelve are the pull request's work. The entry declares `"verify_base": "9c9e94ef9aff73ef60949d41d4323652617d7f0f"` and the check reports `12 commits present`.
- `local-inference-lab/vllm#800` has grown to four commits and this lock replays only the adaptive-spin commit, so the entry records the pull request in its note with `pull_request: null` rather than claiming the whole pull request, and names it in `description_refs` so the registry description still lists it. Carrying the other three would be a separate decision, not a rebase.

`B12X_AUTOTUNE_EXHAUSTIVE` left the generated runtime configuration reference with this bump: upstream collapsed the tuning controls onto `B12X_AUTOTUNE_EFFORT=off|default|audit` (`60344962c`) and rejects the old effort spellings. No shipped recipe sets either name.

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
