# VLLMB12X Source Update Notes

This file records upstream pull requests selected for the next Qwen source-lock update. It is not part of the current `profile.json` lock and does not claim that an image includes these changes. The build embeds the resolved lock and source identities in `/opt/vllmb12x/sources.lock.json` after the update.

## Integration Branches

- vLLM target: [`randomvariable/vllm:cycle/karmic-kraken`](https://github.com/randomvariable/vllm/tree/cycle/karmic-kraken) — the `local-inference-lab/vllm:integration/karmic-kraken-beta` tip plus exactly one commit, the bounded shared-memory broadcast waits of [#800](https://github.com/local-inference-lab/vllm/pull/800).
- B12X: the lock pins [`local-inference-lab/b12x:integration/karmic-kraken-beta`](https://github.com/local-inference-lab/b12x/tree/integration/karmic-kraken-beta) directly and carries no fork change. `randomvariable/b12x:cycle/karmic-kraken` is reset to that same tip, so it exists only to be forked from again.
- The vLLM branch is named for the upstream cycle because `scripts/refresh-vllmb12x.py` composes the published version from `source_ref`; that ref puts the local version segment at `karmic.kraken`.
- Do not push to `local-inference-lab` repositories. The source lock stores immutable commits, not these moving branch names.

## Already supplied by the beta base

The fork branches carried each of these as local commits. The pinned base now supplies them, so the fork branches no longer replay them and a new selection must not re-add them.

| Component | Reaching the base | Change |
| --- | --- | --- |
| vLLM | [#958](https://github.com/local-inference-lab/vllm/pull/958), merged as `85314135e0f9` | Native MXFP8 MTP draft experts through the ModelOpt B12X backend, with the per-rank intermediate at 32-aligned sizes. |
| B12X | [#453](https://github.com/local-inference-lab/b12x/pull/453), merged as `914921dad15d` | Native block-scaled MXFP8 W8A8 expert execution, including 32-aligned intermediate sizes. |
| B12X | [#384](https://github.com/local-inference-lab/b12x/pull/384), merged into the beta composition as `748fa9ea94f2` | Retained prepared launcher programs, one-shot RoCE dtype normalisation, collective preparation coordination and source-hashed preparation-memory extensions. |
| vLLM | [#777](https://github.com/local-inference-lab/vllm/pull/777), closed unmerged, whose added test `tests/config/test_speculative_draft_hf_overrides.py` is present in the base | Preserve dictionary `hf_overrides` when vLLM constructs the Qwen MTP draft model, carrying the target YaRN context geometry into the draft configuration. |

## Carried by the current lock

| Component | Change | Inclusion |
| --- | --- | --- |
| vLLM | [#800](https://github.com/local-inference-lab/vllm/pull/800) bounded shared-memory broadcast waits, the profile port of [vllm-project/vLLM #52917](https://github.com/vllm-project/vllm/pull/52917) | One commit on `cycle/karmic-kraken`, cherry-picked from `arm-spinloop-karmic-gated` (`29ebd2dfb82c`) onto `58d05bc7626d`. Open against `dev/karmic-kraken`; regenerate from the final integration tree before it lands. |
| vLLM | `third_party/vllm_flash_attn_cute_namespace.patch` | Applied at build time. Packaging repair for this builder's symlink install path, which skips the import rewrite the upstream CMake copy performs; carries no upstream change. |

## Retired selections, available for re-selection

None of these is merged upstream. Re-select by building the fork branch from the current `integration/karmic-kraken-beta` tip and cherry-picking in the order listed, resolving against that tip and recording only the resulting full SHA in `profile.json`.

| Component | Change | Where it survives |
| --- | --- | --- |
| vLLM | [#779](https://github.com/local-inference-lab/vllm/pull/779) partial port: Qwen HyperConnection prefill token-row ownership with deferred tensor-parallel reductions, and opt-in recurrent checkpoint coalescing. Nine commits. | `randomvariable/vllm:dev/rv-mxfp8-mtp` and `backup/cycle-karmic-kraken-before-reset`, both at `39e9ae0eabae` |
| vLLM | Six follow-on changes carried with that port: Qwen GDN layer-norm warmup sizing to the norm weight, prefill checkpoint blocks wired into the NVIDIA GDN decoder, AMD `Qwen4Exp` import and PLE guard restoration, an explicit deadline on preparation control reads, and two warmup-fixture alignments. | same |
| B12X | [#386](https://github.com/local-inference-lab/b12x/pull/386) export of prepared internal PLE prefill checkpoints: head commits `087b15b326a5` and `8801c016c429`, which supply the `export_checkpoint` API. | `randomvariable/b12x:feat/mxfp8-moe` and `backup/cycle-karmic-kraken-before-reset`, both at `efbc65547ff8` |
| B12X | [#387](https://github.com/local-inference-lab/b12x/pull/387) reuse of paged representative keys across paired four-head QSA queries, stacked on the #386 head branch `codex/qwen-ple-checkpoint-export`. Never carried by a published lock. | the open pull request branch |

Risk: the #779 coalescing path calls the #386 `export_checkpoint` API, so the two must be re-selected together.

## Runtime Configuration Reference

The source-lock update also publishes `docs/reference/vllmb12x-runtime-configuration.md`. It is a generated, checked inventory of every runtime environment variable and configuration option added or changed over the exact stock-vLLM merge-base by local-inference-lab/vLLM, B12X, and the selected pull requests. It separates supported serving controls from experimental tuning, diagnostics, and build-only controls, and it identifies the Qwen recipe's current settings without treating unset controls as recommendations.
