# VLLMB12X Source Update Notes

This file records upstream pull requests selected for the next Qwen source-lock update. It is not part of the current `profile.json` lock and does not claim that an image includes these changes. The build embeds the resolved lock and source identities in `/opt/vllmb12x/sources.lock.json` after the update.

## Integration Branches

- vLLM target: [`randomvariable/vllm:cycle/karmic-kraken`](https://github.com/randomvariable/vllm/tree/cycle/karmic-kraken), rebased onto `local-inference-lab/vllm:integration/karmic-kraken-beta`.
- B12X target: [`randomvariable/b12x:cycle/karmic-kraken`](https://github.com/randomvariable/b12x/tree/cycle/karmic-kraken), rebased onto `local-inference-lab/b12x:integration/karmic-kraken-beta`.
- Each fork branch is named for the upstream cycle because `scripts/refresh-vllmb12x.py` composes the published version from `source_ref`; these refs put the local version segment at `karmic.kraken`. The superseded `dev/rv-mxfp8-mtp` and `feat/mxfp8-moe` names still resolve on the forks, at the same commits, and can be deleted once no `main` lock names them.
- Do not push to `local-inference-lab` repositories. The source lock stores immutable fork commits, not these moving branch names.

## Already supplied by the beta base

The fork branches carried each of these as local commits. The pinned base now supplies them, so the fork branches no longer replay them and a new selection must not re-add them.

| Component | Reaching the base | Change |
| --- | --- | --- |
| vLLM | [#958](https://github.com/local-inference-lab/vllm/pull/958), merged as `85314135e0f9` | Native MXFP8 MTP draft experts through the ModelOpt B12X backend, with the per-rank intermediate at 32-aligned sizes. |
| B12X | [#453](https://github.com/local-inference-lab/b12x/pull/453), merged as `914921dad15d` | Native block-scaled MXFP8 W8A8 expert execution, including 32-aligned intermediate sizes. |
| B12X | [#384](https://github.com/local-inference-lab/b12x/pull/384), merged into the beta composition as `748fa9ea94f2` | Retained prepared launcher programs, one-shot RoCE dtype normalisation, collective preparation coordination and source-hashed preparation-memory extensions. |
| vLLM | [#777](https://github.com/local-inference-lab/vllm/pull/777), closed unmerged, whose added test `tests/config/test_speculative_draft_hf_overrides.py` is present in the base | Preserve dictionary `hf_overrides` when vLLM constructs the Qwen MTP draft model, carrying the target YaRN context geometry into the draft configuration. |

## vLLM

| Pull request | Selected change | Status |
| --- | --- | --- |
| [local-inference-lab/vLLM #779](https://github.com/local-inference-lab/vllm/pull/779) | Add opt-in Qwen HyperConnection prefill row sharding and recurrent checkpoint coalescing. | Open against `dev/jovian-judgement`. The fork branch carries a nine-commit partial port to Qwen4Exp; HC and coalescing stay opt-in. Requires B12X #386 when `VLLM_QWEN3_8_PREFILL_COALESCE=1` is enabled. |
| [local-inference-lab/vLLM #800](https://github.com/local-inference-lab/vllm/pull/800) | Bounded shared-memory broadcast waits, the profile port of [vllm-project/vLLM #52917](https://github.com/vllm-project/vllm/pull/52917). | Open against `dev/karmic-kraken`. The fork branch carries it as one commit. Regenerate from the final integration tree before it lands. |

## B12X

| Pull request | Selected change | Status |
| --- | --- | --- |
| [local-inference-lab/b12x #386](https://github.com/local-inference-lab/b12x/pull/386) | Export prepared PLE internal prefill checkpoints for recurrent prefix-cache coalescing. | Open against `master`. The fork branch carries its two head commits, `087b15b326a5` and `8801c016c429`, which supply the `export_checkpoint` API that vLLM #779 coalescing calls. |
| [local-inference-lab/b12x #387](https://github.com/local-inference-lab/b12x/pull/387) | Reuse paged representative keys across paired four-head QSA queries. | Open, stacked on the #386 head branch `codex/qwen-ple-checkpoint-export`. Select after #386 to retain the paired Qwen qualification source set. |

The PR heads are not a linear Git stack. Build each fork integration branch from the current `integration/karmic-kraken-beta` tip, then cherry-pick only the open selections above. Resolve conflicts against that tip and record only the resulting full commit SHA in `profile.json`.

## Runtime Configuration Reference

The source-lock update also publishes `docs/reference/vllmb12x-runtime-configuration.md`. It is a generated, checked inventory of every runtime environment variable and configuration option added or changed over the exact stock-vLLM merge-base by local-inference-lab/vLLM, B12X, and the selected pull requests. It separates supported serving controls from experimental tuning, diagnostics, and build-only controls, and it identifies the Qwen recipe's current settings without treating unset controls as recommendations.
