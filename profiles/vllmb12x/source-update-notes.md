# VLLMB12X Source Update Notes

This file records upstream pull requests selected for the next Qwen source-lock update. It is not part of the current `profile.json` lock and does not claim that an image includes these changes. The build embeds the resolved lock and source identities in `/opt/vllmb12x/sources.lock.json` after the update.

## Integration Branches

- vLLM target: [`randomvariable/vllm:dev/rv-jovian-judgement`](https://github.com/randomvariable/vllm/tree/dev/rv-jovian-judgement), based on `local-inference-lab/vllm:dev/jovian-judgement`.
- B12X target: use the matching `randomvariable/b12x:dev/rv-jovian-judgement` ref unless it conflicts, then record the selected B12X ref in `profile.json`.
- Do not push to `local-inference-lab` repositories. The source lock stores immutable fork commits, not these moving branch names.

## vLLM

| Pull request | Selected change | Integration dependency |
| --- | --- | --- |
| [local-inference-lab/vLLM #777](https://github.com/local-inference-lab/vllm/pull/777) | Preserve dictionary `hf_overrides` when vLLM constructs the Qwen MTP draft model. This carries target YaRN context geometry into the draft configuration. | Apply before #779. |
| [local-inference-lab/vLLM #779](https://github.com/local-inference-lab/vllm/pull/779) | Add opt-in Qwen HyperConnection prefill row sharding and recurrent checkpoint coalescing. | Requires B12X #386 when `VLLM_QWEN3_8_PREFILL_COALESCE=1` is enabled. |
| [vllm-project/vLLM #52917](https://github.com/vllm-project/vllm/pull/52917) | Replace fixed shared-memory broadcast spinning with adaptive reader and writer grace periods, plus bounded Arm WFET and Intel WAITPKG waits where the CPU supports them. | Port after #777 and #779. Regenerate the profile patch from the final integration tree. |

## B12X

| Pull request | Selected change | Integration dependency |
| --- | --- | --- |
| [local-inference-lab/b12x #384](https://github.com/local-inference-lab/b12x/pull/384) | Retain prepared launcher programs, normalize one-shot RoCE dtypes, coordinate collective preparation, and hash preparation-memory extensions by source. | Apply first. |
| [local-inference-lab/b12x #386](https://github.com/local-inference-lab/b12x/pull/386) | Export prepared PLE internal prefill checkpoints for recurrent prefix-cache coalescing. | Required by vLLM #779 coalescing. |
| [local-inference-lab/b12x #387](https://github.com/local-inference-lab/b12x/pull/387) | Reuse paged representative keys across paired four-head QSA queries. | Apply after #386 to retain the paired Qwen qualification source set. |

The PR heads are not a linear Git stack. Build each fork integration branch from its named upstream tip, then cherry-pick the listed changes in the order above. Resolve conflicts against that tip and record only the resulting full commit SHA in `profile.json`.

## Runtime Configuration Reference

The source-lock update also publishes `docs/reference/vllmb12x-runtime-configuration.md`. It is a generated, checked inventory of every runtime environment variable and configuration option added or changed over the exact stock-vLLM merge-base by local-inference-lab/vLLM, B12X, and the selected pull requests. It separates supported serving controls from experimental tuning, diagnostics, and build-only controls, and it identifies the Qwen recipe's current settings without treating unset controls as recommendations.
