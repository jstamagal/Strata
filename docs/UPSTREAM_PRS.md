# Upstream pull requests carried by this branch

This branch carries the following open pull requests of [Niko1221/Strata](https://github.com/Niko1221/Strata) on top of
upstream `main`. The work is their authors'. Where a pull request did not apply cleanly it was ported by hand, so the
commit here may carry another author field; this table is the credit. "Changed here" says where the port differs from
the pull request as posted.

| PR | Author | What it does | Commit here | Changed here |
|---|---|---|---|---|
| [#910](https://github.com/Niko1221/Strata/pull/910) | @Hardin22 | Two-GPU decode: bounded attention merge (bitwise) and pipelined PLE read-ahead / early chain | `b8a81ef3`, `fa1f76be` | merged by hand into the N-stage pipeline code of `main` |
| [#1441](https://github.com/Niko1221/Strata/pull/1441) | @ruibeikaa | Prompt chunks sized for a layer split's pipeline (`--prefill-pipe`) | merge `9d51cb20` | none |
| [#1517](https://github.com/Niko1221/Strata/pull/1517) | @sergqwer | Adaptive tier: evictions reach the device's residency table before the next window | `be8b42df` | none |
| [#1533](https://github.com/Niko1221/Strata/pull/1533) | @gfifgfifofich | `--adapt-swaps` above 96 no longer capped; `--vram-reserve-mib` refuses a non-number | `3bfeabcf` | the two fixes only; the prompt-loan commit is left out |
| [#2016](https://github.com/Niko1221/Strata/pull/2016) | @libratechw | QSA top-k register kernel picks the radix digit by a cooperative scan | `c9f81a79` | no gfx1030 default for `STRATA_TOPK_REG_MIN_BLOCKS` |
| [#1749](https://github.com/Niko1221/Strata/pull/1749) | @vlupilin | The wide top-k kernel compiled for HIP | `56b11832` | top-k part only; opt-in on HIP (`STRATA_TOPK_STREAM=1`) |
| [#2042](https://github.com/Niko1221/Strata/pull/2042) | @Xiaomi2008 | `STRATA_EXCHANGE_ROTATE` for packs whose layers differ in expert block size | `b3a41605` | none |
| [#1949](https://github.com/Niko1221/Strata/pull/1949) | @maxious | The AVX2 dot kernels keep their accumulators in registers | `f122ca1d` | none |
| [#1368](https://github.com/Niko1221/Strata/pull/1368) | @sergqwer | Prefill quantizes each token's MoE input once and scatters it | `c63c0d11` | opt-in (`STRATA_QUANT_GATHER=0`); Q4_0 / Q4_1 keep the gather on HIP |
| [#2078](https://github.com/Niko1221/Strata/pull/2078) | @libratechw | Lent expert slots refilled from the GGUF mapping's pieces | `45589de3` | opt-in everywhere (`STRATA_REFILL_PARTS=1`) |
| [#1284](https://github.com/Niko1221/Strata/pull/1284) | @sergqwer | `--adapt-decay` outside (0, 1) is refused | `d282bdc0` | none |
| [#1305](https://github.com/Niko1221/Strata/pull/1305) | @ZackO2o | `--lookup-chain` 1 and 2 size the window cap from the cap in force | `31b2f97d` | none |
| [#1537](https://github.com/Niko1221/Strata/pull/1537) | @zsariboga | A checkpoint where a request continues the live session after a long reply | `c6ec7886` | off unless `STRATA_LIVE_CKPT_MIN=N` is set |
| [#1788](https://github.com/Niko1221/Strata/pull/1788) | @llj098 | `tool_choice` none keeps the tool declarations in the prompt | `bfa027b6` | none |
| [#1455](https://github.com/Niko1221/Strata/pull/1455) | @W1nge | serve reuses the encoding of an unchanged prompt prefix (`STRATA_PROMPT_REUSE=1`) | `d30706d5` | none |
| [#1536](https://github.com/Niko1221/Strata/pull/1536) | @zsariboga | serve reuses a reply's own token ids on the next turn | `7a5d70c7` | the drained-tokens part is also behind `STRATA_PROMPT_REUSE=1` |
| [#1525](https://github.com/Niko1221/Strata/pull/1525) | @sergqwer | Prefill: swiglu + q8_1 in one kernel, MoE combine + hyper-connection write fused | `7150d25e` | ported to HIP; both opt-in; the dequant part is left out |
| [#2253](https://github.com/Niko1221/Strata/pull/2253) | @alanthinker | serve: the conversation cache's RAM gate releases the heap after each eviction and stops evicting when it cannot help | `d9465021` | as written; not measured here |

Commit ids are those of the branch at the time this file was written; `git log --grep 'PR #'` finds them after a rebase.
