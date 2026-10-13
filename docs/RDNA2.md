# Strata on RDNA2 (Radeon RX 6000 / PRO V620, gfx103x)

This fork of [Strata](https://github.com/Niko1221/Strata) is aimed at AMD's RDNA2 cards. Upstream runs on them, but
its fast paths are written for NVIDIA and for AMD's newer cards (matrix cores, WMMA). RDNA2 has neither, so here the
hot kernels are rewritten for what the chip does have, and the engine turns them on by itself on a gfx103x card.

Everything below was measured on one PC: **2x Radeon PRO V620 (gfx1030, 32 GB each), Ryzen 9 9950X, 92 GB RAM,
Linux, ROCm with rocBLAS 5.8.0**, the two cards as a layer split, one conversation at a time. The first card sits on a
PCIe 4.0 x8 link and the second on x4. Other RDNA2 cards (RX 6800 / 6900 / 6700 XT, one card alone) build the same
code but have not been measured with it: a report is welcome.

## What you get

Measured with `tools/strata_bench.py` at temperature 1.0 / top-p 0.95 / top-k 20 (prompt speed) and
`tools/decode_probe.py` (decode: three fixed questions, 400 tokens each, greedy and seeded; one copy-heavy request).
"Stock + the two known tricks" is upstream with `STRATA_HIP_PROMPT_F16=1` and the rocBLAS tuning table.

| Model | | Prompt, 4K / 16K tokens | Decode, greedy / sampled | Copy-heavy answer | First token, agent turn |
|---|---|---:|---:|---:|---:|
| abliterated IQ3_S | stock + the two known tricks | 1,052 / 1,472 tok/s | 78.2 / 75.6 tok/s | 123 tok/s | 3.35 s |
| | this fork | 1,888 / 2,487 tok/s | 121.4 / 111.6 tok/s | 323 tok/s | 2.10 s |
| Swift 1.5 Q4_0 | stock, nothing set | 444 / 587 tok/s | | | 6.93 s |
| | stock + the two known tricks | 843 / 1,150 tok/s | 71.7 / 71.0 tok/s | 95 tok/s | 3.94 s |
| | this fork | about 1,880 / 2,440 tok/s | about 103 / 104 tok/s | about 210 tok/s | about 2.4 s |

One run per number. Prompt speeds repeat within 0.5%; the decode probe repeats within 0.1 tok/s where every expert is
in VRAM (the IQ3 models) and within about 2 tok/s where the adaptive tier moves experts (Q4_0, UD-Q4_K_XL).

## What is on by default on a gfx103x card

The engine sets these itself at start (a switch you set yourself is kept). `STRATA_GFX103X_DEFAULTS=0` turns all of
them off, which gives upstream's behaviour; `STRATA_GFX103X_EXPERIMENTAL=0` turns off only the second group.

### Proven

Bitwise equal to the kernels they replace (parity tests in `tests/hip/`), or the same prompt residuals, or the same
answers end to end.

| Switch | What it does | Measured |
|---|---|---|
| `STRATA_PF_STREAM_ROUTED=1`, `STRATA_PF_DIRECT=1`, `STRATA_PF_NL=1`, `STRATA_PF_Q51=1`, `STRATA_MMQ_RESIDENT_SORT_NE=1` | A prompt chunk copies only the experts its routing picked instead of every expert outside VRAM, and reads each expert where it lies | Q4_0 prompts +40% / +43%, IQ3_S +10% / +9% |
| `STRATA_PF_Q4_WIDE=1` | Q4_0 / Q4_1 prompt experts on 256 x 32 tiles with no shared-memory reads in the dot loop | kernel 3.5x; Q4_0 prompts +18% / +15% |
| `STRATA_DQ8=1`, `STRATA_HC_GATED16=1` | Dense dequant with 16-byte stores; the hyper-connection mix reads FP16 halves directly | IQ3_S prompts +6% |
| `STRATA_GDN_HEAD=1`, `STRATA_GDN_PP=2`, `STRATA_GDN_NOY=1` | The barrier-free GDN recurrence (upstream's, default there only on newer cards) | recurrence 1.76x; prompts +2% |
| `STRATA_HC_WIDE=3`, `STRATA_GR_NORM_FAST=1`, `STRATA_TSUM=1` | Hyper-connection reads one block per row in decode | decode +7% |
| `STRATA_EXP_MODE=8` | Decode routed experts on the shared-memory layouts | decode +3% |
| `STRATA_HEAD_WIDE=1`, `STRATA_ROUTE_PLAN=1`, `STRATA_HEAD_SAMPLE=1` | Head, router and sampler as fewer, wider launches | decode +1% |
| `STRATA_PL_LOOKUP_STREAK=12` | A copied stretch of text stays on 8-token lookup windows (needs `--pipeline-windows 2`) | copy-heavy answers 174 -> 302 tok/s |
| `STRATA_SH_STREAM=1`, `STRATA_QFUSE=1` | Upstream's decode switches | decode +12% / +14% together |

### Experimental

On by default as well, and measured faster here, but either not bitwise (rounding-level differences) or checked on
fewer models and context lengths. These are the ones to test more; say what you see.

| Switch | What it does | Why experimental |
|---|---|---|
| `STRATA_HIP_PROMPT_F16=1` | The prompt's 16-bit GEMMs run FP16 in and out | Not bit-identical to the default route. Prompts +73% to +112% |
| `STRATA_PA_DOT4=1` | A prompt attention kernel for int8 K/V on `v_dot4_i32_i8` | FP32-level error (2.8e-6 against an FP64 reference, the old kernel 2.1e-6), not bitwise. Needle recall checked to 128K. Prompts +4% |
| `STRATA_EXPERT_DOWN_L=1` | The 4-row down kernel behind any gate/up format | Bitwise on synthetic weights; one end-to-end run |
| `STRATA_KV_PREFETCH=1`, `STRATA_SPLIT_MTP_BATCH=1`, `STRATA_KV_PF_RESIDENT=1` | K/V staged ahead and read from VRAM slots on the prompt path; the draft layer's pass batched | Measured on IQ3_S with int8 K/V only. Prompts +7% to +9% from 16K to 128K |
| `STRATA_MMVQ_ROW=1`, `STRATA_MMVQ_SET=1`, `STRATA_NORM_ROPE=2` | Decode dense projections one wavefront per row, a layer's projections of one input in one launch, RMSNorm + RoPE in one launch | One A/B run, IQ3_S: decode 119.2 -> 122.1 greedy, 108.7 -> 111.2 sampled, copy-heavy 322 -> 336 tok/s, the same text |
| `STRATA_TOPK_STREAM=1`, `STRATA_TOPK_AUTO=1` | Wider top-k kernels for deep contexts | Same ids in the parity test; end-to-end gain only at depth |

## Settings that are not defaults

These are choices per model and per PC; they go in the model's config.

- `--pipeline-windows 2` on two cards: decode +12% greedy, +9% sampled, +24% on copy-heavy text.
- `--spec 8 --mtp-max-t 3 --spec-min-p 0.7`: MTP drafts three deep, copied text up to eight. Depth 3 measured best
  for ordinary text (IQ3_S greedy, by depth 2 to 8: 110.6, 114.6, 109.5, 105.0, 100.2, 94.2, 91.6 tok/s).
- Leave `STRATA_SPEC_COUPLED` off: with it a sampled request decodes serially (-7% to -11%).
- `--prefill auto:16384`, `STRATA_SPLIT_OWN=1`, `STRATA_PREFILL_HELP=1` on a layer split: prompts +18% to +34%.
- An even layer split with `--trim-stage-weights` (`"layer_split": "24"`). With the second card on a x4 link, 26 was
  1% to 3% faster than 24 for Q4_0 prompts; 22 and 28 were slower.
- `STRATA_HC_DOWNINJ=1` (down and inject as one GEMM, rounding-level): only with a rocBLAS table that has the N 324
  rows (`tools/hip/gfx1030-rocblas-5.8.0.3d8f334b.txt` does); without them it is slower.
- Doubling the context (`--max-context 524288` with YaRN) is free on the IQ3 models and costs a model that fills the
  cards (UD-Q4_K_XL) about 16% of its decode speed.
- No effect found on this PC: `--pool-workers` (0 to 12), `--host-core`, `STRATA_IO_THREADS`, `--pcie-frac`.

## Building

`setup.sh` builds the engine for the card it finds. The HIP kernels need three patches to llama.cpp
(`third_party/llama.cpp-patches/`); setup applies them when it unpacks llama.cpp.

## Not measured yet

One card alone; RX 6000 cards; Windows; contexts past 133K tokens; the CUDA and SYCL builds of the files this fork
touches (the new kernels are HIP-only or opt-in). Upstream pull requests carried here and their authors:
[UPSTREAM_PRS.md](UPSTREAM_PRS.md).
