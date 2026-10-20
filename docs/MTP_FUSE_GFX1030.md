MTP draft-chain fusion on two Radeon PRO V620 cards.

Measured on 2026-10-20. Both cards are gfx1030, wave32. ROCm 10.2.0, AMD Clang 24.0.0git (6da731510c23).
Measured in a git worktree of branch cx/mtp-chain.
Starting commit: e43d90f648620812a7e9ba17814424842428711d.

The model was sc117-abl IQ3_S. Layer split 24, pipeline windows 2, spec 8, MTP depth cap 3, draft min-p 0.7.
The config used int8 K/V, YaRN x2, maximum context 524288, and the shipped 40525-token draft head.
The prompt arm read 3197 tokens. It did not test the configured maximum context.

STRATA_MTP_FUSE=1 is opt-in and selected only on gfx1030. With it unset, the original GPU allocation and launch paths are used.
The unset arm matched the supplied saved control's text hashes, generated-token counts and draft counts.
The flag uses per-call APIs. It does not change the verifier's process-wide tuning switches.
All 64 verifier captures per arm had matching counts and kernel kinds.

The changes are:

- Q/K/V projections use one native_mmvq_set launch for a one-row step. Multi-row catch-up joins K/V; Q is projected after selecting the row.
- The shared expert's gate and up projections use one SET launch.
- Norm/RoPE uses the gfx1030 apply-expression form per call. The control already used that form for single-token Q/K. Multi-token K now uses one fused call per token, with the original NH-spaced positions.
- Attention gating and q8_1 quantization use one launch.
- Top-10 routing and the draft's grouped-resident plan use one launch. One token's ten distinct picks form ten singleton groups. IDs, weights, pointers, offsets, counts and entry maps were compared bit for bit.
- The Q6_K head reuses the wide head kernel. Its greedy merge and probability calculation share one launch. The probability retains row_top_prob's 1024-thread sum and add order.
- Mapped input staging uses one launch per round or step. At the largest round, redundant overlapping copies are omitted.

The draft has no learned QSA indexer, so the indexer projection pair does not apply.
Packed dense projections and unsupported SET shapes retain separate calls. Resident-tail rerouting retains its separate plan. Coupled sampling retains its original head path.

Captured draft steps fell from 57 to 50 kernels. Each captured round lost 12 kernels.
Counts include input staging, ID mapping, selection and teacher-force updates. All listed draft graph nodes were kernels.
STRATA_VERIFY_NODES=1 supplied the counts. Profiling stamps were off. Kernel names below are shortened to distinguish the variants.

| Draft-step kernel kind | Base | Fuse |
|---|---:|---:|
| `copy_i32_from_mapped_kernel` | 2 | 0 |
| `embed_rows_kernel` | 1 | 1 |
| `norm<1024>` | 2 | 2 |
| `native_quantize_q8_1_kernel` | 6 | 5 |
| `native_small_mmvq_kernel<Q8_0,8,false>` | 7 | 2 |
| `native_mmvq_multi_kernel<Q8_0,4 columns>` | 1 | 1 |
| `add_streams_broadcast_kernel` | 1 | 1 |
| `gr_norm_split_fast_kernel` | 2 | 2 |
| `gr_down_rows_kernel` | 2 | 2 |
| `gr_up_q32e_kernel` | 2 | 2 |
| `norm_rope_kernel` | 2 | 2 |
| `kv_append_q8_kernel` | 1 | 1 |
| `window_ids_kernel` | 1 | 1 |
| `attn_chunk_kernel_pre75` | 1 | 1 |
| `attn_merge_v2_kernel` | 1 | 1 |
| `gate` | 1 | 0 |
| `bf16_f32_mmvf_kernel<256>` | 3 | 3 |
| `route` | 1 | 0 |
| `group_resident_kernel` | 1 | 0 |
| `quantize_q8_0_scaled_kernel` | 1 | 1 |
| `gu_grouped_t_kernel` | 1 | 1 |
| `swiglu_quantize_q8_0_scaled_kernel` | 1 | 1 |
| `down_grouped_t_kernel` | 1 | 1 |
| `native_swiglu_quantize_q8_1_kernel` | 1 | 1 |
| `native_small_mmvq_kernel<Q8_0,8,true>` | 1 | 1 |
| `sigmoid_scale_rows_vec4_kernel` | 1 | 1 |
| `combine_k10_vec4` | 1 | 1 |
| `post` | 1 | 1 |
| `weighted_rms_norm<1024>` | 1 | 1 |
| `down_silu` | 1 | 1 |
| `bf16_f32_mmvf_kernel<160>` | 1 | 1 |
| `pre_gated` | 1 | 1 |
| `native_mmvq_row_kernel<Q6_K>` | 1 | 0 |
| `sampler_greedy_kernel` | 1 | 0 |
| `row_top_prob_kernel` | 1 | 0 |
| `map_ids_kernel` | 1 | 1 |
| `mtp_select_kernel` | 1 | 1 |
| `force_token_kernel` | 1 | 1 |
| `mtp_mapped_copies_kernel` | 0 | 1 |
| `native_mmvq_rowset_kernel` | 0 | 2 |
| `native_gate_quantize_q8_1_kernel` | 0 | 1 |
| `route_grouped` | 0 | 1 |
| `head_q6k_wide3_kernel<1>` | 0 | 1 |
| `head_pick_prob_kernel` | 0 | 1 |
| Total | 57 | 50 |

| Catch-up rows | Base kernels | Fuse kernels |
|---:|---:|---:|
| 1 | 62 | 50 |
| 2 | 66 | 54 |
| 3 | 69 | 57 |
| 4 | 70 | 58 |
| 5 | 73 | 61 |
| 6 | 74 | 62 |
| 7 | 78 | 66 |
| 8 | 79 | 67 |

The targeted HIP configure and build passed. Targets were strata, hip_mtp_fuse_parity, hip_norm_rope_parity, hip_qsa_fuse_parity and hip_head_parity.
The build used the supplied Release/gfx1030 configuration and -j16 with nice -n 10.
Every build and GPU command used /tmp/strata-gpu.lock. Each command had a timeout below ten minutes after acquiring the lock.
The final parity suite ran serially under one locked CTest command: 12/12 passed in 4.58 seconds.

- Both cards passed projection SET and head checks with STRATA_TSUM=0 and =1.
- Each MTP parity run checked 1024 changing router replays, 64 changing copy replays for each of 2/5/7 spans, and projection pairs/triples with Q8_0/Q4_0, 512/640 rows and 1..8 columns. Copy spans included the largest round's 81920-word residual.
- Each head run checked 76 new draft-head cases, plus 608 existing head cases and 400 staged-sampler cases. Logits, greedy IDs and probability bytes matched. Ties and non-finite inputs were included.
- The apply-expression norm/RoPE form matched 442368 values on each card. Gate plus q8_1 matched floats and quantized bytes for 14 shapes on each card. The existing BF16-pair check also passed 28 cases per card.

The parity checks used synthetic inputs. The real-model checks used saved text hashes and draft counters.

All four probes used this engine SHA256:
`7420af14ee811d3a5c64ec5cb2016391a2548e602a1b1415a3eebc6c469de175`.
A later test-only rebuild expanded the parity shapes. The engine checksum stayed unchanged.
The arms were copied from the supplied control. They differ only in log path and STRATA_MTP_FUSE=1.
Each probe used --warm and the supplied decode_probe.py prompts, seed and sampling settings. Runs alternated base, fuse, base, fuse.

| Run | Greedy tok/s | Sampled tok/s | Copy tok/s |
|---|---:|---:|---:|
| base-1 | 125.8 | 115.8 | 346.0 |
| fuse-1 | 126.9 | 116.0 | 346.4 |
| base-2 | 126.8 | 115.8 | 346.1 |
| fuse-2 | 126.9 | 115.9 | 347.0 |

Two-run arithmetic means, greedy / sampled / copy tok/s: base 126.3 / 115.8 / 346.1; fuse 126.9 / 116.0 / 346.7.
Observed mean changes: 0.5 / 0.1 / 0.2 percent. Greedy controls varied by 1.1 tok/s.
The speed changes are small. Two pairs do not establish a stable greedy speedup beyond that control variation.

Every run had hashes 2d3827caed25 (greedy) and 130618deee8b (sampled).
The full stored per-question hashes also matched. Kept was 80% greedy, 77% sampled and 98% copy in every run.
Exact per-question counts matched: greedy offered [276,170,131], accepted [248,124,95]; sampled offered [219,180,120], accepted [196,116,88].
Copy accepted 344 of 349 offered drafts in every run.

Every probe row follows, unchanged from cx/probe/probe-rows.txt:

```text
base-1                                       | T0  164.2  105.0  104.0 pooled  125.8 kept 80% gen 400/260/202 sha 2d3827caed25 | T1  161.3   95.2  100.0 pooled  115.8 kept 77% gen 320/273/196 sha 130618deee8b | prompt 3197 tok  1794.8 | copy  346.0 gen 400 kept 98% | serial n
fuse-1                                       | T0  166.6  105.6  104.7 pooled  126.9 kept 80% gen 400/260/202 sha 2d3827caed25 | T1  161.6   95.3  100.1 pooled  116.0 kept 77% gen 320/273/196 sha 130618deee8b | prompt 3197 tok  1799.2 | copy  346.4 gen 400 kept 98% | serial n
base-2                                       | T0  166.4  105.6  104.6 pooled  126.8 kept 80% gen 400/260/202 sha 2d3827caed25 | T1  161.4   95.1  100.0 pooled  115.8 kept 77% gen 320/273/196 sha 130618deee8b | prompt 3197 tok  1761.9 | copy  346.1 gen 400 kept 98% | serial n
fuse-2                                       | T0  166.5  105.6  104.7 pooled  126.9 kept 80% gen 400/260/202 sha 2d3827caed25 | T1  161.7   95.2  100.1 pooled  115.9 kept 77% gen 320/273/196 sha 130618deee8b | prompt 3197 tok  1805.9 | copy  347.0 gen 400 kept 98% | serial n
```

The probe commands were:

```sh
cd "$W"   # the measuring worktree
STRATA_VERIFY_NODES=1 flock /tmp/strata-gpu.lock timeout -k 10 -s INT 550 python3 tools/decode_probe.py cx/base.json base-1 --warm --out cx/probe
STRATA_VERIFY_NODES=1 flock /tmp/strata-gpu.lock timeout -k 10 -s INT 550 python3 tools/decode_probe.py cx/fuse.json fuse-1 --warm --out cx/probe
STRATA_VERIFY_NODES=1 flock /tmp/strata-gpu.lock timeout -k 10 -s INT 550 python3 tools/decode_probe.py cx/base.json base-2 --warm --out cx/probe
STRATA_VERIFY_NODES=1 flock /tmp/strata-gpu.lock timeout -k 10 -s INT 550 python3 tools/decode_probe.py cx/fuse.json fuse-2 --warm --out cx/probe
```

Not proven: other model packs, other GPUs, one-card serving, CUDA/SYCL/Metal builds, long contexts, prolonged stress and extra flag combinations.
Packed heads, Q4 draft heads, resident-tail rerouting, coupled sampling and RoPE tables were not probed end to end. The full project test suite was not run.
The supplied copy arm records no text hash. Its token and draft counts matched.
A sustained speed gain, especially for greedy decode, needs more measurements. The launch reduction and the recorded parity checks are established for this box and configuration.

Raw probe JSON and engine logs, capture counts and parity output stayed in the measuring worktree (cx/probe, cx/node-counts.json, cx/parity-suite-expanded.log); they are not part of this repository.
