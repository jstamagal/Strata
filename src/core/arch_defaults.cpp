// src/core/arch_defaults.cpp - see include/strata/core/arch_defaults.hpp.
#include "strata/core/arch_defaults.hpp"

#include "strata/kernels/gfx_arch.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {

namespace {
bool arch_is_gfx103x(const char* gcn) {
    return strata::kernels::gfx_arch_is(gcn, "gfx1030") || strata::kernels::gfx_arch_is(gcn, "gfx1031") ||
           strata::kernels::gfx_arch_is(gcn, "gfx1032");
}
}  // namespace

std::vector<std::pair<std::string, std::string>> arch_default_env(const char* gcn_arch) {
    std::vector<std::pair<std::string, std::string>> t;
    if (strata::kernels::gfx_arch_is_gfx12(gcn_arch)) {
        // gfx1200 / gfx1201 (RDNA4): only the PROMPT half of the gfx1151 table.  Reported exact (identical answers) on an R9700
        // in #1478 and re-checked bitwise here (notes-142-amd.md); the decode half measured slower on that card, so it stays out.
        // STRATA_HCD_EXACT is left out: it needs the gfx1151 hipBLASLt tuning table, finds none on gfx12, says so on every start
        // and does nothing (measured: with and without it identical, 0.0 to +0.1%).
        // STRATA_GDN_CONVL2 is AMD-only on purpose: it is not bitwise on CUDA 13.3 sm_120 (#1522), and this table is never
        // consulted by a CUDA build.
        const char* off12 = std::getenv("STRATA_GFX12_DEFAULTS");
        if (off12 != nullptr && off12[0] == '0') return t;
        t = {
            {"STRATA_GDN_HEAD", "1"},
            {"STRATA_GDN_PP", "2"},
            {"STRATA_GDN_CONVL2", "1"},
            {"STRATA_GDN_NOY", "1"},
            {"STRATA_CVEC_FUSE", "1"},
        };
        return t;
    }
    if (arch_is_gfx103x(gcn_arch)) {
        // gfx1030 / gfx1031 / gfx1032 (RDNA2), measured on 2x Radeon PRO V620 (docs/AMD_HIP.md, docs/RDNA2.md).
        // STRATA_GFX103X_DEFAULTS=0 turns the whole table off; STRATA_GFX103X_EXPERIMENTAL=0 only its second half.
        const char* off103 = std::getenv("STRATA_GFX103X_DEFAULTS");
        if (off103 != nullptr && off103[0] == '0') return t;
        // PROVEN: bitwise equal to the default kernels (parity tests) or the same residuals / the same answers end to end.
        t = {
            // prompt
            {"STRATA_GDN_HEAD", "1"},              // the barrier-free GDN recurrence: bitwise; recurrence 1.76x
            {"STRATA_GDN_PP", "2"},
            {"STRATA_GDN_NOY", "1"},
            {"STRATA_PF_Q4_WIDE", "1"},            // Q4_0 / Q4_1 prompt experts on tiles without LDS reads: bitwise; Q4_0 prompts +15-18%
            {"STRATA_DQ8", "1"},                   // dense dequant with 16-byte stores: bitwise; with HC_GATED16 prompts +6%
            {"STRATA_HC_GATED16", "1"},            // the hyper-connection mix reads the up projection's halves: bitwise
            {"STRATA_PF_STREAM_ROUTED", "1"},      // only the routed streamed experts are copied: same residuals; Q4_0 prompts +40%
            {"STRATA_PF_DIRECT", "1"},             // experts read where they lie (llama.cpp patch 0003): same residuals
            {"STRATA_PF_NL", "1"},                 // IQ4_NL down products on the no-LDS tile: MMQ's bits
            {"STRATA_PF_Q51", "1"},                // Q5_1 down products on the same tile: MMQ's bits
            {"STRATA_MMQ_RESIDENT_SORT_NE", "1"},  // same bits
            // decode
            {"STRATA_SH_STREAM", "1"},             // the shared expert on its own stream
            {"STRATA_QFUSE", "1"},                 // activation q8_1 images written by their producers
            {"STRATA_HC_WIDE", "3"},               // hyper-connection reads one block per row: bitwise; with the next two decode +7%
            {"STRATA_GR_NORM_FAST", "1"},          // same answers
            {"STRATA_TSUM", "1"},                  // several warp sums as one transposed butterfly: same answers
            {"STRATA_EXP_MODE", "8"},              // routed experts on the LDS layouts: bitwise; decode +3-4%
            {"STRATA_HEAD_WIDE", "1"},             // head + greedy pick + non-finite check in one launch: bitwise
            {"STRATA_ROUTE_PLAN", "1"},            // router top-10 and the all-resident plan in one launch: bitwise
            {"STRATA_HEAD_SAMPLE", "1"},           // sampled draws as nodes of the window graph: bitwise
            {"STRATA_PL_LOOKUP_STREAK", "12"},     // a copied stretch stays on lookup windows: same answers; copy-heavy text +70%
        };
        const char* exp103 = std::getenv("STRATA_GFX103X_EXPERIMENTAL");
        if (exp103 != nullptr && exp103[0] == '0') return t;
        // EXPERIMENTAL: on, and measured faster on that box, but either rounding-level (not bitwise) or checked on fewer
        // models and contexts.  More testing is wanted on each (docs/RDNA2.md says what).
        const std::pair<const char*, const char*> exp[] = {
            {"STRATA_HIP_PROMPT_F16", "1"},        // the prompt's 16-bit GEMMs FP16 in and out: not bit-identical; prompts +73-112%
            {"STRATA_PA_DOT4", "1"},               // int8-KV prompt attention on v_dot4_i32_i8: FP32-level error, not bitwise; prompts +4%
            {"STRATA_EXPERT_DOWN_L", "1"},         // the 4-row down kernel behind any gate/up format: bitwise in the lab, one end-to-end run
            {"STRATA_KV_PREFETCH", "1"},           // the next streamed layer's K/V staged ahead: prompts +5% at depth
            {"STRATA_SPLIT_MTP_BATCH", "1"},       // the draft layer's pass after a prompt chunk batched (layer split)
            {"STRATA_TOPK_STREAM", "1"},           // the wide top-k kernel on HIP: same ids in the parity test
            {"STRATA_TOPK_AUTO", "1"},             // decode top-k picks its kernel by the real depth: same ids in the parity test
            {"STRATA_KV_PF_RESIDENT", "1"},        // the prompt path reads streamed K/V from the VRAM slots: bit-exact on six requests
            {"STRATA_MMVQ_ROW", "1"},              // decode dense projections one wavefront per row: same text, one A/B run
            {"STRATA_MMVQ_SET", "1"},              // a layer's projections of one input in one launch
            {"STRATA_NORM_ROPE", "2"},             // RMSNorm + RoPE in one launch
        };
        for (const auto& kv : exp) t.emplace_back(kv.first, kv.second);
        return t;
    }
    if (!strata::kernels::gfx_arch_is_gfx1151(gcn_arch)) return t;
    const char* off = std::getenv("STRATA_GFX1151_DEFAULTS");
    if (off != nullptr && off[0] == '0') return t;
    // Exact (bitwise) on Aurora, each with its measured gain in aurora_s23.md:
    t = {
        // prompt
        {"STRATA_GDN_HEAD", "1"},        // the GDN recurrence, four lanes a column + the grid-stride norm (S23)
        {"STRATA_GDN_PP", "2"},          //   two columns per lane: 64K prompt +2.9% (stream A, round 2)
        {"STRATA_GDN_CONVL2", "1"},      // conv + q/k L2 norm in one kernel: 64K -0.3 s (stream A)
        {"STRATA_GDN_NOY", "1"},         // no dead FP32 store of the norm (stream A)
        {"STRATA_CVEC_FUSE", "1"},       // a steered layer's write, control vector and next norm in one pass (S23)
        {"STRATA_HCD_EXACT", "1"},       // the HC down GEMM in hipBLASLt solution 1176's k order: 64K +1.8% (stream S)
        {"STRATA_PF_PAD", "1"},          // padded GEMM row strides (acts with STRATA_PF_GEMM; inert without it) (S23)
        {"STRATA_TOPK_REG_MIN_BLOCKS", "4096"},   // the register top-k from 16K cells, not 28K: same ids; per 2,048 queries the ref takes 1.0 / 2.4 / 3.9 ms at 16K / 20K / 24K cells, the scan 0.87 / 0.93 / 1.00
        // decode
        {"STRATA_Q8_PACKED", "1"},       // the packed Q8_0 decode layout: +2.3% (S26)
        {"STRATA_Q6_PACKED", "1"},       // the packed Q6_K heads (S26)
        {"STRATA_MMVF_ROWS", "1"},       // bf16 multi-row GEMV, 4 rows per block (S25)
        {"STRATA_ATTN_LANECELL", "1"},   // decode attention scores, one cell per thread (S25)
        {"STRATA_EXPERT_V2", "1"},       // the grouped decode experts, IQ3_S gate/up + IQ4_NL down (S26)
        {"STRATA_TSUM", "1"},            // several warp sums as one transposed butterfly (S26)
        {"STRATA_LFUSE", "1"},           // fewer launches around the shared expert and the KV append (S26)
        {"STRATA_GDN_SPLIT", "1"},       // the GDN step over 4 blocks per head (S25/S26)
        {"STRATA_QFUSE", "1"},           // activation q8_1 images written by their producers (S26)
        {"STRATA_PLE_BATCH", "1"},       // the verify window's PLE key / value projections at once (S25)
        {"STRATA_SH_STREAM", "1"},       // the shared expert on its own stream: decode +1.8% / +6.7% (UD-Q4_K_XL) (140-m)
    };
    return t;
}

std::vector<std::string> apply_arch_defaults(const char* gcn_arch) {
    std::vector<std::string> set;
    for (const auto& kv : arch_default_env(gcn_arch)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;   // the user's setting wins, whatever it is
#if defined(_WIN32)
        _putenv_s(kv.first.c_str(), kv.second.c_str());
#else
        setenv(kv.first.c_str(), kv.second.c_str(), 0);
#endif
        set.push_back(kv.first);
    }
    if (!set.empty()) {
        const bool g12 = strata::kernels::gfx_arch_is_gfx12(gcn_arch);
        const bool g103 = arch_is_gfx103x(gcn_arch);
        if (g103)
            std::fprintf(stderr, "strata: gfx103x (RDNA2): %zu speed switches on by default, proven and experimental "
                                 "(STRATA_GFX103X_DEFAULTS=0 turns all off, STRATA_GFX103X_EXPERIMENTAL=0 the experimental "
                                 "ones: PROMPT_F16 PA_DOT4 EXPERT_DOWN_L KV_PREFETCH SPLIT_MTP_BATCH TOPK_STREAM TOPK_AUTO "
                                 "KV_PF_RESIDENT MMVQ_ROW MMVQ_SET NORM_ROPE; a switch you set is kept): ", set.size());
        else
        std::fprintf(stderr, "strata: %s: %zu exact speed switches on by default (%s=0 turns "
                             "them off; a switch you set is kept): ",
                     g12 ? "gfx12 (RDNA4)" : "gfx1151 (Strix Halo)", set.size(),
                     g12 ? "STRATA_GFX12_DEFAULTS" : "STRATA_GFX1151_DEFAULTS");
        for (size_t i = 0; i < set.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", set[i].c_str() + 7);
        std::fprintf(stderr, "\n");
    }
    return set;
}

}  // namespace strata::core
