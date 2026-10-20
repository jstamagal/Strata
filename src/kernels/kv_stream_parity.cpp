// src/kernels/kv_stream_parity.cpp - KV streaming (kv_stream.hpp) against a fully resident pool (GPU, no model).
//
// Decodes a synthetic sequence twice: into a fully resident pool with the identity page table, and into a streamed
// layer (host copy + a small VRAM slot pool that must evict constantly). Every few cells a batch of 1-8 queries
// with selections shaped like the indexer's (whole 4-cell blocks, a partial threshold block, the tail block, half of
// them recent) is attended in both; `kv_stream_resolve` runs before the streamed attention. Checks:
//   1. the attention outputs are BITWISE equal (the streamed readers see exactly the resident values);
//   2. the residency map is consistent after every call (slot_block and page_table invert each other);
//   3. no call overflowed, and the hit/miss counters add up;
//   4. a ring (the MTP drafter's layout) restored from the host copy reads the same values as the resident pool;
//   5. kv_stream_resolve_range (the prompt path's STRATA_KV_PF_RESIDENT=1) leaves a whole run of blocks resident, and
//      the attention reads the resident pool's values from it with no resolve of its own.
// INT8, FP16, Q4_0 (PR #21) and hybrid K8V4 pools; K8V4's appends are the engine's folded calls (layer.cpp), its host
// copy written through kv_hybrid_k_half / kv_hybrid_v_half.
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <vector>

namespace k = strata::kernels;

namespace {
int g_fail = 0;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}
template <typename T> T* halloc(size_t n) {   // pinned, mapped; returns the device pointer
    void* h = nullptr;
    void* d = nullptr;
    ck(cudaHostAlloc(&h, n * sizeof(T) + 64, cudaHostAllocMapped), "hostalloc");
    std::memset(h, 0, n * sizeof(T) + 64);
    ck(cudaHostGetDevicePointer(&d, h, 0), "devptr");
    return (T*) d;
}

struct Pools {   // one K/V pool set of `pages` pages
    k::KvHostPools p;   // reused as a plain pointer bundle
    void alloc(int64_t pages, const k::QsaShapes& s, int fmt, bool host) {
        const size_t rows = (size_t) pages * s.n_head_kv * s.page_size;
        if (fmt == k::kKvHybrid) {
            const size_t b = rows * k::kv_q4_bytes_per_head((int) s.head_dim);
            p.k_q = host ? halloc<int8_t>(rows * s.head_dim) : dalloc<int8_t>(rows * s.head_dim);
            p.k_scale = host ? halloc<uint16_t>(rows * 4) : dalloc<uint16_t>(rows * 4);
            p.v_q4 = host ? halloc<uint8_t>(b) : dalloc<uint8_t>(b);
        } else if (fmt == k::kKvQ4) {
            const size_t b = rows * k::kv_q4_bytes_per_head((int) s.head_dim);
            p.k_q4 = host ? halloc<uint8_t>(b) : dalloc<uint8_t>(b);
            p.v_q4 = host ? halloc<uint8_t>(b) : dalloc<uint8_t>(b);
        } else if (fmt == k::kKvInt8) {
            p.k_q = host ? halloc<int8_t>(rows * s.head_dim) : dalloc<int8_t>(rows * s.head_dim);
            p.v_q = host ? halloc<int8_t>(rows * s.head_dim) : dalloc<int8_t>(rows * s.head_dim);
            p.k_scale = host ? halloc<uint16_t>(rows * 4) : dalloc<uint16_t>(rows * 4);
            p.v_scale = host ? halloc<uint16_t>(rows * 4) : dalloc<uint16_t>(rows * 4);
        } else {
            p.k_pool = host ? halloc<uint16_t>(rows * s.head_dim) : dalloc<uint16_t>(rows * s.head_dim);
            p.v_pool = host ? halloc<uint16_t>(rows * s.head_dim) : dalloc<uint16_t>(rows * s.head_dim);
        }
    }
    k::QsaAttnPools attn(const int32_t* table) const {
        k::QsaAttnPools a;
        a.k_pool = p.k_pool; a.v_pool = p.v_pool; a.k_q = p.k_q; a.v_q = p.v_q; a.k_scale = p.k_scale;
        a.v_scale = p.v_scale; a.k_q4 = p.k_q4; a.v_q4 = p.v_q4; a.page_table = table;
        return a;
    }
};

void append(const Pools& pl, const int32_t* table, const int32_t* step, const float* kc, const float* vc,
            const k::QsaShapes& s, int fmt, const k::KvHostPools* host) {
    if (fmt == k::kKvHybrid) {   // exactly as qsa_layer: each half folded onto its own pool, host copy by halves
        const k::KvHostPools hk = host ? k::kv_hybrid_k_half(*host) : k::KvHostPools{},
                             hv = host ? k::kv_hybrid_v_half(*host) : k::KvHostPools{};
        k::kv_append_q8_step(pl.p.k_q, pl.p.k_q, pl.p.k_scale, pl.p.k_scale, table, step, kc, kc, s, nullptr,
                             host ? &hk : nullptr);
        k::kv_append_q4_step(pl.p.v_q4, pl.p.v_q4, table, step, vc, vc, s, nullptr, host ? &hv : nullptr);
    } else if (fmt == k::kKvQ4) k::kv_append_q4_step(pl.p.k_q4, pl.p.v_q4, table, step, kc, vc, s, nullptr, host);
    else if (fmt == k::kKvInt8) k::kv_append_q8_step(pl.p.k_q, pl.p.v_q, pl.p.k_scale, pl.p.v_scale, table, step, kc, vc, s, nullptr, host);
    else k::kv_append_step(pl.p.k_pool, pl.p.v_pool, table, step, kc, vc, s, nullptr, host);
}

// A selection like qsa_block_topk's: ascending cells, whole blocks, the tail block's cells, and one partial block.
std::vector<int32_t> selection(int64_t n_kv, int64_t width, std::mt19937& rng) {
    std::vector<int32_t> ids;
    if (n_kv <= width) {
        for (int64_t c = 0; c < n_kv; ++c) ids.push_back((int32_t) c);
        return ids;
    }
    const int64_t n_bid = n_kv / 4, tail = n_kv - n_bid * 4;
    std::set<int64_t> blocks;
    const int64_t want_full = (width - tail) / 4, part = (width - tail) % 4;
    std::uniform_int_distribution<int64_t> any(0, n_bid - 1), recent(std::max<int64_t>(0, n_bid - 2048), n_bid - 1);
    while ((int64_t) blocks.size() < want_full + (part ? 1 : 0)) blocks.insert(rng() % 2 ? recent(rng) : any(rng));
    int64_t partial_block = part ? *std::next(blocks.begin(), (long) (rng() % blocks.size())) : -1;
    for (int64_t b : blocks) {
        const int64_t take = b == partial_block ? part : 4;
        for (int64_t i = 0; i < take; ++i) ids.push_back((int32_t) (b * 4 + i));
    }
    for (int64_t c = n_bid * 4; c < n_kv; ++c) ids.push_back((int32_t) c);
    std::sort(ids.begin(), ids.end());
    return ids;
}

bool run(int fmt) {
    const char* name = fmt == k::kKvHybrid ? "k8v4" : fmt == k::kKvQ4 ? "q4_0" : fmt == k::kKvInt8 ? "int8" : "fp16";
    k::QsaShapes s = k::qsa_real_shapes();
    const int64_t N = 40000, n_blocks = (N + 3) / 4, n_slots = 8 * 516 + 700;   // must evict: slots < blocks
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
#if defined(STRATA_USE_HIP)
    constexpr int64_t NQ = 32;
#else
    constexpr int64_t NQ = 8;
#endif
    const int64_t H = s.n_head_kv, D = s.head_dim, NH = s.n_head;
    std::mt19937 rng(7 + 4 * fmt);
    std::normal_distribution<float> nd(0.f, 1.f);

    Pools ref, slots, host;
    ref.alloc(n_blocks, s, fmt, false);
    slots.alloc(n_slots, s, fmt, false);
    host.alloc(n_blocks, s, fmt, true);
    int32_t* ident = dalloc<int32_t>(n_blocks);
    {
        std::vector<int32_t> t(n_blocks);
        for (int64_t i = 0; i < n_blocks; ++i) t[i] = (int32_t) i;
        ck(cudaMemcpy(ident, t.data(), n_blocks * 4, cudaMemcpyHostToDevice), "ident");
    }
    k::KvStreamMap m;
    m.page_table = dalloc<int32_t>(n_blocks);
    m.slot_block = dalloc<int32_t>(n_slots); m.slot_stamp = dalloc<int32_t>(n_slots); m.slot_ref = dalloc<int32_t>(n_slots);
    m.miss_block = dalloc<int32_t>(n_slots); m.miss_slot = dalloc<int32_t>(n_slots);
    m.ctl = dalloc<int32_t>(k::kKvCtlInts);
    m.n_blocks = n_blocks; m.n_slots = n_slots;
    k::kv_stream_reset(m, nullptr);

    float* kc = dalloc<float>(H * D);
    float* vc = dalloc<float>(H * D);
    int32_t* step = dalloc<int32_t>(k::kStepCount);
    int32_t* steps = dalloc<int32_t>(NQ * k::kStepCount);
    int32_t* ids = dalloc<int32_t>(NQ * cap);
    float* q = dalloc<float>(NQ * NH * D);
    const uint64_t scr = k::qsa_decode_attn_scratch_floats(cap, s);
    float* scratch = dalloc<float>(NQ * scr);
    float* out_ref = dalloc<float>(NQ * NH * D);
    float* out_str = dalloc<float>(NQ * NH * D);
    std::vector<float> hk(H * D), hv(H * D), hq(NQ * NH * D), a((size_t) NQ * NH * D), b2((size_t) NQ * NH * D);
    std::vector<int32_t> pt(n_blocks), sb(n_slots);

    int batches = 0, bad = 0, range_calls = 0, range_checked = 0, range_ahead = 0;
    for (int64_t pos = 0; pos < N; ++pos) {
        for (auto& x : hk) x = nd(rng) * ((pos % 17 == 0) ? 30.f : 1.f);
        for (auto& x : hv) x = nd(rng);
        const int32_t st[4] = {(int32_t) pos, (int32_t) (pos + 1), (int32_t) ((pos + 1) / 4), 0};
        ck(cudaMemcpy(kc, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice), "k");
        ck(cudaMemcpy(vc, hv.data(), hv.size() * 4, cudaMemcpyHostToDevice), "v");
        ck(cudaMemcpy(step, st, sizeof(st), cudaMemcpyHostToDevice), "step");
        append(ref, ident, step, kc, vc, s, fmt, nullptr);
        append(slots, m.page_table, step, kc, vc, s, fmt, &host.p);
        if (pos % 131 != 130 && pos != N - 1) continue;
        // a batch: n_q queries at the last n_q positions (as a verify window), each with its own selection
        const int n_q = 1 + (int) (rng() % 8);
        std::vector<int32_t> hids((size_t) NQ * cap, 0), hst(NQ * 4, 0);
        for (int t = 0; t < n_q; ++t) {
            const int64_t p = std::max<int64_t>(0, pos - (n_q - 1 - t)), n_kv = p + 1;
            const int64_t width = std::min<int64_t>(n_kv, cap);
            const std::vector<int32_t> sel = selection(n_kv, width, rng);
            if ((int64_t) sel.size() != width) { std::fprintf(stderr, "selection size %zu != %lld\n", sel.size(), (long long) width); return false; }
            std::copy(sel.begin(), sel.end(), hids.begin() + (size_t) t * cap);
            hst[t * 4 + 0] = (int32_t) p; hst[t * 4 + 1] = (int32_t) n_kv; hst[t * 4 + 2] = (int32_t) (n_kv / 4);
            hst[t * 4 + 3] = (int32_t) width;
        }
        for (auto& x : hq) x = nd(rng);
        ck(cudaMemcpy(ids, hids.data(), hids.size() * 4, cudaMemcpyHostToDevice), "ids");
        ck(cudaMemcpy(steps, hst.data(), hst.size() * 4, cudaMemcpyHostToDevice), "steps");
        ck(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q");
        // 5. every third batch first makes a run of blocks resident (kv_stream_resolve_range, the prompt path's
        //    STRATA_KV_PF_RESIDENT): the whole run is in the table afterwards, and one query whose selection is `cap`
        //    consecutive cells of the run reads the resident pool's values from it WITHOUT a resolve of its own
        if (batches % 3 == 2) {
            // every other one runs past the written cells (up to 64 blocks that hold no cell yet, as a prompt chunk's
            // new blocks): those get slots without a read, and the appends that follow must fill slot and host alike
            const int64_t nbk = (pos + 4) / 4;
            const int64_t len = std::min<int64_t>(nbk, 1 + (int64_t) (rng() % (uint64_t) (n_slots - 64)));
            const int64_t ahead = (range_calls & 1) ? std::min<int64_t>(64, n_blocks - nbk) : 0;
            const int64_t rb0 = ahead > 0 ? nbk - len : (int64_t) (rng() % (uint64_t) (nbk - len + 1));
            const int64_t rb1 = rb0 + len + ahead;
            range_ahead += rb1 > nbk;
            k::kv_stream_resolve_range(m, slots.attn(m.page_table), host.p, fmt, rb0, rb1, nbk, s, nullptr);
            ++range_calls;
            ck(cudaDeviceSynchronize(), "range");
            ck(cudaMemcpy(pt.data(), m.page_table, n_blocks * 4, cudaMemcpyDeviceToHost), "pt");
            for (int64_t bk = rb0; bk < rb1; ++bk)
                if (pt[bk] < 0) { if (bad++ < 5) std::fprintf(stderr, "  %s: block %lld of the range [%lld, %lld) is not resident\n", name, (long long) bk, (long long) rb0, (long long) rb1); break; }
            const int64_t c0 = rb0 * 4, c1 = std::min<int64_t>(rb1 * 4, pos + 1), rw = std::min<int64_t>(c1 - c0, cap);
            if (rw > 0) {
                std::vector<int32_t> rids((size_t) cap, 0);
                for (int64_t i = 0; i < rw; ++i) rids[(size_t) i] = (int32_t) (c1 - rw + i);
                const int32_t rst[4] = {(int32_t) pos, (int32_t) (pos + 1), (int32_t) ((pos + 1) / 4), (int32_t) rw};
                ck(cudaMemcpy(ids, rids.data(), rids.size() * 4, cudaMemcpyHostToDevice), "rids");
                ck(cudaMemcpy(steps, rst, sizeof(rst), cudaMemcpyHostToDevice), "rsteps");
                ck(cudaMemcpy(q, hq.data(), (size_t) NH * D * 4, cudaMemcpyHostToDevice), "rq");
                k::qsa_decode_attn_batch(q, ref.attn(ident), ids, steps, cap, s, scratch, out_ref, 1, nullptr);
                k::qsa_decode_attn_batch(q, slots.attn(m.page_table), ids, steps, cap, s, scratch, out_str, 1, nullptr);
                ck(cudaDeviceSynchronize(), "range attention");
                ck(cudaMemcpy(a.data(), out_ref, (size_t) NH * D * 4, cudaMemcpyDeviceToHost), "ra");
                ck(cudaMemcpy(b2.data(), out_str, (size_t) NH * D * 4, cudaMemcpyDeviceToHost), "rb");
                if (std::memcmp(a.data(), b2.data(), (size_t) NH * D * 4) != 0) {
                    if (bad++ < 5) std::fprintf(stderr, "  %s pos %lld: the attention over the range [%lld, %lld) differs\n", name, (long long) pos, (long long) rb0, (long long) rb1);
                }
                ++range_checked;
            }
            ck(cudaMemcpy(ids, hids.data(), hids.size() * 4, cudaMemcpyHostToDevice), "ids");
            ck(cudaMemcpy(steps, hst.data(), hst.size() * 4, cudaMemcpyHostToDevice), "steps");
            ck(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q");
        }
        k::qsa_decode_attn_batch(q, ref.attn(ident), ids, steps, cap, s, scratch, out_ref, n_q, nullptr);
        k::kv_stream_resolve(m, slots.attn(m.page_table), host.p, fmt, ids, steps, n_q, cap, s, nullptr);
        k::qsa_decode_attn_batch(q, slots.attn(m.page_table), ids, steps, cap, s, scratch, out_str, n_q, nullptr);
        ck(cudaDeviceSynchronize(), "batch");
        ck(cudaMemcpy(a.data(), out_ref, (size_t) n_q * NH * D * 4, cudaMemcpyDeviceToHost), "a");
        ck(cudaMemcpy(b2.data(), out_str, (size_t) n_q * NH * D * 4, cudaMemcpyDeviceToHost), "b");
        if (std::memcmp(a.data(), b2.data(), (size_t) n_q * NH * D * 4) != 0) {
            if (bad++ < 5) std::fprintf(stderr, "  %s pos %lld n_q %d: streamed attention differs\n", name, (long long) pos, n_q);
        }
        // the map inverts itself
        ck(cudaMemcpy(pt.data(), m.page_table, n_blocks * 4, cudaMemcpyDeviceToHost), "pt");
        ck(cudaMemcpy(sb.data(), m.slot_block, n_slots * 4, cudaMemcpyDeviceToHost), "sb");
        int64_t resident = 0;
        for (int64_t bk = 0; bk < n_blocks; ++bk) {
            if (pt[bk] < -1 || pt[bk] >= n_slots || (pt[bk] >= 0 && sb[pt[bk]] != bk)) {
                if (bad++ < 5) std::fprintf(stderr, "  map broken at block %lld (table %d)\n", (long long) bk, pt[bk]);
                break;
            }
            resident += pt[bk] >= 0;
        }
        for (int64_t sl = 0; sl < n_slots; ++sl)
            if (sb[sl] >= 0 && pt[sb[sl]] != sl) { if (bad++ < 5) std::fprintf(stderr, "  slot %lld not in the table\n", (long long) sl); break; }
        ++batches;
    }
    const k::KvStreamCounters c = k::kv_stream_counters(m);
    std::printf("  %s: %d batches, %llu block lookups, %llu misses (%.1f%% hit), overflow %d, %d failures\n",
                name, batches, (unsigned long long) c.lookups, (unsigned long long) c.misses,
                c.lookups ? 100.0 * (double) (c.lookups - c.misses) / (double) c.lookups : 0.0, (int) c.overflow, bad);
    std::printf("  %s: %d range resolves, %d of them read back through the attention, %d reached past the written cells\n",
                name, range_calls, range_checked, range_ahead);
    if (c.overflow || c.calls != (uint64_t) (batches + range_calls) || c.misses == 0 || c.misses >= c.lookups) ++bad;
    if (range_checked == 0 || range_ahead == 0) ++bad;

#if defined(STRATA_USE_HIP)
    // 6. A prompt larger than the slot pool: full-batch identity attention versus bounded
    // groups resolved through slots, including a short final group and repeated evictions.
    {
        k::kv_stream_reset(m, nullptr);
        std::vector<int32_t> hids((size_t) NQ * cap), hst(NQ * k::kStepCount);
        for (int64_t t = 0; t < NQ; ++t) {
            const auto sel = selection(N - NQ + t + 1, cap, rng);
            std::copy(sel.begin(), sel.end(), hids.begin() + t * cap);
            hst[t * 4] = (int32_t) (N - NQ + t);
            hst[t * 4 + 1] = hst[t * 4] + 1;
            hst[t * 4 + 2] = hst[t * 4 + 1] / 4;
            hst[t * 4 + 3] = (int32_t) cap;
        }
        ck(cudaMemcpy(ids, hids.data(), hids.size() * 4, cudaMemcpyHostToDevice), "prompt ids");
        ck(cudaMemcpy(steps, hst.data(), hst.size() * 4, cudaMemcpyHostToDevice), "prompt steps");
        if (!k::qsa_prompt_attn_batch(q, ref.attn(ident), ids, steps, cap, s, out_ref, NQ, nullptr))
            k::qsa_decode_attn_batch(q, ref.attn(ident), ids, steps, cap, s, scratch, out_ref, NQ, nullptr);
        const int64_t batch = n_slots / (cap / s.page_size + 2);
        for (int64_t t = 0; t < NQ; t += batch) {
            const int64_t nb = std::min(batch, NQ - t);
            k::kv_stream_resolve(m, slots.attn(m.page_table), host.p, fmt, ids + t * cap,
                                 steps + t * k::kStepCount, nb, cap, s, nullptr);
            if (!k::qsa_prompt_attn_batch(q + t * NH * D, slots.attn(m.page_table), ids + t * cap,
                                         steps + t * k::kStepCount, cap, s, out_str + t * NH * D, nb, nullptr))
                k::qsa_decode_attn_batch(q + t * NH * D, slots.attn(m.page_table), ids + t * cap,
                                         steps + t * k::kStepCount, cap, s, scratch, out_str + t * NH * D, nb, nullptr);
        }
        ck(cudaDeviceSynchronize(), "prompt groups");
        ck(cudaMemcpy(a.data(), out_ref, (size_t) NQ * NH * D * 4, cudaMemcpyDeviceToHost), "prompt ref");
        ck(cudaMemcpy(b2.data(), out_str, (size_t) NQ * NH * D * 4, cudaMemcpyDeviceToHost), "prompt slots");
        const bool same = std::memcmp(a.data(), b2.data(), (size_t) NQ * NH * D * 4) == 0;
        const auto pc = k::kv_stream_counters(m);
        const bool ok = same && !pc.overflow && pc.misses > (uint64_t) n_slots;
        std::printf("  %s prompt groups: %lld queries, batch %lld, %llu misses, overflow %d: %s\n", name,
                    (long long) NQ, (long long) batch, (unsigned long long) pc.misses, (int) pc.overflow,
                    ok ? "identical with eviction" : "FAIL");
        if (!ok) ++bad;
    }

#endif

    // 4. a ring over the same host copy: blocks [b1 - R, b1) restored, read through `b % R`
    {
        const int64_t R = 1500, b1 = n_blocks, b0 = b1 - R;
        Pools ring;
        ring.alloc(R, s, fmt, false);
        int32_t* rt = dalloc<int32_t>(n_blocks);
        k::kv_ring_table(rt, n_blocks, R, nullptr);
        k::kv_ring_restore(ring.attn(rt), host.p, fmt, b0, b1, R, s, nullptr);
        const int64_t width = cap;
        std::vector<int32_t> hids((size_t) cap), hst = {(int32_t) (N - 1), (int32_t) N, (int32_t) (N / 4), (int32_t) width};
        for (int64_t i = 0; i < width; ++i) hids[i] = (int32_t) (N - width + i);   // the window's last cells
        ck(cudaMemcpy(ids, hids.data(), hids.size() * 4, cudaMemcpyHostToDevice), "ids");
        ck(cudaMemcpy(steps, hst.data(), 16, cudaMemcpyHostToDevice), "steps");
        k::qsa_decode_attn_batch(q, ref.attn(ident), ids, steps, cap, s, scratch, out_ref, 1, nullptr);
        k::qsa_decode_attn_batch(q, ring.attn(rt), ids, steps, cap, s, scratch, out_str, 1, nullptr);
        ck(cudaDeviceSynchronize(), "ring");
        ck(cudaMemcpy(a.data(), out_ref, (size_t) NH * D * 4, cudaMemcpyDeviceToHost), "a");
        ck(cudaMemcpy(b2.data(), out_str, (size_t) NH * D * 4, cudaMemcpyDeviceToHost), "b");
        const bool ok = std::memcmp(a.data(), b2.data(), (size_t) NH * D * 4) == 0;
        std::printf("  %s ring restore: %s\n", name, ok ? "identical" : "DIFFERS");
        if (!ok) ++bad;
    }
    return bad == 0;
}
}  // namespace

int main() {
    std::printf("kv_stream_parity: streamed vs resident KV, bitwise\n");
    const bool a = run(k::kKvInt8), b = run(k::kKvF16), c = run(k::kKvQ4), d = run(k::kKvHybrid);
    if (!a || !b || !c || !d) ++g_fail;
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
