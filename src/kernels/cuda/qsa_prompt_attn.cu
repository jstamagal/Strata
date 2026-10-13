// src/kernels/cuda/qsa_prompt_attn.cu - see include/strata/kernels/qsa_prompt_attn.hpp.
#include "strata/core/emulate.hpp"
#include <cstdlib>
#include <cstring>
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace strata::kernels {
namespace {

constexpr int HD = 256;           // head_dim
constexpr int G = 12;             // query heads per KV head
#ifndef D1_CH
#define D1_CH 32
#endif
constexpr int CH = D1_CH;         // cells per chunk
constexpr int THREADS = 128;      // 4 warps: scores by cell (8 each), p.v by dimension (64 each = one int8 scale group)
constexpr int QS = HD + 8;        // q row stride in halves (bank-conflict-free fragment loads)

// The MMA below needs sm_75 or newer (Turing runs it as two k=8 steps); cp.async needs sm_80. Builds for pre-sm_75
// cards compile the MMA to a trap; qsa_prompt_attn_batch refuses such a device at run time, so the old kernel runs
// there.  Turing compiles cp_async16 to a trap as well and takes the v1 kernel instead of launch_i8.
#if defined(__HIPCC__)          // AMD: no mma.sync / cp.async; the host keeps the old kernel (below)
#define STRATA_PA_SM80 0
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define STRATA_PA_SM80 1
#else
#define STRATA_PA_SM80 0
#endif

// m16n8k16 with f16 inputs needs sm_80.  Turing (sm_75) has m16n8k8 with the SAME A/B/C register mapping, so the
// k=16 step is two k=8 steps on the fragments as they are already laid out: a[0]/a[1] are rows gid/gid+8 at k columns
// 2*tig..2*tig+1 (b[0]'s k rows), a[2]/a[3] the same rows at k columns 2*tig+8..2*tig+9 (b[1]'s k rows).  The
// products then add into the same FP32 C registers in the order hi-part-0, hi-part-1, which is the order the k16
// instruction accumulates in as well - but the sum now rounds twice, so the two paths do not agree bit for bit.
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
#if !STRATA_PA_SM80 && (defined(__HIPCC__) || !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 750)
    __trap();   // AMD and pre-Turing builds: no mma.sync (the host keeps the old kernel there)
#elif !STRATA_PA_SM80
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[2]), "r"(a[3]), "r"(b[1]));
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

// Two int8 codes (low byte first) as an exact half2: 1024 + (c + 128) built in the mantissa, minus 1152.
__device__ __forceinline__ uint32_t i8x2_to_h2(uint32_t x) {
    uint32_t y = ((x & 0xffu) | ((x & 0xff00u) << 8)) ^ 0x00800080u;
    y |= 0x64006400u;
    __half2 h = *reinterpret_cast<__half2*>(&y);
    h = __hsub2(h, __halves2half2(__float2half(1152.f), __float2half(1152.f)));
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t pack_h2(float lo_k, float hi_k) {   // element k in the low half
    __half2 h = __floats2half2_rn(lo_k, hi_k);
    return *reinterpret_cast<uint32_t*>(&h);
}

// KV_MODE 1: int8 codes + fp16 scale per 64 values. KV_MODE 0: fp16 values (scales 1).
// KV_MODE 3 (hybrid K8V4): K as mode 1, V as mode 0 - the row's q4_0 blocks are dequantized to fp16 at
// gather, so everything downstream of the load is the mode-0 V path; the caller un-rotates the output.
// KV_MODE 4 (Q4_0 K and V, `--kv q4_0`): mode 1 with a scale per 32 values - each q4_0 block's codes enter as exact
// int8 (code - 8) and its fp16 scale is applied in FP32, as mode 1's are; the caller rotates q and un-rotates the
// output (kv_q4.hpp).
template <int KV_MODE>
struct Smem {
    using KElem = typename std::conditional<KV_MODE == 0, __half, int8_t>::type;
    using VElem = typename std::conditional<KV_MODE == 1 || KV_MODE == 4, int8_t, __half>::type;
    static constexpr int KROW = KV_MODE == 0 ? HD + 8 : HD + 16;   // elements; 16-byte aligned rows, banks spread
    static constexpr int VROW = (KV_MODE == 1 || KV_MODE == 4) ? HD + 16 : HD + 8;
    static constexpr int NG = KV_MODE == 4 ? HD / QK4_0 : 4;      // scale groups per row (q4_0: 8 of 32, else 4 of 64)
    __half qh[16][QS];
    __half ql[16][QS];
    KElem k[CH][KROW];
    VElem v[CH][VROW];
    float ks[CH][NG];
    float vs[CH][NG];
    float s[16][CH + 1];
    float qmax[THREADS / 32];
    float alpha[16];
    float lsum[16];
    float mrow[16];
    long long row[CH];
};

template <int KV_MODE>
__global__ void __launch_bounds__(THREADS) prompt_attn_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                              const int32_t* __restrict__ ids,
                                                              const int32_t* __restrict__ steps, int n_kv_heads,
                                                              int page_size, float scale_log2, float* __restrict__ attn,
                                                              int cap) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem<KV_MODE>& S = *reinterpret_cast<Smem<KV_MODE>*>(smem_raw);
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int gid = lane >> 2, tig = lane & 3;

    // q: 12 heads + 4 zero rows, scaled by a power of two that puts its largest value near 2^14 (exact, and the
    // lo halves stay out of FP16's subnormal range), then split into hi + lo halves
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);                 // qm < 2^qe
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    for (int i = t; i < 16 * HD; i += THREADS) {
        const int h = i / HD, d = i % HD;
        const float x = h < G ? q[(size_t) h * HD + d] * qup : 0.0f;
        const __half hi = __float2half_rn(x);
        S.qh[h][d] = hi;
        S.ql[h][d] = __float2half_rn(x - __half2float(hi));
    }
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

    for (int c0 = 0; c0 < n; c0 += CH) {
        const int nh = min(CH, n - c0);
        if (t < CH) {
            long long r = -1;
            if (t < nh) {
                const int cell = ids[c0 + t];
                const long long page = (long long) p.page_table[cell / page_size];
                r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
            }
            S.row[t] = r;
        }
        __syncthreads();   // rows ready; the previous chunk's p.v is done with k, v, s
        // gather the chunk's K and V rows (16-byte pieces; K8V4's V as q4_0 blocks dequantized to fp16)
        // and their scales
        if constexpr (KV_MODE == 4) {
            // q4_0 K and V: one (K or V, cell, block) per item - the block's 16 code bytes (2-byte aligned: eight
            // 16-bit loads) become 32 exact int8 codes (element j in the low nibble of byte j, j + 16 in the high
            // one), its fp16 scale goes to ks/vs
            constexpr int BLKS = HD / QK4_0;
            constexpr int BYTES = BLKS * (int) sizeof(block_q4_0);
            for (int i = t; i < 2 * CH * BLKS; i += THREADS) {
                const int kv = i / (CH * BLKS), rem = i % (CH * BLKS), c = rem / BLKS, b = rem % BLKS;
                const long long r = S.row[c];
                uint4 lo = make_uint4(0, 0, 0, 0), hi = make_uint4(0, 0, 0, 0);
                float d = 0.0f;
                if (r >= 0) {
                    const block_q4_0* blk = reinterpret_cast<const block_q4_0*>((kv == 0 ? p.k_q4 : p.v_q4) + r * BYTES) + b;
                    d = __half2float(__ushort_as_half(__ldg(&blk->d)));
                    const uint16_t* q16 = reinterpret_cast<const uint16_t*>(blk->qs);
                    uint32_t w[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        w[j] = (uint32_t) __ldg(q16 + 2 * j) | ((uint32_t) __ldg(q16 + 2 * j + 1) << 16);
                    // per byte: the low nibble minus 8 is element j, the high one minus 8 element j + 16
                    lo = make_uint4(__vsub4(w[0] & 0x0F0F0F0Fu, 0x08080808u), __vsub4(w[1] & 0x0F0F0F0Fu, 0x08080808u),
                                    __vsub4(w[2] & 0x0F0F0F0Fu, 0x08080808u), __vsub4(w[3] & 0x0F0F0F0Fu, 0x08080808u));
                    hi = make_uint4(__vsub4((w[0] >> 4) & 0x0F0F0F0Fu, 0x08080808u),
                                    __vsub4((w[1] >> 4) & 0x0F0F0F0Fu, 0x08080808u),
                                    __vsub4((w[2] >> 4) & 0x0F0F0F0Fu, 0x08080808u),
                                    __vsub4((w[3] >> 4) & 0x0F0F0F0Fu, 0x08080808u));
                }
                int8_t* dst = kv == 0 ? &S.k[c][b * QK4_0] : &S.v[c][b * QK4_0];
                reinterpret_cast<uint4*>(dst)[0] = lo;
                reinterpret_cast<uint4*>(dst)[1] = hi;
                (kv == 0 ? S.ks : S.vs)[c][b] = d;
            }
        } else {
            constexpr int KPIECES = HD * (int) sizeof(typename Smem<KV_MODE>::KElem) / 16;   // per K row
            for (int i = t; i < CH * KPIECES; i += THREADS) {
                const int c = i / KPIECES, pc = i % KPIECES;
                const long long r = S.row[c];
                uint4 kx = make_uint4(0, 0, 0, 0);
                if (r >= 0) {
                    if constexpr (KV_MODE == 0)
                        kx = __ldg(reinterpret_cast<const uint4*>(p.k_pool + r * HD) + pc);
                    else   // modes 1 and 3: the K side is INT8
                        kx = __ldg(reinterpret_cast<const uint4*>(p.k_q + r * HD) + pc);
                }
                *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(&S.k[c][0]) + pc * 16) = kx;
            }
            if constexpr (KV_MODE == 3) {   // V: dequantize the row's q4_0 blocks straight into the fp16 V row
                constexpr int BLKS = HD / QK4_0;
                constexpr int BYTES = BLKS * (int) sizeof(block_q4_0);
                for (int i = t; i < CH * BLKS; i += THREADS) {
                    const int c = i / BLKS, b = i % BLKS;
                    const long long r = S.row[c];
#pragma unroll
                    for (int j = 0; j < QK4_0; ++j) S.v[c][b * QK4_0 + j] = __half(0);
                    if (r >= 0) {
                        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + r * BYTES) + b;
                        const float d = __half2float(__ushort_as_half(blk->d));
#pragma unroll
                        for (int j = 0; j < QK4_0 / 2; ++j) {
                            S.v[c][b * QK4_0 + j] = __float2half_rn((float) ((int)(blk->qs[j] & 0x0F) - 8) * d);
                            S.v[c][b * QK4_0 + j + QK4_0 / 2] =
                                __float2half_rn((float) ((int)(blk->qs[j] >> 4) - 8) * d);
                        }
                    }
                }
            } else {
                constexpr int VPIECES = HD * (int) sizeof(typename Smem<KV_MODE>::VElem) / 16;   // per V row
                for (int i = t; i < CH * VPIECES; i += THREADS) {
                    const int c = i / VPIECES, pc = i % VPIECES;
                    const long long r = S.row[c];
                    uint4 vx = make_uint4(0, 0, 0, 0);
                    if (r >= 0) {
                        if constexpr (KV_MODE == 1)
                            vx = __ldg(reinterpret_cast<const uint4*>(p.v_q + r * HD) + pc);
                        else
                            vx = __ldg(reinterpret_cast<const uint4*>(p.v_pool + r * HD) + pc);
                    }
                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(&S.v[c][0]) + pc * 16) = vx;
                }
            }
            for (int i = t; i < CH * 4; i += THREADS) {
                const int c = i / 4, g = i % 4;
                const long long r = S.row[c];
                float a = 0.0f, b = 0.0f;
                if (r >= 0) {
                    if constexpr (KV_MODE == 1) {
                        a = __half2float(__ushort_as_half(p.k_scale[r * (HD / KV_Q8_GROUP) + g]));
                        b = __half2float(__ushort_as_half(p.v_scale[r * (HD / KV_Q8_GROUP) + g]));
                    } else if constexpr (KV_MODE == 3) {   // K as int8, V dequantized to fp16 (scale 1)
                        a = __half2float(__ushort_as_half(p.k_scale[r * (HD / KV_Q8_GROUP) + g]));
                        b = 1.0f;
                    } else {
                        a = b = 1.0f;
                    }
                }
                S.ks[c][g] = a;
                S.vs[c][g] = b;
            }
        }
        __syncthreads();
        // scores: warp w takes cells 8w..8w+7 (one n-tile) over all 256 dims, per scale group (64 dims; q4_0's 32)
        constexpr int NG = Smem<KV_MODE>::NG, KPG = HD / 16 / NG;   // groups per row, 16-dim MMA steps per group
#pragma unroll
        for (int nt = 0; nt < CH / 32; ++nt) {
            const int cb = (warp + 4 * nt) * 8;
            float sc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                float tg[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int kk = 0; kk < KPG; ++kk) {
                    const int k0 = (g * KPG + kk) * 16;
                    uint32_t ah[4], al[4], b[2];
                    ah[0] = *reinterpret_cast<const uint32_t*>(&S.qh[gid][k0 + 2 * tig]);
                    ah[1] = *reinterpret_cast<const uint32_t*>(&S.qh[gid + 8][k0 + 2 * tig]);
                    ah[2] = *reinterpret_cast<const uint32_t*>(&S.qh[gid][k0 + 2 * tig + 8]);
                    ah[3] = *reinterpret_cast<const uint32_t*>(&S.qh[gid + 8][k0 + 2 * tig + 8]);
                    al[0] = *reinterpret_cast<const uint32_t*>(&S.ql[gid][k0 + 2 * tig]);
                    al[1] = *reinterpret_cast<const uint32_t*>(&S.ql[gid + 8][k0 + 2 * tig]);
                    al[2] = *reinterpret_cast<const uint32_t*>(&S.ql[gid][k0 + 2 * tig + 8]);
                    al[3] = *reinterpret_cast<const uint32_t*>(&S.ql[gid + 8][k0 + 2 * tig + 8]);
                    if constexpr (KV_MODE != 0) {   // modes 1 and 3: the K side is INT8 codes
                        b[0] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(&S.k[cb + gid][k0 + 2 * tig]));
                        b[1] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(&S.k[cb + gid][k0 + 2 * tig + 8]));
                    } else {
                        b[0] = *reinterpret_cast<const uint32_t*>(&S.k[cb + gid][k0 + 2 * tig]);
                        b[1] = *reinterpret_cast<const uint32_t*>(&S.k[cb + gid][k0 + 2 * tig + 8]);
                    }
                    mma16816(tg, ah, b);
#ifndef D1_NO_QLO
                    mma16816(tg, al, b);
#endif
                }
                const float s0 = S.ks[cb + 2 * tig][g], s1 = S.ks[cb + 2 * tig + 1][g];
                sc[0] = fmaf(tg[0], s0, sc[0]);
                sc[1] = fmaf(tg[1], s1, sc[1]);
                sc[2] = fmaf(tg[2], s0, sc[2]);
                sc[3] = fmaf(tg[3], s1, sc[3]);
            }
            const int c = cb + 2 * tig;
            S.s[gid][c] = c < nh ? sc[0] * qdown : -INFINITY;
            S.s[gid][c + 1] = c + 1 < nh ? sc[1] * qdown : -INFINITY;
            S.s[gid + 8][c] = c < nh ? sc[2] * qdown : -INFINITY;
            S.s[gid + 8][c + 1] = c + 1 < nh ? sc[3] * qdown : -INFINITY;
        }
        __syncthreads();
        // online softmax: row t/8, 4 cells per thread, 8 threads per row (lanes 8r..8r+7 of a warp)
        {
            constexpr int PER = CH / 8;
            const int r = t >> 3, sub = t & 7;
            float x[PER], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < PER; ++j) { x[j] = S.s[r][sub * PER + j]; mx = fmaxf(mx, x[j]); }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.s[r][sub * PER + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        // p.v: warp w owns dims [64w, 64w+64), which is int8 scale group w (q4_0: groups 2w and 2w+1, four n-tiles
        // each). A group's scale is folded into p relative to the chunk's largest magnitude (q4_0's scales are signed:
        // ggml's d = max / -8), times 2^14 (|p'| <= 2^14: inside FP16's range, its lo half out of the subnormals); the
        // chunk's sum is then added to the running one in FP32 with the factor taken back out
        {
            constexpr int GPW = NG / 4, JPG = 8 / GPW;   // scale groups per warp, n-tiles per group
            float vdown_g[GPW];
            float tmp[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j) tmp[j][0] = tmp[j][1] = tmp[j][2] = tmp[j][3] = 0.0f;
#pragma unroll
            for (int gi = 0; gi < GPW; ++gi) {
            const int vg = warp * GPW + gi;
            float vmax = 0.0f;
#pragma unroll
            for (int c = lane; c < CH; c += 32) vmax = fmaxf(vmax, fabsf(S.vs[c][vg]));
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f;
            vdown_g[gi] = vmax * (1.0f / 16384.0f);
#pragma unroll
            for (int ks = 0; ks < CH / 16; ++ks) {
                const int cA = ks * 16 + 2 * tig, cB = cA + 8;
                const float w0 = S.vs[cA][vg] * vup, w1 = S.vs[cA + 1][vg] * vup, w2 = S.vs[cB][vg] * vup,
                            w3 = S.vs[cB + 1][vg] * vup;
                const float p00 = S.s[gid][cA] * w0, p01 = S.s[gid][cA + 1] * w1;
                const float p10 = S.s[gid + 8][cA] * w0, p11 = S.s[gid + 8][cA + 1] * w1;
                const float p02 = S.s[gid][cB] * w2, p03 = S.s[gid][cB + 1] * w3;
                const float p12 = S.s[gid + 8][cB] * w2, p13 = S.s[gid + 8][cB + 1] * w3;
                uint32_t ah[4], al[4];
                ah[0] = pack_h2(p00, p01);
                ah[1] = pack_h2(p10, p11);
                ah[2] = pack_h2(p02, p03);
                ah[3] = pack_h2(p12, p13);
                {
                    const __half2* h = reinterpret_cast<const __half2*>(ah);
                    float2 f;
                    f = __half22float2(h[0]); al[0] = pack_h2(p00 - f.x, p01 - f.y);
                    f = __half22float2(h[1]); al[1] = pack_h2(p10 - f.x, p11 - f.y);
                    f = __half22float2(h[2]); al[2] = pack_h2(p02 - f.x, p03 - f.y);
                    f = __half22float2(h[3]); al[3] = pack_h2(p12 - f.x, p13 - f.y);
                }
#pragma unroll
                for (int jj = 0; jj < JPG; ++jj) {
                    const int j = gi * JPG + jj;
                    const int d = warp * 64 + j * 8 + gid;
                    uint32_t b[2];
                    if constexpr (KV_MODE == 1 || KV_MODE == 4) {
                        const uint32_t x0 = (uint8_t) S.v[cA][d] | ((uint32_t) (uint8_t) S.v[cA + 1][d] << 8);
                        const uint32_t x1 = (uint8_t) S.v[cB][d] | ((uint32_t) (uint8_t) S.v[cB + 1][d] << 8);
                        b[0] = i8x2_to_h2(x0);
                        b[1] = i8x2_to_h2(x1);
                    } else {
                        const __half2 h0 = __halves2half2(S.v[cA][d], S.v[cA + 1][d]);
                        const __half2 h1 = __halves2half2(S.v[cB][d], S.v[cB + 1][d]);
                        b[0] = *reinterpret_cast<const uint32_t*>(&h0);
                        b[1] = *reinterpret_cast<const uint32_t*>(&h1);
                    }
                    mma16816(tmp[j], ah, b);
#ifndef D1_NO_PLO
                    mma16816(tmp[j], al, b);
#endif
                }
            }
            }
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float vdown = vdown_g[j / JPG];
                acc[j][0] = fmaf(acc[j][0], a0, tmp[j][0] * vdown);
                acc[j][1] = fmaf(acc[j][1], a0, tmp[j][1] * vdown);
                acc[j][2] = fmaf(acc[j][2], a1, tmp[j][2] * vdown);
                acc[j][3] = fmaf(acc[j][3], a1, tmp[j][3] * vdown);
            }
        }
    }
    __syncthreads();
    const float l0 = S.lsum[gid], l1 = S.lsum[gid + 8];
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = warp * 64 + j * 8 + 2 * tig;
        *reinterpret_cast<float2*>(attn + (size_t) gid * HD + d) = make_float2(acc[j][0] * i0, acc[j][1] * i0);
        if (gid + 8 < G)
            *reinterpret_cast<float2*>(attn + (size_t) (gid + 8) * HD + d) = make_float2(acc[j][2] * i1, acc[j][3] * i1);
    }
}

// ---- v2 (int8 KV): warp w owns dims [64w, 64w+64) for both q.k and p.v, which is also int8 scale group w. So a
// warp needs only its own 64-byte slice of each K and V row: it gathers it itself with cp.async into its own
// double-buffered stage while it computes the previous chunk, and q stays in registers. Only the q.k partial sums
// (one per dim group) cross warps, and they are added in a fixed order: the result is deterministic.
constexpr int CH2 = 32;

struct Smem2 {
    int8_t kv[2][4][2][CH2][64];   // stage, warp, K/V, cell, 64 dims in 16-byte pieces XOR-swizzled by the cell
    float sc[2][4][2][CH2];        // stage, warp, K/V scale of the cell for the warp's group
    float part[4][16][CH2 + 1];    // q.k per dim group
    float p[16][CH2 + 1];
    float qmax[4];
    float alpha[16];
    float lsum[16];
    float mrow[16];
};

__device__ __forceinline__ int swz(int cell, int byte) {   // byte offset of (cell, byte) in a stage slice
    return cell * 64 + ((((byte >> 4) ^ (cell >> 1)) & 3) << 4) + (byte & 15);
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
#if !STRATA_PA_SM80
    __trap();
#else
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem), "r"(valid ? 16 : 0));
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if STRATA_PA_SM80
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
__device__ __forceinline__ void cp_async_wait1() {
#if STRATA_PA_SM80
    asm volatile("cp.async.wait_group 1;\n" ::);
#endif
}

__global__ void __launch_bounds__(THREADS) prompt_attn_i8_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                 const int32_t* __restrict__ ids,
                                                                 const int32_t* __restrict__ steps, int n_kv_heads,
                                                                 int page_size, float scale_log2,
                                                                 float* __restrict__ attn, int cap) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem2& S = *reinterpret_cast<Smem2*>(smem_raw);
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int dim0 = warp * 64;

    // q: the power-of-two prescale over all 12 heads (as v1), then this warp's 64 dims as hi/lo A fragments
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    uint32_t qh[4][4], ql[4][4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = gid + (r & 1) * 8, col = dim0 + kk * 16 + 2 * tig + (r >> 1) * 8;
            float2 x = make_float2(0.f, 0.f);
            if (row < G) x = *reinterpret_cast<const float2*>(q + (size_t) row * HD + col);
            x.x *= qup;
            x.y *= qup;
            const __half2 hi = __floats2half2_rn(x.x, x.y);
            const float2 hf = __half22float2(hi);
            const __half2 lo = __floats2half2_rn(x.x - hf.x, x.y - hf.y);
            qh[kk][r] = *reinterpret_cast<const uint32_t*>(&hi);
            ql[kk][r] = *reinterpret_cast<const uint32_t*>(&lo);
        }
    }

    // the chunk pipeline: cells two chunks ahead, their pool rows one chunk ahead, the data (cp.async) one ahead
    const int n_chunks = (n + CH2 - 1) / CH2;
    auto cell_of = [&](int c) -> int { return c < n ? __ldg(ids + c) : -1; };
    auto row_of = [&](int cell) -> long long {
        if (cell < 0) return -1;
        const long long page = (long long) __ldg(p.page_table + cell / page_size);
        return (page * n_kv_heads + kvh) * page_size + (cell % page_size);
    };
    auto issue = [&](long long r, int st, float& ksr, float& vsr) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = lane + 32 * j, cell = idx >> 2, pc = idx & 3;
            const long long rr = __shfl_sync(0xffffffffu, r, cell);
            const bool ok = rr >= 0;
            const size_t off = ok ? (size_t) rr * HD + dim0 + pc * 16 : 0;
            cp_async16(&S.kv[st][warp][0][0][0] + swz(cell, pc * 16), p.k_q + off, ok);
            cp_async16(&S.kv[st][warp][1][0][0] + swz(cell, pc * 16), p.v_q + off, ok);
        }
        ksr = r >= 0 ? __half2float(__ushort_as_half(__ldg(p.k_scale + r * (HD / KV_Q8_GROUP) + warp))) : 0.0f;
        vsr = r >= 0 ? __half2float(__ushort_as_half(__ldg(p.v_scale + r * (HD / KV_Q8_GROUP) + warp))) : 0.0f;
    };
    float ksn, vsn;
    issue(row_of(cell_of(lane)), 0, ksn, vsn);
    cp_async_commit();
    S.sc[0][warp][0][lane] = ksn;
    S.sc[0][warp][1][lane] = vsn;
    long long r_next = row_of(cell_of(CH2 + lane));
    int cell_next2 = cell_of(2 * CH2 + lane);

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

    for (int ci = 0; ci < n_chunks; ++ci) {
        const int st = ci & 1, c0 = ci * CH2;
        const bool more = ci + 1 < n_chunks;
        if (more) issue(r_next, st ^ 1, ksn, vsn);
        cp_async_commit();
        r_next = row_of(cell_next2);
        cell_next2 = cell_of((ci + 3) * CH2 + lane);
        cp_async_wait1();
        __syncwarp();
        const int8_t* K = &S.kv[st][warp][0][0][0];
        const int8_t* V = &S.kv[st][warp][1][0][0];
        // q.k over this warp's 64 dims, times the cell's K scale for this group
#pragma unroll
        for (int nt = 0; nt < CH2 / 8; ++nt) {
            float tg[4] = {0.f, 0.f, 0.f, 0.f};
            const int cell = nt * 8 + gid;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                uint32_t b[2];
                b[0] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(K + swz(cell, kk * 16 + 2 * tig)));
                b[1] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(K + swz(cell, kk * 16 + 2 * tig + 8)));
                mma16816(tg, qh[kk], b);
                mma16816(tg, ql[kk], b);
            }
            const int c = nt * 8 + 2 * tig;
            const float s0 = S.sc[st][warp][0][c], s1 = S.sc[st][warp][0][c + 1];
            S.part[warp][gid][c] = tg[0] * s0;
            S.part[warp][gid][c + 1] = tg[1] * s1;
            S.part[warp][gid + 8][c] = tg[2] * s0;
            S.part[warp][gid + 8][c + 1] = tg[3] * s1;
        }
        __syncthreads();
        // online softmax over the four groups' sum (fixed order): row t/8, 4 cells per thread
        {
            const int r = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                x[j] = c0 + c < n ? (((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c]) * qdown
                                  : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.p[r][sub * 4 + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        // p.v over this warp's 64 dims (as v1)
        {
            float vmax = S.sc[st][warp][1][lane];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            float tmp[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j) tmp[j][0] = tmp[j][1] = tmp[j][2] = tmp[j][3] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < CH2 / 16; ++ks) {
                const int cA = ks * 16 + 2 * tig, cB = cA + 8;
                const float w0 = S.sc[st][warp][1][cA] * vup, w1 = S.sc[st][warp][1][cA + 1] * vup,
                            w2 = S.sc[st][warp][1][cB] * vup, w3 = S.sc[st][warp][1][cB + 1] * vup;
                const float p00 = S.p[gid][cA] * w0, p01 = S.p[gid][cA + 1] * w1;
                const float p10 = S.p[gid + 8][cA] * w0, p11 = S.p[gid + 8][cA + 1] * w1;
                const float p02 = S.p[gid][cB] * w2, p03 = S.p[gid][cB + 1] * w3;
                const float p12 = S.p[gid + 8][cB] * w2, p13 = S.p[gid + 8][cB + 1] * w3;
                uint32_t ah[4], al[4];
                ah[0] = pack_h2(p00, p01);
                ah[1] = pack_h2(p10, p11);
                ah[2] = pack_h2(p02, p03);
                ah[3] = pack_h2(p12, p13);
                {
                    const __half2* h = reinterpret_cast<const __half2*>(ah);
                    float2 f;
                    f = __half22float2(h[0]); al[0] = pack_h2(p00 - f.x, p01 - f.y);
                    f = __half22float2(h[1]); al[1] = pack_h2(p10 - f.x, p11 - f.y);
                    f = __half22float2(h[2]); al[2] = pack_h2(p02 - f.x, p03 - f.y);
                    f = __half22float2(h[3]); al[3] = pack_h2(p12 - f.x, p13 - f.y);
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int d = j * 8 + gid;
                    const uint32_t x0 = (uint8_t) V[swz(cA, d)] | ((uint32_t) (uint8_t) V[swz(cA + 1, d)] << 8);
                    const uint32_t x1 = (uint8_t) V[swz(cB, d)] | ((uint32_t) (uint8_t) V[swz(cB + 1, d)] << 8);
                    uint32_t b[2];
                    b[0] = i8x2_to_h2(x0);
                    b[1] = i8x2_to_h2(x1);
                    mma16816(tmp[j], ah, b);
                    mma16816(tmp[j], al, b);
                }
            }
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j][0] = fmaf(acc[j][0], a0, tmp[j][0] * vdown);
                acc[j][1] = fmaf(acc[j][1], a0, tmp[j][1] * vdown);
                acc[j][2] = fmaf(acc[j][2], a1, tmp[j][2] * vdown);
                acc[j][3] = fmaf(acc[j][3], a1, tmp[j][3] * vdown);
            }
        }
        if (more) {
            S.sc[st ^ 1][warp][0][lane] = ksn;
            S.sc[st ^ 1][warp][1][lane] = vsn;
        }
        __syncwarp();
    }
    __syncthreads();
    const float l0 = S.lsum[gid], l1 = S.lsum[gid + 8];
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = dim0 + j * 8 + 2 * tig;
        *reinterpret_cast<float2*>(attn + (size_t) gid * HD + d) = make_float2(acc[j][0] * i0, acc[j][1] * i0);
        if (gid + 8 < G)
            *reinterpret_cast<float2*>(attn + (size_t) (gid + 8) * HD + d) = make_float2(acc[j][2] * i1, acc[j][3] * i1);
    }
}

bool launch_i8(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
               const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t st) {
    static bool attr[64] = {};   // the shared-memory opt-in is per device (a layer split runs this on several)
    int dev = 0;
    cudaGetDevice(&dev);
    const int bytes = (int) sizeof(Smem2);
    if (dev < 0 || dev >= 64) return false;
    if (!attr[dev]) {
        if (cudaFuncSetAttribute(prompt_attn_i8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) !=
            cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        attr[dev] = true;
    }
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        prompt_attn_i8_kernel<<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, bytes, st>>>(
            q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
            (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prompt_attn_batch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

template <int KV_MODE>
bool launch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
            const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t st) {
    static bool attr[64] = {};   // per device, as above
    int dev = 0;
    cudaGetDevice(&dev);
    const int bytes = (int) sizeof(Smem<KV_MODE>);
    if (dev < 0 || dev >= 64) return false;
    if (!attr[dev]) {
        if (cudaFuncSetAttribute(prompt_attn_kernel<KV_MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) !=
            cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        attr[dev] = true;
    }
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        prompt_attn_kernel<KV_MODE><<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, bytes, st>>>(
            q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
            (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prompt_attn_batch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}


// PR #600: compiled only into the experimental build (-DSTRATA_EXPERIMENTAL_SM60=ON), the one build that runs on a
// Volta card; the ready-made engine has none of it (no extra kernels to load, the same code as before).
#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
// ---- sm_70 (Volta): v1's structure on mma.m8n8k4 ------------------------------------------------------------------
// Volta's tensor cores only have m8n8k4 (FP16 in, FP32 accumulate): per warp FOUR independent 8x8x4 products, one per
// quad-pair (QP q = lanes 4q..4q+3 and 4q+16..4q+19; thread j = (lane & 3) + 4 * (lane >> 4) within it). Layout, measured on
// a V100 (tools-free microtest, and equal to turbomind's SM70_MMA_884): the thread holds row j of A (k 0..3 as two
// registers) and column j of B (k 0..3); C register i is row (lane & 1) + 4 * (lane >> 4) + 2 * ((i >> 1) & 1),
// column (lane & 2) + 4 * (i >> 2) + (i & 1) of its QP's 8x8 tile.
//   q.k: warp w owns dims [64w, 64w+64) = int8 scale group w; QP q takes cells 8q..8q+7 of the 32-cell chunk; 16 k-steps
//        of 4 dims, two 8-row blocks (12 heads + 4 zero rows), q as hi + lo halves. The warp's partial (times the
//        cells' K scale) goes to S.part[w]; the softmax adds the four groups in a fixed order (deterministic).
//   p.v: warp w owns dims [64w, 64w+64) again; QP q takes the 16 dims 64w+16q.. (two n-tiles of 8); k = the 32 cells in
//        steps of 4; p carries the V scale (as v1) and is split into hi + lo halves.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
#define STRATA_PA_VOLTA 1
#else
#define STRATA_PA_VOLTA 0
#endif

__device__ __forceinline__ void mma884(float* c, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
#if STRATA_PA_VOLTA
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#else
    __trap();   // m8n8k4 is a Volta instruction; the host only launches this kernel on sm_70
#endif
}

template <int KV_MODE>
struct Smem70 {
    using KElem = typename std::conditional<KV_MODE == 0, __half, int8_t>::type;
    using VElem = typename std::conditional<KV_MODE == 1, int8_t, __half>::type;
    static constexpr int KROW = KV_MODE == 0 ? HD + 8 : HD + 16;
    static constexpr int VROW = KV_MODE == 1 ? HD + 16 : HD + 8;
    __half qh[16][QS];
    __half ql[16][QS];
    KElem k[CH][KROW];
    VElem v[CH][VROW];
    float ks[CH][4];
    float vs[CH][4];
    float part[4][16][CH + 4];   // q.k per dim group (scaled by the cell's K scale)
    float s[16][CH + 4];         // the chunk's probabilities
    float qmax[THREADS / 32];
    float alpha[16];
    float lsum[16];
    float mrow[16];
    long long row[CH];
};

template <int KV_MODE>
__global__ void __launch_bounds__(THREADS) prompt_attn_v70_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                  const int32_t* __restrict__ ids,
                                                                  const int32_t* __restrict__ steps, int n_kv_heads,
                                                                  int page_size, float scale_log2,
                                                                  float* __restrict__ attn, int cap) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem70<KV_MODE>& S = *reinterpret_cast<Smem70<KV_MODE>*>(smem_raw);
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int qp = (lane >> 2) & 3, jj = (lane & 3) + 4 * (lane >> 4);   // quad-pair, thread within it
    const int crow = (lane & 1) + 4 * (lane >> 4);                        // C row of register 0 (+2 per register pair)
    const int ccol = lane & 2;                                           // C column of register 0 (+4 per half, +1 per reg)

    // q: 12 heads + 4 zero rows, scaled by a power of two that puts its largest value near 2^14 (exact, and the
    // lo halves stay out of FP16's subnormal range), then split into hi + lo halves
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);                 // qm < 2^qe
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    for (int i = t; i < 16 * HD; i += THREADS) {
        const int h = i / HD, d = i % HD;
        const float x = h < G ? q[(size_t) h * HD + d] * qup : 0.0f;
        const __half hi = __float2half_rn(x);
        S.qh[h][d] = hi;
        S.ql[h][d] = __float2half_rn(x - __half2float(hi));
    }
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }

    float acc[2][2][8];
#pragma unroll
    for (int rb = 0; rb < 2; ++rb)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[rb][nt][i] = 0.0f;

    for (int c0 = 0; c0 < n; c0 += CH) {
        const int nh = min(CH, n - c0);
        if (t < CH) {
            long long r = -1;
            if (t < nh) {
                const int cell = ids[c0 + t];
                const long long page = (long long) p.page_table[cell / page_size];
                r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
            }
            S.row[t] = r;
        }
        __syncthreads();   // rows ready; the previous chunk's p.v is done with k, v, s
        // gather the chunk's K and V rows (16-byte pieces; K8V4's V as q4_0 blocks dequantized to fp16)
        // and their scales
        {
            constexpr int KPIECES = HD * (int) sizeof(typename Smem<KV_MODE>::KElem) / 16;   // per K row
            for (int i = t; i < CH * KPIECES; i += THREADS) {
                const int c = i / KPIECES, pc = i % KPIECES;
                const long long r = S.row[c];
                uint4 kx = make_uint4(0, 0, 0, 0);
                if (r >= 0) {
                    if constexpr (KV_MODE == 0)
                        kx = __ldg(reinterpret_cast<const uint4*>(p.k_pool + r * HD) + pc);
                    else   // modes 1 and 3: the K side is INT8
                        kx = __ldg(reinterpret_cast<const uint4*>(p.k_q + r * HD) + pc);
                }
                *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(&S.k[c][0]) + pc * 16) = kx;
            }
            if constexpr (KV_MODE == 3) {   // V: dequantize the row's q4_0 blocks straight into the fp16 V row
                constexpr int BLKS = HD / QK4_0;
                constexpr int BYTES = BLKS * (int) sizeof(block_q4_0);
                for (int i = t; i < CH * BLKS; i += THREADS) {
                    const int c = i / BLKS, b = i % BLKS;
                    const long long r = S.row[c];
#pragma unroll
                    for (int j = 0; j < QK4_0; ++j) S.v[c][b * QK4_0 + j] = __half(0);
                    if (r >= 0) {
                        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + r * BYTES) + b;
                        const float d = __half2float(__ushort_as_half(blk->d));
#pragma unroll
                        for (int j = 0; j < QK4_0 / 2; ++j) {
                            S.v[c][b * QK4_0 + j] = __float2half_rn((float) ((int)(blk->qs[j] & 0x0F) - 8) * d);
                            S.v[c][b * QK4_0 + j + QK4_0 / 2] =
                                __float2half_rn((float) ((int)(blk->qs[j] >> 4) - 8) * d);
                        }
                    }
                }
            } else {
                constexpr int VPIECES = HD * (int) sizeof(typename Smem<KV_MODE>::VElem) / 16;   // per V row
                for (int i = t; i < CH * VPIECES; i += THREADS) {
                    const int c = i / VPIECES, pc = i % VPIECES;
                    const long long r = S.row[c];
                    uint4 vx = make_uint4(0, 0, 0, 0);
                    if (r >= 0) {
                        if constexpr (KV_MODE == 1)
                            vx = __ldg(reinterpret_cast<const uint4*>(p.v_q + r * HD) + pc);
                        else
                            vx = __ldg(reinterpret_cast<const uint4*>(p.v_pool + r * HD) + pc);
                    }
                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(&S.v[c][0]) + pc * 16) = vx;
                }
            }
            for (int i = t; i < CH * 4; i += THREADS) {
                const int c = i / 4, g = i % 4;
                const long long r = S.row[c];
                float a = 0.0f, b = 0.0f;
                if (r >= 0) {
                    if constexpr (KV_MODE == 1) {
                        a = __half2float(__ushort_as_half(p.k_scale[r * (HD / KV_Q8_GROUP) + g]));
                        b = __half2float(__ushort_as_half(p.v_scale[r * (HD / KV_Q8_GROUP) + g]));
                    } else if constexpr (KV_MODE == 3) {   // K as int8, V dequantized to fp16 (scale 1)
                        a = __half2float(__ushort_as_half(p.k_scale[r * (HD / KV_Q8_GROUP) + g]));
                        b = 1.0f;
                    } else {
                        a = b = 1.0f;
                    }
                }
                S.ks[c][g] = a;
                S.vs[c][g] = b;
            }
        }
        __syncthreads();
        // scores: warp w = dim group w (64 dims), QP q = cells 8q..8q+7. The hi and lo halves of q accumulate in
        // separate m8n8k4 chains (tg, tgl), added in FP32 at the end: Volta's tensor cores truncate as they
        // accumulate, and in one chain behind the hi products most of the lo half was lost (error vs FP64 at 32K
        // 5.1e-6 -> 2.3e-6, the FP32 kernel's level). The same for p.v below.
        {
            float tg[2][8], tgl[2][8];
#pragma unroll
            for (int rb = 0; rb < 2; ++rb)
#pragma unroll
                for (int i = 0; i < 8; ++i) tg[rb][i] = tgl[rb][i] = 0.0f;
            const int cell_b = qp * 8 + jj;
#pragma unroll
            for (int ks = 0; ks < 16; ++ks) {
                const int d0 = warp * 64 + ks * 4;
                uint32_t b0, b1;
                if constexpr (KV_MODE != 0) {   // modes 1 and 3: the K side is INT8 codes, exact in FP16
                    const uint32_t raw = *reinterpret_cast<const uint32_t*>(&S.k[cell_b][d0]);
                    b0 = i8x2_to_h2(raw & 0xffffu);
                    b1 = i8x2_to_h2(raw >> 16);
                } else {
                    const uint2 raw = *reinterpret_cast<const uint2*>(&S.k[cell_b][d0]);
                    b0 = raw.x;
                    b1 = raw.y;
                }
#pragma unroll
                for (int rb = 0; rb < 2; ++rb) {
                    const uint2 ah = *reinterpret_cast<const uint2*>(&S.qh[rb * 8 + jj][d0]);
                    const uint2 al = *reinterpret_cast<const uint2*>(&S.ql[rb * 8 + jj][d0]);
                    mma884(tg[rb], ah.x, ah.y, b0, b1);
                    mma884(tgl[rb], al.x, al.y, b0, b1);
                }
            }
#pragma unroll
            for (int rb = 0; rb < 2; ++rb)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int row = rb * 8 + crow + 2 * ((i >> 1) & 1);
                    const int cell = qp * 8 + ccol + 4 * (i >> 2) + (i & 1);
                    S.part[warp][row][cell] = (tg[rb][i] + tgl[rb][i]) * S.ks[cell][warp];
                }
        }
        __syncthreads();
        // online softmax: row t/8, 4 cells per thread, 8 threads per row (lanes 8r..8r+7 of a warp)
        {
            constexpr int PER = CH / 8;
            const int r = t >> 3, sub = t & 7;
            float x[PER], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const int c = sub * PER + j;
                const float sum4 = ((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c];
                x[j] = c < nh ? sum4 * qdown : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.s[r][sub * PER + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        __syncthreads();
        // p.v: warp w owns dims [64w, 64w+64) (int8 scale group w), QP q its 16 dims 64w+16q..; the V scale is folded
        // into p relative to the chunk's largest, times 2^14 (as v1)
        {
            float vmax = 0.0f;
#pragma unroll
            for (int c = lane; c < CH; c += 32) vmax = fmaxf(vmax, S.vs[c][warp]);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            float tmp[2][2][8], tmpl[2][2][8];   // the hi and lo halves of p in separate chains (see the scores)
#pragma unroll
            for (int rb = 0; rb < 2; ++rb)
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                    for (int i = 0; i < 8; ++i) tmp[rb][nt][i] = tmpl[rb][nt][i] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < CH / 4; ++ks) {
                const int c0 = ks * 4;
                float w[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) w[e] = S.vs[c0 + e][warp] * vup;
                uint2 ah[2], al[2];
#pragma unroll
                for (int rb = 0; rb < 2; ++rb) {
                    const float4 pv = *reinterpret_cast<const float4*>(&S.s[rb * 8 + jj][c0]);
                    const float p0 = pv.x * w[0], p1 = pv.y * w[1], p2 = pv.z * w[2], p3 = pv.w * w[3];
                    ah[rb].x = pack_h2(p0, p1);
                    ah[rb].y = pack_h2(p2, p3);
                    const float2 f01 = __half22float2(*reinterpret_cast<const __half2*>(&ah[rb].x));
                    const float2 f23 = __half22float2(*reinterpret_cast<const __half2*>(&ah[rb].y));
                    al[rb].x = pack_h2(p0 - f01.x, p1 - f01.y);
                    al[rb].y = pack_h2(p2 - f23.x, p3 - f23.y);
                }
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int d = warp * 64 + qp * 16 + nt * 8 + jj;
                    uint32_t b0, b1;
                    if constexpr (KV_MODE == 1) {
                        const uint32_t x0 = (uint8_t) S.v[c0][d] | ((uint32_t) (uint8_t) S.v[c0 + 1][d] << 8);
                        const uint32_t x1 = (uint8_t) S.v[c0 + 2][d] | ((uint32_t) (uint8_t) S.v[c0 + 3][d] << 8);
                        b0 = i8x2_to_h2(x0);
                        b1 = i8x2_to_h2(x1);
                    } else {
                        const __half2 h0 = __halves2half2(S.v[c0][d], S.v[c0 + 1][d]);
                        const __half2 h1 = __halves2half2(S.v[c0 + 2][d], S.v[c0 + 3][d]);
                        b0 = *reinterpret_cast<const uint32_t*>(&h0);
                        b1 = *reinterpret_cast<const uint32_t*>(&h1);
                    }
#pragma unroll
                    for (int rb = 0; rb < 2; ++rb) {
                        mma884(tmp[rb][nt], ah[rb].x, ah[rb].y, b0, b1);
                        mma884(tmpl[rb][nt], al[rb].x, al[rb].y, b0, b1);
                    }
                }
            }
#pragma unroll
            for (int rb = 0; rb < 2; ++rb) {
                float a_[2];
                a_[0] = S.alpha[rb * 8 + crow];
                a_[1] = S.alpha[rb * 8 + crow + 2];
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                        acc[rb][nt][i] = fmaf(acc[rb][nt][i], a_[(i >> 1) & 1], (tmp[rb][nt][i] + tmpl[rb][nt][i]) * vdown);
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int rb = 0; rb < 2; ++rb)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = rb * 8 + crow + 2 * h;
            if (row >= G) continue;
            const float l = S.lsum[row];
            const float inv = l > 0.0f ? 1.0f / l : 0.0f;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int i = hh * 4 + h * 2;               // registers i, i+1: row h, columns ccol + 4 * hh + {0, 1}
                    const int d = warp * 64 + qp * 16 + nt * 8 + ccol + 4 * hh;
                    *reinterpret_cast<float2*>(attn + (size_t) row * HD + d) =
                        make_float2(acc[rb][nt][i] * inv, acc[rb][nt][i + 1] * inv);
                }
        }
}

template <int KV_MODE>
bool launch70(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
              const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t st) {
    static bool attr[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    const int bytes = (int) sizeof(Smem70<KV_MODE>);
    if (dev < 0 || dev >= 64) return false;
    if (!attr[dev]) {
        if (cudaFuncSetAttribute(prompt_attn_v70_kernel<KV_MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) !=
            cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        attr[dev] = true;
    }
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        prompt_attn_v70_kernel<KV_MODE><<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, bytes, st>>>(
            q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
            (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prompt_attn_batch (sm_70): %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}
#else   // AMD, or a build without STRATA_EXPERIMENTAL_SM60: no m8n8k4 kernel (the caller keeps the old one)
template <int KV_MODE>
bool launch70(const float*, const QsaAttnPools&, const int32_t*, const int32_t*, int64_t, const QsaShapes&, float*,
              int64_t, cudaStream_t) {
    return false;
}
#endif  // !__HIPCC__ && STRATA_EXPERIMENTAL_SM60

#if defined(__HIPCC__)
// ---- S6: the int8-KV prompt attention on RDNA4 matrix cores (opt-in: STRATA_HIP_WMMA=1, gfx12 only). The design of
// the v2 kernel above with gfx12's v_wmma_f32_16x16x16_f16 (wave32) in place of m16n8k16: wave w owns dims
// [64w, 64w+64) (int8 scale group w) for q.k and p.v; q and p are split into FP16 hi + lo parts, the int8 codes enter
// exactly as FP16, the scales are applied in FP32 and the four groups' q.k partials are added in a fixed order.
// FP32-level accuracy, deterministic, but not bitwise equal to qsa_decode_attn_batch (another summation order).
// Fragment layout (16x16x16, wave32, checked on gfx1201): A lane l holds A[l % 16][(l / 16) * 8 + i], B lane l holds
// B[(l / 16) * 8 + i][l % 16], C/D lane l holds D[(l / 16) * 8 + i][l % 16], i = 0..7.
// gfx11 (RDNA3 / RDNA3.5: gfx110x, gfx115x) has the same instruction with another layout: A lane l holds all 16 k of
// row l % 16 (lanes 16..31 repeat lanes 0..15), B lane l all 16 k of column l % 16, and C/D lane l holds
// D[2 * i + l / 16][l % 16]. PA_KL is the k values per lane (8 or 16), PA_KOFF(half) the first one, PA_ROW the C row.
#if defined(__gfx1200__) || defined(__gfx1201__)
#define STRATA_PA_WMMA 1
#define STRATA_PA_GFX11 0
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || \
    defined(__gfx1151__)
#define STRATA_PA_WMMA 1
#define STRATA_PA_GFX11 1
#else
#define STRATA_PA_WMMA 0
#define STRATA_PA_GFX11 0
#endif
#if STRATA_PA_GFX11
constexpr int PA_KL = 16;
#define PA_KOFF(half) 0
#define PA_ROW(i, half) (2 * (i) + (half))
#else
constexpr int PA_KL = 8;
#define PA_KOFF(half) ((half) * 8)
#define PA_ROW(i, half) ((half) * 8 + (i))
#endif
typedef _Float16 wh8 __attribute__((ext_vector_type(PA_KL)));
typedef float wf8 __attribute__((ext_vector_type(8)));
constexpr int WCH = 32;          // cells per chunk (two 16-cell tiles)
constexpr int WVS = 80;          // staged V row stride in bytes (64 codes, 16-byte aligned, banks spread)
struct alignas(16) SmemW {
    float part[4][16][WCH + 1];  // q.k per dim group
    float p[16][WCH + 1];
    float vs[4][WCH];            // V scale per (group, cell)
    int valid[WCH];              // the cell has a resident pool row
    uint8_t v[4][WCH][WVS];      // per wave: the chunk's V codes of its group, one row per cell
    float qmax[4];
    float alpha[16];
    float lsum[16];
    float mrow[16];
};

__device__ __forceinline__ wf8 wmma_f16(wh8 a, wh8 b, wf8 c) {
#if STRATA_PA_WMMA && STRATA_PA_GFX11
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#elif STRATA_PA_WMMA
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
    __builtin_trap();
    return c;
#endif
}

__device__ __forceinline__ wh8 i8_to_h(const uint32_t* x) {   // PA_KL codes, exact: |code| <= 128
    wh8 h;
#pragma unroll
    for (int i = 0; i < PA_KL; ++i) h[i] = (_Float16) (int) (int8_t) (x[i / 4] >> (8 * (i % 4)));
    return h;
}

// SPLIT = true (default): q and p carried as FP16 hi + lo pairs (two WMMAs each: FP32-level). S23 (opt-in
// STRATA_PA_FAST=1): SPLIT = false, one FP16 image each - half the matrix work, rounding-level (FP16 attention as the
// usual flash-attention kernels), quality-gated
template <bool SPLIT>
__global__ void __launch_bounds__(THREADS) prompt_attn_wmma_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                   const int32_t* __restrict__ ids,
                                                                   const int32_t* __restrict__ steps, int n_kv_heads,
                                                                   int page_size, float scale_log2,
                                                                   float* __restrict__ attn, int cap) {
#if STRATA_PA_WMMA
    __shared__ SmemW S;
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int col = lane & 15, half = lane >> 4;   // fragment column / k half
    const int dim0 = warp * 64;

    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    // q A fragments of this wave's 64 dims: row = col (heads 12..15 zero), k = dim0 + kk*16 + PA_KOFF(half) + i
    wh8 qh[4], ql[4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        float x[PA_KL];
        if (col < G) {
            const float4* src = reinterpret_cast<const float4*>(q + (size_t) col * HD + dim0 + kk * 16 + PA_KOFF(half));
#pragma unroll
            for (int j = 0; j < PA_KL / 4; ++j) {
                const float4 a = src[j];
                x[4 * j] = a.x; x[4 * j + 1] = a.y; x[4 * j + 2] = a.z; x[4 * j + 3] = a.w;
            }
        } else {
#pragma unroll
            for (int i = 0; i < PA_KL; ++i) x[i] = 0.0f;
        }
#pragma unroll
        for (int i = 0; i < PA_KL; ++i) {
            const float v = x[i] * qup;
            const _Float16 hi = (_Float16) v;
            qh[kk][i] = hi;
            ql[kk][i] = (_Float16) (v - (float) hi);
        }
    }

    wf8 acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j] = wf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const int n_chunks = (n + WCH - 1) / WCH;
    for (int ci = 0; ci < n_chunks; ++ci) {
        const int c0 = ci * WCH;
        // this lane's cell of the chunk (lane = cell index): its pool row, or -1 (past the selection, or a block the
        // KV streaming left non-resident: masked as in the decode kernel)
        long long myrow = -1;
        if (c0 + lane < n) {
            const int cell = __ldg(ids + c0 + lane);
            const long long page = (long long) __ldg(p.page_table + cell / page_size);
            if (page >= 0) myrow = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        // stage this wave's V slice of the chunk (64 bytes of each row) and the V scale; the K fragments come straight
        // from global memory (8 consecutive codes per lane)
        {
            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0, v2 = v0, v3 = v0;
            float vsc = 0.0f;
            if (myrow >= 0) {
                const uint4* src = reinterpret_cast<const uint4*>(p.v_q + myrow * HD + dim0);
                v0 = __ldg(src); v1 = __ldg(src + 1); v2 = __ldg(src + 2); v3 = __ldg(src + 3);
                vsc = __half2float(__ushort_as_half(__ldg(p.v_scale + myrow * (HD / KV_Q8_GROUP) + warp)));
            }
            uint4* dst = reinterpret_cast<uint4*>(&S.v[warp][lane][0]);
            dst[0] = v0; dst[1] = v1; dst[2] = v2; dst[3] = v3;
            S.vs[warp][lane] = vsc;
            if (warp == 0) S.valid[lane] = myrow >= 0;
        }
        // q.k over this wave's 64 dims for two 16-cell tiles, times the cell's K scale for this group
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const long long rr = __shfl(myrow, nt * 16 + col);
            wf8 s = wf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            float ksc = 0.0f;
            if (rr >= 0) ksc = __half2float(__ushort_as_half(__ldg(p.k_scale + rr * (HD / KV_Q8_GROUP) + warp)));
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                uint32_t kx[PA_KL / 4] = {};
                if (rr >= 0) {
                    const uint32_t* src = reinterpret_cast<const uint32_t*>(p.k_q + rr * HD + dim0 + kk * 16 + PA_KOFF(half));
#if STRATA_PA_GFX11
                    const uint4 v = __ldg(reinterpret_cast<const uint4*>(src));
                    kx[0] = v.x; kx[1] = v.y; kx[2] = v.z; kx[3] = v.w;
#else
                    const uint2 v = __ldg(reinterpret_cast<const uint2*>(src));
                    kx[0] = v.x; kx[1] = v.y;
#endif
                }
                const wh8 b = i8_to_h(kx);
                s = wmma_f16(qh[kk], b, s);
                if constexpr (SPLIT) s = wmma_f16(ql[kk], b, s);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) S.part[warp][PA_ROW(i, half)][nt * 16 + col] = s[i] * ksc;
        }
        __syncthreads();
        // online softmax over the four groups' sum (fixed order): row t/8, 4 cells per thread
        {
            const int r = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                // past the selection, or a masked cell (non-resident page, as in the decode kernel): no weight
                x[j] = S.valid[c] ? (((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c]) * qdown
                                  : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.p[r][sub * 4 + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        // p.v over this wave's 64 dims: 4 tiles of 16 dims, 2 k-steps of 16 cells
        {
            float vmax = S.vs[warp][lane];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            wf8 tmp[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) tmp[j] = wf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const int cb = ks * 16 + PA_KOFF(half);   // this lane's PA_KL cells
                wh8 ah, al;
#pragma unroll
                for (int i = 0; i < PA_KL; ++i) {
                    const float pv = S.p[col][cb + i] * (S.vs[warp][cb + i] * vup);
                    const _Float16 hi = (_Float16) pv;
                    ah[i] = hi;
                    al[i] = (_Float16) (pv - (float) hi);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int d = j * 16 + col;
                    wh8 b;
#pragma unroll
                    for (int i = 0; i < PA_KL; ++i) b[i] = (_Float16) (int) (int8_t) S.v[warp][cb + i][d];
                    tmp[j] = wmma_f16(ah, b, tmp[j]);
                    if constexpr (SPLIT) tmp[j] = wmma_f16(al, b, tmp[j]);
                }
            }
            float a[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) a[i] = S.alpha[PA_ROW(i, half)];
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[j][i] = fmaf(acc[j][i], a[i], tmp[j][i] * vdown);
        }
        // no barrier here: `v`/`vs` are this wave's own, and `p`/`alpha`/`valid` are rewritten only after the next
        // chunk's first barrier (which every wave reaches after its p.v)
    }
    float inv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float l = S.lsum[PA_ROW(i, half)];
        inv[i] = l > 0.0f ? 1.0f / l : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = PA_ROW(i, half);
            if (row < G) attn[(size_t) row * HD + dim0 + j * 16 + col] = acc[j][i] * inv[i];
        }
#else
    __builtin_trap();
#endif
}

// gfx12 (RDNA4) and gfx11 (RDNA3 / RDNA3.5), only on request: the output differs from the default kernel's in its
// last bits
bool hip_wmma_usable() {
    static const bool want = [] {
        const char* e = std::getenv("STRATA_HIP_WMMA");
        return e != nullptr && e[0] == '1';
    }();
    if (!want) return false;
    static int arch[64] = {};   // 0 unknown, 1 gfx11 / gfx12, 2 other
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) { (void) hipGetLastError(); return false; }
    if (arch[dev] == 0) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, dev) != hipSuccess) { (void) hipGetLastError(); return false; }
        arch[dev] = std::strncmp(prop.gcnArchName, "gfx12", 5) == 0 || strata::kernels::gfx_arch_is_gfx11_wmma(prop.gcnArchName)
                        ? 1 : 2;
        static bool told = false;
        if (!told) {
            told = true;
            std::fprintf(stderr, "strata: STRATA_HIP_WMMA: the prompt attention on matrix cores %s (%s)\n",
                         arch[dev] == 1 ? "on" : "unavailable", prop.gcnArchName);
        }
    }
    return arch[dev] == 1;
}

bool launch_wmma(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                 const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t st) {
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        static const bool fast = [] { const char* e = std::getenv("STRATA_PA_FAST"); return e != nullptr && e[0] == '1'; }();
        static const int64_t min_t = [] { const char* e = std::getenv("STRATA_PF_SWITCH_MIN_T"); return e ? (int64_t) std::atoll(e) : (int64_t) 0; }();
        if (fast && n_q >= min_t)
            prompt_attn_wmma_kernel<false><<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, 0, st>>>(
                q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
                (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
        else
            prompt_attn_wmma_kernel<true><<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, 0, st>>>(
                q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
                (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prompt_attn_batch (wmma): %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

// ---- gfx103x (RDNA2, no matrix cores), opt-in (STRATA_PA_DOT4=1): the int8-KV prompt attention on v_dot4_i32_i8.
// The design of the kernels above - one block per (query, KV head), an online softmax over chunks of the selection,
// no split-K scratch and no merge kernel - with the signed byte dot product as the multiplier.  The stored int8
// codes are its second operand as they lie in the pool (no conversion); the first is q, or p times the V scale,
// scaled so the largest of the block (q) or of the head's chunk (p) is 8,355,000, rounded to an integer and cut into
// three balanced base-256 digits (23 bits).  Each digit's dot product is exact in int32 (a 64-dim scale group or a
// 64-cell chunk: |sum| <= 2^20); the three are combined in FP32 and the int8 scales are applied in FP32.
// FP32-level accuracy (qsa_prompt_attn_parity: 2.8e-6 against FP64 where the FP32 kernel has 2.1e-6, output scale
// 3.6), deterministic, not bitwise equal to qsa_decode_attn_batch (another summation order, exp2).
// ND = 2 (STRATA_PA_DOT4=2): two digits, 15 bits of q and p - rounding-level (4.9e-4 on the same test, which it
// fails), as STRATA_PA_FAST is for the matrix-core kernel.
//   q.k : 64 / LW lanes hold one (cell, head triple), each LW K words read straight from the pool and the same
//         words of the triple's q digits in registers: 3 x LW x ND dot4 per lane and cell, then a DPP sum over the
//         lanes.  No shared-memory traffic inside a step.
//   p.v : thread = (4 dims, head triple): the chunk's V codes are staged transposed (one word = one dim of 4 cells),
//         so a dot4 runs over 4 cells: 3 x 4 x ND dot4 per 4 cells.
// Measured on a Radeon PRO V620 (gfx1030), 2,048 queries of 2,051 cells at 32K, int8 KV: 20.5 ms the FP32 kernel
// (batches of 32, STRATA_ATTN_MERGE_V2=1), 13.2 ms this one, 10.0 ms with two digits.
#if (defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || \
     defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__)) && \
    __has_builtin(__builtin_amdgcn_sdot4) && __has_builtin(__builtin_amdgcn_mov_dpp)
#define STRATA_PA_DOT4 1
#else
#define STRATA_PA_DOT4 0
#endif
constexpr int DTH = 256;         // threads: 8 waves
constexpr int DCH = 64;          // cells per chunk (32 measured slower: 14.8 against 13.2 ms for 2,048 queries at 32K)
struct alignas(16) SmemD {       // 42.7 KB
    uint32_t qd[G][3][HD / 4];        // q digits: [head][digit][word of 4 dims]
    uint32_t vt[DCH / 4][HD];         // V codes, transposed: [cell quad][dim], byte j = cell 4 * quad + j
    uint32_t wd[4][DCH / 4][G][4];    // p * V scale digits: [scale group][cell quad][head][digit], byte j = cell j
    float sc[G][DCH];                 // q.k
    float vs[DCH][4];                 // V scale per (cell, group); 0 for a masked cell
    long long row[DCH];               // pool row, or -1
    float qmax[DTH / 32];
    float alpha[G];
    float lsum[G];
    float mrow[G];
    float vdown[G];
};

__device__ __forceinline__ int dot4_i8(uint32_t a, uint32_t b, int c) {
#if STRATA_PA_DOT4
    return __builtin_amdgcn_sdot4((int) a, (int) b, c, false);
#else
    __builtin_trap();
    return c;
#endif
}
// the value of lane ^ M within the lane's 16-lane row (DPP row_xmask, in registers)
template <int M> __device__ __forceinline__ float row_x(float v) {
#if STRATA_PA_DOT4
    return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x160 | M, 0xF, 0xF, false));
#else
    __builtin_trap();
    return v;
#endif
}
// ND balanced base-256 digits of four integers, digit d of element e in byte e of out[d]
template <int ND> __device__ __forceinline__ void digits4(const int* v4, uint32_t* out) {
#pragma unroll
    for (int d = 0; d < ND; ++d) out[d] = 0u;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        int v = v4[e];
#pragma unroll
        for (int d = 0; d < ND; ++d) {
            const int b = d == ND - 1 ? v : (int) (int8_t) v;
            out[d] |= ((uint32_t) b & 0xffu) << (8 * e);
            v = (v - b) >> 8;
        }
    }
}
template <int ND> __device__ __forceinline__ float undigit(const int* a) {
    if constexpr (ND == 3) return fmaf((float) a[2], 65536.0f, fmaf((float) a[1], 256.0f, (float) a[0]));
    else return fmaf((float) a[1], 256.0f, (float) a[0]);
}

// LW: K words (of 4 dims) a lane holds in q.k.  8 (default): half a scale group, one cell per wave, 3 x 8 x ND q-digit
// registers (127 VGPRs at ND = 3).  16 (STRATA_PA_DOT4_CFG=16): a whole group, two cells per wave, twice the registers
// (208 VGPRs) - fewer instructions but measured slower on a V620 (19.1 against 13.2 ms for 2,048 queries at 32K).
// 4 (a quarter group, a 16-lane row per (cell, head triple)) was measured too: 15.1 ms, more steps per chunk.
template <int ND, int LW>
__global__ void __launch_bounds__(DTH) prompt_attn_dot4_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                               const int32_t* __restrict__ ids,
                                                               const int32_t* __restrict__ steps, int n_kv_heads,
                                                               int page_size, float scale_log2,
                                                               float* __restrict__ attn, int cap) {
#if STRATA_PA_DOT4
    __shared__ SmemD S;
    // |q * qup|, |p * vs * vup| <= QMAX: ND balanced digits (127 * 65536 + 127 * 256 + 127 = 8355711), less a margin
    // for the products' rounding
    constexpr float QMAX = ND == 3 ? 8355000.0f : 32600.0f;
    constexpr int CPS = LW * (DTH / 32) / 8;   // cells per q.k step: 16, 8 or 4
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
    // q.k: 64 / LW lanes hold one (cell, head triple): the lane's slice of LW words, its head triple and its cell of
    // the step's CPS
    const int unit = t / (64 / LW);                       // 4 * CPS units
    const int sl_k = lane & (64 / LW - 1), ht_k = unit & 3, cis_k = unit >> 2;
    const int g_k = sl_k * LW / 16;
    const int w4 = t & 63, ht_v = t >> 6, g_v = w4 >> 4;                  // p.v: word of 4 dims, head triple, its group

    float qm = 0.0f;
    for (int i = t; i < G * HD; i += DTH) qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[wave] = qm;
    if (t < G) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }
    __syncthreads();
    qm = 0.0f;
#pragma unroll
    for (int i = 0; i < DTH / 32; ++i) qm = fmaxf(qm, S.qmax[i]);
    const float qup = qm > 0.0f ? QMAX / qm : 0.0f, qdown = scale_log2 * (qm * (1.0f / QMAX));
    // the q digits: word t % 64 of heads t / 64, + 4, + 8
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int h = (t >> 6) + 4 * j;
        const float4 a = *reinterpret_cast<const float4*>(q + (size_t) h * HD + 4 * w4);
        const int v4[4] = {(int) rintf(a.x * qup), (int) rintf(a.y * qup), (int) rintf(a.z * qup), (int) rintf(a.w * qup)};
        uint32_t d[ND];
        digits4<ND>(v4, d);
#pragma unroll
        for (int k = 0; k < ND; ++k) S.qd[h][k][w4] = d[k];
    }

    float acc[3][4];
#pragma unroll
    for (int h = 0; h < 3; ++h)
#pragma unroll
        for (int k = 0; k < 4; ++k) acc[h][k] = 0.0f;

    const int n_chunks = (n + DCH - 1) / DCH;
    for (int ci = 0; ci < n_chunks; ++ci) {
        const int c0 = ci * DCH;
        const int n_here = min(DCH, n - c0);
        // the chunk's pool rows and V scales (row / vs were last read before the previous chunk's third barrier)
        if (t < DCH) {
            long long r = -1;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (t < n_here) {
                const int cell = __ldg(ids + c0 + t);
                const long long page = (long long) __ldg(p.page_table + cell / page_size);
                if (page >= 0) {
                    r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
                    const uint16_t* sp = p.v_scale + r * (HD / KV_Q8_GROUP);
                    v = make_float4(__half2float(__ushort_as_half(__ldg(sp))), __half2float(__ushort_as_half(__ldg(sp + 1))),
                                    __half2float(__ushort_as_half(__ldg(sp + 2))), __half2float(__ushort_as_half(__ldg(sp + 3))));
                }
            }
            S.row[t] = r;
            *reinterpret_cast<float4*>(&S.vs[t][0]) = v;
        }
        __syncthreads();   // (1) also: the q digits are written (first chunk), the previous chunk's p.v has read vt / wd
        // ---- q.k
        {
            uint32_t qd[3][ND][LW];
#pragma unroll
            for (int h = 0; h < 3; ++h)
#pragma unroll
                for (int d = 0; d < ND; ++d)
#pragma unroll
                    for (int j = 0; j < LW / 4; ++j) {
                        const uint4 x = *reinterpret_cast<const uint4*>(&S.qd[3 * ht_k + h][d][LW * sl_k + 4 * j]);
                        qd[h][d][4 * j] = x.x; qd[h][d][4 * j + 1] = x.y; qd[h][d][4 * j + 2] = x.z; qd[h][d][4 * j + 3] = x.w;
                    }
            const int n_steps = (n_here + CPS - 1) / CPS;
            // step s's K words and K scale of this lane (zeros for a masked cell)
            const auto load_k = [&](int s, uint32_t* kw, float& ksc) {
#pragma unroll
                for (int j = 0; j < LW; ++j) kw[j] = 0u;
                ksc = 0.0f;
                const long long r = S.row[s * CPS + cis_k];
                if (r >= 0) {
                    const uint4* src = reinterpret_cast<const uint4*>(p.k_q + r * HD + sl_k * (LW * 4));
#pragma unroll
                    for (int j = 0; j < LW / 4; ++j) {
                        const uint4 x = __ldg(src + j);
                        kw[4 * j] = x.x; kw[4 * j + 1] = x.y; kw[4 * j + 2] = x.z; kw[4 * j + 3] = x.w;
                    }
                    ksc = __half2float(__ushort_as_half(__ldg(p.k_scale + r * (HD / KV_Q8_GROUP) + g_k)));
                }
            };
            // (loading step s + 1's K before step s's products was measured: 13.4 against 13.2 ms, no gain)
            for (int s = 0; s < n_steps; ++s) {
                const int c = s * CPS + cis_k;
                uint32_t kw[LW];
                float ksc = 0.0f;
                load_k(s, kw, ksc);
#pragma unroll
                for (int h = 0; h < 3; ++h) {
                    int a[ND];
#pragma unroll
                    for (int d = 0; d < ND; ++d) a[d] = 0;
#pragma unroll
                    for (int j = 0; j < LW; ++j)
#pragma unroll
                        for (int d = 0; d < ND; ++d) a[d] = dot4_i8(qd[h][d][j], kw[j], a[d]);
                    float f = undigit<ND>(a) * ksc;
                    f += row_x<1>(f);
                    f += row_x<2>(f);
                    if constexpr (LW <= 8) f += row_x<4>(f);
                    if constexpr (LW == 4) f += row_x<8>(f);
                    if (sl_k == 0) S.sc[3 * ht_k + h][c] = f;
                }
            }
        }
        // ---- the chunk's V codes, transposed: 4 cells x 4 dims per (thread, quad)
#pragma unroll
        for (int i = 0; i < DCH / 16; ++i) {
            const int cq = (t >> 6) + 4 * i;
            uint32_t a[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const long long r = S.row[4 * cq + j];
                a[j] = r >= 0 ? __ldg(reinterpret_cast<const uint32_t*>(p.v_q + r * HD + 4 * w4)) : 0u;
            }
            const uint32_t t0 = (a[0] & 0x00ff00ffu) | ((a[1] & 0x00ff00ffu) << 8);
            const uint32_t t1 = ((a[0] >> 8) & 0x00ff00ffu) | (a[1] & 0xff00ff00u);
            const uint32_t u0 = (a[2] & 0x00ff00ffu) | ((a[3] & 0x00ff00ffu) << 8);
            const uint32_t u1 = ((a[2] >> 8) & 0x00ff00ffu) | (a[3] & 0xff00ff00u);
            *reinterpret_cast<uint4*>(&S.vt[cq][4 * w4]) =
                make_uint4((t0 & 0xffffu) | (u0 << 16), (t1 & 0xffffu) | (u1 << 16), (t0 >> 16) | (u0 & 0xffff0000u),
                           (t1 >> 16) | (u1 & 0xffff0000u));
        }
        __syncthreads();   // (2)
        // ---- online softmax (log2 domain), then p times the V scale as digits: row t / 16 = head, 4 cells per thread
        if (t < G * 16) {
            const int h = t >> 4, sub = t & 15;
            static_assert(DCH == 64, "4 cells per thread: a cell quad's digits are one thread's");
            float x[4], vsc[4][4], mx = -INFINITY, vm = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                x[j] = S.row[c] >= 0 ? S.sc[h][c] * qdown : -INFINITY;
                mx = fmaxf(mx, x[j]);
                const float4 v = *reinterpret_cast<const float4*>(&S.vs[c][0]);
                vsc[j][0] = v.x; vsc[j][1] = v.y; vsc[j][2] = v.z; vsc[j][3] = v.w;
                vm = fmaxf(vm, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
            }
            mx = fmaxf(mx, row_x<1>(mx)); mx = fmaxf(mx, row_x<2>(mx)); mx = fmaxf(mx, row_x<4>(mx)); mx = fmaxf(mx, row_x<8>(mx));
            vm = fmaxf(vm, row_x<1>(vm)); vm = fmaxf(vm, row_x<2>(vm)); vm = fmaxf(vm, row_x<4>(vm)); vm = fmaxf(vm, row_x<8>(vm));
            const float m_old = S.mrow[h];
            const float m_new = fmaxf(m_old, mx);
            float e[4], sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                e[j] = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                sum += e[j];
            }
            sum += row_x<1>(sum); sum += row_x<2>(sum); sum += row_x<4>(sum); sum += row_x<8>(sum);
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[h] = a;
                S.lsum[h] = fmaf(S.lsum[h], a, sum);
                S.mrow[h] = m_new;
            }
            // the weights p * vs are rounded relative to this chunk's largest for this head (emax * vm bounds them):
            // a chunk of small weights keeps its relative precision.  Below 2^-100 of the running maximum the chunk
            // adds nothing an FP32 accumulator would keep (and 1 / emax would overflow): it is dropped.
            const float emax = mx == -INFINITY ? 0.0f : exp2f(mx - m_new);
            const float wmax = emax * vm;
            const bool keep = emax > 7.9e-31f && wmax > 1e-30f;
            const float vup = keep ? QMAX / wmax : 0.0f;
            if (sub == 0) S.vdown[h] = keep ? wmax * (1.0f / QMAX) : 0.0f;
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const int v4[4] = {(int) rintf((e[0] * vsc[0][g]) * vup), (int) rintf((e[1] * vsc[1][g]) * vup),
                                   (int) rintf((e[2] * vsc[2][g]) * vup), (int) rintf((e[3] * vsc[3][g]) * vup)};
                uint32_t d[3] = {0u, 0u, 0u};
                digits4<ND>(v4, d);
                *reinterpret_cast<uint4*>(&S.wd[g][sub][h][0]) = make_uint4(d[0], d[1], d[2], 0u);
            }
        }
        __syncthreads();   // (3)
        // ---- p.v
        {
            int ia[3][4][ND];
#pragma unroll
            for (int h = 0; h < 3; ++h)
#pragma unroll
                for (int k = 0; k < 4; ++k)
#pragma unroll
                    for (int d = 0; d < ND; ++d) ia[h][k][d] = 0;
            const int n_cq = (n_here + 3) / 4;
            for (int cq = 0; cq < n_cq; ++cq) {
                const uint4 vv = *reinterpret_cast<const uint4*>(&S.vt[cq][4 * w4]);
                const uint32_t v[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
                for (int h = 0; h < 3; ++h) {
                    const uint4 ww = *reinterpret_cast<const uint4*>(&S.wd[g_v][cq][3 * ht_v + h][0]);
                    const uint32_t w[3] = {ww.x, ww.y, ww.z};
#pragma unroll
                    for (int k = 0; k < 4; ++k)
#pragma unroll
                        for (int d = 0; d < ND; ++d) ia[h][k][d] = dot4_i8(w[d], v[k], ia[h][k][d]);
                }
            }
#pragma unroll
            for (int h = 0; h < 3; ++h) {
                const float al = S.alpha[3 * ht_v + h], vdown = S.vdown[3 * ht_v + h];
#pragma unroll
                for (int k = 0; k < 4; ++k) acc[h][k] = fmaf(acc[h][k], al, undigit<ND>(ia[h][k]) * vdown);
            }
        }
        // no barrier here: the next chunk writes row / vs (read last before barrier 3) and then waits at barrier 1
    }
#pragma unroll
    for (int h = 0; h < 3; ++h) {
        const float l = S.lsum[3 * ht_v + h];
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        *reinterpret_cast<float4*>(attn + (size_t) (3 * ht_v + h) * HD + 4 * w4) =
            make_float4(acc[h][0] * inv, acc[h][1] * inv, acc[h][2] * inv, acc[h][3] * inv);
    }
#else
    __builtin_trap();
#endif
}

// gfx103x only, and only on request: 0 off, 3 or 2 digits
int hip_dot4_digits() {
    static const int want = [] {
        const char* e = std::getenv("STRATA_PA_DOT4");
        return e == nullptr ? 0 : e[0] == '1' ? 3 : e[0] == '2' ? 2 : 0;
    }();
    if (want == 0) return 0;
    static int arch[64] = {};   // 0 unknown, 1 gfx103x, 2 other
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) { (void) hipGetLastError(); return 0; }
    if (arch[dev] == 0) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, dev) != hipSuccess) { (void) hipGetLastError(); return 0; }
        arch[dev] = std::strncmp(prop.gcnArchName, "gfx103", 6) == 0 ? 1 : 2;
        static bool told = false;
        if (!told) {
            told = true;
            std::fprintf(stderr, "strata: STRATA_PA_DOT4: the prompt attention on v_dot4_i32_i8, %d digits, %s (%s)\n",
                         want, arch[dev] == 1 ? "on" : "unavailable", prop.gcnArchName);
        }
    }
    return arch[dev] == 1 ? want : 0;
}

bool launch_dot4(int digits, const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                 int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t st) {
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        // STRATA_PA_DOT4_CFG=16 (A/B): 16 K words per lane instead of 8
        static const bool wide = [] { const char* e = std::getenv("STRATA_PA_DOT4_CFG"); return e && std::atoi(e) == 16; }();
#define STRATA_PA_DOT4_GO(ND_, LW_)                                                                                   \
    prompt_attn_dot4_kernel<ND_, LW_><<<dim3((unsigned) nb, (unsigned) s.n_head_kv), DTH, 0, st>>>(                  \
        q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv, (int) s.page_size, \
        scale_log2, attn + q0 * s.n_head * HD, (int) cap)
        if (digits == 2 && wide) STRATA_PA_DOT4_GO(2, 16);
        else if (digits == 2) STRATA_PA_DOT4_GO(2, 8);
        else if (wide) STRATA_PA_DOT4_GO(3, 16);
        else STRATA_PA_DOT4_GO(3, 8);
#undef STRATA_PA_DOT4_GO
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prompt_attn_batch (dot4): %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}
#endif

}  // namespace

bool qsa_prompt_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return true;
    bool volta = false, turing = false;   // per call, from the CURRENT device (a layer split can mix Turing with newer cards)
    {   // sm_75 or newer: the MMA above compiles for both.  sm_80+ runs the cp.async kernel (launch_i8); Turing has
        // no cp.async, so it runs the v1 kernel (launch<1>, same accuracy, another summation order).  An older card
        // keeps the old kernel.
        // #371: the compute capability with its minor - sm_70 (V100) has no m16n8k8 (the kernels trap below sm_75)
#if defined(STRATA_HIP_GFX906)
        return false;   // gfx906: the tensor-core and WMMA kernels are not for it; the FP32 kernel runs
#endif
        static int cc[64] = {};
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
        if (cc[dev] == 0) {
            int major = 0, minor = 0;
            if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
                cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            // STRATA_QSA_WARP=1|attn (an A/B arm): the pre-sm_80 kernels on any card, as RTX 20 runs them
            const char* w = std::getenv("STRATA_QSA_WARP");
            cc[dev] = w && (!std::strcmp(w, "1") || !std::strcmp(w, "attn")) ? 75
                      : 10 * strata::cc_major_of(major) + strata::cc_minor_of(minor);
        }
        if (cc[dev] < 70) return false;
        volta = cc[dev] < 75;     // sm_70: the m8n8k4 kernel (STRATA_PA_VOLTA); STRATA_QSA_WARP=1 still forces the old one
        turing = cc[dev] < 80;
    }
#if defined(__HIPCC__)
    // the tensor-core kernels are compiled out on AMD (its major version is not a CUDA sm); RDNA4 has its own int8-KV
    // matrix-core kernel, opt-in (STRATA_HIP_WMMA=1); everything else keeps the old kernel
    (void) turing; (void) volta;
    // RDNA2: the dot4 kernel, opt-in (STRATA_PA_DOT4=1, or 2 for the two-digit one)
    if (pools.k_q != nullptr && pools.v_q != nullptr && pools.k_scale != nullptr && pools.v_scale != nullptr &&
        pools.k_q4 == nullptr && pools.v_q4 == nullptr && s.head_dim == HD && s.n_head == (int64_t) G * s.n_head_kv &&
        cap > 0 && ids && steps && pools.page_table) {
        if (const int digits = hip_dot4_digits(); digits != 0)
            return launch_dot4(digits, q, pools, ids, steps, cap, s, attn, n_q, (cudaStream_t) stream);
    }
    if (pools.k_q != nullptr && pools.v_q != nullptr && pools.k_scale != nullptr && pools.v_scale != nullptr &&
        pools.k_q4 == nullptr && pools.v_q4 == nullptr && s.head_dim == HD && s.n_head == (int64_t) G * s.n_head_kv &&
        cap > 0 && ids && steps && pools.page_table && hip_wmma_usable())
        return launch_wmma(q, pools, ids, steps, cap, s, attn, n_q, (cudaStream_t) stream);
    return false;
#endif
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !ids || !steps || !pools.page_table)
        return false;
    cudaStream_t st = (cudaStream_t) stream;
    if (pools.k_q4 != nullptr) {   // Q4_0 K and V (--kv q4_0): mode 4.  STRATA_PROMPT_ATTN_Q4=0: the old kernel (A/B)
        static const bool q4_off = [] {
            const char* v = std::getenv("STRATA_PROMPT_ATTN_Q4");
            return v != nullptr && v[0] == '0';
        }();
        // sm_80+ only: on Turing mode 4 would run as pairs of m16n8k8 MMAs, which no parity run has checked yet
        if (q4_off || turing || pools.v_q4 == nullptr) return false;
        return launch<4>(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    if (pools.k_q != nullptr && pools.v_q4 != nullptr) {   // hybrid K8V4: int8 K + dequantized-q4 V
        if (!pools.k_scale) return false;
        if (volta) return launch70<3>(q, pools, ids, steps, cap, s, attn, n_q, st);
        return launch<3>(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    if (pools.k_q != nullptr) {
        if (!pools.v_q || !pools.k_scale || !pools.v_scale) return false;
        // STRATA_PROMPT_ATTN_V1=1 (debug): the first version, same accuracy, another summation order - the control
        // for how far the model amplifies an FP32-level change.  Turing always takes it: v2's cp.async does not
        // exist before sm_80.
        static const bool v1 = std::getenv("STRATA_PROMPT_ATTN_V1") != nullptr;
        if (volta) return launch70<1>(q, pools, ids, steps, cap, s, attn, n_q, st);
        if (v1 || turing) return launch<1>(q, pools, ids, steps, cap, s, attn, n_q, st);
        return launch_i8(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    if (!pools.k_pool || !pools.v_pool) return false;
    if (volta) return launch70<0>(q, pools, ids, steps, cap, s, attn, n_q, st);
    return launch<0>(q, pools, ids, steps, cap, s, attn, n_q, st);
}

}  // namespace strata::kernels
