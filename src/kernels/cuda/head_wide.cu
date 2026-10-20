// src/kernels/cuda/head_wide.cu - STRATA_HEAD_WIDE=1 (HIP, opt-in): the verify window's output head.
//
// The head is a Q6_K matrix of 248,320 rows of 2,560 weights (10 blocks of 210 bytes, 497 MiB) times the window's
// 1-5 activation columns.  native_mmvq's kernel is one workgroup of 4 waves per one or two rows; thread (wave w,
// lane l) owns word l of blocks w, w + 4, w + 8 and reads, for every block and every column, two activation words and
// two block scales from memory.  On a Radeon PRO V620 (gfx1030) that is 0.97 ms for one column - the weight read,
// 538 GB/s - and 1.64 ms for five: past two columns the launch is bound by the per-column work, not by the weights.
// (tests/hip/head_lab.cpp: synthetic weights of the real shape, medians of 9 batches of 20 calls, the card idle.)
//
// What a thread reads of the ACTIVATIONS does not depend on the row.  Here the workgroups are persistent (workgroup
// g computes rows g, g + grid, ...): a thread loads its activation words ONCE, into registers, and the activation
// blocks' scales once into a table in shared memory.  The weights of the next row are requested while the current
// row is computed, so they are in flight across the columns' arithmetic and the row's barrier.  A row then costs its
// weight loads and, per column, two dot4, two integer multiplies and the float tail.
//
// EVERY LOGIT IS native_mmvq's, BIT FOR BIT.  The integer dot products are exact; the float tail is the same
// expression (q6_q8_dot_impl's, with the per-thread sum over its blocks in block order - the fused multiply-adds
// chain through a thread's blocks, which is why a thread keeps all of its blocks); the four waves' partial sums meet
// in lane order (((w0 + w1) + w2) + w3, the default kernel's `tmp += partial[l]` for l = 0, 1, 2) and the 32 lanes
// are summed by the xor butterfly (s26_tsum.cuh's transposed form: the same adds).  The one difference is which wave
// does the last step: the wave that owns blocks 3 and 7 (the lightest) reads the other three from shared memory.
//
// The greedy pick and the non-finite check ride in the launch: the reducing wave already holds every logit it
// writes, so each workgroup keeps the first of its rows in the selection order (the largest value above -inf, the
// lowest row on a tie: rows ascend within a workgroup and the compare is a strict `>`) and whether any of its logits
// was non-finite, and a second, small launch merges the workgroups' candidates by the same order (the larger value,
// the smaller index on a tie) - sample_tokens' greedy answer, and logits_nonfinite_rows' flag.
//
// Two kernels: head_q6k_wide3_kernel (the default) and head_q6k_wide_kernel (STRATA_HEAD_WIDE_V=2: the activation
// scales in registers too and one xor butterfly per column; more registers per column, so fewer waves resident per
// SIMD at 4 and 5 columns - 1.08 / 1.33 ms against 1.03 / 1.11 ms with the pick, where native_mmvq + the greedy pick +
// the non-finite check are 1.43 / 1.70 ms).
#include "strata/kernels/head_wide.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "s26_tsum.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int kRowIn = 2560;           // weights per row
constexpr int kBlocks = kRowIn / 256;  // Q6_K blocks per row
constexpr int kBlockBytes = 210;       // block_q6_K: ql[128] qh[64] scales[16] d (f16)
constexpr int kMaxGrid = 16384;        // workgroups (the scratch's bound)
constexpr int kDefaultGrid = 4096;

struct HeadCand {
    float v;
    int32_t i;
    uint32_t bad;
    uint32_t pad;
};

int g_grid_override = 0;
int g_variant_override = 0;

int head_grid() {
    if (g_grid_override > 0) return g_grid_override < kMaxGrid ? g_grid_override : kMaxGrid;
    static const int grid = [] {
        const char* v = std::getenv("STRATA_HEAD_WIDE_GRID");
        const int g = v ? std::atoi(v) : 0;
        return g > 0 ? (g < kMaxGrid ? g : kMaxGrid) : kDefaultGrid;
    }();
    return grid;
}

#if defined(__HIPCC__)

struct Q81 {
    half2 ds;
    int8_t qs[32];
};

__device__ __forceinline__ int load_i32(const uint8_t* p) {   // 2-byte aligned at worst: one dword load on gfx10
    int v;
    __builtin_memcpy(&v, p, 4);
    return v;
}

__device__ __forceinline__ float xor_sum(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o, 32);
    return x;
}

// STRATA_HEAD_WIDE_V=2.  grid `row_step` workgroups of 4 waves; cand[g * NC + j]: workgroup g's candidate and flag of
// column j.  The weights of the NEXT row are requested while this row is computed (each block slot's five loads
// right after its values are decoded), so they are in flight across the columns' arithmetic and the barrier: a wave
// that waits at the barrier is not also waiting to start its loads.
template <int NC>
__launch_bounds__(128)
__global__ void head_q6k_wide_kernel(const uint8_t* __restrict__ w, const Q81* __restrict__ x, float* __restrict__ y,
                                     int n_out, int row_step, HeadCand* __restrict__ cand) {
    const int lane = int(threadIdx.x), wave = int(threadIdx.y);
    const int iqs = lane;
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int qh_word = 8 * (iqs / 16) + iqs % 8;
    const int slots = wave < kBlocks - 8 ? 3 : 2;   // blocks wave, wave + 4 (, wave + 8)
    // this thread's activations: per column and block slot, two words and their blocks' scales
    int u0[NC][3], u1[NC][3];
    float e0[NC][3], e1[NC][3];
#pragma unroll
    for (int s = 0; s < 3; ++s) {
        const int kbx = wave + 4 * s < kBlocks ? wave + 4 * s : wave;   // (the absent slot repeats slot 0, unused)
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const Q81* xb = x + j * (kRowIn / 32) + kbx * 8 + bq8_offset;
            u0[j][s] = reinterpret_cast<const int*>(xb[0].qs)[iqs % 8];
            e0[j][s] = __low2float(xb[0].ds);
            u1[j][s] = reinterpret_cast<const int*>(xb[2].qs)[iqs % 8];
            e1[j][s] = __low2float(xb[2].ds);
        }
    }
    __shared__ float partial[2][3][NC][32];
    float bv[NC];
    int bi[NC];
    uint32_t bad = 0u;
#pragma unroll
    for (int j = 0; j < NC; ++j) { bv[j] = __int_as_float(0xff800000); bi[j] = n_out; }
    // the raw words of this thread's blocks of one row
    int rl[3], rh[3], rs0[3], rs1[3];
    half rd[3];
    const std::size_t lane_ql = 4 * iqs, lane_qh = 128 + 4 * qh_word, lane_sc = 192 + scale_offset;
    auto load_row = [&](int row) {
        const uint8_t* wr = w + std::size_t(row) * (kBlocks * kBlockBytes) + wave * kBlockBytes;
#pragma unroll
        for (int s = 0; s < 3; ++s) {
            if (s < slots) {
                const uint8_t* wb = wr + s * (4 * kBlockBytes);
                rl[s] = load_i32(wb + lane_ql);
                rh[s] = load_i32(wb + lane_qh);
                rs0[s] = reinterpret_cast<const int8_t*>(wb)[lane_sc];
                rs1[s] = reinterpret_cast<const int8_t*>(wb)[lane_sc + 4];
                rd[s] = *reinterpret_cast<const half*>(wb + 208);
            }
        }
    };
    int row = int(blockIdx.x);
    if (row < n_out) load_row(row);
    int buf = 0;
    for (; row < n_out; row += row_step, buf ^= 1) {
        const int next = row + row_step < n_out ? row + row_step : row;   // (the last row reloads itself, unused)
        const uint8_t* wn = w + std::size_t(next) * (kBlocks * kBlockBytes) + wave * kBlockBytes;
        float tmp[NC];
#pragma unroll
        for (int j = 0; j < NC; ++j) tmp[j] = 0.0f;
#pragma unroll
        for (int s = 0; s < 3; ++s) {
            if (s < slots) {
                const int vl = rl[s];
                const int vh = rh[s] >> vh_shift;
                const int sc0 = rs0[s], sc1 = rs1[s];
                const float d = __half2float(rd[s]);
                // the weights minus 32, bytewise: every byte of (vil | vih) is 0..63, so adding 0x60 never carries
                // and flipping the top bit completes the + 0xe0 - __vsubss4(vil | vih, 0x20202020)'s bytes
                const int vi0 = int((uint32_t((vl & 0x0f0f0f0f) | ((vh << 4) & 0x30303030)) + 0x60606060u) ^ 0x80808080u);
                const int vi1 = int((uint32_t(((vl >> 4) & 0x0f0f0f0f) | (vh & 0x30303030)) + 0x60606060u) ^ 0x80808080u);
                {   // the next row's words of this slot
                    const uint8_t* wb = wn + s * (4 * kBlockBytes);
                    rl[s] = load_i32(wb + lane_ql);
                    rh[s] = load_i32(wb + lane_qh);
                    rs0[s] = reinterpret_cast<const int8_t*>(wb)[lane_sc];
                    rs1[s] = reinterpret_cast<const int8_t*>(wb)[lane_sc + 4];
                    rd[s] = *reinterpret_cast<const half*>(wb + 208);
                }
#pragma unroll
                for (int j = 0; j < NC; ++j) {
                    float sumf = 0.0f;                                         // q6_q8_dot_impl's tail, verbatim
                    sumf += e0[j][s] * (STRATA_DP4A(vi0, u0[j][s], 0) * sc0);
                    sumf += e1[j][s] * (STRATA_DP4A(vi1, u1[j][s], 0) * sc1);
                    tmp[j] += d * sumf;
                }
            }
        }
        if (wave != 3) {
#pragma unroll
            for (int j = 0; j < NC; ++j) partial[buf][wave][j][lane] = tmp[j];
        }
        __syncthreads();
        if (wave == 3) {
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                float t = partial[buf][0][j][lane];
                t += partial[buf][1][j][lane];
                t += partial[buf][2][j][lane];
                t += tmp[j];
                const float sum = xor_sum(t);
                if (lane == 0) y[std::size_t(j) * n_out + row] = sum;
                if (sum > bv[j]) { bv[j] = sum; bi[j] = row; }
                bad |= uint32_t((__float_as_uint(sum) & 0x7f800000u) == 0x7f800000u) << j;
            }
        }
    }
    if (wave == 3 && lane == 0) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            HeadCand c;
            c.v = bv[j]; c.i = bi[j]; c.bad = (bad >> j) & 1u; c.pad = 0u;
            cand[int(blockIdx.x) * NC + j] = c;
        }
    }
}

// The default kernel: the activation words stay in registers, the activation blocks' scales come from a table in
// shared memory (one read of two per column and block slot), and the reducing wave sums the columns with one
// transposed butterfly (s26_tsum.cuh: the xor butterfly's adds, each lane ending with one column's sum), so a lane
// tracks one column's pick.
template <int NC>
__launch_bounds__(128)
__global__ void head_q6k_wide3_kernel(const uint8_t* __restrict__ w, const Q81* __restrict__ x, float* __restrict__ y,
                                      int n_out, int row_step, HeadCand* __restrict__ cand) {
    const int lane = int(threadIdx.x), wave = int(threadIdx.y);
    const int iqs = lane;
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int qh_word = 8 * (iqs / 16) + iqs % 8;
    const int slots = wave < kBlocks - 8 ? 3 : 2;
    constexpr int XB = kRowIn / 32;   // Q8_1 blocks per column
    __shared__ float etab[NC * XB];
    for (int i = wave * 32 + lane; i < NC * XB; i += 128) etab[i] = __low2float(x[i].ds);
    int u0[NC][3], u1[NC][3];
#pragma unroll
    for (int s = 0; s < 3; ++s) {
        const int kbx = wave + 4 * s < kBlocks ? wave + 4 * s : wave;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const Q81* xb = x + j * XB + kbx * 8 + bq8_offset;
            u0[j][s] = reinterpret_cast<const int*>(xb[0].qs)[iqs % 8];
            u1[j][s] = reinterpret_cast<const int*>(xb[2].qs)[iqs % 8];
        }
    }
    __syncthreads();
    const float* et = etab + wave * 8 + bq8_offset;   // + j * XB + s * 32 (+ 2)
    __shared__ float partial[2][3][NC][32];
    constexpr int P = s26ts::pow2_ceil(NC);
    const int my_col = s26ts::tsum_token<P>(lane);
    const bool writer = my_col < NC && lane == s26ts::tsum_lane<P>(my_col);
    float bv = __int_as_float(0xff800000);
    int bi = n_out;
    uint32_t bad = 0u;
    int rl[3], rh[3], rs0[3], rs1[3];
    half rd[3];
    const std::size_t lane_ql = 4 * iqs, lane_qh = 128 + 4 * qh_word, lane_sc = 192 + scale_offset;
    int row = int(blockIdx.x);
    if (row < n_out) {
        const uint8_t* wr = w + std::size_t(row) * (kBlocks * kBlockBytes) + wave * kBlockBytes;
#pragma unroll
        for (int s = 0; s < 3; ++s) {
            if (s < slots) {
                const uint8_t* wb = wr + s * (4 * kBlockBytes);
                rl[s] = load_i32(wb + lane_ql);
                rh[s] = load_i32(wb + lane_qh);
                rs0[s] = reinterpret_cast<const int8_t*>(wb)[lane_sc];
                rs1[s] = reinterpret_cast<const int8_t*>(wb)[lane_sc + 4];
                rd[s] = *reinterpret_cast<const half*>(wb + 208);
            }
        }
    }
    int buf = 0;
    for (; row < n_out; row += row_step, buf ^= 1) {
        const int next = row + row_step < n_out ? row + row_step : row;
        const uint8_t* wn = w + std::size_t(next) * (kBlocks * kBlockBytes) + wave * kBlockBytes;
        float tmp[NC];
#pragma unroll
        for (int j = 0; j < NC; ++j) tmp[j] = 0.0f;
#pragma unroll
        for (int s = 0; s < 3; ++s) {
            if (s < slots) {
                const int vl = rl[s];
                const int vh = rh[s] >> vh_shift;
                const int sc0 = rs0[s], sc1 = rs1[s];
                const float d = __half2float(rd[s]);
                const int vi0 = int((uint32_t((vl & 0x0f0f0f0f) | ((vh << 4) & 0x30303030)) + 0x60606060u) ^ 0x80808080u);
                const int vi1 = int((uint32_t(((vl >> 4) & 0x0f0f0f0f) | (vh & 0x30303030)) + 0x60606060u) ^ 0x80808080u);
                {
                    const uint8_t* wb = wn + s * (4 * kBlockBytes);
                    rl[s] = load_i32(wb + lane_ql);
                    rh[s] = load_i32(wb + lane_qh);
                    rs0[s] = reinterpret_cast<const int8_t*>(wb)[lane_sc];
                    rs1[s] = reinterpret_cast<const int8_t*>(wb)[lane_sc + 4];
                    rd[s] = *reinterpret_cast<const half*>(wb + 208);
                }
#pragma unroll
                for (int j = 0; j < NC; ++j) {
                    const float e0 = et[j * XB + s * 32], e1 = et[j * XB + s * 32 + 2];
                    float sumf = 0.0f;                                         // q6_q8_dot_impl's tail, verbatim
                    sumf += e0 * (STRATA_DP4A(vi0, u0[j][s], 0) * sc0);
                    sumf += e1 * (STRATA_DP4A(vi1, u1[j][s], 0) * sc1);
                    tmp[j] += d * sumf;
                }
            }
        }
        if (wave != 3) {
#pragma unroll
            for (int j = 0; j < NC; ++j) partial[buf][wave][j][lane] = tmp[j];
        }
        __syncthreads();
        if (wave == 3) {
            float v[P];
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                float t = partial[buf][0][j][lane];
                t += partial[buf][1][j][lane];
                t += partial[buf][2][j][lane];
                t += tmp[j];
                v[j] = t;
            }
#pragma unroll
            for (int j = NC; j < P; ++j) v[j] = 0.0f;
            const float sum = s26ts::tsum<P>(v, lane);
            if (writer) y[std::size_t(my_col) * n_out + row] = sum;
            if (sum > bv) { bv = sum; bi = row; }
            bad |= uint32_t((__float_as_uint(sum) & 0x7f800000u) == 0x7f800000u);
        }
    }
    if (wave == 3 && writer) {
        HeadCand c;
        c.v = bv; c.i = bi; c.bad = bad; c.pad = 0u;
        cand[int(blockIdx.x) * NC + my_col] = c;
    }
}

// one block of 256 threads per column: the first of the workgroups' candidates in the selection order, and the OR of
// their flags
__launch_bounds__(256)
__global__ void head_pick_kernel(const HeadCand* __restrict__ cand, int grid, int ncols, int n_out,
                                 int32_t* __restrict__ out, uint32_t* __restrict__ flags) {
    const int j = int(blockIdx.x);
    float bv = __int_as_float(0xff800000);
    int best = n_out;
    uint32_t bad = 0u;
    for (int g = int(threadIdx.x); g < grid; g += int(blockDim.x)) {
        const HeadCand c = cand[g * ncols + j];
        if (c.v > bv || (c.v == bv && c.i < best)) { bv = c.v; best = c.i; }
        bad |= c.bad;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, off, 32);
        const int oi = __shfl_xor_sync(0xffffffffu, best, off, 32);
        bad |= __shfl_xor_sync(0xffffffffu, bad, off, 32);
        if (ov > bv || (ov == bv && oi < best)) { bv = ov; best = oi; }
    }
    __shared__ float sv[8];
    __shared__ int si[8];
    __shared__ uint32_t sb[8];
    const int wave = int(threadIdx.x >> 5), lane = int(threadIdx.x & 31);
    if (lane == 0) { sv[wave] = bv; si[wave] = best; sb[wave] = bad; }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int k = 1; k < 8; ++k) {
            if (sv[k] > bv || (sv[k] == bv && si[k] < best)) { bv = sv[k]; best = si[k]; }
            bad |= sb[k];
        }
        if (out != nullptr) out[j] = best < n_out ? best : 0;
        if (flags != nullptr) {
            flags[j * kNonfiniteBlocks] = bad ? 1u : 0u;
#pragma unroll
            for (int b = 1; b < kNonfiniteBlocks; ++b) flags[j * kNonfiniteBlocks + b] = 0u;
        }
    }
}

// The draft also needs the chosen token's probability. Merge candidates first,
// then copy row_top_prob_kernel's arithmetic and geometry verbatim.
__launch_bounds__(1024)
__global__ void head_pick_prob_kernel(const HeadCand* __restrict__ cand, int grid, int n_out,
                                      const float* __restrict__ logits, int32_t* __restrict__ out,
                                      float* __restrict__ probability) {
    float bv = __int_as_float(0xff800000);
    int best = n_out;
    for (int g = int(threadIdx.x); g < grid; g += int(blockDim.x)) {
        const HeadCand c = cand[g];
        if (c.v > bv || (c.v == bv && c.i < best)) { bv = c.v; best = c.i; }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, off, 32);
        const int oi = __shfl_xor_sync(0xffffffffu, best, off, 32);
        if (ov > bv || (ov == bv && oi < best)) { bv = ov; best = oi; }
    }
    __shared__ float sv[32], part[32];
    __shared__ int si[32], pick;
    const int wave = int(threadIdx.x >> 5), lane = int(threadIdx.x & 31);
    if (lane == 0) { sv[wave] = bv; si[wave] = best; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 1; k < 32; ++k)
            if (sv[k] > bv || (sv[k] == bv && si[k] < best)) { bv = sv[k]; best = si[k]; }
        pick = best < n_out ? best : 0;
        *out = pick;
    }
    __syncthreads();
    const float m = logits[pick];
    float s = 0.0f;
    for (int i = threadIdx.x; i < n_out; i += blockDim.x) s += __expf(logits[i] - m);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int w = 0; w < (int) (blockDim.x >> 5); ++w) tot += part[w];
        *probability = 1.0f / tot;
    }
}

int head_variant() {
    if (g_variant_override > 0) return g_variant_override;
    static const int v = [] {
        const char* e = std::getenv("STRATA_HEAD_WIDE_V");
        const int x = e ? std::atoi(e) : 0;
        return x == 2 ? 2 : 3;
    }();
    return v;
}

template <int NC>
void launch_head(const void* weights, const void* x, float* y, int n_out, int grid, HeadCand* cand, cudaStream_t s) {
    if (head_variant() == 3)
        head_q6k_wide3_kernel<NC><<<unsigned(grid), dim3(32, 4), 0, s>>>(static_cast<const uint8_t*>(weights),
                                                                       static_cast<const Q81*>(x), y, n_out, grid, cand);
    else
    head_q6k_wide_kernel<NC><<<unsigned(grid), dim3(32, 4), 0, s>>>(static_cast<const uint8_t*>(weights),
                                                                  static_cast<const Q81*>(x), y, n_out, grid, cand);
}

#endif  // __HIPCC__

}  // namespace

bool head_wide_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_HEAD_WIDE");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

void head_wide_set_grid(int grid) { g_grid_override = grid; }
void head_wide_set_variant(int v) { g_variant_override = v; }

std::size_t head_wide_scratch_bytes() { return std::size_t(kMaxGrid) * kHeadWideMaxCols * sizeof(HeadCand); }

bool head_wide_supported(int ggml_type, int n_in, int n_out, int ncols) {
#if defined(__HIPCC__)
    return ggml_type == 14 && n_in == kRowIn && n_out >= 1 && ncols >= 1 && ncols <= kHeadWideMaxCols;
#else
    (void) ggml_type; (void) n_in; (void) n_out; (void) ncols;
    return false;
#endif
}

void head_wide(const void* weights, const void* x_q8_1, float* logits, int n_in, int n_out, int ncols, void* scratch,
               int32_t* out, uint32_t* nf_flags, void* stream) {
    if (!head_wide_supported(14, n_in, n_out, ncols) || !weights || !x_q8_1 || !logits || !scratch || !stream)
        throw std::invalid_argument("head_wide: needs a 2560-wide Q6_K head, 1..8 columns, device buffers and a stream");
#if defined(__HIPCC__)
    const auto s = static_cast<cudaStream_t>(stream);
    auto* cand = static_cast<HeadCand*>(scratch);
    const int grid = n_out < head_grid() ? n_out : head_grid();
    switch (ncols) {
        case 1: launch_head<1>(weights, x_q8_1, logits, n_out, grid, cand, s); break;
        case 2: launch_head<2>(weights, x_q8_1, logits, n_out, grid, cand, s); break;
        case 3: launch_head<3>(weights, x_q8_1, logits, n_out, grid, cand, s); break;
        case 4: launch_head<4>(weights, x_q8_1, logits, n_out, grid, cand, s); break;
        case 5: launch_head<5>(weights, x_q8_1, logits, n_out, grid, cand, s); break;
        case 6: launch_head<6>(weights, x_q8_1, logits, n_out, grid, cand, s); break;
        case 7: launch_head<7>(weights, x_q8_1, logits, n_out, grid, cand, s); break;
        default: launch_head<8>(weights, x_q8_1, logits, n_out, grid, cand, s); break;
    }
    if (out != nullptr || nf_flags != nullptr)
        head_pick_kernel<<<unsigned(ncols), 256, 0, s>>>(cand, grid, ncols, n_out, out, nf_flags);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("head_wide launch: ") + cudaGetErrorString(e));
#endif
}

void head_wide_draft(const void* weights, const void* x_q8_1, float* logits, int n_in, int n_out, void* scratch,
                     int32_t* out, float* probability, void* stream) {
    if (!head_wide_supported(14, n_in, n_out, 1) || !weights || !x_q8_1 || !logits || !scratch || !out || !probability || !stream)
        throw std::invalid_argument("head_wide_draft: needs one Q6_K column and device buffers");
#if defined(__HIPCC__)
    const auto s = static_cast<cudaStream_t>(stream);
    auto* cand = static_cast<HeadCand*>(scratch);
    const int grid = n_out < head_grid() ? n_out : head_grid();
    launch_head<1>(weights, x_q8_1, logits, n_out, grid, cand, s);
    head_pick_prob_kernel<<<1, 1024, 0, s>>>(cand, grid, n_out, logits, out, probability);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("head_wide_draft launch: ") + cudaGetErrorString(e));
#endif
}

}  // namespace strata::kernels
