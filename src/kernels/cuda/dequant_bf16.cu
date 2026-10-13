// src/kernels/cuda/dequant_bf16.cu - see include/strata/kernels/dequant_bf16.hpp.
//
// Arithmetic transcribed from ggml/src/ggml-quants.c at the pinned llama.cpp (MIT License, Copyright (c) 2023-2026
// The ggml authors): dequantize_row_q2_0/q4_0/q5_0/q8_0/q3_K/q4_K/q5_K/q6_K/iq4_nl/iq4_xs.
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

__device__ __forceinline__ float h2f(const uint8_t* p) {
    return __half2float(__ushort_as_half((uint16_t) (p[0] | (p[1] << 8))));
}
__device__ __forceinline__ uint16_t f2bf(float f) {
    uint32_t u = __float_as_uint(f);
    // a NaN (a NaN scale in the block) stays a quiet NaN, as in ggml_compute_fp32_to_bf16 and `bf16_from_f32`:
    // the rounding add below would carry it into -0 or inf
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 64u);
    u += 0x7fffu + ((u >> 16) & 1u);          // round to nearest even
    return (uint16_t) (u >> 16);
}
__device__ __forceinline__ void put(uint16_t* o, int i, float v) { o[i] = f2bf(v); }
struct H16 { uint16_t v; };
__device__ __forceinline__ void put(H16* o, int i, float v) { o[i].v = __half_as_ushort(__float2half_rn(v)); }
__device__ __forceinline__ void put(float* o, int i, float v) { o[i] = v; }

__constant__ int8_t kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// One 32-element group `g` (row-major over the whole slice); `out` points at that group's 32 outputs.
template <int TYPE, typename T>
__device__ __forceinline__ void group32(const uint8_t* row_blocks, int gi_in_row, T* out) {
    if constexpr (TYPE == 42) {                                   // Q2_0: 64 per block of 18 B
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 2) * 18;
        const float d = h2f(b);
        const int e0 = (gi_in_row % 2) * 32;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const int e = e0 + j;
            const int q = (b[2 + e / 4] >> ((e % 4) * 2)) & 3;
            put(out, j, (float) (q - 1) * d);
        }
    } else if constexpr (TYPE == 2) {                              // Q4_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 18;
        const float d = h2f(b);
        for (int j = 0; j < 16; ++j) {
            put(out, j, (float) ((b[2 + j] & 0x0F) - 8) * d);
            put(out, j + 16, (float) ((b[2 + j] >> 4) - 8) * d);
        }
    } else if constexpr (TYPE == 6) {                              // Q5_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 22;
        const float d = h2f(b);
        const uint32_t qh = (uint32_t) b[2] | ((uint32_t) b[3] << 8) | ((uint32_t) b[4] << 16) | ((uint32_t) b[5] << 24);
        for (int j = 0; j < 16; ++j) {
            const int xh0 = ((qh >> j) << 4) & 0x10;
            const int xh1 = (qh >> (j + 12)) & 0x10;
            put(out, j, (float) (((b[6 + j] & 0x0F) | xh0) - 16) * d);
            put(out, j + 16, (float) (((b[6 + j] >> 4) | xh1) - 16) * d);
        }
    } else if constexpr (TYPE == 7) {                              // Q5_1
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 24;
        const float d = h2f(b), m = h2f(b + 2);
        const uint32_t qh = (uint32_t) b[4] | ((uint32_t) b[5] << 8) |
                            ((uint32_t) b[6] << 16) | ((uint32_t) b[7] << 24);
        for (int j = 0; j < 16; ++j) {
            const int xh0 = ((qh >> j) & 1) << 4;
            const int xh1 = ((qh >> (j + 16)) & 1) << 4;
            put(out, j, (float) ((b[8 + j] & 15) | xh0) * d + m);
            put(out, j + 16, (float) ((b[8 + j] >> 4) | xh1) * d + m);
        }
    } else if constexpr (TYPE == 8) {                              // Q8_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 34;
        const float d = h2f(b);
        for (int j = 0; j < 32; ++j) put(out, j, (float) (int8_t) b[2 + j] * d);
    } else if constexpr (TYPE == 20) {                             // IQ4_NL
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 18;
        const float d = h2f(b);
        for (int j = 0; j < 16; ++j) {
            put(out, j, d * (float) kv_iq4nl[b[2 + j] & 0xf]);
            put(out, j + 16, d * (float) kv_iq4nl[b[2 + j] >> 4]);
        }
    } else if constexpr (TYPE == 11) {                             // Q3_K: hmask[32] qs[64] scales[12] d
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 110;
        const int gi = gi_in_row % 8, n = gi / 4, jj = gi % 4;
        const uint8_t* hm = b;
        const uint8_t* q = b + 32 + n * 32;
        const uint8_t* sc = b + 96;
        const float d_all = h2f(b + 108);
        uint32_t aux[4];
        memcpy(aux, sc, 12);
        const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu, tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
        const int shift = 2 * jj;
        const uint8_t m = (uint8_t) (1u << (n * 4 + jj));
        for (int t = 0; t < 32; ++t) {
            const int is = n * 8 + jj * 2 + (t >= 16 ? 1 : 0);
            const float dl = d_all * (float) (scales[is] - 32);
            put(out, t, dl * (float) ((int) ((q[t] >> shift) & 3) - ((hm[t] & m) ? 0 : 4)));
        }
    } else if constexpr (TYPE == 12) {                             // Q4_K: d dmin scales[12] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 144;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* q = b + 16 + 32 * j64;
        for (int l = 0; l < 32; ++l) put(out, l, d1 * (float) (hi ? (q[l] >> 4) : (q[l] & 0xF)) - m1);
    } else if constexpr (TYPE == 13) {                             // Q5_K: d dmin scales[12] qh[32] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 176;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* qh = b + 16;
        const uint8_t* ql = b + 48 + 32 * j64;
        const uint8_t u = (uint8_t) (1u << (2 * j64 + hi));
        for (int l = 0; l < 32; ++l) {
            const int nib = hi ? (ql[l] >> 4) : (ql[l] & 0xF);
            put(out, l, d1 * (float) (nib + ((qh[l] & u) ? 16 : 0)) - m1);
        }
    } else if constexpr (TYPE == 14) {                             // Q6_K: ql[128] qh[64] scales[16] d
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 210;
        const int gi = gi_in_row % 8, n = gi / 4, qu = gi % 4;
        const uint8_t* ql = b + 64 * n;
        const uint8_t* qh = b + 128 + 32 * n;
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192) + 8 * n;
        const float d = h2f(b + 208);
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            int q;
            if (qu == 0) q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            else if (qu == 1) q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            else if (qu == 2) q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
            else q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
            put(out, l, d * (float) sc[is + 2 * qu] * (float) (q - 32));
        }
    } else if constexpr (TYPE == 23) {                             // IQ4_XS: d scales_h scales_l[4] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 136;
        const int ib = gi_in_row % 8;
        const float d = h2f(b);
        const uint16_t scales_h = (uint16_t) (b[2] | (b[3] << 8));
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (float) (ls - 32);
        const uint8_t* qs = b + 8 + 16 * ib;
        for (int j = 0; j < 16; ++j) {
            put(out, j, dl * (float) kv_iq4nl[qs[j] & 0xf]);
            put(out, j + 16, dl * (float) kv_iq4nl[qs[j] >> 4]);
        }
    }
}

template <int TYPE, typename T>
__global__ void dequant_kernel(const uint8_t* __restrict__ blocks, int64_t row_bytes, int64_t row0, int64_t rows,
                               int64_t groups_per_row, int64_t ld, T* __restrict__ out) {
    const int64_t g = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= rows * groups_per_row) return;
    const int64_t r = g / groups_per_row, gi = g % groups_per_row;
    group32<TYPE>(blocks + (row0 + r) * row_bytes, (int) gi, out + r * ld + gi * 32);
}

// Q8_0 with four threads per block, 8 values each: the warp reads 8 consecutive blocks and writes 512 consecutive
// bytes (a thread per block wrote 2-byte values 64 bytes apart: ~110 GB/s, a quarter of a prompt projection's time)
template <typename T>
__global__ void dequant_q8_0_kernel(const uint8_t* __restrict__ blocks, int64_t row_bytes, int64_t row0, int64_t rows,
                                    int64_t groups_per_row, T* __restrict__ out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * groups_per_row * 4) return;
    const int64_t g = i >> 2, r = g / groups_per_row, gi = g % groups_per_row;
    const int piece = (int) (i & 3);
    const uint8_t* b = blocks + (row0 + r) * row_bytes + gi * 34;
    const float d = h2f(b);
    const uint16_t* q = (const uint16_t*) (b + 2 + 8 * piece);   // blocks are 2-byte aligned
    __align__(16) T v[8];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint16_t w = q[k];
        put(v, 2 * k, (float) (int8_t) (w & 0xFF) * d);
        put(v, 2 * k + 1, (float) (int8_t) (w >> 8) * d);
    }
    uint4* o = (uint4*) (out + g * 32 + 8 * piece);
#pragma unroll
    for (int k = 0; k < (int) (8 * sizeof(T) / 16); ++k) o[k] = ((const uint4*) v)[k];
}

// STRATA_DQ8=1 (opt-in, the same bits): the Q8_0 kernel's layout for Q4_0, Q5_0, Q4_K, Q5_K, Q6_K and IQ4_XS - four
// threads per 32-value group, 8 values each, so a wave writes 512 consecutive bytes of FP16 with one 16-byte store
// per thread (a thread per group writes 2-byte values 64 bytes apart: a Q5_K / Q6_K 10240 x 2560 projection takes
// 2.0 ms on a Radeon PRO V620, the GEMM it feeds 3.0 ms at 2048 tokens).  Every value is group32's expression.
template <int TYPE> constexpr bool kPiece8 = TYPE == 2 || TYPE == 6 || TYPE == 12 || TYPE == 13 || TYPE == 14 || TYPE == 23;
__device__ __forceinline__ uint64_t ld8(const uint8_t* p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}
template <int TYPE, typename T>
__device__ __forceinline__ void piece8(const uint8_t* row_blocks, int gi_in_row, int piece, T* v) {
    const int l0 = 8 * piece;
    if constexpr (TYPE == 2) {                                     // Q4_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 18;
        const float d = h2f(b);
        const uint64_t q = ld8(b + 2 + (l0 & 15));
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int byte = (int) ((q >> (8 * k)) & 0xFF);
            put(v, k, (float) ((piece < 2 ? (byte & 0x0F) : (byte >> 4)) - 8) * d);
        }
    } else if constexpr (TYPE == 6) {                              // Q5_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 22;
        const float d = h2f(b);
        const uint32_t qh = (uint32_t) b[2] | ((uint32_t) b[3] << 8) | ((uint32_t) b[4] << 16) | ((uint32_t) b[5] << 24);
        const uint64_t q = ld8(b + 6 + (l0 & 15));
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int j = (l0 & 15) + k;
            const int byte = (int) ((q >> (8 * k)) & 0xFF);
            if (piece < 2) {
                const int xh0 = ((qh >> j) << 4) & 0x10;
                put(v, k, (float) (((byte & 0x0F) | xh0) - 16) * d);
            } else {
                const int xh1 = (qh >> (j + 12)) & 0x10;
                put(v, k, (float) (((byte >> 4) | xh1) - 16) * d);
            }
        }
    } else if constexpr (TYPE == 12) {                             // Q4_K
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 144;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint64_t q = ld8(b + 16 + 32 * j64 + l0);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int byte = (int) ((q >> (8 * k)) & 0xFF);
            put(v, k, d1 * (float) (hi ? (byte >> 4) : (byte & 0xF)) - m1);
        }
    } else if constexpr (TYPE == 13) {                             // Q5_K
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 176;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint64_t qh = ld8(b + 16 + l0);
        const uint64_t ql = ld8(b + 48 + 32 * j64 + l0);
        const int u = 1 << (2 * j64 + hi);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int lo = (int) ((ql >> (8 * k)) & 0xFF), h = (int) ((qh >> (8 * k)) & 0xFF);
            const int nib = hi ? (lo >> 4) : (lo & 0xF);
            put(v, k, d1 * (float) (nib + ((h & u) ? 16 : 0)) - m1);
        }
    } else if constexpr (TYPE == 14) {                             // Q6_K
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 210;
        const int gi = gi_in_row % 8, n = gi / 4, qu = gi % 4;
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192) + 8 * n;
        const float d = h2f(b + 208);
        const uint64_t ql = ld8(b + 64 * n + l0 + ((qu & 1) ? 32 : 0));
        const uint64_t qh = ld8(b + 128 + 32 * n + l0);
        const int is = piece / 2;
        const float s = (float) sc[is + 2 * qu];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int lo = (int) ((ql >> (8 * k)) & 0xFF), h = (int) ((qh >> (8 * k)) & 0xFF);
            const int q = (qu < 2 ? (lo & 0xF) : (lo >> 4)) | (((h >> (2 * qu)) & 3) << 4);
            put(v, k, d * s * (float) (q - 32));
        }
    } else if constexpr (TYPE == 23) {                             // IQ4_XS
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 136;
        const int ib = gi_in_row % 8;
        const float d = h2f(b);
        const uint16_t scales_h = (uint16_t) (b[2] | (b[3] << 8));
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (float) (ls - 32);
        const uint64_t q = ld8(b + 8 + 16 * ib + (l0 & 15));
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int byte = (int) ((q >> (8 * k)) & 0xFF);
            put(v, k, dl * (float) kv_iq4nl[piece < 2 ? (byte & 0xf) : (byte >> 4)]);
        }
    }
}
template <int TYPE, typename T>
__global__ void dequant_piece8_kernel(const uint8_t* __restrict__ blocks, int64_t row_bytes, int64_t row0, int64_t rows,
                                      int64_t groups_per_row, int64_t ld, T* __restrict__ out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * groups_per_row * 4) return;
    const int64_t g = i >> 2, r = g / groups_per_row, gi = g % groups_per_row;
    const int piece = (int) (i & 3);
    __align__(16) T v[8];
    piece8<TYPE, T>(blocks + (row0 + r) * row_bytes, (int) gi, piece, v);
    uint4* o = (uint4*) (out + r * ld + gi * 32 + 8 * piece);
#pragma unroll
    for (int k = 0; k < (int) (8 * sizeof(T) / 16); ++k) o[k] = ((const uint4*) v)[k];
}
std::atomic<int> g_dq8{-1};   // -1: STRATA_DQ8 decides; 0 / 1: forced (dequant_set_piece8, the parity test)
bool dq8_on() {
    const int f = g_dq8.load(std::memory_order_relaxed);
    if (f >= 0) return f != 0;
    static const bool env = [] { const char* e = std::getenv("STRATA_DQ8"); return e && e[0] == '1'; }();
    return env;
}

bool geometry(int type, int& block_elems, int& block_bytes) {
    switch (type) {
    case 2: block_elems = 32; block_bytes = 18; return true;
    case 6: block_elems = 32; block_bytes = 22; return true;
    case 7: block_elems = 32; block_bytes = 24; return true;
    case 8: block_elems = 32; block_bytes = 34; return true;
    case 20: block_elems = 32; block_bytes = 18; return true;
    case 11: block_elems = 256; block_bytes = 110; return true;
    case 12: block_elems = 256; block_bytes = 144; return true;
    case 13: block_elems = 256; block_bytes = 176; return true;
    case 14: block_elems = 256; block_bytes = 210; return true;
    case 23: block_elems = 256; block_bytes = 136; return true;
    case 42: block_elems = 64; block_bytes = 18; return true;
    default: return false;
    }
}

template <typename T>
void launch(int type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, T* out, void* stream,
            int64_t ld = 0) {
    int be = 0, bb = 0;
    if (!geometry(type, be, bb) || cols % be != 0 || rows <= 0) {
        std::fprintf(stderr, "dequant: unsupported type %d or shape %lld x %lld\n", type, (long long) rows,
                     (long long) cols);
        std::exit(1);
    }
    const int64_t row_bytes = cols / be * bb, gpr = cols / 32, total = rows * gpr;
    if (ld <= 0) ld = cols;
    const unsigned grid = (unsigned) ((total + 255) / 256);
    const uint8_t* p = (const uint8_t*) blocks;
    cudaStream_t st = (cudaStream_t) stream;
    if (type == 8 && ld == cols && (reinterpret_cast<uintptr_t>(blocks) & 1) == 0 &&
        (reinterpret_cast<uintptr_t>(out) & 15) == 0) {   // contiguous rows only (the kernel writes out + g * 32)
        dequant_q8_0_kernel<T><<<(unsigned) ((4 * total + 255) / 256), 256, 0, st>>>(p, row_bytes, row0, rows, gpr, out);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { std::fprintf(stderr, "dequant launch: %s\n", cudaGetErrorString(e)); std::exit(1); }
        return;
    }
    if (dq8_on() && (reinterpret_cast<uintptr_t>(out) & 15) == 0 && (ld * (int64_t) sizeof(T)) % 16 == 0) {
        const unsigned grid4 = (unsigned) ((4 * total + 255) / 256);
        bool done = true;
#define STRATA_DQ8(TY) dequant_piece8_kernel<TY, T><<<grid4, 256, 0, st>>>(p, row_bytes, row0, rows, gpr, ld, out); break
        switch (type) {
        case 2: STRATA_DQ8(2);
        case 6: STRATA_DQ8(6);
        case 12: STRATA_DQ8(12);
        case 13: STRATA_DQ8(13);
        case 14: STRATA_DQ8(14);
        case 23: STRATA_DQ8(23);
        default: done = false;
        }
#undef STRATA_DQ8
        if (done) {
            const cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) { std::fprintf(stderr, "dequant launch: %s\n", cudaGetErrorString(e)); std::exit(1); }
            return;
        }
    }
#define STRATA_DQ(TY) dequant_kernel<TY, T><<<grid, 256, 0, st>>>(p, row_bytes, row0, rows, gpr, ld, out); break
    switch (type) {
    case 2: STRATA_DQ(2);
    case 6: STRATA_DQ(6);
    case 7: STRATA_DQ(7);
    case 8: STRATA_DQ(8);
    case 11: STRATA_DQ(11);
    case 12: STRATA_DQ(12);
    case 13: STRATA_DQ(13);
    case 14: STRATA_DQ(14);
    case 20: STRATA_DQ(20);
    case 23: STRATA_DQ(23);
    case 42: STRATA_DQ(42);
    }
#undef STRATA_DQ
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "dequant launch: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

}  // namespace

void dequant_set_piece8(int mode) { g_dq8.store(mode < 0 ? -1 : (mode != 0 ? 1 : 0), std::memory_order_relaxed); }

bool dequant_bf16_supported(int ggml_type) noexcept {
    int a, b;
    return geometry(ggml_type, a, b);
}

namespace {
// plan v0.3 P6: the i-quant formats (llama.cpp's dequantizers, iq_kernels.cu)
bool iq_only(int t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29; }
}  // namespace

void dequant_bf16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                  void* stream) {
    launch<uint16_t>(ggml_type, blocks, row0, rows, cols, out, stream);
}

void dequant_f16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                 void* stream) {
    if (iq_only(ggml_type)) {
        iq_dequant_f16(ggml_type, (const uint8_t*) blocks + (size_t) row0 * iq_row_bytes(ggml_type, cols), rows * cols,
                       out, stream);
        return;
    }
    launch<H16>(ggml_type, blocks, row0, rows, cols, reinterpret_cast<H16*>(out), stream);
}

bool dequant_f16_ld(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, int64_t ld,
                    uint16_t* out, void* stream) {
    int be = 0, bb = 0;
    if (iq_only(ggml_type) || !geometry(ggml_type, be, bb) || cols % be != 0 || rows <= 0 || ld < cols || ld % 8 != 0)
        return false;
    launch<H16>(ggml_type, blocks, row0, rows, cols, reinterpret_cast<H16*>(out), stream, ld);
    return true;
}

void dequant_f32(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, float* out, void* stream) {
    if (iq_only(ggml_type)) {
        iq_dequant_f32(ggml_type, (const uint8_t*) blocks + (size_t) row0 * iq_row_bytes(ggml_type, cols), rows * cols,
                       out, stream);
        return;
    }
    launch<float>(ggml_type, blocks, row0, rows, cols, out, stream);
}

}  // namespace strata::kernels
