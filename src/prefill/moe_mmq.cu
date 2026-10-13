// src/prefill/moe_mmq.cu - see include/strata/prefill/moe_mmq.hpp.  llama.cpp's MMQ (ggml-cuda, MIT) is compiled
// from the pinned llama.cpp checkout the build already takes ggml from; src/prefill/ggml_cuda_host.cu supplies the
// few host symbols of ggml-cuda.cu it references.
#include "strata/prefill/moe_mmq.hpp"
#if defined(__HIPCC__)
#include "strata/prefill/moe_q4.hpp"   // Q4_0 / Q4_1 experts: Strata's own RDNA2 kernel, reached through this contract
#endif

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

#include <cstdio>
#include <cstdlib>

namespace strata::prefill::mmq {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill mmq: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int64_t pad512(int64_t n) { return (n + 511) / 512 * 512; }

__global__ void copy16_kernel(const uint4* __restrict__ a, int64_t na, const uint4* __restrict__ b, int64_t nb,
                              uint4* __restrict__ ab_dst, const uint4* __restrict__ c, int64_t nc, uint4* __restrict__ c_dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < na + nb + nc) c_dst[i - na - nb] = c[i - na - nb];
}
// copy16_kernel for an MMQ group: blockIdx.y is the expert (first + y)
struct GroupArgs {
    const uint8_t* blob[kGatherGroupMax];
    int64_t up_off, down_off, gu_stride, d_stride;   // in uint4
};
__global__ void copy16_group_kernel(GroupArgs ga, int first, int64_t na, int64_t nc, uint4* __restrict__ gu_dst,
                                    uint4* __restrict__ d_dst) {
    const int q = first + (int) blockIdx.y;
    const uint4* src = (const uint4*) ga.blob[q];
    uint4* ab = gu_dst + (int64_t) q * ga.gu_stride;
    uint4* cd = d_dst + (int64_t) q * ga.d_stride;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < na) ab[i] = src[i];
    else if (i < 2 * na) ab[i] = src[ga.up_off + (i - na)];
    else if (i < 2 * na + nc) cd[i - 2 * na] = src[ga.down_off + (i - 2 * na)];
}
__global__ void copy1_kernel(const uint8_t* __restrict__ a, int64_t na, const uint8_t* __restrict__ b, int64_t nb,
                             uint8_t* __restrict__ ab_dst, const uint8_t* __restrict__ c, int64_t nc, uint8_t* __restrict__ c_dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < na + nb + nc) c_dst[i - na - nb] = c[i - na - nb];
}

// Strata blob: gate/up codes [1280][640 B], down codes [2560][160 B], gate/up scales [1280][40] f16, down scales
// [2560][10] f16 (the layout of prefill/kernels.cu's blob_dequant_kernel).  A GGUF Q2_0 block is {f16 d; 16 code
// bytes} with the same 2-bit codes in the same order, so a block is a scale and a 16-byte run of codes.
__global__ void strata_q2_kernel(const uint8_t* __restrict__ blob, uint16_t* __restrict__ gu, uint16_t* __restrict__ dn) {
    constexpr size_t O_D_CODES = (size_t) 1280 * 640, O_GU_SC = O_D_CODES + (size_t) 2560 * 160,
                     O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one GGUF block
    const int64_t n_gu = 1280LL * 40, n_d = 2560LL * 10;
    const uint8_t* codes;
    const uint16_t* scale;
    uint16_t* out;
    if (i < n_gu) {
        const int64_t row = i / 40, b = i % 40;
        codes = blob + row * 640 + b * 16;
        scale = (const uint16_t*) (blob + O_GU_SC) + row * 40 + b;
        out = gu + i * 9;
    } else if (i < n_gu + n_d) {
        const int64_t j = i - n_gu, row = j / 10, b = j % 10;
        codes = blob + O_D_CODES + row * 160 + b * 16;
        scale = (const uint16_t*) (blob + O_D_SC) + row * 10 + b;
        out = dn + j * 9;
    } else {
        return;
    }
    const uint4 q = *(const uint4*) codes;
    const uint16_t* qh = (const uint16_t*) &q;
    out[0] = *scale;
#pragma unroll
    for (int k = 0; k < 8; ++k) out[1 + k] = qh[k];
}

__global__ void swiglu_kernel(const float* __restrict__ gu, float* __restrict__ h, int64_t rows, int64_t n_ff,
                              bool interleaved) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, k = i % n_ff;
    const float* row = gu + r * 2 * n_ff;
    const float g = interleaved ? row[2 * k] : row[k], u = interleaved ? row[2 * k + 1] : row[n_ff + k];
    h[i] = g / (1.0f + __expf(-g)) * u;
}

// swiglu_kernel (n_ff 640), then quantize_mmq_q8_1<D4>'s rows of its output, in one pass: a block per row (128
// threads, 4 values each, blockIdx.y the row's 512-value half), H never stored.  Each value is swiglu_kernel's
// expression and each 32-value block quantize_mmq_q8_1's (the same unit and flags: the same bits).
template <bool IL>
__global__ void __launch_bounds__(128) swiglu_quant_kernel(const float* __restrict__ gu, block_q8_1_mmq* __restrict__ y,
                                                           int64_t rows) {
    constexpr int64_t n_ff = 640, ne0 = 1024;   // the padded row: pad512(640)
    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= ne0) return;
    const int64_t r = blockIdx.x;
    float4 xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (i0 < n_ff) {
        const float* row = gu + r * 2 * n_ff;
        float4 g4, u4;
        if constexpr (IL) {   // gate 2k, up 2k + 1
            const float4 a = *reinterpret_cast<const float4*>(row + 2 * i0), b = *reinterpret_cast<const float4*>(row + 2 * i0 + 4);
            g4 = make_float4(a.x, a.z, b.x, b.z);
            u4 = make_float4(a.y, a.w, b.y, b.w);
        } else {
            g4 = *reinterpret_cast<const float4*>(row + i0);
            u4 = *reinterpret_cast<const float4*>(row + n_ff + i0);
        }
        xi.x = g4.x / (1.0f + __expf(-g4.x)) * u4.x;
        xi.y = g4.y / (1.0f + __expf(-g4.y)) * u4.y;
        xi.z = g4.z / (1.0f + __expf(-g4.z)) * u4.z;
        xi.w = g4.w / (1.0f + __expf(-g4.w)) * u4.w;
    }
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = 32 / 8; offset > 0; offset >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    const float d_inv = 127.0f / amax;
    char4 c;
    c.x = roundf(xi.x * d_inv);
    c.y = roundf(xi.y * d_inv);
    c.z = roundf(xi.z * d_inv);
    c.w = roundf(xi.w * d_inv);
    const float d = 1.0f / d_inv;
    const int64_t k_block = i0 / QK8_1_MMQ, iqs = i0 % QK8_1_MMQ;
    block_q8_1_mmq* yb = y + k_block * rows + r;
    reinterpret_cast<char4*>(yb->qs)[iqs / 4] = c;
    if (iqs % 32 == 0) yb->d4[iqs / 32] = d;
}

__global__ void iota_kernel(int32_t* dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (int32_t) i;
}

#if defined(__HIPCC__)   // #820: only the HIP dense-MMQ path (Gemm::native_mmq) uses these two
__global__ void f16_to_f32_kernel(const uint16_t* __restrict__ x, float* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(__ushort_as_half(x[i]));
}

__global__ void set_bounds_kernel(int32_t* d, int32_t rows) {
    d[0] = 0;
    d[1] = rows;
}

#endif

unsigned blocks(int64_t n) { return (unsigned) ((n + 255) / 256); }

#if defined(__HIPCC__)
// ---- Q5_1 down products on a 256 x 32 tile without shared memory (STRATA_PF_Q51=1, opt-in, gfx103x) -------------------
// Unsloth's UD-Q4_K_XL keeps the down projection of 43 of its 48 layers in Q5_1.  This is the tile of moe_q4.cu's
// q4_sload_kernel / nl_sload_kernel for that format, here because it gives llama.cpp's MMQ kernel output bit for bit by
// calling MMQ's own arithmetic: a thread owns one weight row and decodes 128 of its values (4 blocks of {half2 dm;
// 4 bytes qh; 16 bytes qs}) into 32 registers exactly as ggml_cuda_mmq_load_tiles_q5_1 does into shared memory, the
// tile's token rows come by scalar loads from the block_q8_1_mmq rows quantize() wrote (DS4: {half d, half s} per 32
// values), and each 32 values add vec_dot_q8_1_q8_1_impl<8> to the output's sum in K order, as
// ggml_cuda_mmq_vec_dot_q8_1_q8_1_dp4a does.  What MMQ does and this does not: its K steps of 256 values add four
// more blocks past a 640-column row, times zero activations (d and s zero): terms of +-0, which can only turn an
// output of exactly -0 into +0.  It never reads past a matrix, so the product can read an expert in its cache slot
// (STRATA_PF_DIRECT=1) and the down gather goes too.
struct ArgsQ51 {
    const char* w[kGatherGroupMax];
    const int* y;            // block_q8_1_mmq rows, DS4
    const int32_t* bounds;   // n + 1
    const int32_t* ids;
    float* dst;
    int64_t ld_dst, ncols_y;
    int w_rows, w_cols;
};

__device__ __forceinline__ int q51_ld32(const void* p) {
    int v;
    __builtin_memcpy(&v, p, 4);
    return v;
}

__global__ void __launch_bounds__(256) q51_sload_kernel(ArgsQ51 a) {
    constexpr int TJ = 32;
    constexpr int CW = sizeof(block_q8_1_mmq) / sizeof(int);   // 36
    const int t = threadIdx.x;
    const int e = blockIdx.z;
    const int r0 = a.bounds[e], r1 = a.bounds[e + 1];
    const int j0 = r0 + (int) blockIdx.y * TJ;
    if (j0 >= r1) return;
    const int nj = min(TJ, r1 - j0);
    const int i = (int) blockIdx.x * 256 + t;
    const int chunks = a.w_cols / QK8_1_MMQ;
    const block_q5_1* wr = (const block_q5_1*) a.w[e] + (size_t) (i < a.w_rows ? i : 0) * (size_t) (a.w_cols / QK5_1);
    float acc[TJ];
#pragma unroll
    for (int q = 0; q < TJ; ++q) acc[q] = 0.0f;
    for (int c = 0; c < chunks; ++c) {
        int w[32];
        half2 dm[4];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const block_q5_1* bx = wr + c * 4 + b;
            dm[b] = bx->dm;
            const int qh0 = q51_ld32(bx->qh);
#pragma unroll
            for (int k = 0; k < 4; ++k) {   // ggml_cuda_mmq_load_tiles_q5_1's two words
                const int ql = q51_ld32(bx->qs + 4 * k);
                const int qh = qh0 >> (4 * k);
                int qs0 = (ql >> 0) & 0x0F0F0F0F;
                qs0 |= (qh << 4) & 0x00000010;
                qs0 |= (qh << 11) & 0x00001000;
                qs0 |= (qh << 18) & 0x00100000;
                qs0 |= (qh << 25) & 0x10000000;
                int qs1 = (ql >> 4) & 0x0F0F0F0F;
                qs1 |= (qh >> 12) & 0x00000010;
                qs1 |= (qh >> 5) & 0x00001000;
                qs1 |= (qh << 2) & 0x00100000;
                qs1 |= (qh << 9) & 0x10000000;
                w[b * 8 + k] = qs0;
                w[b * 8 + 4 + k] = qs1;
            }
        }
        const int* aj = a.y + ((size_t) c * (size_t) a.ncols_y + (size_t) j0) * CW;
#pragma unroll
        for (int jj = 0; jj < TJ; ++jj, aj += CW) {
            if (jj < nj) {
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    int u[8];
#pragma unroll
                    for (int k = 0; k < 8; ++k) u[k] = aj[4 + b * 8 + k];
                    half2 ds;
                    __builtin_memcpy(&ds, aj + b, 4);
                    acc[jj] += vec_dot_q8_1_q8_1_impl<QR5_1 * VDR_Q5_1_Q8_1_MMQ>(&w[b * 8], u, dm[b], ds);
                }
            }
        }
    }
    if (i >= a.w_rows) return;
#pragma unroll
    for (int jj = 0; jj < TJ; ++jj) {
        if (jj < nj) a.dst[(int64_t) a.ids[j0 + jj] * a.ld_dst + i] = acc[jj];
    }
}

bool q51_on() {
    static const bool env = [] { const char* v = std::getenv("STRATA_PF_Q51"); return v != nullptr && v[0] == '1'; }();
    return env && q4::gfx103x();
}
// the down shape only (K 640), as moe_q4.cu's IQ4_NL tile: at K 2560 MMQ's own tile is the faster one
bool q51_fits(int t, int64_t w_cols) { return t == (int) GGML_TYPE_Q5_1 && w_cols == 640 && q51_on(); }

void q51_run(const Product& p, cudaStream_t s) {
    if (p.n > kGatherGroupMax || p.ids == nullptr || ((uintptr_t) p.xq & 3) != 0) {
        std::fprintf(stderr, "prefill q5_1 experts: %d experts, ids %p: not covered\n", p.n, (const void*) p.ids);
        std::exit(1);
    }
    static bool said = false;
    if (!said) {
        said = true;
        std::fprintf(stderr, "strata: prompt kernels: Q5_1 down experts on the 256 x 32 tiles (STRATA_PF_Q51=1: MMQ's "
                             "output bits)\n");
    }
    ArgsQ51 a{};
    for (int e = 0; e < p.n; ++e) a.w[e] = p.each ? (const char*) p.w0[e] : (const char*) p.w + (size_t) e * p.expert_bytes;
    a.y = (const int*) p.xq;
    a.bounds = p.bounds;
    a.ids = p.ids;
    a.dst = p.dst;
    a.ld_dst = p.ld_dst;
    a.ncols_y = p.total_rows;
    a.w_rows = (int) p.w_rows;
    a.w_cols = (int) p.w_cols;
    const dim3 g((unsigned) ((p.w_rows + 255) / 256), (unsigned) ((p.max_rows + 31) / 32), (unsigned) p.n);
    q51_sload_kernel<<<g, 256, 0, s>>>(a);
    ck(cudaGetLastError(), "product (q5_1)");
}
#endif

}  // namespace

bool built() { return true; }

bool direct_blobs(int t, bool down) {
#if defined(__HIPCC__)
    if (q4::covers(t)) return q4::direct();   // STRATA_PF_DIRECT=1 on gfx103x, Strata's own kernel
    static const bool env = [] { const char* v = std::getenv("STRATA_PF_DIRECT"); return v != nullptr && v[0] == '1'; }();
    if (!env) return false;
    if (q4::nl_fits(t, down ? 640 : 2560)) return true;   // STRATA_PF_NL=1: Strata's IQ4_NL tile (down)
    if (q51_fits(t, down ? 640 : 2560)) return true;       // STRATA_PF_Q51=1: the Q5_1 tile (down)
#if defined(MMQ_X_EACH_MAX)             // llama.cpp with patch 0003
    static_assert(MMQ_X_EACH_MAX >= kGatherGroupMax, "a group's pointers fit the kernel's table");
    return !down && supported(t);
#else
    return false;
#endif
#else
    (void) t; (void) down;
    return false;
#endif
}

bool supported(int t) {
#if defined(__HIPCC__)
    if (q4::covers(t)) return q4::enabled();
#endif
    switch ((ggml_type) t) {
        case GGML_TYPE_Q2_0:
#ifdef STRATA_ORCA_Q4KS_MMQ
        case GGML_TYPE_Q5_0:   // #296 (Q4_K and Q5_1: STRATA_MMQ_KQUANTS)
#endif
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q8_0:   // the draft layer's dense matrices (E-9)
#ifdef STRATA_MMQ_KQUANTS
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q5_1:   // Unsloth's UD-Q4_K_XL experts (CUDA)
#if defined(__HIPCC__) || defined(STRATA_Q6K_EXPERTS)
        case GGML_TYPE_Q6_K:   // HIP: the dense GGUF projections (STRATA_DENSE_MMQ); CUDA: only the opt-in -DSTRATA_Q6K_EXPERTS=ON build
#endif
#endif
            return true;
        default:
            return false;
    }
}

bool fits(int t, int64_t w_rows) {
#if defined(__HIPCC__)
    if (q4::covers(t)) return q4::fits(t, w_rows);   // no llama.cpp tile behind it: nothing to look up
#endif
    if (!supported(t)) return false;
    // mul_mat_q_case's choice: the "fallback" configs when the rows are not a multiple of 128; then
    // mul_mat_q_switch_J's loop - a tile size whose config exists for this card and fits its shared memory
    const bool fallback = w_rows % 128 != 0;
    const ggml_cuda_device_info& info = ggml_cuda_info();
    for (int id = 0; id < info.device_count; ++id) {
        const int cc = info.devices[id].cc;
        const size_t smpbo = info.devices[id].smpbo;
        bool any = false;
        for (int J = 8; J <= 128 && !any; J += 8) {
            const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config((ggml_type) t, J, fallback, cc);
            any = c.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(c, cc) <= smpbo;
        }
        if (!any) {
            static bool said[GGML_TYPE_COUNT] = {};
            if (t >= 0 && t < GGML_TYPE_COUNT && !said[t]) {
                said[t] = true;
                std::fprintf(stderr, "strata: prompt kernels: llama.cpp's MMQ has no tile for %s (%lld rows) on GPU %d "
                                     "(cc %d, %zu bytes of shared memory per block): that product takes the non-MMQ path "
                                     "(#420)\n", ggml_type_name((ggml_type) t), (long long) w_rows, id, cc, smpbo);
            }
            return false;
        }
    }
    return true;
}

size_t matrix_bytes(int t, int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (cols / ggml_blck_size((ggml_type) t)) * ggml_type_size((ggml_type) t);
}

size_t q8_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) pad512(cols) * sizeof(block_q8_1_mmq) / (4 * QK8_1) + 128 * sizeof(block_q8_1_mmq);
}

void quantize(const float* x, const int32_t* ids, void* xq, int t, int64_t cols, int64_t ld, int64_t rows, void* stream) {
    if (rows <= 0) return;
#if defined(__HIPCC__)
    if (q4::covers(t)) {   // its own chunks ({float d[4]; s8 q[128]}), within q8_bytes
        q4::quantize(x, ids, xq, cols, ld, rows, stream);
        return;
    }
#endif
    quantize_mmq_q8_1_cuda(x, ids, xq, (ggml_type) t, cols, ld, rows * ld, rows * ld, pad512(cols), rows, 1, 1,
                           (cudaStream_t) stream);
    ck(cudaGetLastError(), "quantize");
}

void quantize_scatter(const float* x, const int32_t* slot, const int32_t* src, void* xq, int t, int64_t cols, int64_t ld,
                      int64_t tokens, int k_used, void* stream) {
    if (tokens <= 0 || k_used <= 0) return;
    // Opt-in in this tree (upstream: on, STRATA_QUANT_GATHER=1 restores the gather): STRATA_QUANT_GATHER=0 turns the
    // scatter on; unset, quantize() over `src` as before.
    static const bool gather = [] { const char* e = std::getenv("STRATA_QUANT_GATHER"); return e == nullptr || std::atoi(e) != 0; }();
    if (gather) {
        quantize(x, src, xq, t, cols, ld, tokens * k_used, stream);
        return;
    }
#if defined(__HIPCC__)
    // Q4_0 / Q4_1 on HIP: quantize() writes the RDNA2 int8 kernel's own chunks (q4::quantize), not llama.cpp's
    // q8_1 rows, and there is no scatter for those.  The gather, or that kernel would read the wrong layout.
    if (q4::covers(t)) {
        quantize(x, src, xq, t, cols, ld, tokens * k_used, stream);
        return;
    }
#endif
    quantize_scatter_mmq_q8_1_cuda(x, slot, xq, (ggml_type) t, cols, ld, pad512(cols), tokens, tokens * k_used, k_used,
                                   (cudaStream_t) stream);
    ck(cudaGetLastError(), "quantize_scatter");
}

Context::Context() {
    int dev = 0;
    cudaGetDevice(&dev);
    ctx_ = new ggml_backend_cuda_context(dev);
}
Context::~Context() { delete (ggml_backend_cuda_context*) ctx_; }

void Context::run(const Product& p, void* stream) {
    if (p.n <= 0 || p.max_rows <= 0) return;
#if defined(__HIPCC__)
    if (q4::covers(p.type)) {
        q4::run(p, stream);
        return;
    }
#endif
#if defined(__HIPCC__)
    if (q4::nl_fits(p.type, p.w_cols)) {   // STRATA_PF_NL=1: IQ4_NL on Strata's tiles, MMQ's activations and bits
        q4::nl_run(p, stream);
        return;
    }
    if (q51_fits(p.type, p.w_cols)) {      // STRATA_PF_Q51=1: Q5_1 down likewise
        q51_run(p, (cudaStream_t) stream);
        return;
    }
#endif
    const ggml_type t = (ggml_type) p.type;
    const int64_t qk = ggml_blck_size(t), bpr = p.w_cols / qk;
    mmq_args a = {(const char*) p.w, t, (const int*) p.xq, p.ids, p.bounds, p.dst, nullptr,
                        p.w_cols, p.w_rows, p.total_rows, bpr, p.total_rows, p.ld_dst,
                        p.n, p.n, (int64_t) (p.expert_bytes / ggml_type_size(t)), 0, 0,
                        1, 1, 0, 0, 0,
                        p.max_rows, p.max_rows};
    if (p.each) {
#if defined(MMQ_X_EACH_MAX)
        // each expert's matrix where it lies: its rows must be one run (gate rows, then up rows right behind them)
        const size_t split_bytes = (size_t) p.w_split * (size_t) bpr * ggml_type_size(t);
        if (p.n > MMQ_X_EACH_MAX) {
            std::fprintf(stderr, "prefill mmq: %d experts in a product over their own blobs (at most %d)\n", p.n, MMQ_X_EACH_MAX);
            std::exit(1);
        }
        for (int e = 0; e < p.n; ++e) {
            if ((const char*) p.w1[e] != (const char*) p.w0[e] + split_bytes && p.w_split < p.w_rows) {
                std::fprintf(stderr, "prefill mmq: an expert's gate and up rows are not one run\n");
                std::exit(1);
            }
            a.x_each.p[e] = (const char*) p.w0[e];
        }
        a.x_each.n = p.n;
        static bool said = false;
        if (!said) {
            said = true;
            std::fprintf(stderr, "strata: prompt kernels: MMQ gate/up products read the experts from their own blobs, no "
                                 "group copy (STRATA_PF_DIRECT=1)\n");
        }
#else
        std::fprintf(stderr, "prefill mmq: this build's llama.cpp has no per-channel weight pointers (patch 0003)\n");
        std::exit(1);
#endif
    }
    auto& ctx = *(ggml_backend_cuda_context*) ctx_;
    const cudaStream_t s = (cudaStream_t) stream;
    switch (t) {
#ifdef STRATA_ORCA_Q4KS_MMQ
        case GGML_TYPE_Q5_0: mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, a, s); break;
#endif
        case GGML_TYPE_Q2_0: mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, a, s); break;
        case GGML_TYPE_IQ2_XXS: mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, a, s); break;
        case GGML_TYPE_IQ2_XS: mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, a, s); break;
        case GGML_TYPE_IQ2_S: mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, a, s); break;
        case GGML_TYPE_IQ3_XXS: mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, a, s); break;
        case GGML_TYPE_IQ3_S: mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, a, s); break;
        case GGML_TYPE_IQ4_NL: mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, a, s); break;
        case GGML_TYPE_IQ4_XS: mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, a, s); break;
        case GGML_TYPE_Q8_0: mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, a, s); break;
#ifdef STRATA_MMQ_KQUANTS
        case GGML_TYPE_Q4_K: mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, a, s); break;
        case GGML_TYPE_Q5_K: mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, a, s); break;
#if defined(__HIPCC__) || defined(STRATA_Q6K_EXPERTS)
        case GGML_TYPE_Q6_K: mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, a, s); break;
#endif
        case GGML_TYPE_Q5_1: mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, a, s); break;
#endif
        default:
            std::fprintf(stderr, "prefill mmq: type %d is not covered\n", (int) t);
            std::exit(1);
    }
    ck(cudaGetLastError(), "mul_mat_q");
}

void gather_native(const void* gate, const void* up, size_t gu_half_bytes, const void* down, size_t d_bytes,
                   void* gu_dst, void* d_dst, void* stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    const bool a16 = ((uintptr_t) gate | (uintptr_t) up | (uintptr_t) down | (uintptr_t) gu_dst | (uintptr_t) d_dst |
                      gu_half_bytes | d_bytes) % 16 == 0;
    if (a16) {
        const int64_t na = (int64_t) gu_half_bytes / 16, nc = (int64_t) d_bytes / 16;
        copy16_kernel<<<blocks(2 * na + nc), 256, 0, s>>>((const uint4*) gate, na, (const uint4*) up, na, (uint4*) gu_dst,
                                                          (const uint4*) down, nc, (uint4*) d_dst);
    } else {
        const int64_t na = (int64_t) gu_half_bytes, nc = (int64_t) d_bytes;
        copy1_kernel<<<blocks(2 * na + nc), 256, 0, s>>>((const uint8_t*) gate, na, (const uint8_t*) up, na,
                                                         (uint8_t*) gu_dst, (const uint8_t*) down, nc, (uint8_t*) d_dst);
    }
    ck(cudaGetLastError(), "gather_native");
}

bool gather_native_group(const GatherGroup& g, size_t up_off, size_t gu_half_bytes, size_t down_off, size_t d_bytes,
                         void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride, void* stream) {
    if (g.first < 0 || g.n <= g.first || g.n > kGatherGroupMax) return false;
    uintptr_t a = (uintptr_t) gu_dst | (uintptr_t) d_dst | up_off | gu_half_bytes | down_off | d_bytes | gu_stride | d_stride;
    for (int q = g.first; q < g.n; ++q) a |= (uintptr_t) g.blob[q];
    if (a % 16 != 0) return false;
    GroupArgs ga{};
    for (int q = g.first; q < g.n; ++q) ga.blob[q] = g.blob[q];
    ga.up_off = (int64_t) up_off / 16;
    ga.down_off = (int64_t) down_off / 16;
    ga.gu_stride = (int64_t) gu_stride / 16;
    ga.d_stride = (int64_t) d_stride / 16;
    const int64_t na = (int64_t) gu_half_bytes / 16, nc = (int64_t) d_bytes / 16;
    copy16_group_kernel<<<dim3(blocks(2 * na + nc), (unsigned) (g.n - g.first)), 256, 0, (cudaStream_t) stream>>>(
        ga, g.first, na, nc, (uint4*) gu_dst, (uint4*) d_dst);
    ck(cudaGetLastError(), "gather_native_group");
    return true;
}

void gather_strata_q2(const uint8_t* blob, void* gu_dst, void* d_dst, void* stream) {
    strata_q2_kernel<<<blocks(1280LL * 40 + 2560LL * 10), 256, 0, (cudaStream_t) stream>>>(blob, (uint16_t*) gu_dst,
                                                                                         (uint16_t*) d_dst);
    ck(cudaGetLastError(), "gather_strata_q2");
}

void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream) {
    if (rows <= 0) return;
    swiglu_kernel<<<blocks(rows * n_ff), 256, 0, (cudaStream_t) stream>>>(gu, h, rows, n_ff, interleaved);
    ck(cudaGetLastError(), "swiglu");
}

bool swiglu_quant_ok(int t) {
#if defined(__HIPCC__)
    // Q4_0 / Q4_1 on HIP: quantize() writes the RDNA2 int8 kernel's own chunks (q4::quantize), not q8_1 D4 rows
    if (q4::covers(t)) return false;
#endif
    // the q8_1 D4 rows quantize() writes for a down product's H (K 640)
    return supported(t) && mmq_get_q8_1_ds_layout((ggml_type) t) == MMQ_Q8_1_DS_LAYOUT_D4;
}

void swiglu_quant(const float* gu, void* hq, int64_t rows, bool interleaved, void* stream) {
    if (rows <= 0) return;
    const dim3 grid((unsigned) rows, 2);
    if (interleaved) swiglu_quant_kernel<true><<<grid, 128, 0, (cudaStream_t) stream>>>(gu, (block_q8_1_mmq*) hq, rows);
    else swiglu_quant_kernel<false><<<grid, 128, 0, (cudaStream_t) stream>>>(gu, (block_q8_1_mmq*) hq, rows);
    ck(cudaGetLastError(), "swiglu_quant");
}

void iota(int32_t* dst, int64_t n, void* stream) {
    if (n <= 0) return;
    iota_kernel<<<blocks(n), 256, 0, (cudaStream_t) stream>>>(dst, n);
    ck(cudaGetLastError(), "iota");
}

#if defined(__HIPCC__)
void f16_to_f32(const uint16_t* x, float* y, int64_t n, void* stream) {
    if (n <= 0) return;
    f16_to_f32_kernel<<<blocks(n), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    ck(cudaGetLastError(), "f16_to_f32");
}

void set_bounds(int32_t* dst, int32_t rows, void* stream) {
    set_bounds_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(dst, rows);
    ck(cudaGetLastError(), "set_bounds");
}
#endif

}  // namespace strata::prefill::mmq
