// tests/hip/head_lab.cpp - the decode window's output head in isolation: the Q6_K vocabulary projection for 1-5
// columns, the greedy pick, the non-finite check and the sampled chain, timed on one stream (event to event, the
// median of 9 batches of 20 calls).  Synthetic weights of the real shape (248320 x 2560, 497 MiB).
//   head_lab [n_vocab] [device]
#include "strata/kernels/head_wide.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

using namespace strata::kernels;

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

static double timed(cudaStream_t s, const std::function<void()>& f, int reps = 20, int batches = 9) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<float> v;
    f(); cudaStreamSynchronize(s);
    for (int b = 0; b < batches; ++b) {
        cudaEventRecord(e0, s);
        for (int i = 0; i < reps; ++i) f();
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        v.push_back(ms * 1000.0f / (float) reps);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    const int NV = argc > 1 ? std::atoi(argv[1]) : 248320, N = 2560, MAXT = 8;
    if (argc > 2) ck(cudaSetDevice(std::atoi(argv[2])), "device");
    cudaStream_t s; ck(cudaStreamCreate(&s), "stream");
    const size_t wb = native_mmvq_weight_bytes(14, N, NV);
    std::vector<uint8_t> hw(wb);
    std::mt19937_64 rng(42);
    for (size_t i = 0; i + 8 <= wb; i += 8) { const uint64_t r = rng(); std::memcpy(&hw[i], &r, 8); }
    for (size_t b = 0; b < wb / 210; ++b) {   // block_q6_K: ql[128] qh[64] scales[16] d(f16): a small finite d
        const uint16_t d = (uint16_t) (0x2800 + (rng() & 0x3ff));
        std::memcpy(&hw[b * 210 + 208], &d, 2);
    }
    void *w, *xq; float *x, *y; int *out; uint32_t* nf;
    ck(cudaMalloc(&w, wb), "w"); ck(cudaMemcpy(w, hw.data(), wb, cudaMemcpyHostToDevice), "up");
    ck(cudaMalloc(&xq, native_q8_1_bytes(N, MAXT)), "xq");
    ck(cudaMalloc((void**) &x, (size_t) MAXT * N * 4), "x");
    ck(cudaMalloc((void**) &y, (size_t) MAXT * NV * 4), "y");
    ck(cudaMalloc((void**) &out, MAXT * 4), "out"); ck(cudaMalloc((void**) &nf, MAXT * 4), "nf");
    std::vector<float> hx((size_t) MAXT * N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : hx) v = nd(rng);
    ck(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x up");
    std::printf("head lab: %d x %d Q6_K (%.1f MiB)\n", NV, N, wb / 1048576.0);
    for (int T = 1; T <= 5; ++T) {
        const double q = timed(s, [&] { native_quantize_q8_1(x, xq, N, T, s); });
        const double m = timed(s, [&] { native_mmvq(14, w, xq, y, N, NV, T, s); });
        SamplerParams g; g.greedy = true; g.temperature = 0.0f;
        const double a = timed(s, [&] { sample_tokens(y, T, NV, nullptr, 0, g, out, s); });
        const double f = timed(s, [&] { logits_nonfinite_rows(y, T, NV, nf, s); });
        SamplerParams sp; sp.seed = 7; sp.counter = 100; sp.gumbel = true;
        const double sm = timed(s, [&] { sample_tokens(y, T, NV, nullptr, 0, sp, out, s); });
        void* scratch = nullptr;
        static void* sc = nullptr;
        if (!sc) ck(cudaMalloc(&sc, head_wide_scratch_bytes()), "scratch");
        scratch = sc;
        for (int var : {2, 3}) {
            head_wide_set_variant(var);
            std::printf("T %d: wide head v%d + pick + flag:", T, var);
            for (int grid : {2304, 4096, 8192}) {
                head_wide_set_grid(grid);
                const double hwd = timed(s, [&] { head_wide(w, xq, y, N, NV, T, scratch, out, nf, s); });
                std::printf(" grid %d %7.1f us |", grid, hwd);
            }
            std::printf("\n");
        }
        head_wide_set_variant(0);
        std::printf("T %d: quantize %6.1f us | head %7.1f us (%5.1f GB/s) | greedy %6.1f us | nonfinite %6.1f us | "
                    "sampled chain %6.1f us\n", T, q, m, wb / m / 1e3, a, f, sm);
    }
    return 0;
}
