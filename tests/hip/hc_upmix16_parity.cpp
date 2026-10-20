// tests/hip/hc_upmix16_parity.cpp - STRATA_HC_UPMIX16: the hyper-connection up projection with the mix as its epilogue
// (gr_upmix16, v_dot2) against the FP16 path's default pair (Gemm::bf16 with STRATA_HC_GATED16's FP16 output - the
// weight converted to FP16, rocBLAS FP16 in / FP16 out - then gr_mix_r reading the halves), on synthetic inputs of the
// model's shapes (lo16 T x 320 FP16, w_up 10240 x 320 BF16).  Both round each sum to FP16 once and then run the same
// mix arithmetic; they add the 320 products in different orders, so `mixed` is compared bit for bit where it can be
// and both against an FP64 reference of the same FP16 operands on sampled tokens: the fused one must be as close to
// it as the default, finite, and the same bits run to run.
//   hip_hc_upmix16_parity            the checks (T = 1 .. 8192)
//   hip_hc_upmix16_parity --bench    and the timings of both at T = 1024 / 2048 / 4096 (medians of 20 alternating runs)
// HIP on gfx103x only; elsewhere it skips (77).  STRATA_ROCBLAS_TUNING picks the default's table as in the engine.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;

uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
float from_bf16(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float f16v(uint16_t h) {
    __half v;
    std::memcpy(&v, &h, 2);
    return __half2float(v);
}
uint16_t f16b(float f) {
    const __half v = __float2half(f);
    uint16_t h;
    std::memcpy(&h, &v, 2);
    return h;
}
bool ck(hipError_t e, const char* what) {
    if (e != hipSuccess) std::printf("FAIL: %s: %s\n", what, hipGetErrorString(e));
    return e == hipSuccess;
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    if (!ck(hipMalloc(&p, n * sizeof(T)), "hipMalloc")) std::exit(2);
    return p;
}

struct Inputs {
    int64_t T;
    std::vector<uint16_t> lo16, wu;   // lo16: FP16 bits, wu: BF16
    std::vector<float> R, rs, w;
    uint16_t *d_lo16, *d_wu;
    float *d_R, *d_rs, *d_w;
    Inputs(int64_t t, uint32_t seed) : T(t) {
        std::mt19937 g(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        lo16.resize((size_t) T * LR);
        for (auto& x : lo16) x = f16b(nd(g) * 0.5f);   // silu outputs: O(1) and below
        wu.resize((size_t) D * LR);
        for (auto& x : wu) x = to_bf16(nd(g) * 0.12f);   // gated ~ N(0, ~1): sigmoid in its live range
        R.resize((size_t) T * D);
        for (auto& x : R) x = nd(g) * 3.0f;
        rs.resize((size_t) T * HC);
        for (auto& x : rs) x = 0.2f + 0.3f * std::fabs(nd(g));
        w.resize(D);
        for (auto& x : w) x = 1.0f + 0.1f * nd(g);
        d_lo16 = dalloc<uint16_t>(lo16.size());
        d_wu = dalloc<uint16_t>(wu.size());
        d_R = dalloc<float>(R.size());
        d_rs = dalloc<float>(rs.size());
        d_w = dalloc<float>(w.size());
        (void) hipMemcpy(d_lo16, lo16.data(), lo16.size() * 2, hipMemcpyHostToDevice);
        (void) hipMemcpy(d_wu, wu.data(), wu.size() * 2, hipMemcpyHostToDevice);
        (void) hipMemcpy(d_R, R.data(), R.size() * 4, hipMemcpyHostToDevice);
        (void) hipMemcpy(d_rs, rs.data(), rs.size() * 4, hipMemcpyHostToDevice);
        (void) hipMemcpy(d_w, w.data(), w.size() * 4, hipMemcpyHostToDevice);
    }
    ~Inputs() { (void) hipFree(d_lo16); (void) hipFree(d_wu); (void) hipFree(d_R); (void) hipFree(d_rs); (void) hipFree(d_w); }
    double exact(int64_t t, int d) const {   // mixed[t][d] in FP64 from the FP16 operands both paths multiply
        double s = 0.0;
        for (int c = 0; c < HC; ++c) {
            const int j = c * N + d;
            double g = 0.0;
            for (int k = 0; k < LR; ++k)
                g += (double) f16v(lo16[(size_t) t * LR + k]) * (double) f16v(f16b(from_bf16(wu[(size_t) j * LR + k])));
            const double x = (double) R[(size_t) t * D + j] * rs[(size_t) t * HC + c] * w[j];
            s += x / (1.0 + std::exp(-g));
        }
        return s / HC;
    }
};

struct Out {
    float* mixed;
    uint16_t *m16, *mh;
    explicit Out(int64_t T) { mixed = dalloc<float>((size_t) T * N); m16 = dalloc<uint16_t>((size_t) T * N); mh = dalloc<uint16_t>((size_t) T * N); }
    ~Out() { (void) hipFree(mixed); (void) hipFree(m16); (void) hipFree(mh); }
};

void run_default(strata::prefill::Gemm& gm, const Inputs& in, float* gated, const Out& o, hipStream_t st) {
    gm.keep_f16_next();
    gm.bf16(in.d_lo16, in.d_wu, gated, in.T, D, LR);
    strata::prefill::gr_mix_r(in.d_R, in.d_rs, in.d_w, gated, o.mixed, o.m16, in.T, st, o.mh, nullptr, gm.kept_f16());
}
bool run_fused(const Inputs& in, const Out& o, hipStream_t st) {
    return strata::prefill::gr_upmix16(in.d_lo16, in.d_wu, in.d_R, in.d_rs, in.d_w, o.mixed, o.m16, o.mh, in.T, st);
}

int check(strata::prefill::Gemm& gm, hipStream_t st, float* gated, int64_t T, uint32_t seed) {
    Inputs in(T, seed);
    Out ref(T), got(T), again(T);
    run_default(gm, in, gated, ref, st);
    if (!run_fused(in, got, st) || !run_fused(in, again, st)) { std::printf("FAIL T=%lld: gr_upmix16 refused\n", (long long) T); return 1; }
    if (!ck(hipStreamSynchronize(st), "run")) return 1;
    const size_t n = (size_t) T * N;
    std::vector<float> a(n), b(n), b2(n);
    std::vector<uint16_t> a16(n), b16(n), ah(n), bh(n);
    (void) hipMemcpy(a.data(), ref.mixed, n * 4, hipMemcpyDeviceToHost);
    (void) hipMemcpy(b.data(), got.mixed, n * 4, hipMemcpyDeviceToHost);
    (void) hipMemcpy(b2.data(), again.mixed, n * 4, hipMemcpyDeviceToHost);
    (void) hipMemcpy(a16.data(), ref.m16, n * 2, hipMemcpyDeviceToHost);
    (void) hipMemcpy(b16.data(), got.m16, n * 2, hipMemcpyDeviceToHost);
    (void) hipMemcpy(ah.data(), ref.mh, n * 2, hipMemcpyDeviceToHost);
    (void) hipMemcpy(bh.data(), got.mh, n * 2, hipMemcpyDeviceToHost);
    double md = 0, mv = 0;
    size_t bit_diff = 0, img_diff = 0, nonfinite = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(b[i])) ++nonfinite;
        md = std::max(md, (double) std::fabs(a[i] - b[i]));
        mv = std::max(mv, (double) std::fabs(a[i]));
        bit_diff += std::memcmp(&a[i], &b[i], 4) != 0;
        img_diff += a16[i] != b16[i] || ah[i] != bh[i];
    }
    const bool same = std::memcmp(b.data(), b2.data(), n * 4) == 0;
    std::mt19937 g(seed ^ 0x5a5a);
    const int samples = (int) std::min<int64_t>(T, 12);
    double ea = 0, eb = 0, ev = 0;
    for (int i = 0; i < samples; ++i) {
        const int64_t t = T <= 12 ? i : (int64_t) (g() % (uint32_t) T);
        for (int d = 0; d < N; d += 13) {
            const double x = in.exact(t, d);
            ea += (a[(size_t) t * N + d] - x) * (a[(size_t) t * N + d] - x);
            eb += (b[(size_t) t * N + d] - x) * (b[(size_t) t * N + d] - x);
            ev += x * x;
        }
    }
    ea = std::sqrt(ea / ev);
    eb = std::sqrt(eb / ev);
    const double rel = md / std::max(mv, 1e-30);
    const bool bad = !same || nonfinite != 0 || rel > 1e-3 || eb > 1.25 * ea + 1e-7;
    std::printf("%s T=%-5lld mixed max |diff| / max |value| %.2e, values differing %.3f%% (FP16 images %.3f%%); vs FP64 "
                "(RMS rel, sampled): default %.2e, fused %.2e%s\n", bad ? "FAIL" : "ok  ", (long long) T, rel,
                100.0 * (double) bit_diff / (double) n, 100.0 * (double) img_diff / (double) n, ea, eb,
                same ? "" : "  NOT DETERMINISTIC");
    return bad ? 1 : 0;
}

float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

void bench(strata::prefill::Gemm& gm, hipStream_t st, float* gated, int64_t T) {
    Inputs in(T, 7);
    Out o(T);
    hipEvent_t e0, e1;
    (void) hipEventCreate(&e0);
    (void) hipEventCreate(&e1);
    std::vector<float> tp, tf;
    for (int i = 0; i < 23; ++i) {
        for (int k = 0; k < 2; ++k) {
            const bool fused = (k == 0) == ((i & 1) != 0);
            (void) hipEventRecord(e0, st);
            if (fused) run_fused(in, o, st);
            else run_default(gm, in, gated, o, st);
            (void) hipEventRecord(e1, st);
            (void) hipEventSynchronize(e1);
            float ms;
            (void) hipEventElapsedTime(&ms, e0, e1);
            if (i >= 3) (fused ? tf : tp).push_back(ms);
        }
    }
    std::printf("  T=%-5lld default pair (convert + rocBLAS + gr_mix_r) %.3f ms | fused %.3f ms (%.1f TFLOP/s of its GEMM)\n",
                (long long) T, median(tp), median(tf), 2.0 * T * D * LR / median(tf) / 1e9);
    (void) hipEventDestroy(e0);
    (void) hipEventDestroy(e1);
}

}  // namespace

int main(int argc, char** argv) {
    {
        int dev = 0;
        hipDeviceProp_t p{};
        if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&p, dev) != hipSuccess) {
            std::printf("hc_upmix16_parity: no HIP device, skipped\n");
            return 77;
        }
        if (std::strncmp(p.gcnArchName, "gfx103", 6) != 0) {
            std::printf("hc_upmix16_parity: %s, the kernel is gfx103x only: skipped\n", p.gcnArchName);
            return 77;
        }
    }
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    hipStream_t st = nullptr;
    if (!ck(hipStreamCreate(&st), "stream")) return 1;
    strata::prefill::Gemm gm;
    std::string err;
    if (!gm.init(st, (int64_t) D * LR, err)) { std::printf("FAIL: %s\n", err.c_str()); return 1; }
    gm.set_f16_io(true);
    strata::prefill::set_act_f16(true);
    float* gated = dalloc<float>((size_t) 8192 * D);
    int fails = 0;
    uint32_t seed = 501;
    for (const int64_t T : {1, 7, 127, 128, 129, 600, 2047, 2048, 4096, 8192}) fails += check(gm, st, gated, T, seed++);
    if (do_bench) {
        std::printf("--bench: the up projection + mix (medians of 20 alternating runs)\n");
        for (const int64_t T : {1024, 2048, 4096}) bench(gm, st, gated, T);
    }
    (void) hipFree(gated);
    std::printf("hc_upmix16_parity: %d failures\n", fails);
    return fails ? 1 : 0;
}
