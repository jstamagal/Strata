// tests/hip/hc_downsk_parity.cpp - STRATA_HC_DOWNSK: the hyper-connection [down; inject] product as Gemm::hc_down_sk's
// split-K kernel + gr_silu_inj_sk, against the FP16 path's default pair (Gemm::bf16_pair - the weights converted to
// FP16, rocBLAS FP16 in / FP16 out, widened - then gr_silu_inj), on synthetic inputs of the model's shapes (T x 10240
// FP16 image, N 320 + 4).  Both round each sum to FP16 once; they add the K = 10240 products in different orders, so
// the check is against an FP64 reference of the same FP16 inputs on sampled tokens: the split-K outputs must be as
// close to it as the default's, finite, and the same bits from run to run.
//   hip_hc_downsk_parity            the checks (T = 1 .. 8192, S = 1 .. 20)
//   hip_hc_downsk_parity --bench    and the timings of both at T = 1024 / 2048 / 4096 (medians of 20 alternating runs)
// HIP on gfx103x only (the kernel's device); elsewhere it skips (77).  STRATA_ROCBLAS_TUNING picks the default's
// solution table as in the engine.
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

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, NO = LR + HC;

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
// what the FP16 path computes with: the BF16 weight as FP16 (bf16_to_f16_rows; none of these leave FP16's range)
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
float bf16_as_f16(uint16_t h) { return f16v(f16b(from_bf16(h))); }
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
    std::vector<uint16_t> x, wd, wi;   // x: FP16 bits (the xn16 image), wd / wi: BF16
    uint16_t *d_x, *d_wd, *d_wi;
    Inputs(int64_t t, uint32_t seed) : T(t) {
        std::mt19937 g(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        x.resize((size_t) T * D);
        for (auto& v : x) v = f16b(nd(g));   // RMS-normed rows times a ~1 weight
        wd.resize((size_t) LR * D);
        for (auto& v : wd) v = to_bf16(nd(g) * 0.02f);
        wi.resize((size_t) HC * D);
        for (auto& v : wi) v = to_bf16(nd(g) * 0.02f);
        d_x = dalloc<uint16_t>(x.size());
        d_wd = dalloc<uint16_t>(wd.size());
        d_wi = dalloc<uint16_t>(wi.size());
        hipMemcpy(d_x, x.data(), x.size() * 2, hipMemcpyHostToDevice);
        hipMemcpy(d_wd, wd.data(), wd.size() * 2, hipMemcpyHostToDevice);
        hipMemcpy(d_wi, wi.data(), wi.size() * 2, hipMemcpyHostToDevice);
    }
    ~Inputs() { hipFree(d_x); hipFree(d_wd); hipFree(d_wi); }
    double exact(int64_t t, int n) const {   // the product in FP64 from the FP16 values both paths multiply
        const uint16_t* w = n < LR ? &wd[(size_t) n * D] : &wi[(size_t) (n - LR) * D];
        const uint16_t* xr = &x[(size_t) t * D];
        double s = 0.0;
        for (int k = 0; k < D; ++k) s += (double) f16v((xr[k])) * (double) bf16_as_f16(w[k]);
        return s;
    }
};

struct Out {
    uint16_t* lo16;
    float* inj;
    explicit Out(int64_t T) { lo16 = dalloc<uint16_t>((size_t) T * LR); inj = dalloc<float>((size_t) T * HC); }
    ~Out() { hipFree(lo16); hipFree(inj); }
    void get(int64_t T, std::vector<uint16_t>& l, std::vector<float>& i) const {
        l.resize((size_t) T * LR);
        i.resize((size_t) T * HC);
        hipMemcpy(l.data(), lo16, l.size() * 2, hipMemcpyDeviceToHost);
        hipMemcpy(i.data(), inj, i.size() * 4, hipMemcpyDeviceToHost);
    }
};

bool run_default(strata::prefill::Gemm& gm, const Inputs& in, float* buf, const Out& o, hipStream_t st) {
    if (!gm.bf16_pair(in.d_x, in.d_wd, LR, in.d_wi, HC, buf, in.T, D, NO)) return false;
    strata::prefill::gr_silu_inj(buf, NO, o.lo16, o.inj, in.T, st);
    return true;
}
bool run_sk(strata::prefill::Gemm& gm, const Inputs& in, float* buf, const Out& o, int S, hipStream_t st) {
    if (!gm.hc_down_sk(in.d_x, in.d_wd, LR, in.d_wi, HC, buf, in.T, D, S)) return false;
    strata::prefill::gr_silu_inj_sk(buf, S, o.lo16, o.inj, in.T, st);
    return true;
}

double silu_q(double v) { const double x = v / HC; return x / (1.0 + std::exp(-x)); }

int check(strata::prefill::Gemm& gm, hipStream_t st, float* buf, int64_t T, uint32_t seed) {
    Inputs in(T, seed);
    Out ref(T), got(T), again(T);
    if (!run_default(gm, in, buf, ref, st)) { std::printf("FAIL T=%lld: bf16_pair refused\n", (long long) T); return 1; }
    std::vector<uint16_t> lr;
    std::vector<float> ir;
    if (!ck(hipStreamSynchronize(st), "default")) return 1;
    ref.get(T, lr, ir);
    // FP64 on sampled tokens (all of them up to 16): the pre-activation sums, every output column
    std::mt19937 g(seed ^ 0x5a5a);
    const int samples = (int) std::min<int64_t>(T, 16);
    std::vector<int64_t> ts(samples);
    for (int i = 0; i < samples; ++i) ts[i] = T <= 16 ? i : (int64_t) (g() % (uint32_t) T);
    std::vector<double> ex((size_t) samples * NO);
    for (int i = 0; i < samples; ++i)
        for (int n = 0; n < NO; ++n) ex[(size_t) i * NO + n] = in.exact(ts[i], n);
    // each path's error against FP64: lo16 (as the silu of the sum) and inj (the sum), RMS relative to the RMS value
    auto err = [&](const std::vector<uint16_t>& l, const std::vector<float>& inj, double& el, double& ei) {
        double e2 = 0, r2 = 0, f2 = 0, q2 = 0;
        for (int i = 0; i < samples; ++i) {
            for (int n = 0; n < LR; ++n) {
                const double r = silu_q(ex[(size_t) i * NO + n]);
                const double v = f16v((l[(size_t) ts[i] * LR + n]));
                e2 += (v - r) * (v - r); r2 += r * r;
            }
            for (int c = 0; c < HC; ++c) {
                const double r = ex[(size_t) i * NO + LR + c], v = inj[(size_t) ts[i] * HC + c];
                f2 += (v - r) * (v - r); q2 += r * r;
            }
        }
        el = std::sqrt(e2 / std::max(r2, 1e-300));
        ei = std::sqrt(f2 / std::max(q2, 1e-300));
    };
    double rl, ri;
    err(lr, ir, rl, ri);
    int fails = 0;
    for (const int S : {1, 2, 4, 5, 8, 10, 16, 20}) {
        if (!run_sk(gm, in, buf, got, S, st)) { std::printf("FAIL T=%lld S=%d: hc_down_sk refused\n", (long long) T, S); ++fails; continue; }
        if (!run_sk(gm, in, buf, again, S, st)) { ++fails; continue; }
        if (!ck(hipStreamSynchronize(st), "split-K")) return fails + 1;
        std::vector<uint16_t> lg, la;
        std::vector<float> ig, ia;
        got.get(T, lg, ig);
        again.get(T, la, ia);
        const bool same = lg == la && std::memcmp(ig.data(), ia.data(), ig.size() * 4) == 0;
        size_t nonfinite = 0, l_diff = 0;
        for (size_t i = 0; i < lg.size(); ++i) {
            if (!std::isfinite(f16v((lg[i])))) ++nonfinite;
            l_diff += lg[i] != lr[i];
        }
        for (float v : ig) nonfinite += !std::isfinite(v);
        double gl, gi;
        err(lg, ig, gl, gi);
        // as close to FP64 as the default (its FP16 rounding dominates both), finite, deterministic
        const bool bad = !same || nonfinite != 0 || gl > 1.25 * rl + 1e-6 || gi > 1.5 * ri + 1e-6;
        std::printf("%s T=%-5lld S=%-2d lo16 differing from the default %6.3f%%; vs FP64 (RMS rel, sampled): "
                    "lo16 default %.2e split-K %.2e, inj default %.2e split-K %.2e%s\n", bad ? "FAIL" : "ok  ",
                    (long long) T, S, 100.0 * (double) l_diff / (double) lg.size(), rl, gl, ri, gi,
                    same ? "" : "  NOT DETERMINISTIC");
        fails += bad;
    }
    return fails;
}

float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

void bench(strata::prefill::Gemm& gm, hipStream_t st, float* buf, int64_t T) {
    Inputs in(T, 7);
    Out o(T);
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    const int Ss[] = {0, 2, 4, 5, 8, 10};   // 0: the default pair
    std::vector<std::vector<float>> t(6);
    for (int i = 0; i < 23; ++i) {
        for (int k = 0; k < 6; ++k) {
            const int j = (k + i) % 6;   // rotate which runs first
            hipEventRecord(e0, st);
            if (Ss[j] == 0) run_default(gm, in, buf, o, st);
            else run_sk(gm, in, buf, o, Ss[j], st);
            hipEventRecord(e1, st);
            hipEventSynchronize(e1);
            float ms;
            hipEventElapsedTime(&ms, e0, e1);
            if (i >= 3) t[j].push_back(ms);
        }
    }
    const double flop = 2.0 * T * NO * D;
    std::printf("  T=%-5lld default pair (convert + rocBLAS + widen + gr_silu_inj) %.3f ms (%.1f TFLOP/s)", (long long) T,
                median(t[0]), flop / median(t[0]) / 1e9);
    for (int j = 1; j < 6; ++j) std::printf(" | S=%d %.3f ms", Ss[j], median(t[j]));
    std::printf("\n");
    hipEventDestroy(e0);
    hipEventDestroy(e1);
}

}  // namespace

int main(int argc, char** argv) {
    {
        int dev = 0;
        hipDeviceProp_t p{};
        if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&p, dev) != hipSuccess) {
            std::printf("hc_downsk_parity: no HIP device, skipped\n");
            return 77;
        }
        if (std::strncmp(p.gcnArchName, "gfx103", 6) != 0) {
            std::printf("hc_downsk_parity: %s, the kernel is gfx103x only: skipped\n", p.gcnArchName);
            return 77;
        }
    }
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    hipStream_t st = nullptr;
    if (!ck(hipStreamCreate(&st), "stream")) return 1;
    strata::prefill::Gemm gm;
    std::string err;
    if (!gm.init(st, (int64_t) NO * D, err)) { std::printf("FAIL: %s\n", err.c_str()); return 1; }
    gm.set_f16_io(true);
    strata::prefill::set_act_f16(true);
    const int64_t Tmax = 8192;
    float* buf = dalloc<float>((size_t) Tmax * D);   // `gated`: the default's output rows, the split-K parts
    int fails = 0;
    uint32_t seed = 301;
    for (const int64_t T : {1, 7, 31, 64, 127, 128, 129, 600, 1536, 2047, 2048, 4096, 8192}) fails += check(gm, st, buf, T, seed++);
    if (do_bench) {
        std::printf("--bench: the [down; inject] product and its silu (medians of 20 rotating runs)\n");
        for (const int64_t T : {1024, 1536, 2048, 4096}) bench(gm, st, buf, T);
    }
    hipFree(buf);
    std::printf("hc_downsk_parity: %d failures\n", fails);
    return fails ? 1 : 0;
}
