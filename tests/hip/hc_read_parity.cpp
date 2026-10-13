// tests/hip/hc_read_parity.cpp - the hyper-connection read of the FP16 prompt path (STRATA_HIP_PROMPT_F16=1) with
// STRATA_HC_GATED16 and STRATA_HC_DOWNINJ against the default sequence, through the engine's own Gemm and kernels.
//
//   default:   norm -> down GEMM (N 320) -> SiLU -> up GEMM (N 10240) + widening -> inject GEMM (N 4) -> mix
//   GATED16:   the up GEMM's FP16 output is not widened; the mix reads the halves.  Must be the same bits.
//   DOWNINJ:   down and inject as one GEMM (N 324).  Another rocBLAS kernel may run, so the sums can round
//              differently: both are compared with an FP64 reference of the same FP16 operands.
//
// Then the time of each sequence at 2048 tokens.  STRATA_ROCBLAS_TUNING is honoured (run it with the table the
// engine runs with).  GPU, synthetic, no model.  Exit 77 when the device is not on the FP16 prompt path.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CHECK(call) do { const auto e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); std::exit(2); } } while (0)

namespace {
constexpr int64_t N = 2560, HC = 4, D = N * HC, LR = 320;
constexpr float EPS = 1e-6f;

uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
double bf16_val(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
double f16_val(uint16_t h) {
    __half v;
    std::memcpy(&v, &h, 2);
    return __half2float(v);
}
template <typename T> T* dev(size_t n) {
    void* p = nullptr;
    CHECK(hipMalloc(&p, n * sizeof(T)));
    return (T*) p;
}
template <typename T> std::vector<T> pull(const T* d, size_t n, hipStream_t s) {
    std::vector<T> h(n);
    CHECK(hipStreamSynchronize(s));
    CHECK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return h;
}
struct Err { double rel = 0, max = 0; };
template <typename A> Err err_of(const A& got, const std::vector<double>& ref) {
    double e2 = 0, r2 = 0, mx = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double) got[i] - ref[i];
        e2 += d * d; r2 += ref[i] * ref[i]; mx = std::max(mx, std::fabs(d));
    }
    return {std::sqrt(e2 / std::max(r2, 1e-300)), mx};
}

struct Bufs {
    float *R, *wn, *rs, *lo, *gated, *inj, *mixed;
    uint16_t *xn16, *lo16, *wd, *wu, *wi, *mixed_bf, *mixed_h;
};

// mode bit 0: GATED16, bit 1: DOWNINJ
void hc_read(strata::prefill::Gemm& g, const Bufs& b, int64_t T, int mode, hipStream_t s) {
    using namespace strata::prefill;
    gr_norm_rs(b.R, b.wn, EPS, b.rs, b.xn16, T, s);
    bool pair = false;
    if (mode & 2) {
        pair = g.bf16_pair(b.xn16, b.wd, LR, b.wi, HC, b.gated, T, D, LR + HC);
        if (pair) gr_silu_inj(b.gated, LR + HC, b.lo16, b.inj, T, s);
    }
    if (!pair) {
        g.bf16(b.xn16, b.wd, b.lo, T, LR, D);
        gr_silu(b.lo, b.lo16, T, s);
    }
    if (mode & 1) g.keep_f16_next();
    g.bf16(b.lo16, b.wu, b.gated, T, D, LR);
    const bool g16 = (mode & 1) && g.kept_f16();
    if (!pair) g.bf16(b.xn16, b.wi, b.inj, T, HC, D);
    gr_mix_r(b.R, b.rs, b.wn, b.gated, b.mixed, b.mixed_bf, T, s, b.mixed_h, nullptr, g16);
    if ((mode & 1) && !g16) { std::printf("FAIL: the GEMM did not keep its FP16 output\n"); std::exit(1); }
    if ((mode & 2) && !pair) { std::printf("FAIL: the stacked down + inject product was refused\n"); std::exit(1); }
}
}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setenv("STRATA_HIP_PROMPT_F16", "1", 0);
    hipStream_t stream;
    CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    if (!strata::prefill::prompt_f16()) { std::printf("SKIP: not on the FP16 prompt path\n"); return 77; }
    strata::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(stream, (int64_t) D * N, err)) { std::printf("FAIL: %s\n", err.c_str()); return 1; }
    gemm.set_f16_io(true);
    strata::prefill::set_act_f16(true);

    const int64_t TMAX = 2048;
    Bufs b{};
    b.R = dev<float>(TMAX * D); b.wn = dev<float>(D); b.rs = dev<float>(TMAX * HC); b.lo = dev<float>(TMAX * LR);
    b.gated = dev<float>(TMAX * D); b.inj = dev<float>(TMAX * HC); b.mixed = dev<float>(TMAX * N);
    b.xn16 = dev<uint16_t>(TMAX * D); b.lo16 = dev<uint16_t>(TMAX * LR); b.wd = dev<uint16_t>(LR * D);
    b.wu = dev<uint16_t>(D * LR); b.wi = dev<uint16_t>(HC * D); b.mixed_bf = dev<uint16_t>(TMAX * N);
    b.mixed_h = dev<uint16_t>(TMAX * N);

    std::mt19937 rng(20261013);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> R((size_t) TMAX * D), wn(D);
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    for (auto& v : R) v = 3.0f * nd(rng);
    for (auto& v : wn) v = 1.0f + 0.2f * nd(rng);
    for (auto& v : wd) v = to_bf16(0.02f * nd(rng));
    for (auto& v : wu) v = to_bf16(0.08f * nd(rng));
    for (auto& v : wi) v = to_bf16(0.02f * nd(rng));
    CHECK(hipMemcpy(b.R, R.data(), R.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(b.wn, wn.data(), wn.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(b.wd, wd.data(), wd.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(b.wu, wu.data(), wu.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(b.wi, wi.data(), wi.size() * 2, hipMemcpyHostToDevice));

    bool all = true;
    for (const int64_t T : {(int64_t) 37, (int64_t) 300, (int64_t) 2048}) {
        struct Out { std::vector<float> mixed, inj; std::vector<uint16_t> mixed_bf, mixed_h, lo16, xn16; } o[4];
        for (int mode = 0; mode < 4; ++mode) {
            CHECK(hipMemsetAsync(b.gated, 0x5A, (size_t) TMAX * D * 4, stream));
            hc_read(gemm, b, T, mode, stream);
            o[mode].mixed = pull(b.mixed, T * N, stream);
            o[mode].inj = pull(b.inj, T * HC, stream);
            o[mode].mixed_bf = pull(b.mixed_bf, T * N, stream);
            o[mode].mixed_h = pull(b.mixed_h, T * N, stream);
            o[mode].lo16 = pull(b.lo16, T * LR, stream);
            o[mode].xn16 = pull(b.xn16, T * D, stream);
        }
        const auto same = [&](int a, int c) {
            return o[a].mixed.size() == o[c].mixed.size() &&
                   std::memcmp(o[a].mixed.data(), o[c].mixed.data(), o[a].mixed.size() * 4) == 0 &&
                   o[a].mixed_bf == o[c].mixed_bf && o[a].mixed_h == o[c].mixed_h && o[a].lo16 == o[c].lo16 &&
                   std::memcmp(o[a].inj.data(), o[c].inj.data(), o[a].inj.size() * 4) == 0;
        };
        const bool g16 = same(0, 1), g16_pair = same(2, 3);
        std::printf("%s T=%-5lld GATED16 against the default: %s;  DOWNINJ+GATED16 against DOWNINJ: %s\n",
                    g16 && g16_pair ? "PASS" : "FAIL", (long long) T, g16 ? "the same bits" : "DIFFERENT",
                    g16_pair ? "the same bits" : "DIFFERENT");
        all = all && g16 && g16_pair;
        std::printf("     T=%-5lld DOWNINJ against the default (not required to match): %s\n", (long long) T,
                    same(0, 2) ? "the same bits here" : "different bits");
        if (T > 300) continue;
        // FP64 reference of the down / inject products from the FP16 image the kernels wrote, then SiLU, up, mix
        const int64_t TR = std::min<int64_t>(T, 48);
        std::vector<double> lo_ref((size_t) TR * LR), inj_ref((size_t) TR * HC), silu_ref((size_t) TR * LR);
        std::vector<double> xv((size_t) TR * D);
        for (size_t i = 0; i < xv.size(); ++i) xv[i] = f16_val(o[0].xn16[i]);
        std::vector<double> wdv(wd.size()), wiv(wi.size());
        for (size_t i = 0; i < wd.size(); ++i) wdv[i] = bf16_val(wd[i]);
        for (size_t i = 0; i < wi.size(); ++i) wiv[i] = bf16_val(wi[i]);
        for (int64_t t = 0; t < TR; ++t) {
            for (int64_t n = 0; n < LR; ++n) {
                double s = 0;
                for (int64_t k = 0; k < D; ++k) s += xv[t * D + k] * wdv[n * D + k];
                lo_ref[t * LR + n] = s;
                const double x = s / HC;
                silu_ref[t * LR + n] = x / (1.0 + std::exp(-x));
            }
            for (int64_t n = 0; n < HC; ++n) {
                double s = 0;
                for (int64_t k = 0; k < D; ++k) s += xv[t * D + k] * wiv[n * D + k];
                inj_ref[t * HC + n] = s;
            }
        }
        for (int mode : {0, 2}) {
            std::vector<double> silu_got((size_t) TR * LR);
            for (size_t i = 0; i < silu_got.size(); ++i) silu_got[i] = f16_val(o[mode].lo16[i]);
            const Err es = err_of(silu_got, silu_ref), ei = err_of(o[mode].inj, inj_ref);
            // the mixed output against the default's (how far the rounding of lo carries)
            double e2 = 0, r2 = 0;
            for (size_t i = 0; i < o[0].mixed.size(); ++i) {
                const double d = (double) o[mode].mixed[i] - o[0].mixed[i];
                e2 += d * d; r2 += (double) o[0].mixed[i] * o[0].mixed[i];
            }
            const bool ok = es.rel < 2e-3 && ei.rel < 2e-3;
            std::printf("%s T=%-5lld %-8s against FP64: silu(down) rel L2 %.3e max %.3e;  inject rel L2 %.3e max %.3e;  "
                        "mixed against the default's rel L2 %.3e\n", ok ? "PASS" : "FAIL", (long long) T,
                        mode == 0 ? "default" : "DOWNINJ", es.rel, es.max, ei.rel, ei.max, std::sqrt(e2 / std::max(r2, 1e-300)));
            all = all && ok;
        }
    }
    std::printf("\nhyper-connection read at 2048 tokens (ms, the median of 9 runs of 10):\n");
    const char* const names[4] = {"default", "GATED16", "DOWNINJ", "DOWNINJ + GATED16"};
    for (int mode = 0; mode < 4; ++mode) {
        std::vector<double> runs;
        for (int r = 0; r < 10; ++r) {
            CHECK(hipStreamSynchronize(stream));
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 10; ++i) hc_read(gemm, b, 2048, mode, stream);
            CHECK(hipStreamSynchronize(stream));
            if (r > 0) runs.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 10);
        }
        std::sort(runs.begin(), runs.end());
        std::printf("  %-18s %.3f ms\n", names[mode], runs[runs.size() / 2]);
    }
    std::printf("%s\n", all ? "ALL PASS" : "FAILED");
    return all ? 0 : 1;
}
