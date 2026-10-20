// tests/hip/pf_nowiden_parity.cpp - STRATA_PF_NOWIDEN: the readers of the GDN qkv / gate and QSA q projections take
// the FP16-out GEMM's output as it is (row t's values as FP16 at the start of its FP32 row) and widen each value as
// they read it.  Each reader is run on the widened FP32 rows and on the halves, on the same synthetic values, and
// the outputs must be the same bits: gdn_conv (the fused conv + L2 kernel of STRATA_GDN_CONVL2 from 65 tokens, the
// fallback widening below), the recurrence's out norm (STRATA_GDN_HEAD=1, STRATA_GDN_NOY=1, as the gfx1030 stage
// configs run it, and the widening fallback with STRATA_GDN_NOY unset is covered by the variant-0 call), split_q and
// gate_attn, the shared expert's swiglu_pair and the write-back gr_write_norm_rs; and Gemm::f16 with keep_f16_next() leaves halves whose widening is the default call's output.
// HIP only (the FP16 prompt path); exit 77 off the FP16 path.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
constexpr int64_t C = 10240, HV = 48, S = 128, ZV = HV * S, QF = 24 * 512;

uint16_t f16b(float f) {
    const __half v = __float2half(f);
    uint16_t h;
    std::memcpy(&h, &v, 2);
    return h;
}
float f16v(uint16_t h) {
    __half v;
    std::memcpy(&v, &h, 2);
    return __half2float(v);
}
template <typename T> T* dev(size_t n) {
    void* p = nullptr;
    if (hipMalloc(&p, n * sizeof(T)) != hipSuccess) { std::printf("FAIL: hipMalloc\n"); std::exit(2); }
    return (T*) p;
}
template <typename T> std::vector<T> pull(const T* d, size_t n) {
    std::vector<T> h(n);
    (void) hipDeviceSynchronize();
    (void) hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost);
    return h;
}
// a T x n FP16 output both ways: `wide` the floats, `half` the FP16 at the start of each FP32 row (the rest of the
// row holds a NaN pattern a wrong reader would pick up)
struct Out16 {
    float *wide, *half;
    Out16(int64_t T, int64_t n, std::mt19937& g, float scale) {
        std::normal_distribution<float> nd(0.0f, scale);
        std::vector<uint16_t> h((size_t) (T * n));
        for (auto& v : h) v = f16b(nd(g));
        std::vector<float> w((size_t) (T * n));
        std::vector<uint32_t> raw((size_t) (T * n), 0x7fc00001u);
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t c = 0; c < n; ++c) w[(size_t) (t * n + c)] = f16v(h[(size_t) (t * n + c)]);
            std::memcpy(reinterpret_cast<uint16_t*>(raw.data() + t * n), h.data() + t * n, (size_t) n * 2);
        }
        wide = dev<float>(w.size());
        half = dev<float>(raw.size());
        (void) hipMemcpy(wide, w.data(), w.size() * 4, hipMemcpyHostToDevice);
        (void) hipMemcpy(half, raw.data(), raw.size() * 4, hipMemcpyHostToDevice);
    }
    ~Out16() { (void) hipFree(wide); (void) hipFree(half); }
};
float* upload(const std::vector<float>& v) {
    float* d = dev<float>(v.size());
    (void) hipMemcpy(d, v.data(), v.size() * 4, hipMemcpyHostToDevice);
    return d;
}
std::vector<float> randv(size_t n, std::mt19937& g, float lo, float hi) {
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = u(g);
    return v;
}
int report(const char* what, int64_t T, bool same) {
    std::printf("%s %-26s T=%-5lld %s\n", same ? "ok  " : "FAIL", what, (long long) T, same ? "the same bits" : "DIFFERENT BITS");
    return same ? 0 : 1;
}

int conv(int64_t T, uint32_t seed) {
    std::mt19937 g(seed);
    Out16 q(T, C, g, 1.0f);
    const std::vector<float> hist0 = randv((size_t) C * 3, g, -1, 1), w = randv((size_t) C * 4, g, -0.5f, 0.5f);
    float *ha = upload(hist0), *hb = upload(hist0), *dw = upload(w);
    float *oa = dev<float>((size_t) (T * C)), *ob = dev<float>((size_t) (T * C));
    strata::prefill::gdn_conv(ha, q.wide, dw, oa, T, 1e-6f, nullptr, false);
    strata::prefill::gdn_conv(hb, q.half, dw, ob, T, 1e-6f, nullptr, true);
    const bool same = pull(oa, (size_t) (T * C)) == pull(ob, (size_t) (T * C)) &&
                      pull(ha, (size_t) C * 3) == pull(hb, (size_t) C * 3);
    (void) hipFree(ha); (void) hipFree(hb); (void) hipFree(dw); (void) hipFree(oa); (void) hipFree(ob);
    return report("gdn_conv", T, same);
}

int rec(int variant, int64_t T, uint32_t seed) {
    std::mt19937 g(seed);
    Out16 z(T, ZV, g, 1.5f);
    const std::vector<float> st0 = randv((size_t) (HV * S * S), g, -0.05f, 0.05f), h = randv((size_t) (T * C), g, -0.3f, 0.3f),
                             gate = randv((size_t) (T * HV), g, -0.5f, -0.01f), beta = randv((size_t) (T * HV), g, 0.1f, 0.9f),
                             gamma = randv((size_t) S, g, 0.5f, 1.5f);
    float *sa = upload(st0), *sb = upload(st0), *dh = upload(h), *dg = upload(gate), *db = upload(beta), *dgm = upload(gamma);
    float *ya = dev<float>((size_t) (T * ZV)), *yb = dev<float>((size_t) (T * ZV));
    uint16_t *a16 = dev<uint16_t>((size_t) (T * ZV)), *b16 = dev<uint16_t>((size_t) (T * ZV));
    strata::prefill::gdn_recurrence_variant(variant, sa, dh, dg, db, z.wide, dgm, 1e-6f, ya, a16, T, nullptr, 0, false);
    strata::prefill::gdn_recurrence_variant(variant, sb, dh, dg, db, z.half, dgm, 1e-6f, yb, b16, T, nullptr, 0, true);
    const bool same = pull(a16, (size_t) (T * ZV)) == pull(b16, (size_t) (T * ZV)) &&
                      pull(sa, st0.size()) == pull(sb, st0.size());
    for (float* p : {sa, sb, dh, dg, db, dgm, ya, yb}) (void) hipFree(p);
    (void) hipFree(a16); (void) hipFree(b16);
    return report(variant == 1 ? "recurrence norm (head)" : "recurrence (widened)", T, same);
}

int qsa(int64_t T, uint32_t seed) {
    std::mt19937 g(seed);
    Out16 q(T, QF, g, 2.0f);
    const std::vector<float> attn = randv((size_t) (T * 24 * 256), g, -1, 1);
    float* da = upload(attn);
    float *qa = dev<float>((size_t) (T * 24 * 256)), *qb = dev<float>((size_t) (T * 24 * 256));
    uint16_t *oa = dev<uint16_t>((size_t) (T * 24 * 256)), *ob = dev<uint16_t>((size_t) (T * 24 * 256));
    strata::prefill::split_q(q.wide, qa, T, nullptr, false);
    strata::prefill::split_q(q.half, qb, T, nullptr, true);
    strata::prefill::gate_attn(da, q.wide, oa, T, nullptr, 0, false);
    strata::prefill::gate_attn(da, q.half, ob, T, nullptr, 0, true);
    const bool s1 = pull(qa, (size_t) (T * 24 * 256)) == pull(qb, (size_t) (T * 24 * 256));
    const bool s2 = pull(oa, (size_t) (T * 24 * 256)) == pull(ob, (size_t) (T * 24 * 256));
    for (float* p : {da, qa, qb}) (void) hipFree(p);
    (void) hipFree(oa); (void) hipFree(ob);
    return report("split_q", T, s1) + report("gate_attn", T, s2);
}

int shexp(int64_t T, uint32_t seed) {
    std::mt19937 g(seed);
    Out16 a(T, 640, g, 2.0f), b(T, 640, g, 2.0f);
    uint16_t *o0 = dev<uint16_t>((size_t) (T * 640)), *o1 = dev<uint16_t>((size_t) (T * 640)), *o2 = dev<uint16_t>((size_t) (T * 640));
    strata::prefill::swiglu_pair(a.wide, b.wide, o0, T, nullptr);
    strata::prefill::swiglu_pair(a.half, b.half, o1, T, nullptr, true, true);
    strata::prefill::swiglu_pair(a.half, b.wide, o2, T, nullptr, true, false);
    const auto r0 = pull(o0, (size_t) (T * 640));
    const bool same = r0 == pull(o1, (size_t) (T * 640)) && r0 == pull(o2, (size_t) (T * 640));
    (void) hipFree(o0); (void) hipFree(o1); (void) hipFree(o2);
    return report("swiglu_pair", T, same);
}

int write_norm(int64_t T, uint32_t seed) {
    constexpr int64_t N = 2560, HC = 4, D = N * HC;
    std::mt19937 g(seed);
    Out16 bo(T, N, g, 0.5f);
    const std::vector<float> R0 = randv((size_t) (T * D), g, -3, 3), inj = randv((size_t) (T * HC), g, -2, 2),
                             w = randv((size_t) D, g, 0.8f, 1.2f);
    float *ra = upload(R0), *rb = upload(R0), *di = upload(inj), *dw = upload(w);
    float *sa = dev<float>((size_t) (T * HC)), *sb = dev<float>((size_t) (T * HC));
    uint16_t *xa = dev<uint16_t>((size_t) (T * D)), *xb = dev<uint16_t>((size_t) (T * D));
    strata::prefill::gr_write_norm_rs(ra, bo.wide, di, HC, dw, 1e-6f, sa, xa, T, nullptr);
    strata::prefill::gr_write_norm_rs(rb, bo.half, di, HC, dw, 1e-6f, sb, xb, T, nullptr, nullptr, 0, false, true);
    const bool same = pull(ra, R0.size()) == pull(rb, R0.size()) && pull(sa, (size_t) (T * HC)) == pull(sb, (size_t) (T * HC)) &&
                      pull(xa, (size_t) (T * D)) == pull(xb, (size_t) (T * D));
    for (float* p : {ra, rb, di, dw, sa, sb}) (void) hipFree(p);
    (void) hipFree(xa); (void) hipFree(xb);
    return report("gr_write_norm_rs", T, same);
}

int gemm(strata::prefill::Gemm& gm, int64_t T, int64_t N, int64_t K, uint32_t seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0, 1);
    std::vector<uint16_t> x((size_t) (T * K)), w((size_t) (N * K));
    for (auto& v : x) v = f16b(nd(g));
    for (auto& v : w) v = f16b(nd(g) * 0.05f);
    uint16_t *dx = dev<uint16_t>(x.size()), *dw = dev<uint16_t>(w.size());
    (void) hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice);
    (void) hipMemcpy(dw, w.data(), w.size() * 2, hipMemcpyHostToDevice);
    float *ya = dev<float>((size_t) (T * N)), *yb = dev<float>((size_t) (T * N));
    gm.f16(dx, dw, ya, T, N, K);
    gm.keep_f16_next();
    gm.f16(dx, dw, yb, T, N, K);
    const bool kept = gm.kept_f16();
    strata::prefill::widen16_rows(yb, N, N, T, nullptr);
    const bool same = kept && pull(ya, (size_t) (T * N)) == pull(yb, (size_t) (T * N));
    // and the request never outlives its call: the next one widens
    gm.f16(dx, dw, yb, T, N, K);
    const bool once = !gm.kept_f16() && pull(ya, (size_t) (T * N)) == pull(yb, (size_t) (T * N));
    for (float* p : {ya, yb}) (void) hipFree(p);
    (void) hipFree(dx); (void) hipFree(dw);
    return report("Gemm::f16 kept halves", T, same) + report("Gemm::f16 next call", T, once);
}
}  // namespace

int main() {
    setenv("STRATA_HIP_PROMPT_F16", "1", 1);
    setenv("STRATA_GDN_CONVL2", "1", 1);
    setenv("STRATA_GDN_HEAD", "1", 1);
    setenv("STRATA_GDN_PP", "2", 1);
    setenv("STRATA_GDN_NOY", "1", 1);
    hipStream_t st = nullptr;
    (void) hipStreamCreate(&st);
    strata::prefill::Gemm gm;
    std::string err;
    if (!strata::prefill::prompt_f16()) { std::printf("pf_nowiden_parity: not on the FP16 prompt path, skipped\n"); return 77; }
    if (!gm.init(nullptr, 2560 * 2560, err)) { std::printf("FAIL: %s\n", err.c_str()); return 1; }
    gm.set_f16_io(true);
    strata::prefill::set_act_f16(true);
    int fails = 0;
    uint32_t seed = 11;
    for (const int64_t T : {1, 7, 64, 65, 300, 2048}) {
        fails += conv(T, seed++);
        fails += rec(1, T, seed++);
        fails += rec(0, T, seed++);
        fails += qsa(T, seed++);
        fails += shexp(T, seed++);
        fails += write_norm(T, seed++);
        fails += gemm(gm, T, 2560, 2560, seed++);
    }
    std::printf("pf_nowiden_parity: %d failures\n", fails);
    return fails ? 1 : 0;
}
