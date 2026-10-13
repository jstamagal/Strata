// tests/hip/prefill_q4_parity.cpp - the RDNA2 Q4_0 / Q4_1 prompt-expert kernel (src/prefill/moe_q4.cu) through the
// mmq::quantize / mmq::Context::run contract, against a host reference that does the same arithmetic in FP64:
// the same activation rounding (d = max|x| / 127 over 32 values, q = rint(x / d)), the weights' own codes and scales.
// So the tolerance is tight (FP32 accumulation only): a wrong nibble order, block stride, row bound or scale shows.
// Then one timing of the two real shapes.  Exit 77 when the kernel is not enabled on this card.
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/moe_q4.hpp"
#include "ggml.h"
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strata::prefill;

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ok(hipMalloc(&p, n ? n : 16), "hipMalloc"); }
    ~Dev() { if (p) hipFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
};

std::vector<uint8_t> quantize_rows(ggml_type type, int64_t rows, int64_t cols, std::mt19937& rng) {
    const auto* tr = ggml_get_type_traits(type);
    if (!tr || !tr->from_float_ref) throw std::runtime_error("no host quantizer for the type");
    const size_t rb = ggml_row_size(type, cols);
    std::vector<uint8_t> out((size_t) rows * rb);
    std::vector<float> row((size_t) cols);
    std::normal_distribution<float> g(0.0f, 0.03f);
    for (int64_t r = 0; r < rows; ++r) {
        for (auto& v : row) v = g(rng);
        if (r % 97 == 5) std::fill(row.begin(), row.begin() + 64, 0.0f);   // some all-zero blocks
        tr->from_float_ref(row.data(), out.data() + (size_t) r * rb, cols);
    }
    return out;
}

// one weight row times one activation row, as the kernel defines it, in FP64
double ref_dot(ggml_type type, const uint8_t* wrow, const float* x, int64_t cols) {
    const bool q41 = type == GGML_TYPE_Q4_1;
    const int bb = q41 ? 20 : 18;
    double acc = 0.0;
    for (int64_t b = 0; b < cols / 32; ++b) {
        const uint8_t* blk = wrow + b * bb;
        uint16_t dh, mh = 0;
        std::memcpy(&dh, blk, 2);
        if (q41) std::memcpy(&mh, blk + 2, 2);
        const double dw = ggml_fp16_to_fp32(dh), mw = q41 ? ggml_fp16_to_fp32(mh) : 0.0;
        const uint8_t* qs = blk + (q41 ? 4 : 2);
        const float* xb = x + b * 32;
        float amax = 0.0f;
        for (int k = 0; k < 32; ++k) amax = std::fmax(amax, std::fabs(xb[k]));
        const float d = amax / 127.0f, inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        long sq = 0, sa = 0;
        for (int k = 0; k < 32; ++k) {
            const int a = (int) std::rint(xb[k] * inv);
            const int q = k < 16 ? (qs[k] & 15) : (qs[k - 16] >> 4);
            sq += (long) q * a;
            sa += a;
        }
        acc += q41 ? (double) d * (dw * (double) sq + mw * (double) sa) : dw * (double) d * (double) (sq - 8 * sa);
    }
    return acc;
}

// `variant`: q4::set_variant's (0 the 64 x 32 tile, 1 the 256 x 32 ones), or 2: the 256 x 32 ones reading each expert
// through its own two pointers (Product::each: the halves of its rows apart and swapped, at an odd address);
// `keep`: the output, for a bitwise compare
bool check(ggml_type type, int64_t w_rows, int64_t cols, const std::vector<int>& counts, unsigned seed, int variant = 0,
           std::vector<float>* keep = nullptr) {
    q4::set_variant(variant == 2 ? 1 : variant);
    std::mt19937 rng(seed);
    const int n = (int) counts.size();
    int total = 0, max_rows = 0;
    std::vector<int32_t> bounds(1, 0);
    for (int c : counts) { total += c; max_rows = std::max(max_rows, c); bounds.push_back(total); }
    const size_t eb = mmq::matrix_bytes(type, w_rows, cols);
    std::vector<uint8_t> w((size_t) n * eb);
    for (int e = 0; e < n; ++e) {
        const auto q = quantize_rows(type, w_rows, cols, rng);
        std::memcpy(w.data() + (size_t) e * eb, q.data(), eb);
    }
    std::vector<float> x((size_t) total * cols);
    std::normal_distribution<float> g(0.0f, 1.0f);
    for (auto& v : x) v = g(rng);
    if (total > 2) std::fill(x.begin() + cols, x.begin() + 2 * cols, 0.0f);   // one all-zero token row
    std::vector<int32_t> ids((size_t) total);
    for (int i = 0; i < total; ++i) ids[(size_t) i] = total - 1 - i;           // dst rows reversed

    hipStream_t s = nullptr;
    Dev dw(w.size()), dx(x.size() * 4), dq(mmq::q8_bytes(total, cols)), db(bounds.size() * 4), di(ids.size() * 4),
        dd((size_t) total * w_rows * 4);
    ok(hipMemcpy(dw.p, w.data(), w.size(), hipMemcpyHostToDevice), "w");
    ok(hipMemcpy(dx.p, x.data(), x.size() * 4, hipMemcpyHostToDevice), "x");
    ok(hipMemcpy(db.p, bounds.data(), bounds.size() * 4, hipMemcpyHostToDevice), "bounds");
    ok(hipMemcpy(di.p, ids.data(), ids.size() * 4, hipMemcpyHostToDevice), "ids");
    ok(hipMemset(dd.p, 0xff, (size_t) total * w_rows * 4), "dst");            // NaN bits: an unwritten output shows
    mmq::Context ctx;
    mmq::quantize((const float*) dx.p, nullptr, dq.p, type, cols, cols, total, s);
    mmq::Product p;
    p.w = dw.p; p.type = type; p.w_rows = w_rows; p.w_cols = cols; p.expert_bytes = eb; p.n = n;
    p.xq = dq.p; p.bounds = (const int32_t*) db.p; p.ids = (const int32_t*) di.p;
    p.total_rows = total; p.max_rows = max_rows; p.dst = (float*) dd.p; p.ld_dst = w_rows;
    // variant 2: expert e at 3 + e * (eb + 10) of a second buffer, rows [split, w_rows) first, then 6 bytes, then rows [0, split)
    const size_t rbw = ggml_row_size(type, cols);
    const int64_t split = w_rows / 2;
    const size_t lo = (size_t) split * rbw, hi = eb - lo;
    std::vector<uint8_t> w2(3 + (size_t) n * (eb + 10), 0xa5);
    for (int e = 0; e < n; ++e) {
        uint8_t* base = w2.data() + 3 + (size_t) e * (eb + 10);
        std::memcpy(base, w.data() + (size_t) e * eb + lo, hi);
        std::memcpy(base + hi + 6, w.data() + (size_t) e * eb, lo);
    }
    Dev dw2(w2.size());
    if (variant == 2) {
        if (n > mmq::kGatherGroupMax) throw std::runtime_error("too many experts for Product::each");
        ok(hipMemcpy(dw2.p, w2.data(), w2.size(), hipMemcpyHostToDevice), "w2");
        p.w = nullptr; p.expert_bytes = 0; p.each = true; p.w_split = split;
        for (int e = 0; e < n; ++e) {
            const uint8_t* base = (const uint8_t*) dw2.p + 3 + (size_t) e * (eb + 10);
            p.w1[e] = base;
            p.w0[e] = base + hi + 6;
        }
    }
    ctx.run(p, s);
    ok(hipDeviceSynchronize(), "sync");
    std::vector<float> y((size_t) total * w_rows);
    ok(hipMemcpy(y.data(), dd.p, y.size() * 4, hipMemcpyDeviceToHost), "y");

    const size_t rb = ggml_row_size(type, cols);
    double err2 = 0, ref2 = 0, worst = 0, zero_row = 0;
    long bad = 0;
    for (int e = 0; e < n; ++e)
        for (int r = bounds[(size_t) e]; r < bounds[(size_t) e + 1]; ++r)
            for (int64_t i = 0; i < w_rows; i += (w_rows > 256 ? 7 : 1)) {   // every 7th output row of the big shapes
                const double ref = ref_dot(type, w.data() + (size_t) e * eb + (size_t) i * rb, x.data() + (size_t) r * cols, cols);
                const float got = y[(size_t) ids[(size_t) r] * w_rows + (size_t) i];
                if (!std::isfinite(got)) { ++bad; continue; }
                const double d = (double) got - ref;
                err2 += d * d; ref2 += ref * ref; worst = std::max(worst, std::fabs(d));
                if (r == 1) zero_row = std::max(zero_row, std::fabs((double) got));
            }
    const double rel = ref2 > 0 ? std::sqrt(err2 / ref2) : 0;
    const bool pass = bad == 0 && rel < 2e-5 && zero_row == 0.0;
    if (keep) *keep = y;
    std::printf("%s v%d %-4s W[%lld x %lld] experts %d rows %d (max %d): rel L2 %.2e, worst |diff| %.2e, zero row max %.1e, "
                "unwritten %ld\n", pass ? "PASS" : "FAIL", variant, ggml_type_name(type), (long long) w_rows, (long long) cols, n,
                total, max_rows, rel, worst, zero_row, bad);
    return pass;
}

void time_shape(ggml_type type, int64_t w_rows, int64_t cols, int n, int rows_each) {
    std::mt19937 rng(7);
    const int total = n * rows_each;
    const size_t eb = mmq::matrix_bytes(type, w_rows, cols);
    std::vector<uint8_t> w((size_t) n * eb);
    const auto q = quantize_rows(type, w_rows, cols, rng);
    for (int e = 0; e < n; ++e) std::memcpy(w.data() + (size_t) e * eb, q.data(), eb);
    std::vector<float> x((size_t) total * cols);
    std::normal_distribution<float> g(0.0f, 1.0f);
    for (auto& v : x) v = g(rng);
    std::vector<int32_t> bounds((size_t) n + 1), ids((size_t) total);
    for (int e = 0; e <= n; ++e) bounds[(size_t) e] = e * rows_each;
    for (int i = 0; i < total; ++i) ids[(size_t) i] = i;
    Dev dw(w.size()), dx(x.size() * 4), dq(mmq::q8_bytes(total, cols)), db(bounds.size() * 4), di(ids.size() * 4),
        dd((size_t) total * w_rows * 4);
    ok(hipMemcpy(dw.p, w.data(), w.size(), hipMemcpyHostToDevice), "w");
    ok(hipMemcpy(dx.p, x.data(), x.size() * 4, hipMemcpyHostToDevice), "x");
    ok(hipMemcpy(db.p, bounds.data(), bounds.size() * 4, hipMemcpyHostToDevice), "bounds");
    ok(hipMemcpy(di.p, ids.data(), ids.size() * 4, hipMemcpyHostToDevice), "ids");
    mmq::Context ctx;
    mmq::Product p;
    p.w = dw.p; p.type = type; p.w_rows = w_rows; p.w_cols = cols; p.expert_bytes = eb; p.n = n;
    p.xq = dq.p; p.bounds = (const int32_t*) db.p; p.ids = (const int32_t*) di.p;
    p.total_rows = total; p.max_rows = rows_each; p.dst = (float*) dd.p; p.ld_dst = w_rows;
    double best_q = 1e30, best_p = 1e30;
    for (int rep = 0; rep < 6; ++rep) {
        ok(hipDeviceSynchronize(), "sync");
        auto t0 = std::chrono::steady_clock::now();
        mmq::quantize((const float*) dx.p, nullptr, dq.p, type, cols, cols, total, nullptr);
        ok(hipDeviceSynchronize(), "sync");
        auto t1 = std::chrono::steady_clock::now();
        ctx.run(p, nullptr);
        ok(hipDeviceSynchronize(), "sync");
        auto t2 = std::chrono::steady_clock::now();
        if (rep) {
            best_q = std::min(best_q, std::chrono::duration<double, std::milli>(t1 - t0).count());
            best_p = std::min(best_p, std::chrono::duration<double, std::milli>(t2 - t1).count());
        }
    }
    const double mac = (double) total * (double) w_rows * (double) cols;
    std::printf("time %-4s W[%lld x %lld] %d experts x %d rows: quantize %.3f ms, product %.3f ms (%.1f G MAC/s)\n",
                ggml_type_name(type), (long long) w_rows, (long long) cols, n, rows_each, best_q, best_p, mac / best_p / 1e6);
}
}  // namespace

int main() {
    try {
        if (!mmq::supported(GGML_TYPE_Q4_0)) {
            std::printf("SKIP: the Q4_0 kernel is not enabled on this card (gfx103x, STRATA_PF_Q4 not 0)\n");
            return 77;
        }
        bool all = true;
        struct C { ggml_type type; int64_t w_rows, cols; std::vector<int> counts; };
        const std::vector<C> cases = {
            {GGML_TYPE_Q4_0, 128, 256, {33, 0, 1, 70}},       // small, every output row checked
            {GGML_TYPE_Q4_1, 128, 256, {5, 64, 31}},
            {GGML_TYPE_Q4_0, 100, 384, {40}},                 // rows not a multiple of 64
            {GGML_TYPE_Q4_0, 300, 384, {40, 7}},              // rows not a multiple of 256
            {GGML_TYPE_Q4_0, 1280, 2560, {37, 0, 70, 32}},    // gate + up
            {GGML_TYPE_Q4_0, 2560, 640, {37, 0, 70, 32}},     // down
            {GGML_TYPE_Q4_1, 2560, 640, {19, 65, 3}},         // down on the six Q4_1 layers
        };
        unsigned seed = 1;
        for (const C& c : cases) {
            // each tile against FP64, and the 256 x 32 tiles (STRATA_PF_Q4_WIDE=1) against the 64 x 32 one bit for bit
            std::vector<float> y0, yv;
            all &= check(c.type, c.w_rows, c.cols, c.counts, seed, 0, &y0);
            for (int v = 1; v <= 2; ++v) {
                all &= check(c.type, c.w_rows, c.cols, c.counts, seed, v, &yv);
                const bool same = y0.size() == yv.size() && std::memcmp(y0.data(), yv.data(), y0.size() * 4) == 0;
                std::printf("%s v%d against v0: %s\n", same ? "PASS" : "FAIL", v, same ? "the same bits" : "outputs differ");
                all &= same;
            }
            ++seed;
        }
        for (int v = 0; v <= 1; ++v) {
            q4::set_variant(v);
            std::printf("variant %d\n", v);
            time_shape(GGML_TYPE_Q4_0, 1280, 2560, 16, 320);
            time_shape(GGML_TYPE_Q4_0, 2560, 640, 16, 320);
            time_shape(GGML_TYPE_Q4_1, 2560, 640, 16, 320);
        }
        q4::set_variant(-1);
        std::printf(all ? "ALL PASS\n" : "FAILED\n");
        return all ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
