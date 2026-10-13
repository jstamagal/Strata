// tests/hip/prefill_expert_bench.cpp - the prompt path's expert products (mmq::quantize / mmq::Context::run) on the
// two real shapes, gate+up W[1280 x 2560] and down W[2560 x 640], for every weight type the packs on gfx103x use:
// a parity check against an FP64 reference and a timing in G MAC/s.
//
// The weights are random bytes with a sane block scale: every bit pattern of these formats decodes to finite
// weights, so no host quantizer is needed (the i-quant ones take minutes on these shapes).  The reference is
// ggml's own dequantizer times the activations in FP64, so the check covers both the weight decode and the
// activation rounding: it passes at the level of the int8 activation rounding (about 1e-2 relative), and it prints
// the error so two kernels compare.  With `--against-default` (and a STRATA_* switch in the environment that a
// child process is started without) it is the tool for an A/B of a new kernel; here it only prints one build's
// numbers.  Rows per expert: 320 each (a 16K-token chunk routes 163,840 rows to 512 experts), and a skewed set.
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/moe_q4.hpp"
#include "ggml.h"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// `rows` rows of random blocks: every byte random, each block's leading f16 scale set to a value near 0.02
// (and a Q4_1 / Q5_1 block's f16 minimum, which follows it, to a small negative one)
std::vector<uint8_t> random_rows(ggml_type type, int64_t rows, int64_t cols, std::mt19937& rng) {
    const size_t bs = ggml_type_size(type), rb = ggml_row_size(type, cols);
    std::vector<uint8_t> out((size_t) rows * rb);
    for (size_t i = 0; i + 4 <= out.size(); i += 4) {
        const uint32_t v = rng();
        std::memcpy(out.data() + i, &v, 4);
    }
    std::uniform_real_distribution<float> u(0.01f, 0.03f);
    for (size_t b = 0; b * bs < out.size(); ++b) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(u(rng));
        std::memcpy(out.data() + b * bs, &d, 2);
        if (type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_1) {
            const ggml_fp16_t m = ggml_fp32_to_fp16(-8.0f * u(rng));
            std::memcpy(out.data() + b * bs + 2, &m, 2);
        }
    }
    return out;
}

struct Case {
    ggml_type type;
    int64_t w_rows, cols;
    std::vector<int> counts;
};

struct Result { double rel = 0, ms_q = 0, ms_p = 0; uint64_t hash = 0; bool finite = true; };

uint64_t fnv(const void* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

Result run_case(const Case& c, bool check, unsigned seed, int reps) {
    std::mt19937 rng(seed);
    const int n = (int) c.counts.size();
    int total = 0, max_rows = 0;
    std::vector<int32_t> bounds(1, 0);
    for (int k : c.counts) { total += k; max_rows = std::max(max_rows, k); bounds.push_back(total); }
    const size_t eb = mmq::matrix_bytes(c.type, c.w_rows, c.cols);
    const size_t tail = 4096;   // MMQ reads past the last row of a 640-column matrix (see MMQ_TAIL in prefill.cpp)
    std::vector<uint8_t> w((size_t) n * eb + tail, 0);
    // a few distinct experts, repeated: the bytes differ per expert without minutes of setup
    const int distinct = std::min(n, 3);
    for (int e = 0; e < distinct; ++e) {
        const auto q = random_rows(c.type, c.w_rows, c.cols, rng);
        for (int k = e; k < n; k += distinct) std::memcpy(w.data() + (size_t) k * eb, q.data(), eb);
    }
    std::vector<float> x((size_t) total * c.cols);
    std::normal_distribution<float> g(0.0f, 1.0f);
    for (auto& v : x) v = g(rng);
    std::vector<int32_t> ids((size_t) total);
    for (int i = 0; i < total; ++i) ids[(size_t) i] = i;

    Dev dw(w.size()), dx(x.size() * 4), dq(mmq::q8_bytes(total, c.cols)), db(bounds.size() * 4), di(ids.size() * 4),
        dd((size_t) total * c.w_rows * 4);
    ok(hipMemcpy(dw.p, w.data(), w.size(), hipMemcpyHostToDevice), "w");
    ok(hipMemcpy(dx.p, x.data(), x.size() * 4, hipMemcpyHostToDevice), "x");
    ok(hipMemcpy(db.p, bounds.data(), bounds.size() * 4, hipMemcpyHostToDevice), "bounds");
    ok(hipMemcpy(di.p, ids.data(), ids.size() * 4, hipMemcpyHostToDevice), "ids");
    ok(hipMemset(dd.p, 0xff, (size_t) total * c.w_rows * 4), "dst");
    mmq::Context ctx;
    mmq::Product p;
    p.w = dw.p; p.type = c.type; p.w_rows = c.w_rows; p.w_cols = c.cols; p.expert_bytes = eb; p.n = n;
    p.xq = dq.p; p.bounds = (const int32_t*) db.p; p.ids = (const int32_t*) di.p;
    p.total_rows = total; p.max_rows = max_rows; p.dst = (float*) dd.p; p.ld_dst = c.w_rows;
    Result r;
    r.ms_q = r.ms_p = 1e30;
    for (int rep = 0; rep < reps; ++rep) {
        ok(hipDeviceSynchronize(), "sync");
        const auto t0 = std::chrono::steady_clock::now();
        mmq::quantize((const float*) dx.p, nullptr, dq.p, c.type, c.cols, c.cols, total, nullptr);
        ok(hipDeviceSynchronize(), "sync");
        const auto t1 = std::chrono::steady_clock::now();
        ctx.run(p, nullptr);
        ok(hipDeviceSynchronize(), "sync");
        const auto t2 = std::chrono::steady_clock::now();
        if (rep || reps == 1) {
            r.ms_q = std::min(r.ms_q, std::chrono::duration<double, std::milli>(t1 - t0).count());
            r.ms_p = std::min(r.ms_p, std::chrono::duration<double, std::milli>(t2 - t1).count());
        }
    }
    std::vector<float> y((size_t) total * c.w_rows);
    ok(hipMemcpy(y.data(), dd.p, y.size() * 4, hipMemcpyDeviceToHost), "y");
    r.hash = fnv(y.data(), y.size() * 4);
    for (float v : y) if (!std::isfinite(v)) { r.finite = false; break; }
    if (check) {
        // FP64: ggml's dequantized weights times the float activations (so the difference is the kernel's own
        // activation rounding plus its arithmetic); every 13th weight row and every 5th activation row
        const auto* tr = ggml_get_type_traits(c.type);
        if (!tr || !tr->to_float) throw std::runtime_error("no host dequantizer for the type");
        const size_t rb = ggml_row_size(c.type, c.cols);
        std::vector<float> wf((size_t) c.cols);
        double err2 = 0, ref2 = 0;
        for (int e = 0; e < n; ++e)
            for (int64_t i = e % 13; i < c.w_rows; i += 13) {
                tr->to_float(w.data() + (size_t) e * eb + (size_t) i * rb, wf.data(), c.cols);
                for (int rr = bounds[(size_t) e] + (int) (i % 5); rr < bounds[(size_t) e + 1]; rr += 5) {
                    const float* xr = x.data() + (size_t) rr * c.cols;
                    double acc = 0;
                    for (int64_t k = 0; k < c.cols; ++k) acc += (double) wf[(size_t) k] * (double) xr[k];
                    const double d = (double) y[(size_t) rr * c.w_rows + (size_t) i] - acc;
                    err2 += d * d; ref2 += acc * acc;
                }
            }
        r.rel = ref2 > 0 ? std::sqrt(err2 / ref2) : 0;
    }
    return r;
}

const char* shape_name(int64_t w_rows) { return w_rows == 1280 ? "gate+up" : "down"; }
}  // namespace

int main(int argc, char** argv) {
    try {
        bool only_time = false;
        std::vector<ggml_type> types = {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_S,
                                        GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, GGML_TYPE_Q2_0};
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--time-only") == 0) only_time = true;
            else if (std::strcmp(argv[i], "--type") == 0 && i + 1 < argc) types = {(ggml_type) std::atoi(argv[++i])};
            else if (std::strcmp(argv[i], "--q4-variant") == 0 && i + 1 < argc) q4::set_variant(std::atoi(argv[++i]));
        }
        // the skewed set: 16 experts of a 16K chunk as the router fills them (a few busy, most near the mean, some idle)
        const std::vector<int> skew = {1210, 12, 640, 333, 87, 301, 0, 455, 290, 1, 160, 702, 318, 64, 329, 218};
        bool all = true;
        for (ggml_type t : types) {
            for (int shape = 0; shape < 2; ++shape) {
                const int64_t w_rows = shape == 0 ? 1280 : 2560, cols = shape == 0 ? 2560 : 640;
                if (cols % ggml_blck_size(t) != 0) continue;   // no pack has it (a 256-value block in 640 columns)
                if (!mmq::fits(t, w_rows)) {
                    std::printf("skip %-8s %-7s: no kernel for it on this card\n", ggml_type_name(t), shape_name(w_rows));
                    continue;
                }
                if (!only_time) {
                    const Result r = run_case({t, w_rows, cols, {37, 0, 70, 32, 1, 129}}, true, 11u + (unsigned) t, 1);
                    const bool pass = r.finite && r.rel < 0.03;
                    all &= pass;
                    std::printf("%s %-8s %-7s W[%lld x %lld] 6 experts 269 rows: rel L2 vs FP64 %.3e, hash %016llx\n",
                                pass ? "PASS" : "FAIL", ggml_type_name(t), shape_name(w_rows), (long long) w_rows,
                                (long long) cols, r.rel, (unsigned long long) r.hash);
                }
                const Result u = run_case({t, w_rows, cols, std::vector<int>(16, 320)}, false, 7, 7);
                const Result s = run_case({t, w_rows, cols, skew}, false, 7, 7);
                const double mac_u = 16.0 * 320.0 * (double) w_rows * (double) cols;
                double rows_s = 0;
                for (int k : skew) rows_s += k;
                const double mac_s = rows_s * (double) w_rows * (double) cols;
                std::printf("time %-8s %-7s 16 x 320 rows: quantize %.3f ms, product %.3f ms = %.0f G MAC/s | skewed %d rows: "
                            "%.3f ms = %.0f G MAC/s\n", ggml_type_name(t), shape_name(w_rows), u.ms_q, u.ms_p,
                            mac_u / u.ms_p / 1e6, (int) rows_s, s.ms_p, mac_s / s.ms_p / 1e6);
            }
        }
        std::printf(all ? "ALL PASS\n" : "FAILED\n");
        return all ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
