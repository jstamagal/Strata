// tests/hip/prefill_expert_bench.cpp - the prompt path's expert products (mmq::quantize / mmq::Context::run) on the
// two real shapes, gate+up W[1280 x 2560] and down W[2560 x 640], for every weight type the packs on gfx103x use:
// a parity check against an FP64 reference and a timing in G MAC/s.
//
// The weights are random bytes with a sane block scale: every bit pattern of these formats decodes to finite
// weights, so no host quantizer is needed (the i-quant ones take minutes on these shapes).  The reference is
// ggml's own dequantizer times the activations in FP64, so the check covers both the weight decode and the
// activation rounding: it passes at the level of the int8 activation rounding (about 1e-2 relative).
// --write-reference PREFIX saves every output; --check-reference PREFIX compares every output against it.
// Run with STRATA_PF_IQ2S=0 STRATA_PF_IQ3=0, then both =1 with otherwise identical arguments.
// STRATA_PF_DIRECT=1 exercises separate expert blobs too. IQ down tests use 768 columns: 640 is not block-aligned.
// Product timings are best of five after one warmup. --rows 40 models the usual two-card 2K chunk.
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
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::string reference_prefix;
bool write_reference = false;
int reference_case = 0;
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
        if (type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_1 || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
            // Q4_K / Q5_K: {f16 d; f16 dmin; 6-bit scales and minima; codes} - d and dmin scale 6-bit integers
            const bool k = type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K;
            if (k) { const ggml_fp16_t dk = ggml_fp32_to_fp16(u(rng) / 32.0f); std::memcpy(out.data() + b * bs, &dk, 2); }
            const ggml_fp16_t m = ggml_fp32_to_fp16((k ? -0.25f : -8.0f) * u(rng));
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

// `each`: the experts lie apart on the device (in reverse order, a gap of 4,112 bytes between two) and the product
// reads each where it lies (Product::each, STRATA_PF_DIRECT=1) - the output must be the gathered one's, bit for bit
Result run_case(const Case& c, bool check, unsigned seed, int reps, bool each = false) {
    std::mt19937 rng(seed);
    const int n = (int) c.counts.size();
    int total = 0, max_rows = 0;
    std::vector<int32_t> bounds(1, 0);
    for (int k : c.counts) { total += k; max_rows = std::max(max_rows, k); bounds.push_back(total); }
    const int64_t ld = c.w_rows + (c.w_rows == 259 ? 7 : 0);
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
    for (int i = 0; i < total; ++i) ids[(size_t) i] = c.w_rows == 259 ? total - 1 - i : i;

    const size_t gap = 4112, eb_apart = eb + gap;
    std::vector<uint8_t> w_apart;
    if (each) {
        w_apart.assign((size_t) n * eb_apart + tail, 0x7f);   // 0x7f7f: an FP16 NaN wherever a kernel reads past a matrix
        for (int e = 0; e < n; ++e) std::memcpy(w_apart.data() + (size_t) (n - 1 - e) * eb_apart, w.data() + (size_t) e * eb, eb);
    }
    Dev dwa(w_apart.size());
    if (each) ok(hipMemcpy(dwa.p, w_apart.data(), w_apart.size(), hipMemcpyHostToDevice), "w apart");
    Dev dw(w.size()), dx(x.size() * 4), dq(mmq::q8_bytes(total, c.cols)), db(bounds.size() * 4), di(ids.size() * 4),
        dd((size_t) total * ld * 4);
    ok(hipMemcpy(dw.p, w.data(), w.size(), hipMemcpyHostToDevice), "w");
    ok(hipMemcpy(dx.p, x.data(), x.size() * 4, hipMemcpyHostToDevice), "x");
    ok(hipMemcpy(db.p, bounds.data(), bounds.size() * 4, hipMemcpyHostToDevice), "bounds");
    ok(hipMemcpy(di.p, ids.data(), ids.size() * 4, hipMemcpyHostToDevice), "ids");
    if (ld == c.w_rows) {
        ok(hipMemset(dd.p, 0xff, (size_t) total * ld * 4), "dst");
    } else {
        std::vector<float> initial((size_t) total * ld, std::numeric_limits<float>::quiet_NaN());
        for (int r = 0; r < total; ++r)
            std::fill(initial.begin() + r * ld + c.w_rows, initial.begin() + (r + 1) * ld, 0.0f);
        ok(hipMemcpy(dd.p, initial.data(), initial.size() * sizeof(float), hipMemcpyHostToDevice), "dst padding");
    }
    mmq::Context ctx;
    mmq::Product p;
    p.w = dw.p; p.type = c.type; p.w_rows = c.w_rows; p.w_cols = c.cols; p.expert_bytes = eb; p.n = n;
    p.xq = dq.p; p.bounds = (const int32_t*) db.p; p.ids = (const int32_t*) di.p;
    p.total_rows = total; p.max_rows = max_rows; p.dst = (float*) dd.p; p.ld_dst = ld;
    if (each) {
        if (n > mmq::kGatherGroupMax) throw std::runtime_error("too many experts for a product over their own blobs");
        p.each = true;
        p.w_split = c.w_rows == 1280 ? 640 : c.w_rows;   // gate/up: the up rows as a run of their own
        const size_t rb = ggml_row_size(c.type, c.cols);
        for (int e = 0; e < n; ++e) {
            p.w0[e] = (const uint8_t*) dwa.p + (size_t) (n - 1 - e) * eb_apart;
            p.w1[e] = (const uint8_t*) p.w0[e] + (p.w_split < c.w_rows ? (size_t) p.w_split * rb : 0);
        }
        p.w = nullptr;
    }
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
    std::vector<float> y((size_t) total * ld);
    ok(hipMemcpy(y.data(), dd.p, y.size() * 4, hipMemcpyDeviceToHost), "y");
    if (!reference_prefix.empty()) {
        const std::string path = reference_prefix + "." + std::to_string(reference_case++) + ".bin";
        FILE* f = std::fopen(path.c_str(), write_reference ? "wb" : "rb");
        if (!f) throw std::runtime_error("cannot open reference: " + path);
        if (write_reference) {
            const size_t count = std::fwrite(y.data(), sizeof(float), y.size(), f);
            std::fclose(f);
            if (count != y.size()) throw std::runtime_error("short reference write");
        } else {
            std::vector<float> ref(y.size());
            const size_t count = std::fread(ref.data(), sizeof(float), ref.size(), f);
            const bool eof = std::fgetc(f) == EOF;
            std::fclose(f);
            if (count != ref.size() || !eof) throw std::runtime_error("reference size mismatch");
            double err2 = 0, ref2 = 0, max_abs = 0;
            size_t different = 0;
            for (size_t k = 0; k < y.size(); ++k) {
                if (!std::isfinite(y[k]) || !std::isfinite(ref[k])) throw std::runtime_error("nonfinite parity output");
                const double d = (double) y[k] - ref[k];
                err2 += d*d; ref2 += (double) ref[k]*ref[k];
                max_abs = std::max(max_abs, std::abs(d));
                different += std::memcmp(&y[k], &ref[k], sizeof(float)) != 0;
                if (std::abs(d) > 2e-4 + 2e-5*std::abs(ref[k])) throw std::runtime_error("MMQ element parity failed");
            }
            std::printf("MMQ parity %zu outputs: %zu different bits, rel L2 %.3e, max abs %.3e\n",
                        y.size(), different, std::sqrt(err2/std::max(ref2, 1e-30)), max_abs);
        }
    }
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
                    const double d = (double) y[(size_t) ids[rr] * ld + (size_t) i] - acc;
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
        int devices = 0;
        if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
            std::fprintf(stderr, "skip: no HIP device\n");
            return 77;
        }
        bool only_time = false;
        std::vector<ggml_type> types = {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_S,
                                        GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, GGML_TYPE_Q2_0,
                                        GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0};
        // --rows N: the uniform case's rows per expert (default 320: a 16K chunk; a 2,048-token chunk of the two-card
        // pipeline routes 40 to an expert, where a kernel's per-expert weight decode weighs 8 times more)
        int uni_rows = 320;
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--time-only") == 0) only_time = true;
            else if (std::strcmp(argv[i], "--write-reference") == 0 && i + 1 < argc) {
                write_reference = true; reference_prefix = argv[++i];
            } else if (std::strcmp(argv[i], "--check-reference") == 0 && i + 1 < argc) reference_prefix = argv[++i];
            else if (std::strcmp(argv[i], "--type") == 0 && i + 1 < argc) types = {(ggml_type) std::atoi(argv[++i])};
            else if (std::strcmp(argv[i], "--rows") == 0 && i + 1 < argc) uni_rows = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "--nl-tile") == 0 && i + 1 < argc) q4::set_nl_tile(std::atoi(argv[++i]));
            else if (std::strcmp(argv[i], "--q4-variant") == 0 && i + 1 < argc) q4::set_variant(std::atoi(argv[++i]));
        }
        // the skewed set: 16 experts of a 16K chunk as the router fills them (a few busy, most near the mean, some idle)
        const std::vector<int> skew = {1210, 12, 640, 333, 87, 301, 0, 455, 290, 1, 160, 702, 318, 64, 329, 218};
        bool all = true;
        for (ggml_type t : types) {
            const bool iq_tile = t == GGML_TYPE_IQ2_S || t == GGML_TYPE_IQ3_XXS || t == GGML_TYPE_IQ3_S;
            if (iq_tile && !only_time) {
                const Result edge = run_case({t, 259, 256, {0, 1, 15, 16, 17, 31, 32, 33, 40, 63, 64, 65}}, true, 101, 1);
                const bool pass = edge.finite && edge.rel < 0.03;
                all &= pass;
                std::printf("%s %s edge rows, reversed ids, padded destination: rel L2 %.3e\n",
                            pass ? "PASS" : "FAIL", ggml_type_name(t), edge.rel);
            }
            for (int shape = 0; shape < 2; ++shape) {
                const int64_t w_rows = shape == 0 ? 1280 : 2560, cols = shape == 0 ? 2560 : (iq_tile ? 768 : 640);
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
                // STRATA_PF_DIRECT=1: a product over the experts' own blobs gives the gathered one's bits
                const bool can_each = mmq::direct_blobs(t, shape == 1);
                if (can_each && !only_time) {
                    const Result a = run_case({t, w_rows, cols, {37, 0, 70, 32, 1, 129}}, false, 11u + (unsigned) t, 1);
                    const Result b = run_case({t, w_rows, cols, {37, 0, 70, 32, 1, 129}}, false, 11u + (unsigned) t, 1, true);
                    const bool same = a.hash == b.hash && b.finite;
                    all &= same;
                    std::printf("%s %-8s %-7s experts read from their own blobs: %s (hash %016llx)\n", same ? "PASS" : "FAIL",
                                ggml_type_name(t), shape_name(w_rows), same ? "the same bits" : "outputs differ",
                                (unsigned long long) b.hash);
                }
                if (can_each) {
                    const Result ue = run_case({t, w_rows, cols, std::vector<int>(16, uni_rows)}, false, 7, 6, true);
                    std::printf("time %-8s %-7s 16 x %d rows, from their own blobs: product %.3f ms = %.0f G MAC/s\n",
                                ggml_type_name(t), shape_name(w_rows), uni_rows, ue.ms_p,
                                16.0 * (double) uni_rows * (double) w_rows * (double) cols / ue.ms_p / 1e6);
                }
                const Result u = run_case({t, w_rows, cols, std::vector<int>(16, uni_rows)}, false, 7, 6);
                const Result s = run_case({t, w_rows, cols, skew}, false, 7, 6);
                const double mac_u = 16.0 * (double) uni_rows * (double) w_rows * (double) cols;
                double rows_s = 0;
                for (int k : skew) rows_s += k;
                const double mac_s = rows_s * (double) w_rows * (double) cols;
                std::printf("time %-8s %-7s 16 x %d rows: quantize %.3f ms, product %.3f ms = %.0f G MAC/s | skewed %d rows: "
                            "%.3f ms = %.0f G MAC/s\n", ggml_type_name(t), shape_name(w_rows), uni_rows, u.ms_q, u.ms_p,
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
