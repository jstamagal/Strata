// tests/hip/norm_rope_parity.cpp - the fused RMSNorm + RoPE launch (native_qsa_rms_norm_rope, STRATA_NORM_ROPE) against
// the two launches it replaces (native_qsa_rms_norm_weighted, then native_rope_apply), bit for bit, for each of its
// three forms of the rotation on THIS card: head widths 128 and 256, the q / gate split (stride 2 x width) and the
// in-place call, no scaling and YaRN.  Says where a form differs (the rotated low half, the rotated high half, the
// columns past the rotation).  Exit 0 when the form given as the argument (default 1 = STRATA_NORM_ROPE=2) is equal.
// GPU, synthetic, no model.
#include <hip/hip_runtime.h>
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(call) do { const auto e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); std::exit(2); } } while (0)

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    using namespace strata::kernels;
    const int want = argc > 1 ? std::atoi(argv[1]) : 1;
    hipStream_t stream;
    CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    std::mt19937 rng(20261013);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    RopeScaling none, yarn2, yarn4;
    yarn2.type = RopeScalingType::YaRN; yarn2.factor = 2.0; yarn2.ext_factor = 1.0;
    yarn4.type = RopeScalingType::YaRN; yarn4.factor = 4.0; yarn4.ext_factor = 1.0;
    const RopeScaling* scalings[] = {&none, &yarn2, &yarn4};
    const char* sname[] = {"no scaling", "YaRN x2", "YaRN x4"};
    const char* fname[] = {"explicit fused multiply-adds (STRATA_NORM_ROPE=1)", "as native_rope_apply writes it (=2)",
                           "contraction off (=3)"};
    const int rows = 192, n_rot = 64;
    long long total_bad[3] = {0, 0, 0};
    for (int form = 0; form < 3; ++form) {
        long long lo = 0, hi = 0, tail = 0, n = 0;
        for (int hd : {128, 256})
            for (int si = 0; si < 3; ++si) {
                std::vector<float> xs((size_t) rows * 2 * hd), gamma((size_t) hd);
                std::vector<int> pos((size_t) rows);
                for (auto& v : xs) v = nd(rng);
                for (auto& g : gamma) g = 0.5f + std::fabs(nd(rng));
                for (int r = 0; r < rows; ++r) pos[(size_t) r] = r < 8 ? r : (int) (rng() % 300000);
                float *d_xs, *d_g, *d_ref, *d_fus;
                int* d_pos;
                CHECK(hipMalloc((void**) &d_xs, xs.size() * 4));
                CHECK(hipMalloc((void**) &d_g, gamma.size() * 4));
                CHECK(hipMalloc((void**) &d_ref, (size_t) rows * hd * 4));
                CHECK(hipMalloc((void**) &d_fus, (size_t) rows * hd * 4));
                CHECK(hipMalloc((void**) &d_pos, pos.size() * 4));
                CHECK(hipMemcpy(d_xs, xs.data(), xs.size() * 4, hipMemcpyHostToDevice));
                CHECK(hipMemcpy(d_g, gamma.data(), gamma.size() * 4, hipMemcpyHostToDevice));
                CHECK(hipMemcpy(d_pos, pos.data(), pos.size() * 4, hipMemcpyHostToDevice));
                std::vector<float> packed((size_t) rows * hd);
                for (int r = 0; r < rows; ++r) std::memcpy(&packed[(size_t) r * hd], &xs[(size_t) r * 2 * hd], (size_t) hd * 4);
                for (int inplace = 0; inplace < 2; ++inplace) {
                    CHECK(hipMemcpy(d_ref, packed.data(), packed.size() * 4, hipMemcpyHostToDevice));
                    native_qsa_rms_norm_weighted(d_ref, d_g, d_ref, hd, rows, 1e-6f, stream);
                    native_rope_apply(d_ref, d_ref, rows, hd, n_rot, *scalings[si], d_pos, stream);
                    native_norm_rope_tune(form);
                    if (inplace) {
                        CHECK(hipMemcpy(d_fus, packed.data(), packed.size() * 4, hipMemcpyHostToDevice));
                        (form == 1 ? native_qsa_rms_norm_rope_exact : native_qsa_rms_norm_rope)(d_fus, hd, d_g, d_fus, rows, hd, n_rot, 1e-6f, *scalings[si], d_pos, stream);
                    } else {
                        (form == 1 ? native_qsa_rms_norm_rope_exact : native_qsa_rms_norm_rope)(d_xs, 2 * hd, d_g, d_fus, rows, hd, n_rot, 1e-6f, *scalings[si], d_pos, stream);
                    }
                    native_norm_rope_tune(-1);
                    CHECK(hipStreamSynchronize(stream));
                    std::vector<float> href((size_t) rows * hd), hfus((size_t) rows * hd);
                    CHECK(hipMemcpy(href.data(), d_ref, href.size() * 4, hipMemcpyDeviceToHost));
                    CHECK(hipMemcpy(hfus.data(), d_fus, hfus.size() * 4, hipMemcpyDeviceToHost));
                    for (int r = 0; r < rows; ++r)
                        for (int d = 0; d < hd; ++d) {
                            const size_t i = (size_t) r * hd + d;
                            ++n;
                            if (std::memcmp(&href[i], &hfus[i], 4) == 0) continue;
                            if (d < n_rot / 2) ++lo; else if (d < n_rot) ++hi; else ++tail;
                        }
                }
                (void) sname;
                hipFree(d_xs); hipFree(d_g); hipFree(d_ref); hipFree(d_fus); hipFree(d_pos);
            }
        total_bad[form] = lo + hi + tail;
        std::printf("form %d, %-50s: %s (%lld values; differ: %lld rotated low, %lld rotated high, %lld past the rotation)\n",
                    form, fname[form], total_bad[form] ? "DIFFERENT" : "bitwise equal", n, lo, hi, tail);
    }
    const bool ok = want >= 0 && want < 3 && total_bad[want] == 0;
    std::printf("%s: form %d on this card\n", ok ? "PASSED" : "FAILED", want);
    return ok ? 0 : 1;
}
