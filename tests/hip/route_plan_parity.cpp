// tests/hip/route_plan_parity.cpp - STRATA_ROUTE_PLAN's fused launch (the router's top-10 and the all-resident plan
// in one block) against the two launches it replaces (native_router_top10_multi, then resident_plan), word for word:
// the ids, the weights and the whole plan block, for 1 to 8 tokens, random logits (with ties and repeated experts
// across tokens), uniform and per-slot offsets, and a window with an expert that is not resident (an empty plan and
// the error flag).  GPU, synthetic, no model.
#include <hip/hip_runtime.h>
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(call) do { const auto e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); std::exit(2); } } while (0)

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    using namespace strata::kernels;
    hipStream_t stream;
    CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    const int NE = 512, K = 10, MAXT = 8;
    const long long capx = (long long) MAXT * K;
    const size_t plan_words = (size_t) 16 * capx + 64;
    float *d_logits, *d_w[2];
    int32_t *d_ids[2], *d_res, *d_plan[2];
    unsigned long long* d_off;
    uint32_t* d_err[2];
    CHECK(hipMalloc((void**) &d_logits, (size_t) MAXT * NE * 4));
    CHECK(hipMalloc((void**) &d_res, NE * 4));
    CHECK(hipMalloc((void**) &d_off, 4096 * 8));
    for (int a = 0; a < 2; ++a) {
        CHECK(hipMalloc((void**) &d_w[a], (size_t) MAXT * K * 4));
        CHECK(hipMalloc((void**) &d_ids[a], (size_t) MAXT * K * 4));
        CHECK(hipMalloc((void**) &d_plan[a], plan_words * 4));
        CHECK(hipMalloc((void**) &d_err[a], 64));
    }
    const uint8_t* cache_base = (const uint8_t*) 0x7000000000ull;   // never dereferenced: the plan stores addresses
    std::mt19937 rng(20261013);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    int bad = 0, cases = 0;
    for (int trial = 0; trial < 240; ++trial) {
        const int n = 1 + trial % MAXT;
        const bool offsets = (trial / 8) % 2 == 1, missing = trial % 29 == 28;
        std::vector<float> logits((size_t) MAXT * NE);
        for (auto& v : logits) v = nd(rng);
        if (trial % 3 == 1)   // later tokens repeat the first token's row (the same experts in several rows)
            for (int t = 1; t < n; ++t)
                if (rng() & 1) std::memcpy(&logits[(size_t) t * NE], &logits[0], NE * 4);
        if (trial % 5 == 2)   // exact ties
            for (int t = 0; t < n; ++t)
                for (int i = 0; i < 40; ++i) logits[(size_t) t * NE + rng() % NE] = 3.25f;
        std::vector<int32_t> res(NE);
        for (int e = 0; e < NE; ++e) res[e] = (int32_t) ((e * 7 + trial) % 4096);
        std::vector<unsigned long long> off(4096);
        for (int i = 0; i < 4096; ++i) off[i] = (unsigned long long) i * 1000003ull + (rng() % 4096) * 16;
        CHECK(hipMemcpy(d_logits, logits.data(), logits.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(d_off, off.data(), off.size() * 8, hipMemcpyHostToDevice));
        std::vector<int32_t> ids[2], plan[2];
        std::vector<float> w[2];
        uint32_t err[2] = {0, 0};
        for (int pass = 0; pass < (missing ? 2 : 1); ++pass) {
            if (pass == 1) {   // an expert the first token routes to is not resident
                CHECK(hipMemcpy(ids[0].data(), d_ids[0], (size_t) n * K * 4, hipMemcpyDeviceToHost));
                res[(size_t) ids[0][rng() % K]] = -1;
            }
            CHECK(hipMemcpy(d_res, res.data(), NE * 4, hipMemcpyHostToDevice));
            for (int a = 0; a < 2; ++a) {
                CHECK(hipMemsetAsync(d_w[a], 0x5a, (size_t) MAXT * K * 4, stream));
                CHECK(hipMemsetAsync(d_ids[a], 0x5a, (size_t) MAXT * K * 4, stream));
                CHECK(hipMemsetAsync(d_plan[a], 0x5a, plan_words * 4, stream));
                CHECK(hipMemsetAsync(d_err[a], 0, 64, stream));
                if (a == 0) {
                    native_router_top10_multi(d_logits, d_ids[0], d_w[0], n, stream);
                    resident_plan(d_ids[0], n * K, K, d_res, NE, cache_base, offsets ? d_off : nullptr, 2097152, d_plan[0],
                                  capx, nullptr, 0, stream, d_err[0]);
                } else {
                    route_plan(d_logits, d_ids[1], d_w[1], n, d_res, NE, cache_base, offsets ? d_off : nullptr, 2097152,
                               d_plan[1], capx, stream, d_err[1]);
                }
                CHECK(hipStreamSynchronize(stream));
                ids[a].resize((size_t) MAXT * K); w[a].resize((size_t) MAXT * K); plan[a].resize(plan_words);
                CHECK(hipMemcpy(ids[a].data(), d_ids[a], ids[a].size() * 4, hipMemcpyDeviceToHost));
                CHECK(hipMemcpy(w[a].data(), d_w[a], w[a].size() * 4, hipMemcpyDeviceToHost));
                CHECK(hipMemcpy(plan[a].data(), d_plan[a], plan_words * 4, hipMemcpyDeviceToHost));
                CHECK(hipMemcpy(&err[a], d_err[a], 4, hipMemcpyDeviceToHost));
            }
            ++cases;
            const bool same = ids[0] == ids[1] && std::memcmp(w[0].data(), w[1].data(), w[0].size() * 4) == 0 &&
                              plan[0] == plan[1] && err[0] == err[1];
            const bool err_ok = err[0] == (pass == 1 ? 1u : 0u);
            const bool planned = pass == 1 || (plan[0][0] > 0 && plan[0][0] <= n * K && plan[0][1] == n * K);
            if (!same || !err_ok || !planned) {
                if (bad++ < 8)
                    std::printf("  trial %d (%d tokens%s%s): %s%s%s\n", trial, n, offsets ? ", slot offsets" : "",
                                pass == 1 ? ", one expert missing" : "", same ? "" : "the fused launch differs ",
                                err_ok ? "" : "the reference's error flag is not what the case expects ",
                                planned ? "" : "the reference plan is empty");
            }
        }
    }
    std::printf("%s: route_plan against native_router_top10_multi + resident_plan, %d windows of 1-8 tokens, %d differ\n",
                bad ? "FAILED" : "PASSED", cases, bad);
    return bad ? 1 : 0;
}
