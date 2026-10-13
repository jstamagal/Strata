// tests/hip/route_plan_parity.cpp - STRATA_ROUTE_PLAN's one launch against native_router_top10_multi (one token:
// native_router_top10) followed by resident_plan: the ids, the weights and every byte of the plan buffer, for 1-8
// tokens, logits with ties and tokens that share experts, a residency table with and without holes (the plan-error
// path).  --bench: 48 layers of router GEMV + top-10 + plan as one captured graph, both ways, microseconds per layer.
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/route_plan.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    const int NE = 512, KK = 10, MAXT = 8, N = 2560;
    const long long capx = (long long) MAXT * KK;
    const size_t plan_words = (size_t) (((4 + (capx + 1) + 2 * capx) + 1) & ~1ll) + 4 * capx + capx + 1 + 16;
    cudaStream_t s; ck(cudaStreamCreate(&s), "stream");
    float *dlog, *wa, *wb; int32_t *ia, *ib, *pa, *pb, *dres; unsigned long long* dso; uint32_t *ea, *eb;
    ck(cudaMalloc((void**) &dlog, MAXT * NE * 4), "log");
    ck(cudaMalloc((void**) &wa, MAXT * KK * 4), "wa"); ck(cudaMalloc((void**) &wb, MAXT * KK * 4), "wb");
    ck(cudaMalloc((void**) &ia, MAXT * KK * 4), "ia"); ck(cudaMalloc((void**) &ib, MAXT * KK * 4), "ib");
    ck(cudaMalloc((void**) &pa, plan_words * 4), "pa"); ck(cudaMalloc((void**) &pb, plan_words * 4), "pb");
    ck(cudaMalloc((void**) &dres, NE * 4), "res"); ck(cudaMalloc((void**) &dso, NE * 8), "so");
    ck(cudaMalloc((void**) &ea, 64), "ea"); ck(cudaMalloc((void**) &eb, 64), "eb");
    const uint8_t* cache_base = (const uint8_t*) 0x100000000ull;   // only its value enters the plan
    std::mt19937_64 rng(77);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    std::vector<float> hl(MAXT * NE), ha(MAXT * KK), hb(MAXT * KK);
    std::vector<int32_t> hia(MAXT * KK), hib(MAXT * KK), hpa(plan_words), hpb(plan_words), hres(NE);
    std::vector<unsigned long long> hso(NE);
    long long cases = 0, bad = 0, err_cases = 0;
    for (int trial = 0; trial < 600; ++trial) {
        const int T = 1 + (int) (rng() % MAXT);
        for (auto& v : hl) v = nd(rng);
        if (trial % 3 == 0)   // tokens that share most experts: one row copied with small changes
            for (int t = 1; t < MAXT; ++t)
                for (int e = 0; e < NE; ++e) hl[t * NE + e] = hl[e] + (rng() % 5 == 0 ? nd(rng) * 0.3f : 0.0f);
        if (trial % 4 == 1)   // ties: quantized logits
            for (auto& v : hl) v = std::round(v * 2.0f) / 2.0f;
        if (trial % 50 == 7) for (int e = 0; e < NE; ++e) hl[e] = 0.0f;   // an all-equal row
        for (int e = 0; e < NE; ++e) { hres[e] = (int32_t) ((e * 7919) % 100000); hso[e] = (rng() % (1ull << 34)) & ~63ull; }
        const bool holes = trial % 10 == 9;
        if (holes) for (int e = 0; e < NE; e += 1 + (int) (rng() % 3)) hres[e] = -1;
        const bool use_so = trial % 2 == 0;
        ck(cudaMemcpy(dlog, hl.data(), hl.size() * 4, cudaMemcpyHostToDevice), "log up");
        ck(cudaMemcpy(dres, hres.data(), NE * 4, cudaMemcpyHostToDevice), "res up");
        ck(cudaMemcpy(dso, hso.data(), NE * 8, cudaMemcpyHostToDevice), "so up");
        for (void* p : {(void*) wa, (void*) wb, (void*) ia, (void*) ib}) ck(cudaMemset(p, 0x5a, MAXT * KK * 4), "fill");
        ck(cudaMemset(pa, 0x3c, plan_words * 4), "fill"); ck(cudaMemset(pb, 0x3c, plan_words * 4), "fill");
        ck(cudaMemset(ea, 0, 64), "fill"); ck(cudaMemset(eb, 0, 64), "fill");
        if (T == 1) K::native_router_top10(dlog, ia, wa, s);
        else K::native_router_top10_multi(dlog, ia, wa, T, s);
        K::resident_plan(ia, T * KK, KK, dres, NE, cache_base, use_so ? dso : nullptr, 4096, pa, capx, nullptr, 0, s, ea);
        K::native_route_plan(dlog, ib, wb, T, dres, NE, cache_base, use_so ? dso : nullptr, 4096, pb, capx, eb, s);
        ck(cudaStreamSynchronize(s), "sync");
        uint32_t e1 = 0, e2 = 0;
        ck(cudaMemcpy(ha.data(), wa, MAXT * KK * 4, cudaMemcpyDeviceToHost), "wa"); ck(cudaMemcpy(hb.data(), wb, MAXT * KK * 4, cudaMemcpyDeviceToHost), "wb");
        ck(cudaMemcpy(hia.data(), ia, MAXT * KK * 4, cudaMemcpyDeviceToHost), "ia"); ck(cudaMemcpy(hib.data(), ib, MAXT * KK * 4, cudaMemcpyDeviceToHost), "ib");
        ck(cudaMemcpy(hpa.data(), pa, plan_words * 4, cudaMemcpyDeviceToHost), "pa"); ck(cudaMemcpy(hpb.data(), pb, plan_words * 4, cudaMemcpyDeviceToHost), "pb");
        ck(cudaMemcpy(&e1, ea, 4, cudaMemcpyDeviceToHost), "ea"); ck(cudaMemcpy(&e2, eb, 4, cudaMemcpyDeviceToHost), "eb");
        ++cases;
        err_cases += e1 != 0;
        const bool same = std::memcmp(ha.data(), hb.data(), MAXT * KK * 4) == 0 && std::memcmp(hia.data(), hib.data(), MAXT * KK * 4) == 0 &&
                          std::memcmp(hpa.data(), hpb.data(), plan_words * 4) == 0 && e1 == e2;
        if (!same) {
            ++bad;
            std::printf("MISMATCH trial %d T %d holes %d: ids %s, weights %s, plan %s, err %u vs %u\n", trial, T, (int) holes,
                        std::memcmp(hia.data(), hib.data(), MAXT * KK * 4) ? "DIFFER" : "equal",
                        std::memcmp(ha.data(), hb.data(), MAXT * KK * 4) ? "DIFFER" : "equal",
                        std::memcmp(hpa.data(), hpb.data(), plan_words * 4) ? "DIFFER" : "equal", e1, e2);
        }
    }
    std::printf("route_plan_parity: %lld cases, %lld mismatches (%lld with a non-resident expert)\n", cases, bad, err_cases);
    // ---- STRATA_ROUTE_PLAN=2: the projection in the launch too, against bf16_gemv_fp32_mmvf(_multi) + the top-10 + the
    // plan.  Every iteration has new activations and one of 8 weight matrices, and reuses the same logits buffer: a
    // stale logit (a read that did not see another workgroup's store) would show as a mismatch.
    long long fcases = 0, fbad = 0;
    {
        const int NW = 8;
        const int iters = argc > 2 ? std::atoi(argv[2]) : 20000;
        std::vector<uint16_t*> dw(NW);
        std::vector<uint16_t> w((size_t) NE * N);
        for (int l = 0; l < NW; ++l) {
            for (auto& x : w) { const float f = 0.05f * nd(rng); uint32_t u; std::memcpy(&u, &f, 4); x = (uint16_t) (u >> 16); }
            ck(cudaMalloc((void**) &dw[l], w.size() * 2), "w"); ck(cudaMemcpy(dw[l], w.data(), w.size() * 2, cudaMemcpyHostToDevice), "w up");
        }
        const int NX = 16;
        std::vector<float> xh((size_t) NX * MAXT * N);
        for (auto& x : xh) x = nd(rng);
        float *dx, *dlog2; uint32_t* done;
        ck(cudaMalloc((void**) &dx, xh.size() * 4), "x"); ck(cudaMemcpy(dx, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice), "x up");
        ck(cudaMalloc((void**) &dlog2, MAXT * NE * 4), "log2");
        ck(cudaMalloc((void**) &done, 256), "done"); ck(cudaMemset(done, 0, 256), "done0");
        for (int e = 0; e < NE; ++e) hres[e] = (int32_t) ((e * 7919) % 100000);
        ck(cudaMemcpy(dres, hres.data(), NE * 4, cudaMemcpyHostToDevice), "res up");
        std::vector<float> l1(MAXT * NE), l2(MAXT * NE);
        if (!K::native_route_fused(dx, N, dw[0], dlog2, done, ib, wb, 1, dres, NE, cache_base, nullptr, 4096, pb, capx, eb, s)) {
            std::printf("route_plan_parity: the fused launch is not available here\n");
        } else {
            ck(cudaStreamSynchronize(s), "first fused");
            for (int it = 0; it < iters; ++it) {
                const int T = 1 + (int) (rng() % MAXT);
                const float* xx = dx + (size_t) (rng() % NX) * MAXT * N;
                const uint16_t* ww = dw[rng() % NW];
                if (T == 1) { K::bf16_gemv_fp32_mmvf(xx, ww, dlog, N, NE, s); K::native_router_top10(dlog, ia, wa, s); }
                else { K::bf16_gemv_fp32_mmvf_multi(xx, N, ww, dlog, NE, N, NE, T, s); K::native_router_top10_multi(dlog, ia, wa, T, s); }
                K::resident_plan(ia, T * KK, KK, dres, NE, cache_base, nullptr, 4096, pa, capx, nullptr, 0, s, ea);
                K::native_route_fused(xx, N, ww, dlog2, done, ib, wb, T, dres, NE, cache_base, nullptr, 4096, pb, capx, eb, s);
                ck(cudaStreamSynchronize(s), "sync");
                ck(cudaMemcpy(ha.data(), wa, T * KK * 4, cudaMemcpyDeviceToHost), "wa"); ck(cudaMemcpy(hb.data(), wb, T * KK * 4, cudaMemcpyDeviceToHost), "wb");
                ck(cudaMemcpy(hia.data(), ia, T * KK * 4, cudaMemcpyDeviceToHost), "ia"); ck(cudaMemcpy(hib.data(), ib, T * KK * 4, cudaMemcpyDeviceToHost), "ib");
                ck(cudaMemcpy(l1.data(), dlog, T * NE * 4, cudaMemcpyDeviceToHost), "l1"); ck(cudaMemcpy(l2.data(), dlog2, T * NE * 4, cudaMemcpyDeviceToHost), "l2");
                uint32_t dn = 1;
                ck(cudaMemcpy(&dn, done, 4, cudaMemcpyDeviceToHost), "done");
                ++fcases;
                bool same = std::memcmp(ha.data(), hb.data(), T * KK * 4) == 0 && std::memcmp(hia.data(), hib.data(), T * KK * 4) == 0 &&
                            std::memcmp(l1.data(), l2.data(), T * NE * 4) == 0 && dn == 0;
                if (it % 16 == 0) {   // the plan's bytes too (the copies are most of an iteration)
                    ck(cudaMemcpy(hpa.data(), pa, plan_words * 4, cudaMemcpyDeviceToHost), "pa"); ck(cudaMemcpy(hpb.data(), pb, plan_words * 4, cudaMemcpyDeviceToHost), "pb");
                    const int groups = hpa[0];
                    same = same && hpa[0] == hpb[0] && hpa[1] == hpb[1] && hpa[2] == hpb[2] &&
                           std::memcmp(&hpa[4], &hpb[4], (size_t) (groups + 1) * 4) == 0 &&
                           std::memcmp(&hpa[4 + capx + 1], &hpb[4 + capx + 1], (size_t) T * KK * 4) == 0 &&
                           std::memcmp(&hpa[4 + capx + 1 + capx], &hpb[4 + capx + 1 + capx], (size_t) T * KK * 4) == 0;
                }
                if (!same) {
                    if (++fbad <= 10)
                        std::printf("FUSED MISMATCH it %d T %d: logits %s ids %s weights %s done %u\n", it, T,
                                    std::memcmp(l1.data(), l2.data(), T * NE * 4) ? "DIFFER" : "equal",
                                    std::memcmp(hia.data(), hib.data(), T * KK * 4) ? "DIFFER" : "equal",
                                    std::memcmp(ha.data(), hb.data(), T * KK * 4) ? "DIFFER" : "equal", dn);
                }
            }
            std::printf("route_plan_parity: fused projection %lld cases, %lld mismatches\n", fcases, fbad);
            bad += fbad;
        }
    }
    if (bench) {
        const int L = 48;
        std::vector<uint16_t> w((size_t) NE * N);
        for (auto& x : w) { const float f = 0.05f * nd(rng); uint32_t u; std::memcpy(&u, &f, 4); x = (uint16_t) (u >> 16); }
        std::vector<uint16_t*> dw(L);
        for (int l = 0; l < L; ++l) { ck(cudaMalloc((void**) &dw[l], w.size() * 2), "w"); ck(cudaMemcpy(dw[l], w.data(), w.size() * 2, cudaMemcpyHostToDevice), "w up"); }
        std::vector<float> xh((size_t) MAXT * N);
        for (auto& x : xh) x = nd(rng);
        float* dx; ck(cudaMalloc((void**) &dx, xh.size() * 4), "x"); ck(cudaMemcpy(dx, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice), "x up");
        for (int e = 0; e < NE; ++e) hres[e] = e;
        ck(cudaMemcpy(dres, hres.data(), NE * 4, cudaMemcpyHostToDevice), "res up");
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int T : {1, 2, 3, 4, 5}) {
            double us[3] = {0, 0, 0};
            uint32_t* bdone; ck(cudaMalloc((void**) &bdone, 256), "bdone"); ck(cudaMemset(bdone, 0, 256), "bdone0");
            for (int mode = 0; mode < 3; ++mode) {
                cudaGraph_t g; cudaGraphExec_t ge;
                ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
                for (int l = 0; l < L; ++l) {
                    if (mode == 2) {
                        K::native_route_fused(dx, N, dw[l], dlog, bdone, ib, wb, T, dres, NE, cache_base, nullptr, 4096, pb, capx, eb, s);
                        continue;
                    }
                    if (T == 1) K::bf16_gemv_fp32_mmvf(dx, dw[l], dlog, N, NE, s);
                    else K::bf16_gemv_fp32_mmvf_multi(dx, N, dw[l], dlog, NE, N, NE, T, s);
                    if (mode == 0) {
                        if (T == 1) K::native_router_top10(dlog, ia, wa, s);
                        else K::native_router_top10_multi(dlog, ia, wa, T, s);
                        K::resident_plan(ia, T * KK, KK, dres, NE, cache_base, nullptr, 4096, pa, capx, nullptr, 0, s, ea);
                    } else {
                        K::native_route_plan(dlog, ib, wb, T, dres, NE, cache_base, nullptr, 4096, pb, capx, eb, s);
                    }
                }
                ck(cudaStreamEndCapture(s, &g), "end capture");
                ck(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0), "instantiate");
                for (int i = 0; i < 5; ++i) ck(cudaGraphLaunch(ge, s), "launch");
                ck(cudaStreamSynchronize(s), "sync");
                std::vector<float> ms;
                for (int r = 0; r < 31; ++r) {
                    cudaEventRecord(e0, s); ck(cudaGraphLaunch(ge, s), "launch"); cudaEventRecord(e1, s);
                    ck(cudaEventSynchronize(e1), "esync");
                    float m = 0; cudaEventElapsedTime(&m, e0, e1); ms.push_back(m);
                }
                std::sort(ms.begin(), ms.end());
                us[mode] = 1e3 * ms[ms.size() / 2] / L;
                cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
            }
            std::printf("BENCH T %d: GEMV + top-10 + plan %6.2f us per layer | GEMV + route_plan %6.2f us per layer | fused %6.2f us per layer\n", T, us[0], us[1], us[2]);
        }
    }
    return bad == 0 ? 0 : 1;
}
