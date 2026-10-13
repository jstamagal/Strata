// tests/hip/head_parity.cpp - STRATA_HEAD_WIDE's launch against the kernels it replaces, bit for bit: the logits
// against native_mmvq (Q6_K, 1-8 columns), the pick against sample_tokens' greedy pick, the flag against
// logits_nonfinite_rows.  Synthetic Q6_K rows of random bytes, with duplicated rows (ties), an all-zero column, a
// column of tiny values, and (second pass) blocks whose scale is infinite or NaN.  Several grids and row counts,
// among them counts that are not multiples of the grid.  Run it with and without STRATA_TSUM=1.
#include "strata/kernels/head_wide.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::kernels;

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

int main(int argc, char** argv) {
    const int N = 2560, MAXT = kHeadWideMaxCols;
    const int big = argc > 1 ? std::atoi(argv[1]) : 248320;
    if (!head_wide_supported(14, N, big, 1)) { std::printf("head_parity: head_wide is not supported here, skipped\n"); return 77; }
    cudaStream_t s; ck(cudaStreamCreate(&s), "stream");
    std::mt19937_64 rng(20261013);
    const size_t wb = native_mmvq_weight_bytes(14, N, big);
    std::vector<uint8_t> hw(wb);
    void *w, *xq, *scratch; float *x, *ya, *yb; int *oa, *ob; uint32_t *fa, *fb;
    ck(cudaMalloc(&w, wb), "w");
    ck(cudaMalloc(&xq, native_q8_1_bytes(N, MAXT)), "xq");
    ck(cudaMalloc(&scratch, head_wide_scratch_bytes()), "scratch");
    ck(cudaMalloc((void**) &x, (size_t) MAXT * N * 4), "x");
    ck(cudaMalloc((void**) &ya, (size_t) MAXT * big * 4), "ya"); ck(cudaMalloc((void**) &yb, (size_t) MAXT * big * 4), "yb");
    ck(cudaMalloc((void**) &oa, MAXT * 4), "oa"); ck(cudaMalloc((void**) &ob, MAXT * 4), "ob");
    ck(cudaMalloc((void**) &fa, MAXT * kNonfiniteBlocks * 4), "fa"); ck(cudaMalloc((void**) &fb, MAXT * kNonfiniteBlocks * 4), "fb");
    std::vector<float> ha((size_t) MAXT * big), hb((size_t) MAXT * big), hx((size_t) MAXT * N);
    std::vector<int> pa(MAXT), pb(MAXT);
    std::vector<uint32_t> na(MAXT * kNonfiniteBlocks), nb(MAXT * kNonfiniteBlocks);
    long long cases = 0, bad = 0, nonfinite_cols = 0, ties = 0;
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 0; i + 8 <= wb; i += 8) { const uint64_t r = rng(); std::memcpy(&hw[i], &r, 8); }
        for (size_t b = 0; b < wb / 210; ++b) {
            uint16_t d = (uint16_t) (0x2800 + (rng() & 0x3ff));                  // a small positive scale
            if (rng() % 7 == 0) d |= 0x8000;                                     // some negative
            if (rng() % 97 == 0) d = (uint16_t) (rng() % 2 ? 0x0000 : 0x8000);   // +0 / -0: the sign of a zero sum
            if (pass == 1 && rng() % 200000 == 0) d = (uint16_t) (rng() % 2 ? 0x7c00 : 0x7e00);   // inf / NaN
            std::memcpy(&hw[b * 210 + 208], &d, 2);
        }
        // ties: rows copied over later rows, and over the last row
        for (int k = 0; k < 64; ++k) {
            const size_t a = rng() % big, b = rng() % big;
            std::memcpy(&hw[b * 2100], &hw[a * 2100], 2100);
        }
        ck(cudaMemcpy(w, hw.data(), wb, cudaMemcpyHostToDevice), "w up");
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : hx) v = nd(rng);
        for (int i = 0; i < N; ++i) hx[(size_t) 2 * N + i] = 0.0f;               // column 2: zeros
        for (int i = 0; i < N; ++i) hx[(size_t) 5 * N + i] *= 1e-30f;            // column 5: tiny
        for (int i = 0; i < N; i += 3) hx[(size_t) 6 * N + i] = 0.0f;            // column 6: sparse
        ck(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x up");
        for (int n_out : {big, big - 1, 1000, 577, 3, 1}) {
            if (n_out < 1 || n_out > big) continue;
            for (int T = 1; T <= MAXT; ++T) {
                native_quantize_q8_1(x, xq, N, T, s);
                native_mmvq(14, w, xq, ya, N, n_out, T, s);
                SamplerParams g; g.greedy = true; g.temperature = 0.0f;
                sample_tokens(ya, T, n_out, nullptr, 0, g, oa, s);
                logits_nonfinite_rows(ya, T, n_out, fa, s);
                ck(cudaStreamSynchronize(s), "reference");
                ck(cudaMemcpy(ha.data(), ya, (size_t) T * n_out * 4, cudaMemcpyDeviceToHost), "ya");
                ck(cudaMemcpy(pa.data(), oa, T * 4, cudaMemcpyDeviceToHost), "oa");
                ck(cudaMemcpy(na.data(), fa, T * kNonfiniteBlocks * 4, cudaMemcpyDeviceToHost), "fa");
                for (int gv = 0; gv < 8; ++gv) {
                    const int grid = (const int[]){0, 64, 1000, 16384}[gv % 4];
                    if (n_out != big && grid == 16384) continue;
                    head_wide_set_grid(grid);
                    head_wide_set_variant(gv < 4 ? 2 : 3);
                    ck(cudaMemset(yb, 0xff, (size_t) T * n_out * 4), "clear");
                    head_wide(w, xq, yb, N, n_out, T, scratch, ob, fb, s);
                    ck(cudaStreamSynchronize(s), "head_wide");
                    ck(cudaMemcpy(hb.data(), yb, (size_t) T * n_out * 4, cudaMemcpyDeviceToHost), "yb");
                    ck(cudaMemcpy(pb.data(), ob, T * 4, cudaMemcpyDeviceToHost), "ob");
                    ck(cudaMemcpy(nb.data(), fb, T * kNonfiniteBlocks * 4, cudaMemcpyDeviceToHost), "fb");
                    ++cases;
                    const bool same = std::memcmp(ha.data(), hb.data(), (size_t) T * n_out * 4) == 0;
                    bool pick = true, flag = true;
                    for (int t = 0; t < T; ++t) {
                        pick = pick && pa[t] == pb[t];
                        uint32_t a = 0, b = 0;
                        for (int k = 0; k < kNonfiniteBlocks; ++k) { a |= na[t * kNonfiniteBlocks + k]; b |= nb[t * kNonfiniteBlocks + k]; }
                        flag = flag && (a != 0) == (b != 0);
                        if (gv == 0) {
                            nonfinite_cols += a != 0;
                            const float best = ha[(size_t) t * n_out + pa[t]];
                            int eq = 0;
                            for (int r = 0; r < n_out; ++r) eq += ha[(size_t) t * n_out + r] == best;
                            ties += eq > 1;
                        }
                    }
                    if (!same || !pick || !flag) {
                        ++bad;
                        size_t first = 0;
                        while (first < (size_t) T * n_out && std::memcmp(&ha[first], &hb[first], 4) == 0) ++first;
                        std::printf("MISMATCH variant %d pass %d n_out %d T %d grid %d: logits %s (first at col %zu row %zu: %a vs %a), "
                                    "pick %s, flag %s\n", gv < 4 ? 2 : 3, pass, n_out, T, grid, same ? "equal" : "DIFFER",
                                    first / (size_t) n_out, first % (size_t) n_out, same ? 0.0 : (double) ha[first],
                                    same ? 0.0 : (double) hb[first], pick ? "equal" : "DIFFERS", flag ? "equal" : "DIFFERS");
                    }
                }
            }
        }
    }
    // ---- sample_tokens_staged (STRATA_HEAD_SAMPLE) against sample_tokens: the same token for the same parameters,
    // read from device memory instead of baked; a staged greedy request leaves `out` alone
    long long scases = 0, sbad = 0;
    {
        const int NV = big;
        SamplerParams* dp; void* sscr;
        ck(cudaMalloc((void**) &dp, sizeof(SamplerParams) + 64), "dp");
        const size_t sb = sample_tokens_staged_scratch_bytes(MAXT, NV);
        if (sb == 0) { std::printf("head_parity: the staged sampler cannot run here (vocabulary or sampler switches)\n"); return 1; }
        ck(cudaMalloc(&sscr, sb), "sscr");
        native_quantize_q8_1(x, xq, N, MAXT, s);
        native_mmvq(14, w, xq, ya, N, NV, MAXT, s);
        // spread the logits like a head's: scale so the top few carry visible probability
        ck(cudaStreamSynchronize(s), "logits");
        ck(cudaMemcpy(ha.data(), ya, (size_t) MAXT * NV * 4, cudaMemcpyDeviceToHost), "ya");
        float amax = 0.0f;
        for (size_t i = 0; i < (size_t) MAXT * NV; ++i) if (std::isfinite(ha[i]) && std::fabs(ha[i]) > amax) amax = std::fabs(ha[i]);
        for (size_t i = 0; i < (size_t) MAXT * NV; ++i) ha[i] = std::isfinite(ha[i]) ? ha[i] * (12.0f / (amax > 0 ? amax : 1.0f)) : ha[i];
        for (int k = 0; k < 40; ++k) ha[(size_t) (rng() % MAXT) * NV + rng() % NV] = 11.5f + 0.01f * (float) (rng() % 100);   // near ties at the top
        ck(cudaMemcpy(ya, ha.data(), (size_t) MAXT * NV * 4, cudaMemcpyHostToDevice), "ya up");
        for (int trial = 0; trial < 400; ++trial) {
            SamplerParams sp;
            const int ks[] = {1, 2, 5, 20, 40, 64, 0, 100};
            const float tps[] = {1.0f, 0.95f, 0.8f, 0.5f, 0.1f};
            const float mps[] = {0.0f, 0.0f, 0.05f, 0.3f};
            const float temps[] = {1.0f, 0.7f, 1.5f, 0.2f, 3.0f};
            sp.top_k = ks[rng() % 8]; sp.top_p = tps[rng() % 5]; sp.min_p = mps[rng() % 4];
            sp.temperature = temps[rng() % 5]; sp.seed = rng(); sp.counter = rng() % 600000;
            sp.gumbel = rng() % 2 != 0; sp.min_keep = 1 + (int) (rng() % 2);
            if (trial % 25 == 24) { sp.greedy = true; }
            if (trial % 25 == 23) { sp.temperature = 0.0f; }
            const int T = 1 + (int) (rng() % MAXT);
            ck(cudaMemcpy(dp, &sp, sizeof sp, cudaMemcpyHostToDevice), "dp up");
            for (int t = 0; t < MAXT; ++t) pa[t] = pb[t] = -7 - t;
            ck(cudaMemcpy(oa, pa.data(), MAXT * 4, cudaMemcpyHostToDevice), "oa up");
            ck(cudaMemcpy(ob, pb.data(), MAXT * 4, cudaMemcpyHostToDevice), "ob up");
            const bool sampled = !sp.greedy && sp.temperature > 0.0f;
            if (sampled) sample_tokens(ya, T, NV, nullptr, 0, sp, oa, s);
            if (!sample_tokens_staged(ya, T, NV, dp, sscr, ob, s)) { std::printf("staged sampler did not launch\n"); return 1; }
            ck(cudaStreamSynchronize(s), "sampled");
            ck(cudaMemcpy(pa.data(), oa, MAXT * 4, cudaMemcpyDeviceToHost), "oa");
            ck(cudaMemcpy(pb.data(), ob, MAXT * 4, cudaMemcpyDeviceToHost), "ob");
            ++scases;
            bool same = true;
            for (int t = 0; t < MAXT; ++t) same = same && pa[t] == pb[t];
            if (!same) {
                ++sbad;
                std::printf("SAMPLER MISMATCH trial %d T %d top_k %d top_p %g min_p %g temp %g gumbel %d greedy %d: %d %d %d vs %d %d %d\n",
                            trial, T, sp.top_k, sp.top_p, sp.min_p, sp.temperature, (int) sp.gumbel, (int) sp.greedy, pa[0],
                            pa[1], pa[2], pb[0], pb[1], pb[2]);
            }
        }
    }
    std::printf("head_parity: staged sampler %lld cases, %lld mismatches\n", scases, sbad);
    bad += sbad;
    head_wide_set_grid(0);
    head_wide_set_variant(0);
    std::printf("head_parity: %lld cases, %lld mismatches (%lld columns with a tied maximum, %lld non-finite columns "
                "among the default-grid cases)\n", cases, bad, ties, nonfinite_cols);
    return bad == 0 ? 0 : 1;
}
