// tests/hip/dq8_parity.cpp - STRATA_DQ8's dequantization kernels (four threads per 32-value group, 16-byte stores)
// against the one-thread-per-group kernels, bit for bit, for every type they cover and every output format (FP16,
// BF16, FP32, FP16 with a padded row stride), on random blocks.  Then the time of each on the prompt path's largest
// dense projection (10240 x 2560).  GPU, synthetic, no model.
#include <hip/hip_runtime.h>
#include "strata/kernels/dequant_bf16.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(call) do { const auto e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); std::exit(2); } } while (0)

namespace {
struct Type { int id; const char* name; int block_elems, block_bytes, d_off, d_count; };
// d_off / d_count: where the block's FP16 scales sit (kept finite and small; every other byte is random)
const Type kTypes[] = {{2, "Q4_0", 32, 18, 0, 1},   {6, "Q5_0", 32, 22, 0, 1},    {12, "Q4_K", 256, 144, 0, 2},
                       {13, "Q5_K", 256, 176, 0, 2}, {14, "Q6_K", 256, 210, 208, 1}, {23, "IQ4_XS", 256, 136, 0, 1},
                       {8, "Q8_0", 32, 34, 0, 1},    {11, "Q3_K", 256, 110, 108, 1}};

uint16_t f16_bits(float f) {   // a finite value well inside FP16: sign, exponent 2^-9 .. 2^-2, random mantissa
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = u >> 31, exp = ((u >> 23) & 0xFF) - 127 + 15, man = (u >> 13) & 0x3FF;
    return (uint16_t) ((sign << 15) | (exp << 10) | man);
}

uint64_t fnv(const void* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ ((const uint8_t*) p)[i]) * 1099511628211ull;
    return h;
}
}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    hipStream_t stream;
    CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    using strata::kernels::dequant_set_piece8;
    bool all = true;
    std::mt19937 rng(20261013);
    std::uniform_real_distribution<float> scale(0.002f, 0.2f);
    for (const Type& t : kTypes) {
        const int64_t rows_total = 70, row0 = 3, rows = 64, cols = 2560, ld = cols + 64;
        const size_t row_bytes = (size_t) cols / t.block_elems * t.block_bytes;
        std::vector<uint8_t> blocks(rows_total * row_bytes);
        for (auto& b : blocks) b = (uint8_t) rng();
        for (size_t o = 0; o + t.block_bytes <= blocks.size(); o += t.block_bytes)
            for (int k = 0; k < t.d_count; ++k) {
                const uint16_t h = f16_bits((rng() & 1 ? -1.0f : 1.0f) * scale(rng));
                std::memcpy(&blocks[o + t.d_off + 2 * k], &h, 2);
            }
        void *dblocks = nullptr, *dout = nullptr;
        CHECK(hipMalloc(&dblocks, blocks.size()));
        CHECK(hipMalloc(&dout, (size_t) rows * ld * 4));
        CHECK(hipMemcpy(dblocks, blocks.data(), blocks.size(), hipMemcpyHostToDevice));
        std::vector<uint8_t> got[2];
        const char* const fmt[4] = {"f16", "bf16", "f32", "f16 ld"};
        for (int f = 0; f < 4; ++f) {
            const size_t bytes = f == 2 ? (size_t) rows * cols * 4 : f == 3 ? (size_t) rows * ld * 2 : (size_t) rows * cols * 2;
            for (int mode = 0; mode < 2; ++mode) {
                dequant_set_piece8(mode);
                CHECK(hipMemsetAsync(dout, 0xA5, (size_t) rows * ld * 4, stream));
                if (f == 0) strata::kernels::dequant_f16(t.id, dblocks, row0, rows, cols, (uint16_t*) dout, stream);
                else if (f == 1) strata::kernels::dequant_bf16(t.id, dblocks, row0, rows, cols, (uint16_t*) dout, stream);
                else if (f == 2) strata::kernels::dequant_f32(t.id, dblocks, row0, rows, cols, (float*) dout, stream);
                else if (!strata::kernels::dequant_f16_ld(t.id, dblocks, row0, rows, cols, ld, (uint16_t*) dout, stream)) {
                    std::printf("FAIL %s %s: dequant_f16_ld refused\n", t.name, fmt[f]);
                    all = false;
                }
                CHECK(hipStreamSynchronize(stream));
                got[mode].assign((size_t) rows * ld * 4, 0);   // the whole buffer: bytes outside the output must match too
                CHECK(hipMemcpy(got[mode].data(), dout, got[mode].size(), hipMemcpyDeviceToHost));
            }
            const bool same = got[0] == got[1];
            size_t nonfinite = 0;
            if (f == 2) {
                const float* v = (const float*) got[0].data();
                for (size_t i = 0; i < (size_t) rows * cols; ++i) nonfinite += !(v[i] - v[i] == 0.0f);
            }
            std::printf("%s %-6s %-6s %s  fnv %016llx / %016llx%s\n", same ? "PASS" : "FAIL", t.name, fmt[f],
                        same ? "the same bits" : "DIFFERENT", (unsigned long long) fnv(got[0].data(), bytes),
                        (unsigned long long) fnv(got[1].data(), bytes), nonfinite ? "  (non-finite values in the reference)" : "");
            all = all && same && nonfinite == 0;
        }
        CHECK(hipFree(dblocks));
        CHECK(hipFree(dout));
    }
    // timings: 10240 x 2560 into FP16, the median of 9 runs of 20 launches each
    std::printf("\ntime per 10240 x 2560 dequantization to FP16 (ms; one thread per group -> STRATA_DQ8):\n");
    for (const Type& t : kTypes) {
        const int64_t rows = 10240, cols = 2560;
        const size_t row_bytes = (size_t) cols / t.block_elems * t.block_bytes;
        std::vector<uint8_t> blocks(rows * row_bytes);
        for (auto& b : blocks) b = (uint8_t) rng();
        void *dblocks = nullptr, *dout = nullptr;
        CHECK(hipMalloc(&dblocks, blocks.size()));
        CHECK(hipMalloc(&dout, (size_t) rows * cols * 2));
        CHECK(hipMemcpy(dblocks, blocks.data(), blocks.size(), hipMemcpyHostToDevice));
        double ms[2] = {};
        for (int mode = 0; mode < 2; ++mode) {
            dequant_set_piece8(mode);
            std::vector<double> runs;
            for (int r = 0; r < 10; ++r) {
                CHECK(hipStreamSynchronize(stream));
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 20; ++i) strata::kernels::dequant_f16(t.id, dblocks, 0, rows, cols, (uint16_t*) dout, stream);
                CHECK(hipStreamSynchronize(stream));
                if (r > 0) runs.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 20);
            }
            std::sort(runs.begin(), runs.end());
            ms[mode] = runs[runs.size() / 2];
        }
        std::printf("  %-6s %.3f -> %.3f ms\n", t.name, ms[0], ms[1]);
        CHECK(hipFree(dblocks));
        CHECK(hipFree(dout));
    }
    dequant_set_piece8(-1);
    std::printf("%s\n", all ? "ALL PASS" : "FAILED");
    return all ? 0 : 1;
}
