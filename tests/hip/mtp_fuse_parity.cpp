// STRATA_MTP_FUSE's draft-specific pairs against their original calls, bit for bit.
// Reused graphs receive changing mapped inputs. Synthetic data, no model required.
#include <hip/hip_runtime.h>
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::kernels;
#define CHECK(call) do { const auto e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); std::exit(2); } } while (0)
namespace {
std::mt19937 rng(20261020);
void* alloc(size_t bytes) { void* p; CHECK(hipMalloc(&p, bytes)); return p; }
void upload(void* d, const void* h, size_t bytes) { CHECK(hipMemcpy(d, h, bytes, hipMemcpyHostToDevice)); }
bool equal(const void* a, const void* b, size_t bytes) {
    std::vector<uint8_t> x(bytes), y(bytes);
    CHECK(hipMemcpy(x.data(), a, bytes, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(y.data(), b, bytes, hipMemcpyDeviceToHost));
    return x == y;
}
hipGraphExec_t end_capture(hipStream_t s) {
    hipGraph_t graph; hipGraphExec_t exec;
    CHECK(hipStreamEndCapture(s, &graph));
    CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CHECK(hipGraphDestroy(graph));
    return exec;
}

bool route_plan(hipStream_t s) {
    constexpr int blob = 4096;
    auto* base = (uint8_t*) alloc(512 * blob);
    auto* logits = (float*) alloc(512 * 4);
    struct Plan { int32_t *ids, *start, *counts, *dst, *tok; float* weights; unsigned long long* ptr; };
    Plan p[2];
    for (auto& a : p) {
        a.ids = (int32_t*) alloc(16 * 4); a.weights = (float*) alloc(16 * 4);
        a.ptr = (unsigned long long*) alloc(16 * 8); a.start = (int32_t*) alloc(17 * 4);
        a.counts = (int32_t*) alloc(4 * 4); a.dst = (int32_t*) alloc(16 * 4); a.tok = (int32_t*) alloc(16 * 4);
        for (auto* v : {a.ids, a.start, a.counts, a.dst, a.tok}) CHECK(hipMemset(v, 0x5a, v == a.counts ? 16 : v == a.start ? 68 : 64));
        CHECK(hipMemset(a.weights, 0x5a, 64)); CHECK(hipMemset(a.ptr, 0x5a, 128));
    }
    CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
    native_router_top10(logits, p[0].ids, p[0].weights, s);
    moe_group_resident(p[0].ids, 10, 10, base, blob, p[0].ptr, p[0].start, p[0].counts, p[0].dst, p[0].tok, s);
    const bool took = native_router_top10_grouped(logits, p[1].ids, p[1].weights, base, blob, p[1].ptr,
                                                 p[1].start, p[1].counts, p[1].dst, p[1].tok, s);
    auto graph = end_capture(s);
    bool ok = took;
    std::vector<float> h(512);
    for (int rep = 0; rep < 1024 && ok; ++rep) {
        for (auto& x : h) x = rep % 4 == 0 ? 0.0f : rep % 4 == 1 ? float(rng() % 5) : float(int(rng() % 20001) - 10000) / 10;
        upload(logits, h.data(), 512 * 4);
        CHECK(hipGraphLaunch(graph, s)); CHECK(hipStreamSynchronize(s));
        ok = equal(p[0].ids, p[1].ids, 64) && equal(p[0].weights, p[1].weights, 64) &&
             equal(p[0].ptr, p[1].ptr, 128) && equal(p[0].start, p[1].start, 68) &&
             equal(p[0].counts, p[1].counts, 16) && equal(p[0].dst, p[1].dst, 64) && equal(p[0].tok, p[1].tok, 64);
    }
    CHECK(hipGraphExecDestroy(graph));
    for (auto& a : p) {
        for (auto* v : {a.ids, a.start, a.counts, a.dst, a.tok}) CHECK(hipFree(v));
        CHECK(hipFree(a.weights)); CHECK(hipFree(a.ptr));
    }
    CHECK(hipFree(base)); CHECK(hipFree(logits));
    std::printf("top-10 + grouped plan: 1024 changing replays, IDs/weights/plan/guards %s\n", ok ? "bitwise equal" : "DIFFERENT");
    return ok;
}

bool mapped_copies(hipStream_t s) {
    const int words[] = {8, 64, 384, 2, 81920, 4, 48};
    bool ok = true;
    int32_t *h[7], *src[7], *a[7], *b[7];
    MappedCopy copies[7];
    for (int i = 0; i < 7; ++i) {
        CHECK(hipHostMalloc((void**) &h[i], words[i] * 4, hipHostMallocMapped));
        CHECK(hipHostGetDevicePointer((void**) &src[i], h[i], 0));
        if (i == 4) src[i] = (int32_t*) alloc(words[i] * 4); // the round's residual is device memory
        a[i] = (int32_t*) alloc((words[i] + 4) * 4); b[i] = (int32_t*) alloc((words[i] + 4) * 4);
        CHECK(hipMemset(a[i], 0x5a, (words[i] + 4) * 4)); CHECK(hipMemset(b[i], 0x5a, (words[i] + 4) * 4));
        copies[i] = {b[i], src[i], words[i]};
    }
    for (int n : {2, 5, 7}) {
        CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
        for (int i = 0; i < n; ++i) {
            if (i == 4) copy_from_mapped((float*) a[i], (float*) src[i], words[i], s);
            else copy_i32_from_mapped(a[i], src[i], words[i], s);
        }
        copy_from_mapped_fused(copies, n, s);
        auto graph = end_capture(s);
        for (int rep = 0; rep < 64; ++rep) {
            for (int i = 0; i < n; ++i) for (int j = 0; j < words[i]; ++j) h[i][j] = (int32_t) rng();
            if (n > 4) upload(src[4], h[4], words[4] * 4);
            CHECK(hipGraphLaunch(graph, s)); CHECK(hipStreamSynchronize(s));
            for (int i = 0; i < n; ++i) ok &= equal(a[i], b[i], (words[i] + 4) * 4);
        }
        CHECK(hipGraphExecDestroy(graph));
    }
    for (int i = 0; i < 7; ++i) {
        CHECK(hipHostFree(h[i])); CHECK(hipFree(a[i])); CHECK(hipFree(b[i]));
        if (i == 4) CHECK(hipFree(src[i]));
    }
    std::printf("mapped copies: 2/5/7 spans, 64 changing replays each, bytes/guards %s\n", ok ? "bitwise equal" : "DIFFERENT");
    return ok;
}

bool projections(hipStream_t s) {
    constexpr int ni = 2560, max_cols = 8;
    bool ok = true;
    // Deliberately leave the process-wide SET switch OFF. Explicit calls must remain local.
    native_mmvq_set_tune(0);
    for (int type : {8, 2}) for (int small : {512, 640}) {
        const int no[] = {small, small, 12288};
        void* weights[3]; float *ref[3], *got[3]; NativeMmvqItem items[3];
        const int block_bytes = type == 8 ? 34 : 18;
        for (int i = 0; i < 3; ++i) {
            std::vector<uint8_t> w(native_mmvq_weight_bytes(type, ni, no[i]));
            for (auto& x : w) x = (uint8_t) rng();
            for (size_t off = 0; off < w.size(); off += block_bytes) {
                const uint16_t scale = (uint16_t) (0x2400 + rng() % 1024);
                std::memcpy(w.data() + off, &scale, 2);
            }
            weights[i] = alloc(w.size()); upload(weights[i], w.data(), w.size());
            ref[i] = (float*) alloc(max_cols * no[i] * 4); got[i] = (float*) alloc(max_cols * no[i] * 4);
            items[i] = {type, weights[i], got[i], no[i]};
        }
        auto* x = (float*) alloc(max_cols * ni * 4); void* xq = alloc(native_q8_1_bytes(ni, max_cols));
        std::vector<float> h(max_cols * ni);
        for (int nc = 1; nc <= max_cols; ++nc) for (int count : {2, 3}) {
            CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
            native_quantize_q8_1(x, xq, ni, nc, s);
            for (int i = 0; i < count; ++i) native_mmvq(type, weights[i], xq, ref[i], ni, no[i], nc, s);
            ok &= native_mmvq_set_explicit(items, count, xq, ni, nc, s);
            auto graph = end_capture(s);
            for (int rep = 0; rep < 8; ++rep) {
                for (auto& v : h) v = float(int(rng() % 20001) - 10000) / 10000;
                upload(x, h.data(), h.size() * 4);
                CHECK(hipGraphLaunch(graph, s)); CHECK(hipStreamSynchronize(s));
                for (int i = 0; i < count; ++i) ok &= equal(ref[i], got[i], nc * no[i] * 4);
            }
            CHECK(hipGraphExecDestroy(graph));
        }
        for (int i = 0; i < 3; ++i) { CHECK(hipFree(weights[i])); CHECK(hipFree(ref[i])); CHECK(hipFree(got[i])); }
        CHECK(hipFree(x)); CHECK(hipFree(xq));
    }
    native_mmvq_set_tune(-1);
    std::printf("explicit K/V and Q/K/V sets: Q8_0/Q4_0, 512/640 rows, 1..8 columns, 8 changing replays, floats %s\n", ok ? "bitwise equal" : "DIFFERENT");
    return ok;
}
}
int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    hipDeviceProp_t prop{}; CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("MTP fusion parity: %s (%s)\n", prop.name, prop.gcnArchName);
    if (std::strncmp(prop.gcnArchName, "gfx1030", 7) != 0) return 77;
    hipStream_t s; CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    const bool rp = route_plan(s), cp = mapped_copies(s), mm = projections(s);
    CHECK(hipStreamDestroy(s));
    std::printf("%s\n", rp && cp && mm ? "PASSED" : "FAILED");
    return rp && cp && mm ? 0 : 1;
}
