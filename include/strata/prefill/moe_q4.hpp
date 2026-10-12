// include/strata/prefill/moe_q4.hpp - the prompt path's experts for plain llama-quantize Q4_0 files (bartowski's
// Flash-Next Q4_0: gate/up Q4_0, down Q4_0 on 42 layers and Q4_1 on 6) on Strata's own int8 kernel for RDNA2
// (gfx103x, v_dot4_i32_i8).  llama.cpp's MMQ has no instance for these two types in this build, so without this
// kernel their prompt experts take the dequantize-to-FP16 + GEMM path.
//
// It serves the same contract as mmq::Product / mmq::quantize (moe_mmq.hpp) and is reached through them: the
// activation buffer is sized by mmq::q8_bytes, the group gather is mmq's, only the quantizer and the product differ.
#pragma once

#include "strata/prefill/moe_mmq.hpp"

#include <cstdint>

namespace strata::prefill::q4 {

/// ggml type 2 (Q4_0) or 3 (Q4_1).
bool covers(int ggml_type);
/// The kernel is in this build, the current device is gfx103x, and STRATA_PF_Q4 is not 0 (default on there).
bool enabled();
/// `covers`, `enabled`, and the shape is one the kernel takes (rows of any count; see run for the columns).
bool fits(int ggml_type, int64_t w_rows);

/// Row i of the output is row ids[i] of x (row i when ids is null), `cols` values (a multiple of 128) of `ld`
/// floats per row, as chunks of 128 values: {float d[4]; int8 q[128]}, one scale per 32 values, q = rint(x / d),
/// d = max |x| / 127 over its 32 values.  A row takes (cols / 128) * 144 bytes, which mmq::q8_bytes covers.
void quantize(const float* x, const int32_t* ids, void* xq, int64_t cols, int64_t ld, int64_t rows, void* stream);

/// mmq::Product for type Q4_0 or Q4_1, activations from `quantize` above.  w_cols a multiple of 128.
void run(const mmq::Product& p, void* stream);

}  // namespace strata::prefill::q4
