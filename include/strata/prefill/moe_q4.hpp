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

/// Which product kernel `run` launches: 0 the 64 x 32 tile (the default), 1 the 256 x 32 tiles (what
/// STRATA_PF_Q4_WIDE=1 selects: the same output bits, see moe_q4.cu), -1 back to the environment's choice.  For tests
/// and benches: an A/B in one process.
void set_variant(int v);

/// STRATA_PF_DIRECT=1 (opt-in) and `enabled`: the prompt path gives `run` each expert's own blob (Product::each)
/// instead of a copy of it in the group buffer.  Such a product runs on the 256 x 32 tiles whatever STRATA_PF_Q4_WIDE says.
bool direct();

/// STRATA_PF_NL=1 (opt-in, gfx103x): IQ4_NL down products (`w_cols` 640) on the 256-row tiles instead of
/// llama.cpp's MMQ, from MMQ's own activations (mmq::quantize / swiglu_quant for that type), with MMQ's output bits
/// (see moe_q4.cu for the one exception, the sign of an exact zero).  Never reads past a matrix, so Product::each
/// works for a down matrix too.
bool nl_on();
bool nl_fits(int ggml_type, int64_t w_cols);
void nl_run(const mmq::Product& p, void* stream);
/// Tests: the token rows of a tile (32, 48 or 64); 0: the launch's own choice.
void set_nl_tile(int tj);

}  // namespace strata::prefill::q4
