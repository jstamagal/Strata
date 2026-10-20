// include/strata/kernels/head_wide.hpp - STRATA_HEAD_WIDE=1 (HIP, opt-in): the verify window's output head as one
// launch that keeps each thread's activations in registers, with the greedy pick and the non-finite check riding in
// it, and one small launch that merges the picks.  src/kernels/cuda/head_wide.cu has the details; hip_head_parity
// checks every output bit against native_mmvq, sample_tokens' greedy pick and logits_nonfinite_rows.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

inline constexpr int kHeadWideMaxCols = 8;

// STRATA_HEAD_WIDE=1 (anything else, or unset: off)
bool head_wide_enabled();
// Whether head_wide can run this head: HIP, a Q6_K matrix (ggml type 14) of 2560-wide rows in the GGUF layout,
// 1..kHeadWideMaxCols columns.
bool head_wide_supported(int ggml_type, int n_in, int n_out, int ncols);
// The device scratch head_wide needs (the workgroups' picks), for any column count.
std::size_t head_wide_scratch_bytes();

// logits[j * n_out + r] = row r of `weights` (Q6_K, GGUF layout) times column j of `x_q8_1` (native_quantize_q8_1's
// image, ncols columns): the bits native_mmvq(14, ...) writes.  out[j] (may be null) = the greedy pick of column j,
// sample_tokens' with greedy parameters and no history (the lowest index of the largest value above -inf, 0 when
// there is none).  nf_flags (may be null): kNonfiniteBlocks words per column as logits_nonfinite_rows leaves them
// read as "any word set" (word 0 holds the column's flag, the others are zero).  Device pointers (`out` and
// `nf_flags` may be mapped host memory); capturable.  Throws std::invalid_argument where head_wide_supported is false.
void head_wide(const void* weights, const void* x_q8_1, float* logits, int n_in, int n_out, int ncols, void* scratch,
               int32_t* out, uint32_t* nf_flags, void* stream);
// One draft column: the same head, then greedy merge + row_top_prob in one launch.
// The probability uses row_top_prob's 1024-thread sum and exactly the same adds.
void head_wide_draft(const void* weights, const void* x_q8_1, float* logits, int n_in, int n_out, void* scratch,
                     int32_t* out, float* probability, void* stream);

// The lab's knobs: workgroups of the persistent launch (0: the default, STRATA_HEAD_WIDE_GRID).
void head_wide_set_grid(int grid);
// The kernel variant (0: the default, STRATA_HEAD_WIDE_V; 2 or 3: see the .cu).
void head_wide_set_variant(int v);

}  // namespace strata::kernels
