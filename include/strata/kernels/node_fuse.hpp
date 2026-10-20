#pragma once
// STRATA_NODE_FUSE (opt-in, off by default): fewer nodes in the verifier's window graph, every value bitwise the
// launches each fusion replaces.  =1 turns all of them on; any other value is a bit mask of the single fusions (for
// measuring one at a time):
//   2  the MoE combine of a layer inside the next layer's first hyper-connection norm (fused_gr.cu)
//   4  the QSA layer's three per-head RMSNorm + RoPE launches (k, q, indexer q) as one (native_rope.cu)
//   8  the GDN conv + L2 norm and the a/b gates inside the recurrence kernel (verify_kernels.cu)
//  16  the shared expert reads the MoE input's q8_1 image the hyper-connection read already wrote (no quantize)
//  32  the shared expert's gate and up projections in one launch (native_mmvq_set, STRATA_MMVQ_SET's kernel)
//  64  the QSA indexer tail's window snapshot made by the indexer append (no copy launch before it)
// and two existing one-launch forms the window graph did not take on HIP (their own switches turn them on as well):
// 128  the n-gram block's post-projection steps for the window in one pass (STRATA_DF_PLE=1's path)
// 256  the head's hyper-connection read for the window's tokens in one launch set (STRATA_HEAD_MIX_MULTI=1's path),
//      with STRATA_QFUSE also writing the head projection's q8_1 input
#include <cstdlib>

namespace strata::kernels {
enum : int { kNodeFuseCombine = 2, kNodeFuseNormRope = 4, kNodeFuseGdn = 8, kNodeFuseShQ8 = 16, kNodeFuseShSet = 32, kNodeFuseIdxSnap = 64,
             kNodeFusePle = 128, kNodeFuseHeadMix = 256 };
inline int node_fuse_mask() {
    static const int m = [] {
        const char* v = std::getenv("STRATA_NODE_FUSE");
        const int n = v != nullptr ? std::atoi(v) : 0;
        return n == 1 ? 0x7fffffff : n < 0 ? 0 : n;
    }();
    return m;
}
inline bool node_fuse(int bit) { return (node_fuse_mask() & bit) != 0; }
}  // namespace strata::kernels
