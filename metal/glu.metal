#include <metal_stdlib>
using namespace metal;

/*
    DS4 GLU/SwiGLU argument block

    Represents a fused feed-forward activation stage where:
    - src0 = gate projection
    - src1 = up projection
    - dst  = fused output
*/
struct ds4_metal_args_glu {
    int32_t  ne00;   // inner feature dimension
    uint64_t nb01;   // stride src0 per token
    int32_t  ne10;   // (unused but kept for symmetry / graph alignment)
    uint64_t nb11;   // stride src1 per token
    int32_t  ne0;    // vector length per threadgroup
    uint64_t nb1;    // stride dst per token

    int32_t  i00;    // src0 column offset (tensor slicing)
    int32_t  i10;    // src1 column offset

    float    alpha;   // optional scaling factor (not used in base kernel)
    float    limit;   // optional clamp threshold (reserved for stability)
};

/*
    Numerically stable SiLU (Sigmoid Linear Unit)

    SiLU(x) = x * sigmoid(x)
            = x / (1 + exp(-x))

    This formulation is stable for FP32 inference workloads and widely used
    in transformer FFN blocks (e.g., SwiGLU / LLaMA-style MLPs).
*/
static inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

/*
    SwiGLU kernel (fused activation stage)

    Computes:
        output = SiLU(gate) * up_projection

    This is a core component of modern transformer FFNs:
    - Gate branch controls information flow
    - Up branch carries feature expansion
    - Elementwise fusion improves expressivity vs GELU/ReLU
*/
kernel void kernel_swiglu_f32(
        constant ds4_metal_args_glu & args,
        device const char * src0,
        device const char * src1,
        device char * dst,
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint ntg[[threads_per_threadgroup]]) {

    /*
        Compute row pointers (token-level addressing)

        Each threadgroup processes one sequence position (tgpig)
        Each thread processes a subset of hidden dimensions (tpitg)
    */

    device const float * gate =
        (device const float *)((device const char *)src0 + tgpig * args.nb01)
        + args.i00;

    device const float * up =
        (device const float *)((device const char *)src1 + tgpig * args.nb11)
        + args.i10;

    device float * out =
        (device float *)((device char *)dst + tgpig * args.nb1);

    /*
        Vectorized loop over hidden dimension

        - Strided by threadgroup size for coalesced memory access
        - Each thread processes multiple feature channels
    */
    for (int i = tpitg; i < args.ne0; i += ntg) {

        // Load fused FFN projections
        const float gate_val = gate[i];
        const float up_val   = up[i];

        /*
            SwiGLU activation:
            1. Apply SiLU non-linearity on gate branch
            2. Elementwise multiply with up projection
        */
        const float activated_gate = silu(gate_val);
        float result = activated_gate * up_val;

        /*
            Optional numerical stabilization hooks (reserved):

            if (args.limit > 0.0f) {
                result = clamp(result, -args.limit, args.limit);
            }

            if (args.alpha != 1.0f) {
                result *= args.alpha;
            }
        */

        out[i] = result;
    }
}
