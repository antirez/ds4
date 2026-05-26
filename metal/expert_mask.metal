// DS4 Metal expert ownership mask kernel.
// Zeros router_weights for experts not owned by this rank.
// Runs in the same command buffer as the router select kernel —
// no batch break needed for CPU masking.

struct ds4_metal_args_expert_mask {
    int32_t expert_start;
    int32_t expert_end;
    uint32_t n_expert_used;
};

// Single-token decode path: zero weights for non-owned experts.
kernel void kernel_expert_mask(
        constant ds4_metal_args_expert_mask & args,
        device const int32_t * selected [[buffer(1)]],
        device       float   * weights  [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n_expert_used) return;
    const int32_t expert_id = selected[tid];
    if (expert_id < args.expert_start || expert_id >= args.expert_end) {
        weights[tid] = 0.0f;
    }
}

// Single-token compact: rewrite selected[] and weights[] to only owned experts.
// Outputs new count to count_out[0]. Single-threaded (6 elements — not worth parallelizing).
kernel void kernel_expert_compact(
        constant ds4_metal_args_expert_mask & args,
        device       int32_t * selected  [[buffer(1)]],
        device       float   * weights   [[buffer(2)]],
        device       uint    * count_out [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    uint out_idx = 0;
    for (uint i = 0; i < args.n_expert_used; i++) {
        const int32_t expert_id = selected[i];
        if (expert_id >= args.expert_start && expert_id < args.expert_end) {
            selected[out_idx] = selected[i];
            weights[out_idx] = weights[i];
            out_idx++;
        }
    }
    count_out[0] = out_idx;
    // Zero remaining slots so kernel doesn't read garbage
    for (uint i = out_idx; i < args.n_expert_used; i++) {
        selected[i] = 0;
        weights[i] = 0.0f;
    }
}

struct ds4_metal_args_expert_mask_batch {
    int32_t expert_start;
    int32_t expert_end;
    uint32_t n_expert_used;
    uint32_t total;          // n_tokens * n_expert_used
};

// Batch (prefill) path: n_tokens * n_expert_used threads.
kernel void kernel_expert_mask_batch(
        constant ds4_metal_args_expert_mask_batch & args,
        device const int32_t * selected [[buffer(1)]],
        device       float   * weights  [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.total) return;
    const int32_t expert_id = selected[tid];
    if (expert_id < args.expert_start || expert_id >= args.expert_end) {
        weights[tid] = 0.0f;
    }
}
