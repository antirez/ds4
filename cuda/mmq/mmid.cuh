#pragma once

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, cudaStream_t stream);

// ds4 local (P5): whether the large-n global-memory mm_ids path is enabled
// (default on; DS4_MMID_LARGE=0 reverts callers to the past-cap refusal).
bool ds4_mmid_large_enabled(void);
