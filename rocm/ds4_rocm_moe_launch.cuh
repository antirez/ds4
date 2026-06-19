/* Mixed IQ2_XXS-gate/Q2_K-down models already compute routed mid activations
 * as float.  Reuse the newer Q2_K expert-batch/WMMA down kernels instead of
 * re-quantizing mid to Q8_K and taking the older qwarp down path.  This keeps
 * the CyberNeurova all-Q2 path untouched while giving the standard IQ2 mix the
 * same fast Q2 down projection used by q2k_path. */
static int routed_moe_q2_float_down_launch(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *down,
        const ds4_gpu_tensor *mid,
        const half *mid_h_hot,
        int hot_mid_f16,
        const char *down_w,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *sorted_pairs,
        uint32_t *hot_experts_dev,
        uint32_t n_tokens,
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes) {
    if (!out || !down || !mid || !down_w || !counts || !offsets || !sorted_pairs ||
        n_tokens == 0u || n_total_expert == 0u || n_total_expert > DS4_ROCM_MAX_N_EXPERT ||
        n_expert != DS4_ROCM_N_EXPERT_USED ||
        (expert_mid_dim % CUDA_QK_K) != 0u || expert_mid_dim == 0u || out_dim == 0u ||
        !cuda_tensor_has_elems3(mid, n_tokens, n_expert, expert_mid_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(down, n_tokens, n_expert, out_dim, sizeof(float)) ||
        !cuda_tensor_has_elems2(out, n_tokens, out_dim, sizeof(float))) {
        return 0;
    }

    uint32_t h_counts[DS4_ROCM_MAX_N_EXPERT] = {0};
    if (!cuda_ok(cudaMemcpy(h_counts, counts, n_total_expert * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                 "routed_moe iq2/q2 float-down counts copy")) {
        return 0;
    }

    const uint32_t down_tile = 4u;
    const uint32_t down_rpb = 16u;
    const uint32_t down_threads = down_rpb * 32u;
    const size_t down_shmem = (size_t)down_tile * 256u * sizeof(float);
    const int use_f16_down = (out_dim & 1u) == 0u;
    half *down_h = use_f16_down ? (half *)down->ptr : NULL;

    uint32_t hot_count = 0u;
    uint32_t hot_max = 0u;
    const uint32_t hot_threshold = 8u;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    const int use_wmma_hot = hot_experts_dev &&
        !g_quality_mode &&
        (expert_mid_dim % 16u) == 0u && (out_dim % 16u) == 0u;
#else
    const int use_wmma_hot = 0;
#endif
    uint32_t h_hot[DS4_ROCM_MAX_N_EXPERT] = {0};
    if (use_wmma_hot) {
        for (uint32_t e = 0; e < n_total_expert; e++) {
            const uint32_t c = h_counts[e];
            if (c >= hot_threshold) {
                h_hot[hot_count++] = e;
                if (c > hot_max) hot_max = c;
            }
        }
    }

    const uint32_t scalar_max = hot_count != 0u ? hot_threshold : 0u;
    const dim3 down_grid((out_dim + down_rpb - 1u) / down_rpb, n_total_expert, 1u);
    if (use_f16_down) {
        if (down_tile == 4u) {
            moe_down_q2K_expert_batch_sharedmid_kernel<4,false,true><<<down_grid, down_threads, down_shmem>>>(
                    NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes);
        } else if (down_tile == 8u) {
            moe_down_q2K_expert_batch_sharedmid_kernel<8,false,true><<<down_grid, down_threads, down_shmem>>>(
                    NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes);
        } else {
            moe_down_q2K_expert_batch_sharedmid_kernel<16,false,true><<<down_grid, down_threads, down_shmem>>>(
                    NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes);
        }
    } else if (down_tile == 4u) {
        moe_down_q2K_expert_batch_sharedmid_kernel<4><<<down_grid, down_threads, down_shmem>>>(
                (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                down_expert_bytes, down_row_bytes);
    } else if (down_tile == 8u) {
        moe_down_q2K_expert_batch_sharedmid_kernel<8><<<down_grid, down_threads, down_shmem>>>(
                (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                down_expert_bytes, down_row_bytes);
    } else {
        moe_down_q2K_expert_batch_sharedmid_kernel<16><<<down_grid, down_threads, down_shmem>>>(
                (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                down_expert_bytes, down_row_bytes);
    }
    if (!cuda_ok(cudaGetLastError(), "routed_moe iq2/q2 float-down scalar launch")) return 0;
    if (hot_count != 0u &&
        !cuda_ok(cudaMemcpy(hot_experts_dev, h_hot, hot_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                 "routed_moe iq2/q2 float-down hot copy")) {
        return 0;
    }

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    if (use_wmma_hot && hot_count != 0u) {
        constexpr uint32_t bm = 16u, bn = 16u, bk = 16u;
        const int no_n2 = 0;
        const uint32_t wmma_mtiles = 4u;
        if (!no_n2) {
            if (wmma_mtiles == 4u) {
                constexpr uint32_t mt = 4u;
                const dim3 block(32u * mt, 1u, 1u);
                const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                                (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
                const size_t shmem_n2 = (mt * bm * bk + 2u * bk * bn) * sizeof(half) +
                                        (2u * mt * bm * bn) * sizeof(float);
                if (use_f16_down && hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else if (use_f16_down) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else if (hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,false><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                }
            } else if (wmma_mtiles == 16u) {
                constexpr uint32_t mt = 16u;
                const dim3 block(32u * mt, 1u, 1u);
                const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                                (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
                const size_t shmem_n2 = (mt * bm * bk + 2u * bk * bn) * sizeof(half) +
                                        (2u * mt * bm * bn) * sizeof(float);
                if (use_f16_down && hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<16,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else if (use_f16_down) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<16,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else if (hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<16,16,16,16,true,false><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else {
                    moe_down_q2K_hotlist_wmma_n2_kernel<16,16,16,16><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                }
            } else {
                constexpr uint32_t mt = 8u;
                const dim3 block(32u * mt, 1u, 1u);
                const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                                (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
                const size_t shmem_n2 = (mt * bm * bk + 2u * bk * bn) * sizeof(half) +
                                        (2u * mt * bm * bn) * sizeof(float);
                if (use_f16_down && hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else if (use_f16_down) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else if (hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,false><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                } else {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
                }
            }
        } else if (wmma_mtiles == 16u) {
            constexpr uint32_t mt = 16u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid(out_dim / bn, (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
            const size_t shmem = (mt * bm * bk + bk * bn) * sizeof(half) +
                                 (mt * bm * bn) * sizeof(float);
            moe_down_q2K_hotlist_wmma_kernel<16,16,16,16><<<grid, block, shmem>>>(
                    (float *)down->ptr, down_w, (const float *)mid->ptr,
                    counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
        } else if (wmma_mtiles == 4u) {
            constexpr uint32_t mt = 4u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid(out_dim / bn, (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
            const size_t shmem = (mt * bm * bk + bk * bn) * sizeof(half) +
                                 (mt * bm * bn) * sizeof(float);
            moe_down_q2K_hotlist_wmma_kernel<4,16,16,16><<<grid, block, shmem>>>(
                    (float *)down->ptr, down_w, (const float *)mid->ptr,
                    counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
        } else {
            constexpr uint32_t mt = 8u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid(out_dim / bn, (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
            const size_t shmem = (mt * bm * bk + bk * bn) * sizeof(half) +
                                 (mt * bm * bn) * sizeof(float);
            moe_down_q2K_hotlist_wmma_kernel<8,16,16,16><<<grid, block, shmem>>>(
                    (float *)down->ptr, down_w, (const float *)mid->ptr,
                    counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
        }
        if (!cuda_ok(cudaGetLastError(), "routed_moe iq2/q2 float-down wmma launch")) return 0;
    }
#endif

    const uint64_t n = (uint64_t)n_tokens * out_dim;
    if (use_f16_down && (out_dim & 1u) == 0u) {
        const uint64_t n2 = (uint64_t)n_tokens * (out_dim >> 1u);
        moe_sum_f16x2_kernel<<<(n2 + 255u) / 256u, 256>>>(
                (float *)out->ptr, down_h, out_dim, n_expert, n_tokens);
    } else if (use_f16_down) {
        moe_sum_f16_kernel<<<(n + 255u) / 256u, 256>>>(
                (float *)out->ptr, down_h, out_dim, n_expert, n_tokens);
    } else {
        moe_sum_kernel<<<(n + 255u) / 256u, 256>>>(
                (float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
    }
    return cuda_ok(cudaGetLastError(), "routed_moe iq2/q2 float-down sum launch");
}

typedef struct {
    int q4k_path;
    int iq2_path;
    int q2k_path;
    uint64_t gate_bytes;
    uint64_t down_bytes;
} routed_moe_launch_plan;

static int routed_moe_build_plan(
        const ds4_gpu_tensor *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        const ds4_gpu_tensor *mid,
        const ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        const ds4_gpu_tensor *x,
        uint32_t n_tokens,
        routed_moe_launch_plan *plan) {
    if (!plan) return 0;
    memset(plan, 0, sizeof(*plan));
    if (!out || !gate || !up || !mid || !down || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_total_expert == 0u ||
        n_expert == 0u || n_expert > DS4_ROCM_N_EXPERT_USED ||
        expert_in_dim == 0u || expert_mid_dim == 0u || out_dim == 0u ||
        expert_in_dim % CUDA_QK_K != 0 || expert_mid_dim % CUDA_QK_K != 0 ||
        !cuda_tensor_has_elems2(x, n_tokens, expert_in_dim, sizeof(float)) ||
        !cuda_tensor_has_elems2(selected, n_tokens, n_expert, sizeof(int32_t)) ||
        !cuda_tensor_has_elems2(weights, n_tokens, n_expert, sizeof(float)) ||
        !cuda_tensor_has_elems3(gate, n_tokens, n_expert, expert_mid_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(up, n_tokens, n_expert, expert_mid_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(mid, n_tokens, n_expert, expert_mid_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(down, n_tokens, n_expert, out_dim, sizeof(float)) ||
        !cuda_tensor_has_elems2(out, n_tokens, out_dim, sizeof(float))) {
        return 0;
    }
    plan->q4k_path = (gate_type == 12u && down_type == 12u);
    plan->iq2_path = (gate_type == 16u && down_type == 10u);
    plan->q2k_path = (gate_type == 10u && down_type == 10u);
    if (!plan->q4k_path && !plan->iq2_path && !plan->q2k_path) return 0;
    if (!cuda_u64_mul_checked(n_total_expert, gate_expert_bytes, &plan->gate_bytes) ||
        !cuda_u64_mul_checked(n_total_expert, down_expert_bytes, &plan->down_bytes) ||
        !cuda_model_range_fits(model_size, gate_offset, plan->gate_bytes) ||
        !cuda_model_range_fits(model_size, up_offset, plan->gate_bytes) ||
        !cuda_model_range_fits(model_size, down_offset, plan->down_bytes)) {
        return 0;
    }
    return 1;
}

static int routed_moe_full_table_is_cached(
        const void *model_map,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_bytes,
        uint64_t down_bytes) {
    return cuda_model_range_is_cached(model_map, gate_offset, gate_bytes) &&
           cuda_model_range_is_cached(model_map, up_offset, gate_bytes) &&
           cuda_model_range_is_cached(model_map, down_offset, down_bytes);
}

static int routed_moe_launch(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float clamp,
        const ds4_gpu_tensor *x,
        uint32_t layer_index,
        uint32_t n_tokens) {
    routed_moe_launch_plan plan;
    if (!routed_moe_build_plan(out, gate, up, mid, down, model_map, model_size,
                               gate_offset, up_offset, down_offset, gate_type, down_type,
                               gate_expert_bytes, down_expert_bytes, expert_in_dim,
                               expert_mid_dim, out_dim, selected, weights, n_total_expert, n_expert, x,
                               n_tokens, &plan)) {
        return 0;
    }
    const int q4k_path = plan.q4k_path;
    const int iq2_path = plan.iq2_path;
    const int q2k_path = plan.q2k_path;
    const uint64_t gate_bytes = plan.gate_bytes;
    const uint64_t down_bytes = plan.down_bytes;
    const ds4_gpu_tensor *selected_exec = selected;
    const char *gate_w = NULL;
    const char *up_w = NULL;
    const char *down_w = NULL;
    const char **gate_slot_ptrs = NULL;
    const char **up_slot_ptrs = NULL;
    const char **down_slot_ptrs = NULL;
    const char **resident_gate_slot_ptrs = NULL;
    const char **resident_up_slot_ptrs = NULL;
    const char **missing_gate_slot_ptrs = NULL;
    const char **missing_up_slot_ptrs = NULL;
    const uint8_t *stream_batch_pair_missing = NULL;
    uint32_t stream_resident_mask = 0;
    uint32_t stream_missing_mask = 0;
    uint32_t stream_batch_unique = 0;
    uint32_t stream_batch_resident_count = 0;
    uint32_t stream_batch_missing_count = 0;
    const int stream_full_layer =
        n_tokens > 1u &&
        cuda_stream_layer_expert_cache_apply(model_map,
                                             layer_index,
                                             n_total_expert,
                                             gate_offset,
                                             up_offset,
                                             down_offset,
                                             gate_expert_bytes,
                                             down_expert_bytes,
                                             &gate_w,
                                             &up_w,
                                             &down_w);
    const int batch_stream_split_selected =
        !stream_full_layer &&
        n_tokens > 1u &&
        iq2_path &&
        n_expert == DS4_ROCM_N_EXPERT_USED &&
        cuda_stream_batch_selected_apply_split(model_map,
                                               layer_index,
                                               n_total_expert,
                                               n_expert,
                                               n_tokens,
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes,
                                               &selected_exec,
                                               &resident_gate_slot_ptrs,
                                               &resident_up_slot_ptrs,
                                               &missing_gate_slot_ptrs,
                                               &missing_up_slot_ptrs,
                                               &down_slot_ptrs,
                                               &stream_batch_pair_missing,
                                               &stream_batch_resident_count,
                                               &stream_batch_missing_count,
                                               &stream_batch_unique);
    const int batch_stream_selected =
        !stream_full_layer &&
        !batch_stream_split_selected &&
        n_tokens > 1u &&
        iq2_path &&
        n_expert == DS4_ROCM_N_EXPERT_USED &&
        cuda_stream_batch_selected_prepare(model_map,
                                           model_size,
                                           layer_index,
                                           selected,
                                           n_tokens,
                                           n_total_expert,
                                           n_expert,
                                           gate_offset,
                                           up_offset,
                                           down_offset,
                                           gate_expert_bytes,
                                           down_expert_bytes,
                                           &selected_exec,
                                           &gate_slot_ptrs,
                                           &up_slot_ptrs,
                                           &down_slot_ptrs,
                                           &stream_batch_unique);
    const int split_selected =
        n_tokens == 1u &&
        getenv("DS4_ROCM_DISABLE_STREAMING_SPLIT_SELECTED") == NULL &&
        cuda_stream_selected_apply_split(model_map,
                                         layer_index,
                                         n_total_expert,
                                         n_expert,
                                         gate_expert_bytes,
                                         down_expert_bytes,
                                         &selected_exec,
                                         &gate_w,
                                         &up_w,
                                         &down_w,
                                         &gate_slot_ptrs,
                                         &up_slot_ptrs,
                                         &down_slot_ptrs,
                                         &stream_resident_mask,
                                         &stream_missing_mask);
    const int compact_selected =
        split_selected ||
        (n_tokens == 1u &&
        cuda_stream_selected_apply(model_map,
                                   layer_index,
                                   n_total_expert,
                                   n_expert,
                                   gate_expert_bytes,
                                   down_expert_bytes,
                                   &selected_exec,
                                   &gate_w,
                                   &up_w,
                                   &down_w));
    const int full_table_cached =
        !stream_full_layer &&
        routed_moe_full_table_is_cached(model_map,
                                        gate_offset,
                                        up_offset,
                                        down_offset,
                                        gate_bytes,
                                        down_bytes);
    if (!compact_selected && !batch_stream_selected && !batch_stream_split_selected) {
        if (g_ssd_streaming_mode &&
            n_total_expert > n_expert &&
            !stream_full_layer &&
            !full_table_cached) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "SSD streaming routed MoE missing compact selected experts "
                    "(layer=%u tokens=%u total_experts=%u selected=%u); full expert table is not mapped\n",
                    layer_index,
                    n_tokens,
                    n_total_expert,
                    n_expert);
            return 0;
        }
        if (!stream_full_layer) {
            gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_bytes, "moe_gate");
            up_w = cuda_model_range_ptr(model_map, up_offset, gate_bytes, "moe_up");
            down_w = cuda_model_range_ptr(model_map, down_offset, down_bytes, "moe_down");
        }
    }
    if (batch_stream_selected || batch_stream_split_selected) {
        if (!down_slot_ptrs ||
            stream_batch_unique == 0) {
            return 0;
        }
        if (batch_stream_selected && (!gate_slot_ptrs || !up_slot_ptrs)) return 0;
        if (batch_stream_split_selected &&
            (!resident_gate_slot_ptrs || !resident_up_slot_ptrs ||
             !missing_gate_slot_ptrs || !missing_up_slot_ptrs ||
             !stream_batch_pair_missing ||
             stream_batch_missing_count == 0)) {
            return 0;
        }
    } else if (!gate_w || !up_w || !down_w) {
        return 0;
    }

    int ok = 1;
    const uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    const uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
    const uint64_t xq_count = (uint64_t)n_tokens * xq_blocks;
    const uint64_t midq_count = (uint64_t)n_tokens * n_expert * midq_blocks;
    const uint64_t xq_bytes = xq_count * sizeof(cuda_block_q8_K);
    const uint64_t midq_bytes = midq_count * sizeof(cuda_block_q8_K);
    if (!q2k_path && down->bytes >= xq_bytes && gate->bytes >= midq_bytes) {
        cuda_block_q8_K *xq = (cuda_block_q8_K *)down->ptr;
        cuda_block_q8_K *midq = (cuda_block_q8_K *)gate->ptr;
        const uint32_t pair_count = n_tokens * n_expert;
        const uint32_t use_sorted_pairs = n_tokens > 1u &&
            (!q4k_path || n_tokens >= 32u);
        const uint32_t use_expert_tiles = use_sorted_pairs;
        const uint32_t expert_tile_m = 8u;
        const uint32_t write_gate_up = 0u;
        const uint32_t use_p2_sorted = 0u;
        const uint32_t use_atomic_down = use_expert_tiles && n_tokens >= 128u;
        const uint32_t use_gate_row2048 = !q4k_path && use_expert_tiles && n_tokens >= 128u;
        const uint32_t use_down_tile16 = !q4k_path && use_atomic_down && n_tokens >= 128u;
        const uint32_t use_decode_lut_gate =
            n_tokens == 1u && xq_blocks <= 16u;
        const uint32_t gate_row_span = 1024u;
        const uint32_t down_row_span = 1024u;
        const uint32_t use_down_row2048 = !q4k_path && use_atomic_down && use_down_tile16;
        const uint32_t use_direct_down_sum6 =
            n_tokens == 1u && n_expert == 6u;
        uint32_t *sorted_pairs = NULL;
        uint32_t *sorted_offsets = NULL;
        uint32_t *sorted_counts = NULL;
        uint32_t *tile_total = NULL;
        uint32_t *tile_experts = NULL;
        uint32_t *tile_starts = NULL;
        uint32_t *tile16_total = NULL;
        uint32_t *tile16_experts = NULL;
        uint32_t *tile16_starts = NULL;
        uint32_t *iq2_gate_hot_dev = NULL;
        uint32_t tile_capacity = 0;
        uint32_t tile16_capacity = 0;
        dim3 xq_grid(xq_blocks, n_tokens, 1);
        q8_K_quantize_kernel<<<xq_grid, 256>>>(xq, (const float *)x->ptr, expert_in_dim, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "routed_moe x quantize launch");
        if (ok && (batch_stream_selected || batch_stream_split_selected)) {
            dim3 qgrid((expert_mid_dim + 127u) / 128u, pair_count, 1);
            if (batch_stream_split_selected) {
                if (stream_batch_resident_count != 0u) {
                    moe_gate_up_mid_qwarp32_ptrs_split_kernel<<<qgrid, 256>>>(
                            (float *)gate->ptr,
                            (float *)up->ptr,
                            (float *)mid->ptr,
                            resident_gate_slot_ptrs,
                            resident_up_slot_ptrs,
                            stream_batch_pair_missing,
                            0u,
                            xq,
                            (const int32_t *)selected_exec->ptr,
                            (const float *)weights->ptr,
                            gate_row_bytes,
                            xq_blocks,
                            expert_mid_dim,
                            n_expert,
                            0xffffffffu,
                            clamp);
                    ok = cuda_ok(cudaGetLastError(),
                                 "routed_moe streaming batch resident gate/up launch");
                }
                if (!ok) {
                    (void)cuda_stream_batch_selected_finish_pending_missing();
                } else {
                    ok = cuda_stream_batch_selected_finish_pending_missing();
                }
                if (ok && stream_batch_missing_count != 0u) {
                    moe_gate_up_mid_qwarp32_ptrs_split_kernel<<<qgrid, 256>>>(
                            (float *)gate->ptr,
                            (float *)up->ptr,
                            (float *)mid->ptr,
                            missing_gate_slot_ptrs,
                            missing_up_slot_ptrs,
                            stream_batch_pair_missing,
                            1u,
                            xq,
                            (const int32_t *)selected_exec->ptr,
                            (const float *)weights->ptr,
                            gate_row_bytes,
                            xq_blocks,
                            expert_mid_dim,
                            n_expert,
                            0xffffffffu,
                            clamp);
                    ok = cuda_ok(cudaGetLastError(),
                                 "routed_moe streaming batch missing gate/up launch");
                }
            } else {
                moe_gate_up_mid_qwarp32_ptrs_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        0xffffffffu,
                        clamp);
                ok = cuda_ok(cudaGetLastError(),
                             "routed_moe streaming batch gate/up launch");
            }
            if (ok) {
                dim3 midq_grid(midq_blocks, pair_count, 1);
                q8_K_quantize_kernel<<<midq_grid, 256>>>(
                        midq,
                        (const float *)mid->ptr,
                        expert_mid_dim,
                        pair_count);
                ok = cuda_ok(cudaGetLastError(), "routed_moe streaming batch mid quantize launch");
            }
            if (ok) {
                dim3 dgrid((out_dim + 31u) / 32u, n_tokens, 1);
                moe_down_sum6_qwarp32_ptrs_batch_kernel<<<dgrid, 256>>>(
                        (float *)out->ptr,
                        down_slot_ptrs,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert,
                        n_tokens);
                ok = cuda_ok(cudaGetLastError(), "routed_moe streaming batch down launch");
            }
            return ok;
        }
        if (ok && use_sorted_pairs) {
            const uint32_t bucket_count = n_total_expert;
            const uint64_t counts_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
            const uint64_t offsets_bytes = (uint64_t)(bucket_count + 1u) * sizeof(uint32_t);
            const uint64_t cursors_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
            const uint64_t sorted_bytes = (uint64_t)pair_count * sizeof(uint32_t);
            tile_capacity = (pair_count + expert_tile_m - 1u) / expert_tile_m + bucket_count;
            tile16_capacity = use_down_tile16 ? ((pair_count + 15u) / 16u + bucket_count) : 0u;
            const uint64_t tile_offsets_bytes = (uint64_t)(bucket_count + 1u) * sizeof(uint32_t);
            const uint64_t tile_total_bytes = sizeof(uint32_t);
            const uint64_t tile_experts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile_starts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile16_offsets_bytes = use_down_tile16 ? (uint64_t)(bucket_count + 1u) * sizeof(uint32_t) : 0u;
            const uint64_t tile16_total_bytes = use_down_tile16 ? sizeof(uint32_t) : 0u;
            const uint64_t tile16_experts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile16_starts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile_offsets_off = counts_bytes + offsets_bytes + cursors_bytes + sorted_bytes;
            const uint64_t tile_total_off = tile_offsets_off + tile_offsets_bytes;
            const uint64_t tile_experts_off = tile_total_off + tile_total_bytes;
            const uint64_t tile_starts_off = tile_experts_off + tile_experts_bytes;
            const uint64_t tile16_offsets_off = tile_starts_off + tile_starts_bytes;
            const uint64_t tile16_total_off = tile16_offsets_off + tile16_offsets_bytes;
            const uint64_t tile16_experts_off = tile16_total_off + tile16_total_bytes;
            const uint64_t tile16_starts_off = tile16_experts_off + tile16_experts_bytes;
            const uint64_t iq2_gate_hot_off = tile16_starts_off + tile16_starts_bytes;
            const uint64_t iq2_gate_hot_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
            const uint64_t scratch_bytes = iq2_gate_hot_off + iq2_gate_hot_bytes;
            uint8_t *scratch = (uint8_t *)cuda_tmp_alloc(scratch_bytes,
                                                         "routed_moe sorted pairs");
            if (!scratch) {
                ok = 0;
            } else {
                uint32_t *counts = (uint32_t *)scratch;
                uint32_t *offsets = (uint32_t *)(scratch + counts_bytes);
                uint32_t *cursors = (uint32_t *)(scratch + counts_bytes + offsets_bytes);
                sorted_pairs = (uint32_t *)(scratch + counts_bytes + offsets_bytes + cursors_bytes);
                sorted_offsets = offsets;
                sorted_counts = counts;
                uint32_t *tile_offsets = (uint32_t *)(scratch + tile_offsets_off);
                tile_total = (uint32_t *)(scratch + tile_total_off);
                tile_experts = (uint32_t *)(scratch + tile_experts_off);
                tile_starts = (uint32_t *)(scratch + tile_starts_off);
                uint32_t *tile16_offsets = use_down_tile16 ? (uint32_t *)(scratch + tile16_offsets_off) : NULL;
                tile16_total = use_down_tile16 ? (uint32_t *)(scratch + tile16_total_off) : NULL;
                tile16_experts = use_down_tile16 ? (uint32_t *)(scratch + tile16_experts_off) : NULL;
                tile16_starts = use_down_tile16 ? (uint32_t *)(scratch + tile16_starts_off) : NULL;
                iq2_gate_hot_dev = (uint32_t *)(scratch + iq2_gate_hot_off);
                ok = cuda_ok(cudaMemset(counts, 0, counts_bytes), "routed_moe sorted counts clear");
                if (ok) {
                    moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                        counts,
                        (const int32_t *)selected_exec->ptr,
                        pair_count,
                        bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted count launch");
                }
                if (ok) {
                    moe_prefix_sorted_pairs_kernel<<<1, 1>>>(offsets, cursors, counts, bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted prefix launch");
                }
                if (ok) {
                    moe_scatter_sorted_pairs_deterministic_kernel<<<bucket_count, 1u>>>(
                        sorted_pairs,
                        offsets,
                        (const int32_t *)selected_exec->ptr,
                        pair_count,
                        bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted scatter launch");
                }
                if (ok && use_expert_tiles) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile_offsets, tile_total, counts, expert_tile_m, bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile offsets launch");
                }
                if (ok && use_expert_tiles) {
                    moe_build_expert_tiles_kernel<<<(bucket_count + 255u) / 256u, 256>>>(
                            tile_experts, tile_starts, tile_offsets, counts, expert_tile_m, bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tiles launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile16_offsets, tile16_total, counts, 16u, bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 offsets launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tiles_kernel<<<(bucket_count + 255u) / 256u, 256>>>(
                            tile16_experts, tile16_starts, tile16_offsets, counts, 16u, bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 launch");
                }
            }
        }
        uint32_t iq2_gate_hot_count = 0u;
        uint32_t iq2_gate_hot_max = 0u;
        const uint32_t iq2_gate_hot_threshold = 8u;
        const uint32_t iq2_down_hot_threshold = 8u;
        uint32_t h_iq2_gate_hot[DS4_ROCM_MAX_N_EXPERT] = {0};
        const uint32_t use_iq2_gate_wmma =
            ok && iq2_path && n_tokens > 1u && n_expert == 6u && !write_gate_up &&
            sorted_pairs && sorted_offsets && sorted_counts && tile_experts && iq2_gate_hot_dev && use_expert_tiles &&
            (expert_in_dim % 16u) == 0u && (expert_mid_dim % 16u) == 0u &&
            !g_quality_mode;
        if (use_iq2_gate_wmma) {
            uint32_t h_counts[DS4_ROCM_MAX_N_EXPERT] = {0};
            if (!cuda_ok(cudaMemcpy(h_counts, sorted_counts, n_total_expert * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                         "routed_moe iq2 gate wmma counts copy")) {
                ok = 0;
            } else {
                for (uint32_t e = 0; e < n_total_expert; e++) {
                    const uint32_t c = h_counts[e];
                    if (c >= iq2_gate_hot_threshold) {
                        h_iq2_gate_hot[iq2_gate_hot_count++] = e;
                        if (c > iq2_gate_hot_max) iq2_gate_hot_max = c;
                    }
                }
                if (iq2_gate_hot_count != 0u &&
                    !cuda_ok(cudaMemcpy(iq2_gate_hot_dev, h_iq2_gate_hot,
                                        iq2_gate_hot_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                             "routed_moe iq2 gate hot copy")) {
                    ok = 0;
                }
            }
        }
        const uint32_t iq2_gate_scalar_max = iq2_gate_hot_count != 0u ? iq2_gate_hot_threshold : 0u;
        const int use_iq2_hot_f16_mid = use_iq2_gate_wmma && iq2_gate_hot_count != 0u &&
            iq2_gate_hot_threshold == iq2_down_hot_threshold && (out_dim & 1u) == 0u &&
            !g_quality_mode;
        half *iq2_hot_mid_h = use_iq2_hot_f16_mid ? (half *)gate->ptr : NULL;
        const int use_iq2_x_f16 = use_iq2_gate_wmma && iq2_gate_hot_count != 0u &&
            up->bytes >= (uint64_t)n_tokens * expert_in_dim * sizeof(half);
        half *iq2_x_h = use_iq2_x_f16 ? (half *)up->ptr : NULL;
        if (ok && use_iq2_x_f16) {
            const uint64_t xh_count = (uint64_t)n_tokens * expert_in_dim;
            f32_to_f16_kernel<<<(xh_count + 255u) / 256u, 256>>>(iq2_x_h, (const float *)x->ptr, xh_count);
            ok = cuda_ok(cudaGetLastError(), "routed_moe iq2 gate x f16 launch");
        }
        int split_gateup_done = 0;
        if (ok && split_selected) {
            const int split_supported =
                iq2_path &&
                n_tokens == 1u &&
                n_expert == DS4_ROCM_N_EXPERT_USED &&
                !q4k_path &&
                !sorted_pairs &&
                stream_resident_mask != 0 &&
                stream_missing_mask != 0;
            if (split_supported) {
                dim3 qgrid((expert_mid_dim + 127u) / 128u, n_tokens * n_expert, 1);
                if (use_decode_lut_gate) {
                    moe_gate_up_mid_decode_lut_qwarp32_ptrs_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        stream_resident_mask,
                        clamp);
                } else {
                    moe_gate_up_mid_qwarp32_ptrs_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        stream_resident_mask,
                        clamp);
                }
                ok = cuda_ok(cudaGetLastError(), "routed_moe split resident gate/up launch");
                if (!ok) {
                    (void)cuda_stream_selected_finish_pending_missing(0);
                } else {
                    ok = cuda_stream_selected_finish_pending_missing(0);
                }
                if (ok && use_decode_lut_gate) {
                    moe_gate_up_mid_decode_lut_qwarp32_ptrs_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        stream_missing_mask,
                        clamp);
                } else if (ok) {
                    moe_gate_up_mid_qwarp32_ptrs_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        stream_missing_mask,
                        clamp);
                }
                if (ok) ok = cuda_ok(cudaGetLastError(), "routed_moe split missing gate/up launch");
                split_gateup_done = ok;
            } else {
                ok = cuda_stream_selected_finish_pending_missing(
                        stream_resident_mask | stream_missing_mask);
            }
        }
        if (ok && !split_gateup_done) {
            dim3 mgrid((expert_mid_dim + 31u) / 32u, n_tokens * n_expert, 1);
            if (ok && sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts && tile_total && tile_experts && tile_starts) {
                if (q4k_path) {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    if (expert_tile_m == 8u) {
                        moe_gate_up_mid_q4K_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            0u, write_gate_up, clamp);
                    } else {
                        moe_gate_up_mid_q4K_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            0u, write_gate_up, clamp);
                    }
                } else if (use_gate_row2048) {
                    if (gate_row_span == 512u) {
                        dim3 tgrid((expert_mid_dim + 511u) / 512u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<512><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            iq2_gate_scalar_max, write_gate_up, clamp);
                    } else if (gate_row_span == 1024u) {
                        dim3 tgrid((expert_mid_dim + 1023u) / 1024u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<1024><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            iq2_gate_scalar_max, write_gate_up, clamp);
                    } else {
                        dim3 tgrid((expert_mid_dim + 2047u) / 2048u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_row2048_kernel<<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            iq2_gate_scalar_max, write_gate_up, clamp);
                    }
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    moe_gate_up_mid_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        iq2_gate_scalar_max, write_gate_up, clamp);
                } else {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    moe_gate_up_mid_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        iq2_gate_scalar_max, write_gate_up, clamp);
                }
            } else if (ok && sorted_pairs && use_p2_sorted) {
                dim3 p2_mgrid((expert_mid_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_gate_up_mid_sorted_p2_qwarp32_kernel<<<p2_mgrid, 256>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    sorted_pairs,
                    (const int32_t *)selected_exec->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    xq_blocks,
                    expert_mid_dim,
                    n_expert,
                    pair_count,
                    clamp);
            } else if (ok && sorted_pairs) {
                if (q4k_path) {
                    moe_gate_up_mid_q4K_sorted_qwarp32_kernel<<<mgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        sorted_pairs,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                } else {
                    moe_gate_up_mid_sorted_qwarp32_kernel<<<mgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        sorted_pairs,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                }
            } else if (ok) {
                dim3 qgrid((expert_mid_dim + 127u) / 128u, n_tokens * n_expert, 1);
                if (q4k_path) {
                    moe_gate_up_mid_decode_q4K_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else if (use_decode_lut_gate) {
                    moe_gate_up_mid_decode_lut_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        0xffffffffu,
                        clamp);
                } else {
                    moe_gate_up_mid_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        0xffffffffu,
                        clamp);
                }
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
            if (ok && use_iq2_gate_wmma && iq2_gate_hot_count != 0u) {
                constexpr uint32_t bm = 16u, bn = 16u, bk = 16u;
                const uint32_t wmma_mtiles = 4u;
                if (wmma_mtiles == 4u) {
                    constexpr uint32_t mt = 4u;
                    const dim3 block(32u * mt, 1u, 1u);
                    const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                                    (iq2_gate_hot_max + mt * bm - 1u) / (mt * bm),
                                    iq2_gate_hot_count);
                    const size_t shmem_n2 = (mt * bm * bk + 4u * bk * bn) * sizeof(half) +
                                            (4u * mt * bm * bn) * sizeof(float);
                    if (use_iq2_hot_f16_mid && use_iq2_x_f16) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<4,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                                NULL, iq2_hot_mid_h, gate_w, up_w, (const float *)x->ptr, iq2_x_h,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else if (use_iq2_hot_f16_mid) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<4,16,16,16,true><<<grid, block, shmem_n2>>>(
                                NULL, iq2_hot_mid_h, gate_w, up_w, (const float *)x->ptr, NULL,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else if (use_iq2_x_f16) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<4,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, iq2_x_h,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<4,16,16,16><<<grid, block, shmem_n2>>>(
                                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, NULL,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    }
                } else {
                    constexpr uint32_t mt = 8u;
                    const dim3 block(32u * mt, 1u, 1u);
                    const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                                    (iq2_gate_hot_max + mt * bm - 1u) / (mt * bm),
                                    iq2_gate_hot_count);
                    const size_t shmem_n2 = (mt * bm * bk + 4u * bk * bn) * sizeof(half) +
                                            (4u * mt * bm * bn) * sizeof(float);
                    if (use_iq2_hot_f16_mid && use_iq2_x_f16) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<8,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                                NULL, iq2_hot_mid_h, gate_w, up_w, (const float *)x->ptr, iq2_x_h,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else if (use_iq2_hot_f16_mid) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<8,16,16,16,true><<<grid, block, shmem_n2>>>(
                                NULL, iq2_hot_mid_h, gate_w, up_w, (const float *)x->ptr, NULL,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else if (use_iq2_x_f16) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<8,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, iq2_x_h,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<8,16,16,16><<<grid, block, shmem_n2>>>(
                                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, NULL,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    }
                }
                ok = cuda_ok(cudaGetLastError(), "routed_moe iq2 wmma hot gate/up launch");
            }
#endif
        }
        const uint32_t use_iq2_q2_float_down =
            ok && iq2_path && n_tokens > 1u && n_expert == 6u &&
            sorted_pairs && sorted_offsets && sorted_counts && tile_experts;
        if (ok && !use_iq2_q2_float_down) {
            dim3 midq_grid(midq_blocks, n_tokens * n_expert, 1);
            q8_K_quantize_kernel<<<midq_grid, 256>>>(midq, (const float *)mid->ptr, expert_mid_dim, n_tokens * n_expert);
            ok = cuda_ok(cudaGetLastError(), "routed_moe mid quantize launch");
        }
        int split_ptr_down_done = 0;
        if (ok && split_gateup_done) {
            moe_down_sum6_qwarp32_ptrs_kernel<<<(out_dim + 31u) / 32u, 256>>>(
                    (float *)out->ptr,
                    down_slot_ptrs,
                    midq,
                    down_row_bytes,
                    midq_blocks,
                    out_dim);
            ok = cuda_ok(cudaGetLastError(), "routed_moe split ptr down launch");
            split_ptr_down_done = ok;
        }
        if (ok) {
            if (split_ptr_down_done) {
                /* The split pointer-table path writes the final token row. */
            } else if (use_iq2_q2_float_down) {
                ok = routed_moe_q2_float_down_launch(
                        out, down, mid, iq2_hot_mid_h, use_iq2_hot_f16_mid, down_w,
                        sorted_counts, sorted_offsets, sorted_pairs, tile_experts,
                        n_tokens, n_total_expert, n_expert, expert_mid_dim, out_dim,
                        down_expert_bytes, down_row_bytes);
            } else {
            dim3 dgrid((out_dim + 31u) / 32u, n_tokens * n_expert, 1);
            uint32_t *down_tile_total = tile_total;
            uint32_t *down_tile_experts = tile_experts;
            uint32_t *down_tile_starts = tile_starts;
            uint32_t down_tile_capacity = tile_capacity;
            if (use_down_tile16 && tile16_total && tile16_experts && tile16_starts) {
                down_tile_total = tile16_total;
                down_tile_experts = tile16_experts;
                down_tile_starts = tile16_starts;
                down_tile_capacity = tile16_capacity;
            }
            if (use_direct_down_sum6) {
                dim3 sgrid((out_dim + 31u) / 32u, 1, 1);
                if (q4k_path) {
                    moe_down_q4K_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim);
                } else {
                    moe_down_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim);
                }
            } else if (use_atomic_down) {
                uint64_t n = (uint64_t)n_tokens * out_dim;
                zero_kernel<<<(n + 255u) / 256u, 256>>>((float *)out->ptr, n);
                ok = cuda_ok(cudaGetLastError(), "routed_moe atomic zero launch");
            }
            if (use_direct_down_sum6) {
                /* The direct decode kernel writes the final token row. */
            } else if (sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts &&
                down_tile_total && down_tile_experts && down_tile_starts) {
                if (q4k_path) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    if (expert_tile_m == 8u) {
                        moe_down_q4K_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else {
                        moe_down_q4K_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    }
                } else if (use_down_row2048) {
                    if (down_row_span == 512u) {
                        dim3 tgrid((out_dim + 511u) / 512u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<512><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else if (down_row_span == 1024u) {
                        dim3 tgrid((out_dim + 1023u) / 1024u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<1024><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else {
                        dim3 tgrid((out_dim + 2047u) / 2048u, down_tile_capacity, 1);
                        moe_down_expert_tile16_row2048_kernel<<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    }
                } else if (use_down_tile16) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile16_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                }
            } else if (sorted_pairs && use_p2_sorted) {
                dim3 p2_dgrid((out_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_down_sorted_p2_qwarp32_kernel<<<p2_dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    sorted_pairs,
                    (const int32_t *)selected_exec->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert,
                    pair_count);
            } else if (sorted_pairs) {
                if (q4k_path) {
                    moe_down_q4K_sorted_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        sorted_pairs,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                } else {
                    moe_down_sorted_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        sorted_pairs,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                }
            } else {
                if (q4k_path) {
                    moe_down_q4K_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                } else {
                    moe_down_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                }
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
            }
        }
        if (ok && !use_atomic_down && !use_direct_down_sum6 && !use_iq2_q2_float_down) {
            uint64_t n = (uint64_t)n_tokens * out_dim;
            moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
            ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
        }
        if (ok && compact_selected) ok = cuda_stream_selected_mark_inflight();
        return ok;
    }

    if (q2k_path && n_expert == 6u && n_tokens >= 32u && !cuda_runtime_config()->graph_dump) {
        const uint32_t pair_count = n_tokens * n_expert;
        const uint32_t bucket_count = n_total_expert;
        const uint64_t counts_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
        const uint64_t offsets_bytes = (uint64_t)(bucket_count + 1u) * sizeof(uint32_t);
        const uint64_t cursors_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
        const uint64_t sorted_bytes = (uint64_t)pair_count * sizeof(uint32_t);
        const uint64_t hot_gate_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
        const uint64_t hot_down_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
        const uint64_t f16_low_gate_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
        const uint64_t f16_low_down_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        const int moe_wmma_hot = !g_quality_mode &&
                                 expert_in_dim % 16u == 0u &&
                                 expert_mid_dim % 16u == 0u &&
                                 out_dim % 16u == 0u;
#else
        const int moe_wmma_hot = 0;
#endif
        const uint64_t f16_mid_bytes = moe_wmma_hot ? (uint64_t)pair_count * expert_mid_dim * sizeof(__half) : 0ull;
        const uint64_t f16_down_bytes = moe_wmma_hot ? (uint64_t)pair_count * out_dim * sizeof(__half) : 0ull;
        const uint64_t wmma_x_bytes = moe_wmma_hot ? (uint64_t)n_tokens * expert_in_dim * sizeof(__half) : 0ull;
        auto align256 = [](uint64_t v) -> uint64_t { return (v + 255ull) & ~255ull; };
        const uint64_t base_scratch_end = counts_bytes + offsets_bytes + cursors_bytes + sorted_bytes +
                                          hot_gate_bytes + hot_down_bytes +
                                          f16_low_gate_bytes + f16_low_down_bytes;
        const uint64_t f16_mid_off = align256(base_scratch_end);
        const uint64_t f16_down_off = align256(f16_mid_off + f16_mid_bytes);
        const uint64_t wmma_x_off = align256(f16_down_off + f16_down_bytes);
        const uint64_t scratch_bytes = align256(wmma_x_off + wmma_x_bytes);
        uint8_t *scratch = (uint8_t *)cuda_tmp_alloc(scratch_bytes, "routed_moe q2 expert batch buckets");
        if (!scratch) return 0;
        uint32_t *counts = (uint32_t *)scratch;
        uint32_t *offsets = (uint32_t *)(scratch + counts_bytes);
        uint32_t *cursors = (uint32_t *)(scratch + counts_bytes + offsets_bytes);
        uint32_t *sorted_pairs = (uint32_t *)(scratch + counts_bytes + offsets_bytes + cursors_bytes);
        const uint64_t wmma_list_base = counts_bytes + offsets_bytes + cursors_bytes + sorted_bytes;
        uint32_t *wmma_gate_hot_dev = (uint32_t *)(scratch + wmma_list_base);
        uint32_t *wmma_down_hot_dev = (uint32_t *)(scratch + wmma_list_base + hot_gate_bytes);
        uint32_t *wmma_gate_f16_low_dev = (uint32_t *)(scratch + wmma_list_base + hot_gate_bytes + hot_down_bytes);
        uint32_t *wmma_down_f16_low_dev = (uint32_t *)(scratch + wmma_list_base + hot_gate_bytes + hot_down_bytes + f16_low_gate_bytes);
        __half *wmma_mid_h = moe_wmma_hot ? (__half *)(scratch + f16_mid_off) : NULL;
        __half *wmma_down_h = moe_wmma_hot ? (__half *)(scratch + f16_down_off) : NULL;
        __half *wmma_x_h = moe_wmma_hot ? (__half *)(scratch + wmma_x_off) : NULL;
        ok = cuda_ok(cudaMemset(counts, 0, counts_bytes), "routed_moe q2 expert counts clear");
        if (ok) {
            moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                    counts,
                    (const int32_t *)selected_exec->ptr,
                    pair_count,
                    bucket_count);
            ok = cuda_ok(cudaGetLastError(), "routed_moe q2 expert count launch");
        }
        if (ok) {
            moe_prefix_sorted_pairs_kernel<<<1, 1>>>(offsets, cursors, counts, bucket_count);
            ok = cuda_ok(cudaGetLastError(), "routed_moe q2 expert prefix launch");
        }
        if (ok) {
            moe_scatter_sorted_pairs_deterministic_kernel<<<bucket_count, 1u>>>(
                    sorted_pairs,
                    offsets,
                    (const int32_t *)selected_exec->ptr,
                    pair_count,
                    bucket_count);
            ok = cuda_ok(cudaGetLastError(), "routed_moe q2 expert scatter launch");
        }
        if (ok && moe_wmma_hot) {
            const uint64_t xh_count = (uint64_t)n_tokens * expert_in_dim;
            f32_to_f16_kernel<<<(xh_count + 255u) / 256u, 256>>>(wmma_x_h, (const float *)x->ptr, xh_count);
            ok = cuda_ok(cudaGetLastError(), "routed_moe q2 wmma x f16 launch");
        }
        if (!ok) return 0;

        uint32_t wmma_f16_hot_count = 0u, wmma_f16_hot_max = 0u;
        uint32_t wmma_f16_low_count = 0u, wmma_f16_low_max = 0u;
        uint32_t h_counts[DS4_ROCM_MAX_N_EXPERT] = {0};
        uint32_t h_f16_hot[DS4_ROCM_MAX_N_EXPERT] = {0};
        uint32_t h_f16_low[DS4_ROCM_MAX_N_EXPERT] = {0};
        const uint32_t wmma_hot_threshold = 8u;
        const uint32_t wmma_f16_low_threshold = 64u;
        if (moe_wmma_hot) {
            if (!cuda_ok(cudaMemcpy(h_counts, counts, bucket_count * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                         "routed_moe q2 wmma counts copy")) return 0;
            for (uint32_t e = 0; e < bucket_count; e++) {
                const uint32_t c = h_counts[e];
                if (c >= wmma_hot_threshold) {
                    if (c < wmma_f16_low_threshold) {
                        h_f16_low[wmma_f16_low_count++] = e;
                        if (c > wmma_f16_low_max) wmma_f16_low_max = c;
                    } else {
                        h_f16_hot[wmma_f16_hot_count++] = e;
                        if (c > wmma_f16_hot_max) wmma_f16_hot_max = c;
                    }
                }
            }
        }
        const uint32_t gate_rpb = 16u;
        const uint32_t down_rpb = 16u;
        const uint32_t gate_threads = gate_rpb * 32u;
        const uint32_t down_threads = down_rpb * 32u;
        const size_t gate_shmem = 4u * 256u * sizeof(float);
        const size_t down_shmem = 4u * 256u * sizeof(float);
        const uint32_t scalar_max = moe_wmma_hot && (wmma_f16_low_count != 0u || wmma_f16_hot_count != 0u)
            ? wmma_hot_threshold : 0u;
        dim3 gate_grid((expert_mid_dim + gate_rpb - 1u) / gate_rpb, bucket_count, 1);
        moe_gate_up_mid_q2K_expert_batch_sharedx_kernel<4><<<gate_grid, gate_threads, gate_shmem>>>(
                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, (const float *)weights->ptr,
                counts, offsets, sorted_pairs, 1u, scalar_max, expert_in_dim, expert_mid_dim,
                gate_expert_bytes, gate_row_bytes, clamp);
        if (!cuda_ok(cudaGetLastError(), "routed_moe q2 expert gate/up launch")) return 0;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        if (moe_wmma_hot && wmma_f16_low_count != 0u) {
            constexpr uint32_t mt4 = 4u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt4, 1u, 1u);
            const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_low_max + mt4 * bm - 1u) / (mt4 * bm),
                            wmma_f16_low_count);
            const size_t shmem_n2 = (mt4 * bm * bk + 4u * bk * bn) * sizeof(half) +
                                    (4u * mt4 * bm * bn) * sizeof(float);
            if (!cuda_ok(cudaMemcpy(wmma_gate_f16_low_dev, h_f16_low,
                                    wmma_f16_low_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-low hot copy")) return 0;
            moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                    NULL, wmma_mid_h, gate_w, up_w, (const float *)x->ptr, wmma_x_h, (const float *)weights->ptr,
                    counts, offsets, sorted_pairs, wmma_gate_f16_low_dev, wmma_f16_low_count,
                    expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, clamp);
            if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma f16-low gate/up launch")) return 0;
        }
        if (moe_wmma_hot && wmma_f16_hot_count != 0u) {
            constexpr uint32_t mt = 8u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_hot_max + mt * bm - 1u) / (mt * bm),
                            wmma_f16_hot_count);
            const size_t shmem_n2 = (mt * bm * bk + 4u * bk * bn) * sizeof(half) +
                                    (4u * mt * bm * bn) * sizeof(float);
            if (!cuda_ok(cudaMemcpy(wmma_gate_hot_dev, h_f16_hot,
                                    wmma_f16_hot_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-mid hot copy")) return 0;
            moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                    NULL, wmma_mid_h, gate_w, up_w, (const float *)x->ptr, wmma_x_h, (const float *)weights->ptr,
                    counts, offsets, sorted_pairs, wmma_gate_hot_dev, wmma_f16_hot_count,
                    expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, clamp);
            if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma f16-mid gate/up launch")) return 0;
        }
#endif
        dim3 down_grid((out_dim + down_rpb - 1u) / down_rpb, bucket_count, 1);
        if (moe_wmma_hot) {
            moe_down_q2K_expert_batch_sharedmid_kernel<4,false,true><<<down_grid, down_threads, down_shmem>>>(
                    NULL, wmma_down_h, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes);
        } else {
            moe_down_q2K_expert_batch_sharedmid_kernel<4><<<down_grid, down_threads, down_shmem>>>(
                    (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes);
        }
        if (!cuda_ok(cudaGetLastError(), "routed_moe q2 expert down launch")) return 0;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        if (moe_wmma_hot && wmma_f16_low_count != 0u) {
            constexpr uint32_t mt4 = 4u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt4, 1u, 1u);
            const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_low_max + mt4 * bm - 1u) / (mt4 * bm),
                            wmma_f16_low_count);
            const size_t shmem_n2 = (mt4 * bm * bk + 2u * bk * bn) * sizeof(half) +
                                    (2u * mt4 * bm * bn) * sizeof(float);
            if (!cuda_ok(cudaMemcpy(wmma_down_f16_low_dev, h_f16_low,
                                    wmma_f16_low_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-low down hot copy")) return 0;
            moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                    NULL, wmma_down_h, down_w, NULL, wmma_mid_h,
                    counts, offsets, sorted_pairs, wmma_down_f16_low_dev, wmma_f16_low_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
            if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma f16-low down launch")) return 0;
        }
        if (moe_wmma_hot && wmma_f16_hot_count != 0u) {
            constexpr uint32_t mt = 8u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_hot_max + mt * bm - 1u) / (mt * bm),
                            wmma_f16_hot_count);
            const size_t shmem_n2 = (mt * bm * bk + 2u * bk * bn) * sizeof(half) +
                                    (2u * mt * bm * bn) * sizeof(float);
            if (!cuda_ok(cudaMemcpy(wmma_down_hot_dev, h_f16_hot,
                                    wmma_f16_hot_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-mid down hot copy")) return 0;
            moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                    NULL, wmma_down_h, down_w, NULL, wmma_mid_h,
                    counts, offsets, sorted_pairs, wmma_down_hot_dev, wmma_f16_hot_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
            if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma f16-mid down launch")) return 0;
        }
#endif
        const uint64_t n = (uint64_t)n_tokens * out_dim;
        if (moe_wmma_hot) {
            if ((out_dim & 1u) == 0u) {
                const uint64_t n2 = n >> 1u;
                moe_sum_f16x2_kernel<<<(n2 + 255u) / 256u, 256>>>(
                        (float *)out->ptr, wmma_down_h, out_dim, n_expert, n_tokens);
            } else {
                moe_sum_f16_kernel<<<(n + 255u) / 256u, 256>>>(
                        (float *)out->ptr, wmma_down_h, out_dim, n_expert, n_tokens);
            }
        } else {
            moe_sum_kernel<<<(n + 255u) / 256u, 256>>>(
                    (float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
        }
        ok = cuda_ok(cudaGetLastError(), "routed_moe q2 expert sum launch");
        return ok;
    }

    const ds4_rocm_runtime_config *cfg = cuda_runtime_config();
    if (q2k_path && n_expert == 6u) {
        uint32_t rows_per_block = cfg->moe_decode_rpb;
        const uint32_t threads = rows_per_block * 32u;
        const int store_gate_up = (g_quality_mode || cfg->graph_dump) ? 1 : 0;
        const uint64_t xq_gate_bytes = (uint64_t)n_tokens * xq_blocks * sizeof(cuda_block_q8_K);
        const int q8k_gateup = !g_quality_mode && n_tokens == 1u &&
            n_expert == 6u && down->bytes >= xq_gate_bytes;
        int ok_gateup = 1;
        if (q8k_gateup) {
            cuda_block_q8_K *xq_gate = (cuda_block_q8_K *)down->ptr;
            dim3 xq_grid(xq_blocks, n_tokens, 1);
            q8_K_quantize_kernel<<<xq_grid, 256>>>(xq_gate, (const float *)x->ptr, expert_in_dim, n_tokens);
            ok_gateup = cuda_ok(cudaGetLastError(), "routed_moe q2 oldhip q8k gate input quantize launch");
            if (ok_gateup) {
                dim3 gate_grid((expert_mid_dim + 255u) / 256u, n_tokens * n_expert, 1);
                moe_gate_up_mid_q2K_decode_q8_qwarp32_kernel<<<gate_grid, 256u>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq_gate,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        (uint32_t)store_gate_up,
                        clamp);
                ok_gateup = cuda_ok(cudaGetLastError(), "routed_moe q2 oldhip q8k gate/up launch");
            }
        } else if (rows_per_block == 1u) {
            dim3 gate_grid(expert_mid_dim, n_tokens * n_expert, 1);
            moe_gate_up_mid_q2K_rows_rpb1_w32_kernel<<<gate_grid, 32u>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    (const float *)x->ptr,
                    (const int32_t *)selected_exec->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    expert_in_dim,
                    expert_mid_dim,
                    n_expert,
                    clamp,
                    store_gate_up);
            ok_gateup = cuda_ok(cudaGetLastError(), "routed_moe q2 oldhip rows gate/up launch");
        } else {
            dim3 gate_grid((expert_mid_dim + rows_per_block - 1u) / rows_per_block, n_tokens * n_expert, 1);
            moe_gate_up_mid_q2K_rows_w32_kernel<<<gate_grid, threads>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    (const float *)x->ptr,
                    (const int32_t *)selected_exec->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    expert_in_dim,
                    expert_mid_dim,
                    n_expert,
                    clamp,
                    store_gate_up);
            ok_gateup = cuda_ok(cudaGetLastError(), "routed_moe q2 oldhip rows gate/up launch");
        }
        if (!ok_gateup) return 0;
        int ok_decode_moe = 1;
        const uint64_t midq_bytes = (uint64_t)n_tokens * n_expert * midq_blocks * sizeof(cuda_block_q8_K);
        const int q8k_down = !g_quality_mode && n_tokens == 1u &&
            n_expert == 6u && down->bytes >= midq_bytes;
        if (q8k_down) {
            cuda_block_q8_K *midq = (cuda_block_q8_K *)down->ptr;
            dim3 midq_grid(midq_blocks, n_tokens * n_expert, 1);
            q8_K_quantize_kernel<<<midq_grid, 256>>>(midq, (const float *)mid->ptr, expert_mid_dim, n_tokens * n_expert);
            ok_decode_moe = cuda_ok(cudaGetLastError(), "routed_moe q2 oldhip q8k mid quantize launch");
            if (ok_decode_moe) {
                moe_down_sum6_qwarp32_kernel<<<(out_dim + 31u) / 32u, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim);
                ok_decode_moe = cuda_ok(cudaGetLastError(), "routed_moe q2 oldhip q8k down launch");
            }
        } else {
            dim3 down_grid((out_dim + rows_per_block - 1u) / rows_per_block, n_tokens, 1);
            moe_down_q2K_sum_rows_w32_kernel<<<down_grid, threads>>>(
                    (float *)out->ptr,
                    down_w,
                    (const float *)mid->ptr,
                    (const int32_t *)selected_exec->ptr,
                    n_tokens,
                    expert_mid_dim,
                    out_dim,
                    down_expert_bytes,
                    down_row_bytes);
            ok_decode_moe = cuda_ok(cudaGetLastError(), "routed_moe q2 oldhip rows down launch");
        }
        return ok_decode_moe;
    }

    if (ok) {
        dim3 mgrid(expert_mid_dim, n_tokens * n_expert, 1);
        if (q2k_path) {
            moe_gate_up_mid_q2K_f32_kernel<<<mgrid, 256>>>(
                (float *)gate->ptr,
                (float *)up->ptr,
                (float *)mid->ptr,
                gate_w,
                up_w,
                (const float *)x->ptr,
                (const int32_t *)selected_exec->ptr,
                (const float *)weights->ptr,
                gate_expert_bytes,
                gate_row_bytes,
                expert_in_dim,
                expert_mid_dim,
                n_expert,
                clamp);
        } else {
            moe_gate_up_mid_f32_kernel<<<mgrid, 256>>>(
                (float *)gate->ptr,
                (float *)up->ptr,
                (float *)mid->ptr,
                gate_w,
                up_w,
                (const float *)x->ptr,
                (const int32_t *)selected_exec->ptr,
                (const float *)weights->ptr,
                gate_expert_bytes,
                gate_row_bytes,
                expert_in_dim,
                expert_mid_dim,
                n_expert,
                clamp);
        }
        ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
    }
    if (ok) {
        dim3 dgrid(out_dim, n_tokens * n_expert, 1);
        moe_down_f32_kernel<<<dgrid, 256>>>(
            (float *)down->ptr,
            down_w,
            (const float *)mid->ptr,
            (const int32_t *)selected_exec->ptr,
            down_expert_bytes,
            down_row_bytes,
            expert_mid_dim,
            out_dim,
            n_expert);
        ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
    }
    if (ok) {
        uint64_t n = (uint64_t)n_tokens * out_dim;
        moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
    }
    if (ok && compact_selected) ok = cuda_stream_selected_mark_inflight();
    return ok;
}

extern "C" int ds4_gpu_routed_moe_one_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index) {
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x, layer_index, 1);
}
extern "C" int ds4_gpu_routed_moe_batch_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens, bool *mid_is_f16) {
    if (mid_is_f16) *mid_is_f16 = false;
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x, layer_index, n_tokens);
}
