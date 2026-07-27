#include "pulsar_engine_internal.h"



/* Release every GPU tensor owned by the whole-model graph runtime. */
void gpu_graph_free(pulsar_gpu_graph *g) {
    pulsar_gpu_tensor_free(g->descr_diag_pos);
    pulsar_gpu_tensor_free(g->descr_diag_seq);
    pulsar_gpu_tensor_free(g->batch_positions);
    pulsar_gpu_tensor_free(g->batch_seq_id);
    free(g->ms_positions);
    free(g->ms_seq_id);
    g->ms_positions = NULL;
    g->ms_seq_id = NULL;
    pulsar_gpu_tensor_free(g->directional_steering_dirs);
    pulsar_gpu_tensor_free(g->batch_ffn_out);
    pulsar_gpu_tensor_free(g->batch_routed_out);
    pulsar_gpu_tensor_free(g->batch_routed_down);
    pulsar_gpu_tensor_free(g->batch_routed_mid);
    pulsar_gpu_tensor_free(g->batch_routed_up);
    pulsar_gpu_tensor_free(g->batch_routed_gate);
    pulsar_gpu_tensor_free(g->batch_router_weights);
    pulsar_gpu_tensor_free(g->batch_router_selected);
    pulsar_gpu_tensor_free(g->batch_router_probs);
    pulsar_gpu_tensor_free(g->batch_router_logits);
    pulsar_gpu_tensor_free(g->batch_shared_out);
    pulsar_gpu_tensor_free(g->batch_shared_mid);
    pulsar_gpu_tensor_free(g->batch_shared_up);
    pulsar_gpu_tensor_free(g->batch_shared_gate);
    pulsar_gpu_tensor_free(g->batch_ffn_norm);
    pulsar_gpu_tensor_free(g->batch_ffn_cur);
    pulsar_gpu_tensor_free(g->batch_after_attn_hc);
    pulsar_gpu_tensor_free(g->batch_attn_out);
    pulsar_gpu_tensor_free(g->batch_attn_low);
    pulsar_gpu_tensor_free(g->batch_heads);
    pulsar_gpu_tensor_free(g->batch_indexer_weights);
    pulsar_gpu_tensor_free(g->batch_indexer_q);
    pulsar_gpu_tensor_free(g->batch_comp_sc);
    pulsar_gpu_tensor_free(g->batch_comp_kv);
    pulsar_gpu_tensor_free(g->batch_kv);
    pulsar_gpu_tensor_free(g->batch_kv_raw);
    pulsar_gpu_tensor_free(g->batch_q);
    pulsar_gpu_tensor_free(g->batch_qr_norm);
    pulsar_gpu_tensor_free(g->batch_qr);
    pulsar_gpu_tensor_free(g->batch_attn_norm);
    pulsar_gpu_tensor_free(g->batch_attn_cur);
    pulsar_gpu_tensor_free(g->batch_hc_split);
    pulsar_gpu_tensor_free(g->batch_hc_mix);
    pulsar_gpu_tensor_free(g->batch_flat_hc);
    pulsar_gpu_tensor_free(g->batch_next_hc);
    pulsar_gpu_tensor_free(g->batch_cur_hc);
    pulsar_gpu_tensor_free(g->prefill_tokens);
    pulsar_gpu_tensor_free(g->logits);
    pulsar_gpu_tensor_free(g->spec_logits);
    pulsar_gpu_tensor_free(g->dspark_main_x);
    for (int i = 0; i < 3; i++) {
        pulsar_gpu_tensor_free(g->dspark_target_h[i]);
        pulsar_gpu_tensor_free(g->dspark_raw_cache[i]);
        pulsar_gpu_tensor_free(g->dspark_bulk_h[i]);
        pulsar_gpu_tensor_free(g->dspark_prompt_h[i]);
        pulsar_gpu_tensor_free(g->dspark_target_h_batch[i]);
    }
    for (uint32_t il = 0; il < PULSAR_MAX_LAYER; il++) {
        pulsar_gpu_tensor_free(g->spec_comp_kv_save[il]);
        pulsar_gpu_tensor_free(g->spec_comp_sc_save[il]);
        pulsar_gpu_tensor_free(g->spec_icomp_kv_save[il]);
        pulsar_gpu_tensor_free(g->spec_icomp_sc_save[il]);
    }
    pulsar_gpu_tensor_free(g->spec_comp_scratch_row);
    pulsar_gpu_tensor_free(g->dspark_concat);
    pulsar_gpu_tensor_free(g->dspark_proj_out);
    pulsar_gpu_tensor_free(g->dspark_seed_kv);
    pulsar_gpu_tensor_free(g->dspark_seed_norm);
    pulsar_gpu_tensor_free(g->dspark_seed_rot);
    pulsar_gpu_tensor_free(g->dspark_markov_logits);
    pulsar_gpu_tensor_free(g->output_norm);
    pulsar_gpu_tensor_free(g->output_embd);
    pulsar_gpu_tensor_free(g->output_weights);
    pulsar_gpu_tensor_free(g->output_pre);
    pulsar_gpu_tensor_free(g->after_ffn_hc);
    pulsar_gpu_tensor_free(g->ffn_out);
    pulsar_gpu_tensor_free(g->routed_out);
    pulsar_gpu_tensor_free(g->routed_down);
    pulsar_gpu_tensor_free(g->routed_mid);
    pulsar_gpu_tensor_free(g->routed_up);
    pulsar_gpu_tensor_free(g->routed_gate);
    pulsar_gpu_tensor_free(g->router_weights);
    pulsar_gpu_tensor_free(g->router_selected);
    pulsar_gpu_tensor_free(g->router_probs);
    pulsar_gpu_tensor_free(g->router_logits);
    pulsar_gpu_tensor_free(g->shared_out);
    pulsar_gpu_tensor_free(g->shared_mid);
    pulsar_gpu_tensor_free(g->shared_up);
    pulsar_gpu_tensor_free(g->shared_gate);
    pulsar_gpu_tensor_free(g->ffn_norm);
    pulsar_gpu_tensor_free(g->ffn_cur);
    pulsar_gpu_tensor_free(g->after_attn_hc);
    pulsar_gpu_tensor_free(g->attn_out);
    pulsar_gpu_tensor_free(g->attn_low);
    pulsar_gpu_tensor_free(g->heads);
    pulsar_gpu_tensor_free(g->comp_sc_cur);
    pulsar_gpu_tensor_free(g->comp_kv_cur);
    pulsar_gpu_tensor_free(g->attn_comp_stage);
    pulsar_gpu_tensor_free(g->attn_comp_dequant);
    pulsar_gpu_tensor_free(g->idx_comp_stage);
    pulsar_gpu_tensor_free(g->comp_mask);
    pulsar_gpu_tensor_free(g->comp_selected);
    pulsar_gpu_tensor_free(g->indexer_scores);
    pulsar_gpu_tensor_free(g->indexer_weights);
    pulsar_gpu_tensor_free(g->indexer_q);
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        pulsar_gpu_tensor_free(g->layer_raw_cache[il]);
    }
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        pulsar_gpu_tensor_free(g->layer_attn_comp_cache[il]);
    }
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        pulsar_gpu_tensor_free(g->layer_attn_state_kv[il]);
    }
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        pulsar_gpu_tensor_free(g->layer_attn_state_score[il]);
    }
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        pulsar_gpu_tensor_free(g->layer_index_comp_cache[il]);
    }
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        pulsar_gpu_tensor_free(g->layer_index_state_kv[il]);
    }
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        pulsar_gpu_tensor_free(g->layer_index_state_score[il]);
    }
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        pulsar_gpu_tensor_free(g->spec_attn_state_kv[il]);
        pulsar_gpu_tensor_free(g->spec_attn_state_score[il]);
        pulsar_gpu_tensor_free(g->spec_index_state_kv[il]);
        pulsar_gpu_tensor_free(g->spec_index_state_score[il]);
    }
    /* Bank-pool slabs (the layer_* pointers freed above were views into
     * these when the pool was enabled; view frees release no memory). */
    for (uint32_t il = 0; il < PULSAR_MAX_LAYER; il++) {
        pulsar_gpu_tensor_free(g->banks.raw[il]);
        /* Per-bank split comp/index allocations (increment 2a): each bank is its
         * own cudaMallocManaged so the eviction guard can cudaFree it directly. */
        for (uint32_t bk = 0; bk < PULSAR_MSEQ_MAX; bk++) {
            pulsar_gpu_tensor_free(g->banks.comp[il][bk]);
            pulsar_gpu_tensor_free(g->banks.index[il][bk]);
        }
        pulsar_gpu_tensor_free(g->banks.comp_bases[il]);
        pulsar_gpu_tensor_free(g->banks.index_bases[il]);
        pulsar_gpu_tensor_free(g->banks.askv[il]);
        pulsar_gpu_tensor_free(g->banks.assc[il]);
        pulsar_gpu_tensor_free(g->banks.iskv[il]);
        pulsar_gpu_tensor_free(g->banks.issc[il]);
    }
    /* Option F per-bank drafter-ring slabs (dspark_raw_cache[i]/dspark_prompt_h[i]
     * freed above were bank views into these). */
    for (int i = 0; i < 3; i++) {
        pulsar_gpu_tensor_free(g->banks.dspark_raw[i]);
        pulsar_gpu_tensor_free(g->banks.dspark_prompt[i]);
    }
    /* plan-33 inc C: partial-fork boundary-row stash. */
    pulsar_gpu_tensor_free(g->emit_stash_comp);
    pulsar_gpu_tensor_free(g->emit_stash_index);
    g->emit_stash_comp = NULL;
    g->emit_stash_index = NULL;
    /* The batched-copy tables cache raw device pointers into the state tensors
     * freed above; drop them so a rebuilt graph re-prepares fresh tables. */
    pulsar_gpu_batched_copy_free(g->spec_snap_copies);
    pulsar_gpu_batched_copy_free(g->spec_restore_copies);
    g->spec_snap_copies = NULL;
    g->spec_restore_copies = NULL;
    g->spec_frontier_copy_n = 0;
    g->spec_frontier_copy_init = 0;
    pulsar_gpu_tensor_free(g->kv);
    pulsar_gpu_tensor_free(g->kv_raw);
    pulsar_gpu_tensor_free(g->q);
    pulsar_gpu_tensor_free(g->qr_norm);
    pulsar_gpu_tensor_free(g->qr);
    pulsar_gpu_tensor_free(g->attn_norm);
    pulsar_gpu_tensor_free(g->attn_cur);
    pulsar_gpu_tensor_free(g->hc_comb);
    pulsar_gpu_tensor_free(g->hc_post);
    pulsar_gpu_tensor_free(g->hc_pre);
    pulsar_gpu_tensor_free(g->hc_split);
    pulsar_gpu_tensor_free(g->hc_mix);
    pulsar_gpu_tensor_free(g->flat_hc);
    pulsar_gpu_tensor_free(g->cur_hc);
    memset(g, 0, sizeof(*g));
}



bool gpu_tensor_fill_f32(pulsar_gpu_tensor *t, float v, uint64_t n) {
    return pulsar_gpu_tensor_fill_f32(t, v, n) != 0;
}



/* =========================================================================
 * Directional Steering.
 * =========================================================================
 *
 * A steering file contains one normalized 4096-wide direction per layer.  When
 * enabled, the GPU graph edits selected block outputs in-place:
 *
 *     y = y - scale * v * dot(v, y)
 *
 * Positive scales remove the represented direction from the activation.
 * Negative scales add it.  This is deliberately explicit and opt-in; with zero
 * scales, the release graph does not allocate the direction tensor and follows
 * the normal inference path.
 */

bool gpu_graph_load_directional_steering(
        pulsar_gpu_graph *g,
        const char      *path,
        float            attn_scale,
        float            ffn_scale) {
    if (attn_scale == 0.0f && ffn_scale == 0.0f) return true;

    if (!path || !path[0]) {
        fprintf(stderr, "pulsar: directional steering needs --dir-steering-file\n");
        return false;
    }

    const uint64_t n = (uint64_t)PULSAR_N_LAYER * PULSAR_N_EMBD;
    float *dirs = (float *)xmalloc((size_t)n * sizeof(dirs[0]));
    bool ok = read_f32_binary_file(path, dirs, n);
    if (ok) {
        g->directional_steering_dirs = pulsar_gpu_tensor_alloc(n * sizeof(dirs[0]));
        ok = g->directional_steering_dirs != NULL &&
             pulsar_gpu_tensor_write(g->directional_steering_dirs, 0, dirs, n * sizeof(dirs[0])) != 0;
    }
    free(dirs);

    if (!ok) {
        fprintf(stderr, "pulsar: failed to load directional steering vectors from %s\n", path);
        return false;
    }
    g->directional_steering_attn_scale = attn_scale;
    g->directional_steering_ffn_scale = ffn_scale;
    fprintf(stderr, "pulsar: directional steering enabled: %s attn=%g ffn=%g\n",
            path, (double)attn_scale, (double)ffn_scale);
    return true;
}

