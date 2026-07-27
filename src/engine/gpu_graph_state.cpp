#include "ds4_engine_internal.h"



/* Release every GPU tensor owned by the whole-model graph runtime. */
void gpu_graph_free(ds4_gpu_graph *g) {
    ds4_gpu_tensor_free(g->descr_diag_pos);
    ds4_gpu_tensor_free(g->descr_diag_seq);
    ds4_gpu_tensor_free(g->batch_positions);
    ds4_gpu_tensor_free(g->batch_seq_id);
    free(g->ms_positions);
    free(g->ms_seq_id);
    g->ms_positions = NULL;
    g->ms_seq_id = NULL;
    ds4_gpu_tensor_free(g->directional_steering_dirs);
    ds4_gpu_tensor_free(g->batch_ffn_out);
    ds4_gpu_tensor_free(g->batch_routed_out);
    ds4_gpu_tensor_free(g->batch_routed_down);
    ds4_gpu_tensor_free(g->batch_routed_mid);
    ds4_gpu_tensor_free(g->batch_routed_up);
    ds4_gpu_tensor_free(g->batch_routed_gate);
    ds4_gpu_tensor_free(g->batch_router_weights);
    ds4_gpu_tensor_free(g->batch_router_selected);
    ds4_gpu_tensor_free(g->batch_router_probs);
    ds4_gpu_tensor_free(g->batch_router_logits);
    ds4_gpu_tensor_free(g->batch_shared_out);
    ds4_gpu_tensor_free(g->batch_shared_mid);
    ds4_gpu_tensor_free(g->batch_shared_up);
    ds4_gpu_tensor_free(g->batch_shared_gate);
    ds4_gpu_tensor_free(g->batch_ffn_norm);
    ds4_gpu_tensor_free(g->batch_ffn_cur);
    ds4_gpu_tensor_free(g->batch_after_attn_hc);
    ds4_gpu_tensor_free(g->batch_attn_out);
    ds4_gpu_tensor_free(g->batch_attn_low);
    ds4_gpu_tensor_free(g->batch_heads);
    ds4_gpu_tensor_free(g->batch_indexer_weights);
    ds4_gpu_tensor_free(g->batch_indexer_q);
    ds4_gpu_tensor_free(g->batch_comp_sc);
    ds4_gpu_tensor_free(g->batch_comp_kv);
    ds4_gpu_tensor_free(g->batch_kv);
    ds4_gpu_tensor_free(g->batch_kv_raw);
    ds4_gpu_tensor_free(g->batch_q);
    ds4_gpu_tensor_free(g->batch_qr_norm);
    ds4_gpu_tensor_free(g->batch_qr);
    ds4_gpu_tensor_free(g->batch_attn_norm);
    ds4_gpu_tensor_free(g->batch_attn_cur);
    ds4_gpu_tensor_free(g->batch_hc_split);
    ds4_gpu_tensor_free(g->batch_hc_mix);
    ds4_gpu_tensor_free(g->batch_flat_hc);
    ds4_gpu_tensor_free(g->batch_next_hc);
    ds4_gpu_tensor_free(g->batch_cur_hc);
    ds4_gpu_tensor_free(g->prefill_tokens);
    ds4_gpu_tensor_free(g->logits);
    ds4_gpu_tensor_free(g->spec_logits);
    ds4_gpu_tensor_free(g->dspark_main_x);
    for (int i = 0; i < 3; i++) {
        ds4_gpu_tensor_free(g->dspark_target_h[i]);
        ds4_gpu_tensor_free(g->dspark_raw_cache[i]);
        ds4_gpu_tensor_free(g->dspark_bulk_h[i]);
        ds4_gpu_tensor_free(g->dspark_prompt_h[i]);
        ds4_gpu_tensor_free(g->dspark_target_h_batch[i]);
    }
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor_free(g->spec_comp_kv_save[il]);
        ds4_gpu_tensor_free(g->spec_comp_sc_save[il]);
        ds4_gpu_tensor_free(g->spec_icomp_kv_save[il]);
        ds4_gpu_tensor_free(g->spec_icomp_sc_save[il]);
    }
    ds4_gpu_tensor_free(g->spec_comp_scratch_row);
    ds4_gpu_tensor_free(g->dspark_concat);
    ds4_gpu_tensor_free(g->dspark_proj_out);
    ds4_gpu_tensor_free(g->dspark_seed_kv);
    ds4_gpu_tensor_free(g->dspark_seed_norm);
    ds4_gpu_tensor_free(g->dspark_seed_rot);
    ds4_gpu_tensor_free(g->dspark_markov_logits);
    ds4_gpu_tensor_free(g->output_norm);
    ds4_gpu_tensor_free(g->output_embd);
    ds4_gpu_tensor_free(g->output_weights);
    ds4_gpu_tensor_free(g->output_pre);
    ds4_gpu_tensor_free(g->after_ffn_hc);
    ds4_gpu_tensor_free(g->ffn_out);
    ds4_gpu_tensor_free(g->routed_out);
    ds4_gpu_tensor_free(g->routed_down);
    ds4_gpu_tensor_free(g->routed_mid);
    ds4_gpu_tensor_free(g->routed_up);
    ds4_gpu_tensor_free(g->routed_gate);
    ds4_gpu_tensor_free(g->router_weights);
    ds4_gpu_tensor_free(g->router_selected);
    ds4_gpu_tensor_free(g->router_probs);
    ds4_gpu_tensor_free(g->router_logits);
    ds4_gpu_tensor_free(g->shared_out);
    ds4_gpu_tensor_free(g->shared_mid);
    ds4_gpu_tensor_free(g->shared_up);
    ds4_gpu_tensor_free(g->shared_gate);
    ds4_gpu_tensor_free(g->ffn_norm);
    ds4_gpu_tensor_free(g->ffn_cur);
    ds4_gpu_tensor_free(g->after_attn_hc);
    ds4_gpu_tensor_free(g->attn_out);
    ds4_gpu_tensor_free(g->attn_low);
    ds4_gpu_tensor_free(g->heads);
    ds4_gpu_tensor_free(g->comp_sc_cur);
    ds4_gpu_tensor_free(g->comp_kv_cur);
    ds4_gpu_tensor_free(g->attn_comp_stage);
    ds4_gpu_tensor_free(g->attn_comp_dequant);
    ds4_gpu_tensor_free(g->idx_comp_stage);
    ds4_gpu_tensor_free(g->comp_mask);
    ds4_gpu_tensor_free(g->comp_selected);
    ds4_gpu_tensor_free(g->indexer_scores);
    ds4_gpu_tensor_free(g->indexer_weights);
    ds4_gpu_tensor_free(g->indexer_q);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_raw_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_comp_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_state_kv[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_state_score[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_comp_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_state_kv[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_state_score[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->spec_attn_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_attn_state_score[il]);
        ds4_gpu_tensor_free(g->spec_index_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_index_state_score[il]);
    }
    /* Bank-pool slabs (the layer_* pointers freed above were views into
     * these when the pool was enabled; view frees release no memory). */
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor_free(g->banks.raw[il]);
        /* Per-bank split comp/index allocations (increment 2a): each bank is its
         * own cudaMallocManaged so the eviction guard can cudaFree it directly. */
        for (uint32_t bk = 0; bk < DS4_MSEQ_MAX; bk++) {
            ds4_gpu_tensor_free(g->banks.comp[il][bk]);
            ds4_gpu_tensor_free(g->banks.index[il][bk]);
        }
        ds4_gpu_tensor_free(g->banks.comp_bases[il]);
        ds4_gpu_tensor_free(g->banks.index_bases[il]);
        ds4_gpu_tensor_free(g->banks.askv[il]);
        ds4_gpu_tensor_free(g->banks.assc[il]);
        ds4_gpu_tensor_free(g->banks.iskv[il]);
        ds4_gpu_tensor_free(g->banks.issc[il]);
    }
    /* Option F per-bank drafter-ring slabs (dspark_raw_cache[i]/dspark_prompt_h[i]
     * freed above were bank views into these). */
    for (int i = 0; i < 3; i++) {
        ds4_gpu_tensor_free(g->banks.dspark_raw[i]);
        ds4_gpu_tensor_free(g->banks.dspark_prompt[i]);
    }
    /* plan-33 inc C: partial-fork boundary-row stash. */
    ds4_gpu_tensor_free(g->emit_stash_comp);
    ds4_gpu_tensor_free(g->emit_stash_index);
    g->emit_stash_comp = NULL;
    g->emit_stash_index = NULL;
    /* The batched-copy tables cache raw device pointers into the state tensors
     * freed above; drop them so a rebuilt graph re-prepares fresh tables. */
    ds4_gpu_batched_copy_free(g->spec_snap_copies);
    ds4_gpu_batched_copy_free(g->spec_restore_copies);
    g->spec_snap_copies = NULL;
    g->spec_restore_copies = NULL;
    g->spec_frontier_copy_n = 0;
    g->spec_frontier_copy_init = 0;
    ds4_gpu_tensor_free(g->kv);
    ds4_gpu_tensor_free(g->kv_raw);
    ds4_gpu_tensor_free(g->q);
    ds4_gpu_tensor_free(g->qr_norm);
    ds4_gpu_tensor_free(g->qr);
    ds4_gpu_tensor_free(g->attn_norm);
    ds4_gpu_tensor_free(g->attn_cur);
    ds4_gpu_tensor_free(g->hc_comb);
    ds4_gpu_tensor_free(g->hc_post);
    ds4_gpu_tensor_free(g->hc_pre);
    ds4_gpu_tensor_free(g->hc_split);
    ds4_gpu_tensor_free(g->hc_mix);
    ds4_gpu_tensor_free(g->flat_hc);
    ds4_gpu_tensor_free(g->cur_hc);
    memset(g, 0, sizeof(*g));
}



bool gpu_tensor_fill_f32(ds4_gpu_tensor *t, float v, uint64_t n) {
    return ds4_gpu_tensor_fill_f32(t, v, n) != 0;
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
        ds4_gpu_graph *g,
        const char      *path,
        float            attn_scale,
        float            ffn_scale) {
    if (attn_scale == 0.0f && ffn_scale == 0.0f) return true;

    if (!path || !path[0]) {
        fprintf(stderr, "ds4: directional steering needs --dir-steering-file\n");
        return false;
    }

    const uint64_t n = (uint64_t)DS4_N_LAYER * DS4_N_EMBD;
    float *dirs = (float *)xmalloc((size_t)n * sizeof(dirs[0]));
    bool ok = read_f32_binary_file(path, dirs, n);
    if (ok) {
        g->directional_steering_dirs = ds4_gpu_tensor_alloc(n * sizeof(dirs[0]));
        ok = g->directional_steering_dirs != NULL &&
             ds4_gpu_tensor_write(g->directional_steering_dirs, 0, dirs, n * sizeof(dirs[0])) != 0;
    }
    free(dirs);

    if (!ok) {
        fprintf(stderr, "ds4: failed to load directional steering vectors from %s\n", path);
        return false;
    }
    g->directional_steering_attn_scale = attn_scale;
    g->directional_steering_ffn_scale = ffn_scale;
    fprintf(stderr, "ds4: directional steering enabled: %s attn=%g ffn=%g\n",
            path, (double)attn_scale, (double)ffn_scale);
    return true;
}

