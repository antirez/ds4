#include "ds4_engine_internal.h"



float rms_abs_diff(const float *a, const float *b, uint64_t n) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        const double d = (double)a[i] - (double)b[i];
        ss += d * d;
    }
    return n ? (float)sqrt(ss / (double)n) : 0.0f;
}



uint64_t argmax_f32(const float *x, uint64_t n) {
    uint64_t best = 0;
    for (uint64_t i = 1; i < n; i++) {
        if (x[i] > x[best]) best = i;
    }
    return best;
}




void print_vec_stats(const char *name, const float *x, uint64_t n) {
    float minv = DS4_POS_INF;
    float maxv = DS4_NEG_INF;
    double ss = 0.0;

    for (uint64_t i = 0; i < n; i++) {
        const float v = x[i];
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
        ss += (double)v * v;
    }

    printf("%s: min=%g max=%g rms=%g\n",
        name, minv, maxv, sqrt(ss / (double)n));
}



void gpu_graph_debug_dump_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_f32,
        uint32_t          il,
        uint32_t          pos) {
    if (!t || n_f32 == 0 || !gpu_graph_debug_wants(name, il, pos)) return;
    const char *prefix = getenv("DS4_CUDA_GRAPH_DUMP_PREFIX");

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    float *buf = (float *)xmalloc((size_t)n_f32 * sizeof(buf[0]));
    if (ds4_gpu_tensor_read(t, 0, buf, n_f32 * sizeof(buf[0])) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.bin", prefix, name, il, pos);
        if (write_f32_binary_file(path, buf, n_f32)) {
            fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
        }
    }
    free(buf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume GPU command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}


/* Same dump for an HC residual CARRIER (BF16 storage; task #62). The plain f32
 * dump above would read n*sizeof(float) from a buffer holding n*DS4_HC_ELT_SIZE
 * bytes — a 2x out-of-bounds read — so carriers MUST come through here. */
void gpu_graph_debug_dump_hc_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_elems,
        uint32_t          il,
        uint32_t          pos) {
    if (!t || n_elems == 0 || !gpu_graph_debug_wants(name, il, pos)) return;
    const char *prefix = getenv("DS4_CUDA_GRAPH_DUMP_PREFIX");

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    float *buf = (float *)xmalloc((size_t)n_elems * sizeof(buf[0]));
    if (ds4_read_hc_carrier_f32(t, 0, buf, n_elems) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.bin", prefix, name, il, pos);
        if (write_f32_binary_file(path, buf, n_elems)) {
            fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
        }
    }
    free(buf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume GPU command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}



void gpu_graph_debug_dump_f16_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_f16,
        uint32_t          il,
        uint32_t          pos) {
    if (!t || n_f16 == 0 || !gpu_graph_debug_wants(name, il, pos)) return;
    const char *prefix = getenv("DS4_CUDA_GRAPH_DUMP_PREFIX");

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    uint16_t *hbuf = (uint16_t *)xmalloc((size_t)n_f16 * sizeof(hbuf[0]));
    float *fbuf = (float *)xmalloc((size_t)n_f16 * sizeof(fbuf[0]));
    if (ds4_gpu_tensor_read(t, 0, hbuf, n_f16 * sizeof(hbuf[0])) != 0) {
        for (uint64_t i = 0; i < n_f16; i++) fbuf[i] = f16_to_f32(hbuf[i]);
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.bin", prefix, name, il, pos);
        if (write_f32_binary_file(path, fbuf, n_f16)) {
            fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
        }
    }
    free(fbuf);
    free(hbuf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume GPU command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}



void gpu_graph_debug_dump_i32_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_i32,
        uint32_t          il,
        uint32_t          pos) {
    if (!t || n_i32 == 0 || !gpu_graph_debug_wants(name, il, pos)) return;
    const char *prefix = getenv("DS4_CUDA_GRAPH_DUMP_PREFIX");

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    int32_t *buf = (int32_t *)xmalloc((size_t)n_i32 * sizeof(buf[0]));
    if (ds4_gpu_tensor_read(t, 0, buf, n_i32 * sizeof(buf[0])) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.i32", prefix, name, il, pos);
        FILE *fp = fopen(path, "wb");
        if (fp) {
            if (fwrite(buf, sizeof(buf[0]), (size_t)n_i32, fp) == (size_t)n_i32) {
                fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
            }
            fclose(fp);
        }
    }
    free(buf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume GPU command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}



bool gpu_graph_needs_ffn_out(const ds4_gpu_graph *g, uint32_t il, uint32_t pos) {
    return gpu_graph_directional_steering_ffn_enabled(g) ||
           g->materialize_ffn_out ||
           gpu_graph_debug_wants("ffn_out", il, pos);
}



bool gpu_graph_ensure_ffn_out(ds4_gpu_graph *g) {
    if (!g->ffn_out) {
        g->ffn_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    }
    return g->ffn_out != NULL;
}



bool gpu_graph_ensure_batch_ffn_out(ds4_gpu_graph *g) {
    if (!g->batch_ffn_out) {
        g->batch_ffn_out = ds4_gpu_tensor_alloc((uint64_t)g->prefill_cap * DS4_N_EMBD * sizeof(float));
    }
    return g->batch_ffn_out != NULL;
}



/* =========================================================================
 * GPU Release Graph Allocation.
 * ========================================================================= */

/* Derived capacities and tensor dimensions for one session's GPU graph.
 * Computed by gpu_graph_compute_dims and shared by the allocator
 * (gpu_graph_alloc_raw_cap) and the sizing estimate (gpu_graph_session_bytes)
 * so admission control and the allocator can never disagree about the derived
 * quantities.  The per-buffer byte expressions still appear in both functions;
 * the server's estimate-vs-actual reconciliation (>10% drift warning) is the
 * enforcement that they stay in sync. */
typedef struct {
    uint32_t raw_cap;
    uint32_t raw_window;
    uint32_t ctx_size;
    uint32_t prefill_cap;
    uint32_t comp_cap;
    uint32_t attn_comp_stage_cap;          /* only meaningful under DS4_ATTN_PACK */
    uint32_t layer_comp_cap[DS4_MAX_LAYER];
    uint64_t hc_dim;
    uint64_t mix_hc;
    uint64_t q_rank;
    uint64_t q_dim;
    uint64_t low_dim;
    uint64_t shared_dim;
    uint64_t routed_mid_dim;
    uint64_t vocab_dim;
    uint64_t comp_width_max;
    uint64_t indexer_q_dim;
} gpu_graph_dims;

static void gpu_graph_compute_dims(
        gpu_graph_dims *d,
        const ds4_weights       *weights,
        const ds4_layer_weights *layer,
        uint32_t                 raw_cap,
        uint32_t                 ctx_size,
        uint32_t                 prefill_cap) {
    memset(d, 0, sizeof(*d));
    if (raw_cap == 0) raw_cap = 1;
    if (ctx_size == 0) ctx_size = raw_cap;
    if (prefill_cap == 0) prefill_cap = 1;
    uint32_t raw_window = DS4_N_SWA;
    if (raw_window > ctx_size) raw_window = ctx_size;
    if (raw_window == 0) raw_window = 1;
    if (raw_cap < raw_window) raw_cap = raw_window;
    if (raw_cap > ctx_size) raw_cap = ctx_size;
    if (raw_cap == 0) raw_cap = 1;
    d->raw_cap = raw_cap;
    d->raw_window = raw_window;
    d->ctx_size = ctx_size;
    d->prefill_cap = prefill_cap;

    uint32_t min_ratio = UINT32_MAX;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) min_ratio = ctx_size ? ctx_size : 1u;
    d->comp_cap = ctx_size / min_ratio + 2u;
    if (d->comp_cap < 2u) d->comp_cap = 2u;
    d->attn_comp_stage_cap = prefill_cap / min_ratio + 2u;
    if (d->attn_comp_stage_cap < 2u) d->attn_comp_stage_cap = 2u;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) {
            d->layer_comp_cap[il] = 0;
        } else {
            d->layer_comp_cap[il] = ctx_size / ratio + 2u;
            if (d->layer_comp_cap[il] < 2u) d->layer_comp_cap[il] = 2u;
        }
    }

    d->hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    d->mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    d->q_rank = layer->attn_q_a->dim[1];
    d->q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    d->low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    d->shared_dim = layer->ffn_gate_shexp->dim[1];
    d->routed_mid_dim = layer->ffn_gate_exps->dim[1];
    d->vocab_dim = weights->output ? weights->output->dim[1] : DS4_N_VOCAB;
    d->comp_width_max = 2ull * (DS4_N_HEAD_DIM > DS4_N_INDEXER_HEAD_DIM
        ? DS4_N_HEAD_DIM
        : DS4_N_INDEXER_HEAD_DIM);
    d->indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
}



/* TRUE total per-session GPU byte cost: the sum of every ds4_gpu_tensor that
 * gpu_graph_alloc_raw_cap (and, when enable_spec, gpu_graph_init_dspark_target)
 * allocates for one session.  This is what admission control must price a
 * session at — the packed KV estimate (ds4_context_memory_estimate_packed)
 * covers only the persistent KV rows plus the indexer score/mask scratch and
 * undercounts the real cost by roughly an order of magnitude (2026-07-13
 * incident: three ctx=65536 slots admitted at 0.5 GiB each consumed the whole
 * GB10 and hard-locked the machine).
 *
 * KEEP IN SYNC with the allocators below (gpu_graph_alloc_raw_cap,
 * gpu_graph_bank_slabs_alloc, gpu_graph_init_dspark_target — same order,
 * same expressions).
 * The server reconciles this estimate against the measured allocation delta
 * after every session create and logs a loud warning on >10% drift, so a
 * missed buffer surfaces on the first live run instead of as an
 * under-admission OOM.
 *
 * EXCLUSION LIST — intentionally unaccounted per-session allocations (each
 * negligible, absorbed by DS4_SERVER_MEM_FLOOR_BYTES; do not re-derive):
 *   - the lazy gpu_graph_ensure_ffn_out/gpu_graph_ensure_batch_ffn_out
 *     buffers (only allocated under steering/diagnostics);
 *   - directional-steering dirs (gpu_graph_load_directional_steering,
 *     ~DS4_N_LAYER*DS4_N_EMBD floats — these ARE inside the measured create
 *     delta, so they show up in reconciliation but never in this estimate);
 *   - the spec snapshot/restore descriptor tables that
 *     ds4_gpu_batched_copy_prepare lazily cudaMallocs on the first fused
 *     spec step (~KBs; raw cudaMalloc, outside both this estimate and the
 *     ds4_gpu_tensor byte counter that reconciliation measures). */
uint64_t gpu_graph_session_bytes(
        const ds4_weights       *weights,
        const ds4_layer_weights *layer,
        uint32_t                 raw_cap,
        uint32_t                 ctx_size,
        uint32_t                 prefill_cap,
        bool                     enable_spec) {
    return gpu_graph_session_bytes_banked(weights, layer, raw_cap, ctx_size,
                                          prefill_cap, enable_spec,
                                          gpu_graph_bank_pool_n());
}

uint64_t gpu_graph_session_bytes_banked(
        const ds4_weights       *weights,
        const ds4_layer_weights *layer,
        uint32_t                 raw_cap,
        uint32_t                 ctx_size,
        uint32_t                 prefill_cap,
        bool                     enable_spec,
        uint32_t                 n_banks_in) {
    gpu_graph_dims dz;
    gpu_graph_compute_dims(&dz, weights, layer, raw_cap, ctx_size, prefill_cap);
    const uint64_t pc = dz.prefill_cap;
    const uint64_t f32 = sizeof(float);
    const uint64_t hc = DS4_HC_ELT_SIZE;   /* HC residual carrier element (BF16); task #62 */

    /* Persistent KV caches (raw ring, packed attn comp, indexer comp) plus the
     * indexer_scores/comp_mask working pair — shared with the managed-vs-device
     * KV placement policy the allocator itself uses.  In bank-pool mode the
     * persistent KV slabs (and the per-bank compressor state lanes below)
     * scale with the pool size; everything else is shared by all banks. */
    const uint64_t n_banks = n_banks_in < 1u ? 1u : n_banks_in;
    uint64_t kv_cache_bytes = 0;
    uint64_t total = gpu_graph_context_bytes_for_kv_policy(
            dz.ctx_size, dz.raw_cap, dz.prefill_cap, &kv_cache_bytes);
    if (n_banks > 1u) total += (n_banks - 1u) * kv_cache_bytes;

    /* Per-layer attention/indexer state — one lane per bank — and the
     * single-lane spec shadows (spec never runs against a shared pool). */
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint64_t attn_state = (uint64_t)coff * DS4_N_HEAD_DIM *
                                    coff * ratio * f32;
        total += n_banks * 2ull * attn_state;             /* layer_attn_state_kv/score */
        if (enable_spec) total += 2ull * attn_state;      /* spec_attn_state_kv/score */
        if (ratio == 4) {
            const uint64_t index_state = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM *
                                         coff * ratio * f32;
            total += n_banks * 2ull * index_state;        /* layer_index_state_kv/score */
            if (enable_spec) total += 2ull * index_state; /* spec_index_state_kv/score */
        }
    }

    /* Single-token graph buffers. */
    total += dz.hc_dim * hc;                              /* cur_hc (carrier) */
    total += dz.hc_dim * f32;                             /* flat_hc (RMSNorm out, f32) */
    total += 2ull * dz.mix_hc * f32;                      /* hc_mix, hc_split (views free) */
    total += 2ull * DS4_N_EMBD * f32;                     /* attn_cur, attn_norm */
    total += 2ull * dz.q_rank * f32;                      /* qr, qr_norm */
    total += dz.q_dim * f32;                              /* q */
    total += 2ull * DS4_N_HEAD_DIM * f32;                 /* kv_raw, kv */
    total += 2ull * dz.comp_width_max * f32;              /* comp_kv_cur, comp_sc_cur */
    if (gpu_graph_attn_pack_enabled()) {
        total += (uint64_t)dz.attn_comp_stage_cap * DS4_N_HEAD_DIM * f32; /* attn_comp_stage */
        total += (uint64_t)dz.comp_cap * DS4_N_HEAD_DIM * f32;            /* attn_comp_dequant */
    }
    if (gpu_graph_idx_fp4_enabled()) {
        total += (uint64_t)dz.comp_cap * DS4_N_INDEXER_HEAD_DIM * f32;    /* idx_comp_stage */
    }
    total += dz.indexer_q_dim * f32;                      /* indexer_q */
    total += (uint64_t)DS4_N_INDEXER_HEAD * f32;          /* indexer_weights */
    total += (uint64_t)(DS4_N_INDEXER_TOP_K ? DS4_N_INDEXER_TOP_K : 1u) *
             pc * sizeof(uint32_t);                       /* comp_selected */
    total += dz.q_dim * f32;                              /* heads */
    total += dz.low_dim * f32;                            /* attn_low */
    total += (uint64_t)DS4_N_EMBD * f32;                  /* attn_out */
    total += dz.hc_dim * hc;                              /* after_attn_hc (carrier) */
    total += 2ull * DS4_N_EMBD * f32;                     /* ffn_cur, ffn_norm */
    total += 3ull * dz.shared_dim * f32;                  /* shared_gate/up/mid */
    total += (uint64_t)DS4_N_EMBD * f32;                  /* shared_out */
    total += 2ull * DS4_N_EXPERT * f32;                   /* router_logits, router_probs */
    total += (uint64_t)DS4_N_EXPERT_USED * (sizeof(int) + f32); /* router_selected/weights */
    total += 3ull * DS4_N_EXPERT_USED * dz.routed_mid_dim * f32; /* routed_gate/up/mid */
    total += (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * f32;     /* routed_down */
    total += (uint64_t)DS4_N_EMBD * f32;                  /* routed_out */
    total += dz.hc_dim * hc;                              /* after_ffn_hc (carrier) */
    total += 2ull * DS4_N_HC * f32;                       /* output_pre, output_weights */
    total += 2ull * DS4_N_EMBD * f32;                     /* output_embd, output_norm */
    total += dz.vocab_dim * f32;                          /* logits */
    total += pc * sizeof(int32_t);                        /* prefill_tokens */
    /* Multi-row logits readback slab.  Allocated unconditionally (NOT under
     * enable_spec): besides the DSpark draft/verify passes it is the output
     * buffer of every batched multi-row head — gpu_graph_verify_suffix_tops
     * and the Tier-2 batched multi-session decode driver
     * (gpu_graph_decode_multiseq_batch) both read their rows out of it, and
     * those paths must work with speculation disabled (--no-dspark,
     * ds4-bench/ds4-eval/agent, or any model without a merged drafter). */
    total += 16ull * DS4_N_VOCAB * f32;                   /* spec_logits */

    /* Batch (prefill working set) buffers — the pc-scaled bulk that dominates
     * the non-KV cost (~4 GiB at pc=4096 on Flash). */
    total += 2ull * pc * dz.hc_dim * hc;                  /* batch_cur/next_hc (carriers) */
    total += pc * dz.hc_dim * f32;                        /* batch_flat_hc (RMSNorm out, f32) */
    total += 2ull * pc * dz.mix_hc * f32;                 /* batch_hc_mix/split */
    total += 2ull * pc * DS4_N_EMBD * f32;                /* batch_attn_cur/norm */
    total += 2ull * pc * dz.q_rank * f32;                 /* batch_qr/qr_norm */
    total += pc * dz.q_dim * f32;                         /* batch_q */
    total += 2ull * pc * DS4_N_HEAD_DIM * f32;            /* batch_kv_raw/kv */
    total += 2ull * pc * dz.comp_width_max * f32;         /* batch_comp_kv/sc */
    total += pc * dz.indexer_q_dim * f32;                 /* batch_indexer_q */
    total += pc * DS4_N_INDEXER_HEAD * f32;               /* batch_indexer_weights */
    total += pc * dz.q_dim * f32;                         /* batch_heads */
    total += pc * dz.low_dim * f32;                       /* batch_attn_low */
    total += pc * DS4_N_EMBD * f32;                       /* batch_attn_out */
    total += pc * dz.hc_dim * hc;                         /* batch_after_attn_hc (carrier) */
    total += 2ull * pc * DS4_N_EMBD * f32;                /* batch_ffn_cur/norm */
    total += 3ull * pc * dz.shared_dim * f32;             /* batch_shared_gate/up/mid */
    total += pc * DS4_N_EMBD * f32;                       /* batch_shared_out */
    total += 2ull * pc * DS4_N_EXPERT * f32;              /* batch_router_logits/probs */
    total += pc * DS4_N_EXPERT_USED * (sizeof(int) + f32); /* batch_router_selected/weights */
    total += 3ull * pc * DS4_N_EXPERT_USED * dz.routed_mid_dim * f32; /* batch_routed_gate/up/mid */
    total += pc * DS4_N_EXPERT_USED * DS4_N_EMBD * f32;   /* batch_routed_down */
    total += pc * DS4_N_EMBD * f32;                       /* batch_routed_out */

    /* DSpark drafter graph state (gpu_graph_init_dspark_target), allocated by
     * ds4_session_create whenever the engine has a drafter loaded — the
     * production merged GGUF auto-enables it. */
    if (enable_spec) {
        total += 3ull * DS4_N_EMBD * f32;                 /* dspark_target_h[3] */
        total += 3ull * 17 * DS4_N_EMBD * f32;            /* dspark_target_h_batch[3] */
        /* Option F: dspark_raw_cache[3] + dspark_prompt_h[3] are BANKED slabs
         * (n_banks lanes) so the N=2 spec-time-slice lane keeps a warm ring per
         * bank; the rest of the drafter state is shared across banks. */
        total += (uint64_t)n_banks * 3ull * DS4_DSPARK_DRAFT_WINDOW * DS4_N_HEAD_DIM * f32; /* dspark_raw_cache[3] */
        total += (uint64_t)DS4_N_EMBD * f32;              /* dspark_main_x */
        total += 3ull * pc * DS4_N_EMBD * f32;            /* dspark_bulk_h[3] */
        total += (uint64_t)n_banks * 3ull * DS4_DSPARK_DRAFT_WINDOW * DS4_N_EMBD * f32; /* dspark_prompt_h[3] */
        const uint64_t attn_w = 2ull * DS4_N_HEAD_DIM;
        const uint64_t idx_w = 2ull * DS4_N_INDEXER_HEAD_DIM;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            if (ratio == 0) continue;
            total += 2ull * 17 * attn_w * f32;            /* spec_comp_kv/sc_save */
            if (ratio == 4) total += 2ull * 17 * idx_w * f32; /* spec_icomp_kv/sc_save */
        }
        total += (uint64_t)DS4_N_HEAD_DIM * f32;          /* spec_comp_scratch_row */
        total += 3ull * DS4_N_EMBD * f32;                 /* dspark_concat */
        total += (uint64_t)DS4_N_EMBD * f32;              /* dspark_proj_out */
        total += 3ull * DS4_N_HEAD_DIM * f32;             /* dspark_seed_kv/norm/rot */
        total += (uint64_t)DS4_N_VOCAB * f32;             /* dspark_markov_logits */
    }
    return total;
}



/* Tier-2 overcommit (task #55, increment 1): the DEMAND-PAGED (cudaMallocManaged,
 * physical-on-touch) bytes of ONE bank's ctx-scaled comp + index caches at the
 * given context.  This is exactly the comp/index portion of
 * gpu_graph_kv_cache_bytes_for_context (steering.c) MINUS the eager raw ring —
 * the part the overcommit auto-size reserves as VA only and does NOT charge at
 * admission (the eager floor is charged; physical materializes as the frontier
 * grows, tracked by gpu_graph_touched_kv_bytes).  Row widths track the ACTUAL
 * packed storage (DS4_ATTN_PACK attn comp + MXFP4 indexer), matching the slab
 * allocator in gpu_graph_bank_slabs_alloc. */
uint64_t gpu_graph_demand_paged_bytes_per_bank(uint32_t ctx_size) {
    const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
    const uint64_t idx_row = gpu_graph_idx_fp4_enabled()
        ? DS4_ENGINE_IDXFP4_ROWBYTES
        : (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float);
    uint64_t bytes = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint64_t comp_cap = (uint64_t)(ctx_size / ratio + 2u);
        bytes += comp_cap * attn_row;
        if (ratio == 4) bytes += comp_cap * idx_row;
    }
    return bytes;
}



/* DS4_KV_MANAGED: measurement override for the managed-vs-device placement of
 * the PERSISTENT KV caches (raw ring + attn/index comp caches), for the
 * multi-session UVM over-provisioning design (plan addendum 2026-07-14 —
 * first-touch fault cost must be measured before committing to
 * always-managed slot KV). Unset/empty → the size-based policy
 * (ds4_gpu_should_use_managed_kv_cache) decides, as before. "1"/"on" →
 * force cudaMallocManaged; "0"/"off"/"false" → force device cudaMalloc.
 * Read once per process (static cache) and consulted only at graph
 * allocation (session create) — never on a token/layer path. */
static int gpu_graph_kv_managed_override(void) {
    static int cached = -2;
    if (cached == -2) {
        const char *v = getenv("DS4_KV_MANAGED");
        if (!v || !v[0]) {
            cached = -1;
        } else if (strcmp(v, "0") == 0 || strcasecmp(v, "off") == 0 ||
                   strcasecmp(v, "false") == 0) {
            cached = 0;
        } else if (strcmp(v, "1") == 0 || strcasecmp(v, "on") == 0 ||
                   strcasecmp(v, "true") == 0) {
            cached = 1;
        } else {
            /* Unrecognized values fall back to the size policy instead of
             * silently force-flipping a measurement flag (a typo'd "of[f]"
             * must not force-manage the KV of a 128 GB box). */
            fprintf(stderr,
                    "ds4: DS4_KV_MANAGED=\"%s\" not recognized "
                    "(want 1/on/true or 0/off/false); using size policy\n", v);
            cached = -1;
        }
    }
    return cached;
}



/* DS4_MSEQ_BANKS: Tier-2 bank-pool size for the next graph allocation.
 * 1 (default) keeps the classic single-session cache layout; 2..DS4_MSEQ_MAX
 * allocates the fixed per-bank slabs (ds4_bank_slabs) and installs bank-0
 * views, so every existing single-session path runs unmodified against
 * bank 0.  Read once per process at graph allocation — never on a
 * token/layer path.  Interim wiring: later increments make the server pass
 * the pool size explicitly instead. */
uint32_t gpu_graph_bank_pool_n(void) {
    static long cached = -1;
    if (cached < 0) {
        const char *v = getenv("DS4_MSEQ_BANKS");
        long n = 1;
        if (v && v[0]) {
            char *end = NULL;
            n = strtol(v, &end, 10);
            /* Tolerate trailing whitespace (e.g. "2\n" from a shell here-doc),
             * like the DS4_CUDA_DECODE_INDEXER_SPARSE_THRESHOLD parser. */
            while (end && isspace((unsigned char)*end)) end++;
            if (end == v || (end && *end != '\0') || n < 1) {
                fprintf(stderr,
                        "ds4: DS4_MSEQ_BANKS=\"%s\" not recognized (want 1..%u); "
                        "bank pool disabled\n", v, DS4_MSEQ_MAX);
                n = 1;
            } else if (n > (long)DS4_MSEQ_MAX) {
                fprintf(stderr, "ds4: DS4_MSEQ_BANKS=%ld clamped to %u\n",
                        n, DS4_MSEQ_MAX);
                n = (long)DS4_MSEQ_MAX;
            }
        }
        cached = n;
    }
    return (uint32_t)cached;
}



/* Allocate the fixed per-bank KV slabs (layout: ds4_bank_slabs in
 * ds4_engine_internal.h; design adapted from Entrpi/ds4 v0.2 — citation at
 * the struct).  Per-bank byte expressions MUST match the single-session
 * branches in gpu_graph_alloc_raw_cap below and the pricing in
 * gpu_graph_session_bytes.  Compressor state lanes are primed for every bank
 * here (kv = 0, score = -inf), exactly like a fresh single-session graph, so
 * a later bank admit starts from the same state a cold session would. */
static bool gpu_graph_bank_slabs_alloc(
        ds4_gpu_graph      *g,
        uint32_t              n_banks,
        bool                  managed_kv_cache,
        const gpu_graph_dims *dz) {
    ds4_bank_slabs *b = &g->banks;
    const uint64_t raw_elem = gpu_graph_raw_f16_enabled() ? sizeof(uint16_t)
                                                          : sizeof(float);
    b->n_banks = n_banks;
    b->cur_bank = 0;
    b->raw_bank_bytes = (uint64_t)dz->raw_cap * DS4_N_HEAD_DIM * raw_elem;
    bool ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        /* uint32 row-ABI audit: batched kernels address rows as
         * seq_id * cap + local in uint32; reject the geometry up front. */
        if ((uint64_t)n_banks * dz->raw_cap > 4294967296ull ||
            (uint64_t)n_banks * dz->layer_comp_cap[il] > 4294967296ull) {
            fprintf(stderr,
                    "ds4: bank pool geometry overflows the uint32 row ABI "
                    "(%u banks x raw_cap %u / comp_cap %u)\n",
                    n_banks, dz->raw_cap, dz->layer_comp_cap[il]);
            return false;
        }
        b->raw[il] = gpu_graph_alloc_kv_cache_tensor(
                managed_kv_cache, (uint64_t)n_banks * b->raw_bank_bytes);
        ok = b->raw[il] != NULL;
        if (!ok || ratio == 0) continue;

        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
        const uint64_t attn_rows = (uint64_t)coff * ratio;
        const uint64_t attn_lane = attn_width * attn_rows * sizeof(float);
        b->comp_bank_bytes[il] = (uint64_t)dz->layer_comp_cap[il] *
                                 gpu_graph_attn_comp_cache_row_bytes();
        b->astate_bank_bytes[il] = attn_lane;
        /* Increment 2a: one cudaMallocManaged PER BANK (not one n_banks*bytes
         * slab) so the eviction guard can cudaFree a single idle bank's physical
         * directly. Uniform stride (comp_bank_bytes[il] is bank-independent). */
        void *comp_ptr_h[DS4_MSEQ_MAX];
        for (uint32_t bk = 0; ok && bk < n_banks; bk++) {
            b->comp[il][bk] = ds4_gpu_tensor_alloc_managed(b->comp_bank_bytes[il]);
            ok = b->comp[il][bk] != NULL;
            if (ok) comp_ptr_h[bk] = ds4_gpu_tensor_device_ptr(b->comp[il][bk]);
        }
        b->askv[il] = ds4_gpu_tensor_alloc((uint64_t)n_banks * attn_lane);
        b->assc[il] = ds4_gpu_tensor_alloc((uint64_t)n_banks * attn_lane);
        /* Device base-pointer table (indexed by seq_id) the batched READ kernels
         * use instead of base + seq_id*comp_cap over one slab. */
        if (ok) b->comp_bases[il] = ds4_gpu_tensor_alloc((uint64_t)n_banks * sizeof(void *));
        ok = ok && b->askv[il] && b->assc[il] && b->comp_bases[il] &&
             ds4_gpu_tensor_write(b->comp_bases[il], 0, comp_ptr_h,
                                  (uint64_t)n_banks * sizeof(void *)) &&
             gpu_tensor_fill_f32(b->askv[il], 0.0f,
                                 (uint64_t)n_banks * attn_width * attn_rows) &&
             gpu_tensor_fill_f32(b->assc[il], DS4_NEG_INF,
                                 (uint64_t)n_banks * attn_width * attn_rows);
        if (ok && ratio == 4) {
            const uint64_t index_width = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM;
            const uint64_t index_rows = (uint64_t)coff * ratio;
            const uint64_t index_lane = index_width * index_rows * sizeof(float);
            const uint64_t index_row_bytes = gpu_graph_idx_fp4_enabled()
                ? DS4_ENGINE_IDXFP4_ROWBYTES
                : (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float);
            b->index_bank_bytes[il] = (uint64_t)dz->layer_comp_cap[il] *
                                      index_row_bytes;
            b->istate_bank_bytes[il] = index_lane;
            void *index_ptr_h[DS4_MSEQ_MAX];
            for (uint32_t bk = 0; ok && bk < n_banks; bk++) {
                b->index[il][bk] = ds4_gpu_tensor_alloc_managed(b->index_bank_bytes[il]);
                ok = b->index[il][bk] != NULL;
                if (ok) index_ptr_h[bk] = ds4_gpu_tensor_device_ptr(b->index[il][bk]);
            }
            b->iskv[il] = ds4_gpu_tensor_alloc((uint64_t)n_banks * index_lane);
            b->issc[il] = ds4_gpu_tensor_alloc((uint64_t)n_banks * index_lane);
            if (ok) b->index_bases[il] = ds4_gpu_tensor_alloc((uint64_t)n_banks * sizeof(void *));
            ok = ok && b->iskv[il] && b->issc[il] && b->index_bases[il] &&
                 ds4_gpu_tensor_write(b->index_bases[il], 0, index_ptr_h,
                                      (uint64_t)n_banks * sizeof(void *)) &&
                 gpu_tensor_fill_f32(b->iskv[il], 0.0f,
                                     (uint64_t)n_banks * index_width * index_rows) &&
                 gpu_tensor_fill_f32(b->issc[il], DS4_NEG_INF,
                                     (uint64_t)n_banks * index_width * index_rows);
        }
    }
    /* plan-33 inc C: the partial-fork boundary-row stash (one packed comp row +
     * one packed index row per (bank, layer); a few hundred KB total). */
    if (ok) {
        const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
        const uint64_t idx_row = gpu_graph_idx_fp4_enabled()
            ? DS4_ENGINE_IDXFP4_ROWBYTES : (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float);
        g->emit_stash_comp = ds4_gpu_tensor_alloc((uint64_t)n_banks * DS4_N_LAYER * attn_row);
        g->emit_stash_index = ds4_gpu_tensor_alloc((uint64_t)n_banks * DS4_N_LAYER * idx_row);
        ok = g->emit_stash_comp && g->emit_stash_index;
    }
    return ok;
}



/* Write one bank's entry in the comp/index base-pointer tables (device arrays).
 * A NULL slab nulls the entry so a stray batched-kernel read of an evicted bank
 * faults instead of touching freed pages. */
static bool bank_bases_set(ds4_gpu_graph *g, uint32_t il, uint32_t bank,
                           ds4_gpu_tensor *comp_slab, ds4_gpu_tensor *index_slab) {
    ds4_bank_slabs *b = &g->banks;
    bool ok = true;
    if (b->comp_bases[il]) {
        void *cp = comp_slab ? ds4_gpu_tensor_device_ptr(comp_slab) : NULL;
        ok = ds4_gpu_tensor_write(b->comp_bases[il], (uint64_t)bank * sizeof(void *),
                                  &cp, sizeof(void *)) != 0;
    }
    if (ok && b->index_bases[il]) {
        void *ip = index_slab ? ds4_gpu_tensor_device_ptr(index_slab) : NULL;
        ok = ds4_gpu_tensor_write(b->index_bases[il], (uint64_t)bank * sizeof(void *),
                                  &ip, sizeof(void *)) != 0;
    }
    return ok;
}

/* Tier-2 task #55 increment 2b — EVICTION RECLAIM primitive. Free ONE idle bank's
 * own comp/index physical by a DIRECT cudaFree of its per-bank split allocations
 * (the only primitive that returns physical on GB10; Step-1/2a reclaim gate).
 * Nulls the slab pointers and their base-table entries, and ZEROES this bank's
 * frontier counters (ms_n_comp/ms_n_index_comp) so touched_kv_bytes stops counting
 * a freed bank — else the guard's projected never drops after a spill and it
 * cascades, evicting every idle bank on one breach (review finding 2). The disk
 * snapshot preserves the real counts for restore. MUST NOT be the installed (cur)
 * bank — the caller repoints away first. The eager raw ring + state lanes
 * (contiguous, bounded floor) are NOT freed and survive the cycle in place.
 *
 * Contract (review finding 4 — no ambiguous half-evicted state): returns FALSE
 * ONLY on a precondition refusal (bad args / cur bank) where NOTHING was freed and
 * the bank stays live. Once freeing begins it returns TRUE — the bank IS physically
 * evicted; a base-table device-write failure is logged HARD but does not un-evict
 * the bank (the stale table entry is never read while the bank is evicted, and
 * restore rebuilds it), so the caller can unconditionally mark it spilled. */
bool gpu_graph_bank_free_physical(ds4_gpu_graph *g, uint32_t bank) {
    if (!g || g->banks.n_banks == 0 || bank >= g->banks.n_banks) return false;
    if (bank == g->banks.cur_bank) return false;   /* precondition: never free cur */
    ds4_bank_slabs *b = &g->banks;
    bool table_ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        ds4_gpu_tensor_free(b->comp[il][bank]);
        b->comp[il][bank] = NULL;
        if (ratio == 4) {
            ds4_gpu_tensor_free(b->index[il][bank]);
            b->index[il][bank] = NULL;
        }
        table_ok = bank_bases_set(g, il, bank, NULL, NULL) && table_ok;
        /* Finding 2: a freed bank contributes 0 resident KV. */
        g->ms_n_comp[bank][il] = 0;
        g->ms_n_index_comp[bank][il] = 0;
    }
    /* plan-33: an evicted bank's boundary stash is meaningless — disarm the
     * emit-restore hook so a later cold refill cannot restore stale bytes. */
    g->ms_emit_keep[bank] = 0u;
    if (!table_ok) {
        fprintf(stderr,
                "ds4: WARNING free_physical bank %u: base-table NULL device-write "
                "failed (CUDA); bank IS evicted, table is rebuilt on restore\n", bank);
    }
    return true;   /* bank is physically evicted (slabs freed) */
}

/* Tier-2 task #55 increment 2b — RESTORE alloc primitive. Reallocate ONE evicted
 * bank's comp/index physical (fresh cudaMallocManaged: VA reserved, physical on
 * touch) and rebuild its base-table entries to the new pointers. The caller then
 * reloads the bank's KV (H2D from the disk snapshot) into these. Idempotent: a
 * slab already present is left untouched. Returns false on OOM. */
bool gpu_graph_bank_alloc_physical(ds4_gpu_graph *g, uint32_t bank) {
    if (!g || g->banks.n_banks == 0 || bank >= g->banks.n_banks) return false;
    ds4_bank_slabs *b = &g->banks;
    bool ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        if (!b->comp[il][bank]) {
            b->comp[il][bank] = ds4_gpu_tensor_alloc_managed(b->comp_bank_bytes[il]);
            ok = b->comp[il][bank] != NULL;
        }
        if (ok && ratio == 4 && !b->index[il][bank]) {
            b->index[il][bank] = ds4_gpu_tensor_alloc_managed(b->index_bank_bytes[il]);
            ok = b->index[il][bank] != NULL;
        }
        if (ok) ok = bank_bases_set(g, il, bank, b->comp[il][bank],
                                    ratio == 4 ? b->index[il][bank] : NULL);
    }
    if (!ok) {
        /* TRANSACTIONAL: roll the bank back to fully-evicted rather than leaving
         * layers 0..k allocated.  A half-allocated bank is the dangerous state —
         * callers (server_bank_restore_spilled) return false, so the slot stays
         * marked spilled, but the bank now LOOKS partly live.  Returning it to a
         * clean evicted state keeps gpu_graph_bank_is_evicted's answer truthful
         * and lets a later retry start from scratch. */
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (ds4_layer_compress_ratio(il) == 0) continue;
            if (b->comp[il][bank])  { ds4_gpu_tensor_free(b->comp[il][bank]);  b->comp[il][bank] = NULL; }
            if (b->index[il][bank]) { ds4_gpu_tensor_free(b->index[il][bank]); b->index[il][bank] = NULL; }
            (void)bank_bases_set(g, il, bank, NULL, NULL);
        }
    }
    return ok;
}

/* True when `bank`'s comp/index physical is currently freed (evicted) — the
 * server checks this before restoring on a returning request. */
bool gpu_graph_bank_is_evicted(const ds4_gpu_graph *g, uint32_t bank) {
    if (!g || g->banks.n_banks == 0 || bank >= g->banks.n_banks) return false;
    /* Scan EVERY compressed layer, not just the first.  The old version sampled
     * layer 0 and returned, on the assumption that slabs are freed in lockstep.
     * That holds for eviction, but NOT for a FAILED (re)allocation:
     * gpu_graph_bank_alloc_physical can fail partway and leave layers 0..k
     * allocated, at which point a layer-0 sample reports the bank LIVE while it
     * is really half-built — and bank_fork_copy would then read NULL/garbage
     * slabs from it (silent cross-conversation KV corruption).  A bank is only
     * "live" when every compressed layer has physical. */
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (ds4_layer_compress_ratio(il) == 0) continue;
        if (g->banks.comp[il][bank] == NULL) return true;
    }
    return false;
}

/* plan-33 inc C: base alignment for a partial cut = LCM of the layer compress
 * ratios (128 on Flash): a multiple-of-LCM cut leaves every ratio-128 layer with
 * an EMPTY in-progress group; only ratio-4 layers straddle (boundary row). */
uint32_t ds4_partial_fork_base_align(void) {
    static uint32_t a = 0;
    if (a == 0u) {
        uint32_t m = 4u;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const uint32_t r = ds4_layer_compress_ratio(il);
            if (r > m) m = r;
        }
        a = m;
    }
    return a;
}

/* plan-33 inc C: byte-REPLACE the recomputed ratio-4 boundary row with the stash.
 * Fires after any ratio-4 emit that wrote rows starting below the bank's keep
 * threshold (R/4+1); self-deactivates once emits move past. Byte-copy — NEVER
 * re-encode (MXFP4 QAT is non-idempotent; MXFP8 pack byte-copy is trivially
 * bit-exact too). Same-stream D2D: ordered after the emit's store and before any
 * later attention read. No-op when the pool/stash is absent or keep==0. */
bool gpu_graph_emit_keep_restore(ds4_gpu_graph *g, uint32_t il, uint32_t bank,
                                 uint32_t row0, uint32_t rows, bool indexer) {
    if (!g || rows == 0u || bank >= DS4_MSEQ_MAX) return true;
    const uint32_t keep = g->ms_emit_keep[bank];
    if (keep == 0u || row0 >= keep) return true;
    if (ds4_layer_compress_ratio(il) != 4u) return true;
    ds4_gpu_tensor *stash = indexer ? g->emit_stash_index : g->emit_stash_comp;
    if (!stash) return true;
    const uint64_t row_bytes = indexer
        ? (gpu_graph_idx_fp4_enabled() ? DS4_ENGINE_IDXFP4_ROWBYTES
                                       : (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float))
        : gpu_graph_attn_comp_cache_row_bytes();
    const uint32_t keep4 = keep - 1u;              /* the boundary row index R/4 */
    ds4_gpu_tensor *cache = indexer ? gpu_graph_bank_index_comp_view(g, il, bank)
                                    : gpu_graph_bank_attn_comp_view(g, il, bank);
    if (!cache) return false;
    const bool ok = ds4_gpu_tensor_copy(cache, (uint64_t)keep4 * row_bytes,
                                        stash,
                                        ((uint64_t)bank * DS4_N_LAYER + il) * row_bytes,
                                        row_bytes) != 0;
    ds4_gpu_tensor_free(cache);
    return ok;
}

/* Tier-2 PATH-A PARTIAL-CUT FORK (plan-33 increment C, the risky core). Clone
 * bank src's KV TRUNCATED at position R into dst (src==dst = in-place truncate:
 * no copies, counters/stash only). Preconds: pool on, R >= align, R % align == 0,
 * R+4 <= src_len (the boundary row R/4 pools [R-4, R+4) — all inside the
 * validated prefix). Wrapped-ring guard: if src's ring has scrolled past
 * R - raw_window, the replay's attention would read scrolled-out raw rows —
 * REFUSE (caller cold-prefills). Per layer: raw [0,R) (or the whole wrapped
 * ring); ratio-4 comp/index rows [0, R/4+1) — ONE row past the counter (the
 * byte-valid boundary row), counters set to R/4 so it is present-but-invisible;
 * ratio-128 rows [0, R/128) (group closed exactly at a 128-multiple cut); state
 * lanes copied for hygiene (the replay re-seeds ratio-4 state from raw). The
 * boundary rows are stashed (packed bytes, from src) and ms_emit_keep[dst] =
 * R/4+1 arms the emit-restore hook. Caller validates tokens + pins src FIRST. */
bool gpu_graph_bank_fork_copy_cut(ds4_gpu_graph *g, uint32_t src, uint32_t dst,
                                  uint32_t R, uint32_t src_len) {
    if (!g || g->banks.n_banks == 0) return false;
    if (src >= g->banks.n_banks || dst >= g->banks.n_banks) return false;
    const uint32_t align = ds4_partial_fork_base_align();
    if (R < align || (R % align) != 0u || (uint64_t)R + 4u > src_len) return false;
    if (gpu_graph_bank_is_evicted(g, src)) return false;
    if (src != dst && gpu_graph_bank_is_evicted(g, dst) &&
        !gpu_graph_bank_alloc_physical(g, dst)) return false;
    ds4_bank_slabs *b = &g->banks;
    /* Wrapped-ring window guard: ring holds positions [oldest, src_len); the
     * replay from R reads raw rows [R - raw_window, R). */
    const uint32_t rcap = g->raw_cap;
    const uint64_t oldest = src_len > rcap ? (uint64_t)src_len - rcap : 0u;
    if ((uint64_t)R < oldest + g->raw_window) return false;   /* scrolled out */
    if (!g->emit_stash_comp || !g->emit_stash_index) return false;
    const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
    const uint64_t idx_row = gpu_graph_idx_fp4_enabled()
        ? DS4_ENGINE_IDXFP4_ROWBYTES : (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float);
    const uint32_t keep4 = R / 4u;
    const uint64_t raw_row_bytes = b->raw_bank_bytes / rcap;
    bool ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (src != dst) {
            const uint64_t raw_bytes = (uint64_t)(src_len <= rcap ? R : rcap) * raw_row_bytes;
            if (raw_bytes)
                ok = ds4_gpu_tensor_copy(b->raw[il], (uint64_t)dst * b->raw_bank_bytes,
                                         b->raw[il], (uint64_t)src * b->raw_bank_bytes,
                                         raw_bytes) != 0;
            if (ok && ratio != 0) {
                const uint64_t crows = ratio == 4u ? (uint64_t)keep4 + 1u
                                                   : (uint64_t)R / ratio;
                if (crows)
                    ok = ds4_gpu_tensor_copy(b->comp[il][dst], 0, b->comp[il][src], 0,
                                             crows * attn_row) != 0;
                if (ok) ok = ds4_gpu_tensor_copy(b->askv[il], (uint64_t)dst * b->astate_bank_bytes[il],
                                                 b->askv[il], (uint64_t)src * b->astate_bank_bytes[il],
                                                 b->astate_bank_bytes[il]) != 0;
                if (ok) ok = ds4_gpu_tensor_copy(b->assc[il], (uint64_t)dst * b->astate_bank_bytes[il],
                                                 b->assc[il], (uint64_t)src * b->astate_bank_bytes[il],
                                                 b->astate_bank_bytes[il]) != 0;
                if (ok && ratio == 4u) {
                    ok = ds4_gpu_tensor_copy(b->index[il][dst], 0, b->index[il][src], 0,
                                             ((uint64_t)keep4 + 1u) * idx_row) != 0;
                    if (ok) ok = ds4_gpu_tensor_copy(b->iskv[il], (uint64_t)dst * b->istate_bank_bytes[il],
                                                     b->iskv[il], (uint64_t)src * b->istate_bank_bytes[il],
                                                     b->istate_bank_bytes[il]) != 0;
                    if (ok) ok = ds4_gpu_tensor_copy(b->issc[il], (uint64_t)dst * b->istate_bank_bytes[il],
                                                     b->issc[il], (uint64_t)src * b->istate_bank_bytes[il],
                                                     b->istate_bank_bytes[il]) != 0;
                }
            }
        }
        if (!ok) break;
        /* Boundary-row stash (from SRC's rows — identical to dst's copy, and the
         * only source for the src==dst truncate) + counters at frontier R. */
        if (ratio == 4u) {
            ok = ds4_gpu_tensor_copy(g->emit_stash_comp,
                                     ((uint64_t)dst * DS4_N_LAYER + il) * attn_row,
                                     b->comp[il][src], (uint64_t)keep4 * attn_row,
                                     attn_row) != 0;
            if (ok) ok = ds4_gpu_tensor_copy(g->emit_stash_index,
                                     ((uint64_t)dst * DS4_N_LAYER + il) * idx_row,
                                     b->index[il][src], (uint64_t)keep4 * idx_row,
                                     idx_row) != 0;
            g->ms_n_comp[dst][il] = keep4;
            g->ms_n_index_comp[dst][il] = keep4;
        } else if (ratio != 0) {
            g->ms_n_comp[dst][il] = R / ratio;
            g->ms_n_index_comp[dst][il] = 0u;
        } else {
            g->ms_n_comp[dst][il] = 0u;
            g->ms_n_index_comp[dst][il] = 0u;
        }
    }
    if (ok) g->ms_emit_keep[dst] = keep4 + 1u;
    return ok;
}

/* Tier-2 PATH-A FULL-PREFIX FORK (plan-33 increment A). Device-side D2D clone of
 * bank `src`'s entire committed KV into bank `dst`: per layer the raw ring (whole
 * bank region — position-indexed, stale slots harmlessly copied), the comp
 * frontier rows (ms_n_comp[src] rows at offset 0 of the split alloc), the ratio-4
 * index frontier rows, and the attn/index compressor state lanes; the per-bank
 * frontier counters are mirrored src->dst. No captured-graph invalidation (pure
 * D2D + host counters). The CALLER (ds4_session_bank_fork) has already memcmp-
 * validated the request prefix against src's committed history and pinned src
 * against eviction — this routine performs the copy only. Refuses if src is
 * evicted (no physical to clone; the caller restores from disk first) and
 * reallocs dst if it was freed. Returns false on a bad geometry / copy error. */
bool gpu_graph_bank_fork_copy(ds4_gpu_graph *g, uint32_t src, uint32_t dst) {
    if (!g || g->banks.n_banks == 0) return false;
    if (src >= g->banks.n_banks || dst >= g->banks.n_banks || src == dst) return false;
    if (gpu_graph_bank_is_evicted(g, src)) return false;   /* caller restores src first */
    if (gpu_graph_bank_is_evicted(g, dst) && !gpu_graph_bank_alloc_physical(g, dst)) return false;
    ds4_bank_slabs *b = &g->banks;
    const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
    const uint64_t idx_row = gpu_graph_idx_fp4_enabled()
        ? DS4_ENGINE_IDXFP4_ROWBYTES : (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float);
    bool ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        /* Raw ring: copy the whole bank region (bounded by raw_cap; the ring is
         * position-indexed so any stale slots are never read at dst's pos). */
        ok = ds4_gpu_tensor_copy(b->raw[il], (uint64_t)dst * b->raw_bank_bytes,
                                 b->raw[il], (uint64_t)src * b->raw_bank_bytes,
                                 b->raw_bank_bytes) != 0;
        g->ms_n_comp[dst][il] = g->ms_n_comp[src][il];
        g->ms_n_index_comp[dst][il] = g->ms_n_index_comp[src][il];
        if (!ok || ratio == 0) continue;
        const uint64_t csz = (uint64_t)g->ms_n_comp[src][il] * attn_row;
        if (ok && csz) ok = ds4_gpu_tensor_copy(b->comp[il][dst], 0, b->comp[il][src], 0, csz) != 0;
        if (ok) ok = ds4_gpu_tensor_copy(b->askv[il], (uint64_t)dst * b->astate_bank_bytes[il],
                                         b->askv[il], (uint64_t)src * b->astate_bank_bytes[il],
                                         b->astate_bank_bytes[il]) != 0;
        if (ok) ok = ds4_gpu_tensor_copy(b->assc[il], (uint64_t)dst * b->astate_bank_bytes[il],
                                         b->assc[il], (uint64_t)src * b->astate_bank_bytes[il],
                                         b->astate_bank_bytes[il]) != 0;
        if (ratio == 4) {
            const uint64_t isz = (uint64_t)g->ms_n_index_comp[src][il] * idx_row;
            if (ok && isz) ok = ds4_gpu_tensor_copy(b->index[il][dst], 0, b->index[il][src], 0, isz) != 0;
            if (ok) ok = ds4_gpu_tensor_copy(b->iskv[il], (uint64_t)dst * b->istate_bank_bytes[il],
                                             b->iskv[il], (uint64_t)src * b->istate_bank_bytes[il],
                                             b->istate_bank_bytes[il]) != 0;
            if (ok) ok = ds4_gpu_tensor_copy(b->issc[il], (uint64_t)dst * b->istate_bank_bytes[il],
                                             b->issc[il], (uint64_t)src * b->istate_bank_bytes[il],
                                             b->istate_bank_bytes[il]) != 0;
        }
    }
    return ok;
}



/* Re-install the graph's per-layer cache views onto `bank`.  Pure host-side
 * pointer surgery (view wrappers are freed/recreated; no device work), so the
 * caller must guarantee the device is idle with respect to the previous
 * bank's views.  The spec frontier copy tables bake raw device pointers of
 * the state views, so they are dropped for lazy rebuild. */
bool gpu_graph_bank_repoint(ds4_gpu_graph *g, uint32_t bank) {
    if (!g || g->banks.n_banks == 0 || bank >= g->banks.n_banks) return false;
    ds4_bank_slabs *b = &g->banks;
    if (bank == b->cur_bank) return true;
    bool ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        ds4_gpu_tensor_free(g->layer_raw_cache[il]);
        g->layer_raw_cache[il] = ds4_gpu_tensor_view(
                b->raw[il], (uint64_t)bank * b->raw_bank_bytes, b->raw_bank_bytes);
        ok = g->layer_raw_cache[il] != NULL;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (!ok || ratio == 0) continue;
        ds4_gpu_tensor_free(g->layer_attn_comp_cache[il]);
        ds4_gpu_tensor_free(g->layer_attn_state_kv[il]);
        ds4_gpu_tensor_free(g->layer_attn_state_score[il]);
        g->layer_attn_comp_cache[il] = ds4_gpu_tensor_view(
                b->comp[il][bank], 0, b->comp_bank_bytes[il]);
        g->layer_attn_state_kv[il] = ds4_gpu_tensor_view(
                b->askv[il], (uint64_t)bank * b->astate_bank_bytes[il],
                b->astate_bank_bytes[il]);
        g->layer_attn_state_score[il] = ds4_gpu_tensor_view(
                b->assc[il], (uint64_t)bank * b->astate_bank_bytes[il],
                b->astate_bank_bytes[il]);
        ok = g->layer_attn_comp_cache[il] && g->layer_attn_state_kv[il] &&
             g->layer_attn_state_score[il];
        if (ok && ratio == 4) {
            ds4_gpu_tensor_free(g->layer_index_comp_cache[il]);
            ds4_gpu_tensor_free(g->layer_index_state_kv[il]);
            ds4_gpu_tensor_free(g->layer_index_state_score[il]);
            g->layer_index_comp_cache[il] = ds4_gpu_tensor_view(
                    b->index[il][bank], 0, b->index_bank_bytes[il]);
            g->layer_index_state_kv[il] = ds4_gpu_tensor_view(
                    b->iskv[il], (uint64_t)bank * b->istate_bank_bytes[il],
                    b->istate_bank_bytes[il]);
            g->layer_index_state_score[il] = ds4_gpu_tensor_view(
                    b->issc[il], (uint64_t)bank * b->istate_bank_bytes[il],
                    b->istate_bank_bytes[il]);
            ok = g->layer_index_comp_cache[il] && g->layer_index_state_kv[il] &&
                 g->layer_index_state_score[il];
        }
    }
    /* Option F: swap the per-bank DSpark drafter ring views (present only when
     * the drafter is loaded and the ring was banked).  Same host-only pointer
     * surgery as the KV views above, so the spec path transparently reads the
     * active bank's warm window. */
    if (ok && b->dspark_raw[0]) {
        for (int i = 0; i < 3 && ok; i++) {
            ds4_gpu_tensor_free(g->dspark_raw_cache[i]);
            ds4_gpu_tensor_free(g->dspark_prompt_h[i]);
            g->dspark_raw_cache[i] = ds4_gpu_tensor_view(
                    b->dspark_raw[i], (uint64_t)bank * b->dspark_raw_bank_bytes,
                    b->dspark_raw_bank_bytes);
            g->dspark_prompt_h[i] = ds4_gpu_tensor_view(
                    b->dspark_prompt[i], (uint64_t)bank * b->dspark_prompt_bank_bytes,
                    b->dspark_prompt_bank_bytes);
            ok = g->dspark_raw_cache[i] && g->dspark_prompt_h[i];
        }
    }
    /* Stale pointer hygiene (mirrors gpu_graph_free). */
    ds4_gpu_batched_copy_free(g->spec_snap_copies);
    ds4_gpu_batched_copy_free(g->spec_restore_copies);
    g->spec_snap_copies = NULL;
    g->spec_restore_copies = NULL;
    g->spec_frontier_copy_n = 0;
    g->spec_frontier_copy_init = 0;
    if (ok) b->cur_bank = bank;
    return ok;
}



/* ===== Tier-2 banked multiseq step machinery (increment 2) ==============
 *
 * Pool/view accessors + per-bank frontier bookkeeping + the multiseq step
 * arm/disarm pair.  Contracts at the declarations (ds4_engine_internal.h)
 * and at the ms_* fields in ds4_gpu_graph. */

uint32_t gpu_graph_bank_pool_count(const ds4_gpu_graph *g) {
    return g->banks.n_banks ? g->banks.n_banks : 1u;
}

ds4_gpu_tensor *gpu_graph_bank_raw_pool(ds4_gpu_graph *g, uint32_t il) {
    if (!g || il >= DS4_N_LAYER) return NULL;
    return g->banks.n_banks ? g->banks.raw[il] : g->layer_raw_cache[il];
}

/* Nominal comp/index operand for the batched attention/indexer wrappers. With
 * per-bank split allocations there is no single slab; the batched (descriptor)
 * path addresses banks through the base-pointer table (below), so this returns
 * bank 0's allocation as the nominal typed operand (its per-bank size drives the
 * wrappers' buffer-size validation). Pool disabled → the classic tensor. */
ds4_gpu_tensor *gpu_graph_bank_attn_comp_pool(ds4_gpu_graph *g, uint32_t il) {
    if (!g || il >= DS4_N_LAYER) return NULL;
    return g->banks.n_banks ? g->banks.comp[il][0] : g->layer_attn_comp_cache[il];
}

ds4_gpu_tensor *gpu_graph_bank_index_comp_pool(ds4_gpu_graph *g, uint32_t il) {
    if (!g || il >= DS4_N_LAYER) return NULL;
    return g->banks.n_banks ? g->banks.index[il][0] : g->layer_index_comp_cache[il];
}

/* Per-bank comp/index base-pointer tables (device arrays of n_banks pointers,
 * indexed by seq_id) the batched READ kernels use in place of base +
 * seq_id*comp_cap. NULL when the pool is disabled. */
ds4_gpu_tensor *gpu_graph_bank_attn_comp_bases(ds4_gpu_graph *g, uint32_t il) {
    if (!g || il >= DS4_N_LAYER) return NULL;
    return g->banks.n_banks ? g->banks.comp_bases[il] : NULL;
}

ds4_gpu_tensor *gpu_graph_bank_index_comp_bases(ds4_gpu_graph *g, uint32_t il) {
    if (!g || il >= DS4_N_LAYER) return NULL;
    return g->banks.n_banks ? g->banks.index_bases[il] : NULL;
}

/* One accessor body for all six per-(bank,layer) view kinds: a fresh view of
 * the bank's lane in the slab, or (pool disabled) a fresh view wrapping the
 * whole classic tensor — bank 0 only, so a stale bank id cannot silently
 * alias the single session's state. */
static ds4_gpu_tensor *bank_lane_view(ds4_gpu_graph *g,
                                      ds4_gpu_tensor *slab,
                                      const uint64_t *lane_bytes,
                                      ds4_gpu_tensor *classic,
                                      uint32_t il, uint32_t bank) {
    if (!g || il >= DS4_N_LAYER) return NULL;
    if (g->banks.n_banks == 0) {
        if (bank != 0 || !classic) return NULL;
        return ds4_gpu_tensor_view(classic, 0, ds4_gpu_tensor_bytes(classic));
    }
    if (bank >= g->banks.n_banks || !slab) return NULL;
    return ds4_gpu_tensor_view(slab, (uint64_t)bank * lane_bytes[il], lane_bytes[il]);
}

/* Per-bank comp/index view: with split allocations the bank's whole allocation
 * IS its lane (offset 0), so it bypasses the contiguous-slab bank_lane_view. */
static ds4_gpu_tensor *bank_split_view(ds4_gpu_graph *g,
                                       ds4_gpu_tensor *bank_slab,
                                       const uint64_t *lane_bytes,
                                       ds4_gpu_tensor *classic,
                                       uint32_t il, uint32_t bank) {
    if (!g || il >= DS4_N_LAYER) return NULL;
    if (g->banks.n_banks == 0) {
        if (bank != 0 || !classic) return NULL;
        return ds4_gpu_tensor_view(classic, 0, ds4_gpu_tensor_bytes(classic));
    }
    if (bank >= g->banks.n_banks || !bank_slab) return NULL;
    return ds4_gpu_tensor_view(bank_slab, 0, lane_bytes[il]);
}

ds4_gpu_tensor *gpu_graph_bank_attn_comp_view(ds4_gpu_graph *g, uint32_t il, uint32_t bank) {
    return bank_split_view(g, g->banks.n_banks ? g->banks.comp[il][bank] : NULL,
                           g->banks.comp_bank_bytes,
                           g->layer_attn_comp_cache[il], il, bank);
}

ds4_gpu_tensor *gpu_graph_bank_index_comp_view(ds4_gpu_graph *g, uint32_t il, uint32_t bank) {
    return bank_split_view(g, g->banks.n_banks ? g->banks.index[il][bank] : NULL,
                           g->banks.index_bank_bytes,
                           g->layer_index_comp_cache[il], il, bank);
}

ds4_gpu_tensor *gpu_graph_bank_attn_state_kv_view(ds4_gpu_graph *g, uint32_t il, uint32_t bank) {
    return bank_lane_view(g, g->banks.askv[il], g->banks.astate_bank_bytes,
                          g->layer_attn_state_kv[il], il, bank);
}

ds4_gpu_tensor *gpu_graph_bank_attn_state_score_view(ds4_gpu_graph *g, uint32_t il, uint32_t bank) {
    return bank_lane_view(g, g->banks.assc[il], g->banks.astate_bank_bytes,
                          g->layer_attn_state_score[il], il, bank);
}

ds4_gpu_tensor *gpu_graph_bank_index_state_kv_view(ds4_gpu_graph *g, uint32_t il, uint32_t bank) {
    return bank_lane_view(g, g->banks.iskv[il], g->banks.istate_bank_bytes,
                          g->layer_index_state_kv[il], il, bank);
}

ds4_gpu_tensor *gpu_graph_bank_index_state_score_view(ds4_gpu_graph *g, uint32_t il, uint32_t bank) {
    return bank_lane_view(g, g->banks.issc[il], g->banks.istate_bank_bytes,
                          g->layer_index_state_score[il], il, bank);
}

void gpu_graph_bank_counters_capture(ds4_gpu_graph *g, uint32_t bank) {
    if (!g || bank >= DS4_MSEQ_MAX) return;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g->ms_n_comp[bank][il] = g->layer_n_comp[il];
        g->ms_n_index_comp[bank][il] = g->layer_n_index_comp[il];
    }
    /* Option F: the drafter-ring frontier is per-bank too (device rings live in
     * banks.dspark_*), so it rides the same capture/install hand-off. */
    for (int i = 0; i < 3; i++) g->ms_dspark_n_raw[bank][i] = g->dspark_n_raw[i];
    g->ms_dspark_prompt_n[bank] = g->dspark_prompt_n;
    g->ms_dspark_prompt_lo[bank] = g->dspark_prompt_lo;
}

void gpu_graph_bank_counters_install(ds4_gpu_graph *g, uint32_t bank) {
    if (!g || bank >= DS4_MSEQ_MAX) return;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g->layer_n_comp[il] = g->ms_n_comp[bank][il];
        g->layer_n_index_comp[il] = g->ms_n_index_comp[bank][il];
    }
    for (int i = 0; i < 3; i++) g->dspark_n_raw[i] = g->ms_dspark_n_raw[bank][i];
    g->dspark_prompt_n = g->ms_dspark_prompt_n[bank];
    g->dspark_prompt_lo = g->ms_dspark_prompt_lo[bank];
}

/* Tier-2 overcommit (task #55, increment 1): EXACT touched (physically resident)
 * demand-paged KV bytes across the whole pool — Σ over live banks Σ over layers
 * of (comp frontier rows × comp row bytes + index frontier rows × index row
 * bytes).  Deterministic from the position-driven compressor frontier; no
 * cudaMemGetInfo / MemAvailable.  This is the number the increment-2 eviction
 * guard triggers on, and the accounting-exactness gate proves it tracks the real
 * physical delta.  The CURRENT bank's frontier is live in layer_n_comp /
 * layer_n_index_comp; idle banks keep their frontier in ms_n_comp / ms_n_index_comp
 * (captured on switch-away).  Pool disabled (n_banks==0) → pool_count 1, cur 0 →
 * the classic single-session frontier (layer_n_comp) is summed. Only the ctx-
 * scaled comp/index are counted; the eager raw ring + state lanes are the fixed
 * floor and are already resident (not part of the growing touched set). */
/* Exact touched (physically resident) demand-paged comp/index KV of ONE bank,
 * from its compressor frontier.  cur bank reads the live layer_n_comp; idle banks
 * read their captured ms_n_comp.  The increment-2b guard uses this for the
 * per-bank Δ projection and the smallest-frontier victim tie-break. */
uint64_t gpu_graph_bank_touched_kv_bytes(const ds4_gpu_graph *g, uint32_t bank) {
    if (!g) return 0;
    const uint32_t nb = gpu_graph_bank_pool_count(g);
    if (bank >= nb) return 0;
    const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
    const uint64_t idx_row = gpu_graph_idx_fp4_enabled()
        ? DS4_ENGINE_IDXFP4_ROWBYTES
        : (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float);
    const uint32_t cur = g->banks.n_banks ? g->banks.cur_bank : 0u;
    uint64_t bytes = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t ncomp = (bank == cur) ? g->layer_n_comp[il]
                                             : g->ms_n_comp[bank][il];
        bytes += (uint64_t)ncomp * attn_row;
        if (ratio == 4) {
            const uint32_t nidx = (bank == cur) ? g->layer_n_index_comp[il]
                                                : g->ms_n_index_comp[bank][il];
            bytes += (uint64_t)nidx * idx_row;
        }
    }
    return bytes;
}

uint64_t gpu_graph_touched_kv_bytes(const ds4_gpu_graph *g) {
    if (!g) return 0;
    const uint32_t nb = gpu_graph_bank_pool_count(g);
    uint64_t bytes = 0;
    for (uint32_t b = 0; b < nb; b++) bytes += gpu_graph_bank_touched_kv_bytes(g, b);
    return bytes;
}

/* Tier-2 task #55 increment 2b — CONSERVATIVE per-bank comp/index growth over one
 * decode quantum of `q` tokens: Σ_layers( ceil(q/ratio)·comp_row + q·index_row ).
 * The index term charges q (not ceil(q/ratio)) rows — a deliberate over-estimate
 * so the guard fires EARLY (safe side). Position-independent, so total Δ =
 * n_live_growing_banks × this. */
uint64_t gpu_graph_quantum_growth_bytes_per_bank(uint32_t q) {
    const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
    const uint64_t idx_row = gpu_graph_idx_fp4_enabled()
        ? DS4_ENGINE_IDXFP4_ROWBYTES : (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float);
    uint64_t bytes = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        bytes += (uint64_t)((q + ratio - 1u) / ratio) * attn_row;
        if (ratio == 4) bytes += (uint64_t)q * idx_row;
    }
    return bytes;
}

bool gpu_graph_multiseq_step_begin(ds4_gpu_graph *g, const int32_t *pos,
                                   const int32_t *seq, uint32_t n_rows,
                                   bool capture_cur) {
    if (!g || !pos || !seq || n_rows == 0 || n_rows > g->prefill_cap) {
        fprintf(stderr, "ds4: multiseq step rejected: bad args (n_rows=%u)\n",
                n_rows);
        return false;
    }
    if (g->batch_multiseq) {
        fprintf(stderr, "ds4: multiseq step rejected: step already armed\n");
        return false;
    }
    const uint32_t n_banks = gpu_graph_bank_pool_count(g);
    /* Constraint (relaxed for the batched-decode driver): contiguous per-bank
     * runs (each bank at most one run), positions consecutive WITHIN each
     * bank's run, and every run starting at a position > 0.  Banks may sit at
     * unrelated positions — the multi-session shape; the upstream batch
     * stages (RoPE q/kv/indexer-q/inverse, compressor loop, raw scatter,
     * attention, indexer) are all per-row-position driven.  Position-0 rows
     * stay rejected: admission (from-zero) prefill runs on the classic
     * single-bank view path in v1, and negative positions would wrap the
     * uint32 casts below. */
    bool bank_seen[DS4_MSEQ_MAX] = {false};
    int32_t prev_bank = -1;
    for (uint32_t t = 0; t < n_rows; t++) {
        if (seq[t] < 0 || (uint32_t)seq[t] >= n_banks ||
            (uint32_t)seq[t] >= DS4_MSEQ_MAX) {
            fprintf(stderr, "ds4: multiseq step rejected: row %u bank %d "
                            "out of pool (n_banks=%u)\n", t, seq[t], n_banks);
            return false;
        }
        if (seq[t] != prev_bank) {
            if (bank_seen[seq[t]]) {
                fprintf(stderr, "ds4: multiseq step rejected: bank %d rows "
                                "not contiguous\n", seq[t]);
                return false;
            }
            bank_seen[seq[t]] = true;
            prev_bank = seq[t];
            if (pos[t] <= 0) {
                fprintf(stderr, "ds4: multiseq step rejected: bank %d first "
                                "position %d <= 0 (admission prefill is "
                                "single-bank classic)\n", seq[t], pos[t]);
                return false;
            }
        } else if ((int64_t)pos[t] != (int64_t)pos[t - 1] + 1) {
            /* int64 arithmetic: pos[t-1] == INT32_MAX would make the int32
             * increment signed overflow (UB) before the per-row bound below
             * could reject it. */
            fprintf(stderr, "ds4: multiseq step rejected: bank %d positions "
                            "not consecutive within its run (row %u: %d, "
                            "want %lld)\n", seq[t], t, pos[t],
                    (long long)((int64_t)pos[t - 1] + 1));
            return false;
        }
        /* Bound EVERY row (not just each run's first): positions are cast to
         * uint32 downstream (ring slot, visible-comp, RoPE), and the
         * position-derived arithmetic below adds 1. */
        if (pos[t] <= 0 || pos[t] == INT32_MAX) {
            fprintf(stderr, "ds4: multiseq step rejected: bank %d row %u "
                            "position %d out of range (want 0 < pos < "
                            "INT32_MAX)\n", seq[t], t, pos[t]);
            return false;
        }
    }
    /* capture_cur: the classic scalar counters are the CURRENT bank's truth
     * (its per-bank slots may be stale — e.g. still holding a previous
     * step's values).  The capture is COMMITTED only after every rejection
     * point below has passed: a rejected begin must leave ms_n_comp[cur]
     * untouched too (the scalars can hold a previous step's cross-bank
     * superset, which would corrupt the bank's frontier record).  Until
     * then the validation loop reads the scalars directly for cur_bank. */
    const uint32_t cur_bank = g->banks.n_banks ? g->banks.cur_bank : 0u;
    /* DRIVER CONTRACT check: each batched bank's committed frontier is
     * position-true at its first row — floor(first_pos / ratio) compressed
     * rows.  A bank that lags (mid-admission-prefill) must never be
     * co-scheduled: its rows would clamp against the superset and silently
     * diverge from single-session output. */
    for (uint32_t t = 0; t < n_rows; t++) {
        if (t > 0 && seq[t] == seq[t - 1]) continue;
        const uint32_t b = (uint32_t)seq[t];
        const uint32_t p = (uint32_t)pos[t];
        const bool use_scalars = capture_cur && b == cur_bank;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            if (ratio == 0) continue;
            const uint32_t have_comp = use_scalars ? g->layer_n_comp[il]
                                                   : g->ms_n_comp[b][il];
            const uint32_t have_index = use_scalars ? g->layer_n_index_comp[il]
                                                    : g->ms_n_index_comp[b][il];
            if (have_comp != p / ratio ||
                (ratio == 4 && have_index != p / ratio)) {
                fprintf(stderr,
                        "ds4: multiseq step rejected: bank %u frontier not "
                        "position-true at layer %u (pos %u ratio %u: "
                        "n_comp %u want %u, n_index_comp %u)\n",
                        b, il, p, ratio, have_comp, p / ratio, have_index);
                return false;
            }
        }
    }
    /* Lazy descriptor storage: host mirrors + device arrays, prefill_cap
     * entries (a few KB) — never allocated in single-session serving.  On a
     * (transient) device-alloc failure everything is released and reset so a
     * later step re-attempts instead of failing forever. */
    if (!g->ms_positions) {
        g->ms_positions = (int32_t *)xmalloc((size_t)g->prefill_cap * sizeof(int32_t));
        g->ms_seq_id = (int32_t *)xmalloc((size_t)g->prefill_cap * sizeof(int32_t));
        g->batch_positions =
            ds4_gpu_tensor_alloc((uint64_t)g->prefill_cap * sizeof(int32_t));
        g->batch_seq_id =
            ds4_gpu_tensor_alloc((uint64_t)g->prefill_cap * sizeof(int32_t));
        if (!g->batch_positions || !g->batch_seq_id) {
            fprintf(stderr, "ds4: multiseq descriptor alloc failed\n");
            ds4_gpu_tensor_free(g->batch_positions);
            ds4_gpu_tensor_free(g->batch_seq_id);
            g->batch_positions = NULL;
            g->batch_seq_id = NULL;
            free(g->ms_positions);
            free(g->ms_seq_id);
            g->ms_positions = NULL;
            g->ms_seq_id = NULL;
            return false;
        }
    }
    memcpy(g->ms_positions, pos, (size_t)n_rows * sizeof(int32_t));
    memcpy(g->ms_seq_id, seq, (size_t)n_rows * sizeof(int32_t));
    if (!ds4_gpu_tensor_write(g->batch_positions, 0, pos,
                              (uint64_t)n_rows * sizeof(int32_t)) ||
        !ds4_gpu_tensor_write(g->batch_seq_id, 0, seq,
                              (uint64_t)n_rows * sizeof(int32_t))) {
        fprintf(stderr, "ds4: multiseq descriptor upload failed\n");
        return false;
    }
    /* Superset refresh — the ONLY write of the scalar counters during a
     * multiseq step (top of step, before any launch, never mid-forward).
     * The value is the step's emit-inclusive visibility bound: max over
     * rows of (pos+1)/ratio, which every batched bank's frontier reaches
     * once its own emits land (verified by step_end).  Validate EVERY
     * layer's capacity before writing ANY scalar: a rejected begin must
     * leave the graph's classic counters untouched (a partial overwrite
     * would inflate the frontiers a recovering classic caller decodes
     * with). */
    uint32_t sup[DS4_MAX_LAYER];
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        sup[il] = 0;
        if (ratio == 0) continue;
        for (uint32_t t = 0; t < n_rows; t++) {
            const uint32_t v = ((uint32_t)pos[t] + 1u) / ratio;
            if (v > sup[il]) sup[il] = v;
        }
        if (sup[il] > g->layer_comp_cap[il]) {
            fprintf(stderr,
                    "ds4: multiseq step rejected: superset %u exceeds comp "
                    "cap %u at layer %u\n", sup[il], g->layer_comp_cap[il], il);
            return false;
        }
    }
    /* All rejection points passed: NOW commit the cur-bank capture (the
     * scalars are still the pre-step classic values here — the superset
     * write below is the first scalar mutation). */
    if (capture_cur) gpu_graph_bank_counters_capture(g, cur_bank);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        g->layer_n_comp[il] = sup[il];
        if (ratio == 4) g->layer_n_index_comp[il] = sup[il];
    }
    g->batch_multiseq_rows = n_rows;
    g->batch_multiseq = true;
    /* plan-34 inc 2/3/4: arm the M-independent custom GEMMs for the DECODE PREFIX
     * of this step. Row layout (inc 4): decode rows first — one 1-row run per
     * decode bank — then the K-row PREFILL run(s). n_dec = the count of leading
     * rows that live in length-1 runs; every dense GEMM runs those rows through the
     * M-independent custom kernel (byte-identical to a decode-only step of that
     * width — gate-4 neutrality) and the trailing prefill rows through the fast
     * tensor-core path. Decode-only (every run length 1) => n_dec == n_rows (arm
     * all, == inc-2). Pure prefill (first run length>1) => n_dec == 0 (arm none,
     * == inc-3). The scan stops at the first multi-row run: the inc-4 scheduler
     * lays decode rows strictly before the prefill run (a length-1 run appearing
     * AFTER a prefill run is not an inc-4 layout and would be treated as prefill —
     * documented, enforced by the row builder / gate). */
    uint32_t n_dec = 0;
    for (uint32_t t = 0; t < n_rows; ) {
        uint32_t rl = 1;
        while (t + rl < n_rows && seq[t + rl] == seq[t]) rl++;
        if (rl == 1) { n_dec++; t++; } else break;
    }
    ds4_gpu_matmul_set_batch_mneutral((int)n_dec);
    return true;
}

bool gpu_graph_multiseq_step_end(ds4_gpu_graph *g) {
    if (!g || !g->batch_multiseq) return false;
    ds4_gpu_matmul_set_batch_mneutral(0);
    g->batch_multiseq = false;
    const uint32_t n_rows = g->batch_multiseq_rows;
    g->batch_multiseq_rows = 0;
    /* Self-check (host ints only): every batched bank's frontier advanced to
     * exactly its position-derived value — (last_pos+1)/ratio — and the
     * scalar superset (untouched since step top) equals the max over the
     * batched banks.  A miss here is the silent-KV-corruption class; fail
     * loud so the driver aborts the session instead of serving garbage. */
    bool ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        uint32_t sup = 0;
        for (uint32_t t = 0; t < n_rows && ok; t++) {
            if (t + 1 < n_rows && g->ms_seq_id[t + 1] == g->ms_seq_id[t]) continue;
            const uint32_t b = (uint32_t)g->ms_seq_id[t];
            const uint32_t want = ((uint32_t)g->ms_positions[t] + 1u) / ratio;
            if (want > sup) sup = want;
            if (g->ms_n_comp[b][il] != want ||
                (ratio == 4 && g->ms_n_index_comp[b][il] != want)) {
                fprintf(stderr,
                        "ds4: multiseq step_end FAILED: bank %u layer %u "
                        "frontier %u/%u want %u (ratio %u)\n",
                        b, il, g->ms_n_comp[b][il],
                        g->ms_n_index_comp[b][il], want, ratio);
                ok = false;
            }
        }
        if (ok && g->layer_n_comp[il] != sup) {
            fprintf(stderr,
                    "ds4: multiseq step_end FAILED: layer %u superset %u "
                    "mutated mid-step (want %u)\n",
                    il, g->layer_n_comp[il], sup);
            ok = false;
        }
        if (ok && ratio == 4 && g->layer_n_index_comp[il] != sup) {
            fprintf(stderr,
                    "ds4: multiseq step_end FAILED: layer %u indexer superset "
                    "%u mutated mid-step (want %u)\n",
                    il, g->layer_n_index_comp[il], sup);
            ok = false;
        }
    }
    return ok;
}



/* Allocate the GPU graph state for a chosen raw-cache capacity.  The model
 * weights are not copied here; tensors reference the mapped GGUF. */
bool gpu_graph_alloc_raw_cap(
        ds4_gpu_graph *g,
        const ds4_weights     *weights,
        const ds4_layer_weights *layer,
        uint32_t                raw_cap,
        uint32_t                ctx_size,
        uint32_t                prefill_cap,
        bool                    enable_spec) {
    memset(g, 0, sizeof(*g));
    g->comp_ratio_override = -1;
    gpu_graph_dims dz;
    gpu_graph_compute_dims(&dz, weights, layer, raw_cap, ctx_size, prefill_cap);
    raw_cap = dz.raw_cap;
    ctx_size = dz.ctx_size;
    prefill_cap = dz.prefill_cap;
    g->raw_cap = dz.raw_cap;
    g->raw_window = dz.raw_window;
    g->prefill_cap = dz.prefill_cap;

    /* DS4_ATTN_PACK validation lives up here: no allocations have happened
     * yet, so these early returns need no cleanup. */
    if (getenv("DS4_ATTN_MX") != NULL) {
        /* Removed 2026-07-10: superseded by DS4_ATTN_PACK (bit-exact, smaller
         * rows; MX re-quantized the rope dims and cost drafter acceptance).
         * Refuse loudly instead of silently running a different format. */
        fprintf(stderr,
                "ds4: DS4_ATTN_MX has been removed (superseded by DS4_ATTN_PACK); "
                "unset DS4_ATTN_MX (use DS4_ATTN_PACK=0 for the classic f32 comp cache)\n");
        return false;
    }
    if (gpu_graph_attn_pack_enabled() &&
        (DS4_N_ROT != 64u || ((DS4_N_HEAD_DIM - DS4_N_ROT) % 64u) != 0u)) {
        fprintf(stderr,
                "ds4: DS4_ATTN_PACK requires n_rot 64 and 64-aligned nope dims "
                "(head_dim %u / n_rot %u)\n",
                (unsigned)DS4_N_HEAD_DIM, (unsigned)DS4_N_ROT);
        return false;
    }
    g->comp_cap = dz.comp_cap;
    if (gpu_graph_attn_pack_enabled()) {
        g->attn_comp_stage_cap = dz.attn_comp_stage_cap;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g->layer_comp_cap[il] = dz.layer_comp_cap[il];
    }

    const uint64_t hc_dim = dz.hc_dim;
    const uint64_t mix_hc = dz.mix_hc;
    const uint64_t q_rank = dz.q_rank;
    const uint64_t q_dim = dz.q_dim;
    const uint64_t low_dim = dz.low_dim;
    const uint64_t shared_dim = dz.shared_dim;
    const uint64_t routed_mid_dim = dz.routed_mid_dim;
    const uint64_t vocab_dim = dz.vocab_dim;
    const uint64_t comp_width_max = dz.comp_width_max;
    const uint64_t indexer_q_dim = dz.indexer_q_dim;
    const uint64_t pc = prefill_cap;
    uint64_t kv_cache_bytes = 0;
    const uint64_t context_bytes =
        gpu_graph_context_bytes_for_kv_policy(ctx_size, raw_cap, prefill_cap, &kv_cache_bytes);
    const int managed_override = gpu_graph_kv_managed_override();
    const bool managed_kv_cache = managed_override >= 0
        ? managed_override != 0
        : ds4_gpu_should_use_managed_kv_cache(kv_cache_bytes, context_bytes) != 0;
    if (managed_override >= 0) {
        fprintf(stderr,
                "ds4: DS4_KV_MANAGED override: KV caches use %s allocation "
                "(measurement flag; policy would have chosen %s)\n",
                managed_kv_cache ? "MANAGED (cudaMallocManaged)"
                                 : "DEVICE (cudaMalloc)",
                ds4_gpu_should_use_managed_kv_cache(kv_cache_bytes,
                                                    context_bytes) != 0
                    ? "managed" : "device");
    }
    if (managed_kv_cache) {
        /*
         * CUDA device allocations are fastest, but a million-token KV cache is
         * large enough to starve DGX Spark's unified CPU/GPU memory once the
         * model cache and driver allocations are present.  For this one
         * long-lived cache class, managed memory restores the old demand-paged
         * behavior.  It can be slower, but it keeps oversized contexts from
         * turning memory pressure into a machine-wide lockup.
         */
        fprintf(stderr,
                "ds4: CUDA using managed KV cache for ctx=%u "
                "(kv cache %.2f GiB, context buffers %.2f GiB); "
                "this may degrade performance but is needed for very large contexts\n",
                ctx_size,
                (double)kv_cache_bytes / 1073741824.0,
                (double)context_bytes / 1073741824.0);
    }

    /* Tier-2 bank pool: allocate the per-bank slabs first; the per-layer
     * cache pointers below then become bank-0 views instead of owning
     * allocations, and all single-session code runs unmodified. */
    const uint32_t n_banks = gpu_graph_bank_pool_n();
    if (n_banks >= 2u &&
        !gpu_graph_bank_slabs_alloc(g, n_banks, managed_kv_cache, &dz)) {
        gpu_graph_free(g);
        return false;
    }
    const bool banked = g->banks.n_banks != 0;

    g->cur_hc = ds4_gpu_tensor_alloc(hc_dim * DS4_HC_ELT_SIZE);   /* HC residual carrier (BF16); task #62 */
    g->flat_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->hc_mix = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    g->hc_split = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    g->hc_pre = ds4_gpu_tensor_view(g->hc_split, 0, (uint64_t)DS4_N_HC * sizeof(float));
    g->hc_post = ds4_gpu_tensor_view(g->hc_split,
                                       (uint64_t)DS4_N_HC * sizeof(float),
                                       (uint64_t)DS4_N_HC * sizeof(float));
    g->hc_comb = ds4_gpu_tensor_view(g->hc_split,
                                       2ull * DS4_N_HC * sizeof(float),
                                       (uint64_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    g->attn_cur = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->attn_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->qr = ds4_gpu_tensor_alloc(q_rank * sizeof(float));
    g->qr_norm = ds4_gpu_tensor_alloc(q_rank * sizeof(float));
    g->q = ds4_gpu_tensor_alloc(q_dim * sizeof(float));
    g->kv_raw = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    g->kv = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    bool state_init_ok = true;
    const uint64_t raw_elem_bytes = gpu_graph_raw_f16_enabled() ? sizeof(uint16_t)
                                                                : sizeof(float);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g->layer_raw_cache[il] = banked
            ? ds4_gpu_tensor_view(g->banks.raw[il], 0, g->banks.raw_bank_bytes)
            : gpu_graph_alloc_kv_cache_tensor(
                    managed_kv_cache,
                    (uint64_t)raw_cap * DS4_N_HEAD_DIM * raw_elem_bytes);
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0) {
            const uint32_t coff = ratio == 4 ? 2u : 1u;
            const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
            const uint64_t attn_rows = (uint64_t)coff * ratio;
            const uint64_t comp_row_bytes = gpu_graph_attn_comp_cache_row_bytes();
            if (banked) {
                g->layer_attn_comp_cache[il] = ds4_gpu_tensor_view(
                        g->banks.comp[il][0], 0, g->banks.comp_bank_bytes[il]);
                g->layer_attn_state_kv[il] = ds4_gpu_tensor_view(
                        g->banks.askv[il], 0, g->banks.astate_bank_bytes[il]);
                g->layer_attn_state_score[il] = ds4_gpu_tensor_view(
                        g->banks.assc[il], 0, g->banks.astate_bank_bytes[il]);
            } else {
                g->layer_attn_comp_cache[il] = gpu_graph_alloc_kv_cache_tensor(
                        managed_kv_cache,
                        (uint64_t)g->layer_comp_cap[il] * comp_row_bytes);
                g->layer_attn_state_kv[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
                g->layer_attn_state_score[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            }
            if (enable_spec) {
                g->spec_attn_state_kv[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
                g->spec_attn_state_score[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            }
            /* Banked mode primes every bank's state lanes at slab alloc. */
            if (!banked && g->layer_attn_state_kv[il]) {
                state_init_ok = state_init_ok &&
                                gpu_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, attn_width * attn_rows);
            }
            if (!banked && g->layer_attn_state_score[il]) {
                state_init_ok = state_init_ok &&
                                gpu_tensor_fill_f32(g->layer_attn_state_score[il], DS4_NEG_INF, attn_width * attn_rows);
            }

            if (ratio == 4) {
                const uint64_t index_width = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM;
                const uint64_t index_rows = (uint64_t)coff * ratio;
                const uint64_t index_row_bytes = gpu_graph_idx_fp4_enabled()
                    ? DS4_ENGINE_IDXFP4_ROWBYTES
                    : (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float);
                if (banked) {
                    g->layer_index_comp_cache[il] = ds4_gpu_tensor_view(
                            g->banks.index[il][0], 0, g->banks.index_bank_bytes[il]);
                    g->layer_index_state_kv[il] = ds4_gpu_tensor_view(
                            g->banks.iskv[il], 0, g->banks.istate_bank_bytes[il]);
                    g->layer_index_state_score[il] = ds4_gpu_tensor_view(
                            g->banks.issc[il], 0, g->banks.istate_bank_bytes[il]);
                } else {
                    g->layer_index_comp_cache[il] = gpu_graph_alloc_kv_cache_tensor(
                            managed_kv_cache,
                            (uint64_t)g->layer_comp_cap[il] * index_row_bytes);
                    g->layer_index_state_kv[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                    g->layer_index_state_score[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                }
                if (enable_spec) {
                    g->spec_index_state_kv[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                    g->spec_index_state_score[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                }
                if (!banked && g->layer_index_state_kv[il]) {
                    state_init_ok = state_init_ok &&
                                    gpu_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, index_width * index_rows);
                }
                if (!banked && g->layer_index_state_score[il]) {
                    state_init_ok = state_init_ok &&
                                    gpu_tensor_fill_f32(g->layer_index_state_score[il], DS4_NEG_INF, index_width * index_rows);
                }
            }
        }
    }
    g->comp_kv_cur = ds4_gpu_tensor_alloc(comp_width_max * sizeof(float));
    g->comp_sc_cur = ds4_gpu_tensor_alloc(comp_width_max * sizeof(float));
    if (gpu_graph_attn_pack_enabled()) {
        /* f32 staging: the compressor writes real f32 rows here, then the commit
         * step packs them to the persistent DS4_ATTN_PACK comp cache. */
        g->attn_comp_stage = ds4_gpu_tensor_alloc((uint64_t)g->attn_comp_stage_cap *
                                                  DS4_N_HEAD_DIM * sizeof(float));
        /* f32 shadow the prefill attention consumers (and session save) read
         * after the persistent packed comp cache is dequantized (see
         * gpu_graph_attn_comp_read_cache). */
        g->attn_comp_dequant = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap *
                                                    DS4_N_HEAD_DIM * sizeof(float));
    }
    if (gpu_graph_idx_fp4_enabled()) {
        if (DS4_N_INDEXER_HEAD_DIM != 128u) {
            /* The packed loader and QAT+pack kernels hard-code the 68-byte
             * head_dim-128 row; fail loud here instead of deep in a launch. */
            fprintf(stderr,
                    "ds4: DS4_IDX_FP4 requires indexer head_dim 128 (model has %u)\n",
                    DS4_N_INDEXER_HEAD_DIM);
            gpu_graph_free(g);
            return false;
        }
        /* f32 emit/repack staging for the packed indexer cache: comp-cap rows so
         * the compressor writers can keep their absolute row indices. */
        g->idx_comp_stage = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap *
                                                 DS4_N_INDEXER_HEAD_DIM * sizeof(float));
        ds4_gpu_indexer_set_fp4(1);
    }
    g->indexer_q = ds4_gpu_tensor_alloc(indexer_q_dim * sizeof(float));
    g->indexer_weights = ds4_gpu_tensor_alloc((uint64_t)DS4_N_INDEXER_HEAD * sizeof(float));
    /* DS4_PREFILL_SLICE: these two are the only ctx-scaling f32 work buffers
     * with a prefill_cap token dimension; under slicing they only ever hold
     * one <=slice-token span at a time. */
    const uint64_t score_rows = (gpu_graph_prefill_slice() != 0u &&
                                 (uint64_t)gpu_graph_prefill_slice() < pc)
        ? (uint64_t)gpu_graph_prefill_slice() : pc;
    g->indexer_scores = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap * score_rows * sizeof(float));
    g->comp_mask = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap * score_rows * sizeof(float));
    g->comp_selected = ds4_gpu_tensor_alloc((uint64_t)(DS4_N_INDEXER_TOP_K ? DS4_N_INDEXER_TOP_K : 1u) *
                                              pc * sizeof(uint32_t));
    g->heads = ds4_gpu_tensor_alloc(q_dim * sizeof(float));
    g->attn_low = ds4_gpu_tensor_alloc(low_dim * sizeof(float));
    g->attn_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->after_attn_hc = ds4_gpu_tensor_alloc(hc_dim * DS4_HC_ELT_SIZE);   /* HC residual carrier */
    g->ffn_cur = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->ffn_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->shared_gate = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_up = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_mid = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->router_logits = ds4_gpu_tensor_alloc(DS4_N_EXPERT * sizeof(float));
    g->router_probs = ds4_gpu_tensor_alloc(DS4_N_EXPERT * sizeof(float));
    g->router_selected = ds4_gpu_tensor_alloc(DS4_N_EXPERT_USED * sizeof(int));
    g->router_weights = ds4_gpu_tensor_alloc(DS4_N_EXPERT_USED * sizeof(float));
    /* NOTE (task #64): these look debug-only — the qwarp32 kernels' stores are
     * read only by gpu_graph_debug_dump_tensor — but they are NOT safe to skip:
     * routed_moe_*_impl rejects NULL gate/up (moe.cu ~2597/2908) and the batch
     * path REUSES gate->ptr as cuda_block_q8_K staging for quantized mid
     * (moe.cu ~2957). Keep them allocated unconditionally. */
    g->routed_gate = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_up = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_mid = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_down = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
    g->routed_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->after_ffn_hc = ds4_gpu_tensor_alloc(hc_dim * DS4_HC_ELT_SIZE);   /* HC residual carrier */
    g->output_pre = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HC * sizeof(float));
    g->output_weights = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HC * sizeof(float));
    g->output_embd = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->output_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->logits = ds4_gpu_tensor_alloc(vocab_dim * sizeof(float));
    g->prefill_tokens = ds4_gpu_tensor_alloc(pc * sizeof(int32_t));
    /* Shared multi-row logits slab (16 rows).  Unconditional, NOT gated on
     * speculation: every batched multi-row output head writes its rows here —
     * the DSpark draft/verify passes, gpu_graph_verify_suffix_tops, and the
     * Tier-2 batched multi-session decode driver.  It used to be allocated
     * only by gpu_graph_init_dspark_target (session create, dspark_ready
     * only), which left the multiseq driver rejecting every step whenever
     * speculation was off. */
    g->spec_logits = ds4_gpu_tensor_alloc(16ull * DS4_N_VOCAB * sizeof(float));
    g->batch_cur_hc = ds4_gpu_tensor_alloc(pc * hc_dim * DS4_HC_ELT_SIZE);   /* HC residual carrier */
    g->batch_next_hc = ds4_gpu_tensor_alloc(pc * hc_dim * DS4_HC_ELT_SIZE);   /* HC residual carrier */
    g->batch_flat_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_hc_mix = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
    g->batch_hc_split = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
    g->batch_attn_cur = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_attn_norm = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_qr = ds4_gpu_tensor_alloc(pc * q_rank * sizeof(float));
    g->batch_qr_norm = ds4_gpu_tensor_alloc(pc * q_rank * sizeof(float));
    g->batch_q = ds4_gpu_tensor_alloc(pc * q_dim * sizeof(float));
    g->batch_kv_raw = ds4_gpu_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
    g->batch_kv = ds4_gpu_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
    g->batch_comp_kv = ds4_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
    g->batch_comp_sc = ds4_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
    g->batch_indexer_q = ds4_gpu_tensor_alloc(pc * indexer_q_dim * sizeof(float));
    g->batch_indexer_weights = ds4_gpu_tensor_alloc(pc * DS4_N_INDEXER_HEAD * sizeof(float));
    g->batch_heads = ds4_gpu_tensor_alloc(pc * q_dim * sizeof(float));
    g->batch_attn_low = ds4_gpu_tensor_alloc(pc * low_dim * sizeof(float));
    g->batch_attn_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_after_attn_hc = ds4_gpu_tensor_alloc(pc * hc_dim * DS4_HC_ELT_SIZE);   /* HC residual carrier */
    g->batch_ffn_cur = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_ffn_norm = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_shared_gate = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_up = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_mid = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_router_logits = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
    g->batch_router_probs = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
    g->batch_router_selected = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(int));
    g->batch_router_weights = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(float));
    g->batch_routed_gate = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_up = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_mid = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_down = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
    g->batch_routed_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));

    bool layer_cache_ok = true;
    for (uint32_t il = 0; layer_cache_ok && il < DS4_N_LAYER; il++) {
        layer_cache_ok = g->layer_raw_cache[il] != NULL;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (layer_cache_ok && ratio != 0) {
            layer_cache_ok = g->layer_attn_comp_cache[il] != NULL &&
                             g->layer_attn_state_kv[il] != NULL &&
                             g->layer_attn_state_score[il] != NULL &&
                             (!enable_spec ||
                              (g->spec_attn_state_kv[il] != NULL &&
                               g->spec_attn_state_score[il] != NULL));
        }
        if (layer_cache_ok && ratio == 4) {
            layer_cache_ok = g->layer_index_comp_cache[il] != NULL &&
                             (!gpu_graph_idx_fp4_enabled() || g->idx_comp_stage != NULL) &&
                             g->layer_index_state_kv[il] != NULL &&
                             g->layer_index_state_score[il] != NULL &&
                             (!enable_spec ||
                              (g->spec_index_state_kv[il] != NULL &&
                               g->spec_index_state_score[il] != NULL));
        }
    }

    const bool ok = state_init_ok && layer_cache_ok &&
                    g->cur_hc && g->flat_hc && g->hc_mix && g->hc_split &&
                    g->hc_pre && g->hc_post && g->hc_comb &&
                    g->attn_cur && g->attn_norm && g->qr && g->qr_norm &&
                    g->q && g->kv_raw && g->kv &&
                    g->comp_kv_cur && g->comp_sc_cur &&
                    (!gpu_graph_attn_pack_enabled() ||
                     (g->attn_comp_stage && g->attn_comp_dequant)) &&
                    g->indexer_q && g->indexer_weights && g->indexer_scores &&
                    g->comp_mask && g->comp_selected &&
                    g->heads && g->attn_low && g->attn_out &&
                    g->after_attn_hc && g->ffn_cur && g->ffn_norm &&
                    g->shared_gate && g->shared_up && g->shared_mid &&
                    g->shared_out &&
                    g->router_logits && g->router_probs && g->router_selected && g->router_weights &&
                    g->routed_gate && g->routed_up && g->routed_mid &&
                    g->routed_down && g->routed_out &&
                    g->after_ffn_hc &&
                    g->output_pre && g->output_weights && g->output_embd &&
                    g->output_norm && g->logits &&
                    g->prefill_tokens && g->spec_logits &&
                    g->batch_cur_hc && g->batch_next_hc && g->batch_flat_hc &&
                    g->batch_hc_mix && g->batch_hc_split &&
                    g->batch_attn_cur && g->batch_attn_norm &&
                    g->batch_qr && g->batch_qr_norm && g->batch_q &&
                    g->batch_kv_raw && g->batch_kv &&
                    g->batch_comp_kv && g->batch_comp_sc &&
                    g->batch_indexer_q && g->batch_indexer_weights &&
                    g->batch_heads && g->batch_attn_low && g->batch_attn_out &&
                    g->batch_after_attn_hc &&
                    g->batch_ffn_cur && g->batch_ffn_norm &&
                    g->batch_shared_gate && g->batch_shared_up &&
                    g->batch_shared_mid && g->batch_shared_out &&
                    g->batch_router_logits && g->batch_router_probs &&
                    g->batch_router_selected && g->batch_router_weights &&
                    g->batch_routed_gate && g->batch_routed_up &&
                    g->batch_routed_mid && g->batch_routed_down &&
                    g->batch_routed_out;
    if (!ok) gpu_graph_free(g);
    return ok;
}

bool gpu_graph_init_dspark_target(ds4_gpu_graph *g, const uint32_t target_layer_ids[3]) {
    bool ok = true;
    for (int i = 0; i < 3; i++) {
        g->dspark_target_layer_ids[i] = target_layer_ids[i];
        g->dspark_target_h[i] = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        /* Fused-loop batch capture: one anchor hidden per verify-batch position
         * (17 = the spec block clamp of 16 drafts + first_token). */
        g->dspark_target_h_batch[i] = ds4_gpu_tensor_alloc(
            (uint64_t)17 * DS4_N_EMBD * sizeof(float));
        /* Option F: bank the drafter ring when the pool is enabled — one
         * bank-major slab, dspark_raw_cache[i] becomes bank 0's view (repoint
         * swaps it). */
        if (g->banks.n_banks != 0) {
            g->banks.dspark_raw_bank_bytes =
                (uint64_t)DS4_DSPARK_DRAFT_WINDOW * DS4_N_HEAD_DIM * sizeof(float);
            g->banks.dspark_raw[i] = ds4_gpu_tensor_alloc(
                (uint64_t)g->banks.n_banks * g->banks.dspark_raw_bank_bytes);
            g->dspark_raw_cache[i] = g->banks.dspark_raw[i]
                ? ds4_gpu_tensor_view(g->banks.dspark_raw[i], 0,
                                      g->banks.dspark_raw_bank_bytes)
                : NULL;
        } else {
            g->dspark_raw_cache[i] = ds4_gpu_tensor_alloc(
                (uint64_t)DS4_DSPARK_DRAFT_WINDOW * DS4_N_HEAD_DIM * sizeof(float));
        }
        g->dspark_n_raw[i] = 0;
        ok = ok && g->dspark_target_h[i] && g->dspark_target_h_batch[i] &&
             g->dspark_raw_cache[i];
    }
    g->dspark_capture_batch_n = 0;
    g->dspark_main_x = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    ok = ok && g->dspark_main_x;
    /* Bulk prefill capture buffers: always allocated when the drafter is
     * loaded (~200 MB at prefill_cap 4096) -- the prompt-window seeding
     * reduces every chunk through them; DS4_DSPARK_PREFILL_DUMP additionally
     * streams them to disk for retraining data. */
    g->dspark_bulk_n = 0;
    for (int i = 0; i < 3; i++) {
        g->dspark_bulk_h[i] = ds4_gpu_tensor_alloc(
            (uint64_t)g->prefill_cap * DS4_N_EMBD * sizeof(float));
        ok = ok && g->dspark_bulk_h[i];
    }
    /* Prompt-window ring: last <=128 prompt positions' anchor hiddens. */
    g->dspark_prompt_n = 0;
    for (int i = 0; i < 3; i++) {
        if (g->banks.n_banks != 0) {
            g->banks.dspark_prompt_bank_bytes =
                (uint64_t)DS4_DSPARK_DRAFT_WINDOW * DS4_N_EMBD * sizeof(float);
            g->banks.dspark_prompt[i] = ds4_gpu_tensor_alloc(
                (uint64_t)g->banks.n_banks * g->banks.dspark_prompt_bank_bytes);
            g->dspark_prompt_h[i] = g->banks.dspark_prompt[i]
                ? ds4_gpu_tensor_view(g->banks.dspark_prompt[i], 0,
                                      g->banks.dspark_prompt_bank_bytes)
                : NULL;
        } else {
            g->dspark_prompt_h[i] = ds4_gpu_tensor_alloc(
                (uint64_t)DS4_DSPARK_DRAFT_WINDOW * DS4_N_EMBD * sizeof(float));
        }
        ok = ok && g->dspark_prompt_h[i];
    }
    /* Stage-B no-replay rollback: per-position compressor projection saves for
     * every compressed layer (attn comp_width <= 2*DS4_N_HEAD_DIM; indexer width
     * = 2*DS4_N_INDEXER_HEAD_DIM) + one emit-sink scratch row. ~8 MB total. */
    {
        const uint64_t attn_w = 2ull * DS4_N_HEAD_DIM;
        const uint64_t idx_w = 2ull * DS4_N_INDEXER_HEAD_DIM;
        for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            g->spec_comp_kv_save[il] = NULL;
            g->spec_comp_sc_save[il] = NULL;
            g->spec_icomp_kv_save[il] = NULL;
            g->spec_icomp_sc_save[il] = NULL;
            if (ratio == 0) continue;
            g->spec_comp_kv_save[il] = ds4_gpu_tensor_alloc(17ull * attn_w * sizeof(float));
            g->spec_comp_sc_save[il] = ds4_gpu_tensor_alloc(17ull * attn_w * sizeof(float));
            ok = ok && g->spec_comp_kv_save[il] && g->spec_comp_sc_save[il];
            if (ratio == 4) {
                g->spec_icomp_kv_save[il] = ds4_gpu_tensor_alloc(17ull * idx_w * sizeof(float));
                g->spec_icomp_sc_save[il] = ds4_gpu_tensor_alloc(17ull * idx_w * sizeof(float));
                ok = ok && g->spec_icomp_kv_save[il] && g->spec_icomp_sc_save[il];
            }
        }
        g->spec_comp_scratch_row = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
        ok = ok && g->spec_comp_scratch_row;
        g->spec_comp_save_n = 0;
        g->dspark_concat = ds4_gpu_tensor_alloc(3ull * DS4_N_EMBD * sizeof(float));
        g->dspark_proj_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->dspark_seed_kv = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
        g->dspark_seed_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
        g->dspark_seed_rot = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
        g->dspark_markov_logits = ds4_gpu_tensor_alloc((uint64_t)DS4_N_VOCAB * sizeof(float));
        ok = ok && g->dspark_concat && g->dspark_proj_out && g->dspark_seed_kv &&
             g->dspark_seed_norm && g->dspark_seed_rot && g->dspark_markov_logits;
    }
    /* spec_logits is NOT allocated here: it is the shared multi-row logits
     * slab, allocated unconditionally by gpu_graph_alloc_raw_cap (the batched
     * decode driver needs it with speculation disabled). */
    return ok;
}

