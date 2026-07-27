#include "ds4_engine_internal.h"


/* Read an HC residual CARRIER (BF16 storage under task #62) into an f32 host
 * buffer, expanding each stored sample (BF16->f32 is an exact bit-extension:
 * the stored 16 bits are the high half of the f32). Used ONLY by the dev-only
 * layer-0 parity self-test (gpu_graph_decode_test) and the env-gated DSpark
 * dumps — never the production decode path. In the DS4_HC_F32 fallback build the
 * carrier is already f32, so it is a plain read. n is a sample count. */
/* Host-side f32 -> HC carrier store (task #62). Round-to-nearest-even so a host
 * staged write matches the GPU's __float2bfloat16 store path. NaN is
 * CANONICALIZED to 0x7FFF, which is what cvt.rn.bf16.f32 emits on sm_80+ (and
 * what CUDA's software path returns) — passing the payload through in the high
 * half would NOT match. Inf needs no special case: it rounds exactly through
 * the RNE path below. `dst` is raw carrier bytes, n a sample count. */
void ds4_store_hc_carrier_f32(void *dst, const float *src, uint64_t n) {
#ifdef DS4_HC_F32
    memcpy(dst, src, (size_t)n * sizeof(float));
#else
    uint16_t *d = (uint16_t *)dst;
    for (uint64_t i = 0; i < n; i++) {
        uint32_t x;
        memcpy(&x, &src[i], sizeof(x));
        if ((x & 0x7FFFFFFFu) > 0x7F800000u) {       /* NaN -> canonical */
            d[i] = 0x7FFFu;
        } else {
            const uint32_t bias = 0x7FFFu + ((x >> 16) & 1u);   /* RNE; Inf exact */
            d[i] = (uint16_t)((x + bias) >> 16);
        }
    }
#endif
}


int ds4_read_hc_carrier_f32(const ds4_gpu_tensor *t, uint64_t off_elems,
                            float *out, uint64_t n) {
#ifdef DS4_HC_F32
    return ds4_gpu_tensor_read((ds4_gpu_tensor *)t, off_elems * sizeof(float),
                               out, n * sizeof(float));
#else
    uint16_t *tmp = (uint16_t *)xmalloc((size_t)n * sizeof(uint16_t));
    int rc = ds4_gpu_tensor_read((ds4_gpu_tensor *)t, off_elems * DS4_HC_ELT_SIZE,
                                 tmp, n * DS4_HC_ELT_SIZE);
    if (rc == 0) {
        /* Read failed and left tmp uninitialized — do NOT convert it. Callers
         * (the DSpark dumps at session.c) memset `out` to zero beforehand and
         * ignore the return, relying on "zeros on failure"; converting garbage
         * here would silently write heap noise into the dump, and the f32
         * fallback build leaves `out` untouched in the same case. */
        free(tmp);
        return 0;
    }
    for (uint64_t i = 0; i < n; i++) {
        uint32_t bits = (uint32_t)tmp[i] << 16;
        memcpy(&out[i], &bits, sizeof(float));
    }
    free(tmp);
    return rc;
#endif
}




bool gpu_graph_use_reference_hc_decode(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_CUDA_DISABLE_HC_FUSION", &cache);
}



static bool gpu_graph_use_reference_kv_decode(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_CUDA_DISABLE_KV_FUSION", &cache);
}



bool gpu_graph_use_reference_qkv_norm(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_CUDA_DISABLE_QKV_NORM_FUSION", &cache);
}



static bool gpu_graph_use_reference_compressor_pair_proj(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_CUDA_DISABLE_COMPRESSOR_PAIR_PROJ", &cache);
}



static bool gpu_graph_use_reference_hc_norm_decode(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_CUDA_DISABLE_HC_NORM_FUSION", &cache);
}



bool gpu_graph_enable_batch_hc_norm_fusion(void) {
    return !gpu_graph_use_reference_hc_norm_decode();
}



static bool gpu_graph_use_reference_shared_down_hc(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_CUDA_DISABLE_SHARED_DOWN_HC_FUSION", &cache);
}



static bool gpu_graph_use_reference_attn_out_hc(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_CUDA_DISABLE_ATTN_OUT_HC_FUSION", &cache);
}

/* Evaluated every layer on the decode path; cache the flag reads (like the
 * fusion toggles above) instead of scanning environ per layer. */
static bool gpu_graph_disable_shared_gate_up_swiglu(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_CUDA_DISABLE_SHARED_GATE_UP_SWIGLU_FUSION", &cache);
}




static bool gpu_graph_decode_hc_pre(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const ds4_model        *model,
        uint64_t                scale_offset,
        uint64_t                base_offset) {
    if (gpu_graph_use_reference_hc_decode()) {
        return ds4_gpu_hc_split_sinkhorn_tensor(split,
                                                  mix,
                                                  model->map,
                                                  model->size,
                                                  scale_offset,
                                                  base_offset,
                                                  DS4_N_HC,
                                                  DS4_N_HC_SINKHORN_ITER,
                                                  DS4_HC_EPS) != 0 &&
               ds4_gpu_hc_weighted_sum_tensor(out,
                                                 residual_hc,
                                                 split,
                                                 DS4_N_EMBD,
                                                 DS4_N_HC) != 0;
    }

    return ds4_gpu_hc_split_weighted_sum_tensor(out,
                                                  split,
                                                  mix,
                                                  residual_hc,
                                                  model->map,
                                                  model->size,
                                                  scale_offset,
                                                  base_offset,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC,
                                                  DS4_N_HC_SINKHORN_ITER,
                                                  DS4_HC_EPS) != 0;
}



static bool gpu_graph_hc_norm_fusion_check_enabled(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_CUDA_HC_NORM_FUSION_CHECK", &cache);
}



static float gpu_graph_hc_norm_fusion_check_tolerance(void) {
    static int initialized;
    static float tolerance;
    if (initialized) return tolerance;
    tolerance = 2.0e-4f;
    const char *env = getenv("DS4_CUDA_HC_NORM_FUSION_CHECK_TOL");
    if (env && env[0]) {
        char *end = NULL;
        const float v = strtof(env, &end);
        if (end != env && isfinite(v) && v > 0.0f) tolerance = v;
    }
    initialized = 1;
    return tolerance;
}



static bool gpu_graph_check_hc_norm_fusion(
        const char            *label,
        ds4_gpu_tensor        *fused_out,
        ds4_gpu_tensor        *fused_norm,
        const ds4_gpu_tensor  *mix,
        const ds4_gpu_tensor  *residual_hc,
        const ds4_model       *model,
        uint64_t               scale_offset,
        uint64_t               base_offset,
        uint64_t               norm_weight_offset,
        uint32_t               il,
        uint32_t               pos) {
    if (!gpu_graph_hc_norm_fusion_check_enabled()) return true;
    if (!fused_out || !fused_norm || !mix || !residual_hc || !model) return false;

    const uint64_t n_embd = DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    ds4_gpu_tensor *ref_split = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    ds4_gpu_tensor *ref_out = ds4_gpu_tensor_alloc(n_embd * sizeof(float));
    ds4_gpu_tensor *ref_norm = ds4_gpu_tensor_alloc(n_embd * sizeof(float));
    bool ok = ref_split && ref_out && ref_norm;

    if (ok) {
        ok = ds4_gpu_hc_split_sinkhorn_tensor(ref_split,
                                              mix,
                                              model->map,
                                              model->size,
                                              scale_offset,
                                              base_offset,
                                              DS4_N_HC,
                                              DS4_N_HC_SINKHORN_ITER,
                                              DS4_HC_EPS) != 0 &&
             ds4_gpu_hc_weighted_sum_tensor(ref_out,
                                            residual_hc,
                                            ref_split,
                                            DS4_N_EMBD,
                                            DS4_N_HC) != 0 &&
             ds4_gpu_rms_norm_weight_tensor(ref_norm,
                                            ref_out,
                                            model->map,
                                            model->size,
                                            norm_weight_offset,
                                            DS4_N_EMBD,
                                            DS4_RMS_EPS) != 0;
    }

    if (ok) ok = ds4_gpu_end_commands() != 0;

    float *fused_out_cpu = NULL;
    float *ref_out_cpu = NULL;
    float *fused_norm_cpu = NULL;
    float *ref_norm_cpu = NULL;
    if (ok) {
        fused_out_cpu = (float *)xmalloc((size_t)n_embd * sizeof(float));
        ref_out_cpu = (float *)xmalloc((size_t)n_embd * sizeof(float));
        fused_norm_cpu = (float *)xmalloc((size_t)n_embd * sizeof(float));
        ref_norm_cpu = (float *)xmalloc((size_t)n_embd * sizeof(float));
        ok = ds4_gpu_tensor_read(fused_out, 0, fused_out_cpu, n_embd * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(ref_out, 0, ref_out_cpu, n_embd * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(fused_norm, 0, fused_norm_cpu, n_embd * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(ref_norm, 0, ref_norm_cpu, n_embd * sizeof(float)) != 0;
    }

    if (ok) {
        const float out_max = max_abs_diff(fused_out_cpu, ref_out_cpu, n_embd);
        const float out_rms = rms_abs_diff(fused_out_cpu, ref_out_cpu, n_embd);
        const float norm_max = max_abs_diff(fused_norm_cpu, ref_norm_cpu, n_embd);
        const float norm_rms = rms_abs_diff(fused_norm_cpu, ref_norm_cpu, n_embd);
        const float tol = gpu_graph_hc_norm_fusion_check_tolerance();
        fprintf(stderr,
                "ds4: GPU HC norm fusion check %s layer=%u pos=%u "
                "out_max=%g out_rms=%g norm_max=%g norm_rms=%g tol=%g\n",
                label ? label : "hc",
                il,
                pos,
                out_max,
                out_rms,
                norm_max,
                norm_rms,
                tol);
        if (out_max > tol || norm_max > tol) {
            fprintf(stderr,
                    "ds4: GPU HC norm fusion check failed for %s layer=%u pos=%u\n",
                    label ? label : "hc",
                    il,
                    pos);
            ok = false;
        }
    }

    free(fused_out_cpu);
    free(ref_out_cpu);
    free(fused_norm_cpu);
    free(ref_norm_cpu);
    ds4_gpu_tensor_free(ref_norm);
    ds4_gpu_tensor_free(ref_out);
    ds4_gpu_tensor_free(ref_split);

    const bool restart_ok = ds4_gpu_begin_commands() != 0;
    return ok && restart_ok;
}



static bool gpu_graph_decode_kv_store(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          raw_row,
        uint32_t          raw_f16) {
    if (gpu_graph_use_reference_kv_decode()) {
        return ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv, 1, DS4_N_HEAD_DIM, DS4_N_ROT) != 0 &&
               ds4_gpu_store_raw_kv_tensor(raw_cache, kv, raw_cap, raw_row, DS4_N_HEAD_DIM, raw_f16) != 0;
    }

    return ds4_gpu_kv_fp8_store_raw_tensor(kv,
                                             raw_cache,
                                             raw_cap,
                                             raw_row,
                                             DS4_N_HEAD_DIM,
                                             DS4_N_ROT,
                                             raw_f16) != 0;
}



/* The validated packed storage formats and the prefill work-buffer slice
 * default ON (2026-07-09: each proven bit-exact by golden byte-compare; see
 * commits 0647621, 34f6a95, 87f8374, 39c6526).  Their env vars are
 * OFF-switches: unset or "1"/"on" enables, "0"/"off"/"false" restores the
 * classic f32 containers — the escape hatch for regression bisects. */
static int gpu_graph_env_default_flag(const char *name, int def) {
    const char *v = getenv(name);
    if (!v || !v[0]) return def;
    if (strcmp(v, "0") == 0 || strcasecmp(v, "off") == 0 ||
        strcasecmp(v, "false") == 0) return 0;
    return 1;
}

int gpu_graph_attn_pack_enabled(void) {
    static int cached = -1;
    if (cached < 0) cached = gpu_graph_env_default_flag("DS4_ATTN_PACK", 1);
    return cached;
}

int gpu_graph_idx_fp4_enabled(void) {
    static int cached = -1;
    if (cached < 0) cached = gpu_graph_env_default_flag("DS4_IDX_FP4", 1);
    return cached;
}

int gpu_graph_raw_f16_enabled(void) {
    static int cached = -1;
    if (cached < 0) cached = gpu_graph_env_default_flag("DS4_RAW_F16", 1);
    return cached;
}

/* DS4_DECODE_DESCR: Tier-2 A/B diagnostic level (env read once per process
 * and cached — never a per-token getenv).  Not a production mode.
 *
 * Level 1: single-token decode attention AND indexer scan route through the
 * descriptor (banked) entry points as an n_banks=1 pool over the installed
 * cache views (positions=[pos], seq_id=[0]).  At n_tokens==1 the banked
 * dispatch selects the SAME kernels as classic, so this level is byte-exact
 * vs classic under the default configuration — the descriptor-vs-classic
 * gate.
 *
 * Level 2: additionally arms the banked MULTISEQ machinery for the
 * spec-verify batch (imatrix.c: one-bank batch over the current bank —
 * per-bank frontier bookkeeping, banked emit loop, banked raw scatter,
 * banked multi-token attention/indexer with packed-native comp reads).
 * Banked multi-token rows force the generic kernel tiers (WMMA / indexed
 * heads8 stay single-bank), so level 2 is byte-exact vs classic ONLY when
 * both runs pin those tiers off (DS4_CUDA_NO_INDEXER_WMMA=1 and
 * DS4_CUDA_NO_INDEXED_HEADS8=1) — how the Tier-2 gate runs it. */
int gpu_graph_decode_descr_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("DS4_DECODE_DESCR");
        if (v && strcmp(v, "2") == 0) cached = 2;
        else cached = gpu_graph_env_default_flag("DS4_DECODE_DESCR", 0);
    }
    return cached;
}

/* Lazily allocate the 1-row descriptor arrays and refresh positions[0] = pos
 * (seq_id[0] stays 0: the single session lives in bank 0). */
static bool gpu_graph_decode_descr_prepare(ds4_gpu_graph *g, uint32_t pos) {
    if (!g->descr_diag_pos) {
        g->descr_diag_pos = ds4_gpu_tensor_alloc(sizeof(int32_t));
        g->descr_diag_seq = ds4_gpu_tensor_alloc(sizeof(int32_t));
        const int32_t bank0 = 0;
        if (!g->descr_diag_pos || !g->descr_diag_seq ||
            !ds4_gpu_tensor_write(g->descr_diag_seq, 0, &bank0, sizeof(bank0))) {
            fprintf(stderr, "ds4: DS4_DECODE_DESCR descriptor alloc failed\n");
            /* Release and reset BOTH so a later call retries the whole block
             * instead of keying off a half-allocated descr_diag_pos and
             * failing forever (descr_diag_seq NULL / unwritten). */
            ds4_gpu_tensor_free(g->descr_diag_pos);
            ds4_gpu_tensor_free(g->descr_diag_seq);
            g->descr_diag_pos = NULL;
            g->descr_diag_seq = NULL;
            return false;
        }
    }
    const int32_t p = (int32_t)pos;
    return ds4_gpu_tensor_write(g->descr_diag_pos, 0, &p, sizeof(p)) != 0;
}

/* DS4_PREFILL_SLICE=<N>: process the prefill [indexer score -> top-k ->
 * indexed attention] sequence in <=N-token slices so the two ctx-scaling f32
 * work buffers (indexer_scores, comp_mask) are allocated with only N token
 * rows instead of prefill_cap.  Defaults to 512 (validated bit-exact);
 * 0 restores the historical full-chunk buffers. */
uint32_t gpu_graph_prefill_slice(void) {
    static long cached = -1;
    if (cached < 0) {
        const char *e = getenv("DS4_PREFILL_SLICE");
        long v = (e && e[0]) ? strtol(e, NULL, 10) : 512;
        cached = v > 0 ? v : 0;
    }
    return (uint32_t)cached;
}

/* Diagnostic raw-ring row read as f32 regardless of the active storage format. */
static int gpu_graph_read_raw_row_f32(ds4_gpu_graph *g, uint32_t il, uint32_t phys, float *out) {
    if (!gpu_graph_raw_f16_enabled()) {
        return ds4_gpu_tensor_read(g->layer_raw_cache[il],
                                   (uint64_t)phys * DS4_N_HEAD_DIM * sizeof(float),
                                   out, (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    }
    uint16_t *tmp = (uint16_t *)xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(uint16_t));
    int ok = ds4_gpu_tensor_read(g->layer_raw_cache[il],
                                 (uint64_t)phys * DS4_N_HEAD_DIM * sizeof(uint16_t),
                                 tmp, (uint64_t)DS4_N_HEAD_DIM * sizeof(uint16_t));
    if (ok != 0) {
        for (uint32_t d = 0; d < DS4_N_HEAD_DIM; d++) out[d] = f16_to_f32(tmp[d]);
    }
    free(tmp);
    return ok;
}

uint64_t gpu_graph_attn_comp_cache_row_bytes(void) {
    if (gpu_graph_attn_pack_enabled()) return DS4_ENGINE_ATTN_PACK_ROWBYTES;
    return (uint64_t)DS4_N_HEAD_DIM * sizeof(float);
}

/* Comp cache to hand the f32 prefill attention consumers. Normally the
 * persistent cache itself; under DS4_ATTN_PACK storage, dequantize the first
 * n_rows packed rows into the f32 shadow and return that.  The dequant is
 * bit-exact: packed rows decode to exactly the values the f32 cache would
 * hold. */
ds4_gpu_tensor *gpu_graph_attn_comp_read_cache(ds4_gpu_graph *g, uint32_t il, uint32_t n_rows) {
    if (!g || il >= DS4_N_LAYER) return NULL;
    if (!gpu_graph_attn_pack_enabled()) return g->layer_attn_comp_cache[il];
    if (!g->attn_comp_dequant) return NULL;
    if (n_rows == 0) return g->attn_comp_dequant;
    if (n_rows > g->layer_comp_cap[il]) return NULL;
    if (ds4_gpu_attn_pack_dequant_tensor(g->layer_attn_comp_cache[il], g->attn_comp_dequant,
                                         n_rows, DS4_N_HEAD_DIM, DS4_N_ROT) == 0) {
        return NULL;
    }
    return g->attn_comp_dequant;
}

/* Format flag for consumers reading the PERSISTENT comp cache natively (the
 * single-token decode attention).  The prefill/batch consumers read the f32
 * shadow instead (gpu_graph_attn_comp_read_cache) and pass 0. */
uint32_t gpu_graph_attn_comp_cache_is_pack(void) {
    return gpu_graph_attn_pack_enabled() ? 1u : 0u;
}
static bool gpu_graph_weight_is_plain_or_mxfp8(const ds4_tensor *w) {
    return w->type == DS4_TENSOR_F16 || w->type == DS4_TENSOR_FP8_E4M3;
}




ds4_gpu_tensor *gpu_graph_attn_comp_update_target(
        ds4_gpu_graph *g,
        uint32_t       il) {
    return gpu_graph_attn_pack_enabled()
        ? g->attn_comp_stage
        : g->layer_attn_comp_cache[il];
}



uint32_t gpu_graph_attn_comp_update_row(uint32_t row) {
    return gpu_graph_attn_pack_enabled() ? 0u : row;
}



bool gpu_graph_commit_attn_comp_stage(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    if (gpu_graph_attn_pack_enabled()) {
        /* Quantize+pack the `rows` f32 rows staged in attn_comp_stage into the
         * packed comp cache at first_row.  The kernel also fp8-roundtrips the
         * stage rows in place (identical to the plain quantize the f32 path
         * runs), so the stage keeps the exact f32-pipeline values.  This MUST
         * be the only fp8 quantize of the row: re-quantizing an already-
         * roundtripped block is NOT bit-idempotent when the block amax sits on
         * a scale boundary (the recomputed scale can shift one step and
         * re-round small values, e.g. subnormal ties) — that is why the
         * prefill/replay producers pass quantize_fp8=false under pack. */
        if (rows == 0) return true;
        if (!g || il >= DS4_N_LAYER || !g->layer_attn_comp_cache[il] || !g->attn_comp_stage) {
            return false;
        }
        if (first_row > g->layer_comp_cap[il] || rows > g->layer_comp_cap[il] - first_row) {
            return false;
        }
        if (ds4_gpu_attn_pack_quantize_store_tensor(g->attn_comp_stage,
                                                    g->layer_attn_comp_cache[il],
                                                    first_row, rows,
                                                    DS4_N_HEAD_DIM, DS4_N_ROT) == 0) {
            return false;
        }
        /* plan-33 inc C: byte-replace the ratio-4 boundary row after any commit
         * below the fork keep threshold (no-op when ms_emit_keep is 0). */
        return gpu_graph_emit_keep_restore(g, il,
                g->banks.n_banks ? g->banks.cur_bank : 0u, first_row, rows, false);
    }
    /* Classic f32 storage: the compressor wrote the persistent cache directly
     * (gpu_graph_attn_comp_update_target returned it), nothing to commit — but
     * the fork boundary restore still applies to the rows it just wrote. */
    return gpu_graph_emit_keep_restore(g, il,
            g->banks.n_banks ? g->banks.cur_bank : 0u, first_row, rows, false);
}



bool gpu_graph_commit_attn_comp_stage_bank(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       bank,
        uint32_t       first_row,
        uint32_t       rows) {
    if (!gpu_graph_attn_pack_enabled()) {
        /* f32 storage: the multiseq emit loop pointed the compressor at the
         * bank's comp view directly; nothing to commit (mirrors the classic
         * helper). */
        return true;
    }
    if (rows == 0) return true;
    if (!g || il >= DS4_N_LAYER || !g->attn_comp_stage) return false;
    if (first_row > g->layer_comp_cap[il] || rows > g->layer_comp_cap[il] - first_row) {
        return false;
    }
    ds4_gpu_tensor *cache = gpu_graph_bank_attn_comp_view(g, il, bank);
    if (!cache) return false;
    const bool ok = ds4_gpu_attn_pack_quantize_store_tensor(
            g->attn_comp_stage, cache, first_row, rows,
            DS4_N_HEAD_DIM, DS4_N_ROT) != 0;
    ds4_gpu_tensor_free(cache);
    /* plan-33 inc C: boundary-row restore for the explicit-bank commit path. */
    return ok && gpu_graph_emit_keep_restore(g, il, bank, first_row, rows, false);
}



ds4_gpu_tensor *gpu_graph_attn_comp_row_view(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       row) {
    if (gpu_graph_attn_pack_enabled()) {
        return ds4_gpu_tensor_view(g->attn_comp_stage,
                                   0,
                                   (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    }
    return ds4_gpu_tensor_view(g->layer_attn_comp_cache[il],
                               (uint64_t)row * DS4_N_HEAD_DIM * sizeof(float),
                               (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
}



ds4_gpu_tensor *gpu_graph_attn_comp_prefill_target(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    if (gpu_graph_attn_pack_enabled()) {
        return g->attn_comp_stage;
    }
    const uint32_t view_rows = rows ? rows : 1u;
    return ds4_gpu_tensor_view(g->layer_attn_comp_cache[il],
                               (uint64_t)first_row * DS4_N_HEAD_DIM * sizeof(float),
                               (uint64_t)view_rows * DS4_N_HEAD_DIM * sizeof(float));
}



void gpu_graph_attn_comp_prefill_target_free(ds4_gpu_tensor *t) {
    /* Only the pure-f32 path returns a fresh view; the staged pack path
     * returns the persistent attn_comp_stage, which must not be freed. */
    if (!gpu_graph_attn_pack_enabled()) {
        ds4_gpu_tensor_free(t);
    }
}



/* Encode one DS4 decode layer on GPU.  This is the release single-token
 * layer path; diagnostics reuse it so they compare exactly what generation
 * runs. */
bool gpu_graph_indexer_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        uint32_t    n_comp,
        double     *stage_t0);


bool gpu_graph_layer_stage_profile_boundary(
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0);


bool gpu_graph_decode_stage_profile_enabled(uint32_t il);


bool gpu_graph_matmul_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);



/* Decode-only fused RMSNorm + HC-mix GEMV.  Byte-identical to the
 * rms_norm_plain -> matmul_f16 pair it replaces (see ds4_cuda_hc_router.cu);
 * it exists because that pair ran a 1-block kernel and then a 24-block kernel
 * with a 64 KB f32 scratch round trip between them, for ~5.4% of decode.
 * src_hc is an HC residual carrier (BF16 under task #62); the fused kernel
 * reads it via ds4_hc_load, exactly as rms_norm_plain_tensor does.
 * Non-F16 mix weights keep the original two-kernel path. */
static bool gpu_graph_norm_mix_plain(
        ds4_gpu_graph        *g,
        const ds4_model      *model,
        const ds4_tensor     *w,
        uint64_t              hc_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *src_hc,
        ds4_gpu_tensor       *out) {
    if (w->type == DS4_TENSOR_F16) {
        return ds4_gpu_hc_norm_mix_f16_tensor(out, model->map, model->size,
                                              w->abs_offset, hc_dim, out_dim,
                                              src_hc, DS4_RMS_EPS) != 0;
    }
    if (!ds4_gpu_rms_norm_plain_tensor(g->flat_hc, src_hc, (uint32_t)hc_dim, DS4_RMS_EPS)) return false;
    return gpu_graph_matmul_plain_tensor(out, model, w, hc_dim, out_dim, g->flat_hc, 1);
}


bool gpu_graph_encode_decode_layer(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos,
        ds4_gpu_tensor       *raw_cache,
        uint32_t                raw_cap,
        uint32_t                raw_row,
        uint32_t                n_raw,
        int                     token) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint32_t raw_f16 = (uint32_t)gpu_graph_raw_f16_enabled();
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t n_groups = DS4_N_OUT_GROUP;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = DS4_N_LORA_O;
    const uint32_t shared_dim = (uint32_t)layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    const bool compressed = g->comp_ratio_override >= 0
        ? g->comp_ratio_override != 0
        : ds4_layer_compress_ratio(il) != 0;
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    const bool qkv_rms_fused = !gpu_graph_use_reference_qkv_norm();

    bool ok = true;
    const bool decode_stage_profile = gpu_graph_decode_stage_profile_enabled(il);
    double decode_stage_t0 = decode_stage_profile ? now_sec() : 0.0;
#define DS4_CUDA_PROFILE_DECODE_STAGE(name) do { \
        if (ok && decode_stage_profile) { \
            ok = gpu_graph_layer_stage_profile_boundary("decode", (name), il, pos, 1, &decode_stage_t0); \
        } \
    } while (0)
    if (ok) ok = gpu_graph_norm_mix_plain(g, model, layer->hc_attn_fn,
                                          hc_dim, mix_hc, g->cur_hc, g->hc_mix);
    const bool fuse_hc_norm =
        DS4_N_HC == 4 &&
        !gpu_graph_use_reference_hc_decode() &&
        !gpu_graph_use_reference_hc_norm_decode();
    if (ok && fuse_hc_norm) {
        ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(g->attn_cur,
                                                         g->attn_norm,
                                                         g->hc_split,
                                                         g->hc_mix,
                                                         g->cur_hc,
                                                         model->map,
                                                         model->size,
                                                         layer->hc_attn_scale->abs_offset,
                                                         layer->hc_attn_base->abs_offset,
                                                         layer->attn_norm->abs_offset,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC,
                                                         DS4_N_HC_SINKHORN_ITER,
                                                         DS4_HC_EPS,
                                                         DS4_RMS_EPS) != 0;
        if (ok) {
            ok = gpu_graph_check_hc_norm_fusion("attn",
                                                  g->attn_cur,
                                                  g->attn_norm,
                                                  g->hc_mix,
                                                  g->cur_hc,
                                                  model,
                                                  layer->hc_attn_scale->abs_offset,
                                                  layer->hc_attn_base->abs_offset,
                                                  layer->attn_norm->abs_offset,
                                                  il,
                                                  pos);
        }
    } else if (ok) {
        ok = gpu_graph_decode_hc_pre(g->attn_cur,
                                       g->hc_split,
                                       g->hc_mix,
                                       g->cur_hc,
                                       model,
                                       layer->hc_attn_scale->abs_offset,
                                       layer->hc_attn_base->abs_offset);
    }
    DS4_CUDA_PROFILE_DECODE_STAGE("attn_hc_pre");
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_attn_pre_mixes", g->hc_mix, mix_hc, il, pos);
        gpu_graph_debug_dump_tensor("hc_attn_pre_weights", g->hc_pre, DS4_N_HC, il, pos);
        gpu_graph_debug_dump_tensor("hc_attn_pre_post_weights", g->hc_post, DS4_N_HC, il, pos);
        gpu_graph_debug_dump_tensor("hc_attn_pre_comb", g->hc_comb, (uint64_t)DS4_N_HC * DS4_N_HC, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_attn_pre", g->attn_cur, DS4_N_EMBD, il, pos);
    }
    if (ok && !fuse_hc_norm) ok = ds4_gpu_rms_norm_weight_tensor(g->attn_norm, g->attn_cur,
                                                                   model->map, model->size,
                                                                   layer->attn_norm->abs_offset,
                                                                   DS4_N_EMBD, DS4_RMS_EPS) != 0;
    DS4_CUDA_PROFILE_DECODE_STAGE("attn_norm");
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_norm", g->attn_norm, DS4_N_EMBD, il, pos);
    }
    bool qkv_pair_projected = false;
    if (ok && qkv_rms_fused) {
        qkv_pair_projected = ds4_gpu_matmul_mxfp8_pair_tensor(g->qr,
                                                             g->kv_raw,
                                                             model->map,
                                                             model->size,
                                                             layer->attn_q_a->abs_offset,
                                                             layer->attn_kv->abs_offset,
                                                             DS4_N_EMBD,
                                                             q_rank,
                                                             DS4_N_HEAD_DIM,
                                                             g->attn_norm,
                                                             1) != 0;
    }
    if (ok && !qkv_pair_projected) ok = ds4_gpu_matmul_mxfp8_tensor(g->qr,
                                                                    model->map,
                                                                    model->size,
                                                                    layer->attn_q_a->abs_offset,
                                                                    DS4_N_EMBD,
                                                                    q_rank,
                                                                    g->attn_norm,
                                                                    1) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("q_lora", g->qr, q_rank, il, pos);
    }
    if (qkv_rms_fused) {
        if (ok && !qkv_pair_projected) ok = ds4_gpu_matmul_mxfp8_tensor(g->kv_raw, model->map, model->size,
                                                  layer->attn_kv->abs_offset,
                                                  DS4_N_EMBD, DS4_N_HEAD_DIM,
                                                  g->attn_norm, 1) != 0;
        if (ok) {
            gpu_graph_debug_dump_tensor("KVraw", g->kv_raw, DS4_N_HEAD_DIM, il, pos);
        }
        if (ok) ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(g->qr_norm,
                                                             g->qr,
                                                             model->map,
                                                             model->size,
                                                             layer->attn_q_a_norm->abs_offset,
                                                             (uint32_t)q_rank,
                                                             g->kv,
                                                             g->kv_raw,
                                                             layer->attn_kv_a_norm->abs_offset,
                                                             DS4_N_HEAD_DIM,
                                                             1,
                                                             DS4_RMS_EPS) != 0;
    } else {
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->qr_norm, g->qr,
                                                      model->map, model->size,
                                                      layer->attn_q_a_norm->abs_offset,
                                                      (uint32_t)q_rank, DS4_RMS_EPS) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("q_lora_norm", g->qr_norm, q_rank, il, pos);
    }
    if (qkv_rms_fused && ok) {
        gpu_graph_debug_dump_tensor("KVnorm", g->kv, DS4_N_HEAD_DIM, il, pos);
    }
    if (ok) ok = ds4_gpu_matmul_mxfp8_tensor(g->q, model->map, model->size,
                                              layer->attn_q_b->abs_offset,
                                              q_rank, q_dim,
                                              g->qr_norm, 1) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("Qraw", g->q, q_dim, il, pos);
    }
    const bool decode_q_norm_debug = gpu_graph_debug_wants("Qnorm", il, pos);
    bool decode_q_norm_rope_fused = false;
    if (ok && !decode_q_norm_debug) {
        decode_q_norm_rope_fused =
            ds4_gpu_head_rms_norm_rope_tail_tensor(g->q,
                                                   1,
                                                   DS4_N_HEAD,
                                                   DS4_N_HEAD_DIM,
                                                   DS4_N_ROT,
                                                   pos,
                                                   compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                   false,
                                                   freq_base,
                                                   freq_scale,
                                                   ext_factor,
                                                   attn_factor,
                                                   DS4_ROPE_YARN_BETA_FAST,
                                                   DS4_ROPE_YARN_BETA_SLOW,
                                                   DS4_RMS_EPS,
                                                   NULL) != 0;
    }
    if (!decode_q_norm_rope_fused) {
        if (ok) ok = ds4_gpu_head_rms_norm_tensor(g->q, 1, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_RMS_EPS) != 0;
        if (ok) {
            gpu_graph_debug_dump_tensor("Qnorm", g->q, q_dim, il, pos);
        }
        if (ok) ok = ds4_gpu_rope_tail_tensor(g->q, 1, DS4_N_HEAD, DS4_N_HEAD_DIM,
                                                DS4_N_ROT, pos,
                                                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                false, freq_base, freq_scale, ext_factor, attn_factor,
                                                DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW,
                                                NULL) != 0;
    }
    DS4_CUDA_PROFILE_DECODE_STAGE("q_path");
    if (ok) {
        gpu_graph_debug_dump_tensor("Qcur", g->q, q_dim, il, pos);
    }
    if (!qkv_rms_fused) {
        if (ok) ok = ds4_gpu_matmul_mxfp8_tensor(g->kv_raw, model->map, model->size,
                                                  layer->attn_kv->abs_offset,
                                                  DS4_N_EMBD, DS4_N_HEAD_DIM,
                                                  g->attn_norm, 1) != 0;
        if (ok) {
            gpu_graph_debug_dump_tensor("KVraw", g->kv_raw, DS4_N_HEAD_DIM, il, pos);
        }
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->kv, g->kv_raw,
                                                      model->map, model->size,
                                                      layer->attn_kv_a_norm->abs_offset,
                                                      DS4_N_HEAD_DIM, DS4_RMS_EPS) != 0;
        if (ok) {
            gpu_graph_debug_dump_tensor("KVnorm", g->kv, DS4_N_HEAD_DIM, il, pos);
        }
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->kv, 1, DS4_N_HEAD_KV, DS4_N_HEAD_DIM,
                                            DS4_N_ROT, pos,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false, freq_base, freq_scale, ext_factor, attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW,
                                            NULL) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("KVrope", g->kv, DS4_N_HEAD_DIM, il, pos);
    }
    /* RoPE stays as the exact standalone kernel above.  The decode fusion
     * starts after that, where FP8 KV quantization and raw-cache storage can
     * share one pass without changing the trigonometric path. */
    if (ok) ok = gpu_graph_decode_kv_store(g->kv, raw_cache, raw_cap, raw_row, raw_f16);
    DS4_CUDA_PROFILE_DECODE_STAGE("kv_path");
    if (ok) {
        gpu_graph_debug_dump_tensor("KVcur", g->kv, DS4_N_HEAD_DIM, il, pos);
    }

    uint32_t n_comp = 0;
    ds4_gpu_tensor *comp_cache = NULL;
    ds4_gpu_tensor *comp_selected = NULL;
    uint32_t n_selected = 0;
    double decode_index_stage_t0 = 0.0;
    static int decode_index_stage_env = -1;
    const bool decode_index_stage_profile = gpu_graph_env_flag("DS4_CUDA_INDEXER_STAGE_PROFILE", &decode_index_stage_env);
    /* DS4_DECODE_DESCR diagnostic: refresh the 1-row descriptor arrays once
     * per layer (both the banked indexer scan and the banked attention below
     * read them). */
    const int descr_diag = gpu_graph_decode_descr_enabled();
    if (ok && descr_diag) ok = gpu_graph_decode_descr_prepare(g, pos);
    if (ok && compressed) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint32_t comp_width = coff * DS4_N_HEAD_DIM;
        const bool emit = ((pos + 1u) % ratio) == 0u;
        if (!layer->attn_compressor_kv || !layer->attn_compressor_gate ||
            !layer->attn_compressor_ape || !layer->attn_compressor_norm ||
            !gpu_graph_weight_is_plain_or_mxfp8(layer->attn_compressor_kv) ||
            !gpu_graph_weight_is_plain_or_mxfp8(layer->attn_compressor_gate) ||
            layer->attn_compressor_kv->dim[0] != DS4_N_EMBD ||
            layer->attn_compressor_gate->dim[0] != DS4_N_EMBD ||
            layer->attn_compressor_kv->dim[1] != comp_width ||
            layer->attn_compressor_gate->dim[1] != comp_width) {
            fprintf(stderr, "ds4: GPU graph compressor expects paired F16 compressor projections\n");
            ok = false;
        }
        if (ok && emit && g->layer_n_comp[il] >= g->layer_comp_cap[il]) {
            fprintf(stderr, "ds4: GPU graph compressed KV cache capacity exceeded at layer %u\n", il);
            ok = false;
        }
        if (ok && !gpu_graph_use_reference_compressor_pair_proj()) {
            if (layer->attn_compressor_kv->type == DS4_TENSOR_F16) {
                ok = ds4_gpu_matmul_f16_pair_tensor(g->comp_kv_cur,
                                                      g->comp_sc_cur,
                                                      model->map,
                                                      model->size,
                                                      layer->attn_compressor_kv->abs_offset,
                                                      layer->attn_compressor_gate->abs_offset,
                                                      DS4_N_EMBD,
                                                      comp_width,
                                                      g->attn_norm,
                                                      1) != 0;
            } else {
                ok = gpu_graph_matmul_plain_tensor(g->comp_kv_cur, model,
                                                    layer->attn_compressor_kv,
                                                    DS4_N_EMBD, comp_width,
                                                    g->attn_norm, 1) &&
                     gpu_graph_matmul_plain_tensor(g->comp_sc_cur, model,
                                                    layer->attn_compressor_gate,
                                                    DS4_N_EMBD, comp_width,
                                                    g->attn_norm, 1);
            }
        } else {
            if (ok) ok = gpu_graph_matmul_plain_tensor(g->comp_kv_cur, model,
                                                         layer->attn_compressor_kv,
                                                         DS4_N_EMBD, comp_width,
                                                         g->attn_norm, 1);
            if (ok) ok = gpu_graph_matmul_plain_tensor(g->comp_sc_cur, model,
                                                         layer->attn_compressor_gate,
                                                         DS4_N_EMBD, comp_width,
                                                         g->attn_norm, 1);
        }
        const uint32_t comp_row = g->layer_n_comp[il];
        if (ok) ok = ds4_gpu_compressor_update_tensor(g->comp_kv_cur,
                                                        g->comp_sc_cur,
                                                        g->layer_attn_state_kv[il],
                                                        g->layer_attn_state_score[il],
                                                        gpu_graph_attn_comp_update_target(g, il),
                                                        model->map,
                                                        model->size,
                                                        layer->attn_compressor_ape->abs_offset,
                                                        layer->attn_compressor_ape->type,
                                                        layer->attn_compressor_norm->abs_offset,
                                                        layer->attn_compressor_norm->type,
                                                        DS4_N_HEAD_DIM,
                                                        ratio,
                                                        pos,
                                                        gpu_graph_attn_comp_update_row(comp_row),
                                                        DS4_N_ROT,
                                                        compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                        freq_base,
                                                        freq_scale,
                                                        ext_factor,
                                                        attn_factor,
                                                        DS4_ROPE_YARN_BETA_FAST,
                                                        DS4_ROPE_YARN_BETA_SLOW,
                                                        DS4_RMS_EPS) != 0;
        if (ok && emit) {
            ds4_gpu_tensor *comp_row_view = gpu_graph_attn_comp_row_view(g, il, comp_row);
            if (!comp_row_view) {
                ok = false;
            } else if (gpu_graph_attn_pack_enabled()) {
                /* comp_row_view aliases the f32 stage; commit below quantizes,
                 * packs, and roundtrips the stage in place — dump afterwards so
                 * KVcompress shows the same post-roundtrip values as f32 mode. */
            } else {
                ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_row_view, 1, DS4_N_HEAD_DIM, DS4_N_ROT) != 0;
                if (ok) {
                    gpu_graph_debug_dump_tensor("KVcompress", comp_row_view, DS4_N_HEAD_DIM, il, pos);
                }
            }
            if (ok) ok = gpu_graph_commit_attn_comp_stage(g, il, comp_row, 1);
            if (ok && comp_row_view && gpu_graph_attn_pack_enabled()) {
                gpu_graph_debug_dump_tensor("KVcompress", comp_row_view, DS4_N_HEAD_DIM, il, pos);
            }
            ds4_gpu_tensor_free(comp_row_view);
        }
        if (ok && emit) g->layer_n_comp[il]++;

        if (ok && ratio == 4) {
            const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_compressor_kv || !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape || !layer->indexer_compressor_norm ||
                !gpu_graph_weight_is_plain_or_mxfp8(layer->indexer_compressor_kv) ||
                !gpu_graph_weight_is_plain_or_mxfp8(layer->indexer_compressor_gate) ||
                layer->indexer_compressor_kv->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_gate->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_kv->dim[1] != index_width ||
                layer->indexer_compressor_gate->dim[1] != index_width) {
                fprintf(stderr, "ds4: GPU graph indexer compressor expects paired F16 projections\n");
                ok = false;
            }
            if (ok && emit && g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) {
                fprintf(stderr, "ds4: GPU graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                ok = false;
            }
            if (ok && !gpu_graph_use_reference_compressor_pair_proj()) {
                if (layer->indexer_compressor_kv->type == DS4_TENSOR_F16) {
                    ok = ds4_gpu_matmul_f16_pair_tensor(g->comp_kv_cur,
                                                          g->comp_sc_cur,
                                                          model->map,
                                                          model->size,
                                                          layer->indexer_compressor_kv->abs_offset,
                                                          layer->indexer_compressor_gate->abs_offset,
                                                          DS4_N_EMBD,
                                                          index_width,
                                                          g->attn_norm,
                                                          1) != 0;
                } else {
                    ok = gpu_graph_matmul_plain_tensor(g->comp_kv_cur, model,
                                                        layer->indexer_compressor_kv,
                                                        DS4_N_EMBD, index_width,
                                                        g->attn_norm, 1) &&
                         gpu_graph_matmul_plain_tensor(g->comp_sc_cur, model,
                                                        layer->indexer_compressor_gate,
                                                        DS4_N_EMBD, index_width,
                                                        g->attn_norm, 1);
                }
            } else {
                if (ok) ok = gpu_graph_matmul_plain_tensor(g->comp_kv_cur, model,
                                                             layer->indexer_compressor_kv,
                                                             DS4_N_EMBD, index_width,
                                                             g->attn_norm, 1);
                if (ok) ok = gpu_graph_matmul_plain_tensor(g->comp_sc_cur, model,
                                                             layer->indexer_compressor_gate,
                                                             DS4_N_EMBD, index_width,
                                                              g->attn_norm, 1);
            }
            const uint32_t index_row = g->layer_n_index_comp[il];
            const int idx_fp4 = gpu_graph_idx_fp4_enabled();
            if (ok) ok = ds4_gpu_compressor_update_tensor(g->comp_kv_cur,
                                                            g->comp_sc_cur,
                                                            g->layer_index_state_kv[il],
                                                            g->layer_index_state_score[il],
                                                            idx_fp4 ? g->idx_comp_stage
                                                                    : g->layer_index_comp_cache[il],
                                                            model->map,
                                                            model->size,
                                                            layer->indexer_compressor_ape->abs_offset,
                                                            layer->indexer_compressor_ape->type,
                                                            layer->indexer_compressor_norm->abs_offset,
                                                            layer->indexer_compressor_norm->type,
                                                            DS4_N_INDEXER_HEAD_DIM,
                                                            ratio,
                                                            pos,
                                                            index_row,
                                                            DS4_N_ROT,
                                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                            freq_base,
                                                            freq_scale,
                                                            ext_factor,
                                                            attn_factor,
                                                            DS4_ROPE_YARN_BETA_FAST,
                                                            DS4_ROPE_YARN_BETA_SLOW,
                                                            DS4_RMS_EPS) != 0;
            if (ok && emit) {
                ds4_gpu_tensor *index_row_view = ds4_gpu_tensor_view(
                        idx_fp4 ? g->idx_comp_stage : g->layer_index_comp_cache[il],
                        (uint64_t)index_row * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                        (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                if (!index_row_view) {
                    ok = false;
                } else if (idx_fp4) {
                    ok = ds4_gpu_dsv4_indexer_qat_pack_tensor(index_row_view,
                                                               g->layer_index_comp_cache[il],
                                                               index_row,
                                                               1,
                                                               DS4_N_INDEXER_HEAD_DIM) != 0;
                    ds4_gpu_tensor_free(index_row_view);
                } else {
                    ok = ds4_gpu_dsv4_indexer_qat_tensor(index_row_view,
                                                          1,
                                                          DS4_N_INDEXER_HEAD_DIM) != 0;
                    ds4_gpu_tensor_free(index_row_view);
                }
                /* plan-33 inc C: boundary-row restore (decode-fallback site). */
                if (ok) ok = gpu_graph_emit_keep_restore(g, il,
                        g->banks.n_banks ? g->banks.cur_bank : 0u, index_row, 1, true);
            }
            if (ok && emit) g->layer_n_index_comp[il]++;
            const uint32_t decode_sparse_threshold =
                gpu_graph_decode_indexer_sparse_threshold(g);
            if (ok &&
                g->layer_n_comp[il] > decode_sparse_threshold &&
                g->layer_n_index_comp[il] > DS4_N_INDEXER_TOP_K) {
                const uint64_t indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
                if (!layer->indexer_attn_q_b ||
                    !gpu_graph_weight_is_plain_or_mxfp8(layer->indexer_attn_q_b) ||
                    layer->indexer_attn_q_b->dim[0] != q_rank ||
                    layer->indexer_attn_q_b->dim[1] != indexer_q_dim) {
                    fprintf(stderr, "ds4: GPU graph indexer q projection expects F16 weights\n");
                    ok = false;
                }
                if (ok && (!layer->indexer_proj ||
                           !gpu_graph_weight_is_plain_or_mxfp8(layer->indexer_proj) ||
                           layer->indexer_proj->dim[0] != DS4_N_EMBD ||
                           layer->indexer_proj->dim[1] != DS4_N_INDEXER_HEAD)) {
                    fprintf(stderr, "ds4: GPU graph indexer weight projection expects F16 weights\n");
                    ok = false;
                }
                if (ok) ok = gpu_graph_matmul_plain_tensor(g->indexer_q,
                                                              model,
                                                              layer->indexer_attn_q_b,
                                                              q_rank,
                                                              indexer_q_dim,
                                                              g->qr_norm,
                                                              1);
                if (ok) ok = ds4_gpu_rope_tail_tensor(g->indexer_q, 1,
                                                        DS4_N_INDEXER_HEAD,
                                                        DS4_N_INDEXER_HEAD_DIM,
                                                        DS4_N_ROT,
                                                        pos,
                                                        compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                        false,
                                                        freq_base,
                                                        freq_scale,
                                                        ext_factor,
                                                        attn_factor,
                                                        DS4_ROPE_YARN_BETA_FAST,
                                                        DS4_ROPE_YARN_BETA_SLOW,
                                                        NULL) != 0;
                if (ok) ok = ds4_gpu_dsv4_indexer_qat_tensor(g->indexer_q,
                                                              DS4_N_INDEXER_HEAD,
                                                              DS4_N_INDEXER_HEAD_DIM) != 0;
                if (ok) ok = gpu_graph_matmul_plain_tensor(g->indexer_weights, model,
                                                             layer->indexer_proj,
                                                             DS4_N_EMBD, DS4_N_INDEXER_HEAD,
                                                             g->attn_norm, 1);
                const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                if (ok && decode_index_stage_profile) {
                    ok = gpu_graph_indexer_stage_profile_boundary(NULL,
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                if (ok && descr_diag) {
                    /* DS4_DECODE_DESCR: route the single-token indexer scan
                     * through the banked entry (n_banks=1, bank 0 over the
                     * installed views).  Dispatches the SAME direct-one fast
                     * tier as score_one; the banked causal clamp (pos+1)/ratio
                     * equals layer_n_index_comp here (emit-before-read), so
                     * no row goes -INF and the scan is byte-exact vs classic
                     * — gated in the Tier-2 harness. */
                    ok = ds4_gpu_indexer_scores_decode_batch_tensor(g->indexer_scores,
                                                                g->indexer_q,
                                                                g->indexer_weights,
                                                                g->layer_index_comp_cache[il],
                                                                g->layer_n_index_comp[il],
                                                                1,
                                                                pos,
                                                                DS4_N_INDEXER_HEAD,
                                                                DS4_N_INDEXER_HEAD_DIM,
                                                                ds4_layer_compress_ratio(il),
                                                                index_scale,
                                                                g->descr_diag_pos,
                                                                g->descr_diag_seq,
                                                                NULL, /* n_banks==1 diag: installed bank view */
                                                                g->layer_comp_cap[il],
                                                                1) != 0;
                } else if (ok) {
                    ok = ds4_gpu_indexer_score_one_tensor(g->indexer_scores,
                                                                g->indexer_q,
                                                                g->indexer_weights,
                                                                g->layer_index_comp_cache[il],
                                                                g->layer_n_index_comp[il],
                                                                DS4_N_INDEXER_HEAD,
                                                                DS4_N_INDEXER_HEAD_DIM,
                                                                index_scale) != 0;
                }
                if (ok && decode_index_stage_profile) {
                    ok = gpu_graph_indexer_stage_profile_boundary("decode_score",
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                if (ok) ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                           g->indexer_scores,
                                                           g->layer_n_index_comp[il],
                                                           1,
                                                           DS4_N_INDEXER_TOP_K) != 0;
                if (ok && decode_index_stage_profile) {
                    ok = gpu_graph_indexer_stage_profile_boundary("decode_topk",
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                /* Decode used to materialize a dense compressed-row mask and
                 * call the generic gathered FlashAttention wrapper below.
                 * That wrapper scans every compressed row and rejects long
                 * contexts once raw+compressed rows exceed 8192.  Ratio-4 DS4
                 * attention is sparse after indexer top-k, so use the private
                 * indexed attention kernel instead: it scans only SWA raw rows
                 * plus the selected compressed rows, matching prefill and
                 * avoiding the long-context decode failure. */
                if (ok) {
                    comp_selected = g->comp_selected;
                    /*
                     * Contract: the indexer top-k is fixed by the model config
                     * and must remain the full 512 rows.  Do not reduce this for
                     * throughput benchmarks.
                     *
                     * Why: the indexer is not just an implementation detail.  It
                     * decides which compressed memory rows are visible to the
                     * attention kernel.  If we keep only 128/256 rows, the later
                     * indexed-attention math may be perfectly computed, but it is
                     * computed over the wrong candidate set: rows ranked 257-512
                     * are removed before softmax/PV can use them.  Those rows may
                     * carry weak-but-necessary evidence for retrieval, name/number
                     * recall, or long-context disambiguation.  The error is
                     * therefore semantic/algorithmic, not the acceptable kind of
                     * local numerical drift caused by a different reduction order
                     * or Tensor/NAX precision.
                     *
                     * Short prompt tests, first-token agreement, or even a small
                     * official-vector set can miss this because many prompts do
                     * not need the tail of the 512 selected compressed rows.  The
                     * failure appears only when the model needs information that
                     * fell below the reduced cutoff.  Optimizations belong inside
                     * the score/top-k/attention implementation while preserving
                     * DS4_N_INDEXER_TOP_K.
                     */
                    n_selected = DS4_N_INDEXER_TOP_K < g->layer_n_index_comp[il]
                        ? DS4_N_INDEXER_TOP_K
                        : g->layer_n_index_comp[il];
                }
            }
        }

        n_comp = g->layer_n_comp[il];
        comp_cache = g->layer_attn_comp_cache[il];
    }
    DS4_CUDA_PROFILE_DECODE_STAGE("compressor_indexer");

    if (ok) {
        const uint32_t raw_start = gpu_graph_raw_start_for_span(g, pos, n_raw);
        /* DS4_DECODE_DESCR diagnostic (see gpu_graph_decode_descr_enabled):
         * route this step through the banked entries as a 1-bank pool.  The
         * per-row raw derivation (min(pos+1, raw_window) rows ending at pos)
         * matches gpu_graph_raw_span_for_batch/raw_start_for_span exactly,
         * and the per-row (pos+1)/ratio compressed visibility equals the
         * n_comp this step just produced (emit-before-attention).  The
         * descriptor arrays were refreshed once at the top of this layer. */
        if (!ok) {
            /* fall through with ok == false */
        } else if (n_comp != 0 && comp_selected != NULL && n_selected != 0) {
            ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                    g->heads,
                    model->map,
                    model->size,
                    layer->attn_sinks->abs_offset,
                    g->q,
                    raw_cache,
                    g->layer_attn_comp_cache[il],
                    0u, 0u, /* comp f32 (f16/fp8 comp modes removed) */
                    gpu_graph_attn_comp_cache_is_pack(),
                    comp_selected,
                    1,
                    pos,
                    n_raw,
                    raw_cap,
                    raw_start,
                    n_comp,
                    n_selected,
                    g->raw_window,
                    ds4_layer_compress_ratio(il),
                    DS4_N_HEAD,
                    DS4_N_HEAD_DIM,
                    raw_f16,
                    descr_diag ? g->descr_diag_pos : NULL,
                    descr_diag ? g->descr_diag_seq : NULL,
                    NULL, /* n_banks==1 diag: installed bank view is the operand */
                    descr_diag ? g->layer_comp_cap[il] : 0,
                    1) != 0;
            if (ok && decode_index_stage_profile) {
                ok = gpu_graph_indexer_stage_profile_boundary("decode_attention",
                                                                il,
                                                                pos,
                                                                1,
                                                                n_comp,
                                                                &decode_index_stage_t0);
            }
        } else if (descr_diag) {
            /* Non-indexed single-token attention through the banked batch
             * entry (scalar n_raw/raw_start are ignored in banked mode). */
            ok = ds4_gpu_attention_decode_mixed_batch_heads_tensor(g->heads,
                    model->map, model->size,
                    layer->attn_sinks->abs_offset,
                    g->q, raw_cache,
                    n_comp ? comp_cache : NULL,
                    0u, 0u, /* comp f32 (f16/fp8 comp modes removed) */
                    gpu_graph_attn_comp_cache_is_pack(),
                    NULL, 0,
                    1, pos,
                    0, raw_cap, 0, /* n_raw/raw_start unused (banked) */
                    n_comp,
                    g->raw_window,
                    ds4_layer_compress_ratio(il),
                    DS4_N_HEAD, DS4_N_HEAD_DIM,
                    0, raw_f16,
                    g->descr_diag_pos, g->descr_diag_seq,
                    NULL, /* n_banks==1 diag: installed bank view is the operand */
                    g->layer_comp_cap[il], 1) != 0;
        } else {
            ok = ds4_gpu_attention_decode_heads_tensor(g->heads,
                                                         model->map, model->size,
                                                         layer->attn_sinks->abs_offset,
                                                         g->q, raw_cache, n_raw,
                                                         raw_cap,
                                                         raw_start,
                                                         n_comp ? comp_cache : NULL,
                                                         0u, 0u, /* comp f32 (f16/fp8 comp modes removed) */
                                                         gpu_graph_attn_comp_cache_is_pack(),
                                                         n_comp,
                                                         NULL,
                                                         0,
                                                         DS4_N_HEAD, DS4_N_HEAD_DIM,
                                                         raw_f16) != 0;
        }
    }
    DS4_CUDA_PROFILE_DECODE_STAGE("attention");
    if (ok) {
        gpu_graph_debug_dump_tensor("kqv_out", g->heads, q_dim, il, pos);
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->heads,
                                            1, DS4_N_HEAD, DS4_N_HEAD_DIM,
                                            DS4_N_ROT, pos,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            true,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW,
                                            NULL) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("kqv_back", g->heads, q_dim, il, pos);
    }
    const bool fuse_attn_out_hc =
        !gpu_graph_directional_steering_attn_enabled(g) &&
        !gpu_graph_use_reference_attn_out_hc();
    if (ok && fuse_attn_out_hc) {
        ok = ds4_gpu_attention_output_low_tensor(g->attn_low,
                                                   model->map,
                                                   model->size,
                                                   layer->attn_output_a->abs_offset,
                                                   group_dim,
                                                   rank,
                                                   n_groups,
                                                   g->heads) != 0;
        if (ok) {
            ok = ds4_gpu_matmul_fp8_hc_expand_tensor(g->after_attn_hc,
                                                        g->attn_out,
                                                        model->map,
                                                        model->size,
                                                        layer->attn_output_b->abs_offset,
                                                        (uint64_t)n_groups * rank,
                                                        DS4_N_EMBD,
                                                        g->attn_low,
                                                        g->cur_hc,
                                                        g->hc_split,
                                                        DS4_N_EMBD,
                                                        DS4_N_HC) != 0;
        }
    } else if (ok) {
        ok = ds4_gpu_attention_output_batch_tensor(g->attn_out,
                                                     g->attn_low,
                                                     model->map,
                                                     model->size,
                                                     layer->attn_output_a->abs_offset,
                                                     layer->attn_output_b->abs_offset,
                                                     group_dim, rank,
                                                     n_groups, DS4_N_EMBD,
                                                     g->heads, 1) != 0;
    }
    DS4_CUDA_PROFILE_DECODE_STAGE("attn_output");
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_low", g->attn_low, (uint64_t)n_groups * rank, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_out", g->attn_out, DS4_N_EMBD, il, pos);
    }
    if (ok && gpu_graph_directional_steering_attn_enabled(g)) {
        ok = gpu_graph_apply_directional_steering_attn(g, g->attn_out, il, 1);
    }
    if (ok && !fuse_attn_out_hc) {
        ok = ds4_gpu_hc_expand_tensor(g->after_attn_hc, g->attn_out, g->cur_hc,
                                        g->hc_post, g->hc_comb, DS4_N_EMBD, DS4_N_HC) != 0;
    }
    DS4_CUDA_PROFILE_DECODE_STAGE("attn_hc_post");
    if (ok) {
        gpu_graph_debug_dump_hc_tensor("hc_attn_post", g->after_attn_hc, hc_dim, il, pos);
    }
    if (ok) ok = gpu_graph_norm_mix_plain(g, model, layer->hc_ffn_fn,
                                          hc_dim, mix_hc, g->after_attn_hc, g->hc_mix);
    if (ok && fuse_hc_norm) {
        ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(g->ffn_cur,
                                                         g->ffn_norm,
                                                         g->hc_split,
                                                         g->hc_mix,
                                                         g->after_attn_hc,
                                                         model->map,
                                                         model->size,
                                                         layer->hc_ffn_scale->abs_offset,
                                                         layer->hc_ffn_base->abs_offset,
                                                         layer->ffn_norm->abs_offset,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC,
                                                         DS4_N_HC_SINKHORN_ITER,
                                                         DS4_HC_EPS,
                                                         DS4_RMS_EPS) != 0;
        if (ok) {
            ok = gpu_graph_check_hc_norm_fusion("ffn",
                                                  g->ffn_cur,
                                                  g->ffn_norm,
                                                  g->hc_mix,
                                                  g->after_attn_hc,
                                                  model,
                                                  layer->hc_ffn_scale->abs_offset,
                                                  layer->hc_ffn_base->abs_offset,
                                                  layer->ffn_norm->abs_offset,
                                                  il,
                                                  pos);
        }
    } else if (ok) {
        ok = gpu_graph_decode_hc_pre(g->ffn_cur,
                                       g->hc_split,
                                       g->hc_mix,
                                       g->after_attn_hc,
                                       model,
                                       layer->hc_ffn_scale->abs_offset,
                                       layer->hc_ffn_base->abs_offset);
    }
    DS4_CUDA_PROFILE_DECODE_STAGE("ffn_hc_pre");
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_ffn_pre_mixes", g->hc_mix, mix_hc, il, pos);
        gpu_graph_debug_dump_tensor("hc_ffn_pre_weights", g->hc_pre, DS4_N_HC, il, pos);
        gpu_graph_debug_dump_tensor("hc_ffn_pre_post_weights", g->hc_post, DS4_N_HC, il, pos);
        gpu_graph_debug_dump_tensor("hc_ffn_pre_comb", g->hc_comb, (uint64_t)DS4_N_HC * DS4_N_HC, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_ffn_pre", g->ffn_cur, DS4_N_EMBD, il, pos);
    }
    if (ok && !fuse_hc_norm) ok = ds4_gpu_rms_norm_weight_tensor(g->ffn_norm, g->ffn_cur,
                                                                   model->map, model->size,
                                                                   layer->ffn_norm->abs_offset,
                                                                   DS4_N_EMBD, DS4_RMS_EPS) != 0;
    DS4_CUDA_PROFILE_DECODE_STAGE("ffn_norm");
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_norm", g->ffn_norm, DS4_N_EMBD, il, pos);
    }
    uint64_t gate_expert_bytes = 0, gate_row_bytes = 0;
    uint64_t down_expert_bytes = 0, down_row_bytes = 0;
    if (ok) {
        ok = routed_expert_gate_down_layout(layer->ffn_gate_exps, layer->ffn_down_exps,
                                            &gate_expert_bytes, &gate_row_bytes,
                                            &down_expert_bytes, &down_row_bytes);
    }
    if (ok) ok = gpu_graph_matmul_plain_tensor(g->router_logits, model, layer->ffn_gate_inp,
                                                 DS4_N_EMBD, DS4_N_EXPERT, g->ffn_norm, 1);
    if (ok) ok = ds4_gpu_router_select_tensor(g->router_selected, g->router_weights, g->router_probs,
                                                model->map, model->size,
                                                layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                                layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
                                                layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                                                (uint32_t)token,
                                                DS4_N_EXPERT,
                                                DS4_N_EXPERT_USED,
                                                DS4_EXPERT_WEIGHT_SCALE,
                                                0,
                                                0,
                                                layer->ffn_exp_probs_b != NULL,
                                                layer->ffn_gate_tid2eid != NULL,
                                                g->router_logits) != 0;
    DS4_CUDA_PROFILE_DECODE_STAGE("router");
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_logits", g->router_logits, DS4_N_EXPERT, il, pos);
        gpu_graph_debug_dump_tensor("ffn_moe_probs", g->router_probs, DS4_N_EXPERT, il, pos);
        gpu_graph_debug_dump_i32_tensor("ffn_moe_topk", g->router_selected, DS4_N_EXPERT_USED, il, pos);
        gpu_graph_debug_dump_tensor("ffn_moe_weights_scaled", g->router_weights, DS4_N_EXPERT_USED, il, pos);
    }
    const bool keep_ffn_out = gpu_graph_needs_ffn_out(g, il, pos);
    const bool fuse_shared_gate_up =
        !g->quality &&
        !gpu_graph_disable_shared_gate_up_swiglu();
    const bool fuse_shared_down_hc =
        !keep_ffn_out && !gpu_graph_use_reference_shared_down_hc();
    if (ok) ok = ds4_gpu_routed_moe_one_tensor(g->routed_out,
                                                 g->routed_gate,
                                                 g->routed_up,
                                                 g->routed_mid,
                                                 g->routed_down,
                                                 tensor_map_base(model, layer->ffn_gate_exps),
                                                 tensor_map_size(model, layer->ffn_gate_exps),
                                                 layer->ffn_gate_exps->abs_offset,
                                                 layer->ffn_up_exps->abs_offset,
                                                 layer->ffn_down_exps->abs_offset,
                                                 layer->ffn_gate_exps->type,
                                                 layer->ffn_down_exps->type,
                                                 gate_expert_bytes, gate_row_bytes,
                                                 down_expert_bytes, down_row_bytes,
                                                 (uint32_t)expert_in_dim,
                                                 (uint32_t)down_in_dim,
                                                 (uint32_t)routed_out_dim,
                                                 g->router_selected, g->router_weights,
                                                 ds4_layer_n_expert(il),
                                                 DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, g->ffn_norm,
                                                 il) != 0;
    DS4_CUDA_PROFILE_DECODE_STAGE("routed_moe");
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_gate_clamped", g->routed_gate,
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
        gpu_graph_debug_dump_tensor("ffn_moe_up_clamped", g->routed_up,
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", g->routed_mid,
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_down", g->routed_down,
                                      (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_out", g->routed_out, DS4_N_EMBD, il, pos);
    }
    if (ok && fuse_shared_gate_up) {
        ok = ds4_gpu_shared_gate_up_swiglu_mxfp8_tensor(g->shared_gate,
                                                         g->shared_up,
                                                         g->shared_mid,
                                                         model->map,
                                                         model->size,
                                                         layer->ffn_gate_shexp->abs_offset,
                                                         layer->ffn_up_shexp->abs_offset,
                                                         DS4_N_EMBD,
                                                         shared_dim,
                                                         g->ffn_norm,
                                                         DS4_SWIGLU_CLAMP_EXP) != 0;
    } else {
        if (ok) ok = ds4_gpu_matmul_mxfp8_tensor(g->shared_gate, model->map, model->size,
                                                  layer->ffn_gate_shexp->abs_offset,
                                                  DS4_N_EMBD, shared_dim,
                                                  g->ffn_norm, 1) != 0;
        if (ok) ok = ds4_gpu_matmul_mxfp8_tensor(g->shared_up, model->map, model->size,
                                                  layer->ffn_up_shexp->abs_offset,
                                                  DS4_N_EMBD, shared_dim,
                                                  g->ffn_norm, 1) != 0;
        if (ok) ok = ds4_gpu_swiglu_tensor(g->shared_mid, g->shared_gate, g->shared_up,
                                           shared_dim, DS4_SWIGLU_CLAMP_EXP, 1.0f) != 0;
    }
    DS4_CUDA_PROFILE_DECODE_STAGE("shared_gate_up");
    if (ok && fuse_shared_down_hc) {
        ok = ds4_gpu_shared_down_hc_expand_mxfp8_tensor(g->after_ffn_hc,
                                                         g->shared_out,
                                                         model->map,
                                                         model->size,
                                                         layer->ffn_down_shexp->abs_offset,
                                                         shared_dim,
                                                         DS4_N_EMBD,
                                                         g->shared_mid,
                                                         g->routed_out,
                                                         g->after_attn_hc,
                                                         g->hc_split,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC) != 0;
    } else if (ok) {
        ok = ds4_gpu_matmul_mxfp8_tensor(g->shared_out, model->map, model->size,
                                          layer->ffn_down_shexp->abs_offset,
                                          shared_dim, DS4_N_EMBD,
                                          g->shared_mid, 1) != 0;
    }
    DS4_CUDA_PROFILE_DECODE_STAGE("shared_down");
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_shexp", g->shared_out, DS4_N_EMBD, il, pos);
    }
    if (ok && keep_ffn_out) {
        ok = gpu_graph_ensure_ffn_out(g) &&
             ds4_gpu_add_tensor(g->ffn_out, g->shared_out, g->routed_out, DS4_N_EMBD) != 0;
    }
    if (ok && keep_ffn_out) {
        gpu_graph_debug_dump_tensor("ffn_out", g->ffn_out, DS4_N_EMBD, il, pos);
    }
    if (ok && gpu_graph_directional_steering_ffn_enabled(g)) {
        ok = gpu_graph_apply_directional_steering_ffn(g, g->ffn_out, il, 1);
    }
    if (ok && gpu_graph_directional_steering_ffn_enabled(g)) {
        ok = ds4_gpu_hc_expand_tensor(g->after_ffn_hc,
                                        g->ffn_out,
                                        g->after_attn_hc,
                                        g->hc_post,
                                        g->hc_comb,
                                        DS4_N_EMBD,
                                        DS4_N_HC) != 0;
    } else if (ok && !fuse_shared_down_hc) {
        ok = ds4_gpu_hc_expand_add_split_tensor(g->after_ffn_hc,
                                                  g->routed_out,
                                                  g->shared_out,
                                                  g->after_attn_hc,
                                                  g->hc_split,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    }
    DS4_CUDA_PROFILE_DECODE_STAGE("ffn_hc_post");
#undef DS4_CUDA_PROFILE_DECODE_STAGE
    if (ok) {
        gpu_graph_debug_dump_hc_tensor("hc_ffn_post", g->after_ffn_hc, hc_dim, il, pos);
    }
    if (ok) gpu_graph_capture_dspark_target_hc(g, il);
    return ok;
}

void gpu_graph_capture_dspark_target_hc(ds4_gpu_graph *g, uint32_t il) {
    int slot = -1;
    for (int i = 0; i < 3; i++) {
        if (il == g->dspark_target_layer_ids[i]) { slot = i; break; }
    }
    if (slot < 0 || !g->dspark_target_h[slot]) return;

    ds4_gpu_dspark_hc_mean_reduce(g->dspark_target_h[slot],
                                   g->after_ffn_hc,
                                   DS4_N_EMBD, DS4_N_HC);
}



bool gpu_graph_dspark_project_main_x(
        ds4_gpu_graph          *g,
        const ds4_model         *dspark_model,
        const ds4_dspark_weights *w) {
    const uint64_t E = DS4_N_EMBD;
    const uint64_t concat_dim = 3ull * E;

    for (int i = 0; i < 3; i++) {
        if (!g->dspark_target_h[i]) return false;
    }

    /* Persistent scratch: this runs up to 5x per fused spec step and each
     * cudaMalloc/cudaFree pair serializes the device. */
    ds4_gpu_tensor *target_concat = g->dspark_concat;
    ds4_gpu_tensor *proj_out = g->dspark_proj_out;
    if (!target_concat || !proj_out) return false;

    bool ok = true;
    for (int i = 0; i < 3; i++) {
        ok = ds4_gpu_tensor_copy(target_concat, (uint64_t)i * E * sizeof(float),
                                 g->dspark_target_h[i], 0, E * sizeof(float)) != 0;
        if (!ok) break;
    }

    if (ok) {
        ok = ds4_gpu_matmul_mxfp8_tensor(proj_out,
                                          dspark_model->map,
                                          dspark_model->size,
                                          w->main_proj->abs_offset,
                                          concat_dim, E,
                                          target_concat, 1) != 0;
    }

    if (ok) {
        ok = ds4_gpu_rms_norm_weight_tensor(g->dspark_main_x,
                                            proj_out,
                                            dspark_model->map,
                                            dspark_model->size,
                                            w->main_norm->abs_offset,
                                            (uint32_t)E,
                                            DS4_RMS_EPS) != 0;
    }

    return ok;
}

void gpu_graph_dspark_seed_draft_kv(
        ds4_gpu_graph          *g,
        const ds4_model         *dspark_model,
        const ds4_dspark_weights *w,
        uint32_t                 n_rows) {
    const uint64_t kv_bytes = (uint64_t)DS4_N_HEAD_DIM * sizeof(float);
    /* Persistent scratch (dspark_seed_*): the fused loop seeds up to 5 rows per
     * step across 3 layers; per-call cudaMalloc/cudaFree here was ~9 device-
     * serializing pairs per call. */
    ds4_gpu_tensor *kv_out = g->dspark_seed_kv;
    ds4_gpu_tensor *kv_norm = g->dspark_seed_norm;
    ds4_gpu_tensor *kv_rot = g->dspark_seed_rot;
    if (!kv_out || !kv_norm || !kv_rot) return;
    for (int li = 0; li < 3; li++) {
        if (!ds4_gpu_matmul_mxfp8_tensor(kv_out,
                                          dspark_model->map,
                                          dspark_model->size,
                                          w->layer[li].attn_kv->abs_offset,
                                          DS4_N_EMBD, DS4_N_HEAD_DIM,
                                          g->dspark_main_x, 1)) {
            continue;
        }
        if (!ds4_gpu_rms_norm_weight_tensor(kv_norm, kv_out,
                                             dspark_model->map,
                                             dspark_model->size,
                                             w->layer[li].attn_kv_a_norm->abs_offset,
                                             DS4_N_HEAD_DIM, DS4_RMS_EPS)) {
            continue;
        }
        /* Seed one KV row per committed position.  Each row is RoPE'd at its OWN
         * sequence position (dspark_n_raw[li]) and fp8-rounded, matching the
         * draft-forward KV and the DSparkAttention reference (main_kv is rotated
         * at start_pos).  kv_norm holds the un-rotated vector; we rotate a fresh
         * copy per position so multi-row seeds (accepted drafts) land at distinct
         * positions rather than all sharing the first row's rotation. */
        for (uint32_t i = 0; i < n_rows; i++) {
            const uint32_t pos = g->dspark_n_raw[li];
            const uint32_t row = pos % DS4_DSPARK_DRAFT_WINDOW;
            ds4_gpu_tensor_copy(kv_rot, 0, kv_norm, 0, kv_bytes);
            ds4_gpu_rope_tail_tensor(kv_rot, 1, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT,
                                     pos, 0, false,
                                     (float)DS4_ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f,
                                     DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, NULL);
            ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv_rot, 1, DS4_N_HEAD_DIM, DS4_N_ROT);
            ds4_gpu_tensor_copy(g->dspark_raw_cache[li],
                                (uint64_t)row * kv_bytes,
                                kv_rot, 0, kv_bytes);
            g->dspark_n_raw[li]++;
        }
    }
}

bool gpu_graph_dspark_draft_forward(
        ds4_gpu_graph          *g,
        const ds4_model         *base_model,
        const ds4_weights       *base_weights,
        const ds4_model         *dspark_model,
        const ds4_dspark_weights *w,
        ds4_gpu_tensor         *base_logits_out,
        const int32_t            draft_ids[],
        uint32_t                n_draft) {
    if (!g || !base_model || !base_weights || !dspark_model || !w ||
        !base_logits_out || n_draft == 0 || n_draft > 16 ||
        n_draft > g->prefill_cap)
        return false;

    if (!base_weights->token_embd || !base_weights->output)
        return false;

    /* Embed N draft tokens via main model's F16 token_embd → HC-expand */
    ds4_gpu_tensor *tokens_t = ds4_gpu_tensor_alloc((uint64_t)n_draft * sizeof(int32_t));
    if (!tokens_t) return false;
    if (!ds4_gpu_tensor_write(tokens_t, 0, draft_ids, (uint64_t)n_draft * sizeof(int32_t))) {
        ds4_gpu_tensor_free(tokens_t);
        return false;
    }
    bool ok = ds4_gpu_embed_tokens_hc_tensor(g->batch_cur_hc,
                                              tokens_t,
                                              base_model->map,
                                              base_model->size,
                                              base_weights->token_embd->abs_offset,
                                              DS4_N_VOCAB,
                                              n_draft,
                                              DS4_N_EMBD,
                                              DS4_N_HC) != 0;
    ds4_gpu_tensor_free(tokens_t);
    if (!ok) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint32_t n_groups = DS4_N_OUT_GROUP;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = DS4_N_LORA_O;

    const int prev_comp = g->comp_ratio_override;
    g->comp_ratio_override = 0;

    for (uint32_t li = 0; li < 3 && ok; li++) {
        const ds4_layer_weights *layer = &w->layer[li];
        const uint32_t raw_cap = DS4_DSPARK_DRAFT_WINDOW;
        const uint32_t q_rank = (uint32_t)layer->attn_q_a->dim[1];
        /* Draft queries/KV sit at the frontier (the current main_kv was seeded at
         * dspark_n_raw[li]-1 just before this forward), so RoPE them at the real
         * position -- NOT 0.  The reference DSparkAttention rotates draft Q/KV at
         * start_pos+1 while the seeded main_kv is at start_pos. */
        const uint32_t pos0 = g->dspark_n_raw[li];

        /* --- HC pre-processing --- */
        /* Create views from batch working set */
        ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_draft * mix_hc * sizeof(float));
        ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_draft * mix_hc * sizeof(float));
        ds4_gpu_tensor *attn_cur_view = ds4_gpu_tensor_view(
            g->batch_ffn_cur, 0, (uint64_t)n_draft * DS4_N_EMBD * sizeof(float));
        ok = hc_mix_view && hc_split_view && attn_cur_view;
        /* RMS norm: flat HC from batch_cur_hc */
        if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(
            g->batch_flat_hc, g->batch_cur_hc,
            (uint32_t)hc_dim, n_draft, DS4_RMS_EPS) != 0;
        /* HC → mix projection */
        if (ok) ok = gpu_graph_matmul_plain_tensor(
            hc_mix_view, dspark_model,
            layer->hc_attn_fn,
            hc_dim, mix_hc, g->batch_flat_hc, n_draft);
        /* HC split + weighted sum → attn_cur (E-dim) */
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(
            attn_cur_view, hc_split_view, hc_mix_view,
            g->batch_cur_hc,
            dspark_model->map, dspark_model->size,
            layer->hc_attn_scale->abs_offset,
            layer->hc_attn_base->abs_offset,
            DS4_N_EMBD, DS4_N_HC,
            DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS) != 0;
        /* Input RMS norm → batch_attn_norm */
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(
            g->batch_attn_norm, attn_cur_view,
            dspark_model->map, dspark_model->size,
            layer->attn_norm->abs_offset,
            DS4_N_EMBD, n_draft, DS4_RMS_EPS) != 0;

        if (ok) gpu_graph_debug_dump_tensor("dsp_attn_norm", g->batch_attn_norm,
                                             (uint64_t)n_draft * DS4_N_EMBD, li, pos0);
        /* --- Q projection --- */
        if (ok) ok = ds4_gpu_matmul_mxfp8_tensor(
            g->batch_qr, dspark_model->map, dspark_model->size,
            layer->attn_q_a->abs_offset,
            DS4_N_EMBD, q_rank, g->batch_attn_norm, n_draft) != 0;
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(
            g->batch_qr_norm, g->batch_qr,
            dspark_model->map, dspark_model->size,
            layer->attn_q_a_norm->abs_offset,
            q_rank, n_draft, DS4_RMS_EPS) != 0;
        if (ok) ok = ds4_gpu_matmul_mxfp8_tensor(
            g->batch_q, dspark_model->map, dspark_model->size,
            layer->attn_q_b->abs_offset,
            q_rank, DS4_N_HEAD * DS4_N_HEAD_DIM,
            g->batch_qr_norm, n_draft) != 0;
        /* Q head-norm + RoPE */
        if (ok) ok = ds4_gpu_head_rms_norm_rope_tail_tensor(
            g->batch_q, n_draft,
            DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT,
            pos0, 0, false,
            (float)DS4_ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f,
            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, DS4_RMS_EPS,
            NULL) != 0;

        /* --- KV projection --- */
        if (ok) ok = ds4_gpu_matmul_mxfp8_tensor(
            g->batch_kv_raw, dspark_model->map, dspark_model->size,
            layer->attn_kv->abs_offset,
            DS4_N_EMBD, DS4_N_HEAD_DIM,
            g->batch_attn_norm, n_draft) != 0;
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(
            g->batch_kv, g->batch_kv_raw,
            dspark_model->map, dspark_model->size,
            layer->attn_kv_a_norm->abs_offset,
            DS4_N_HEAD_DIM, n_draft, DS4_RMS_EPS) != 0;
        if (ok) ok = ds4_gpu_rope_tail_tensor(
            g->batch_kv, n_draft,
            DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT,
            pos0, 0, false,
            (float)DS4_ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f,
            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, NULL) != 0;
        if (ok) ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(
            g->batch_kv, n_draft, DS4_N_HEAD_DIM, DS4_N_ROT) != 0;

        /* --- Store draft KV transiently in ring buffer for attention --- */
        const uint32_t saved_n_raw = g->dspark_n_raw[li];
        const uint32_t kv_store_pos = saved_n_raw % raw_cap;
        if (ok) ok = ds4_gpu_store_raw_kv_batch_tensor(
            g->dspark_raw_cache[li], g->batch_kv,
            raw_cap, kv_store_pos, n_draft, DS4_N_HEAD_DIM,
            0 /* drafter ring is always f32 */,
            NULL, NULL, 1) != 0;
        const uint32_t vis_raw = saved_n_raw + n_draft;
        const uint32_t cap_raw = vis_raw < raw_cap ? vis_raw : raw_cap;
        const uint32_t raw_start = vis_raw > raw_cap
            ? (vis_raw - raw_cap) % raw_cap : 0;

        /* --- Non-causal raw batch attention ---
         * Queries are at positions [saved_n_raw, saved_n_raw+n_draft).
         * Visible raw entries span [0, vis_raw) — all cached + current draft rows. */
        if (ok) ok = ds4_gpu_attention_decode_raw_batch_heads_tensor(
            g->batch_heads,
            dspark_model->map, dspark_model->size,
            layer->attn_sinks->abs_offset,
            g->batch_q, g->dspark_raw_cache[li],
            n_draft, saved_n_raw,
            cap_raw, raw_cap, raw_start,
            0,
            DS4_N_HEAD, DS4_N_HEAD_DIM,
            1,
            0 /* drafter ring is always f32 */,
            NULL, NULL, 0, 1) != 0;

        if (ok) gpu_graph_debug_dump_tensor("dsp_heads", g->batch_heads,
                                             (uint64_t)n_draft * DS4_N_HEAD * DS4_N_HEAD_DIM, li, pos0);
        /* Inverse-rotate the attention output's rope dims before the o
         * projection (reference: apply_rotary_emb(o, freqs_cis, inverse=True);
         * the verify/prefill path does the same via its "kqv_back" rope).
         * This call was MISSING here: wo_a/wo_b consumed position-rotated
         * rope dims -- 64/512 dims per head scrambled in every drafter block,
         * the ~30-point acceptance bug (#4). Verified vs the torch reference:
         * the engine attention output matches exactly WITHOUT inverse rope
         * (cos 0.99999) and diverges with it (cos 0.57), so everything
         * downstream of this line computed on corrupted features. */
        if (ok) ok = ds4_gpu_rope_tail_tensor(
            g->batch_heads, n_draft,
            DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT,
            pos0, 0, true,
            (float)DS4_ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f,
            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, NULL) != 0;
        /* --- Attention output projection (LoRA grouped) --- */
        if (ok) ok = ds4_gpu_attention_output_batch_tensor(
            g->batch_attn_out, g->batch_attn_low,
            dspark_model->map, dspark_model->size,
            layer->attn_output_a->abs_offset,
            layer->attn_output_b->abs_offset,
            group_dim, rank, n_groups, DS4_N_EMBD,
            g->batch_heads, n_draft) != 0;
        if (ok) gpu_graph_debug_dump_tensor("dsp_attn_out", g->batch_attn_out,
                                             (uint64_t)n_draft * DS4_N_EMBD, li, pos0);

        /* --- HC expand + split → batch_after_attn_hc --- */
        /* View sized to the real draft count: the CUDA side infers n_tokens from
         * out_hc->bytes, and the raw batch tensor is allocated at prefill capacity
         * (4096) -- passing it unviewed made every drafter block expand the FULL
         * capacity, ~2.8 ms per call x3 blocks = ~8 ms of pure waste per spec step. */
        if (ok) {
            ds4_gpu_tensor *after_attn_view = ds4_gpu_tensor_view(
                g->batch_after_attn_hc, 0,
                (uint64_t)n_draft * DS4_N_HC * DS4_N_EMBD * DS4_HC_ELT_SIZE);   /* carrier */
            ok = after_attn_view &&
                 ds4_gpu_hc_expand_split_tensor(
                     after_attn_view, g->batch_attn_out,
                     g->batch_cur_hc, g->batch_hc_split,
                     DS4_N_EMBD, DS4_N_HC) != 0;
            ds4_gpu_tensor_free(after_attn_view);
        }

        /* --- FFN batch (reuses existing function) --- */
        if (ok) ok = gpu_graph_encode_layer_ffn_batch(
            g, dspark_model, layer, li, pos0, n_draft);

        if (ok) gpu_graph_debug_dump_hc_tensor("dsp_after_attn_hc", g->batch_after_attn_hc,
                                             (uint64_t)n_draft * DS4_N_HC * DS4_N_EMBD, li, pos0);
        /* --- HC swap for next layer --- */
        if (ok) {
            ds4_gpu_tensor *tmp = g->batch_cur_hc;
            g->batch_cur_hc = g->batch_next_hc;
            g->batch_next_hc = tmp;
        }
        if (ok) gpu_graph_debug_dump_hc_tensor("dsp_block_out", g->batch_cur_hc,
                                             (uint64_t)n_draft * DS4_N_HC * DS4_N_EMBD, li, pos0);
        /* Draft KV is transient; dspark_n_raw remains at the persistent count.
         * Committed positions are seeded via gpu_graph_dspark_seed_draft_kv(). */

        /* Views over the batch working set are per-iteration host structs;
         * free them each layer or they leak on every speculative block. */
        ds4_gpu_tensor_free(hc_mix_view);
        ds4_gpu_tensor_free(hc_split_view);
        ds4_gpu_tensor_free(attn_cur_view);
    }

    g->comp_ratio_override = prev_comp;

    /* Batch output head → N-token logits in g->spec_logits.  Use the DSpark
     * drafter's OWN hc_head + norm (dspark.2.*) with the shared vocab head, NOT
     * the main model's output head (which was corrupting the draft logits). */
    if (ok) {
        ok = gpu_graph_encode_dspark_output_head_batch(
            g, dspark_model, w, base_model, base_weights, n_draft, DS4_N_VOCAB);
    }

    /* The output head already wrote into g->spec_logits.  Callers that pass a
     * distinct buffer get a copy; the session passes g->spec_logits itself, so
     * skip the self-copy (a same-buffer cudaMemcpy is undefined). */
    if (ok && base_logits_out && base_logits_out != g->spec_logits) {
        const uint64_t logits_bytes = (uint64_t)n_draft * DS4_N_VOCAB * sizeof(float);
        ok = ds4_gpu_tensor_copy(base_logits_out, 0,
                                  g->spec_logits, 0,
                                  logits_bytes) != 0;
    }

    return ok;
}

/* Encode the final HC collapse, output norm, and vocab projection on GPU. */
bool gpu_graph_encode_output_head(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint64_t               vocab_dim) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    bool ok = gpu_graph_norm_mix_plain(g, (const ds4_model *)model, weights->output_hc_fn,
                                       hc_dim, DS4_N_HC, g->cur_hc, g->output_pre);
    if (ok) {
        gpu_graph_debug_dump_tensor("result_hc_pre", g->output_pre, DS4_N_HC, DS4_N_LAYER, 0);
    }
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(g->output_weights,
                                                    g->output_pre,
                                                    model->map,
                                                    model->size,
                                                    weights->output_hc_scale->abs_offset,
                                                    weights->output_hc_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("result_hc_weights", g->output_weights, DS4_N_HC, DS4_N_LAYER, 0);
    }
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(g->output_embd,
                                                  g->cur_hc,
                                                  g->output_weights,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("result_hc", g->output_embd, DS4_N_EMBD, DS4_N_LAYER, 0);
    }
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->output_norm,
                                                  g->output_embd,
                                                  model->map,
                                                  model->size,
                                                  weights->output_norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_RMS_EPS) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("result_norm", g->output_norm, DS4_N_EMBD, DS4_N_LAYER, 0);
    }
    if (ok) {
        if (weights->output->type == DS4_TENSOR_BF16)
            ok = ds4_gpu_matmul_bf16_tensor(g->logits, model->map, model->size,
                                            weights->output->abs_offset, DS4_N_EMBD,
                                            vocab_dim, g->output_norm, 1) != 0;
        else
            ok = ds4_gpu_matmul_mxfp8_tensor(g->logits, model->map, model->size,
                                            weights->output->abs_offset, DS4_N_EMBD,
                                            vocab_dim, g->output_norm, 1) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("result_output", g->logits, vocab_dim, DS4_N_LAYER, 0);
    }
    return ok;
}



/* Batched output head for speculative verification.
 *
 * A target verifier only needs top-1 ids for intermediate draft rows and full
 * logits for the last accepted row.  Running the normal one-row output head in
 * a loop serializes the HC collapse, output norm, and MXFP8 vocab projection.  For
 * tiny speculative suffixes we instead process all rows together and let the GPU reduce
 * each row to a top id; the CPU reads back just those ids plus the last row's
 * logits needed to continue the exact target stream. */
bool gpu_graph_encode_output_head_batch(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint32_t               n_tokens,
        uint64_t               vocab_dim) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap || !g->spec_logits) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    ds4_gpu_tensor *output_pre = NULL;
    ds4_gpu_tensor *output_weights = NULL;
    ds4_gpu_tensor *output_embd = NULL;
    ds4_gpu_tensor *output_norm = NULL;
    ds4_gpu_tensor *logits = NULL;

    bool ok = true;
    output_pre = ds4_gpu_tensor_view(g->batch_hc_mix,
                                       0,
                                       (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    output_weights = ds4_gpu_tensor_view(g->batch_hc_split,
                                           0,
                                           (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    output_embd = ds4_gpu_tensor_view(g->batch_ffn_cur,
                                        0,
                                        (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    output_norm = ds4_gpu_tensor_view(g->batch_ffn_norm,
                                        0,
                                        (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    logits = ds4_gpu_tensor_view(g->spec_logits,
                                   0,
                                   (uint64_t)n_tokens * vocab_dim * sizeof(float));
    ok = output_pre && output_weights && output_embd && output_norm && logits;

    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      g->batch_cur_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = gpu_graph_matmul_plain_tensor(output_pre,
                                                 (const ds4_model *)model,
                                                 weights->output_hc_fn,
                                             hc_dim,
                                             DS4_N_HC,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(output_weights,
                                                    output_pre,
                                                    model->map,
                                                    model->size,
                                                    weights->output_hc_scale->abs_offset,
                                                    weights->output_hc_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(output_embd,
                                                  g->batch_cur_hc,
                                                  output_weights,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(output_norm,
                                                       output_embd,
                                                       model->map,
                                                       model->size,
                                                       weights->output_norm->abs_offset,
                                                       DS4_N_EMBD,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0;
    if (ok) {
        if (weights->output->type == DS4_TENSOR_BF16)
            ok = ds4_gpu_matmul_bf16_tensor(logits, model->map, model->size,
                                            weights->output->abs_offset, DS4_N_EMBD,
                                            vocab_dim, output_norm, n_tokens) != 0;
        else
            ok = ds4_gpu_matmul_mxfp8_tensor(logits, model->map, model->size,
                                            weights->output->abs_offset, DS4_N_EMBD,
                                            vocab_dim, output_norm, n_tokens) != 0;
    }

    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(output_norm);
    ds4_gpu_tensor_free(output_embd);
    ds4_gpu_tensor_free(output_weights);
    ds4_gpu_tensor_free(output_pre);
    return ok;
}

/* DSpark drafter output head.  Collapses the drafter's final HC with the DSpark
 * block's OWN head (dspark.2.hc_head_fn/scale/base) and norm (dspark.2.norm),
 * then projects to vocab with the SHARED main output head (self.head in the
 * reference).  The plain gpu_graph_encode_output_head_batch used the MAIN model's
 * output_hc and output_norm weights for the drafter -- wrong weights that
 * corrupted the draft base logits (base0_hit was ~29%). */
bool gpu_graph_encode_dspark_output_head_batch(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        const ds4_model          *base_model,
        const ds4_weights        *bw,
        uint32_t                  n_tokens,
        uint64_t                  vocab_dim) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap || !g->spec_logits) return false;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    ds4_gpu_tensor *output_pre = ds4_gpu_tensor_view(g->batch_hc_mix, 0, (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    ds4_gpu_tensor *output_weights = ds4_gpu_tensor_view(g->batch_hc_split, 0, (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    ds4_gpu_tensor *output_embd = ds4_gpu_tensor_view(g->batch_ffn_cur, 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *output_norm = ds4_gpu_tensor_view(g->batch_ffn_norm, 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *logits = ds4_gpu_tensor_view(g->spec_logits, 0, (uint64_t)n_tokens * vocab_dim * sizeof(float));
    bool ok = output_pre && output_weights && output_embd && output_norm && logits;
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc, g->batch_cur_hc,
                                                     (uint32_t)hc_dim, n_tokens, DS4_RMS_EPS) != 0;
    if (ok) ok = gpu_graph_matmul_plain_tensor(output_pre, dspark_model, dw->hc_head_fn,
                                               hc_dim, DS4_N_HC, g->batch_flat_hc, n_tokens) != 0;
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(output_weights, output_pre,
                                                  dspark_model->map, dspark_model->size,
                                                  dw->hc_head_scale->abs_offset,
                                                  dw->hc_head_base->abs_offset,
                                                  DS4_N_HC, DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(output_embd, g->batch_cur_hc, output_weights,
                                                DS4_N_EMBD, DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(output_norm, output_embd,
                                                     dspark_model->map, dspark_model->size,
                                                     dw->final_norm->abs_offset,
                                                     DS4_N_EMBD, n_tokens, DS4_RMS_EPS) != 0;
    if (ok) {
        if (bw->output->type == DS4_TENSOR_BF16)
            ok = ds4_gpu_matmul_bf16_tensor(logits, base_model->map, base_model->size,
                                            bw->output->abs_offset, DS4_N_EMBD, vocab_dim,
                                            output_norm, n_tokens) != 0;
        else
            ok = ds4_gpu_matmul_mxfp8_tensor(logits, base_model->map, base_model->size,
                                             bw->output->abs_offset, DS4_N_EMBD, vocab_dim,
                                             output_norm, n_tokens) != 0;
    }
    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(output_norm);
    ds4_gpu_tensor_free(output_embd);
    ds4_gpu_tensor_free(output_weights);
    ds4_gpu_tensor_free(output_pre);
    return ok;
}



bool gpu_graph_matmul_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (w->type == DS4_TENSOR_F16) {
        return ds4_gpu_matmul_f16_tensor(out, model->map, model->size,
                                           w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    if (w->type == DS4_TENSOR_F32) {
        return ds4_gpu_matmul_f32_tensor(out, model->map, model->size,
                                           w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    if (w->type == DS4_TENSOR_FP8_E4M3) {
        return ds4_gpu_matmul_mxfp8_tensor(out, model->map, model->size,
                                            w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    fprintf(stderr, "ds4: plain matmul does not support %s\n", tensor_type_name(w->type));
    return false;
}



bool gpu_graph_matmul_mxfp8_named_tensor(
        const char             *module,
        uint32_t                il,
        uint32_t                pos0,
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    (void)module;
    (void)il;
    (void)pos0;
    const bool ok = ds4_gpu_matmul_mxfp8_tensor(out,
                                                 model->map,
                                                 model->size,
                                                 w->abs_offset,
                                                 in_dim,
                                                 out_dim,
                                                 x,
                                                 n_tok) != 0;
    return ok;
}



/* =========================================================================
 * GPU Diagnostic Comparisons.
 * =========================================================================
 *
 * These routines deliberately allocate CPU-side reference buffers and read
 * GPU tensors back.  They are not part of generation; command-line tests use
 * them to localize drift against the C reference pipeline.
 */




int gpu_graph_decode_test(
        const ds4_model   *model,
        const ds4_weights *weights,
        const token_vec   *prompt,
        bool               quality) {
    if (prompt->len <= 0) {
        fprintf(stderr, "ds4: GPU graph test needs a non-empty prompt\n");
        return 1;
    }

    const int token = prompt->v[0];
    const ds4_layer_weights *layer = &weights->layer[0];
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t vocab_dim = weights->output->dim[1];

    float *plain = (float *)xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_hc = (float *)xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_attn_cur = (float *)xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_post = (float *)xmalloc((size_t)DS4_N_HC * sizeof(float));
    float *cpu_comb = (float *)xmalloc((size_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    float *cpu_attn_norm = (float *)xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_qr_norm = (float *)xmalloc((size_t)q_rank * sizeof(float));
    float *cpu_q = (float *)xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_kv = (float *)xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *cpu_heads = (float *)xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_attn_out = (float *)xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_attn_hc = (float *)xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_ffn_cur = (float *)xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_post = (float *)xmalloc((size_t)DS4_N_HC * sizeof(float));
    float *cpu_ffn_comb = (float *)xmalloc((size_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    float *cpu_ffn_norm = (float *)xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_shared = (float *)xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_routed = (float *)xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_out = (float *)xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_ffn_hc = (float *)xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_logits = (float *)xmalloc((size_t)vocab_dim * sizeof(float));
    float *gpu_hc = (float *)xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_attn_cur = (float *)xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_attn_norm = (float *)xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_q = (float *)xmalloc((size_t)q_dim * sizeof(float));
    float *gpu_kv = (float *)xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *gpu_raw = (float *)xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *gpu_attn_out = (float *)xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_attn_hc = (float *)xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_ffn_cur = (float *)xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_norm = (float *)xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_shared = (float *)xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_routed = (float *)xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_out = (float *)xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_ffn_hc = (float *)xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_logits = (float *)xmalloc((size_t)vocab_dim * sizeof(float));
    int gpu_selected[DS4_MAX_EXPERT_USED];
    float gpu_expert_weight[DS4_MAX_EXPERT_USED];
    float *routed_mid_all = (float *)xmalloc((size_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float));
    block_q8_K *routed_xq = (block_q8_K *)xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(block_q8_K));
    block_q8_K *routed_midq = (block_q8_K *)xmalloc((size_t)DS4_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(block_q8_K));
    int selected[DS4_MAX_EXPERT_USED];
    float expert_weight[DS4_MAX_EXPERT_USED];

    embed_token_f16(model, weights, token, plain);
    hc_from_plain_embedding(cpu_hc, plain, DS4_N_EMBD, DS4_N_HC);
    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          cpu_hc, cpu_attn_cur, cpu_post, cpu_comb);
    layer_attn_norm_one(cpu_attn_norm, model, layer, cpu_attn_cur);
    layer_q_projection_with_lora_one(model, layer, cpu_attn_norm, cpu_q, cpu_qr_norm);
    layer_kv_projection_normed_one(model, layer, cpu_attn_norm, cpu_kv);
    rope_tail_layer_inplace(cpu_q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, false);
    rope_tail_layer_inplace(cpu_kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM);
    layer_attention_rows_one(cpu_heads, model, layer, cpu_q, cpu_kv, 1);
    rope_tail_layer_inplace(cpu_heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, true);
    layer_grouped_out_one(cpu_attn_out, model, layer, cpu_heads);
    hc_post_one(cpu_after_attn_hc, cpu_attn_out, cpu_hc, cpu_post, cpu_comb, DS4_N_EMBD, DS4_N_HC);
    hc_pre_from_state_one(model,
                          layer->hc_ffn_fn,
                          layer->hc_ffn_scale,
                          layer->hc_ffn_base,
                          cpu_after_attn_hc, cpu_ffn_cur, cpu_ffn_post, cpu_ffn_comb);
    rms_norm_weight(cpu_ffn_norm, cpu_ffn_cur, (const float *)tensor_data(model, layer->ffn_norm), DS4_N_EMBD, DS4_RMS_EPS);
    layer_shared_ffn_one(cpu_shared, model, layer, cpu_ffn_norm);
    layer_routed_moe_one_prealloc(cpu_routed,
                                  model,
                                  layer,
                                  cpu_ffn_norm,
                                  0,
                                  token,
                                  DS4_SWIGLU_CLAMP_EXP,
                                  routed_mid_all,
                                  routed_xq,
                                  routed_midq);
    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, token);
        layer_hash_router_weights_one(expert_weight, model, layer, cpu_ffn_norm, selected);
    } else {
        layer_topk_selected_experts(selected, expert_weight, model, layer, cpu_ffn_norm);
    }
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) cpu_ffn_out[i] = cpu_shared[i] + cpu_routed[i];
    hc_post_one(cpu_after_ffn_hc,
                cpu_ffn_out,
                cpu_after_attn_hc,
                cpu_ffn_post,
                cpu_ffn_comb,
                DS4_N_EMBD,
                DS4_N_HC);
    output_logits_one(cpu_logits, model, weights, cpu_after_ffn_hc);

    ds4_gpu_graph g;
    bool ok = gpu_graph_alloc(&g, weights, layer);
    g.quality = quality;
    g.materialize_ffn_out = true;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = ds4_gpu_embed_token_hc_tensor(g.cur_hc,
                                                 model->map,
                                                 model->size,
                                                 weights->token_embd->abs_offset,
                                                 (uint32_t)weights->token_embd->dim[1],
                                                 (uint32_t)token,
                                                     DS4_N_EMBD,
                                                     DS4_N_HC) != 0;
    if (ok) ok = gpu_graph_encode_decode_layer(&g,
                                               model,
                                               layer,
                                               0,
                                               0,
                                               g.layer_raw_cache[0],
                                               g.raw_cap,
                                               0,
                                               1,
                                               token);
    if (ok) {
        ds4_gpu_tensor *embedded_hc = g.cur_hc;
        g.cur_hc = g.after_ffn_hc;
        g.after_ffn_hc = embedded_hc;
    }
    if (ok) ok = gpu_graph_encode_output_head(&g, model, weights, vocab_dim);
    if (ok) ok = ds4_gpu_end_commands() != 0;

    if (ok) {
        ok = ds4_read_hc_carrier_f32(g.after_ffn_hc, 0, gpu_hc, hc_dim) != 0 &&
             ds4_gpu_tensor_read(g.attn_cur, 0, gpu_attn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.attn_norm, 0, gpu_attn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.q, 0, gpu_q, q_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.kv, 0, gpu_kv, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) != 0 &&
             gpu_graph_read_raw_row_f32(&g, 0, 0, gpu_raw) != 0 &&
             ds4_gpu_tensor_read(g.attn_out, 0, gpu_attn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_read_hc_carrier_f32(g.after_attn_hc, 0, gpu_after_attn_hc, hc_dim) != 0 &&
             ds4_gpu_tensor_read(g.ffn_cur, 0, gpu_ffn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.ffn_norm, 0, gpu_ffn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.shared_out, 0, gpu_shared, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.router_selected, 0, gpu_selected, sizeof(gpu_selected)) != 0 &&
             ds4_gpu_tensor_read(g.router_weights, 0, gpu_expert_weight, sizeof(gpu_expert_weight)) != 0 &&
             ds4_gpu_tensor_read(g.routed_out, 0, gpu_routed, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.ffn_out, 0, gpu_ffn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_read_hc_carrier_f32(g.cur_hc, 0, gpu_after_ffn_hc, hc_dim) != 0 &&
             ds4_gpu_tensor_read(g.logits, 0, gpu_logits, vocab_dim * sizeof(float)) != 0;
    }

    if (ok) {
        fprintf(stderr,
                "ds4: GPU graph test layer0 diffs: embed_hc=%g hc_pre=%g attn_norm=%g q_rope=%g kv_rope=%g raw_cache=%g attn_out=%g after_attn_hc=%g ffn_cur=%g ffn_norm=%g shared=%g router_w=%g routed=%g ffn_out=%g after_ffn_hc=%g logits=%g\n",
                max_abs_diff(cpu_hc, gpu_hc, hc_dim),
                max_abs_diff(cpu_attn_cur, gpu_attn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_attn_norm, gpu_attn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_q, gpu_q, q_dim),
                max_abs_diff(cpu_kv, gpu_kv, DS4_N_HEAD_DIM),
                max_abs_diff(cpu_kv, gpu_raw, DS4_N_HEAD_DIM),
                max_abs_diff(cpu_attn_out, gpu_attn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_attn_hc, gpu_after_attn_hc, hc_dim),
                max_abs_diff(cpu_ffn_cur, gpu_ffn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_norm, gpu_ffn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD),
                max_abs_diff(expert_weight, gpu_expert_weight, DS4_N_EXPERT_USED),
                max_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_out, gpu_ffn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_ffn_hc, gpu_after_ffn_hc, hc_dim),
                max_abs_diff(cpu_logits, gpu_logits, vocab_dim));
        if (memcmp(selected, gpu_selected, sizeof(selected)) != 0) {
            fprintf(stderr,
                    "ds4: GPU graph router selected mismatch: cpu=[%d,%d,%d,%d,%d,%d] gpu=[%d,%d,%d,%d,%d,%d]\n",
                    selected[0], selected[1], selected[2], selected[3], selected[4], selected[5],
                    gpu_selected[0], gpu_selected[1], gpu_selected[2], gpu_selected[3], gpu_selected[4], gpu_selected[5]);
        }
        print_vec_stats("GPU graph q", gpu_q, q_dim);
        print_vec_stats("GPU graph kv", gpu_kv, DS4_N_HEAD_DIM);
        print_vec_stats("GPU graph routed", gpu_routed, DS4_N_EMBD);
    } else {
        fprintf(stderr, "ds4: GPU graph test failed while encoding first decode stages\n");
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: GPU synchronize after graph test failure also failed\n");
        }
    }

    gpu_graph_free(&g);
    free(routed_midq);
    free(routed_xq);
    free(routed_mid_all);
    free(gpu_logits);
    free(gpu_after_ffn_hc);
    free(gpu_ffn_out);
    free(gpu_routed);
    free(gpu_shared);
    free(gpu_ffn_norm);
    free(gpu_ffn_cur);
    free(gpu_after_attn_hc);
    free(gpu_attn_out);
    free(gpu_raw);
    free(gpu_kv);
    free(gpu_q);
    free(gpu_attn_norm);
    free(gpu_attn_cur);
    free(gpu_hc);
    free(cpu_kv);
    free(cpu_q);
    free(cpu_attn_out);
    free(cpu_heads);
    free(cpu_ffn_norm);
    free(cpu_routed);
    free(cpu_logits);
    free(cpu_after_ffn_hc);
    free(cpu_ffn_out);
    free(cpu_shared);
    free(cpu_ffn_comb);
    free(cpu_ffn_post);
    free(cpu_ffn_cur);
    free(cpu_after_attn_hc);
    free(cpu_qr_norm);
    free(cpu_attn_norm);
    free(cpu_comb);
    free(cpu_post);
    free(cpu_attn_cur);
    free(cpu_hc);
    free(plain);
    return ok ? 0 : 1;
}






/* =========================================================================
 * GPU Release Decode and Prefill.
 * =========================================================================
 *
 * Everything below is the user-facing GPU backend.  It uses the same layer
 * encoder as diagnostics, but diagnostics are not required for normal command
 * flow and their CPU reads stay outside these generation entry points.
 */

uint32_t gpu_graph_token_split_after_layers(void) {
    uint32_t split_after_layers = 4;
    const char *split_env = getenv("DS4_CUDA_GRAPH_TOKEN_SPLIT_LAYERS");
    if (split_env && split_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(split_env, &end, 10);
        if (end != split_env && v <= DS4_N_LAYER) split_after_layers = (uint32_t)v;
    }
    return split_after_layers;
}

