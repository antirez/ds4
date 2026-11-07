/* Recursive Flint (fork-owned): the decode engine on the host side.
 * Textually included into ds4.c before ds4_engine_open_internal.
 *
 * Builds rec's 50 x 64 per-layer descriptor table from the bound tensors
 * (offsets in 256-byte units into the mapped GGUF), the expert pack offset
 * table and the silu lookup table, opens the Metal engine
 * (ds4_metal_flint.m) and provides the trace check that compares one
 * decode against rec (tests/flint/trace_compare.py).  Sessions, prefill and
 * generation follow in later steps of docs/RECURSIVE_FLINT.md. */

#include "ds4_flint_gpu.h"

static uint32_t g_flint_meta[DS4_FLINT_GPU_META_WORDS];
static uint64_t g_flint_pack_offsets[DS4_FLINT_GPU_N_LAYER * DS4_FLINT_GPU_N_EXPERT];
static uint16_t g_flint_silu[DS4_FLINT_GPU_SILU_ENTRIES];
static bool g_flint_gpu_open;

static uint32_t flint_off256(const ds4_tensor *t) {
    if (!t) ds4_die("flint: descriptor references a missing tensor");
    if (t->abs_offset % 256u != 0) {
        fprintf(stderr, "ds4: flint tensor %.*s is not 256-byte aligned\n", (int)t->name.len, t->name.ptr);
        exit(1);
    }
    if ((t->abs_offset >> 8) > UINT32_MAX) ds4_die("flint: tensor offset exceeds the descriptor range");
    return (uint32_t)(t->abs_offset >> 8);
}

static void flint_linear_desc(uint32_t *d, const ds4_tensor *w, const flint_quant_aux *aux,
                              uint32_t n, uint32_t k) {
    d[0] = flint_off256(w);
    d[1] = flint_off256(aux->scales);
    d[2] = flint_off256(aux->biases);
    d[3] = n;
    d[4] = k;
}

/* rec's layout (FullModel.__init__): [0] bank, [1] global, [2] routed,
 * [3] input norm, [4] post norm, [5] q norm, [6] k norm, [7] sinks, [8] f32
 * router (unused by decode), [9] expert bias (none for Flint), linear blocks
 * of five words at 10 q, 15 k, 20 v, 25 o, 30 gate, 35 up, 40 down, 45/50/55
 * shared gate/up/down, [60] bf16 router, [61]/[62] KV offset and capacity
 * (per session).  Row 49: [3] final norm, 10 embedding, 30 head. */
static void flint_build_metadata(const ds4_weights *w) {
    memset(g_flint_meta, 0, sizeof(g_flint_meta));
    memset(g_flint_pack_offsets, 0, sizeof(g_flint_pack_offsets));
    const uint32_t q_dim = DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t kv_dim = DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        const flint_layer_aux *a = &g_flint_layer_aux[il];
        uint32_t *d = g_flint_meta + il * 64;
        d[1] = ds4_flint_layer_is_global(il) ? 1u : 0u;
        d[2] = il >= DS4_N_LEADING_DENSE ? 1u : 0u;
        d[3] = flint_off256(l->attn_norm);
        d[4] = flint_off256(l->ffn_norm);
        d[5] = flint_off256(l->attn_q_norm);
        d[6] = flint_off256(l->attn_k_norm);
        d[7] = flint_off256(l->attn_sinks);
        flint_linear_desc(d + 10, l->attn_q, &a->attn_q, q_dim, DS4_N_EMBD);
        flint_linear_desc(d + 15, l->attn_k, &a->attn_k, kv_dim, DS4_N_EMBD);
        flint_linear_desc(d + 20, l->attn_v, &a->attn_v, kv_dim, DS4_N_EMBD);
        flint_linear_desc(d + 25, l->attn_output, &a->attn_output, DS4_N_EMBD, q_dim);
        if (!d[2]) {
            flint_linear_desc(d + 30, l->ffn_gate, &a->ffn_gate, DS4_N_FF_DENSE, DS4_N_EMBD);
            flint_linear_desc(d + 35, l->ffn_up, &a->ffn_up, DS4_N_FF_DENSE, DS4_N_EMBD);
            flint_linear_desc(d + 40, l->ffn_down, &a->ffn_down, DS4_N_EMBD, DS4_N_FF_DENSE);
        } else {
            flint_linear_desc(d + 45, l->ffn_gate_shexp, &a->ffn_gate_shexp, DS4_FLINT_N_FF_SHARED, DS4_N_EMBD);
            flint_linear_desc(d + 50, l->ffn_up_shexp, &a->ffn_up_shexp, DS4_FLINT_N_FF_SHARED, DS4_N_EMBD);
            flint_linear_desc(d + 55, l->ffn_down_shexp, &a->ffn_down_shexp, DS4_N_EMBD, DS4_FLINT_N_FF_SHARED);
            d[60] = flint_off256(l->ffn_gate_inp);
            const ds4_tensor *pack = a->ffn_exps_pack;
            if (!pack) ds4_die("flint: expert pack is missing");
            for (uint32_t e = 0; e < DS4_N_EXPERT; e++) {
                g_flint_pack_offsets[il * DS4_FLINT_GPU_N_EXPERT + e] =
                    pack->abs_offset + (uint64_t)e * DS4_FLINT_EXPERT_PACK_BYTES;
            }
        }
    }
    uint32_t *f = g_flint_meta + 49 * 64;
    f[3] = flint_off256(w->output_norm);
    flint_linear_desc(f + 10, w->token_embd, &g_flint_token_embd_aux, DS4_N_VOCAB, DS4_N_EMBD);
    flint_linear_desc(f + 30, w->output, &g_flint_output_aux, DS4_N_VOCAB, DS4_N_EMBD);
}

static float flint_bf16_to_f32(uint16_t bits) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)bits << 16;
    return v.f;
}

/* Round to nearest even, as MLX and Metal do; NaN keeps a quiet payload. */
static uint16_t flint_f32_to_bf16(float f) {
    union { uint32_t u; float f; } v;
    v.f = f;
    if ((v.u & 0x7f800000u) == 0x7f800000u && (v.u & 0x007fffffu)) return (uint16_t)((v.u >> 16) | 0x40u);
    const uint32_t rounding = 0x7fffu + ((v.u >> 16) & 1u);
    return (uint16_t)((v.u + rounding) >> 16);
}

static float flint_ftz(float f) {
    union { uint32_t u; float f; } v;
    v.f = f;
    if ((v.u & 0x7f800000u) == 0 && (v.u & 0x007fffffu)) v.u &= 0x80000000u;
    return v.f;
}

static float flint_bf16_round(float f) {
    return flint_bf16_to_f32(flint_f32_to_bf16(flint_ftz(f)));
}

/* silu(x) = x * sigmoid(x) for every bf16 bit pattern; the shared expert's
 * activation reads it.  The converter stores rec's table (built by mlx on
 * the GPU) as flint.silu_table; this CPU formula mirrors mlx's bf16 sigmoid
 * (tail = 1 / (1 + exp(|x|)), rounded after each op, denormals flushed) and
 * differs from it in one entry, so the stored table wins when present. */
static void flint_build_silu(const ds4_model *m) {
    if (g_flint_silu_table) {
        memcpy(g_flint_silu, tensor_data(m, g_flint_silu_table), sizeof(g_flint_silu));
        return;
    }
    fprintf(stderr, "ds4: flint.silu_table is missing from the GGUF; computing it on the CPU\n");
    for (uint32_t i = 0; i < DS4_FLINT_GPU_SILU_ENTRIES; i++) {
        const float x = flint_ftz(flint_bf16_to_f32((uint16_t)i));
        const float e = flint_bf16_round(expf(fabsf(x)));
        const float tail = flint_bf16_round(1.0f / flint_bf16_round(1.0f + e));
        const float sig = x < 0.0f ? tail : flint_bf16_round(1.0f - tail);
        g_flint_silu[i] = flint_f32_to_bf16(flint_ftz(x * sig));
    }
}

static int flint_engine_open_tail(ds4_engine *e, const ds4_engine_options *opt, ds4_engine **out) {
    (void)opt;
    vocab_load(&e->vocab, &e->model);
    const char *why = NULL;
    if (e->backend != DS4_BACKEND_METAL) {
        why = "needs the Metal backend";
    } else if (!ds4_gpu_init()) {
        why = "could not initialize Metal";
    } else {
        flint_build_metadata(&e->weights);
        flint_build_silu(&e->model);
        if (ds4_flint_gpu_engine_open(e->model.map, e->model.size, g_flint_meta,
                                      g_flint_pack_offsets, g_flint_silu, &why)) {
            g_flint_gpu_open = true;
        }
    }
    if (!g_flint_gpu_open) {
        fprintf(stderr, "ds4: Recursive Flint %s\n", why ? why : "failed to open the GPU engine");
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }
    e->metal_ready = true;
    *out = e;
    return 0;
}

static void flint_engine_close(void) {
    if (!g_flint_gpu_open) return;
    ds4_flint_gpu_engine_release();
    g_flint_gpu_open = false;
}

/* The decode trace check.  Input (little endian): "FLTR", u32 version 1,
 * u32 prefill length P, u32 n_tokens, u32 n_layers (49), i32 tokens[n],
 * then per layer u32 rows followed by K[4][rows][128] and V[4][rows][128]
 * bf16 (global layers: positions 0..rows-1; sliding layers: ring slots).
 * Decodes tokens[i] at position P+i and writes "FLTO", u32 n_steps, u32
 * n_vocab, u32 trace elements, then per step f32 logits and u16 trace. */
static bool flint_read_u32(FILE *f, uint32_t *v) { return fread(v, 4, 1, f) == 1; }

static double flint_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

int ds4_flint_trace_check(ds4_engine *e, const char *in_path, const char *out_path) {
    if (!e || !g_flint_gpu_open) {
        fprintf(stderr, "ds4: the Flint trace check needs an open Flint engine\n");
        return 1;
    }
    FILE *in = fopen(in_path, "rb");
    if (!in) { perror(in_path); return 1; }
    char magic[4];
    uint32_t version = 0, prefill = 0, n_tokens = 0, n_layers = 0;
    if (fread(magic, 1, 4, in) != 4 || memcmp(magic, "FLTR", 4) != 0 ||
        !flint_read_u32(in, &version) || version != 1 || !flint_read_u32(in, &prefill) ||
        !flint_read_u32(in, &n_tokens) || !flint_read_u32(in, &n_layers) || n_layers != DS4_N_LAYER ||
        n_tokens == 0 || n_tokens > 4096) {
        fprintf(stderr, "ds4: %s is not a Flint trace input\n", in_path);
        fclose(in);
        return 1;
    }
    int32_t *tokens = xcalloc(n_tokens, sizeof(int32_t));
    if (fread(tokens, sizeof(int32_t), n_tokens, in) != n_tokens) {
        fprintf(stderr, "ds4: short read of tokens in %s\n", in_path);
        free(tokens); fclose(in);
        return 1;
    }
    const uint32_t ctx_cap = prefill + n_tokens + 1;
    ds4_flint_gpu_session *s = ds4_flint_gpu_session_create(ctx_cap);
    if (!s) { free(tokens); fclose(in); return 1; }
    /* The imported KV is kept so the warm passes below can restore it: a
     * decode writes ring slots and positions the measured pass must see as
     * prefill left them. */
    uint16_t *kv_rows[DS4_FLINT_GPU_N_LAYER] = {0};
    uint32_t kv_count[DS4_FLINT_GPU_N_LAYER] = {0};
    uint16_t *rows = NULL;
    for (uint32_t il = 0; il < n_layers; il++) {
        uint32_t n_rows = 0;
        if (!flint_read_u32(in, &n_rows) || n_rows > ctx_cap) {
            fprintf(stderr, "ds4: bad KV row count in layer %u\n", il);
            goto fail;
        }
        kv_count[il] = n_rows;
        if (n_rows) {
            kv_rows[il] = xcalloc((size_t)n_rows * 8u * 128u, sizeof(uint16_t));   /* K then V, 4 heads each */
            if (fread(kv_rows[il], 128u * sizeof(uint16_t), (size_t)n_rows * 8u, in) != (size_t)n_rows * 8u) {
                fprintf(stderr, "ds4: short KV read in layer %u\n", il);
                goto fail;
            }
        }
    }
#define FLINT_TRACE_IMPORT_KV() \
    for (uint32_t il = 0; il < n_layers; il++) { \
        for (uint32_t slot = 0; slot < 8u && kv_count[il]; slot++) { \
            if (!ds4_flint_gpu_kv_write(s, il, slot < 4u, slot % 4u, 0, kv_count[il], \
                                        kv_rows[il] + (size_t)slot * kv_count[il] * 128u)) { \
                fprintf(stderr, "ds4: KV write failed in layer %u\n", il); \
                goto fail; \
            } \
        } \
    }
    FLINT_TRACE_IMPORT_KV();
    fclose(in);
    in = NULL;
    {
        FILE *outf = fopen(out_path, "wb");
        if (!outf) { perror(out_path); goto fail; }
        const uint32_t header[3] = { n_tokens, DS4_FLINT_GPU_N_VOCAB, DS4_FLINT_GPU_TRACE_ELEMS };
        fwrite("FLTO", 1, 4, outf);
        fwrite(header, 4, 3, outf);
        uint16_t *logits = xcalloc(DS4_FLINT_GPU_N_VOCAB, sizeof(uint16_t));
        float *logits_f = xcalloc(DS4_FLINT_GPU_N_VOCAB, sizeof(float));
        uint16_t *trace = xcalloc(DS4_FLINT_GPU_TRACE_ELEMS, sizeof(uint16_t));
        /* DS4_FLINT_TRACE_WARM=N first runs the same steps N extra times
         * (results discarded) to separate cold page-ins from steady state. */
        const char *warm_env = getenv("DS4_FLINT_TRACE_WARM");
        for (int warm = warm_env ? atoi(warm_env) : 0; warm > 0; warm--) {
            const double w0 = flint_now_ms();
            for (uint32_t i = 0; i < n_tokens; i++) {
                char err[256] = {0};
                const double s0 = flint_now_ms();
                if (!ds4_flint_gpu_decode(s, (uint32_t)tokens[i], prefill + i, prefill, logits, NULL, err, sizeof(err))) {
                    fprintf(stderr, "ds4: Flint warm step %u failed: %s\n", i, err);
                    break;
                }
                fprintf(stderr, "ds4: warm step %u: %.1f ms\n", i, flint_now_ms() - s0);
            }
            fprintf(stderr, "ds4: warm pass: %.1f ms/token\n", (flint_now_ms() - w0) / n_tokens);
            FLINT_TRACE_IMPORT_KV();
        }
        const double t0 = flint_now_ms();
        for (uint32_t i = 0; i < n_tokens; i++) {
            char err[256] = {0};
            if (!ds4_flint_gpu_decode(s, (uint32_t)tokens[i], prefill + i, prefill, logits, trace, err, sizeof(err))) {
                fprintf(stderr, "ds4: Flint decode step %u failed: %s\n", i, err);
                fclose(outf); free(logits); free(logits_f); free(trace);
                goto fail;
            }
            for (uint32_t v = 0; v < DS4_FLINT_GPU_N_VOCAB; v++) logits_f[v] = flint_bf16_to_f32(logits[v]);
            fwrite(logits_f, sizeof(float), DS4_FLINT_GPU_N_VOCAB, outf);
            fwrite(trace, sizeof(uint16_t), DS4_FLINT_GPU_TRACE_ELEMS, outf);
        }
        const double dt = flint_now_ms() - t0;
        fprintf(stderr, "ds4: Flint trace check: %u steps from position %u in %.1f ms (%.1f ms/token), wrote %s\n",
                n_tokens, prefill, dt, dt / n_tokens, out_path);
        fclose(outf); free(logits); free(logits_f); free(trace);
        /* DS4_FLINT_TRACE_KV_LAYER=L appends that layer's whole K and V
         * (4 heads x slots x 128 bf16 each) to OUT.kv for comparison. */
        const char *kv_env = getenv("DS4_FLINT_TRACE_KV_LAYER");
        if (kv_env) {
            const uint32_t il = (uint32_t)atoi(kv_env);
            const uint32_t slots = ds4_flint_layer_is_global(il) ? prefill + n_tokens : DS4_FLINT_GPU_SWA;
            char path[1024];
            snprintf(path, sizeof(path), "%s.kv", out_path);
            FILE *kf = fopen(path, "wb");
            uint16_t *buf = xcalloc((size_t)slots * 128u, sizeof(uint16_t));
            if (kf) {
                fwrite(&slots, 4, 1, kf);
                for (int key = 1; key >= 0; key--) {
                    for (uint32_t head = 0; head < DS4_N_HEAD_KV; head++) {
                        ds4_flint_gpu_kv_read(s, il, key, head, 0, slots, buf);
                        fwrite(buf, sizeof(uint16_t), (size_t)slots * 128u, kf);
                    }
                }
                fclose(kf);
                fprintf(stderr, "ds4: wrote layer %u KV (%u slots) to %s\n", il, slots, path);
            }
            free(buf);
        }
    }
    for (uint32_t il = 0; il < n_layers; il++) free(kv_rows[il]);
    free(rows); free(tokens);
    ds4_flint_gpu_session_free(s);
    return 0;
fail:
    if (in) fclose(in);
    for (uint32_t il = 0; il < n_layers; il++) free(kv_rows[il]);
    free(rows); free(tokens);
    ds4_flint_gpu_session_free(s);
    return 1;
#undef FLINT_TRACE_IMPORT_KV
}
