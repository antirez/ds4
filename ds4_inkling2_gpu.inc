/* Inkling 2 Mini GPU graph (Metal): the same 48-visit schedule as the CPU oracle in ds4_inkling2.inc, one
 * function (ikg_forward_chunk) for chunked prefill and one-token decoding.  Included by ds4_inkling2.inc when a GPU
 * backend exists.  Kernels: metal/inkling2.metal; wrappers: ds4_inkling2_metal.inc.
 *
 * State (mirrors ik_state): per visit slot K and V rows as half (local slots: ring of 1024 + chunk rows, the eight
 * global slots: one row per position), conv tails f32.  Everything else (block starts, loop messages, router
 * results) is recomputed per chunk in the stream buffers below.
 *
 * Streams: stream 0 is the embedding, stream v+1 the output of the v-th visit executed; block starts, the loop bank
 * and the final mix refer to streams by index.  Visits are committed one command buffer each so no single buffer
 * runs for long (the GPU is shared). */

#define IKG_REF(tensor) ((ds4_gpu_ik_ref){ (tensor), 0 })
#define IKG_REFO(tensor, offset) ((ds4_gpu_ik_ref){ (tensor), (uint64_t)(offset) })
#define IKG_NONE ((ds4_gpu_ik_ref){ NULL, 0 })
#define IKG_NSTREAM (1 + IK_NSLOT)

struct ik_gpu {
    const ds4_inkling2 *w;
    const ds4_model *m;
    uint32_t cap, chunk, pos, max_want;
    uint32_t rows[IK_NSLOT];
    ds4_gpu_tensor *k[IK_NSLOT], *v[IK_NSLOT], *conv[IK_NSLOT];
    ds4_gpu_tensor *streams, *mixed, *ibuf[3], *Q, *Kp, *Vp, *Rr, *sink, *gate, *dk, *dv, *Kc, *Vc, *O, *ap, *a, *fm, *h, *f, *fc;
    ds4_gpu_tensor *rlogits, *picks, *wts, *lists, *counts, *tiles, *ac, *yd;
    ds4_gpu_tensor *ctl_g, *ctl_d, *ctl_gain, *ctl_forget, *msg, *tokens, *tau, *want, *logits, *hh, *mo, *forced;
    ds4_gpu_tensor *apart;           /* decode attention partials: [24 heads][apart_splits][130] floats */
    uint32_t apart_splits;
    ds4_gpu_tensor *snap_k[IK_NSLOT], *snap_v[IK_NSLOT], *snap_conv[IK_NSLOT];   /* copy of the sliding-window rings and conv tails */
    uint32_t snap_pos;               /* position the copy was taken at (0: none) */
    uint64_t bytes_state, bytes_scratch;
    uint32_t forced_ntok;            /* DS4_INKLING2_FT_PICKS: router decisions taken from a file */
    int32_t *pick_log;               /* host copy of every decision, [48][cap][8], when picks are being dumped */
    FILE *dump;                      /* DS4_INKLING2_FT_VISITS: per visit l_out sums and expert ids */
};

static ds4_gpu_ik_w ikg_w(const ik_gpu *g, const ds4_tensor *t) {
    ds4_gpu_ik_w w = { g->m->map, g->m->size, t->abs_offset, t->bytes, t->type, 0 };
    return w;
}
static ds4_gpu_ik_w ikg_none(void) { ds4_gpu_ik_w w = {0}; return w; }

/* K/V rows are rounded up to a multiple of 8: the matrix attention reads whole 8-key tiles and otherwise falls back to the slow scalar kernel. */
static uint32_t ikg_rows(uint32_t n) { return (n + 7u) & ~7u; }

/* Bytes of GPU memory a state with these limits needs (K/V caches, conv tails, scratch), for the honest estimate. */
static void ikg_estimate(uint32_t cap, uint32_t chunk, uint32_t max_want, uint64_t *state, uint64_t *scratch) {
    uint64_t st = 0;
    for (int s = 0; s < IK_NSLOT; s++) {
        const uint64_t rows = ikg_rows(ik_layer_is_global(ik_slot_layer(s)) ? cap : IK_EXTENT + chunk);
        st += 2 * rows * IK_HK * IK_D * 2 + (uint64_t)IK_CONVC * 3 * 4;
        if (!ik_layer_is_global(ik_slot_layer(s))) st += 2 * rows * IK_HK * IK_D * 2 + (uint64_t)IK_CONVC * 3 * 4;   /* snapshot copy (ikg_snapshot) */
    }
    const uint64_t NE = (uint64_t)chunk * IK_E * 4;
    uint64_t sc = (uint64_t)IKG_NSTREAM * NE + 3 * NE + 3 * NE;                    /* streams, mixed, ibuf */
    sc += (uint64_t)chunk * (IK_H * IK_D * 2 + IK_HK * IK_D * 4 + IK_H * IK_R + IK_H + 2 * IK_KC * IK_HK) * 4;   /* Q gate Kp Vp Kc Vc Rr sink dk dv */
    sc += (uint64_t)chunk * IK_H * IK_D * 4;                                        /* O */
    sc += 8 * NE;                                                                   /* ap a fm h f fc mo + msg share */
    sc += 4 * NE;                                                                   /* msg */
    sc += (uint64_t)chunk * 2 * IK_E * 4 + (uint64_t)chunk * (64 + 2 * IK_E) * 4;   /* control */
    sc += (uint64_t)chunk * (IK_NEXP * 4 + 8 * 4 + 8 * 4 + 4 + 4);                  /* router logits, wts, picks(one slot), tokens, tau */
    sc += (uint64_t)IK_NSLOT * chunk * 8 * 4 + (uint64_t)IK_NEXP * chunk * 4 + 512 + ((uint64_t)chunk * 8 / 32 + 130) * 8; /* picks of all slots, lists, counts, tiles */
    sc += (uint64_t)chunk * 8 * IK_FF * 4 + (uint64_t)chunk * 8 * IK_E * 4;         /* ac, yd */
    sc += (uint64_t)max_want * (IK_VOCAB * 4 + IK_E * 4 + 4);                       /* logits, hh, want */
    *state = st;
    *scratch = sc;
}

static uint64_t g_ikg_alloc_bytes;   /* what ikg_alloc has handed out (printed after a state is built) */

static ds4_gpu_tensor *ikg_alloc(uint64_t bytes, const char *what, bool zero) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes ? bytes : 16);
    g_ikg_alloc_bytes += bytes;
    if (!t) { fprintf(stderr, "ds4: inkling2 gpu: cannot allocate %.1f MiB for %s\n", (double)bytes / 1048576.0, what); return NULL; }
    if (zero && !ds4_gpu_tensor_fill_f32(t, 0.0f, bytes / 4)) { ds4_gpu_tensor_free(t); return NULL; }
    return t;
}

static void ikg_free(ik_gpu *g) {
    if (!g) return;
    ds4_gpu_tensor **all[] = { &g->streams, &g->mixed, &g->ibuf[0], &g->ibuf[1], &g->ibuf[2], &g->Q, &g->Kp, &g->Vp, &g->Rr, &g->sink,
        &g->gate, &g->dk, &g->dv, &g->Kc, &g->Vc, &g->O, &g->ap, &g->a, &g->fm, &g->h, &g->f, &g->fc, &g->rlogits, &g->picks,
        &g->wts, &g->lists, &g->counts, &g->tiles, &g->ac, &g->yd, &g->ctl_g, &g->ctl_d, &g->ctl_gain, &g->ctl_forget, &g->msg, &g->tokens,
        &g->tau, &g->want, &g->logits, &g->hh, &g->mo, &g->forced, &g->apart };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) if (*all[i]) ds4_gpu_tensor_free(*all[i]);
    for (int s = 0; s < IK_NSLOT; s++) {
        if (g->snap_k[s]) ds4_gpu_tensor_free(g->snap_k[s]);
        if (g->snap_v[s]) ds4_gpu_tensor_free(g->snap_v[s]);
        if (g->snap_conv[s]) ds4_gpu_tensor_free(g->snap_conv[s]);
        if (g->k[s]) ds4_gpu_tensor_free(g->k[s]);
        if (g->v[s]) ds4_gpu_tensor_free(g->v[s]);
        if (g->conv[s]) ds4_gpu_tensor_free(g->conv[s]);
    }
    free(g->pick_log);
    free(g);
}

/* A new state: positions 0..cap-1, chunks of at most `chunk` tokens, up to max_want logit rows per chunk. */
static ik_gpu *ikg_new_ex(const ds4_inkling2 *w, uint32_t cap, uint32_t chunk, uint32_t max_want, bool scratch_only);
static ik_gpu *ikg_new(const ds4_inkling2 *w, uint32_t cap, uint32_t chunk, uint32_t max_want) { return ikg_new_ex(w, cap, chunk, max_want, false); }
/* scratch_only: no K/V or conv state (the shared scratch of batched decode; cap only sizes the attention partials) */
static ik_gpu *ikg_new_ex(const ds4_inkling2 *w, uint32_t cap, uint32_t chunk, uint32_t max_want, bool scratch_only) {
    if (cap == 0 || chunk == 0 || max_want == 0 || max_want > chunk) return NULL;
    ik_gpu *g = xcalloc(1, sizeof(*g));
    g_ikg_alloc_bytes = 0;
    g->w = w; g->m = w->m; g->cap = cap; g->chunk = chunk; g->max_want = max_want;
    ikg_estimate(cap, chunk, max_want, &g->bytes_state, &g->bytes_scratch);
    if (scratch_only) g->bytes_state = 0;
    if (!scratch_only) fprintf(stderr, "ds4: inkling2 gpu: context %u, chunk %u: K/V + conv state %.1f MiB, scratch %.1f MiB (weights are mapped from the file)\n",
            cap, chunk, (double)g->bytes_state / 1048576.0, (double)g->bytes_scratch / 1048576.0);
    bool ok = true;
    for (int s = 0; s < IK_NSLOT && ok && !scratch_only; s++) {
        g->rows[s] = ikg_rows(ik_layer_is_global(ik_slot_layer(s)) ? cap : IK_EXTENT + chunk);
        /* zeroed: the matrix attention reads whole 8-key tiles, rows not written yet must be finite (0 x garbage would poison V products) */
        g->k[s] = ikg_alloc((uint64_t)g->rows[s] * IK_HK * IK_D * 2, "K cache", true);
        g->v[s] = ikg_alloc((uint64_t)g->rows[s] * IK_HK * IK_D * 2, "V cache", true);
        g->conv[s] = ikg_alloc((uint64_t)IK_CONVC * 3 * 4, "conv tail", true);
        ok = g->k[s] && g->v[s] && g->conv[s];
    }
    const uint64_t n = chunk, NE = n * IK_E * 4;
#define IKG_A(field, bytes) do { if (ok) { g->field = ikg_alloc((bytes), #field, false); ok = g->field != NULL; } } while (0)
    IKG_A(streams, (uint64_t)IKG_NSTREAM * NE);
    IKG_A(mixed, 3 * NE); IKG_A(ibuf[0], NE); IKG_A(ibuf[1], NE); IKG_A(ibuf[2], NE);
    IKG_A(Q, n * IK_H * IK_D * 4); IKG_A(Kp, n * IK_HK * IK_D * 4); IKG_A(Vp, n * IK_HK * IK_D * 4);
    IKG_A(Rr, n * IK_H * IK_R * 4); IKG_A(sink, n * IK_H * 4); IKG_A(gate, n * IK_H * IK_D * 4);
    IKG_A(dk, n * IK_KC * IK_HK * 4); IKG_A(dv, n * IK_KC * IK_HK * 4);
    IKG_A(Kc, n * IK_HK * IK_D * 4); IKG_A(Vc, n * IK_HK * IK_D * 4); IKG_A(O, n * IK_H * IK_D * 4);
    IKG_A(ap, NE); IKG_A(a, NE); IKG_A(fm, NE); IKG_A(h, NE); IKG_A(f, NE); IKG_A(fc, NE); IKG_A(mo, NE);
    IKG_A(rlogits, n * IK_NEXP * 4); IKG_A(picks, (uint64_t)IK_NSLOT * n * IK_TOPK * 4); IKG_A(wts, n * IK_TOPK * 4);
    IKG_A(lists, (uint64_t)IK_NEXP * n * 4); IKG_A(counts, IK_NEXP * 4); IKG_A(tiles, ((uint64_t)n * 8 / 32 + 130) * 8);
    IKG_A(ac, n * IK_TOPK * IK_FF * 4); IKG_A(yd, n * IK_TOPK * IK_E * 4);
    IKG_A(ctl_g, n * 2 * IK_E * 4); IKG_A(ctl_d, (n < 4 ? 4 : n) * 64 * 4); IKG_A(ctl_gain, NE); IKG_A(ctl_forget, NE); IKG_A(msg, 4 * NE);
    g->apart_splits = (cap + 127u) / 128u + 1u;   /* the decode attention splits long contexts into 1024-key pieces */
    IKG_A(apart, (uint64_t)24 * g->apart_splits * 130 * 4);
    IKG_A(tokens, n * 4); IKG_A(tau, n * 4); IKG_A(want, (uint64_t)max_want * 4);
    IKG_A(logits, (uint64_t)max_want * IK_VOCAB * 4); IKG_A(hh, (uint64_t)max_want * IK_E * 4);
#undef IKG_A
    if (!ok) { ikg_free(g); return NULL; }
    if (!scratch_only) fprintf(stderr, "ds4: inkling2 gpu: allocated %.1f MiB of tensors (estimate was %.1f MiB)\n",
            (double)g_ikg_alloc_bytes / 1048576.0, (double)(g->bytes_state + g->bytes_scratch) / 1048576.0);
    return g;
}

/* State copy for prefix reuse.  The conv tails and the sliding-window rings cannot be rolled back, but the global slots' rows are
 * never overwritten (one row per position), so only the 40 window slots and the tails are copied (about 250 MiB at chunk 512);
 * restoring then makes the state exactly what it was at snap_pos.  Buffers are allocated on first use.  Returns false (and keeps
 * no snapshot) when anything fails. */
static bool ikg_snapshot(ik_gpu *g) {
    g->snap_pos = 0;
    bool ok = ds4_gpu_begin_commands() != 0;   /* tensor copies are blits inside a command batch */
    for (int s = 0; s < IK_NSLOT && ok; s++) {
        const bool glob = ik_layer_is_global(ik_slot_layer(s));
        const uint64_t kv_bytes = (uint64_t)g->rows[s] * IK_HK * IK_D * 2, conv_bytes = (uint64_t)IK_CONVC * 3 * 4;
        if (!g->snap_conv[s]) g->snap_conv[s] = ikg_alloc(conv_bytes, "snapshot conv", false);
        ok = g->snap_conv[s] && ds4_gpu_tensor_copy(g->snap_conv[s], 0, g->conv[s], 0, conv_bytes);
        if (!ok || glob) continue;
        if (!g->snap_k[s]) g->snap_k[s] = ikg_alloc(kv_bytes, "snapshot K", false);
        if (!g->snap_v[s]) g->snap_v[s] = ikg_alloc(kv_bytes, "snapshot V", false);
        ok = g->snap_k[s] && g->snap_v[s] && ds4_gpu_tensor_copy(g->snap_k[s], 0, g->k[s], 0, kv_bytes) &&
             ds4_gpu_tensor_copy(g->snap_v[s], 0, g->v[s], 0, kv_bytes);
    }
    if (!ds4_gpu_end_commands()) ok = false;
    if (ok) g->snap_pos = g->pos;
    return ok;
}

static bool ikg_restore(ik_gpu *g) {
    if (!g->snap_pos) return false;
    bool ok = ds4_gpu_begin_commands() != 0;
    for (int s = 0; s < IK_NSLOT && ok; s++) {
        const bool glob = ik_layer_is_global(ik_slot_layer(s));
        const uint64_t kv_bytes = (uint64_t)g->rows[s] * IK_HK * IK_D * 2, conv_bytes = (uint64_t)IK_CONVC * 3 * 4;
        ok = ds4_gpu_tensor_copy(g->conv[s], 0, g->snap_conv[s], 0, conv_bytes) != 0;
        if (!ok || glob) continue;
        ok = ds4_gpu_tensor_copy(g->k[s], 0, g->snap_k[s], 0, kv_bytes) && ds4_gpu_tensor_copy(g->v[s], 0, g->snap_v[s], 0, kv_bytes);
    }
    if (!ds4_gpu_end_commands()) ok = false;
    if (ok) g->pos = g->snap_pos;
    return ok;
}

static void ikg_reset(ik_gpu *g) {
    g->pos = 0;
    g->snap_pos = 0;   /* the replay from zero rewrites the global rows the kept copy relies on: it is no longer restorable */
    for (int s = 0; s < IK_NSLOT; s++) ds4_gpu_tensor_fill_f32(g->conv[s], 0.0f, (uint64_t)IK_CONVC * 3);
}

/* Diagnostic: DS4_INKLING2_FT_PICKS (see the oracle) forces the router decisions of this run, taken from a file. */
static bool ikg_set_forced(ik_gpu *g, const int32_t *picks, uint32_t ntok) {
    ds4_gpu_tensor_free(g->forced);
    g->forced = ikg_alloc((uint64_t)IK_NSLOT * ntok * IK_TOPK * 4, "forced picks", false);
    if (!g->forced || !ds4_gpu_tensor_write(g->forced, 0, picks, (uint64_t)IK_NSLOT * ntok * IK_TOPK * 4)) return false;
    g->forced_ntok = ntok;
    return true;
}

/* DS4_INKLING2_PROFILE=1: CPU time spent encoding vs time then waiting for the GPU, per call, printed every 50 decode calls
 * and every prefill call.  If "wait" is small the CPU encode is the limit, if "encode" is small the GPU is. */
static double ikg_now_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6; }
static void ikg_profile_note(int n, double enc_ms, double wait_ms) {
    static int on = -1, cnt = 0;
    static double se = 0, sw = 0;
    if (on < 0) { const char *e = getenv("DS4_INKLING2_PROFILE"); on = e && e[0] && e[0] != '0'; }
    if (!on) return;
    if (n > 1) { fprintf(stderr, "ik-profile: chunk %d tokens: encode %.1f ms, then wait %.1f ms\n", n, enc_ms, wait_ms); return; }
    se += enc_ms; sw += wait_ms;
    if (++cnt == 50) { fprintf(stderr, "ik-profile: 50 decode calls: encode+commit %.2f ms, final wait %.2f ms per token (total %.2f)\n", se / 50, sw / 50, (se + sw) / 50); cnt = 0; se = sw = 0; }
}

typedef struct {
    ik_gpu *g;
    int n;
    uint32_t p0;
    int blocks[IK_MAX_STATES], nb;   /* stream indices of the current block starts */
    int visit;
    const char *err;
    ik_gpu *const *gs;   /* batched decode: gs[t] is the session of token t (state buffers); g is then the shared scratch */
    int cur_slot;
} ikg_run;

#define IKG_CHECK(r, call) do { if (!(r)->err && !(call)) (r)->err = "a Metal dispatch failed (see the message above)"; } while (0)

static ds4_gpu_ik_ref ikg_stream(const ikg_run *r, int idx) {
    return IKG_REFO(r->g->streams, (uint64_t)idx * (uint64_t)r->n * IK_E * 4);
}

/* out = cur + (-tanh(forget) * cur + tanh(gain) * msg), gates from the rms-normalised pair (cur, msg). */
static void ikg_control(ikg_run *r, const ik_layer *L, int j, ds4_gpu_ik_ref out, ds4_gpu_ik_ref cur, ds4_gpu_ik_ref msg) {
    ik_gpu *g = r->g;
    const uint32_t n = (uint32_t)r->n;
    IKG_CHECK(r, ds4_gpu_ik_ctl_prep(IKG_REF(g->ctl_g), cur, msg, n));
    IKG_CHECK(r, ds4_gpu_ik_mm(IKG_REF(g->ctl_d), ikg_w(g, L->ctl_down[j]), 2 * IK_E, 64, IKG_REF(g->ctl_g), n));
    IKG_CHECK(r, ds4_gpu_ik_mm(IKG_REF(g->ctl_gain), ikg_w(g, L->ctl_gain[j]), 64, IK_E, IKG_REF(g->ctl_d), n));
    IKG_CHECK(r, ds4_gpu_ik_mm(IKG_REF(g->ctl_forget), ikg_w(g, L->ctl_forget[j]), 64, IK_E, IKG_REF(g->ctl_d), n));
    IKG_CHECK(r, ds4_gpu_ik_ctl_apply(out, cur, msg, IKG_REF(g->ctl_gain), IKG_REF(g->ctl_forget), n * IK_E));
}

/* conv over one chunk; at n == 1 (fast mode) one dispatch that also updates the tail and adds `addend` (may be IKG_NONE). */
static ds4_gpu_ik_ref ikg_refadd(ds4_gpu_ik_ref a, uint64_t bytes) { if (a.t) a.off += bytes; return a; }

static void ikg_conv(ikg_run *r, ds4_gpu_ik_ref out, ds4_gpu_ik_ref x, ds4_gpu_ik_ref tail, ds4_gpu_ik_w w, ds4_gpu_ik_ref dyn, ds4_gpu_ik_ref addend,
                     uint32_t C, uint32_t hd, float scale) {
    const uint32_t n = (uint32_t)r->n;
    if (r->gs) {   /* batched decode: token t is the next step of session t, with its own conv tail */
        for (uint32_t t = 0; t < n; t++) {
            const ds4_gpu_ik_ref tl = IKG_REFO(r->gs[t]->conv[r->cur_slot], tail.off), o = ikg_refadd(out, (uint64_t)t * C * 4),
                                 xx = ikg_refadd(x, (uint64_t)t * C * 4), dd = ikg_refadd(dyn, (uint64_t)t * IK_KC * IK_HK * 4), aa = ikg_refadd(addend, (uint64_t)t * C * 4);
            if (!ds4_inkling2_exact_mode()) { IKG_CHECK(r, ds4_gpu_ik_conv1(o, xx, tl, w, dd, aa, C, hd, scale)); continue; }
            IKG_CHECK(r, ds4_gpu_ik_conv(o, xx, tl, w, dd, C, hd, scale, 1));
            if (aa.t) IKG_CHECK(r, ds4_gpu_ik_add(o, aa, o, C));
        }
        return;
    }
    if (ds4_gpu_ik_fast1(n)) { IKG_CHECK(r, ds4_gpu_ik_conv1(out, x, tail, w, dyn, addend, C, hd, scale)); return; }
    IKG_CHECK(r, ds4_gpu_ik_conv(out, x, tail, w, dyn, C, hd, scale, n));
    if (addend.t) IKG_CHECK(r, ds4_gpu_ik_add(out, addend, out, n * C));
}

static void ikg_dump_visit(ikg_run *r, int slot, int y_idx) {
    ik_gpu *g = r->g;
    if (!g->dump || r->err) return;
    ds4_gpu_end_commands();
    const size_t NE = (size_t)r->n * IK_E;
    float *y = xmalloc(NE * sizeof(float));
    int32_t *pk = xmalloc((size_t)r->n * IK_TOPK * sizeof(int32_t));
    ds4_gpu_tensor_read(g->streams, (uint64_t)y_idx * NE * 4, y, NE * 4);
    ds4_gpu_tensor_read(g->picks, (uint64_t)slot * r->n * IK_TOPK * 4, pk, (uint64_t)r->n * IK_TOPK * 4);
    for (int t = 0; t < r->n; t++) {
        fprintf(g->dump, "ids visit=%d slot=%d tok=%d pos=%u", r->visit, slot, t, r->p0 + (uint32_t)t);
        for (int k = 0; k < IK_TOPK; k++) fprintf(g->dump, "%c%d", k ? ',' : ' ', pk[t * IK_TOPK + k]);
        fputc('\n', g->dump);
    }
    for (int t = 0; t < r->n; t++)
        fprintf(g->dump, "lout visit=%d slot=%d pos=%u %.9g\n", r->visit, slot, r->p0 + (uint32_t)t, ik_sum(y + (size_t)t * IK_E, IK_E));
    free(y); free(pk);
    ds4_gpu_begin_commands();
}

/* One layer visit (slot `slot`, physical layer ip) over the chunk; x_idx is the input stream, msg the loop messages
 * ([4][n][E] in g->msg) or NULL.  Returns the output stream index (visit + 1). */
static int ikg_visit(ikg_run *r, int slot, int ip, int x_idx, bool has_msg) {
    ik_gpu *g = r->g;
    const ik_layer *L = &g->w->layer[ip];
    const uint32_t n = (uint32_t)r->n;
    r->cur_slot = slot;
    const uint64_t NE4 = (uint64_t)n * IK_E * 4;
    const bool global = ik_layer_is_global(ip);
    const int y_idx = r->visit + 1;
    ds4_gpu_ik_ref x = ikg_stream(r, x_idx);
    ds4_gpu_ik_ref in[3] = { x, x, x };

    /* 1. attention inputs: depth mix of the block starts and x (layer 0 has no queries) */
    if (ip > 0) {
        const uint32_t nq = global ? 3 : 1;
        uint32_t idx[IK_MAX_STATES];
        for (int i = 0; i < r->nb; i++) idx[i] = (uint32_t)r->blocks[i];
        IKG_CHECK(r, ds4_gpu_ik_mix(IKG_REF(g->mixed), IKG_REF(g->streams), n * IK_E, idx, (uint32_t)r->nb, x,
                                    ikg_w(g, L->attn_res_query), nq, n));
        for (int j = 0; j < 3; j++) in[j] = IKG_REFO(g->mixed, (global ? (uint64_t)j : 0) * NE4);
    }
    /* 2./3. loop messages and input norms.  A local layer without messages feeds one normalised input to Q, K and V. */
    const int nin = (has_msg || global) ? 3 : 1;
    ds4_gpu_ik_ref ib[3];
    for (int j = 0; j < 3; j++) ib[j] = IKG_REF(g->ibuf[j < nin ? j : 0]);
    int done3 = 2;
    if (ds4_gpu_ik_fast1(n) && nin == 3) {   /* decode: controllers and the three input norms in 3 dispatches instead of 18 */
        ds4_gpu_ik_ref src3[3] = { in[0], in[1], in[2] };
        if (has_msg) {
            const ds4_gpu_ik_w wd[3] = { ikg_w(g, L->ctl_down[0]), ikg_w(g, L->ctl_down[1]), ikg_w(g, L->ctl_down[2]) };
            const ds4_gpu_ik_w wg_[3] = { ikg_w(g, L->ctl_gain[0]), ikg_w(g, L->ctl_gain[1]), ikg_w(g, L->ctl_gain[2]) };
            const ds4_gpu_ik_w wf_[3] = { ikg_w(g, L->ctl_forget[0]), ikg_w(g, L->ctl_forget[1]), ikg_w(g, L->ctl_forget[2]) };
            done3 = ds4_gpu_ik_ctl_multi(3, wd, wg_, wf_, in[0], global ? IK_E : 0, IKG_REFO(g->msg, 0), IK_E, IKG_REF(g->ctl_d), ib);
            if (done3 == 0) IKG_CHECK(r, 0);
            for (int j = 0; j < 3; j++) src3[j] = ib[j];
        }
        if (done3 != 2 || !has_msg) {
            const ds4_gpu_ik_w wn[3] = { ikg_w(g, global ? L->inp_norm[0] : L->attn_norm), ikg_w(g, global ? L->inp_norm[1] : L->attn_norm),
                                         ikg_w(g, global ? L->inp_norm[2] : L->attn_norm) };
            done3 = ds4_gpu_ik_rms3(3, src3, wn, ib);
            if (done3 == 0) IKG_CHECK(r, 0);
        }
    }
    if (done3 == 2) {
        for (int j = 0; j < nin; j++) {
            ds4_gpu_ik_ref src = in[j];
            if (has_msg) {
                ikg_control(r, L, j, ib[j], in[j], IKG_REFO(g->msg, (uint64_t)j * NE4));
                src = ib[j];
            }
            IKG_CHECK(r, ds4_gpu_ik_rms(ib[j], src, ikg_w(g, global ? L->inp_norm[j] : L->attn_norm), IK_E, n, IKG_NONE));
        }
    }
    /* 4. projections */
    int multi = 2;
    if (ds4_gpu_ik_fast1(n)) {   /* decode: the six Q8_0 projections in one dispatch */
        const ds4_gpu_ik_w mw[6] = { ikg_w(g, L->q), ikg_w(g, L->k), ikg_w(g, L->v), ikg_w(g, L->rel), ikg_w(g, L->sink), ikg_w(g, L->gate) };
        const uint32_t mrows[6] = { IK_H * IK_D, IK_HK * IK_D, IK_HK * IK_D, IK_R * IK_H, IK_H, IK_H * IK_D };
        const ds4_gpu_ik_ref mx[3] = { ib[0], ib[1], ib[2] };
        const uint32_t msel[6] = { 0, 1, 2, 0, 0, 0 };
        const ds4_gpu_ik_ref my[6] = { IKG_REF(g->Q), IKG_REF(g->Kp), IKG_REF(g->Vp), IKG_REF(g->Rr), IKG_REF(g->sink), IKG_REF(g->gate) };
        multi = ds4_gpu_ik_mm_multi(6, mw, mrows, IK_E, mx, msel, my);
        if (multi == 0) IKG_CHECK(r, 0);
    }
    if (multi == 2) {
        IKG_CHECK(r, ds4_gpu_ik_mm(IKG_REF(g->Q), ikg_w(g, L->q), IK_E, IK_H * IK_D, ib[0], n));
        IKG_CHECK(r, ds4_gpu_ik_mm(IKG_REF(g->Kp), ikg_w(g, L->k), IK_E, IK_HK * IK_D, ib[1], n));
        IKG_CHECK(r, ds4_gpu_ik_mm(IKG_REF(g->Vp), ikg_w(g, L->v), IK_E, IK_HK * IK_D, ib[2], n));
        IKG_CHECK(r, ds4_gpu_ik_mm(IKG_REF(g->Rr), ikg_w(g, L->rel), IK_E, IK_R * IK_H, ib[0], n));
        IKG_CHECK(r, ds4_gpu_ik_mm(IKG_REF(g->sink), ikg_w(g, L->sink), IK_E, IK_H, ib[0], n));
        IKG_CHECK(r, ds4_gpu_ik_mm(IKG_REF(g->gate), ikg_w(g, L->gate), IK_E, IK_H * IK_D, ib[0], n));
    }
    ds4_gpu_ik_ref dk = IKG_NONE, dv = IKG_NONE;
    if (global) {
        IKG_CHECK(r, ds4_gpu_ik_mm(IKG_REF(g->dk), ikg_w(g, L->dyn_proj[0]), IK_E, IK_KC * IK_HK, ib[1], n));
        IKG_CHECK(r, ds4_gpu_ik_mm(IKG_REF(g->dv), ikg_w(g, L->dyn_proj[1]), IK_E, IK_KC * IK_HK, ib[2], n));
        dk = IKG_REF(g->dk); dv = IKG_REF(g->dv);
    }
    const float cscale = global ? 1.0f / 3.0f : 1.0f / sqrtf((float)IK_KC + 1.0f);
    ikg_conv(r, IKG_REF(g->Kc), IKG_REF(g->Kp), IKG_REFO(g->conv[slot], 0), ikg_w(g, L->k_conv), dk, IKG_NONE, IK_HK * IK_D, IK_D, cscale);
    ikg_conv(r, IKG_REF(g->Vc), IKG_REF(g->Vp), IKG_REFO(g->conv[slot], 1024 * 3 * 4), ikg_w(g, L->v_conv), dv, IKG_NONE, IK_HK * IK_D, IK_D, cscale);
    /* 5. normalisations; tau scales Q and R on global layers after the clip; K/V go to the half cache */
    IKG_CHECK(r, ds4_gpu_ik_qr_prep(IKG_REF(g->Q), IKG_REF(g->Rr), ikg_w(g, L->q_clip), ikg_w(g, L->rel_clip), IKG_REF(g->tau), global, n));
    if (r->gs) {
        for (uint32_t t = 0; t < n; t++) {
            const ik_gpu *gt = r->gs[t];
            IKG_CHECK(r, ds4_gpu_ik_kv_store(IKG_REFO(g->Kc, (uint64_t)t * IK_HK * IK_D * 4), IKG_REFO(g->Vc, (uint64_t)t * IK_HK * IK_D * 4),
                                             IKG_REF(gt->k[slot]), IKG_REF(gt->v[slot]), gt->pos, gt->rows[slot], 1));
        }
    } else
    IKG_CHECK(r, ds4_gpu_ik_kv_store(IKG_REF(g->Kc), IKG_REF(g->Vc), IKG_REF(g->k[slot]), IKG_REF(g->v[slot]), r->p0, g->rows[slot], n));
    /* 6./7. attention (output gate included), projection, post conv */
    if (r->gs) {
        for (uint32_t t = 0; t < n; t++) {
            const ik_gpu *gt = r->gs[t];
            IKG_CHECK(r, ds4_gpu_ik_attn(IKG_REFO(g->O, (uint64_t)t * IK_H * IK_D * 4), IKG_REFO(g->Q, (uint64_t)t * IK_H * IK_D * 4),
                                         IKG_REFO(g->Rr, (uint64_t)t * IK_H * IK_R * 4), IKG_REFO(g->sink, (uint64_t)t * IK_H * 4),
                                         IKG_REFO(g->gate, (uint64_t)t * IK_H * IK_D * 4), ikg_w(g, L->rel_bias),
                                         IKG_REF(gt->k[slot]), IKG_REF(gt->v[slot]), gt->pos, gt->rows[slot], !global, 1,
                                         IKG_REF(g->apart), g->apart_splits));
        }
    } else
    IKG_CHECK(r, ds4_gpu_ik_attn(IKG_REF(g->O), IKG_REF(g->Q), IKG_REF(g->Rr), IKG_REF(g->sink), IKG_REF(g->gate), ikg_w(g, L->rel_bias),
                                 IKG_REF(g->k[slot]), IKG_REF(g->v[slot]), r->p0, g->rows[slot], !global, n,
                                 IKG_REF(g->apart), g->apart_splits));
    {
        int rc = 2;
        if (ds4_gpu_ik_fast1(n)) {
            const ds4_gpu_ik_w mw[1] = { ikg_w(g, L->o) };
            const uint32_t mrows[1] = { IK_E }, msel[1] = { 0 };
            const ds4_gpu_ik_ref mx[3] = { IKG_REF(g->O), IKG_REF(g->O), IKG_REF(g->O) }, my[1] = { IKG_REF(g->ap) };
            rc = ds4_gpu_ik_mm_multi(1, mw, mrows, IK_H * IK_D, mx, msel, my);
            if (rc == 0) IKG_CHECK(r, 0);
        }
        if (rc == 2) IKG_CHECK(r, ds4_gpu_ik_mm(IKG_REF(g->ap), ikg_w(g, L->o), IK_H * IK_D, IK_E, IKG_REF(g->O), n));
    }
    {   /* post-attention conv; except at the first layer of a block the residual x is added (fused at n == 1) */
        const bool block_start = ip % IK_BLOCK == 0;
        ikg_conv(r, IKG_REF(g->a), IKG_REF(g->ap), IKG_REFO(g->conv[slot], 2048 * 3 * 4), ikg_w(g, L->post_conv), IKG_NONE,
                 block_start ? IKG_NONE : x, IK_E, 1, 1.0f / sqrtf((float)IK_KC + 1.0f));
        /* 8. block bookkeeping: at the first layer of each block the stream restarts from the attention output */
        if (block_start) r->blocks[r->nb++] = x_idx;
    }
    /* 9. MoE on a depth mix of the block starts and ffn_inp */
    {
        uint32_t idx[IK_MAX_STATES];
        for (int i = 0; i < r->nb; i++) idx[i] = (uint32_t)r->blocks[i];
        IKG_CHECK(r, ds4_gpu_ik_mix(IKG_REF(g->fm), IKG_REF(g->streams), n * IK_E, idx, (uint32_t)r->nb, IKG_REF(g->a),
                                    ikg_w(g, L->ffn_res_query), 1, n));
    }
    if (has_msg) {
        int rc = 2;
        if (ds4_gpu_ik_fast1(n)) {
            const ds4_gpu_ik_w wd[1] = { ikg_w(g, L->ctl_down[3]) }, wg_[1] = { ikg_w(g, L->ctl_gain[3]) }, wf_[1] = { ikg_w(g, L->ctl_forget[3]) };
            const ds4_gpu_ik_ref fo[1] = { IKG_REF(g->fm) };
            rc = ds4_gpu_ik_ctl_multi(1, wd, wg_, wf_, IKG_REF(g->fm), 0, IKG_REFO(g->msg, 3 * NE4), 0, IKG_REF(g->ctl_d), fo);
            if (rc == 0) IKG_CHECK(r, 0);
        }
        if (rc == 2) ikg_control(r, L, 3, IKG_REF(g->fm), IKG_REF(g->fm), IKG_REFO(g->msg, 3 * NE4));
    }
    IKG_CHECK(r, ds4_gpu_ik_rms(IKG_REF(g->h), IKG_REF(g->fm), ikg_w(g, L->ffn_norm), IK_E, n, IKG_NONE));
    IKG_CHECK(r, ds4_gpu_ik_mm(IKG_REF(g->rlogits), ikg_w(g, L->router), IK_E, IK_NEXP, IKG_REF(g->h), n));
    ds4_gpu_ik_ref picks = IKG_REFO(g->picks, (uint64_t)slot * n * IK_TOPK * 4);
    IKG_CHECK(r, ds4_gpu_ik_router(picks, IKG_REF(g->wts), IKG_REF(g->rlogits), ikg_w(g, L->router_bias), IKG_REF(g->forced),
                                   g->forced ? g->forced_ntok : 0, (uint32_t)slot, r->p0, n));
    if ((n == 1 || r->gs) && ds4_gpu_ik_moe_decode_ok(L->gate_up->type, L->down->type)) {   /* decode (also a batch of sessions: token index on z): eight experts per dispatch, down + combine fused */
        IKG_CHECK(r, ds4_gpu_ik_moe_gu1(IKG_REF(g->ac), ikg_w(g, L->gate_up), L->gate_up_scale ? ikg_w(g, L->gate_up_scale) : ikg_none(),
                                        IKG_REF(g->h), picks, n));
        IKG_CHECK(r, ds4_gpu_ik_moe_dc1(IKG_REF(g->f), ikg_w(g, L->down), IKG_REF(g->ac), picks, IKG_REF(g->wts),
                                        L->down_scale ? ikg_w(g, L->down_scale) : ikg_none(), n));
    } else {
        IKG_CHECK(r, ds4_gpu_ik_moe_lists(IKG_REF(g->lists), IKG_REF(g->counts), picks, n > 8 && !ds4_inkling2_exact_mode() ? IKG_REF(g->tiles) : IKG_NONE, n));
        int rc = ds4_gpu_ik_moe_gu_mma(IKG_REF(g->ac), ikg_w(g, L->gate_up), L->gate_up_scale ? ikg_w(g, L->gate_up_scale) : ikg_none(),
                                       IKG_REF(g->h), IKG_REF(g->lists), IKG_REF(g->counts), IKG_REF(g->tiles), n);
        if (rc == 0) IKG_CHECK(r, 0);
        if (rc == 2) IKG_CHECK(r, ds4_gpu_ik_moe_gu(IKG_REF(g->ac), ikg_w(g, L->gate_up), L->gate_up_scale ? ikg_w(g, L->gate_up_scale) : ikg_none(),
                                                    IKG_REF(g->h), IKG_REF(g->lists), IKG_REF(g->counts), n));
        rc = ds4_gpu_ik_moe_down_mma(IKG_REF(g->yd), ikg_w(g, L->down), IKG_REF(g->ac), IKG_REF(g->lists), IKG_REF(g->counts), IKG_REF(g->tiles), n);
        if (rc == 0) IKG_CHECK(r, 0);
        if (rc == 2) IKG_CHECK(r, ds4_gpu_ik_moe_down(IKG_REF(g->yd), ikg_w(g, L->down), IKG_REF(g->ac), IKG_REF(g->lists), IKG_REF(g->counts), n));
        IKG_CHECK(r, ds4_gpu_ik_moe_combine(IKG_REF(g->f), IKG_REF(g->yd), IKG_REF(g->wts), picks,
                                            L->down_scale ? ikg_w(g, L->down_scale) : ikg_none(), n));

    }
    if (ds4_gpu_ik_fast1(n) || (r->gs && !ds4_inkling2_exact_mode())) {
        ikg_conv(r, ikg_stream(r, y_idx), IKG_REF(g->f), IKG_REFO(g->conv[slot], 4352 * 3 * 4), ikg_w(g, L->ffn_post_conv), IKG_NONE, IKG_REF(g->a),
                 IK_E, 1, 1.0f / sqrtf((float)IK_KC + 1.0f));
    } else {
        ikg_conv(r, IKG_REF(g->fc), IKG_REF(g->f), IKG_REFO(g->conv[slot], 4352 * 3 * 4), ikg_w(g, L->ffn_post_conv), IKG_NONE, IKG_NONE,
                 IK_E, 1, 1.0f / sqrtf((float)IK_KC + 1.0f));
        IKG_CHECK(r, ds4_gpu_ik_add(ikg_stream(r, y_idx), IKG_REF(g->a), IKG_REF(g->fc), n * IK_E));
    }
    if (!r->err && !ds4_gpu_flush_commands()) r->err = "cannot commit the command buffer";
    ikg_dump_visit(r, slot, y_idx);
    r->visit++;
    return y_idx;
}

/* Run n tokens at positions g->pos.. through the 48 visits.  logits receives IK_VOCAB soft-capped values for each
 * of the chunk rows listed in want[]. */
static int ikg_forward_chunk(ik_gpu *g, const int *tokens, int n, const int *want, int nwant, float *logits, const char **err) {
    *err = NULL;
    if (n <= 0 || (uint32_t)n > g->chunk || (uint64_t)g->pos + (uint64_t)n > g->cap) { *err = "chunk exceeds the state capacity"; return 1; }
    if (nwant < 0 || (uint32_t)nwant > g->max_want) { *err = "too many logit rows requested"; return 1; }
    for (int t = 0; t < n; t++)
        if (tokens[t] < 0 || tokens[t] >= IK_VOCAB) { *err = "token id out of range"; return 1; }
    for (int i = 0; i < nwant; i++)
        if (want[i] < 0 || want[i] >= n) { *err = "logit row outside the chunk"; return 1; }
    if (g->forced && g->pos + (uint32_t)n > g->forced_ntok) { *err = "pick file too short"; return 1; }
    ikg_run r = { .g = g, .n = n, .p0 = g->pos };
    const double prof_t0 = ikg_now_ms();
    float *tau = xmalloc((size_t)n * sizeof(float));
    for (int t = 0; t < n; t++) tau[t] = 1.0f + 0.1f * logf(fmaxf(1.0f, (float)(r.p0 + (uint32_t)t + 1) / 8192.0f));
    const bool wrote = ds4_gpu_tensor_write(g->tokens, 0, tokens, (uint64_t)n * 4) && ds4_gpu_tensor_write(g->tau, 0, tau, (uint64_t)n * 4) &&
                       (nwant == 0 || ds4_gpu_tensor_write(g->want, 0, want, (uint64_t)nwant * 4));
    free(tau);
    if (!wrote) { *err = "cannot upload the chunk inputs"; return 1; }
    if (!ds4_gpu_begin_commands()) { *err = "a GPU command batch is already open"; return 1; }
    const uint32_t un = (uint32_t)n;
    const uint64_t NE4 = (uint64_t)un * IK_E * 4;

    IKG_CHECK(&r, ds4_gpu_ik_embed(ikg_stream(&r, 0), ikg_w(g, g->w->embd), ikg_w(g, g->w->embd_norm), IKG_REF(g->tokens), un));
    int x_idx = 0;
    for (int il = 0; il < IK_LOOP_START && !r.err; il++) x_idx = ikg_visit(&r, il, il, x_idx, false);
    const int prelude_x = x_idx, prelude_nb = r.nb;
    int bank[IK_MAX_STATES], nbank = 0, bank_out = -1;
    for (int pass = 0; pass < IK_LOOP_N && !r.err; pass++) {
        if (pass > 0) {
            nbank = r.nb - (prelude_nb + 1);
            memcpy(bank, r.blocks + prelude_nb + 1, (size_t)nbank * sizeof(int));
            bank_out = x_idx;
            x_idx = prelude_x;
            r.nb = prelude_nb;
        }
        for (int ip = IK_LOOP_START; ip < IK_LOOP_END && !r.err; ip++) {
            const int slot = pass == 0 ? ip : IK_NLAYER + (pass - 1) * (IK_LOOP_END - IK_LOOP_START) + (ip - IK_LOOP_START);
            if (pass > 0) {   /* messages: mix of the previous pass's bank and its last output, one query each for Q, K, V and FFN */
                uint32_t idx[IK_MAX_STATES];
                for (int i = 0; i < nbank; i++) idx[i] = (uint32_t)bank[i];
                IKG_CHECK(&r, ds4_gpu_ik_mix(IKG_REF(g->msg), IKG_REF(g->streams), un * IK_E, idx, (uint32_t)nbank, ikg_stream(&r, bank_out),
                                             ikg_w(g, g->w->layer[ip].msg_query), 4, un));
            }
            x_idx = ikg_visit(&r, slot, ip, x_idx, pass > 0);
        }
    }
    for (int il = IK_LOOP_END; il < IK_NLAYER && !r.err; il++) x_idx = ikg_visit(&r, il, il, x_idx, false);

    /* output: depth mix over every block start and the final stream, norm of the wanted rows, LM head, soft cap */
    if (nwant > 0 && !r.err) {
        uint32_t idx[IK_MAX_STATES];
        for (int i = 0; i < r.nb; i++) idx[i] = (uint32_t)r.blocks[i];
        IKG_CHECK(&r, ds4_gpu_ik_mix(IKG_REF(g->mo), IKG_REF(g->streams), un * IK_E, idx, (uint32_t)r.nb, ikg_stream(&r, x_idx),
                                     ikg_w(g, g->w->out_res_query), 1, un));
        IKG_CHECK(&r, ds4_gpu_ik_rms(IKG_REF(g->hh), IKG_REF(g->mo), ikg_w(g, g->w->out_norm), IK_E, (uint32_t)nwant, IKG_REF(g->want)));
        IKG_CHECK(&r, ds4_gpu_ik_mm(IKG_REF(g->logits), ikg_w(g, g->w->out), IK_E, IK_VOCAB, IKG_REF(g->hh), (uint32_t)nwant));
        IKG_CHECK(&r, ds4_gpu_ik_softcap(IKG_REF(g->logits), (uint32_t)nwant * IK_VOCAB));
    }
    (void)NE4;
    const double prof_t1 = ikg_now_ms();   /* CPU has encoded and committed everything but the last buffer */
    const int ok = ds4_gpu_end_commands();
    ikg_profile_note(n, prof_t1 - prof_t0, ikg_now_ms() - prof_t1);
    if (!r.err && !ok) r.err = "the GPU command buffer failed";
    if (r.err) { *err = r.err; return 1; }
    if (nwant > 0 && !ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)nwant * IK_VOCAB * 4)) { *err = "cannot read the logits back"; return 1; }
    if (nwant > 0 && logits) {
        const uint64_t nl = (uint64_t)nwant * IK_VOCAB;
        /* test hook: any value poisons every call; a value starting with / poisons only while that file exists (lets a test
         * fail one prefill of a running server) */
        const char *poison = getenv("DS4_INKLING2_TEST_POISON_LOGITS");
        if (poison && (poison[0] != '/' || access(poison, F_OK) == 0)) { const uint32_t nan_bits = 0x7fc00000u; memcpy(&logits[nl / 2], &nan_bits, sizeof(nan_bits)); }   /* the check below must catch it */
        /* Bit test (the host build uses fast math, isfinite() may be folded away).  The soft cap bounds real logits, so any NaN or
         * infinity means the GPU computation went wrong: report it instead of sampling token 0 from garbage. */
        uint32_t bad = 0;
        for (uint64_t i = 0; i < nl; i++) {
            uint32_t u;
            memcpy(&u, &logits[i], sizeof(u));
            bad |= (u & 0x7f800000u) == 0x7f800000u;
        }
        if (bad) { *err = "the logits contain NaN or infinity (GPU computation failed)"; return 1; }
    }
    if (g->pick_log) {   /* keep every router decision for DS4_INKLING2_GPU_PICKS_OUT */
        int32_t *tmp = xmalloc((size_t)n * IK_TOPK * sizeof(int32_t));
        for (int s = 0; s < IK_NSLOT; s++) {
            ds4_gpu_tensor_read(g->picks, (uint64_t)s * (uint64_t)n * IK_TOPK * 4, tmp, (uint64_t)n * IK_TOPK * 4);
            memcpy(g->pick_log + ((size_t)s * g->cap + g->pos) * IK_TOPK, tmp, (size_t)n * IK_TOPK * sizeof(int32_t));
        }
        free(tmp);
    }
    g->pos += (uint32_t)n;
    return 0;
}

/* Batched decode: one token for each of ns sessions (2..8) in one pass.  The dense matvecs, the router and the expert kernels
 * see an ns-token batch (weights are read once); K/V store, attention and the conv tails run per session on that session's own buffers.
 * `sc` is a state-less scratch (ikg_new_ex with scratch_only, chunk >= ns, max_want >= ns, cap >= the longest session context).  Each
 * session's logits row is written to logits[t] (IK_VOCAB floats).  On an error (return 1) every session in the batch has half-written
 * state: the caller must invalidate them.  Non-finite logits are not such an error: the command buffer succeeded, every session's
 * state is complete and only the session with the bad row is affected, so it is flagged in bad[t] (its pos does not advance and the
 * caller invalidates it) and the others go on. */
static int ikg_forward_multi(ik_gpu *sc, ik_gpu *const *gs, const int *tokens, int ns, float *const *logits, int *bad, const char **err) {
    *err = NULL;
    if (ns < 2 || (uint32_t)ns > sc->chunk || (uint32_t)ns > sc->max_want) { *err = "batch size outside the scratch"; return 1; }
    for (int t = 0; t < ns; t++) {
        if (tokens[t] < 0 || tokens[t] >= IK_VOCAB) { *err = "token id out of range"; return 1; }
        if ((uint64_t)gs[t]->pos + 1 > gs[t]->cap) { *err = "context is full"; return 1; }
        if (gs[t]->apart_splits > sc->apart_splits && gs[t]->cap > sc->cap) { *err = "scratch is smaller than a session context"; return 1; }
    }
    ikg_run r = { .g = sc, .n = ns, .p0 = 0, .gs = gs };
    float tau[8];
    int want[8];
    for (int t = 0; t < ns; t++) { tau[t] = 1.0f + 0.1f * logf(fmaxf(1.0f, (float)(gs[t]->pos + 1) / 8192.0f)); want[t] = t; }
    if (!ds4_gpu_tensor_write(sc->tokens, 0, tokens, (uint64_t)ns * 4) || !ds4_gpu_tensor_write(sc->tau, 0, tau, (uint64_t)ns * 4) ||
        !ds4_gpu_tensor_write(sc->want, 0, want, (uint64_t)ns * 4)) { *err = "cannot upload the batch inputs"; return 1; }
    if (!ds4_gpu_begin_commands()) { *err = "a GPU command batch is already open"; return 1; }
    const uint32_t un = (uint32_t)ns;
    IKG_CHECK(&r, ds4_gpu_ik_embed(ikg_stream(&r, 0), ikg_w(sc, sc->w->embd), ikg_w(sc, sc->w->embd_norm), IKG_REF(sc->tokens), un));
    int x_idx = 0;
    for (int il = 0; il < IK_LOOP_START && !r.err; il++) x_idx = ikg_visit(&r, il, il, x_idx, false);
    const int prelude_x = x_idx, prelude_nb = r.nb;
    int bank[IK_MAX_STATES], nbank = 0, bank_out = -1;
    for (int pass = 0; pass < IK_LOOP_N && !r.err; pass++) {
        if (pass > 0) {
            nbank = r.nb - (prelude_nb + 1);
            memcpy(bank, r.blocks + prelude_nb + 1, (size_t)nbank * sizeof(int));
            bank_out = x_idx;
            x_idx = prelude_x;
            r.nb = prelude_nb;
        }
        for (int ip = IK_LOOP_START; ip < IK_LOOP_END && !r.err; ip++) {
            const int slot = pass == 0 ? ip : IK_NLAYER + (pass - 1) * (IK_LOOP_END - IK_LOOP_START) + (ip - IK_LOOP_START);
            if (pass > 0) {
                uint32_t idx[IK_MAX_STATES];
                for (int i = 0; i < nbank; i++) idx[i] = (uint32_t)bank[i];
                IKG_CHECK(&r, ds4_gpu_ik_mix(IKG_REF(sc->msg), IKG_REF(sc->streams), un * IK_E, idx, (uint32_t)nbank, ikg_stream(&r, bank_out),
                                             ikg_w(sc, sc->w->layer[ip].msg_query), 4, un));
            }
            x_idx = ikg_visit(&r, slot, ip, x_idx, pass > 0);
        }
    }
    for (int il = IK_LOOP_END; il < IK_NLAYER && !r.err; il++) x_idx = ikg_visit(&r, il, il, x_idx, false);
    if (!r.err) {
        uint32_t idx[IK_MAX_STATES];
        for (int i = 0; i < r.nb; i++) idx[i] = (uint32_t)r.blocks[i];
        IKG_CHECK(&r, ds4_gpu_ik_mix(IKG_REF(sc->mo), IKG_REF(sc->streams), un * IK_E, idx, (uint32_t)r.nb, ikg_stream(&r, x_idx),
                                     ikg_w(sc, sc->w->out_res_query), 1, un));
        IKG_CHECK(&r, ds4_gpu_ik_rms(IKG_REF(sc->hh), IKG_REF(sc->mo), ikg_w(sc, sc->w->out_norm), IK_E, un, IKG_REF(sc->want)));
        IKG_CHECK(&r, ds4_gpu_ik_mm(IKG_REF(sc->logits), ikg_w(sc, sc->w->out), IK_E, IK_VOCAB, IKG_REF(sc->hh), un));
        IKG_CHECK(&r, ds4_gpu_ik_softcap(IKG_REF(sc->logits), un * IK_VOCAB));
    }
    const int ok = ds4_gpu_end_commands();
    if (!r.err && !ok) r.err = "the GPU command buffer failed";
    if (r.err) { *err = r.err; return 1; }
    const char *poison_row = getenv("DS4_INKLING2_TEST_POISON_ROW");   /* test hook: row of the batch to poison */
    for (int t = 0; t < ns; t++) {
        if (!ds4_gpu_tensor_read(sc->logits, (uint64_t)t * IK_VOCAB * 4, logits[t], (uint64_t)IK_VOCAB * 4)) { *err = "cannot read the logits back"; return 1; }
        if (poison_row && poison_row[0] && atoi(poison_row) == t) { const uint32_t nan_bits = 0x7fc00000u; memcpy(&logits[t][IK_VOCAB / 2], &nan_bits, sizeof(nan_bits)); }
        uint32_t nonfinite = 0;
        for (uint64_t i = 0; i < IK_VOCAB; i++) { uint32_t u; memcpy(&u, &logits[t][i], sizeof(u)); nonfinite |= (u & 0x7f800000u) == 0x7f800000u; }
        bad[t] = nonfinite != 0;
    }
    for (int t = 0; t < ns; t++) if (!bad[t]) gs[t]->pos += 1;
    return 0;
}

/* Chunk size of sessions (tokens per forward call): DS4_INKLING2_CHUNK, default 512 (see docs/INKLING2.md). */
static uint32_t ik_default_chunk(void) {
    const char *env = getenv("DS4_INKLING2_CHUNK");
    const long v = env && env[0] ? atol(env) : 0;
    return v > 0 && v <= 4096 ? (uint32_t)v : 512u;
}

/* What `ds4_context_memory_estimate` reports for this family: the K/V + conv state and the per-chunk scratch. */
static ds4_context_memory ik_context_memory(uint32_t ctx, uint32_t chunk) {
    if (chunk == 0) chunk = ik_default_chunk();
    if (chunk > ctx) chunk = ctx;
    uint64_t st = 0, sc = 0;
    ikg_estimate(ctx, chunk, 1, &st, &sc);
    ds4_context_memory m = {0};
    m.prefill_cap = chunk;
    m.raw_cap = ctx;
    m.raw_bytes = st;
    m.scratch_bytes = sc;
    m.total_bytes = st + sc;
    return m;
}

/* Features this family does not have yet answer with an error and a clear message (never a crash). */
static int iks_unsupported(char *err, size_t errlen, const char *what) {
    if (err && errlen) snprintf(err, errlen, "inkling2: %s are not supported", what);
    return 1;
}


/* The honest memory estimate for a server with `nsess` sessions of `ctx` tokens, and a refusal to start when it does not
 * fit the budget (DS4_INKLING2_BUDGET_GB, default half of the machine's RAM): a state that does not fit would be swapped or wired by the
 * GPU driver, which slows every other process on the machine.  Per session: eight full-context slots (K and V, half, 2 KiB per row each)
 * + 40 window slots (1024 + chunk rows) + the snapshot copy of those + conv tails + the per-chunk scratch (the scratch is per session
 * here: sessions do not share it).  Returns 0 when it fits. */
static int ik_check_memory_budget(uint64_t model_bytes, uint32_t ctx, int nsess) {
    uint32_t chunk = ik_default_chunk();
    if (chunk > ctx) chunk = ctx;
    if (nsess < 1) nsess = 1;
    uint64_t st = 0, sc = 0;
    ikg_estimate(ctx, chunk, 1, &st, &sc);
    const double GiB = 1073741824.0;
    const double full = 8.0 * (double)ikg_rows(ctx) * 2.0 * IK_HK * IK_D * 2.0;   /* the 8 full-context slots, K and V */
    uint64_t ram = 0;
#ifdef __APPLE__
    size_t len = sizeof(ram);
    if (sysctlbyname("hw.memsize", &ram, &len, NULL, 0) != 0) ram = 0;
#endif
    double budget = ram ? (double)ram / 2.0 : 0.0;
    const char *env = getenv("DS4_INKLING2_BUDGET_GB");
    const char *src = "half of RAM";
    if (env && env[0] && atof(env) > 0.0) { budget = atof(env) * GiB; src = "DS4_INKLING2_BUDGET_GB"; }
    const double per = (double)(st + sc), total = (double)model_bytes + per * nsess;
    fprintf(stderr, "ds4: inkling2: memory estimate for ctx %u: %.2f GiB per session (%.2f GiB are the 8 full-context slots = 8 x ctx x K and V; "
            "window slots, snapshot copy and conv tails %.2f GiB; scratch %.2f GiB)\n", ctx, per / GiB, full / GiB, ((double)st - full) / GiB, (double)sc / GiB);
    fprintf(stderr, "ds4: inkling2: 1 session %.2f GiB, 8 sessions %.2f GiB, model file (mapped) %.2f GiB; this start: %d session%s, %.2f GiB in total, budget %.2f GiB (%s)\n",
            per / GiB, per * 8 / GiB, (double)model_bytes / GiB, nsess, nsess == 1 ? "" : "s", total / GiB, budget / GiB, src);
    /* what is free right now, with a safety margin: other jobs on the machine keep growing and the model's pages are read in on first
     * use.  Free memory must stay above DS4_INKLING2_FREE_FLOOR_PCT percent of RAM (default 40) after this server's total. */
    double free_pct = -1.0, floor_pct = 40.0;
#ifdef __APPLE__
    { int lvl = 0; size_t ll = sizeof(lvl); if (sysctlbyname("kern.memorystatus_level", &lvl, &ll, NULL, 0) == 0 && lvl >= 0 && lvl <= 100) free_pct = lvl; }
#endif
    const char *fenv = getenv("DS4_INKLING2_FREE_FLOOR_PCT");
    if (fenv && fenv[0] && atof(fenv) >= 0.0) floor_pct = atof(fenv);
    if (free_pct >= 0.0)
        fprintf(stderr, "ds4: inkling2: free memory now %.0f%% (%.1f GiB of %.1f GiB); after this start it would be %.1f GiB, floor %.0f%% = %.1f GiB (DS4_INKLING2_FREE_FLOOR_PCT)\n",
                free_pct, free_pct / 100.0 * (double)ram / GiB, (double)ram / GiB, (free_pct / 100.0 * (double)ram - total) / GiB, floor_pct, floor_pct / 100.0 * (double)ram / GiB);
    const int verdict = ds4_inkling2_memory_verdict(total, (double)ram, free_pct, floor_pct, budget);
    if (verdict == 1) {
        fprintf(stderr, "ds4: inkling2: refusing to start: %.2f GiB do not fit the %.2f GiB budget (lower --ctx or --batched-session, or raise DS4_INKLING2_BUDGET_GB)\n",
                total / GiB, budget / GiB);
        return 1;
    }
    if (verdict == 2) {
        fprintf(stderr, "ds4: inkling2: refusing to start: %.2f GiB would leave %.1f GiB free, under the %.0f%% floor (%.1f GiB). Lower --ctx or --batched-session, free memory, or lower DS4_INKLING2_FREE_FLOOR_PCT\n",
                total / GiB, (free_pct / 100.0 * (double)ram - total) / GiB, floor_pct, floor_pct / 100.0 * (double)ram / GiB);
        return 1;
    }
    return 0;
}
