#include "pulsar_engine_internal.h"

void payload_set_err(char *err, size_t errlen, const char *msg) {
    if (errlen != 0) snprintf(err, errlen, "%s", msg);
}



static void payload_put_u32(uint8_t out[4], uint32_t v) {
    out[0] = (uint8_t)(v);
    out[1] = (uint8_t)(v >> 8);
    out[2] = (uint8_t)(v >> 16);
    out[3] = (uint8_t)(v >> 24);
}



static uint32_t payload_get_u32(const uint8_t in[4]) {
    return (uint32_t)in[0] |
           ((uint32_t)in[1] << 8) |
           ((uint32_t)in[2] << 16) |
           ((uint32_t)in[3] << 24);
}



static int payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes, char *err, size_t errlen) {
    const uint8_t *p = (const uint8_t *)ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fwrite(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to write session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    return 0;
}



static PULSAR_MAYBE_UNUSED int payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes, uint64_t *remaining, char *err, size_t errlen) {
    if (remaining && *remaining < bytes) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    const uint64_t original = bytes;
    uint8_t *p = (uint8_t *)ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fread(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to read session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    if (remaining) *remaining -= original;
    return 0;
}



static PULSAR_MAYBE_UNUSED int payload_write_u32(FILE *fp, uint32_t v, char *err, size_t errlen) {
    uint8_t b[4];
    payload_put_u32(b, v);
    return payload_write_bytes(fp, b, sizeof(b), err, errlen);
}



static PULSAR_MAYBE_UNUSED int payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining, char *err, size_t errlen) {
    uint8_t b[4];
    if (remaining && *remaining < sizeof(b)) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    if (fread(b, 1, sizeof(b), fp) != sizeof(b)) {
        payload_set_err(err, errlen, "failed to read session payload");
        return 1;
    }
    if (remaining) *remaining -= sizeof(b);
    *v = payload_get_u32(b);
    return 0;
}



static int payload_copy_file_bytes(FILE *src, FILE *dst, uint64_t bytes, char *err, size_t errlen) {
    uint8_t *buf = (uint8_t *)xmalloc(PULSAR_SESSION_IO_CHUNK);
    int rc = 0;
    while (bytes != 0) {
        const size_t n = bytes > PULSAR_SESSION_IO_CHUNK ? PULSAR_SESSION_IO_CHUNK : (size_t)bytes;
        if (fread(buf, 1, n, src) != n) {
            payload_set_err(err, errlen, "failed to read staged session payload");
            rc = 1;
            break;
        }
        if (fwrite(buf, 1, n, dst) != n) {
            payload_set_err(err, errlen, "failed to write staged session payload");
            rc = 1;
            break;
        }
        bytes -= n;
    }
    free(buf);
    return rc;
}



static PULSAR_MAYBE_UNUSED uint64_t layer_attn_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * PULSAR_N_HEAD_DIM * coff * ratio * sizeof(float);
}



static PULSAR_MAYBE_UNUSED uint64_t layer_index_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * PULSAR_N_INDEXER_HEAD_DIM * coff * ratio * sizeof(float);
}



/* Only the last logical sliding-window rows are needed from the raw cache.
 * The physical GPU tensor is a ring sized for ubatches, but after restore
 * the next suffix chunk will write its own raw rows before any attention read.
 * Compressed rows are different: sparse attention can select any row from the
 * prefix, so those are persisted up to their live row counts. */
static uint32_t session_raw_live_rows(const pulsar_gpu_graph *g, uint32_t checkpoint_len) {
    uint32_t rows = g->raw_window ? g->raw_window : PULSAR_N_SWA;
    if (rows > g->raw_cap) rows = g->raw_cap;
    if (rows > checkpoint_len) rows = checkpoint_len;
    return rows;
}



/* Return the exact engine-owned payload size, excluding the server's KVC file
 * header and observability text.  This is deliberately based on live row counts
 * rather than capacities so the disk cache scales with saved tokens, not with
 * the maximum context size used to allocate the graph. */
static uint64_t session_payload_live_tensor_bytes(const pulsar_gpu_graph *g, uint32_t checkpoint_len) {
    uint64_t bytes = 0;
    const uint32_t raw_live = session_raw_live_rows(g, checkpoint_len);
    /* Session files always store comp rows as f32 (packed caches dequant to
     * the f32 shadow on save), so payload sizing is format-independent. */
    const uint64_t comp_row = (uint64_t)PULSAR_N_HEAD_DIM * sizeof(float);
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        bytes += (uint64_t)raw_live * PULSAR_N_HEAD_DIM * sizeof(float);
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        bytes += (uint64_t)g->layer_n_comp[il] * comp_row;
        bytes += layer_attn_state_bytes(ratio);
        bytes += layer_attn_state_bytes(ratio);
        if (ratio == 4) {
            bytes += (uint64_t)g->layer_n_index_comp[il] * PULSAR_N_INDEXER_HEAD_DIM * sizeof(float);
            bytes += layer_index_state_bytes(ratio);
            bytes += layer_index_state_bytes(ratio);
        }
    }
    return bytes;
}



/* Accelerator tensors are copied through a fixed-size CPU buffer.  We do not mmap the
 * cache file and we do not allocate a second graph-sized blob just to serialize
 * it; both would be poor fits for this very large model. */
static int payload_write_tensor_span(FILE *fp, const pulsar_gpu_tensor *tensor,
                                     uint64_t offset, uint64_t bytes,
                                     uint8_t *buf, size_t cap, char *err, size_t errlen) {
    if (!tensor || offset > pulsar_gpu_tensor_bytes(tensor) ||
        bytes > pulsar_gpu_tensor_bytes(tensor) - offset)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (pulsar_gpu_tensor_read(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to read accelerator session tensor");
            return 1;
        }
        if (payload_write_bytes(fp, buf, n, err, errlen) != 0) return 1;
        done += n;
    }
    return 0;
}



static int payload_read_tensor_span(FILE *fp, pulsar_gpu_tensor *tensor,
                                    uint64_t offset, uint64_t bytes,
                                    uint8_t *buf, size_t cap, uint64_t *remaining,
                                    char *err, size_t errlen) {
    if (!tensor || offset > pulsar_gpu_tensor_bytes(tensor) ||
        bytes > pulsar_gpu_tensor_bytes(tensor) - offset)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (payload_read_bytes(fp, buf, n, remaining, err, errlen) != 0) return 1;
        if (pulsar_gpu_tensor_write(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to restore accelerator session tensor");
            return 1;
        }
        done += n;
    }
    return 0;
}



/* Session files always store the indexer comp cache as f32 rows.  Under
 * PULSAR_IDX_FP4 the persistent cache is MXKV-FP4-packed, so save dequantizes
 * into the f32 staging first and load repacks from it.  The repack is
 * value-exact for all realistic rows (QAT-roundtripped fp4 values on
 * power-of-two block scales survive re-encoding); the one exception is a
 * 32-block whose amax sits below the mxkv encode floor (1e-20), which would
 * flush to zero — unreachable for RMS-normed indexer rows. */
static int payload_write_index_comp(FILE *fp, pulsar_gpu_graph *g, uint32_t il,
                                    uint32_t n_rows, uint8_t *buf, size_t cap,
                                    char *err, size_t errlen) {
    const uint64_t bytes = (uint64_t)n_rows * PULSAR_N_INDEXER_HEAD_DIM * sizeof(float);
    pulsar_gpu_tensor *src = g->layer_index_comp_cache[il];
    if (gpu_graph_idx_fp4_enabled() && n_rows != 0) {
        if (!g->idx_comp_stage ||
            pulsar_gpu_mxkv_dequant_tensor(g->layer_index_comp_cache[il],
                                        g->idx_comp_stage,
                                        PULSAR_ENGINE_MXKV_FMT_FP4,
                                        n_rows,
                                        PULSAR_N_INDEXER_HEAD_DIM) == 0) {
            payload_set_err(err, errlen, "failed to dequantize fp4 indexer cache for session save");
            return 1;
        }
        src = g->idx_comp_stage;
    }
    return payload_write_tensor_span(fp, src, 0, bytes, buf, cap, err, errlen);
}

static int payload_read_index_comp(FILE *fp, pulsar_gpu_graph *g, uint32_t il,
                                   uint32_t n_rows, uint8_t *buf, size_t cap,
                                   uint64_t *remaining, char *err, size_t errlen) {
    const uint64_t bytes = (uint64_t)n_rows * PULSAR_N_INDEXER_HEAD_DIM * sizeof(float);
    if (!gpu_graph_idx_fp4_enabled() || n_rows == 0) {
        return payload_read_tensor_span(fp, g->layer_index_comp_cache[il], 0, bytes,
                                        buf, cap, remaining, err, errlen);
    }
    if (!g->idx_comp_stage) {
        payload_set_err(err, errlen, "fp4 indexer cache staging missing on session load");
        return 1;
    }
    int rc = payload_read_tensor_span(fp, g->idx_comp_stage, 0, bytes,
                                      buf, cap, remaining, err, errlen);
    if (rc != 0) return rc;
    if (pulsar_gpu_mxkv_pack_tensor(g->idx_comp_stage,
                                 g->layer_index_comp_cache[il],
                                 PULSAR_ENGINE_MXKV_FMT_FP4,
                                 n_rows,
                                 PULSAR_N_INDEXER_HEAD_DIM) == 0) {
        payload_set_err(err, errlen, "failed to repack fp4 indexer cache on session load");
        return 1;
    }
    return 0;
}

/* Attn comp cache spans under PULSAR_ATTN_PACK: session files always store f32
 * rows, so save dequantizes the packed cache into the f32 shadow first and
 * load repacks from it.  Save is bit-exact by construction (packed rows decode
 * to exactly the fp8-roundtripped values the f32 pipeline holds).  Load
 * re-quantizes already-roundtripped rows; that is value-preserving except for
 * blocks whose amax sits exactly on a scale boundary, where the recomputed
 * block scale can shift one step and re-round small values (the same
 * non-idempotency that forced quantize_fp8=false in the pack prefill paths).
 * Sub-1e-3-relative on isolated dims; acceptable for session restore, but do
 * NOT rely on save/load being bit-exact under pack. */
static int payload_write_attn_comp_pack(FILE *fp, pulsar_gpu_graph *g, uint32_t il,
                                        uint32_t n_rows, uint8_t *buf, size_t cap,
                                        char *err, size_t errlen) {
    const uint64_t bytes = (uint64_t)n_rows * PULSAR_N_HEAD_DIM * sizeof(float);
    pulsar_gpu_tensor *src = g->attn_comp_dequant;
    if (n_rows != 0) {
        if (!src ||
            pulsar_gpu_attn_pack_dequant_tensor(g->layer_attn_comp_cache[il],
                                             src, n_rows,
                                             PULSAR_N_HEAD_DIM, PULSAR_N_ROT) == 0) {
            payload_set_err(err, errlen, "failed to dequantize packed attn comp cache for session save");
            return 1;
        }
    }
    if (n_rows == 0) return 0;
    return payload_write_tensor_span(fp, src, 0, bytes, buf, cap, err, errlen);
}

static int payload_read_attn_comp_pack(FILE *fp, pulsar_gpu_graph *g, uint32_t il,
                                       uint32_t n_rows, uint8_t *buf, size_t cap,
                                       uint64_t *remaining, char *err, size_t errlen) {
    if (n_rows == 0) return 0;
    const uint64_t bytes = (uint64_t)n_rows * PULSAR_N_HEAD_DIM * sizeof(float);
    if (!g->attn_comp_dequant) {
        payload_set_err(err, errlen, "packed attn comp cache staging missing on session load");
        return 1;
    }
    int rc = payload_read_tensor_span(fp, g->attn_comp_dequant, 0, bytes,
                                      buf, cap, remaining, err, errlen);
    if (rc != 0) return rc;
    /* Exact-scale repack: file rows are already roundtripped, and the
     * fast-math quantize bucket is not bit-idempotent at scale boundaries. */
    if (pulsar_gpu_attn_pack_repack_tensor(g->attn_comp_dequant,
                                        g->layer_attn_comp_cache[il],
                                        0, n_rows,
                                        PULSAR_N_HEAD_DIM, PULSAR_N_ROT) == 0) {
        payload_set_err(err, errlen, "failed to repack attn comp cache on session load");
        return 1;
    }
    return 0;
}



static PULSAR_MAYBE_UNUSED int payload_write_tensor_span_f16_as_f32(FILE *fp, const pulsar_gpu_tensor *tensor,
                                                                 uint64_t offset_f16, uint64_t count,
                                                                 uint8_t *buf, size_t cap, char *err, size_t errlen) {
    if (!tensor ||
        count > (UINT64_MAX / sizeof(uint16_t)) ||
        count > (UINT64_MAX / sizeof(float)) ||
        offset_f16 > pulsar_gpu_tensor_bytes(tensor) ||
        count * sizeof(uint16_t) > pulsar_gpu_tensor_bytes(tensor) - offset_f16)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the F16 payload");
        return 1;
    }

    size_t cap_elems = cap / (sizeof(uint16_t) + sizeof(float));
    cap_elems &= ~(size_t)1u;
    if (cap_elems == 0) {
        payload_set_err(err, errlen, "session tensor conversion buffer is too small");
        return 1;
    }
    uint16_t *h = (uint16_t *)buf;
    float *f = (float *)(void *)(buf + cap_elems * sizeof(uint16_t));

    uint64_t done = 0;
    while (done < count) {
        const size_t n = count - done > (uint64_t)cap_elems
            ? cap_elems
            : (size_t)(count - done);
        if (pulsar_gpu_tensor_read(tensor, offset_f16 + done * sizeof(uint16_t),
                                h, n * sizeof(uint16_t)) == 0) {
            payload_set_err(err, errlen, "failed to read GPU F16 session tensor");
            return 1;
        }
        for (size_t i = 0; i < n; i++) f[i] = f16_to_f32(h[i]);
        if (payload_write_bytes(fp, f, (uint64_t)n * sizeof(float), err, errlen) != 0) return 1;
        done += n;
    }
    return 0;
}



static PULSAR_MAYBE_UNUSED int payload_read_tensor_span_f32_as_f16(FILE *fp, pulsar_gpu_tensor *tensor,
                                                                uint64_t offset_f16, uint64_t count,
                                                                uint8_t *buf, size_t cap, uint64_t *remaining,
                                                                char *err, size_t errlen) {
    if (!tensor ||
        count > (UINT64_MAX / sizeof(uint16_t)) ||
        count > (UINT64_MAX / sizeof(float)) ||
        offset_f16 > pulsar_gpu_tensor_bytes(tensor) ||
        count * sizeof(uint16_t) > pulsar_gpu_tensor_bytes(tensor) - offset_f16)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the F16 payload");
        return 1;
    }

    size_t cap_elems = cap / (sizeof(uint16_t) + sizeof(float));
    cap_elems &= ~(size_t)1u;
    if (cap_elems == 0) {
        payload_set_err(err, errlen, "session tensor conversion buffer is too small");
        return 1;
    }
    uint16_t *h = (uint16_t *)buf;
    float *f = (float *)(void *)(buf + cap_elems * sizeof(uint16_t));

    uint64_t done = 0;
    while (done < count) {
        const size_t n = count - done > (uint64_t)cap_elems
            ? cap_elems
            : (size_t)(count - done);
        if (payload_read_bytes(fp, f, (uint64_t)n * sizeof(float), remaining, err, errlen) != 0) return 1;
        for (size_t i = 0; i < n; i++) h[i] = f32_to_f16(f[i]);
        if (pulsar_gpu_tensor_write(tensor, offset_f16 + done * sizeof(uint16_t),
                                 h, n * sizeof(uint16_t)) == 0) {
            payload_set_err(err, errlen, "failed to restore GPU F16 session tensor");
            return 1;
        }
        done += n;
    }
    return 0;
}

/* Raw-ring row spans: session files always store f32 rows.  Under PULSAR_RAW_F16
 * the ring holds __half containers, so save expands f16->f32 and load packs
 * f32->f16 — bit-exact both ways because the values are f16-rounded at store
 * time in both modes. */
static int payload_write_raw_row(FILE *fp, pulsar_gpu_graph *g, uint32_t il, uint32_t phys,
                                 uint8_t *buf, size_t cap, char *err, size_t errlen) {
    if (gpu_graph_raw_f16_enabled()) {
        return payload_write_tensor_span_f16_as_f32(fp, g->layer_raw_cache[il],
                (uint64_t)phys * PULSAR_N_HEAD_DIM * sizeof(uint16_t),
                (uint64_t)PULSAR_N_HEAD_DIM, buf, cap, err, errlen);
    }
    return payload_write_tensor_span(fp, g->layer_raw_cache[il],
            (uint64_t)phys * PULSAR_N_HEAD_DIM * sizeof(float),
            (uint64_t)PULSAR_N_HEAD_DIM * sizeof(float), buf, cap, err, errlen);
}

static int payload_read_raw_row(FILE *fp, pulsar_gpu_graph *g, uint32_t il, uint32_t phys,
                                uint8_t *buf, size_t cap, uint64_t *remaining,
                                char *err, size_t errlen) {
    if (gpu_graph_raw_f16_enabled()) {
        return payload_read_tensor_span_f32_as_f16(fp, g->layer_raw_cache[il],
                (uint64_t)phys * PULSAR_N_HEAD_DIM * sizeof(uint16_t),
                (uint64_t)PULSAR_N_HEAD_DIM, buf, cap, remaining, err, errlen);
    }
    return payload_read_tensor_span(fp, g->layer_raw_cache[il],
            (uint64_t)phys * PULSAR_N_HEAD_DIM * sizeof(float),
            (uint64_t)PULSAR_N_HEAD_DIM * sizeof(float), buf, cap, remaining, err, errlen);
}

uint64_t pulsar_session::payload_bytes() {
    auto *s = this;
    if (!s || !s->checkpoint_valid) return 0;
    const pulsar_gpu_graph *g = &s->graph;
    uint64_t bytes = (uint64_t)PULSAR_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    bytes += (uint64_t)s->checkpoint.len * sizeof(uint32_t);
    bytes += (uint64_t)PULSAR_N_VOCAB * sizeof(float);
    bytes += (uint64_t)PULSAR_N_LAYER * sizeof(uint32_t);
    bytes += (uint64_t)PULSAR_N_LAYER * sizeof(uint32_t);
    bytes += session_payload_live_tensor_bytes(g, (uint32_t)s->checkpoint.len);
    return bytes;
}



int pulsar_session_write_staged_payload(const pulsar_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen) {
    if (!payload || !payload->path || !fp) {
        payload_set_err(err, errlen, "invalid staged session payload");
        return 1;
    }
    FILE *src = fopen(payload->path, "rb");
    if (!src) {
        payload_set_err(err, errlen, "failed to open staged session payload");
        return 1;
    }
    int rc = payload_copy_file_bytes(src, fp, payload->bytes, err, errlen);
    if (fclose(src) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close staged session payload");
        return 1;
    }
    return rc;
}



void pulsar_session_payload_file_free(pulsar_session_payload_file *payload) {
    if (!payload) return;
    if (payload->path) {
        unlink(payload->path);
        free(payload->path);
    }
    memset(payload, 0, sizeof(*payload));
}



int pulsar_session::stage_payload(pulsar_session_payload_file *out,
                              char *err, size_t errlen) {
    auto *s = this;
    if (!out) {
        payload_set_err(err, errlen, "invalid session payload staging request");
        return 1;
    }
    memset(out, 0, sizeof(*out));
    if (!s || !s->checkpoint_valid) {
        payload_set_err(err, errlen, "session has no valid checkpoint to stage");
        return 1;
    }

    char tmpl[] = "/tmp/ds4-session-payload.XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        payload_set_err(err, errlen, "failed to create staged session payload");
        return 1;
    }
    FILE *fp = fdopen(fd, "wb");
    if (!fp) {
        int saved = errno;
        close(fd);
        unlink(tmpl);
        if (errlen) snprintf(err, errlen, "failed to open staged session payload: %s",
                             strerror(saved));
        return 1;
    }

    int rc = s->save_payload(fp, err, errlen);
    if (rc == 0 && fflush(fp) != 0) {
        payload_set_err(err, errlen, "failed to flush staged session payload");
        rc = 1;
    }
    off_t pos = -1;
    if (rc == 0) {
        pos = ftello(fp);
        if (pos < 0) {
            payload_set_err(err, errlen, "failed to measure staged session payload");
            rc = 1;
        }
    }
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close staged session payload");
        rc = 1;
    }
    if (rc != 0) {
        unlink(tmpl);
        return 1;
    }
    out->path = pulsar_strdup(tmpl);
    out->bytes = (uint64_t)pos;
    return 0;
}



int pulsar_session::save_payload(FILE *fp, char *err, size_t errlen) {
    auto *s = this;
    if (!s || !fp || !s->checkpoint_valid) {
        payload_set_err(err, errlen, "session has no valid checkpoint to save");
        return 1;
    }
    if (pulsar_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator before snapshot");
        return 1;
    }

    pulsar_gpu_graph *g = &s->graph;
    const uint32_t raw_live = session_raw_live_rows(g, (uint32_t)s->checkpoint.len);
    /* Header fields:
     *   0 magic, 1 version, 2 ctx, 3 prefill chunk, 4 raw cap,
     *   5 raw window, 6 compressed cap, 7 token count,
     *   8 layers, 9 raw head dim, 10 indexer head dim, 11 vocab,
     *   12 live raw rows serialized below.
     */
    uint32_t header[PULSAR_SESSION_PAYLOAD_U32_FIELDS] = {
        PULSAR_SESSION_PAYLOAD_MAGIC,
        PULSAR_SESSION_PAYLOAD_VERSION,
        (uint32_t)s->ctx_size,
        s->prefill_cap,
        g->raw_cap,
        g->raw_window,
        g->comp_cap,
        (uint32_t)s->checkpoint.len,
        PULSAR_N_LAYER,
        PULSAR_N_HEAD_DIM,
        PULSAR_N_INDEXER_HEAD_DIM,
        PULSAR_N_VOCAB,
        raw_live,
    };
    for (uint32_t i = 0; i < PULSAR_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
    }
    for (int i = 0; i < s->checkpoint.len; i++) {
        if (payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen) != 0) return 1;
    }
    if (payload_write_bytes(fp, s->logits, (uint64_t)PULSAR_N_VOCAB * sizeof(float), err, errlen) != 0) return 1;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (payload_write_u32(fp, g->layer_n_comp[il], err, errlen) != 0) return 1;
    }
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (payload_write_u32(fp, g->layer_n_index_comp[il], err, errlen) != 0) return 1;
    }

    uint8_t *buf = (uint8_t *)xmalloc(PULSAR_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = 0; rc == 0 && il < PULSAR_N_LAYER; il++) {
        /* Write the raw ring in logical position order.  The file does not care
         * where the rows happened to live physically in the source graph. */
        const uint32_t raw_first = (uint32_t)s->checkpoint.len - raw_live;
        for (uint32_t r = 0; rc == 0 && r < raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_write_raw_row(fp, g, il, phys,
                                       buf, PULSAR_SESSION_IO_CHUNK, err, errlen);
        }
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        /* Compressed rows are append-only from row zero, so the live prefix is
         * contiguous.  The two compressor state tensors hold the partial window
         * that will become the next compressed row. */
        if (gpu_graph_attn_pack_enabled()) {
            rc = payload_write_attn_comp_pack(fp, g, il,
                                              g->layer_n_comp[il],
                                              buf,
                                              PULSAR_SESSION_IO_CHUNK,
                                              err,
                                              errlen);
        } else {
            rc = payload_write_tensor_span(fp,
                                           g->layer_attn_comp_cache[il],
                                           0,
                                           (uint64_t)g->layer_n_comp[il] * PULSAR_N_HEAD_DIM * sizeof(float),
                                           buf,
                                           PULSAR_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
        }
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_kv[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    PULSAR_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_score[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    PULSAR_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_write_index_comp(fp, g, il,
                                          g->layer_n_index_comp[il],
                                          buf,
                                          PULSAR_SESSION_IO_CHUNK,
                                          err,
                                          errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_kv[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        PULSAR_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_score[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        PULSAR_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
        }
    }
    free(buf);
    return rc;
}



int pulsar_session::load_payload(FILE *fp, uint64_t payload_bytes, char *err, size_t errlen) {
    auto *s = this;
    if (!s || !fp) {
        payload_set_err(err, errlen, "invalid session payload load");
        return 1;
    }
    /* drop speculative lookahead up front, not just on success: a restore
     * that fails midway may already have overwritten GPU state the carry
     * and pendings were conditioned on */
    s->spec.spec_carry_valid = false;
    s->spec.dspark_n_pending = 0;
    spec_quench_reset(s);
    uint64_t remaining = payload_bytes;
    uint32_t h[PULSAR_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < PULSAR_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    if (h[0] != PULSAR_SESSION_PAYLOAD_MAGIC || h[1] != PULSAR_SESSION_PAYLOAD_VERSION) {
        payload_set_err(err, errlen, "unsupported session payload version");
        return 1;
    }
    pulsar_gpu_graph *g = &s->graph;
    const uint32_t saved_ctx = h[2];
    const uint32_t saved_prefill_cap = h[3];
    const uint32_t saved_raw_cap = h[4];
    const uint32_t saved_raw_window = h[5];
    const uint32_t saved_comp_cap = h[6];
    const uint32_t saved_tokens = h[7];
    const uint32_t saved_raw_live = h[12];
    if (saved_ctx > (uint32_t)s->ctx_size || saved_tokens >= (uint32_t)s->ctx_size) {
        payload_set_err(err, errlen, "KV checkpoint does not fit current context");
        return 1;
    }
    if (h[8] != PULSAR_N_LAYER || h[9] != PULSAR_N_HEAD_DIM ||
        h[10] != PULSAR_N_INDEXER_HEAD_DIM || h[11] != PULSAR_N_VOCAB)
    {
        payload_set_err(err, errlen, "KV checkpoint was written for a different Pulsar layout");
        return 1;
    }
    /* prefill_cap is scratch scheduling capacity, not durable KV layout.
     * Old checkpoints remain valid as long as the raw KV window matches. */
    (void)saved_prefill_cap;
    if (saved_raw_window != g->raw_window) {
        payload_set_err(err, errlen, "KV checkpoint graph chunk layout does not match current runtime");
        return 1;
    }
    /* The raw rows in the file are logical rows.  We can restore them into any
     * current ring with enough capacity, but the saved live count must be exactly
     * the last window implied by the saved token count. */
    const uint32_t expected_raw_live = saved_tokens < saved_raw_window ? saved_tokens : saved_raw_window;
    if (saved_raw_cap == 0 || saved_raw_live != expected_raw_live ||
        saved_raw_live > saved_raw_cap || saved_raw_live > g->raw_cap)
    {
        payload_set_err(err, errlen, "KV checkpoint raw ring layout does not match current context");
        return 1;
    }
    if (saved_comp_cap > g->comp_cap) {
        payload_set_err(err, errlen, "KV checkpoint compressed cache is larger than current context");
        return 1;
    }

    token_vec new_checkpoint = {0};
    for (uint32_t i = 0; i < saved_tokens; i++) {
        uint32_t tok = 0;
        if (payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        token_vec_push(&new_checkpoint, (int)tok);
    }
    if (payload_read_bytes(fp, s->logits, (uint64_t)PULSAR_N_VOCAB * sizeof(float),
                           &remaining, err, errlen) != 0)
    {
        token_vec_free(&new_checkpoint);
        return 1;
    }
    uint32_t n_comp[PULSAR_MAX_LAYER];
    uint32_t n_index_comp[PULSAR_MAX_LAYER];
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (payload_read_u32(fp, &n_comp[il], &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        if (n_comp[il] > saved_comp_cap || n_comp[il] > g->layer_comp_cap[il]) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has invalid compressed row count");
            return 1;
        }
    }
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (payload_read_u32(fp, &n_index_comp[il], &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        if (n_index_comp[il] > saved_comp_cap || n_index_comp[il] > g->layer_comp_cap[il]) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has invalid indexer row count");
            return 1;
        }
    }

    if (pulsar_gpu_synchronize() == 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "failed to synchronize accelerator before KV restore");
        return 1;
    }
    s->checkpoint_valid = false;

    uint8_t *buf = (uint8_t *)xmalloc(PULSAR_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = 0; rc == 0 && il < PULSAR_N_LAYER; il++) {
        /* Rebuild the physical raw ring expected by the current graph.  This is
         * why the file stores rows in logical order instead of dumping bytes from
         * the old ring layout. */
        const uint32_t raw_first = saved_tokens - saved_raw_live;
        for (uint32_t r = 0; rc == 0 && r < saved_raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_read_raw_row(fp, g, il, phys,
                                      buf, PULSAR_SESSION_IO_CHUNK, &remaining, err, errlen);
        }
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        if (gpu_graph_attn_pack_enabled()) {
            rc = payload_read_attn_comp_pack(fp, g, il,
                                             n_comp[il],
                                             buf,
                                             PULSAR_SESSION_IO_CHUNK,
                                             &remaining,
                                             err,
                                             errlen);
        } else {
            rc = payload_read_tensor_span(fp,
                                          g->layer_attn_comp_cache[il],
                                          0,
                                          (uint64_t)n_comp[il] * PULSAR_N_HEAD_DIM * sizeof(float),
                                          buf,
                                          PULSAR_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
        }
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                    g->layer_attn_state_kv[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   PULSAR_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   g->layer_attn_state_score[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   PULSAR_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_read_index_comp(fp, g, il,
                                         n_index_comp[il],
                                         buf,
                                         PULSAR_SESSION_IO_CHUNK,
                                         &remaining,
                                         err,
                                         errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_kv[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       PULSAR_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_score[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       PULSAR_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
        }
    }
    free(buf);
    if (rc != 0) {
        token_vec_free(&new_checkpoint);
        return 1;
    }
    if (remaining != 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "KV checkpoint has trailing payload bytes");
        return 1;
    }
    if (pulsar_gpu_synchronize() == 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "failed to synchronize accelerator after KV restore");
        return 1;
    }

    token_vec_free(&s->checkpoint);
    s->checkpoint = new_checkpoint;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        g->layer_n_comp[il] = n_comp[il];
        g->layer_n_index_comp[il] = n_index_comp[il];
    }
    s->checkpoint_valid = true;
    /* a restored state invalidates any in-flight speculative lookahead: the
     * carry token, pre-drafted pendings, AND the drafter's context-KV ring
     * were all conditioned on the replaced state. Leaving the ring makes the
     * next drafts (and therefore the verify batch shapes) depend on whatever
     * ran before the restore — the source of run-to-run tie flips. */
    s->spec.spec_carry_valid = false;
    s->spec.dspark_n_pending = 0;
    spec_quench_reset(s);
    for (int li = 0; li < 3; li++) g->dspark_n_raw[li] = 0;
    g->dspark_prompt_n = 0;
    return 0;
}



int pulsar_session::save_snapshot(pulsar_session_snapshot *snap, char *err, size_t errlen) {
    auto *s = this;
    if (!s || !snap) {
        payload_set_err(err, errlen, "invalid session snapshot save");
        return 1;
    }
    const uint64_t bytes = s->payload_bytes();
    if (bytes == 0) {
        payload_set_err(err, errlen, "session has no valid checkpoint to snapshot");
        return 1;
    }
    if (bytes > (uint64_t)SIZE_MAX) {
        payload_set_err(err, errlen, "session snapshot is too large for this platform");
        return 1;
    }
    if (snap->cap < bytes) {
        uint8_t *p = (uint8_t *)realloc(snap->ptr, (size_t)bytes);
        if (!p) {
            payload_set_err(err, errlen, "out of memory while allocating session snapshot");
            return 1;
        }
        snap->ptr = p;
        snap->cap = bytes;
    }

    FILE *fp = fmemopen(snap->ptr, (size_t)bytes, "wb");
    if (!fp) {
        payload_set_err(err, errlen, "failed to open memory stream for session snapshot");
        return 1;
    }
    const int rc = s->save_payload(fp, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to finalize memory session snapshot");
        return 1;
    }
    if (rc != 0) return 1;
    snap->len = bytes;
    return 0;
}



int pulsar_session::load_snapshot(const pulsar_session_snapshot *snap, char *err, size_t errlen) {
    auto *s = this;
    if (!s || !snap || !snap->ptr || snap->len == 0) {
        payload_set_err(err, errlen, "invalid session snapshot load");
        return 1;
    }
    if (snap->len > (uint64_t)SIZE_MAX) {
        payload_set_err(err, errlen, "session snapshot is too large for this platform");
        return 1;
    }

    FILE *fp = fmemopen((void *)snap->ptr, (size_t)snap->len, "rb");
    if (!fp) {
        payload_set_err(err, errlen, "failed to open memory stream for session snapshot restore");
        return 1;
    }
    const int rc = s->load_payload(fp, snap->len, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close memory session snapshot");
        return 1;
    }
    return rc;
}



void pulsar_session_snapshot_free(pulsar_session_snapshot *snap) {
    if (!snap) return;
    free(snap->ptr);
    memset(snap, 0, sizeof(*snap));
}
