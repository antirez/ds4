/* Inkling 2 Mini Metal wrappers (kernels in metal/inkling2.metal, graph in ds4_inkling2_gpu.inc).
 * Included at the end of ds4_metal.m: it uses the file's static command-buffer and model-view helpers.
 * Every wrapper encodes one dispatch into the active command batch (ds4_gpu_begin_commands) and
 * returns 1 on success.  Weights are bound by file offset from the mapped model; activations are
 * ds4_gpu_tensor + byte offset pairs. */

#include "ds4_inkling2_env.h"

typedef struct {
    id<MTLBuffer> buf;
    NSUInteger    off;
} ikm_bind;

static ds4_gpu_tensor *g_ikm_dummy;

static int ikm_ref(ikm_bind *b, ds4_gpu_ik_ref r, uint64_t min_bytes, const char *what) {
    if (!r.t) {
        if (!g_ikm_dummy) g_ikm_dummy = ds4_gpu_tensor_alloc(4096);
        b->buf = ds4_gpu_tensor_buffer(g_ikm_dummy);
        b->off = ds4_gpu_tensor_offset(g_ikm_dummy);
        return b->buf != nil;
    }
    if (!ds4_gpu_tensor_buffer(r.t) || r.off + min_bytes > ds4_gpu_tensor_bytes(r.t)) {
        fprintf(stderr, "ds4: inkling2 %s buffer is missing or undersized (%" PRIu64 " + %" PRIu64 " > %" PRIu64 ")\n",
                what, r.off, min_bytes, r.t ? ds4_gpu_tensor_bytes(r.t) : 0);
        return 0;
    }
    b->buf = ds4_gpu_tensor_buffer(r.t);
    b->off = ds4_gpu_tensor_offset(r.t) + (NSUInteger)r.off;
    return 1;
}

static int ikm_weight(ikm_bind *b, ds4_gpu_ik_w w, const char *what) {
    if (!w.map) {   /* optional weight absent */
        ds4_gpu_ik_ref none = {0};
        return ikm_ref(b, none, 0, what);
    }
    uint64_t inner = 0;
    if (w.off > w.size || w.bytes > w.size - w.off) {
        fprintf(stderr, "ds4: inkling2 %s range is outside the mapped model\n", what);
        return 0;
    }
    b->buf = ds4_gpu_wrap_model_range(w.map, w.size, w.off, w.bytes, &inner);
    b->off = (NSUInteger)inner;
    return b->buf != nil;
}

static const char *ikm_type_name(uint32_t type) {
    switch (type) {
    case 0u:  return "f32";
    case 1u:  return "f16";
    case 8u:  return "q8_0";
    case 30u: return "bf16";
    case 39u: return "mxfp4";
    case 40u: return "nvfp4";
    default:  return NULL;
    }
}

/* Kernel names are string literals so the pipeline cache can key on the pointer. */
static const char *ikm_kname(const char *family, uint32_t type) {
    static const char *const mm[] = { "kernel_ik_mm_f32", "kernel_ik_mm_f16", "kernel_ik_mm_q8_0", "kernel_ik_mm_bf16", "kernel_ik_mm_mxfp4", "kernel_ik_mm_nvfp4" };
    static const char *const gu[] = { NULL, "kernel_ik_moe_gu_f16", "kernel_ik_moe_gu_q8_0", "kernel_ik_moe_gu_bf16", "kernel_ik_moe_gu_mxfp4", "kernel_ik_moe_gu_nvfp4" };
    static const char *const dn[] = { NULL, "kernel_ik_moe_down_f16", "kernel_ik_moe_down_q8_0", "kernel_ik_moe_down_bf16", "kernel_ik_moe_down_mxfp4", "kernel_ik_moe_down_nvfp4" };
    int i = -1;
    switch (type) { case 0u: i = 0; break; case 1u: i = 1; break; case 8u: i = 2; break; case 30u: i = 3; break; case 39u: i = 4; break; case 40u: i = 5; break; default: break; }
    if (i < 0) return NULL;
    return family[0] == 'm' ? mm[i] : family[0] == 'g' ? gu[i] : dn[i];
}

/* DS4_INKLING2_EXACT=1: only the fixed-order kernels (results do not depend on the number of tokens in a dispatch). */
static int ikm_exact(void) {
    static int v = -1;
    if (v < 0) v = ds4_inkling2_exact_mode();
    return v;
}

static int ikm_fast1_enabled(void) { return !ikm_exact(); }
#define IKM_EXTENT_ 1024u
#define IKM_SPLIT_MIN_KEYS   128u    /* decode attention uses the split kernels above this many keys */
#define IKM_SPLIT_KEYS       1024u   /* keys per split, global slots (about the memory-bandwidth limit: 512 and 2048 measure the same or worse) */
#define IKM_SPLIT_KEYS_LOCAL 128u    /* keys per split, sliding-window slots (at most 1024 keys: 8 splits; one 1024-key walk was latency bound) */
static int ikm_attn_scalar(void) {   /* DS4_INKLING2_ATTN_SCALAR=1 forces the old attention kernel (A/B runs); 0 / empty = off */
    static int v = -1;
    if (v < 0) v = ds4_inkling2_env_flag("DS4_INKLING2_ATTN_SCALAR");
    return v;
}

static uint32_t ikm_row_bytes(uint32_t type, uint32_t in_dim) {
    switch (type) {
    case 0u:  return in_dim * 4u;
    case 1u:
    case 30u: return in_dim * 2u;
    case 8u:  return (in_dim / 32u) * 34u;
    case 39u: return (in_dim / 32u) * 17u;
    case 40u: return (in_dim / 64u) * 36u;
    default:  return 0;
    }
}

/* The Inkling 2 kernels live in their own library (preamble + metal/inkling2.metal), compiled the first time this family
 * needs them: the other model families neither read nor compile the file, so a missing or broken metal/inkling2.metal cannot
 * stop them from starting.  DS4_METAL_INKLING2_SOURCE overrides the path like the other sources. */
static id<MTLLibrary> ikm_library(void) {
    if (g_ik_library || g_ik_library_tried || !g_device) return g_ik_library;
    g_ik_library_tried = 1;
    NSString *path = nil;
    NSString *src = ds4_gpu_read_source_file(@"DS4_METAL_INKLING2_SOURCE", @"metal/inkling2.metal", &path);
    if (!src) return nil;
    NSString *full = [[NSString stringWithUTF8String:ds4_gpu_source] stringByAppendingFormat:@"\n// appended %@\n%@\n", path, src];
    NSError *error = nil;
    g_ik_library = [g_device newLibraryWithSource:full options:[MTLCompileOptions new] error:&error];
    if (!g_ik_library)
        fprintf(stderr, "ds4: inkling2 Metal shader compilation failed (%s): %s\n", [path UTF8String], [[error localizedDescription] UTF8String]);
    return g_ik_library;
}

/* 1 when the Inkling 2 kernel library is available (loads and compiles it if needed). */
int ds4_gpu_ik_ready(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    return ikm_library() != nil;
}

static id<MTLComputePipelineState> ikm_get_pipeline(const char *name) {
    NSString *key = [NSString stringWithFormat:@"ik:%s", name];
    id<MTLComputePipelineState> cached = [g_pipeline_cache objectForKey:key];
    if (cached) return cached == (id<MTLComputePipelineState>)[NSNull null] ? nil : cached;
    id<MTLLibrary> lib = ikm_library();
    id<MTLFunction> fn = lib ? [lib newFunctionWithName:[NSString stringWithUTF8String:name]] : nil;
    NSError *error = nil;
    id<MTLComputePipelineState> pso = fn ? [g_device newComputePipelineStateWithFunction:fn error:&error] : nil;
    if (!pso) {
        fprintf(stderr, "ds4: inkling2 Metal pipeline %s unavailable%s%s\n", name, error ? ": " : "", error ? [[error localizedDescription] UTF8String] : "");
        [g_pipeline_cache setObject:(id<MTLComputePipelineState>)[NSNull null] forKey:key];
        return nil;
    }
    [g_pipeline_cache setObject:pso forKey:key];
    return pso;
}

static int ikm_dispatch(const char *name, const void *args, size_t alen, const ikm_bind *b, int nb,
                        MTLSize grid, MTLSize tg) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    @autoreleasepool {
        static struct { const char *name; id<MTLComputePipelineState> pso; } cache[160];
        static int ncache;
        id<MTLComputePipelineState> pipeline = nil;
        for (int i = 0; i < ncache; i++) if (cache[i].name == name) { pipeline = cache[i].pso; break; }
        if (!pipeline) {
            pipeline = ikm_get_pipeline(name);
            if (!pipeline) return 0;
            if (ncache < 160) { cache[ncache].name = name; cache[ncache].pso = pipeline; ncache++; }
        }
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        if (args) [enc setBytes:args length:alen atIndex:0];
        for (int i = 0; i < nb; i++) [enc setBuffer:b[i].buf offset:b[i].off atIndex:(NSUInteger)(i + 1)];
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        ds4_gpu_end_compute_encoder(cb, enc);
        return ds4_gpu_finish_command_buffer(cb, owned, name);
    }
}

#define IKM_CEIL(a, b) (((a) + (b) - 1u) / (b))
#define IKM_E_      2304u
#define IKM_FF_     1152u

static const char *ikm_mvn_name(int kind, int one, uint32_t n) {
    static const char *const names[3][2][7] = {
        {
            { "kernel_ik_mvn_q8_0_t2", "kernel_ik_mvn_q8_0_t3", "kernel_ik_mvn_q8_0_t4", "kernel_ik_mvn_q8_0_t5", "kernel_ik_mvn_q8_0_t6", "kernel_ik_mvn_q8_0_t7", "kernel_ik_mvn_q8_0_t8" },
            { "kernel_ik_mvn1_q8_0_t2", "kernel_ik_mvn1_q8_0_t3", "kernel_ik_mvn1_q8_0_t4", "kernel_ik_mvn1_q8_0_t5", "kernel_ik_mvn1_q8_0_t6", "kernel_ik_mvn1_q8_0_t7", "kernel_ik_mvn1_q8_0_t8" },
        },
        {
            { "kernel_ik_mvn_f32_t2", "kernel_ik_mvn_f32_t3", "kernel_ik_mvn_f32_t4", "kernel_ik_mvn_f32_t5", "kernel_ik_mvn_f32_t6", "kernel_ik_mvn_f32_t7", "kernel_ik_mvn_f32_t8" },
            { "kernel_ik_mvn1_f32_t2", "kernel_ik_mvn1_f32_t3", "kernel_ik_mvn1_f32_t4", "kernel_ik_mvn1_f32_t5", "kernel_ik_mvn1_f32_t6", "kernel_ik_mvn1_f32_t7", "kernel_ik_mvn1_f32_t8" },
        },
        {
            { "kernel_ik_mvn_bf16_t2", "kernel_ik_mvn_bf16_t3", "kernel_ik_mvn_bf16_t4", "kernel_ik_mvn_bf16_t5", "kernel_ik_mvn_bf16_t6", "kernel_ik_mvn_bf16_t7", "kernel_ik_mvn_bf16_t8" },
            { "kernel_ik_mvn1_bf16_t2", "kernel_ik_mvn1_bf16_t3", "kernel_ik_mvn1_bf16_t4", "kernel_ik_mvn1_bf16_t5", "kernel_ik_mvn1_bf16_t6", "kernel_ik_mvn1_bf16_t7", "kernel_ik_mvn1_bf16_t8" },
        },
    };
    return n >= 2 && n <= 8 ? names[kind][one ? 1 : 0][n - 2] : NULL;
}

int ds4_gpu_ik_mm(ds4_gpu_ik_ref y, ds4_gpu_ik_w w, uint32_t in_dim, uint32_t out_rows, ds4_gpu_ik_ref x, uint32_t n) {
    const char *name = ikm_kname("mm", w.type);
    if (!name || !n || !out_rows || !in_dim) return 0;
    const uint32_t rb = ikm_row_bytes(w.type, in_dim);
    struct { uint32_t in_dim, out_rows, n_tokens, row_bytes; } args = { in_dim, out_rows, n, rb };
    ikm_bind b[3];
    if (!ikm_weight(&b[0], w, "mm weight") || !ikm_ref(&b[1], x, (uint64_t)n * in_dim * 4, "mm x") ||
        !ikm_ref(&b[2], y, (uint64_t)n * out_rows * 4, "mm y")) return 0;
    if (n <= 8 && !ikm_exact() && (w.type == 8u || w.type == 0u || w.type == 30u) && (in_dim % 32u) == 0) {
        /* decode: matvec kernels (lanes split each row, two rows per simdgroup) */
        const int one = out_rows <= 4096;   /* small outputs: one row per simdgroup for more threadgroups */
        name = w.type == 8u ? (one ? "kernel_ik_mv1_q8_0" : "kernel_ik_mv_q8_0")
             : w.type == 0u ? (one ? "kernel_ik_mv1_f32" : "kernel_ik_mv_f32")
                            : (one ? "kernel_ik_mv1_bf16" : "kernel_ik_mv_bf16");
        /* 2..8 tokens: the kernel with the token count fixed at compile time (kernel_ik_mvn*); one token keeps kernel_ik_mv*.
         * ikm_dispatch caches pipelines by name POINTER, so the names must be string literals. */
        const char *tn = name;
        if (n >= 2 && n <= 8) tn = ikm_mvn_name(w.type == 8u ? 0 : w.type == 0u ? 1 : 2, one, n);
        if (!tn) return 0;
        return ikm_dispatch(tn, &args, sizeof(args), b, 3, MTLSizeMake(IKM_CEIL(out_rows, one ? 4u : 8u), 1, 1), MTLSizeMake(128, 1, 1));
    }
    if (n > 8 && !ikm_exact() && (w.type == 8u || w.type == 0u || w.type == 30u) && (in_dim % 64u) == 0) {
        name = w.type == 8u ? "kernel_ik_mma_q8_0" : w.type == 0u ? "kernel_ik_mma_f32" : "kernel_ik_mma_bf16";
        return ikm_dispatch(name, &args, sizeof(args), b, 3, MTLSizeMake(IKM_CEIL(out_rows, 32u), IKM_CEIL(n, 32u), 1),
                            MTLSizeMake(128, 1, 1));
    }
    return ikm_dispatch(name, &args, sizeof(args), b, 3, MTLSizeMake(IKM_CEIL(out_rows, 8u), IKM_CEIL(n, 8u), 1),
                        MTLSizeMake(128, 1, 1));
}

int ds4_gpu_ik_embed(ds4_gpu_ik_ref x, ds4_gpu_ik_w emb, ds4_gpu_ik_w norm, ds4_gpu_ik_ref tokens, uint32_t n) {
    struct { uint32_t n_tokens, type, row_bytes, pad; } args = { n, emb.type, ikm_row_bytes(emb.type, IKM_E_), 0 };
    if (!ikm_type_name(emb.type) || !args.row_bytes) return 0;
    ikm_bind b[4];
    if (!ikm_weight(&b[0], emb, "embedding") || !ikm_ref(&b[1], tokens, (uint64_t)n * 4, "tokens") ||
        !ikm_weight(&b[2], norm, "embedding norm") || !ikm_ref(&b[3], x, (uint64_t)n * IKM_E_ * 4, "embed out")) return 0;
    return ikm_dispatch("kernel_ik_embed", &args, sizeof(args), b, 4, MTLSizeMake(n, 1, 1), MTLSizeMake(256, 1, 1));
}

int ds4_gpu_ik_mix(ds4_gpu_ik_ref out, ds4_gpu_ik_ref streams, uint32_t stream_stride, const uint32_t *idx, uint32_t n_streams,
                   ds4_gpu_ik_ref cur, ds4_gpu_ik_w queries, uint32_t nq, uint32_t n) {
    if (n_streams > 7 || nq > 4 || !n) return 0;
    struct { uint32_t n_tokens, ns, nq, stream_stride; uint32_t idx[8]; } args = { n, n_streams + 1, nq, stream_stride, {0} };
    for (uint32_t i = 0; i < n_streams; i++) args.idx[i] = idx[i];
    ikm_bind b[4];
    if (!ikm_ref(&b[0], streams, 0, "mix streams") || !ikm_ref(&b[1], cur, (uint64_t)n * IKM_E_ * 4, "mix cur") ||
        !ikm_weight(&b[2], queries, "mix queries") || !ikm_ref(&b[3], out, (uint64_t)nq * n * IKM_E_ * 4, "mix out")) return 0;
    return ikm_dispatch(n == 1 && !ikm_exact() ? "kernel_ik_mix_p" : "kernel_ik_mix", &args, sizeof(args), b, 4, MTLSizeMake(n, 1, 1), MTLSizeMake(256, 1, 1));
}

/* y[r] = x[r] * rsqrt(mean(x^2)+eps) * w; w may be absent; rows (int32) optionally gathers the source rows. */
int ds4_gpu_ik_rms(ds4_gpu_ik_ref y, ds4_gpu_ik_ref x, ds4_gpu_ik_w w, uint32_t dim, uint32_t n_rows, ds4_gpu_ik_ref rows) {
    struct { uint32_t dim, n_rows, has_w, gather; } args = { dim, n_rows, w.map != NULL, rows.t != NULL };
    if (n_rows == 1 && dim == IKM_E_ && w.map && !rows.t && ikm_fast1_enabled()) {   /* decode: single-pass kernel */
        const ds4_gpu_ik_ref xs[1] = { x }, ys[1] = { y };
        const int rc = ds4_gpu_ik_rms3(1, xs, &w, ys);
        if (rc != 2) return rc;
    }
    ikm_bind b[4];
    if (!ikm_ref(&b[0], x, 0, "rms x") || !ikm_weight(&b[1], w, "rms weight") ||
        !ikm_ref(&b[2], y, (uint64_t)n_rows * dim * 4, "rms y") || !ikm_ref(&b[3], rows, 0, "rms rows")) return 0;
    return ikm_dispatch("kernel_ik_rms_rows", &args, sizeof(args), b, 4, MTLSizeMake(n_rows, 1, 1), MTLSizeMake(256, 1, 1));
}

int ds4_gpu_ik_ctl_prep(ds4_gpu_ik_ref g, ds4_gpu_ik_ref cur, ds4_gpu_ik_ref msg, uint32_t n) {
    ikm_bind b[3];
    if (!ikm_ref(&b[0], cur, (uint64_t)n * IKM_E_ * 4, "ctl cur") || !ikm_ref(&b[1], msg, (uint64_t)n * IKM_E_ * 4, "ctl msg") ||
        !ikm_ref(&b[2], g, (uint64_t)n * 2 * IKM_E_ * 4, "ctl g")) return 0;
    return ikm_dispatch("kernel_ik_ctl_prep", &n, sizeof(n), b, 3, MTLSizeMake(n, 1, 1), MTLSizeMake(256, 1, 1));
}

int ds4_gpu_ik_ctl_apply(ds4_gpu_ik_ref out, ds4_gpu_ik_ref cur, ds4_gpu_ik_ref msg, ds4_gpu_ik_ref gain, ds4_gpu_ik_ref forget, uint32_t count) {
    ikm_bind b[5];
    const uint64_t by = (uint64_t)count * 4;
    if (!ikm_ref(&b[0], cur, by, "ctl apply cur") || !ikm_ref(&b[1], msg, by, "ctl apply msg") ||
        !ikm_ref(&b[2], gain, by, "ctl apply gain") || !ikm_ref(&b[3], forget, by, "ctl apply forget") ||
        !ikm_ref(&b[4], out, by, "ctl apply out")) return 0;
    return ikm_dispatch("kernel_ik_ctl_apply", &count, sizeof(count), b, 5, MTLSizeMake(IKM_CEIL(count, 256u), 1, 1), MTLSizeMake(256, 1, 1));
}

/* Causal 4-tap conv with residual over x[n][C]; tail (f32 [C][3]) is read by the first tokens and then advanced. */
int ds4_gpu_ik_conv(ds4_gpu_ik_ref out, ds4_gpu_ik_ref x, ds4_gpu_ik_ref tail, ds4_gpu_ik_w w, ds4_gpu_ik_ref dyn,
                    uint32_t C, uint32_t hd, float scale, uint32_t n) {
    struct { uint32_t n_tokens, C, hd, has_dyn; float scale; uint32_t tail_off; } args = { n, C, hd, dyn.t != NULL, scale, 0 };
    ikm_bind b[5];
    if (!ikm_ref(&b[0], x, (uint64_t)n * C * 4, "conv x") || !ikm_ref(&b[1], tail, (uint64_t)C * 12, "conv tail") ||
        !ikm_weight(&b[2], w, "conv weight") || !ikm_ref(&b[3], dyn, 0, "conv dyn") ||
        !ikm_ref(&b[4], out, (uint64_t)n * C * 4, "conv out")) return 0;
    if (!ikm_dispatch("kernel_ik_conv", &args, sizeof(args), b, 5, MTLSizeMake(IKM_CEIL(C, 64u), n, 1), MTLSizeMake(64, 1, 1))) return 0;
    ikm_bind t[2] = { b[0], b[1] };
    return ikm_dispatch("kernel_ik_conv_tail", &args, sizeof(args), t, 2, MTLSizeMake(IKM_CEIL(C, 128u), 1, 1), MTLSizeMake(128, 1, 1));
}

int ds4_gpu_ik_qr_prep(ds4_gpu_ik_ref Q, ds4_gpu_ik_ref R, ds4_gpu_ik_w qclip, ds4_gpu_ik_w rclip, ds4_gpu_ik_ref tau,
                       int global, uint32_t n) {
    struct { uint32_t n_tokens, global; } args = { n, global != 0 };
    ikm_bind b[5];
    if (!ikm_ref(&b[0], Q, (uint64_t)n * 24 * 128 * 4, "qr Q") || !ikm_ref(&b[1], R, (uint64_t)n * 24 * 16 * 4, "qr R") ||
        !ikm_weight(&b[2], qclip, "q clip") || !ikm_weight(&b[3], rclip, "rel clip") || !ikm_ref(&b[4], tau, 0, "tau")) return 0;
    return ikm_dispatch("kernel_ik_qr_prep", &args, sizeof(args), b, 5, MTLSizeMake(24, n, 1), MTLSizeMake(32, 1, 1));
}

int ds4_gpu_ik_kv_store(ds4_gpu_ik_ref Kc, ds4_gpu_ik_ref Vc, ds4_gpu_ik_ref kcache, ds4_gpu_ik_ref vcache,
                        uint32_t p0, uint32_t rows, uint32_t n) {
    struct { uint32_t n_tokens, p0, rows, pad; } args = { n, p0, rows, 0 };
    ikm_bind b[4];
    if (!ikm_ref(&b[0], Kc, (uint64_t)n * 1024 * 4, "kv Kc") || !ikm_ref(&b[1], Vc, (uint64_t)n * 1024 * 4, "kv Vc") ||
        !ikm_ref(&b[2], kcache, (uint64_t)rows * 1024 * 2, "k cache") || !ikm_ref(&b[3], vcache, (uint64_t)rows * 1024 * 2, "v cache")) return 0;
    return ikm_dispatch("kernel_ik_kv_store", &args, sizeof(args), b, 4, MTLSizeMake(8, n, 1), MTLSizeMake(32, 1, 1));
}

int ds4_gpu_ik_attn(ds4_gpu_ik_ref O, ds4_gpu_ik_ref Q, ds4_gpu_ik_ref R, ds4_gpu_ik_ref sink, ds4_gpu_ik_ref gate,
                    ds4_gpu_ik_w relb, ds4_gpu_ik_ref kcache, ds4_gpu_ik_ref vcache,
                    uint32_t p0, uint32_t rows, int local, uint32_t n, ds4_gpu_ik_ref part, uint32_t part_splits) {
    struct { uint32_t n_tokens, p0, rows, local; } args = { n, p0, rows, local != 0 };
    ikm_bind b[8];
    if (!ikm_ref(&b[0], Q, (uint64_t)n * 24 * 128 * 4, "attn Q") || !ikm_ref(&b[1], R, (uint64_t)n * 24 * 16 * 4, "attn R") ||
        !ikm_ref(&b[2], sink, (uint64_t)n * 24 * 4, "attn sink") || !ikm_ref(&b[3], gate, (uint64_t)n * 24 * 128 * 4, "attn gate") ||
        !ikm_weight(&b[4], relb, "rel bias") || !ikm_ref(&b[5], kcache, (uint64_t)rows * 1024 * 2, "attn k cache") ||
        !ikm_ref(&b[6], vcache, (uint64_t)rows * 1024 * 2, "attn v cache") || !ikm_ref(&b[7], O, (uint64_t)n * 24 * 128 * 4, "attn O")) return 0;
    const uint32_t nkeys_all = local ? (p0 + 1u < IKM_EXTENT_ ? p0 + 1u : IKM_EXTENT_) : p0 + 1u;
    if (n == 1 && part.t && part_splits > 1 && nkeys_all > IKM_SPLIT_MIN_KEYS && !ikm_exact() && !ikm_attn_scalar()) {
        /* decode over a long context: split the keys over threadgroups (kernel_ik_attn_split) and merge (kernel_ik_attn_reduce) */
        const uint32_t nkeys = nkeys_all;
        const uint32_t kps = local ? IKM_SPLIT_KEYS_LOCAL : IKM_SPLIT_KEYS;
        uint32_t S = (nkeys + kps - 1u) / kps;
        if (S > part_splits) S = part_splits;
        const uint32_t chunk = (nkeys + S - 1u) / S;
        S = (nkeys + chunk - 1u) / chunk;
        struct { uint32_t n_tokens, p0, rows, local, nsplit, chunk, pad0, pad1; } sa = { 1, p0, rows, local != 0, S, chunk, 0, 0 };
        ikm_bind sb[7], rb[3];
        sb[0] = b[0]; sb[1] = b[1]; sb[2] = b[2]; sb[3] = b[4]; sb[4] = b[5]; sb[5] = b[6];
        if (!ikm_ref(&sb[6], part, (uint64_t)24 * S * 130 * 4, "attn partials")) return 0;
        rb[0] = sb[6]; rb[1] = b[3]; rb[2] = b[7];
        return ikm_dispatch("kernel_ik_attn_split", &sa, sizeof(sa), sb, 7, MTLSizeMake(24, S, 1), MTLSizeMake(256, 1, 1)) &&
               ikm_dispatch("kernel_ik_attn_reduce", &sa, sizeof(sa), rb, 3, MTLSizeMake(24, 1, 1), MTLSizeMake(128, 1, 1));
    }
    if (n > 8 && !ikm_exact() && (rows % 8u) == 0 && !ikm_attn_scalar())   /* prefill: simdgroup-matrix attention */
        return ikm_dispatch("kernel_ik_attn_mma", &args, sizeof(args), b, 8, MTLSizeMake(8, IKM_CEIL(n, 8u), 1), MTLSizeMake(96, 1, 1));
    return ikm_dispatch("kernel_ik_attn", &args, sizeof(args), b, 8, MTLSizeMake(24, n, 1), MTLSizeMake(256, 1, 1));
}

int ds4_gpu_ik_router(ds4_gpu_ik_ref picks, ds4_gpu_ik_ref wts, ds4_gpu_ik_ref logits, ds4_gpu_ik_w bias,
                      ds4_gpu_ik_ref forced, uint32_t force_ntok, uint32_t slot, uint32_t p0, uint32_t n) {
    struct { uint32_t n_tokens, p0, slot, force_ntok; } args = { n, p0, slot, force_ntok };
    ikm_bind b[5];
    if (!ikm_ref(&b[0], logits, (uint64_t)n * 128 * 4, "router logits") || !ikm_weight(&b[1], bias, "router bias") ||
        !ikm_ref(&b[2], picks, (uint64_t)n * 8 * 4, "router picks") || !ikm_ref(&b[3], wts, (uint64_t)n * 8 * 4, "router weights") ||
        !ikm_ref(&b[4], forced, 0, "forced picks")) return 0;
    return ikm_dispatch("kernel_ik_router", &args, sizeof(args), b, 5, MTLSizeMake(n, 1, 1), MTLSizeMake(128, 1, 1));
}

int ds4_gpu_ik_moe_lists(ds4_gpu_ik_ref lists, ds4_gpu_ik_ref counts, ds4_gpu_ik_ref picks, ds4_gpu_ik_ref tiles, uint32_t n) {
    struct { uint32_t n_pairs, list_cap, make_tiles, pad; } args = { n * 8u, n, tiles.t != NULL, 0 };
    ikm_bind b[4];
    if (!ikm_ref(&b[0], picks, (uint64_t)n * 32, "lists picks") || !ikm_ref(&b[1], lists, (uint64_t)128 * n * 4, "lists") ||
        !ikm_ref(&b[2], counts, 128 * 4, "counts") || !ikm_ref(&b[3], tiles, tiles.t ? (uint64_t)(n * 8u / 32u + 130u) * 8 : 0, "tiles")) return 0;
    return ikm_dispatch("kernel_ik_moe_lists", &args, sizeof(args), b, 4, MTLSizeMake(1, 1, 1), MTLSizeMake(128, 1, 1));
}

/* Grouped simdgroup-matrix experts for chunks of more than 8 tokens (fast mode).  Return 2 = not applicable. */
static const char *ikm_mma_kname(const char *family, uint32_t type) {
    static const char *const gu[] = { "kernel_ik_moe_gum_f16", "kernel_ik_moe_gum_q8_0", "kernel_ik_moe_gum_bf16", "kernel_ik_moe_gum_mxfp4", "kernel_ik_moe_gum_nvfp4" };
    static const char *const dn[] = { "kernel_ik_moe_dnm_f16", "kernel_ik_moe_dnm_q8_0", "kernel_ik_moe_dnm_bf16", "kernel_ik_moe_dnm_mxfp4", "kernel_ik_moe_dnm_nvfp4" };
    int i = -1;
    switch (type) { case 1u: i = 0; break; case 8u: i = 1; break; case 30u: i = 2; break; case 39u: i = 3; break; case 40u: i = 4; break; default: break; }
    if (i < 0) return NULL;
    return family[0] == 'g' ? gu[i] : dn[i];
}

int ds4_gpu_ik_moe_gu_mma(ds4_gpu_ik_ref ac, ds4_gpu_ik_w w, ds4_gpu_ik_w gscale, ds4_gpu_ik_ref h, ds4_gpu_ik_ref lists, ds4_gpu_ik_ref counts,
                          ds4_gpu_ik_ref tiles, uint32_t n) {
    const char *name = ikm_mma_kname("gu", w.type);
    if (n <= 8 || ikm_exact() || !name) return 2;
    const uint32_t rb = ikm_row_bytes(w.type, IKM_E_);
    struct { uint32_t in_dim, out_rows, n_tokens, row_bytes; uint64_t expert_bytes; uint32_t list_cap, has_scale; } args =
        { IKM_E_, IKM_FF_, n, rb, (uint64_t)rb * 2u * IKM_FF_, n, gscale.map != NULL };
    const uint32_t slots = n * 8u / 32u + 128u;
    ikm_bind b[8];
    if (!ikm_weight(&b[0], w, "gate_up experts") || !ikm_ref(&b[1], h, (uint64_t)n * IKM_E_ * 4, "moe h") ||
        !ikm_ref(&b[2], lists, (uint64_t)128 * n * 4, "moe lists") || !ikm_ref(&b[3], counts, 128 * 4, "moe counts") ||
        !ikm_weight(&b[4], gscale, "gate_up scale") || !ikm_ref(&b[5], ac, (uint64_t)n * 8 * IKM_FF_ * 4, "moe ac") ||
        !ikm_ref(&b[6], tiles, (uint64_t)(slots + 2) * 8, "moe tiles")) return 0;
    return ikm_dispatch(name, &args, sizeof(args), b, 7, MTLSizeMake(IKM_FF_ / 32u, slots, 1), MTLSizeMake(128, 1, 1)) ? 1 : 0;
}

int ds4_gpu_ik_moe_down_mma(ds4_gpu_ik_ref yd, ds4_gpu_ik_w w, ds4_gpu_ik_ref ac, ds4_gpu_ik_ref lists, ds4_gpu_ik_ref counts, ds4_gpu_ik_ref tiles,
                            uint32_t n) {
    const char *name = ikm_mma_kname("dn", w.type);
    if (n <= 8 || ikm_exact() || !name) return 2;
    const uint32_t rb = ikm_row_bytes(w.type, IKM_FF_);
    struct { uint32_t in_dim, out_rows, n_tokens, row_bytes; uint64_t expert_bytes; uint32_t list_cap, has_scale; } args =
        { IKM_FF_, IKM_E_, n, rb, (uint64_t)rb * IKM_E_, n, 0 };
    const uint32_t slots = n * 8u / 32u + 128u;
    ikm_bind b[7];
    if (!ikm_weight(&b[0], w, "down experts") || !ikm_ref(&b[1], ac, (uint64_t)n * 8 * IKM_FF_ * 4, "moe down ac") ||
        !ikm_ref(&b[2], lists, (uint64_t)128 * n * 4, "moe down lists") || !ikm_ref(&b[3], counts, 128 * 4, "moe down counts") ||
        !ikm_ref(&b[4], yd, (uint64_t)n * 8 * IKM_E_ * 4, "moe yd") || !ikm_ref(&b[5], tiles, (uint64_t)(slots + 2) * 8, "moe tiles")) return 0;
    return ikm_dispatch(name, &args, sizeof(args), b, 6, MTLSizeMake(IKM_E_ / 32u, slots, 1), MTLSizeMake(128, 1, 1)) ? 1 : 0;
}

int ds4_gpu_ik_moe_gu(ds4_gpu_ik_ref ac, ds4_gpu_ik_w w, ds4_gpu_ik_w gscale, ds4_gpu_ik_ref h, ds4_gpu_ik_ref lists,
                      ds4_gpu_ik_ref counts, uint32_t n) {
    const char *name = ikm_kname("gu", w.type);
    if (!name) return 0;
    const uint32_t rb = ikm_row_bytes(w.type, IKM_E_);
    struct { uint32_t in_dim, out_rows, n_tokens, row_bytes; uint64_t expert_bytes; uint32_t list_cap, has_scale; } args =
        { IKM_E_, IKM_FF_, n, rb, (uint64_t)rb * 2u * IKM_FF_, n, gscale.map != NULL };
    ikm_bind b[6];
    if (!ikm_weight(&b[0], w, "gate_up experts") || !ikm_ref(&b[1], h, (uint64_t)n * IKM_E_ * 4, "moe h") ||
        !ikm_ref(&b[2], lists, (uint64_t)128 * n * 4, "moe lists") || !ikm_ref(&b[3], counts, 128 * 4, "moe counts") ||
        !ikm_weight(&b[4], gscale, "gate_up scale") || !ikm_ref(&b[5], ac, (uint64_t)n * 8 * IKM_FF_ * 4, "moe ac")) return 0;
    return ikm_dispatch(name, &args, sizeof(args), b, 6, MTLSizeMake(IKM_CEIL(IKM_FF_, 8u), 128, 1), MTLSizeMake(128, 1, 1));
}

int ds4_gpu_ik_moe_down(ds4_gpu_ik_ref yd, ds4_gpu_ik_w w, ds4_gpu_ik_ref ac, ds4_gpu_ik_ref lists, ds4_gpu_ik_ref counts, uint32_t n) {
    const char *name = ikm_kname("down", w.type);
    if (!name) return 0;
    const uint32_t rb = ikm_row_bytes(w.type, IKM_FF_);
    struct { uint32_t in_dim, out_rows, n_tokens, row_bytes; uint64_t expert_bytes; uint32_t list_cap, has_scale; } args =
        { IKM_FF_, IKM_E_, n, rb, (uint64_t)rb * IKM_E_, n, 0 };
    ikm_bind b[5];
    if (!ikm_weight(&b[0], w, "down experts") || !ikm_ref(&b[1], ac, (uint64_t)n * 8 * IKM_FF_ * 4, "moe down ac") ||
        !ikm_ref(&b[2], lists, (uint64_t)128 * n * 4, "moe down lists") || !ikm_ref(&b[3], counts, 128 * 4, "moe down counts") ||
        !ikm_ref(&b[4], yd, (uint64_t)n * 8 * IKM_E_ * 4, "moe yd")) return 0;
    return ikm_dispatch(name, &args, sizeof(args), b, 5, MTLSizeMake(IKM_CEIL(IKM_E_, 8u), 128, 1), MTLSizeMake(128, 1, 1));
}

int ds4_gpu_ik_moe_combine(ds4_gpu_ik_ref out, ds4_gpu_ik_ref yd, ds4_gpu_ik_ref wts, ds4_gpu_ik_ref picks, ds4_gpu_ik_w dscale, uint32_t n) {
    struct { uint32_t n_tokens, has_scale; } args = { n, dscale.map != NULL };
    ikm_bind b[5];
    if (!ikm_ref(&b[0], yd, (uint64_t)n * 8 * IKM_E_ * 4, "combine yd") || !ikm_ref(&b[1], wts, (uint64_t)n * 32, "combine wts") ||
        !ikm_ref(&b[2], picks, (uint64_t)n * 32, "combine picks") || !ikm_weight(&b[3], dscale, "down scale") ||
        !ikm_ref(&b[4], out, (uint64_t)n * IKM_E_ * 4, "combine out")) return 0;
    return ikm_dispatch("kernel_ik_moe_combine", &args, sizeof(args), b, 5, MTLSizeMake(IKM_CEIL(n * IKM_E_, 256u), 1, 1), MTLSizeMake(256, 1, 1));
}

/* One-token NVFP4 experts (see kernel_ik_moe_gu1_nvfp4): usable when the experts are NVFP4 and EXACT is off. */
int ds4_gpu_ik_moe_decode_ok(uint32_t gu_type, uint32_t down_type) { return gu_type == 40u && down_type == 40u && !ikm_exact(); }

int ds4_gpu_ik_moe_gu1(ds4_gpu_ik_ref ac, ds4_gpu_ik_w w, ds4_gpu_ik_w gscale, ds4_gpu_ik_ref h, ds4_gpu_ik_ref picks, uint32_t n) {
    if (n == 0 || n > 8) return 0;
    const uint32_t rb = ikm_row_bytes(w.type, IKM_E_);
    struct { uint32_t in_dim, out_rows, n_tokens, row_bytes; uint64_t expert_bytes; uint32_t list_cap, has_scale; } args =
        { IKM_E_, IKM_FF_, 1, rb, (uint64_t)rb * 2u * IKM_FF_, 1, gscale.map != NULL };
    ikm_bind b[5];
    if (w.type != 40u || !ikm_weight(&b[0], w, "gate_up experts") || !ikm_ref(&b[1], h, (uint64_t)n * IKM_E_ * 4, "moe h") ||
        !ikm_ref(&b[2], picks, (uint64_t)n * 32, "moe picks") || !ikm_weight(&b[3], gscale, "gate_up scale") ||
        !ikm_ref(&b[4], ac, (uint64_t)n * 8 * IKM_FF_ * 4, "moe ac")) return 0;
    return ikm_dispatch("kernel_ik_moe_gu1_nvfp4", &args, sizeof(args), b, 5, MTLSizeMake(IKM_CEIL(IKM_FF_, 8u), 8, n), MTLSizeMake(128, 1, 1));
}

int ds4_gpu_ik_moe_dc1(ds4_gpu_ik_ref out, ds4_gpu_ik_w w, ds4_gpu_ik_ref ac, ds4_gpu_ik_ref picks, ds4_gpu_ik_ref wts, ds4_gpu_ik_w dscale, uint32_t n) {
    if (n == 0 || n > 8) return 0;
    const uint32_t rb = ikm_row_bytes(w.type, IKM_FF_);
    struct { uint32_t in_dim, out_rows, n_tokens, row_bytes; uint64_t expert_bytes; uint32_t list_cap, has_scale; } args =
        { IKM_FF_, IKM_E_, 1, rb, (uint64_t)rb * IKM_E_, 1, dscale.map != NULL };
    ikm_bind b[6];
    if (w.type != 40u || !ikm_weight(&b[0], w, "down experts") || !ikm_ref(&b[1], ac, (uint64_t)n * 8 * IKM_FF_ * 4, "moe down ac") ||
        !ikm_ref(&b[2], picks, (uint64_t)n * 32, "moe picks") || !ikm_ref(&b[3], wts, (uint64_t)n * 32, "moe wts") || !ikm_weight(&b[4], dscale, "down scale") ||
        !ikm_ref(&b[5], out, (uint64_t)n * IKM_E_ * 4, "moe out")) return 0;
    return ikm_dispatch("kernel_ik_moe_dc1_nvfp4", &args, sizeof(args), b, 6, MTLSizeMake(IKM_E_ / 2u, 1, n), MTLSizeMake(256, 1, 1));
}

/* One-token fast paths (decode): used when n == 1 and DS4_INKLING2_EXACT is off. */
int ds4_gpu_ik_fast1(uint32_t n) { return n == 1u && !ikm_exact(); }

/* conv + tail update + optional residual in one dispatch (n == 1). */
int ds4_gpu_ik_conv1(ds4_gpu_ik_ref out, ds4_gpu_ik_ref x, ds4_gpu_ik_ref tail, ds4_gpu_ik_w w, ds4_gpu_ik_ref dyn, ds4_gpu_ik_ref addend,
                     uint32_t C, uint32_t hd, float scale) {
    struct { uint32_t C, hd, has_dyn, has_add; float scale; } args = { C, hd, dyn.t != NULL, addend.t != NULL, scale };
    ikm_bind b[6];
    if (!ikm_ref(&b[0], x, (uint64_t)C * 4, "conv1 x") || !ikm_ref(&b[1], tail, (uint64_t)C * 12, "conv1 tail") ||
        !ikm_weight(&b[2], w, "conv1 weight") || !ikm_ref(&b[3], dyn, 0, "conv1 dyn") || !ikm_ref(&b[4], addend, 0, "conv1 add") ||
        !ikm_ref(&b[5], out, (uint64_t)C * 4, "conv1 out")) return 0;
    return ikm_dispatch("kernel_ik_conv1", &args, sizeof(args), b, 6, MTLSizeMake(IKM_CEIL(C, 64u), 1, 1), MTLSizeMake(64, 1, 1));
}

/* Up to six Q8_0 matvecs of one token with the same input width in one dispatch (see kernel_ik_mvm_q8).  x holds the (up to three)
 * distinct inputs, xsel[g] picks one per matvec, y[g] is each output.  Returns 1 done, 2 not applicable (caller runs them one by one), 0 error. */
int ds4_gpu_ik_mm_multi(uint32_t nseg, const ds4_gpu_ik_w *w, const uint32_t *rows, uint32_t in_dim, const ds4_gpu_ik_ref *x,
                        const uint32_t *xsel, const ds4_gpu_ik_ref *y) {
    if (!ikm_fast1_enabled() || nseg == 0 || nseg > 6 || (in_dim % 32u) != 0) return 2;
    struct { uint32_t nseg, in_dim, row_bytes, pad; uint32_t row_end[8]; uint32_t xsel[8]; int64_t woff[8]; } args;
    memset(&args, 0, sizeof(args));
    args.nseg = nseg; args.in_dim = in_dim; args.row_bytes = ikm_row_bytes(8u, in_dim);
    ikm_bind b[10];
    uint32_t total = 0;
    for (uint32_t g = 0; g < nseg; g++) {
        if (w[g].type != 8u) return 2;
        ikm_bind wb;
        if (!ikm_weight(&wb, w[g], "multi weight")) return 0;
        if (g == 0) b[0] = wb;
        else if (wb.buf != b[0].buf) return 2;
        args.woff[g] = (int64_t)wb.off - (int64_t)b[0].off;
        total += rows[g];
        args.row_end[g] = total;
        args.xsel[g] = xsel[g];
        if (!ikm_ref(&b[4 + g], y[g], (uint64_t)rows[g] * 4, "multi y")) return 0;
    }
    for (uint32_t g = nseg; g < 6; g++) { ds4_gpu_ik_ref none = {0}; if (!ikm_ref(&b[4 + g], none, 0, "multi y")) return 0; }
    for (int i = 0; i < 3; i++) if (!ikm_ref(&b[1 + i], x[i], (uint64_t)in_dim * 4, "multi x")) return 0;
    return ikm_dispatch("kernel_ik_mvm_q8", &args, sizeof(args), b, 10, MTLSizeMake(IKM_CEIL(total, 4u), 1, 1), MTLSizeMake(128, 1, 1)) ? 1 : 0;
}

/* Same-buffer weight set: binds weights w[0..n) relative to w[0]'s binding; returns 0 on error, 2 if they are in different buffers. */
static int ikm_wset(ikm_bind *base, int64_t *delta, const ds4_gpu_ik_w *w, uint32_t n) {
    for (uint32_t g = 0; g < n; g++) {
        ikm_bind wb;
        if (!ikm_weight(&wb, w[g], "weight set")) return 0;
        if (g == 0) *base = wb;
        else if (wb.buf != base->buf) return 2;
        delta[g] = (int64_t)wb.off - (int64_t)base->off;
    }
    delta[0] = 0;
    return 1;
}

/* Loop controllers, one token, nctl <= 3 in two dispatches.  cur_j = cur + j*cur_stride floats, msg_j = msg + j*msg_stride.
 * out[j] receives the controlled input.  Returns 1 done, 2 not applicable, 0 error. */
int ds4_gpu_ik_ctl_multi(uint32_t nctl, const ds4_gpu_ik_w *wdown, const ds4_gpu_ik_w *wgain, const ds4_gpu_ik_w *wforget,
                         ds4_gpu_ik_ref cur, uint32_t cur_stride, ds4_gpu_ik_ref msg, uint32_t msg_stride, ds4_gpu_ik_ref d,
                         const ds4_gpu_ik_ref *out) {
    if (!ikm_fast1_enabled() || nctl == 0 || nctl > 3) return 2;
    struct { uint32_t nctl, cur_stride, msg_stride, pad; int64_t woff[3]; } a1;
    struct { uint32_t nctl, cur_stride, msg_stride, pad; int64_t wg[3]; int64_t wf[3]; } a2;
    memset(&a1, 0, sizeof(a1)); memset(&a2, 0, sizeof(a2));
    a1.nctl = a2.nctl = nctl; a1.cur_stride = a2.cur_stride = cur_stride; a1.msg_stride = a2.msg_stride = msg_stride;
    ikm_bind bd[5], bg[8]; int rc;
    if ((rc = ikm_wset(&bd[0], a1.woff, wdown, nctl)) != 1) return rc;
    ikm_bind gb, fb;
    if ((rc = ikm_wset(&gb, a2.wg, wgain, nctl)) != 1) return rc;
    if ((rc = ikm_wset(&fb, a2.wf, wforget, nctl)) != 1) return rc;
    if (fb.buf != gb.buf) return 2;
    for (uint32_t j = 0; j < nctl; j++) a2.wf[j] += (int64_t)fb.off - (int64_t)gb.off;
    bg[0] = gb;
    if (!ikm_ref(&bd[1], cur, 0, "ctl cur") || !ikm_ref(&bd[2], msg, 0, "ctl msg") || !ikm_ref(&bd[3], d, (uint64_t)nctl * 256, "ctl d") ||
        !ikm_ref(&bg[1], cur, 0, "ctl cur") || !ikm_ref(&bg[2], msg, 0, "ctl msg") || !ikm_ref(&bg[3], d, (uint64_t)nctl * 256, "ctl d")) return 0;
    for (uint32_t j = 0; j < 3; j++) {
        ds4_gpu_ik_ref none = {0};
        if (!ikm_ref(&bg[4 + j], j < nctl ? out[j] : none, j < nctl ? (uint64_t)IKM_E_ * 4 : 0, "ctl out")) return 0;
    }
    if (!ikm_dispatch("kernel_ik_ctl_down", &a1, sizeof(a1), bd, 4, MTLSizeMake(IKM_CEIL(nctl * 64u, 4u), 1, 1), MTLSizeMake(128, 1, 1))) return 0;
    return ikm_dispatch("kernel_ik_ctl_apply_m", &a2, sizeof(a2), bg, 7, MTLSizeMake(IKM_CEIL(nctl * IKM_E_, 4u), 1, 1), MTLSizeMake(128, 1, 1)) ? 1 : 0;
}

/* Up to three rms norms of one token (own input, weight, output each; in place allowed). */
int ds4_gpu_ik_rms3(uint32_t nrow, const ds4_gpu_ik_ref *x, const ds4_gpu_ik_w *w, const ds4_gpu_ik_ref *y) {
    if (!ikm_fast1_enabled() || nrow == 0 || nrow > 3) return 2;
    struct { uint32_t dim, nrow, pad0, pad1; int64_t woff[3]; } a;
    memset(&a, 0, sizeof(a));
    a.dim = IKM_E_; a.nrow = nrow;
    ikm_bind b[7]; int rc;
    if ((rc = ikm_wset(&b[0], a.woff, w, nrow)) != 1) return rc;
    for (uint32_t j = 0; j < 3; j++) {
        ds4_gpu_ik_ref none = {0};
        if (!ikm_ref(&b[1 + j], j < nrow ? x[j] : none, j < nrow ? (uint64_t)IKM_E_ * 4 : 0, "rms3 x") ||
            !ikm_ref(&b[4 + j], j < nrow ? y[j] : none, j < nrow ? (uint64_t)IKM_E_ * 4 : 0, "rms3 y")) return 0;
    }
    return ikm_dispatch("kernel_ik_rms3", &a, sizeof(a), b, 7, MTLSizeMake(nrow, 1, 1), MTLSizeMake(256, 1, 1)) ? 1 : 0;
}

int ds4_gpu_ik_add(ds4_gpu_ik_ref out, ds4_gpu_ik_ref a, ds4_gpu_ik_ref b_, uint32_t count) {
    ikm_bind b[3];
    const uint64_t by = (uint64_t)count * 4;
    if (!ikm_ref(&b[0], a, by, "add a") || !ikm_ref(&b[1], b_, by, "add b") || !ikm_ref(&b[2], out, by, "add out")) return 0;
    return ikm_dispatch("kernel_ik_add", &count, sizeof(count), b, 3, MTLSizeMake(IKM_CEIL(count, 256u), 1, 1), MTLSizeMake(256, 1, 1));
}

int ds4_gpu_ik_softcap(ds4_gpu_ik_ref x, uint32_t count) {
    ikm_bind b[1];
    if (!ikm_ref(&b[0], x, (uint64_t)count * 4, "softcap x")) return 0;
    return ikm_dispatch("kernel_ik_softcap", &count, sizeof(count), b, 1, MTLSizeMake(IKM_CEIL(count, 256u), 1, 1), MTLSizeMake(256, 1, 1));
}
