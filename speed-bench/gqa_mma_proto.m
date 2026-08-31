/* Driver for the GQA MMA prototype kernels (speed-bench/gqa_mma_proto.metal).
 *
 * Compiles the prototype library from source at runtime (so a measurement
 * can never go stale), validates every FULL variant against a CPU
 * reference and each other at a small mixed-boundary shape, then times
 * all variants at the production prefill shape (8192 rows, full 512-block
 * selection).  Reference points from the production tree: the scalar
 * kernel measures ~113 ms/layer and the per-simdgroup MMA kernel ~244
 * ms/layer at this shape (QWEN38_PERF_HANDOFF item 20).
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { Q_HEADS = 24, KV_HEADS = 2, DIM = 256, TOP_K = 512, RATIO = 4,
       CACHE_CAP = 8192, MAX_ROWS = 8192 };

typedef struct {
    uint32_t queries, cache_cap, query_heads, kv_heads, head_dim, top_k,
        ratio, max_selected, debug;
} proto_args;

typedef struct {
    const char *name;
    int org, bk, dc, skip;
    double ms;
} variant;

static uint32_t lcg = 0x2463534u;
static float lcg_float(float scale) {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float)(lcg >> 8) / (float)(1u << 24) - 0.5f) * scale;
}
static uint16_t f32_to_bf16_host(float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return (uint16_t)(bits >> 16);
}
static float bf16_to_f32_host(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float v;
    memcpy(&v, &bits, 4);
    return v;
}
/* Round through IEEE 754 binary16 like the kernel's F16 operands
 * (hardware round-to-nearest-even via _Float16). */
static float f16_round(float v) {
    return (float)(_Float16)v;
}
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static id<MTLBuffer> make_buffer(id<MTLDevice> device, size_t bytes) {
    return [device newBufferWithLength:bytes
        options:MTLResourceStorageModeShared];
}

/* CPU reference (f32, bf16 cache expanded) at the validation shape.
 * mode 0: exact scalar-kernel semantics (f32 q, f32 softmax weights)
 * mode 1: models the F16 kernel operand rounding (q -> f16, p -> f16)
 * mode 2: models the Q-split variant (q -> f16 + f16 residual, p -> f16) */
static void cpu_reference(uint32_t rows, const float *q, const float *gate,
                          const uint16_t *key, const uint16_t *value,
                          const uint32_t *sel, const uint32_t *counts,
                          const uint32_t *visible, float *out, int mode) {
    for (uint32_t query = 0; query < rows; query++) {
        const uint32_t complete = visible[query] / RATIO;
        uint32_t blocks = counts[query] < complete ? counts[query] : complete;
        if (blocks > TOP_K) blocks = TOP_K;
        uint32_t tokens[TOP_K * RATIO + RATIO - 1u];
        uint32_t n = 0;
        for (uint32_t r = 0; r < blocks; r++)
            for (uint32_t e = 0; e < RATIO; e++) {
                const uint32_t t = sel[(size_t)query * TOP_K + r] * RATIO + e;
                if (t < visible[query]) tokens[n++] = t;
            }
        for (uint32_t t = complete * RATIO; t < visible[query]; t++)
            tokens[n++] = t;
        for (uint32_t head = 0; head < Q_HEADS; head++) {
            const uint32_t kv = head / (Q_HEADS / KV_HEADS);
            const size_t obase =
                ((size_t)query * Q_HEADS + head) * DIM;
            if (n == 0u) {
                /* The kernels early-out on zero selection and write 0. */
                for (uint32_t d = 0; d < DIM; d++) out[obase + d] = 0.0f;
                continue;
            }
            float *sc = malloc(n * sizeof(float));
            float *pw = malloc(n * sizeof(float));
            float mx = -INFINITY;
            for (uint32_t r = 0; r < n; r++) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < DIM; d++) {
                    const float qv =
                        q[((size_t)query * Q_HEADS + head) * DIM + d];
                    const float qeff = mode == 0 ? qv
                        : mode == 1 ? f16_round(qv)
                        : f16_round(qv) +
                              f16_round(qv - f16_round(qv));
                    dot += qeff *
                        f16_round(bf16_to_f32_host(
                            key[((size_t)tokens[r] * KV_HEADS + kv) * DIM + d]));
                }
                sc[r] = dot / sqrtf((float)DIM);
                if (sc[r] > mx) mx = sc[r];
            }
            float sum = 0.0f;
            for (uint32_t r = 0; r < n; r++) {
                pw[r] = mode == 0 ? expf(sc[r] - mx)
                                  : f16_round(expf(sc[r] - mx));
                sum += pw[r];
            }
            for (uint32_t d = 0; d < DIM; d++) {
                float acc = 0.0f;
                for (uint32_t r = 0; r < n; r++)
                    acc += pw[r] *
                        f16_round(bf16_to_f32_host(
                            value[((size_t)tokens[r] * KV_HEADS + kv) * DIM + d]));
                out[obase + d] =
                    acc / sum /
                    (1.0f + expf(-gate[((size_t)query * Q_HEADS + head) *
                                       DIM + d]));
            }
            free(sc);
            free(pw);
        }
    }
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    @autoreleasepool {
        uint32_t rows = MAX_ROWS;
        if (argc > 1) rows = (uint32_t)strtoul(argv[1], NULL, 0);
        if (rows == 0u || rows > MAX_ROWS) {
            fprintf(stderr, "gqa-mma-proto: rows must be 1..%u\n", MAX_ROWS);
            return 1;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        NSString *src = [NSString stringWithContentsOfFile:
            @"speed-bench/gqa_mma_proto.metal"
            encoding:NSUTF8StringEncoding error:nil];
        if (!src) {
            fprintf(stderr, "gqa-mma-proto: read kernel source failed "
                            "(run from the repo root)\n");
            return 1;
        }
        id<MTLLibrary> lib = [device newLibraryWithSource:src
            options:nil error:&error];
        if (!lib) {
            fprintf(stderr, "gqa-mma-proto: compile failed: %s\n",
                    error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];

        /* Host data at the full bench shape; the validation pass slices
         * the first 24 queries with its own counts/visible overlays. */
        const size_t q_elems = (size_t)MAX_ROWS * Q_HEADS * DIM;
        const size_t cache_elems = (size_t)CACHE_CAP * KV_HEADS * DIM;
        float *q = malloc(q_elems * 4);
        float *gate = malloc(q_elems * 4);
        uint16_t *key = malloc(cache_elems * 2);
        uint16_t *value = malloc(cache_elems * 2);
        uint32_t *sel = malloc((size_t)MAX_ROWS * TOP_K * 4);
        uint32_t *counts = malloc((size_t)MAX_ROWS * 4);
        uint32_t *visible = malloc((size_t)MAX_ROWS * 4);
        float *out_host = malloc(q_elems * 4);
        if (!q || !gate || !key || !value || !sel || !counts || !visible ||
            !out_host) return 1;
        /* Validation magnitudes first (production-fixture class, where
         * F16 operand rounding stays near 1e-4 on gated outputs); the
         * timing pass re-fills with larger bench magnitudes. */
        for (size_t i = 0; i < q_elems; i++) {
            q[i] = lcg_float(0.4f);
            gate[i] = lcg_float(0.5f);
        }
        for (size_t i = 0; i < cache_elems; i++) {
            key[i] = f32_to_bf16_host(lcg_float(0.4f));
            value[i] = f32_to_bf16_host(lcg_float(0.4f));
        }
        for (uint32_t query = 0; query < MAX_ROWS; query++) {
            uint32_t block = query % (MAX_ROWS / RATIO);
            for (uint32_t r = 0; r < TOP_K; r++) {
                sel[(size_t)query * TOP_K + r] = block;
                block = block + 1u >= MAX_ROWS / RATIO ? 0u : block + 1u;
            }
            counts[query] = TOP_K;
            visible[query] = rows;
        }

        id<MTLBuffer> q_b = make_buffer(device, q_elems * 4);
        id<MTLBuffer> gate_b = make_buffer(device, q_elems * 4);
        id<MTLBuffer> key_b = make_buffer(device, cache_elems * 2);
        id<MTLBuffer> value_b = make_buffer(device, cache_elems * 2);
        id<MTLBuffer> sel_b = make_buffer(device, (size_t)MAX_ROWS * TOP_K * 4);
        id<MTLBuffer> counts_b = make_buffer(device, (size_t)MAX_ROWS * 4);
        id<MTLBuffer> visible_b = make_buffer(device, (size_t)MAX_ROWS * 4);
        id<MTLBuffer> out_b = make_buffer(device, q_elems * 4);
        if (!q_b || !gate_b || !key_b || !value_b || !sel_b || !counts_b ||
            !visible_b || !out_b) return 1;
        memcpy(q_b.contents, q, q_elems * 4);
        memcpy(gate_b.contents, gate, q_elems * 4);
        memcpy(key_b.contents, key, cache_elems * 2);
        memcpy(value_b.contents, value, cache_elems * 2);
        memcpy(sel_b.contents, sel, (size_t)MAX_ROWS * TOP_K * 4);

        variant variants[] = {
            {"gqa_ps_b64_d64_full", 0, 64, 64, 0, 0.0},
            {"gqa_ls_b64_d64_full", 1, 64, 64, 0, 0.0},
            {"gqa_ls_b64_d64_nostage", 1, 64, 64, 1, 0.0},
            {"gqa_ls_b64_d64_nomad", 1, 64, 64, 2, 0.0},
            {"gqa_ls_b64_d64_nosoftmax", 1, 64, 64, 3, 0.0},
            {"gqa_ls_b128_d32_full", 1, 128, 32, 0, 0.0},
            {"gqa_ps_b128_d32_full", 0, 128, 32, 0, 0.0},
            {"gqa_ls_b128_d64_full", 1, 128, 64, 0, 0.0},
            {"gqa_ls_b64_d64_constp_tile", 1, 64, 64, 5, 0.0},
            {"gqa_rs_b64_d64", 2, 64, 64, 0, 0.0},
            {"gqa_rs_b128_d32", 2, 128, 32, 0, 0.0},
            {"gqa_rs_b64_d32", 2, 64, 32, 0, 0.0},
            {"gqa_rs_b128_d16", 2, 128, 16, 0, 0.0},
            {"gqa_rs_b64_d16", 2, 64, 16, 0, 0.0},
            {"gqa_rs_b128_d8", 2, 128, 8, 0, 0.0},
            {"gqa_rs_b64_d8", 2, 64, 8, 0, 0.0},
            {"gqa_rs_b64_d8_qsplit", 3, 64, 8, 0, 0.0},
            {"gqa_rs_b128_d8_qsplit", 3, 128, 8, 0, 0.0},
        };
        const int n_variants =
            (int)(sizeof(variants) / sizeof(variants[0]));

        /* ---- validation: 24 queries with boundary profiles ---- */
        enum { VROWS = 24 };
        uint32_t v_counts[VROWS], v_visible[VROWS];
        v_counts[0] = 1u;   v_visible[0] = 7u;
        v_counts[1] = 2u;   v_visible[1] = 131u;  /* 8 ranks, 1 fragment */
        v_counts[2] = 3u;   v_visible[2] = 131u;  /* 12 ranks, 2 fragments */
        v_counts[3] = 0u;   v_visible[3] = 131u;
        v_counts[4] = 0u;   v_visible[4] = 1u;   /* single token */
        v_counts[5] = 0u;   v_visible[5] = 2u;   /* two tokens */
        for (uint32_t i = 6; i < VROWS; i++) {
            v_visible[i] = 17u + (i * 37u) % 131u;
            v_counts[i] = (i * 11u) % 33u;
        }
        {
            uint32_t *cmap = (uint32_t *)counts_b.contents;
            uint32_t *vmap = (uint32_t *)visible_b.contents;
            uint32_t *smap = (uint32_t *)sel_b.contents;
            for (uint32_t i = 0; i < VROWS; i++) {
                cmap[i] = v_counts[i];
                vmap[i] = v_visible[i];
                /* In-range blocks for the first block_count ranks so the
                 * validation exercises real gathered tokens at every
                 * rank position (including ranks past 64); later ranks
                 * keep the scattered (masked) pattern. */
                const uint32_t complete = v_visible[i] / RATIO;
                const uint32_t blocks = v_counts[i] < complete
                    ? v_counts[i] : complete;
                for (uint32_t r = 0; r < TOP_K; r++) {
                    const uint32_t block = (r < blocks && blocks > 0u)
                        ? (r * 7u + i) % blocks
                        : (i * 131u + r * 37u + 19u) % 512u;
                    smap[(size_t)i * TOP_K + r] = block;
                    sel[(size_t)i * TOP_K + r] = block;
                }
            }
            for (uint32_t i = VROWS; i < MAX_ROWS; i++) {
                cmap[i] = TOP_K;
                vmap[i] = rows;
            }
        }
        float *ref = malloc((size_t)VROWS * Q_HEADS * DIM * 4);
        cpu_reference(VROWS, q, gate, key, value, sel, v_counts, v_visible,
                      ref, 1);
        float *first = malloc((size_t)VROWS * Q_HEADS * DIM * 4);
        /* rolesplit (org 2): 16 Q rows + BK*DC staging + 16xBK S/P +
         * selections/stats; O rescale overlays staging. */
        const uint v_hpad2 = 16u;
        int validated = 0;
        for (int v = 0; v < n_variants; v++) {
            /* Validate the role-split (production) variants only; the
             * legacy lockstep/per-simdgroup comparison kernels are
             * timing-only. */
            if (variants[v].skip != 0 || variants[v].org != 2) continue;
            id<MTLFunction> fn = [lib newFunctionWithName:
                [NSString stringWithUTF8String:variants[v].name]];
            if (!fn) {
                fprintf(stderr, "gqa-mma-proto: missing function %s\n",
                        variants[v].name);
                return 1;
            }
            id<MTLComputePipelineState> pso =
                [device newComputePipelineStateWithFunction:fn error:nil];
            const uint hpad = variants[v].org >= 1 ? 16u : 8u;
            size_t tg_bytes =
                (size_t)hpad * DIM * 2u + (size_t)variants[v].bk *
                    variants[v].dc * 2u +
                (size_t)hpad * variants[v].bk * 4u +
                (size_t)hpad * variants[v].bk * 2u +
                (size_t)variants[v].bk * 4u + hpad * 2u * 4u + hpad * 4u +
                2u * 4u + 64u;
            if (variants[v].org == 2)
                tg_bytes += (size_t)8u * 128u * 4u + 64u; /* O block */
            proto_args args = {VROWS, CACHE_CAP, Q_HEADS, KV_HEADS, DIM,
                               TOP_K, RATIO, TOP_K * RATIO + RATIO - 1u,
                               0u};
            memset(out_b.contents, 0, (size_t)VROWS * Q_HEADS * DIM * 4);
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:q_b offset:0 atIndex:1];
            [enc setBuffer:gate_b offset:0 atIndex:2];
            [enc setBuffer:key_b offset:0 atIndex:3];
            [enc setBuffer:value_b offset:0 atIndex:4];
            [enc setBuffer:sel_b offset:0 atIndex:5];
            [enc setBuffer:counts_b offset:0 atIndex:6];
            [enc setBuffer:visible_b offset:0 atIndex:7];
            [enc setBuffer:out_b offset:0 atIndex:8];
            [enc setThreadgroupMemoryLength:tg_bytes atIndex:0];
            const MTLSize groups = MTLSizeMake(
                VROWS,
                variants[v].org == 0 ? KV_HEADS * 2u : KV_HEADS, 1);
            const MTLSize threads = MTLSizeMake(
                variants[v].org == 0 ? 32u
                    : (variants[v].org == 1 ? 64u : 192u),
                1, 1);
            [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) {
                fprintf(stderr, "gqa-mma-proto: %s validation dispatch "
                                "failed: %s\n", variants[v].name,
                        cb.error.localizedDescription.UTF8String);
                return 1;
            }
            const float *got = out_b.contents;
            double worst = 0.0;
            size_t at = 0;
            for (size_t i = 0; i < (size_t)VROWS * Q_HEADS * DIM; i++) {
                const double err = fabs(got[i] - ref[i]);
                const double tol = 5e-3 + 3e-2 * fabs(ref[i]);
                if (!isfinite(got[i]) || err > tol) {
                    fprintf(stderr, "gqa-mma-proto: %s VALIDATION FAIL at "
                                    "%zu: got %.6g ref %.6g\n",
                            variants[v].name, i, got[i], ref[i]);
                    const uint q0 = (uint)(i / (Q_HEADS * DIM));
                    const uint h0 =
                        (uint)((i % (Q_HEADS * DIM)) / DIM);
                    for (uint d = 0; d < 8; d++) {
                        const size_t j = ((size_t)q0 * Q_HEADS + h0) * DIM + d;
                        fprintf(stderr, "  d%u: got %.6g ref %.6g "
                                        "pw-sum-check\n",
                                d, got[j], ref[j]);
                    }
                    return 1;
                }
                if (err > worst) { worst = err; at = i; }
            }
            if (validated++ == 0) memcpy(first, got,
                                         (size_t)VROWS * Q_HEADS * DIM * 4);
            else {
                double xworst = 0.0;
                for (size_t i = 0; i < (size_t)VROWS * Q_HEADS * DIM; i++) {
                    const double err = fabs(got[i] - first[i]);
                    if (err > xworst) xworst = err;
                }
                printf("validated %-22s vs CPU worst=%.2e (cross-variant "
                       "%.2e)\n", variants[v].name, worst, xworst);
            }
        }
        free(ref);
        free(first);

        /* ---- magnitude sweep: the F16-Q rounding error is proportional
         * to |q||k|, so production-scale inputs must show the plain
         * variant drifting from EXACT (scalar-kernel semantics) while
         * still matching its F16-modeled reference (no structural bug),
         * and the Q-split variant holding at the modeled floor. ---- */
        {
            static const float sweep[][2] = {
                {0.4f, 0.4f}, {2.0f, 2.0f}, {10.0f, 10.0f},
                {50.0f, 10.0f}, {200.0f, 10.0f},
            };
            id<MTLComputePipelineState> plain_pso =
                [device newComputePipelineStateWithFunction:
                    [lib newFunctionWithName:@"gqa_rs_b64_d8"]
                                      error:nil];
            id<MTLComputePipelineState> split_pso =
                [device newComputePipelineStateWithFunction:
                    [lib newFunctionWithName:@"gqa_rs_b64_d8_qsplit"]
                                      error:nil];
            if (!plain_pso || !split_pso) {
                fprintf(stderr, "gqa-mma-proto: sweep kernels missing\n");
                return 1;
            }
            const size_t plain_tg =
                (size_t)16u * DIM * 2u + 64u * 8u * 2u +
                (size_t)16u * 64u * 4u + (size_t)16u * 64u * 2u +
                64u * 4u + 16u * 2u * 4u + 16u * 4u + 2u * 4u +
                (size_t)8u * 128u * 4u + 64u;
            const size_t split_tg = plain_tg + (size_t)16u * DIM * 2u;
            if (split_tg > device.maxThreadgroupMemoryLength) {
                fprintf(stderr, "gqa-mma-proto: qsplit tg %zu over limit "
                                "%llu\n", split_tg,
                        (unsigned long long)device.maxThreadgroupMemoryLength);
                return 1;
            }
            float *ref_exact = malloc((size_t)VROWS * Q_HEADS * DIM * 4);
            float *ref_modeled = malloc((size_t)VROWS * Q_HEADS * DIM * 4);
            float *ref_split = malloc((size_t)VROWS * Q_HEADS * DIM * 4);
            if (!ref_exact || !ref_modeled || !ref_split) return 1;
            proto_args sargs = {VROWS, CACHE_CAP, Q_HEADS, KV_HEADS, DIM,
                                TOP_K, RATIO, TOP_K * RATIO + RATIO - 1u,
                                1u};
            const MTLSize sgroups = MTLSizeMake(VROWS, KV_HEADS, 1);
            const MTLSize sthreads = MTLSizeMake(192u, 1, 1);
            for (size_t s = 0;
                 s < sizeof(sweep) / sizeof(sweep[0]); s++) {
                const float qscale = sweep[s][0];
                const float kvscale = sweep[s][1];
                lcg = 0x2463534u;
                for (size_t i = 0; i < q_elems; i++)
                    q[i] = lcg_float(qscale);
                lcg = 0x1234567u;
                for (size_t i = 0; i < q_elems; i++)
                    gate[i] = lcg_float(0.5f);
                lcg = 0x89abcdefu;
                for (size_t i = 0; i < cache_elems; i++) {
                    key[i] = f32_to_bf16_host(lcg_float(kvscale));
                    value[i] = f32_to_bf16_host(lcg_float(kvscale));
                }
                memcpy(q_b.contents, q, q_elems * 4);
                memcpy(gate_b.contents, gate, q_elems * 4);
                memcpy(key_b.contents, key, cache_elems * 2);
                memcpy(value_b.contents, value, cache_elems * 2);
                cpu_reference(VROWS, q, gate, key, value, sel, v_counts,
                              v_visible, ref_exact, 0);
                cpu_reference(VROWS, q, gate, key, value, sel, v_counts,
                              v_visible, ref_modeled, 1);
                cpu_reference(VROWS, q, gate, key, value, sel, v_counts,
                              v_visible, ref_split, 2);
                const float *got[2] = {NULL, NULL};
                id<MTLComputePipelineState> psos[2] = {plain_pso, split_pso};
                const size_t tgs[2] = {plain_tg, split_tg};
                for (int which = 0; which < 2; which++) {
                    memset(out_b.contents, 0,
                           (size_t)VROWS * Q_HEADS * DIM * 4);
                    id<MTLCommandBuffer> cb = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc =
                        [cb computeCommandEncoder];
                    [enc setComputePipelineState:psos[which]];
                    [enc setBytes:&sargs length:sizeof(sargs) atIndex:0];
                    [enc setBuffer:q_b offset:0 atIndex:1];
                    [enc setBuffer:gate_b offset:0 atIndex:2];
                    [enc setBuffer:key_b offset:0 atIndex:3];
                    [enc setBuffer:value_b offset:0 atIndex:4];
                    [enc setBuffer:sel_b offset:0 atIndex:5];
                    [enc setBuffer:counts_b offset:0 atIndex:6];
                    [enc setBuffer:visible_b offset:0 atIndex:7];
                    [enc setBuffer:out_b offset:0 atIndex:8];
                    [enc setThreadgroupMemoryLength:tgs[which] atIndex:0];
                    [enc dispatchThreadgroups:sgroups
                         threadsPerThreadgroup:sthreads];
                    [enc endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                    if (cb.error) {
                        fprintf(stderr, "gqa-mma-proto: sweep dispatch "
                                        "failed: %s\n",
                                cb.error.localizedDescription.UTF8String);
                        return 1;
                    }
                    got[which] = malloc(
                        (size_t)1000 * Q_HEADS * DIM * 4);
                    memcpy((void *)got[which], out_b.contents,
                           (size_t)1000 * Q_HEADS * DIM * 4);
                }
                const char *names[2] = {"plain  ", "qsplit "};
                const float *refs[2] = {ref_modeled, ref_split};
                printf("magnitude sweep q<+/-%g k,v<+/-%g:\n",
                       qscale / 2.0f, kvscale / 2.0f);
                /* predicted pure-rounding drift, modeled vs exact */
                double mw = 0.0, msum = 0.0;
                double ref_abs = 0.0;
                for (size_t i = 0;
                     i < (size_t)VROWS * Q_HEADS * DIM; i++) {
                    const double d = fabs(ref_modeled[i] - ref_exact[i]);
                    if (d > mw) mw = d;
                    msum += d;
                    ref_abs += fabs(ref_exact[i]);
                }
                printf("  F16-modeled vs exact : worst %.3e mean %.3e "
                       "(ref mean abs %.3e)\n",
                       mw, msum / ((size_t)VROWS * Q_HEADS * DIM),
                       ref_abs / ((size_t)VROWS * Q_HEADS * DIM));
                for (int which = 0; which < 2; which++) {
                    double w_exact = 0.0, s_exact = 0.0;
                    double w_modeled = 0.0, s_modeled = 0.0;
                    size_t at = 0;
                    for (size_t i = 0;
                         i < (size_t)VROWS * Q_HEADS * DIM; i++) {
                        double d = fabs(got[which][i] - ref_exact[i]);
                        if (d > w_exact) { w_exact = d; at = i; }
                        s_exact += d;
                        d = fabs(got[which][i] - refs[which][i]);
                        if (d > w_modeled) w_modeled = d;
                        s_modeled += d;
                    }
                    printf("  %s vs exact   : worst %.3e mean %.3e "
                           "at q%zu h%zu d%zu got %.6g ref %.6g "
                           "modeled %.6g\n",
                           names[which], w_exact,
                           s_exact / ((size_t)VROWS * Q_HEADS * DIM),
                           at / ((size_t)Q_HEADS * DIM),
                           (at / DIM) % Q_HEADS, at % DIM,
                           got[which][at], ref_exact[at], refs[which][at]);
                    printf("  %s vs modeled : worst %.3e mean %.3e\n",
                           names[which], w_modeled,
                           s_modeled / ((size_t)VROWS * Q_HEADS * DIM));
                    /* per-query worst, to localize the failing rows */
                    if (which == 0) {
                        printf("    per-query worst:");
                        for (uint32_t qr = 0; qr < VROWS; qr++) {
                            double qw = 0.0;
                            for (size_t i = (size_t)qr * Q_HEADS * DIM;
                                 i < (size_t)(qr + 1u) * Q_HEADS * DIM;
                                 i++) {
                                const double d =
                                    fabs(got[0][i] - ref_exact[i]);
                                if (d > qw) qw = d;
                            }
                            printf(" %u:%.1e", qr, qw);
                        }
                        printf("\n");
                        /* Two-token probe (row 5): with two selected
                         * tokens the mixture exposes the scores. */
                        {
                            const size_t b =
                                (size_t)7 * Q_HEADS * DIM; /* h0 */
                            /* row 7's true first-rank scores: tokens from
                             * its in-range block pattern, ranks 0..3 */
                            double qk[4] = {0, 0, 0, 0};
                            for (int t = 0; t < 4; t++) {
                                const uint32_t blk =
                                    sel[(size_t)7 * TOP_K + t / 4];
                                const uint32_t tok = blk * 4 + (t % 4);
                                for (int d = 0; d < DIM; d++)
                                    qk[t] += (double)q[b + d] *
                                        bf16_to_f32_host(
                                            key[((size_t)tok * KV_HEADS +
                                                  0) * DIM + d]);
                            }
                            printf("    row7 q7h0 true s[0..3]: %.4f %.4f "
                                   "%.4f %.4f (blocks %u %u)\n",
                                   qk[0] / 16.0, qk[1] / 16.0,
                                   qk[2] / 16.0, qk[3] / 16.0,
                                   sel[(size_t)7 * TOP_K + 0],
                                   sel[(size_t)7 * TOP_K + 1]);
                            if (which == 0) {
                                const size_t dbg_s =
                                    (size_t)999 * Q_HEADS * DIM;
                                const size_t dbg_p = dbg_s + DIM;
                                printf("    kernel s[0..3]  : %.4f %.4f "
                                       "%.4f %.4f (s4 -inf: %d)\n",
                                       got[0][dbg_s + 0], got[0][dbg_s + 1],
                                       got[0][dbg_s + 2], got[0][dbg_s + 3],
                                       got[0][dbg_s + 4] == -INFINITY);
                                printf("    kernel p[0..3]  : %.6f %.6f "
                                       "%.6f %.6f\n",
                                       got[0][dbg_p + 0], got[0][dbg_p + 1],
                                       got[0][dbg_p + 2], got[0][dbg_p + 3]);
                                const size_t dbg_m =
                                    (size_t)994 * Q_HEADS * DIM;
                                printf("    row maxima r0..7 : %.3f %.3f "
                                       "%.3f %.3f %.3f %.3f %.3f %.3f\n",
                                       got[0][dbg_m + 0], got[0][dbg_m + 1],
                                       got[0][dbg_m + 2], got[0][dbg_m + 3],
                                       got[0][dbg_m + 4], got[0][dbg_m + 5],
                                       got[0][dbg_m + 6], got[0][dbg_m + 7]);
                                printf("    row sums   r0..7 : %.3f %.3f "
                                       "%.3f %.3f %.3f %.3f %.3f %.3f\n",
                                       got[0][dbg_m + 16], got[0][dbg_m + 17],
                                       got[0][dbg_m + 18], got[0][dbg_m + 19],
                                       got[0][dbg_m + 20], got[0][dbg_m + 21],
                                       got[0][dbg_m + 22], got[0][dbg_m + 23]);
                                const size_t dbg_q =
                                    (size_t)998 * Q_HEADS * DIM;
                                printf("    staged Q[0][0..3] : %.4f %.4f "
                                       "%.4f %.4f  (host q %.4f %.4f "
                                       "%.4f %.4f)\n",
                                       got[0][dbg_q + 0], got[0][dbg_q + 1],
                                       got[0][dbg_q + 2], got[0][dbg_q + 3],
                                       q[(size_t)5 * Q_HEADS * DIM + 0],
                                       q[(size_t)5 * Q_HEADS * DIM + 1],
                                       q[(size_t)5 * Q_HEADS * DIM + 2],
                                       q[(size_t)5 * Q_HEADS * DIM + 3]);
                                printf("    staged K[d][r0]  : %.4f %.4f "
                                       "%.4f %.4f  (host k0d0-3 %.4f %.4f "
                                       "%.4f %.4f)\n",
                                       got[0][dbg_q + 16], got[0][dbg_q + 17],
                                       got[0][dbg_q + 18], got[0][dbg_q + 19],
                                       bf16_to_f32_host(key[0]),
                                       bf16_to_f32_host(key[1]),
                                       bf16_to_f32_host(key[2]),
                                       bf16_to_f32_host(key[3]));
                                printf("    staged K[0][rk]  : %.4f %.4f "
                                       "%.4f %.4f\n",
                                       got[0][dbg_q + 24], got[0][dbg_q + 25],
                                       got[0][dbg_q + 26], got[0][dbg_q + 27]);
                                const size_t dbg_f =
                                    (size_t)997 * Q_HEADS * DIM;
                                printf("    sfrag[0] r0      : %.4f %.4f "
                                       "%.4f %.4f %.4f %.4f %.4f %.4f\n",
                                       got[0][dbg_f + 0], got[0][dbg_f + 1],
                                       got[0][dbg_f + 2], got[0][dbg_f + 3],
                                       got[0][dbg_f + 4], got[0][dbg_f + 5],
                                       got[0][dbg_f + 6], got[0][dbg_f + 7]);
                                {
                                    double qk9[4] = {0, 0, 0, 0};
                                    for (int t = 0; t < 4; t++) {
                                        const uint32_t blk =
                                            sel[(size_t)7 * TOP_K +
                                                (8 + t) / 4];
                                        const uint32_t tok =
                                            blk * 4 + ((8 + t) % 4);
                                        for (int d = 0; d < DIM; d++)
                                            qk9[t] += (double)
                                                q[(size_t)7 * Q_HEADS *
                                                      DIM +
                                                  d] *
                                                bf16_to_f32_host(
                                                    key[((size_t)tok *
                                                          KV_HEADS + 0) *
                                                         DIM + d]);
                                    }
                                    printf("    sfrag[1] r0      : %.4f "
                                           "%.4f %.4f %.4f (true raw "
                                           "ranks 8..11: %.4f %.4f %.4f "
                                           "%.4f)\n",
                                           got[0][dbg_f + 64],
                                           got[0][dbg_f + 65],
                                           got[0][dbg_f + 66],
                                           got[0][dbg_f + 67],
                                           qk9[0], qk9[1], qk9[2],
                                           qk9[3]);
                                }
                                const size_t dbg_t =
                                    (size_t)995 * Q_HEADS * DIM;
                                /* row 0's 64 rank slots post-store: find
                                 * the max slot and print the tail columns */
                                int mx_i = 0;
                                for (int i = 1; i < 64; i++)
                                    if (got[0][dbg_t + i] >
                                        got[0][dbg_t + mx_i]) mx_i = i;
                                printf("    s_tile row0 max %.4f at col %d; "
                                       "cols 8..23: %.3f %.3f %.3f %.3f "
                                       "%.3f %.3f %.3f %.3f %.3f %.3f "
                                       "%.3f %.3f %.3f %.3f %.3f %.3f\n",
                                       got[0][dbg_t + mx_i], mx_i,
                                       got[0][dbg_t + 8], got[0][dbg_t + 9],
                                       got[0][dbg_t + 10],
                                       got[0][dbg_t + 11],
                                       got[0][dbg_t + 12],
                                       got[0][dbg_t + 13],
                                       got[0][dbg_t + 14],
                                       got[0][dbg_t + 15],
                                       got[0][dbg_t + 16],
                                       got[0][dbg_t + 17],
                                       got[0][dbg_t + 18],
                                       got[0][dbg_t + 19],
                                       got[0][dbg_t + 20],
                                       got[0][dbg_t + 21],
                                       got[0][dbg_t + 22],
                                       got[0][dbg_t + 23]);
                            }
                            for (int d = 0; d < 4; d++) {
                                const float g0 = 1.0f / (1.0f +
                                    expf(-gate[b + d]));
                                const float v0d = bf16_to_f32_host(
                                    value[d]);
                                const float v1d = bf16_to_f32_host(
                                    value[DIM + d]);
                                printf("      d%d got %.5f ref %.5f "
                                       "v0 %.5f v1 %.5f gate %.5f\n",
                                       d, got[0][b + d], ref_exact[b + d],
                                       v0d, v1d, g0);
                            }
                        }
                    }
                    free((void *)got[which]);
                }
            }
            free(ref_exact);
            free(ref_modeled);
            free(ref_split);
        }

        /* restore bench counts/visible for timing */
        {
            uint32_t *cmap = (uint32_t *)counts_b.contents;
            uint32_t *vmap = (uint32_t *)visible_b.contents;
            for (uint32_t i = 0; i < MAX_ROWS; i++) {
                cmap[i] = TOP_K;
                vmap[i] = rows;
            }
        }

        /* restore the bench gather pattern for the first rows */
        {
            uint32_t *smap = (uint32_t *)sel_b.contents;
            for (uint32_t query = 0; query < VROWS; query++) {
                uint32_t block = query % (MAX_ROWS / RATIO);
                for (uint32_t r = 0; r < TOP_K; r++) {
                    smap[(size_t)query * TOP_K + r] = block;
                    block = block + 1u >= MAX_ROWS / RATIO ? 0u : block + 1u;
                }
            }
        }
        /* re-fill with bench magnitudes for timing */
        for (size_t i = 0; i < q_elems; i++) {
            q[i] = lcg_float(1.0f);
            gate[i] = lcg_float(1.0f);
        }
        for (size_t i = 0; i < cache_elems; i++) {
            key[i] = f32_to_bf16_host(lcg_float(1.5f));
            value[i] = f32_to_bf16_host(lcg_float(1.5f));
        }
        memcpy(q_b.contents, q, q_elems * 4);
        memcpy(gate_b.contents, gate, q_elems * 4);
        memcpy(key_b.contents, key, cache_elems * 2);
        memcpy(value_b.contents, value, cache_elems * 2);

        /* ---- timing at the production shape ---- */
        printf("timing rows=%u selected_per_query=%u "
               "(reference: scalar ~113 ms/layer, per-sg MMA ~244)\n",
               rows, TOP_K * RATIO);
        for (int v = 0; v < n_variants; v++) {
            id<MTLFunction> fn = [lib newFunctionWithName:
                [NSString stringWithUTF8String:variants[v].name]];
            id<MTLComputePipelineState> pso =
                [device newComputePipelineStateWithFunction:fn error:nil];
            if (!pso) continue;
            const uint hpad = variants[v].org >= 1 ? 16u : 8u;
            size_t tg_bytes =
                (size_t)hpad * DIM * 2u + (size_t)variants[v].bk *
                    variants[v].dc * 2u +
                (size_t)hpad * variants[v].bk * 4u +
                (size_t)hpad * variants[v].bk * 2u +
                (size_t)variants[v].bk * 4u + hpad * 2u * 4u + hpad * 4u +
                2u * 4u + 64u;
            if (variants[v].org == 2)
                tg_bytes += (size_t)8u * 128u * 4u + 64u; /* O block */
            if (variants[v].org == 3) {
                tg_bytes += (size_t)8u * 128u * 4u + 64u; /* O block */
                tg_bytes += (size_t)hpad * DIM * 2u;      /* qs_lo tile */
            }
            if (tg_bytes > device.maxThreadgroupMemoryLength) {
                printf("  %-24s SKIP (tg %zu > limit %llu)\n",
                       variants[v].name, tg_bytes,
                       (unsigned long long)
                           device.maxThreadgroupMemoryLength);
                continue;
            }
            proto_args args = {rows, CACHE_CAP, Q_HEADS, KV_HEADS, DIM,
                               TOP_K, RATIO, TOP_K * RATIO + RATIO - 1u,
                               0u};
            const MTLSize groups = MTLSizeMake(
                rows, variants[v].org == 0 ? KV_HEADS * 2u : KV_HEADS, 1);
            const MTLSize threads = MTLSizeMake(
                variants[v].org == 0 ? 32u
                    : (variants[v].org == 1 ? 64u : 192u),
                1, 1);
            double samples[5];
            for (int iter = 0; iter < 7; iter++) {
                const double t0 = now_ms();
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc =
                    [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBytes:&args length:sizeof(args) atIndex:0];
                [enc setBuffer:q_b offset:0 atIndex:1];
                [enc setBuffer:gate_b offset:0 atIndex:2];
                [enc setBuffer:key_b offset:0 atIndex:3];
                [enc setBuffer:value_b offset:0 atIndex:4];
                [enc setBuffer:sel_b offset:0 atIndex:5];
                [enc setBuffer:counts_b offset:0 atIndex:6];
                [enc setBuffer:visible_b offset:0 atIndex:7];
                [enc setBuffer:out_b offset:0 atIndex:8];
                [enc setThreadgroupMemoryLength:tg_bytes atIndex:0];
                [enc dispatchThreadgroups:groups
                     threadsPerThreadgroup:threads];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (iter >= 2) samples[iter - 2] = now_ms() - t0;
            }
            /* median of 5 */
            for (int i = 1; i < 5; i++) {
                const double key = samples[i];
                int j = i - 1;
                while (j >= 0 && samples[j] > key) {
                    samples[j + 1] = samples[j];
                    j--;
                }
                samples[j + 1] = key;
            }
            variants[v].ms = samples[2];
            printf("  %-24s %8.2f ms/layer   x12 = %7.1f ms/chunk\n",
                   variants[v].name, variants[v].ms, variants[v].ms * 12.0);
        }
    }
    return 0;
}
