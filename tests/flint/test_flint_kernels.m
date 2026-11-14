/*
 * Model-free unit tests for metal/flint_prefill.metal (docs/RECURSIVE_FLINT.md,
 * Verification: kernel tests).  Compiles the shader at run time with strict
 * math, runs every kernel on random data and compares against CPU references
 * that follow MLX's arithmetic (bf16 rounding points, f32 accumulation).
 * Tolerances allow the summation-order differences of the MMA tiles.
 *
 *   make test-flint-kernels          (fork.mk)
 *   tests/flint/test_flint_kernels [metal/flint_prefill.metal]
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static id<MTLDevice> dev;
static id<MTLCommandQueue> queue;
static id<MTLLibrary> lib;
static int failures;
static id<MTLBuffer> indirect_resource;   /* buffer reached through GPU addresses: must be made resident */

/* ---- bf16 helpers (round to nearest even, as Metal and MLX) ---- */
static float bf2f(uint16_t b) { union { uint32_t u; float f; } v; v.u = (uint32_t)b << 16; return v.f; }
static uint16_t f2bf(float f) {
    union { uint32_t u; float f; } v; v.f = f;
    if ((v.u & 0x7f800000u) == 0x7f800000u && (v.u & 0x007fffffu)) return (uint16_t)((v.u >> 16) | 0x40u);
    const uint32_t r = 0x7fffu + ((v.u >> 16) & 1u);
    return (uint16_t)((v.u + r) >> 16);
}
static float rbf(float f) { return bf2f(f2bf(f)); }

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint32_t rnd32(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 16);
}
static float rndn(void) {   /* approx normal */
    float s = 0; for (int i = 0; i < 4; i++) s += (float)(rnd32() & 0xffff) / 65535.0f - 0.5f;
    return s * 1.7f;
}
static float rndu(void) { return (float)(rnd32() & 0xffffff) / 16777215.0f; }

static id<MTLBuffer> buf(size_t bytes) {
    id<MTLBuffer> b = [dev newBufferWithLength:bytes ? bytes : 16 options:MTLResourceStorageModeShared];
    memset(b.contents, 0, bytes ? bytes : 16);
    return b;
}
static id<MTLBuffer> buf_bf16_random(size_t n, float scale) {
    id<MTLBuffer> b = buf(n * 2);
    uint16_t *p = b.contents;
    for (size_t i = 0; i < n; i++) p[i] = f2bf(rndn() * scale);
    return b;
}
static id<MTLComputePipelineState> pso(const char *name) {
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    NSError *err = nil;
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!p) { fprintf(stderr, "pipeline %s: %s\n", name, err.localizedDescription.UTF8String); exit(1); }
    return p;
}

/* Runs one kernel.  bufs: NULL-terminated list; indices are positional;
 * buffer off_index (if >= 0) is bound at byte offset off. */
static void run_off(const char *name, NSArray<id<MTLBuffer>> *bufs, int off_index, NSUInteger off, MTLSize grid, MTLSize tg,
                    id<MTLBuffer> indirect, NSUInteger indirect_off) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso(name)];
        for (NSUInteger i = 0; i < bufs.count; i++) [enc setBuffer:bufs[i] offset:((int)i == off_index ? off : 0) atIndex:i];
        if (indirect_resource) [enc useResource:indirect_resource usage:MTLResourceUsageRead];
        if (indirect) [enc dispatchThreadgroupsWithIndirectBuffer:indirect indirectBufferOffset:indirect_off threadsPerThreadgroup:tg];
        else [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) { fprintf(stderr, "%s: %s\n", name, cb.error.localizedDescription.UTF8String); exit(1); }
    }
}
static void run(const char *name, NSArray<id<MTLBuffer>> *bufs, MTLSize grid, MTLSize tg, id<MTLBuffer> indirect, NSUInteger indirect_off) {
    run_off(name, bufs, -1, 0, grid, tg, indirect, indirect_off);
}

typedef struct { uint32_t M, N, K, lda, ldc, ldr; } GemmArgs;
typedef struct { uint32_t M, q_pos0, key_pos0, n_keys, kv_head_stride, window, ldq, ldo; } AttnArgs;
typedef struct { uint32_t M, pos0, rotary, dst_row0, dst_head_stride; } HeadArgs;
typedef struct { uint32_t n_rows, src_head_stride, dst_head_stride, first_pos, ring_initial, ring_side, linear_row0; } RingArgs;
typedef struct { uint32_t expert, begin, count, pad; } Tile;

static id<MTLBuffer> buf_struct(const void *p, size_t n) { id<MTLBuffer> b = buf(n); memcpy(b.contents, p, n); return b; }

/* Compare bf16 outputs against float references: max abs diff, ulp-tolerant. */
static void check_bf16(const char *what, const uint16_t *got, const float *ref, size_t n, float abs_tol, float rel_tol) {
    double maxd = 0; size_t exact = 0, bad = 0, worst = 0;
    for (size_t i = 0; i < n; i++) {
        const float g = bf2f(got[i]);
        const float d = fabsf(g - ref[i]);
        if (g == rbf(ref[i])) exact++;
        if (d > maxd) { maxd = d; worst = i; }
        if (d > abs_tol + rel_tol * fabsf(ref[i])) bad++;
    }
    printf("  %-34s n=%-8zu exact %5.1f%%  max|d| %.3g (ref %.4g at %zu)  %s\n", what, n, 100.0 * exact / n, maxd,
           ref[worst], worst, bad ? "FAIL" : "ok");
    if (bad) { failures++; printf("    %zu elements outside tolerance (abs %.3g rel %.3g)\n", bad, abs_tol, rel_tol); }
    if (bad && getenv("FLINT_TEST_MAP") && n >= 64 * 192) {   /* mismatch map of the first rows (row-major, 192 wide) */
        for (size_t r = 0; r < 16; r++) {
            printf("    ");
            for (size_t c = 0; c < 192; c++) { const size_t i = r * 192 + c; putchar(fabsf(bf2f(got[i]) - ref[i]) > abs_tol + rel_tol * fabsf(ref[i]) ? 'X' : '.'); }
            putchar('\n');
        }
    }
}

/* 4-bit affine matrix [N][K] on the CPU: nibbles + bf16 scales/biases, and its single-rounded bf16 dequant. */
typedef struct { uint8_t *w; uint16_t *s, *b; uint32_t N, K; } Q4;
static Q4 q4_random(uint32_t N, uint32_t K) {
    Q4 m = { malloc((size_t)N * K / 2), malloc((size_t)N * K / 64 * 2), malloc((size_t)N * K / 64 * 2), N, K };
    for (size_t i = 0; i < (size_t)N * K / 2; i++) m.w[i] = (uint8_t)(rnd32() & 0xff);
    for (size_t i = 0; i < (size_t)N * K / 64; i++) { m.s[i] = f2bf(0.01f + 0.04f * rndu()); m.b[i] = f2bf(-0.3f + 0.6f * rndu() - 8.0f * bf2f(m.s[i])); }
    return m;
}
static float q4_at(const Q4 *m, uint32_t n, uint32_t k) {   /* bf16(s*q+b) */
    const uint8_t byte = m->w[(size_t)n * m->K / 2 + k / 2];
    const uint32_t q = (k & 1) ? (byte >> 4) : (byte & 15);
    const size_t gi = (size_t)n * (m->K / 64) + k / 64;
    return rbf(bf2f(m->s[gi]) * (float)q + bf2f(m->b[gi]));
}
static void q4_upload(const Q4 *m, id<MTLBuffer> *w, id<MTLBuffer> *s, id<MTLBuffer> *b) {
    *w = buf_struct(m->w, (size_t)m->N * m->K / 2);
    *s = buf_struct(m->s, (size_t)m->N * m->K / 64 * 2);
    *b = buf_struct(m->b, (size_t)m->N * m->K / 64 * 2);
}
/* MLX nn.silu over bf16: bf16(x * bf16(sigmoid(x))) with sigmoid via the tail form. */
static uint16_t *silu_table(void) {
    uint16_t *t = malloc(65536 * 2);
    for (uint32_t i = 0; i < 65536; i++) {
        const float x = bf2f((uint16_t)i);
        if (isnan(x)) { t[i] = (uint16_t)i; continue; }
        const float tail = 1.0f / (1.0f + expf(fabsf(x)));
        const float sig = rbf(x < 0 ? tail : 1.0f - tail);
        t[i] = f2bf(x * sig);
    }
    return t;
}

static void test_gemm(void) {
    printf("gemm\n");
    const uint32_t M = 100, N = 192, K = 256;
    GemmArgs a = { M, N, K, K, N, N };
    id<MTLBuffer> A = buf_bf16_random((size_t)M * K, 1.0f), C = buf((size_t)M * N * 2), R = buf_bf16_random((size_t)M * N, 1.0f);
    id<MTLBuffer> args = buf_struct(&a, sizeof a), dummy = buf(16);
    Q4 W = q4_random(N, K), W2 = q4_random(N, K);
    id<MTLBuffer> w, s, b, w2, s2, b2; q4_upload(&W, &w, &s, &b); q4_upload(&W2, &w2, &s2, &b2);
    uint16_t *silu = silu_table();
    id<MTLBuffer> siluB = buf_struct(silu, 65536 * 2);
    id<MTLBuffer> Y6 = buf_bf16_random((size_t)M * 6 * N, 0.5f);
    const uint16_t *Ap = A.contents, *Rp = R.contents, *Yp = Y6.contents;
    float *acc = malloc((size_t)M * N * 4), *acc2 = malloc((size_t)M * N * 4), *ref = malloc((size_t)M * N * 4);
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < N; n++) {
            double t = 0, t2 = 0;
            for (uint32_t k = 0; k < K; k++) { const double x = bf2f(Ap[(size_t)m * K + k]); t += x * q4_at(&W, n, k); t2 += x * q4_at(&W2, n, k); }
            acc[(size_t)m * N + n] = (float)t; acc2[(size_t)m * N + n] = (float)t2;
        }
    MTLSize grid = MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1), tg = MTLSizeMake(128, 1, 1);
    /* store */
    run("flint_gemm_q4_store", @[A, w, s, b, C, args, dummy, dummy, dummy, dummy, dummy, dummy], grid, tg, nil, 0);
    for (size_t i = 0; i < (size_t)M * N; i++) ref[i] = rbf(acc[i]);
    check_bf16("q4 store", C.contents, ref, (size_t)M * N, 1e-6f, 1.0f / 128);
    /* residual */
    run("flint_gemm_q4_residual", @[A, w, s, b, C, args, R, dummy, dummy, dummy, dummy, dummy], grid, tg, nil, 0);
    for (size_t i = 0; i < (size_t)M * N; i++) ref[i] = rbf(bf2f(Rp[i]) + rbf(acc[i]));
    check_bf16("q4 residual", C.contents, ref, (size_t)M * N, 1e-6f, 1.0f / 64);
    /* glu */
    run("flint_gemm_q4_glu", @[A, w, s, b, C, args, dummy, w2, s2, b2, siluB, dummy], grid, tg, nil, 0);
    for (size_t i = 0; i < (size_t)M * N; i++) ref[i] = rbf(bf2f(silu[f2bf(acc[i])]) * rbf(acc2[i]));
    check_bf16("q4 glu", C.contents, ref, (size_t)M * N, 1e-3f, 1.0f / 32);
    /* combine */
    run("flint_gemm_q4_combine", @[A, w, s, b, C, args, R, dummy, dummy, dummy, dummy, Y6], grid, tg, nil, 0);
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < N; n++) {
            float sum = bf2f(Yp[((size_t)m * 6) * N + n]);
            for (int r = 1; r < 6; r++) sum = rbf(bf2f(Yp[((size_t)m * 6 + r) * N + n]) + sum);
            const float mlp = rbf(rbf(sum) + rbf(acc[(size_t)m * N + n]));
            ref[(size_t)m * N + n] = rbf(bf2f(Rp[(size_t)m * N + n]) + mlp);
        }
    check_bf16("q4 combine", C.contents, ref, (size_t)M * N, 1e-6f, 1.0f / 64);
    /* bf16 weights */
    id<MTLBuffer> Wb = buf_bf16_random((size_t)N * K, 0.05f);
    const uint16_t *Wbp = Wb.contents;
    run("flint_gemm_bf16_store", @[A, Wb, dummy, dummy, C, args, dummy, dummy, dummy, dummy, dummy, dummy], grid, tg, nil, 0);
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < N; n++) {
            double t = 0;
            for (uint32_t k = 0; k < K; k++) t += (double)bf2f(Ap[(size_t)m * K + k]) * bf2f(Wbp[(size_t)n * K + k]);
            ref[(size_t)m * N + n] = rbf((float)t);
        }
    check_bf16("bf16 store", C.contents, ref, (size_t)M * N, 1e-6f, 1.0f / 128);
    free(acc); free(acc2); free(ref); free(silu);
}

static void test_moe(void) {
    printf("routing and experts\n");
    const uint32_t M = 100, E = 256, NPACK = 8;
    id<MTLBuffer> logits = buf_bf16_random((size_t)M * E, 2.0f);
    /* force expert 0 on every token (100 rows: a 64-row tile and a 64-row
     * remainder), expert 1 on 24 (a 32-row tile), experts 2 and 3 on 12 and
     * 9 (16-row tiles with the random picks); the rest of the routing is
     * random and lands in 8-row tiles */
    {
        uint16_t *lg0 = logits.contents;
        for (uint32_t t = 0; t < M; t++) {
            lg0[(size_t)t * E + 0] = f2bf(10.0f);
            if (t < 24) lg0[(size_t)t * E + 1] = f2bf(10.0f);
            if (t < 12) lg0[(size_t)t * E + 2] = f2bf(10.0f);
            if (t < 9) lg0[(size_t)t * E + 3] = f2bf(10.0f);
        }
    }
    id<MTLBuffer> ids = buf((size_t)M * 6 * 4), wts = buf((size_t)M * 6 * 2), counts = buf(E * 4), starts = buf(257 * 4);
    /* tile tables: 64-row tiles at 0 (capacity CAP64), then one 32-, 16- and 8-row table of 256 */
    const uint32_t CAP64 = 256;
    id<MTLBuffer> tiles = buf((CAP64 + 768) * sizeof(Tile)), dispatch = buf(96), cursors = buf(E * 4), rows = buf((size_t)M * 6 * 4);
    uint32_t Mv = M, cfg[2] = {CAP64, 1};
    id<MTLBuffer> Mb = buf_struct(&Mv, 4), cfgB = buf_struct(cfg, 8);
    run("flint_route_topk", @[logits, ids, wts, counts, Mb], MTLSizeMake((M + 3) / 4, 1, 1), MTLSizeMake(128, 1, 1), nil, 0);
    run("flint_route_offsets", @[counts, starts, tiles, dispatch, cursors, cfgB], MTLSizeMake(1, 1, 1), MTLSizeMake(256, 1, 1), nil, 0);
    run("flint_route_scatter", @[ids, starts, cursors, rows, Mb], MTLSizeMake((M * 6 + 255) / 256, 1, 1), MTLSizeMake(256, 1, 1), nil, 0);
    /* CPU routing reference */
    const uint16_t *lg = logits.contents;
    uint32_t *ref_ids = malloc((size_t)M * 6 * 4); float *ref_w = malloc((size_t)M * 6 * 4);
    uint32_t ref_counts[256] = {0};
    for (uint32_t t = 0; t < M; t++) {
        float p[256]; int used[256] = {0};
        for (uint32_t e = 0; e < E; e++) { const float a = bf2f(lg[(size_t)t * E + e]); const float tail = 1.0f / (1.0f + expf(fabsf(a))); p[e] = a < 0 ? tail : 1.0f - tail; }
        float sum = 0;
        for (int r = 0; r < 6; r++) {
            int best = -1;
            for (uint32_t e = 0; e < E; e++) if (!used[e] && (best < 0 || p[e] > p[best])) best = (int)e;
            used[best] = 1; ref_ids[t * 6 + r] = (uint32_t)best; ref_w[t * 6 + r] = p[best]; sum += p[best]; ref_counts[best]++;
        }
        sum += 1e-20f;
        for (int r = 0; r < 6; r++) ref_w[t * 6 + r] = rbf(ref_w[t * 6 + r] / sum);
    }
    const uint32_t *gid = ids.contents; const uint16_t *gw = wts.contents; const uint32_t *gc = counts.contents, *gs = starts.contents, *gr = rows.contents;
    size_t bad = 0;
    for (uint32_t i = 0; i < M * 6; i++) { if (gid[i] != ref_ids[i]) bad++; if (fabsf(bf2f(gw[i]) - ref_w[i]) > 1.0f / 128 * ref_w[i] + 1e-6f) bad++; }
    for (uint32_t e = 0; e < E; e++) if (gc[e] != 0 || gs[e + 1] - gs[e] != ref_counts[e]) bad++;   /* offsets zero counts for the next layer */
    uint32_t acc = 0; for (uint32_t e = 0; e < E; e++) { if (gs[e] != acc) bad++; acc += ref_counts[e]; } if (gs[256] != M * 6) bad++;
    /* rows: each entry once, inside its expert's segment */
    uint8_t *seen = calloc(M * 8, 1);
    for (uint32_t e = 0; e < E; e++)
        for (uint32_t i = gs[e]; i < gs[e + 1]; i++) {
            const uint32_t entry = gr[i], t = entry >> 3, slot = entry & 7;
            if (t >= M || slot >= 6 || gid[t * 6 + slot] != e || seen[entry]) bad++;
            seen[entry] = 1;
        }
    /* tiles: per height, the expected count; every row of every expert covered exactly once */
    const Tile *tl = tiles.contents; const uint32_t *dp = dispatch.contents;
    const uint32_t hbase[4] = {0, CAP64, CAP64 + 256, CAP64 + 512}, height[4] = {64, 32, 16, 8};
    uint32_t ntiles[4] = {0, 0, 0, 0}, ntotal = 0;
    for (uint32_t e = 0; e < E; e++) {
        const uint32_t c = ref_counts[e], rem = c % 64;
        ntiles[0] += c / 64;
        if (rem) ntiles[rem > 32 ? 0 : rem > 16 ? 1 : rem > 8 ? 2 : 3]++;
    }
    uint8_t *covered = calloc(M * 6, 1);
    for (int h = 0; h < 4; h++) {
        ntotal += ntiles[h];
        if (dp[h * 3] != ntiles[h] || dp[h * 3 + 1] != 8 || dp[12 + h * 3] != ntiles[h] || dp[12 + h * 3 + 1] != 32) bad++;
        for (uint32_t i = 0; i < ntiles[h]; i++) {
            const Tile t = tl[hbase[h] + i];
            if (t.count == 0 || t.count > height[h] || t.begin < gs[t.expert] || t.begin + t.count > gs[t.expert + 1]) { bad++; continue; }
            for (uint32_t r = t.begin; r < t.begin + t.count; r++) { if (covered[r]) bad++; covered[r] = 1; }
        }
    }
    for (uint32_t r = 0; r < M * 6; r++) if (!covered[r]) bad++;
    free(covered);
    printf("  %-34s %s (%u tiles: %u x 64, %u x 32, %u x 16, %u x 8)\n", "topk/offsets/scatter", bad ? "FAIL" : "ok", ntotal, ntiles[0], ntiles[1], ntiles[2], ntiles[3]);
    if (bad) failures++;

    /* experts: 8 fake packs, table entries point to pack e % 8 */
    const size_t PACK = 1769472;
    id<MTLBuffer> packs = buf(PACK * NPACK);
    uint8_t *pk = packs.contents;
    Q4 gate[8], up[8], down[8];
    for (uint32_t e = 0; e < NPACK; e++) {
        gate[e] = q4_random(512, 2048); up[e] = q4_random(512, 2048); down[e] = q4_random(2048, 512);
        uint8_t *p = pk + e * PACK;
        memcpy(p + 0, gate[e].w, 524288); memcpy(p + 524288, gate[e].s, 32768); memcpy(p + 557056, gate[e].b, 32768);
        memcpy(p + 589824, up[e].w, 524288); memcpy(p + 1114112, up[e].s, 32768); memcpy(p + 1146880, up[e].b, 32768);
        memcpy(p + 1179648, down[e].w, 524288); memcpy(p + 1703936, down[e].s, 32768); memcpy(p + 1736704, down[e].b, 32768);
    }
    id<MTLBuffer> table = buf(E * 8);
    uint64_t *tb = table.contents;
    for (uint32_t e = 0; e < E; e++) tb[e] = [packs gpuAddress] + (uint64_t)(e % NPACK) * PACK;
    id<MTLBuffer> X = buf_bf16_random((size_t)M * 2048, 1.0f), act = buf((size_t)M * 6 * 512 * 2), y6 = buf((size_t)M * 6 * 2048 * 2);
    uint16_t *silu = silu_table();
    id<MTLBuffer> siluB = buf_struct(silu, 65536 * 2);
    indirect_resource = packs;
    static const char *const gateup_k[4] = {"flint_moe_gateup_64", "flint_moe_gateup_32", "flint_moe_gateup_16", "flint_moe_gateup_8"};
    static const char *const down_k[4] = {"flint_moe_down_64", "flint_moe_down_32", "flint_moe_down_16", "flint_moe_down_8"};
    for (int h = 0; h < 4; h++)
        run_off(gateup_k[h], @[X, rows, tiles, table, siluB, act], 2, hbase[h] * sizeof(Tile), MTLSizeMake(1, 1, 1), MTLSizeMake(128, 1, 1), dispatch, h * 12);
    for (int h = 0; h < 4; h++)
        run_off(down_k[h], @[act, rows, tiles, table, wts, y6], 2, hbase[h] * sizeof(Tile), MTLSizeMake(1, 1, 1), MTLSizeMake(128, 1, 1), dispatch, 48 + h * 12);
    indirect_resource = nil;
    const uint16_t *Xp = X.contents, *ap = act.contents, *yp = y6.contents;
    /* reference for a sample of sorted rows (full reference is slow) */
    float *ract = malloc(512 * 4), *rdown = malloc(2048 * 4);
    double maxa = 0, maxy = 0; size_t bada = 0, bady = 0, checked = 0;
    for (uint32_t pos = 0; pos < M * 6; pos += 7) {
        const uint32_t entry = gr[pos], t = entry >> 3, slot = entry & 7, e = gid[t * 6 + slot] % NPACK;
        for (uint32_t n = 0; n < 512; n++) {
            double g = 0, u = 0;
            for (uint32_t k = 0; k < 2048; k++) { const double x = bf2f(Xp[(size_t)t * 2048 + k]); g += x * q4_at(&gate[e], n, k); u += x * q4_at(&up[e], n, k); }
            ract[n] = rbf(bf2f(silu[f2bf((float)g)]) * rbf((float)u));
            const float got = bf2f(ap[(size_t)pos * 512 + n]);
            const double d = fabs(got - ract[n]); if (d > maxa) maxa = d;
            if (d > 2e-3 + fabs(ract[n]) / 32) bada++;
        }
        const float wgt = bf2f(gw[t * 6 + slot]);
        for (uint32_t n = 0; n < 2048; n++) {
            double dsum = 0;
            for (uint32_t k = 0; k < 512; k++) dsum += (double)bf2f(ap[(size_t)pos * 512 + k]) * q4_at(&down[e], n, k);   /* from the GPU act (tests down alone) */
            rdown[n] = rbf(rbf((float)dsum) * wgt);
            const float got = bf2f(yp[((size_t)t * 6 + slot) * 2048 + n]);
            const double d = fabs(got - rdown[n]); if (d > maxy) maxy = d;
            if (d > 1e-3 + fabs(rdown[n]) / 64) bady++;
        }
        checked++;
    }
    printf("  %-34s rows %zu max|d| %.3g %s\n", "moe gate/up GLU", checked, maxa, bada ? "FAIL" : "ok");
    printf("  %-34s rows %zu max|d| %.3g %s\n", "moe down * weight", checked, maxy, bady ? "FAIL" : "ok");
    if (bada || bady) failures++;
    free(silu); free(ract); free(rdown); free(seen); free(ref_ids); free(ref_w);
}

static void attention_case(const char *name, uint32_t M, uint32_t q_pos0, uint32_t n_prev, uint32_t window) {
    const uint32_t n_keys = n_prev + M, key_pos0 = q_pos0 - n_prev, cap_rows = (M + 63) / 64 * 64;
    AttnArgs a = { M, q_pos0, key_pos0, n_keys, n_keys * 128, window, 4096, 4096 };
    id<MTLBuffer> Q = buf_bf16_random((size_t)cap_rows * 4096, 1.0f), K = buf_bf16_random((size_t)4 * n_keys * 128, 1.0f);
    id<MTLBuffer> V = buf_bf16_random((size_t)4 * n_keys * 128, 1.0f), O = buf((size_t)cap_rows * 4096 * 2);
    id<MTLBuffer> sinks = buf_bf16_random(32, 2.0f), args = buf_struct(&a, sizeof a);
    run("flint_attention", @[Q, K, K, K, K, V, V, V, V, sinks, O, args], MTLSizeMake((M + 63) / 64, 32, 1), MTLSizeMake(256, 1, 1), nil, 0);
    const uint16_t *q = Q.contents, *k = K.contents, *v = V.contents, *o = O.contents, *sk = sinks.contents;
    double maxd = 0; size_t bad = 0, n = 0;
    double *sc = malloc(n_keys * sizeof(double));
    for (uint32_t i = 0; i < M; i += 3)
        for (uint32_t h = 0; h < 32; h += 5) {
            const uint32_t kvh = h / 8;
            const uint32_t qpos = q_pos0 + i;
            double m = bf2f(sk[h]);
            for (uint32_t j = 0; j < n_keys; j++) {
                const uint32_t kpos = key_pos0 + j;
                bool ok = kpos <= qpos && (!window || qpos - kpos < window);
                if (!ok) { sc[j] = -INFINITY; continue; }
                double s = 0;
                for (uint32_t d = 0; d < 128; d++) s += (double)bf2f(q[(size_t)i * 4096 + h * 128 + d]) * bf2f(k[((size_t)kvh * n_keys + j) * 128 + d]);
                sc[j] = s / sqrt(128.0);
                if (sc[j] > m) m = sc[j];
            }
            double l = exp((double)bf2f(sk[h]) - m);
            for (uint32_t j = 0; j < n_keys; j++) if (sc[j] != -INFINITY) l += exp(sc[j] - m);
            for (uint32_t d = 0; d < 128; d += 9) {
                double acc = 0;
                for (uint32_t j = 0; j < n_keys; j++) if (sc[j] != -INFINITY) acc += exp(sc[j] - m) * bf2f(v[((size_t)kvh * n_keys + j) * 128 + d]);
                const double ref = acc / l;
                const double got = bf2f(o[(size_t)i * 4096 + h * 128 + d]);
                const double dd = fabs(got - ref); if (dd > maxd) maxd = dd;
                if (dd > 0.02 + 0.02 * fabs(ref)) bad++;
                n++;
            }
        }
    printf("  %-34s n=%zu max|d| %.3g %s\n", name, n, maxd, bad ? "FAIL" : "ok");
    if (bad) failures++;
    free(sc);
}

static void test_attention(void) {
    printf("attention\n");
    attention_case("global causal M=100 pos0=37", 100, 37, 37, 0);
    attention_case("global causal M=130 pos0=0", 130, 0, 0, 0);
    attention_case("window 513 M=600 prev=512", 600, 1000, 512, 513);
    attention_case("window 513 M=300 prev=200", 300, 200, 200, 513);
}

static void test_small(void) {
    printf("norms, rope, embedding, ring copy, head row\n");
    /* head norm + rope */
    const uint32_t M = 5;
    id<MTLBuffer> q = buf_bf16_random((size_t)M * 4096, 1.0f), k = buf_bf16_random((size_t)M * 512, 1.0f), v = buf_bf16_random((size_t)M * 512, 1.0f);
    id<MTLBuffer> qw = buf_bf16_random(128, 0.5f), kw = buf_bf16_random(128, 0.5f);
    const uint32_t dst_rows = 40, dst_row0 = 7;
    id<MTLBuffer> kd = buf((size_t)4 * dst_rows * 128 * 2), vd = buf((size_t)4 * dst_rows * 128 * 2);
    uint16_t *q_in = malloc((size_t)M * 4096 * 2); memcpy(q_in, q.contents, (size_t)M * 4096 * 2);
    for (int rotary = 0; rotary < 2; rotary++) {
        memcpy(q.contents, q_in, (size_t)M * 4096 * 2);
        HeadArgs ha = { M, 1234, (uint32_t)rotary, dst_row0, dst_rows * 128 };
        id<MTLBuffer> args = buf_struct(&ha, sizeof ha);
        run("flint_head_norm_rope", @[q, k, v, qw, kw, kd, kd, vd, vd, args], MTLSizeMake((M * 40 + 7) / 8, 1, 1), MTLSizeMake(256, 1, 1), nil, 0);
        const uint16_t *qo = q.contents, *ko = kd.contents, *vo = vd.contents, *kp = k.contents, *vp = v.contents, *qwp = qw.contents, *kwp = kw.contents;
        double maxd = 0; size_t bad = 0, n = 0;
        for (uint32_t t = 0; t < M; t++)
            for (uint32_t h = 0; h < 36; h++) {
                const uint16_t *src = h < 32 ? q_in + (size_t)t * 4096 + h * 128 : kp + (size_t)t * 512 + (h - 32) * 128;
                const uint16_t *got = h < 32 ? qo + (size_t)t * 4096 + h * 128 : ko + (size_t)(h - 32) * dst_rows * 128 + (size_t)(dst_row0 + t) * 128;
                const uint16_t *w = h < 32 ? qwp : kwp;
                float ss = 0; for (int d = 0; d < 128; d++) ss += bf2f(src[d]) * bf2f(src[d]);
                const float inv = 1.0f / sqrtf(ss / 128.0f + 1e-6f);
                float ref[128];
                if (rotary) {
                    for (int pair = 0; pair < 64; pair++) {
                        const float a = rbf(bf2f(w[pair]) * rbf(bf2f(src[pair]) * inv)), b = rbf(bf2f(w[pair + 64]) * rbf(bf2f(src[pair + 64]) * inv));
                        const double theta = (double)(1234 + t) * exp2(-(double)pair / 64.0 * 13.287712379549449);
                        ref[pair] = (float)(a * cos(theta) - b * sin(theta)); ref[pair + 64] = (float)(a * sin(theta) + b * cos(theta));
                    }
                } else for (int d = 0; d < 128; d++) ref[d] = rbf(bf2f(w[d]) * rbf(bf2f(src[d]) * inv));
                for (int d = 0; d < 128; d++) { const double dd = fabs(bf2f(got[d]) - ref[d]); if (dd > maxd) maxd = dd; if (dd > 0.02 + fabs(ref[d]) / 64) bad++; n++; }
            }
        for (uint32_t t = 0; t < M; t++) for (int h = 0; h < 4; h++) for (int d = 0; d < 128; d++)
            if (vo[(size_t)h * dst_rows * 128 + (size_t)(dst_row0 + t) * 128 + d] != vp[(size_t)t * 512 + h * 128 + d]) bad++;
        printf("  %-34s n=%zu max|d| %.3g %s\n", rotary ? "head norm + rope, v copy" : "head norm (global), v copy", n, maxd, bad ? "FAIL" : "ok");
        if (bad) failures++;
    }
    free(q_in);
    /* rmsnorm */
    {
        const uint32_t rows = 9;
        id<MTLBuffer> x = buf_bf16_random((size_t)(rows + 3) * 2048, 1.0f), w = buf_bf16_random(2048, 0.5f), out = buf((size_t)(rows + 3) * 2048 * 2);
        uint32_t off = 3; id<MTLBuffer> offb = buf_struct(&off, 4);
        run("flint_rmsnorm", @[x, w, out, offb], MTLSizeMake(rows, 1, 1), MTLSizeMake(256, 1, 1), nil, 0);
        const uint16_t *xp = x.contents, *wp = w.contents, *op = out.contents;
        float *ref = malloc((size_t)rows * 2048 * 4);
        for (uint32_t r = 0; r < rows; r++) {
            double ss = 0; for (int i = 0; i < 2048; i++) { const double vv = bf2f(xp[(size_t)(r + off) * 2048 + i]); ss += vv * vv; }
            const float inv = (float)(1.0 / sqrt(ss / 2048.0 + 1e-6));
            for (int i = 0; i < 2048; i++) ref[(size_t)r * 2048 + i] = rbf(bf2f(wp[i]) * rbf(bf2f(xp[(size_t)(r + off) * 2048 + i]) * inv));
        }
        check_bf16("rmsnorm (row offset 3)", op + (size_t)off * 2048, ref, (size_t)rows * 2048, 1e-6f, 1.0f / 64);
        free(ref);
    }
    /* embedding */
    {
        const uint32_t V = 300, T = 6;
        id<MTLBuffer> W = buf((size_t)V * 2048), S = buf_bf16_random((size_t)V * 32, 0.01f), B = buf_bf16_random((size_t)V * 32, 0.2f), out = buf((size_t)T * 2048 * 2);
        uint8_t *wp = W.contents; for (size_t i = 0; i < (size_t)V * 2048; i++) wp[i] = (uint8_t)(rnd32() & 0xff);
        int toks[6] = { 0, 299, 17, 17, 150, 42 }; id<MTLBuffer> tb = buf_struct(toks, sizeof toks);
        run("flint_embed", @[tb, W, S, B, out], MTLSizeMake(2048 / 256, T, 1), MTLSizeMake(256, 1, 1), nil, 0);
        const uint16_t *sp = S.contents, *bp = B.contents, *op = out.contents;
        float *ref = malloc((size_t)T * 2048 * 4);
        for (uint32_t t = 0; t < T; t++) for (int i = 0; i < 2048; i++) {
            const size_t gi = (size_t)toks[t] * 32 + i / 64;
            ref[(size_t)t * 2048 + i] = rbf(bf2f(sp[gi]) * (float)wp[(size_t)toks[t] * 2048 + i] + bf2f(bp[gi]));   /* Metal contracts the bf16 multiply-add */
        }
        check_bf16("embedding (bf16 ops)", op, ref, (size_t)T * 2048, 0.0f, 0.0f);
        free(ref);
    }
    /* ring copy: scatter positions into a ring then gather them back */
    {
        const uint32_t n = 300, initial = 2000, first = 1700, win_rows = 600;
        id<MTLBuffer> Kw = buf((size_t)4 * win_rows * 128 * 2), Vw = buf((size_t)4 * win_rows * 128 * 2);
        id<MTLBuffer> Kr = buf((size_t)4 * 513 * 128 * 2), Vr = buf((size_t)4 * 513 * 128 * 2);
        id<MTLBuffer> Kg = buf((size_t)4 * win_rows * 128 * 2), Vg = buf((size_t)4 * win_rows * 128 * 2);
        uint16_t *kw = Kw.contents, *vw = Vw.contents;
        for (int h = 0; h < 4; h++) for (uint32_t r = 0; r < win_rows; r++) for (int d = 0; d < 128; d++) {
            kw[((size_t)h * win_rows + r) * 128 + d] = (uint16_t)(h * 7 + r + d * 3); vw[((size_t)h * win_rows + r) * 128 + d] = (uint16_t)(h + r * 2 + d);
        }
        RingArgs sc = { n, win_rows * 128, 513 * 128, first, initial, 1, 100 };   /* window rows 100.. -> ring slots of positions 1700.. */
        run("flint_ring_copy", @[Kw, Vw, Kr, Vr, buf_struct(&sc, sizeof sc)], MTLSizeMake((4 * n * 16 + 255) / 256, 1, 1), MTLSizeMake(256, 1, 1), nil, 0);
        RingArgs ga = { n, 513 * 128, win_rows * 128, first, initial, 0, 250 };   /* back into window rows 250.. */
        run("flint_ring_copy", @[Kr, Vr, Kg, Vg, buf_struct(&ga, sizeof ga)], MTLSizeMake((4 * n * 16 + 255) / 256, 1, 1), MTLSizeMake(256, 1, 1), nil, 0);
        const uint16_t *kr = Kr.contents, *kg = Kg.contents, *vg = Vg.contents;
        size_t bad = 0;
        for (int h = 0; h < 4; h++) for (uint32_t r = 0; r < n; r++) {
            const uint32_t p = first + r, slot = (p + 513 * 65536 - initial) % 513;
            for (int d = 0; d < 128; d++) {
                if (kr[((size_t)h * 513 + slot) * 128 + d] != kw[((size_t)h * win_rows + 100 + r) * 128 + d]) bad++;
                if (kg[((size_t)h * win_rows + 250 + r) * 128 + d] != kw[((size_t)h * win_rows + 100 + r) * 128 + d]) bad++;
                if (vg[((size_t)h * win_rows + 250 + r) * 128 + d] != vw[((size_t)h * win_rows + 100 + r) * 128 + d]) bad++;
            }
        }
        printf("  %-34s %s\n", "ring scatter + gather", bad ? "FAIL" : "ok");
        if (bad) failures++;
    }
    /* head row */
    {
        const uint32_t N = 256;
        Q4 W = q4_random(N, 2048); id<MTLBuffer> w, s, b; q4_upload(&W, &w, &s, &b);
        id<MTLBuffer> x = buf_bf16_random(2048, 1.0f), out = buf(N * 2);
        uint32_t Nv = N;
        run("flint_head_row", @[x, w, s, b, out, buf_struct(&Nv, 4)], MTLSizeMake(N / 32, 1, 1), MTLSizeMake(256, 1, 1), nil, 0);
        const uint16_t *xp = x.contents, *op = out.contents;
        float *ref = malloc(N * 4);
        for (uint32_t n = 0; n < N; n++) {
            double t = 0;
            for (uint32_t k = 0; k < 2048; k++) {
                const uint8_t byte = W.w[(size_t)n * 1024 + k / 2];
                const uint32_t qv = (k & 1) ? (byte >> 4) : (byte & 15);
                const size_t gi = (size_t)n * 32 + k / 64;
                t += (double)bf2f(xp[k]) * ((double)qv * bf2f(W.s[gi]) + bf2f(W.b[gi]));   /* qmv form: no weight rounding */
            }
            ref[n] = rbf((float)t);
        }
        check_bf16("head row (qmv form)", op, ref, N, 1e-6f, 1.0f / 64);
        free(ref);
    }
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *path = argc > 1 ? argv[1] : "metal/flint_prefill.metal";
        dev = MTLCreateSystemDefaultDevice();
        queue = [dev newCommandQueue];
        NSError *err = nil;
        NSString *src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:path] encoding:NSUTF8StringEncoding error:&err];
        if (!src) { fprintf(stderr, "cannot read %s\n", path); return 1; }
        MTLCompileOptions *opt = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) opt.mathMode = MTLMathModeSafe;
        lib = [dev newLibraryWithSource:src options:opt error:&err];
        if (!lib) { fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }
        test_gemm();
        test_moe();
        test_attention();
        test_small();
        printf(failures ? "FAILED: %d group(s)\n" : "all kernel tests passed\n", failures);
        return failures ? 1 : 0;
    }
}
