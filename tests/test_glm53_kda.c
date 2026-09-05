#include <math.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "ds4.h"
#include "ds4_gpu.h"

#ifdef DS4_ROCM_BUILD
typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} test_block_q4_K;

typedef struct {
    uint16_t d;
    int8_t qs[32];
} test_block_q8_0;

extern int ds4_gpu_matmul_q4_K_tensor(
    ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
    uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
    const ds4_gpu_tensor *x, uint64_t n_rows);

extern int ds4_gpu_glm_attention_indexed_decode_tensor(
    ds4_gpu_tensor *heads, const ds4_gpu_tensor *q,
    const ds4_gpu_tensor *qk_low, const ds4_gpu_tensor *kv_lora_cache,
    const ds4_gpu_tensor *k_rope_cache, const void *model_map,
    uint64_t model_size, uint64_t value_weight_offset,
    const ds4_gpu_tensor *selected, uint32_t n_selected,
    uint32_t cache_cap, bool cache_f16, uint32_t n_head,
    uint32_t kv_lora_dim, uint32_t qk_nope, uint32_t qk_rope,
    uint32_t value_dim, uint32_t n_ctx_orig, float freq_base,
    float freq_scale, float ext_factor, float attn_factor,
    float beta_fast, float beta_slow);
#endif

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static void require_ok(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "%s failed\n", what);
        exit(1);
    }
}

static void require_close(const char *what, float actual, float expected, float tolerance) {
    if (!isfinite(actual) || fabsf(actual - expected) > tolerance) {
        fprintf(stderr, "%s: got %.9g, expected %.9g (tolerance %.9g)\n",
                what, actual, expected, tolerance);
        exit(1);
    }
}

static uint16_t f32_to_bf16(float value) {
    union { float f; uint32_t u; } bits = { .f = value };
    const uint32_t rounding = 0x7fffu + ((bits.u >> 16) & 1u);
    return (uint16_t)((bits.u + rounding) >> 16);
}

static float bf16_to_f32(uint16_t value) {
    union { uint32_t u; float f; } bits = { .u = (uint32_t)value << 16 };
    return bits.f;
}

#ifdef __APPLE__
/* Normal-range only, and truncating rather than rounding.  Both are fine here:
 * the compound-producer fixture uses values with at most seven explicit
 * mantissa bits, well inside the half normal range, so truncation to ten bits
 * is exact and the encoding round-trips. */
static uint16_t f32_to_f16(float value) {
    union { float f; uint32_t u; } b = { .f = value };
    const uint32_t sign = (b.u >> 16) & 0x8000u;
    const int32_t  exp  = (int32_t)((b.u >> 23) & 0xffu) - 127 + 15;
    const uint32_t mant = (b.u >> 13) & 0x3ffu;
    if (exp <= 0 || exp >= 31) return (uint16_t)sign;
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static float f16_to_f32(uint16_t value) {
    const uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
    const uint32_t exp  = (uint32_t)(value >> 10) & 0x1fu;
    const uint32_t mant = (uint32_t)value & 0x3ffu;
    union { uint32_t u; float f; } b;
    b.u = exp == 0 ? sign : (sign | ((exp - 15u + 127u) << 23) | (mant << 13));
    return b.f;
}
#endif

/* Exercises ds4_gpu_glm53_matmul_bf16 at one width.  The reference is
 * accumulated in double and compared with a relative tolerance; a stride or
 * indexing error moves a result far more than that, which is what this is
 * here to catch. */
static void check_bf16_matmul(const uint8_t *model, size_t model_bytes,
                              uint64_t offset, uint32_t in_dim,
                              uint32_t out_dim, uint32_t rows,
                              const char *what) {
    uint16_t *w = (uint16_t *)(void *)((uint8_t *)(uintptr_t)model + offset);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            w[(size_t)o * in_dim + i] = f32_to_bf16(
                0.002f * (float)((int)(o % 11u) - 5) +
                0.001f * (float)((int)(i % 13u) - 6));
        }
    }
    const size_t x_bytes   = (size_t)rows * in_dim * sizeof(float);
    const size_t out_bytes = (size_t)rows * out_dim * sizeof(float);
    float *x        = malloc(x_bytes);
    float *expected = malloc(out_bytes);
    float *actual   = malloc(out_bytes);
    require_ok(x && expected && actual, "wide BF16 host allocation");
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            x[(size_t)r * in_dim + i] =
                0.02f * (float)((int)(i % 17u) - 8) + 0.005f * (float)r;
        }
        for (uint32_t o = 0; o < out_dim; o++) {
            double sum = 0.0;
            for (uint32_t i = 0; i < in_dim; i++) {
                sum += (double)bf16_to_f32(w[(size_t)o * in_dim + i]) *
                       (double)x[(size_t)r * in_dim + i];
            }
            expected[(size_t)r * out_dim + o] = (float)sum;
        }
    }
    ds4_gpu_tensor *gx   = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *gout = ds4_gpu_tensor_alloc(out_bytes);
    require_ok(gx && gout, "wide BF16 tensor allocation");
    require_ok(ds4_gpu_tensor_write(gx, 0, x, x_bytes), "wide BF16 input write");

    require_ok(ds4_gpu_glm53_matmul_bf16(gout, model, model_bytes, offset,
                                         in_dim, out_dim, gx, 1), what);
    require_ok(ds4_gpu_tensor_read(gout, 0, actual, out_dim * sizeof(float)),
               "wide BF16 decode output read");
    for (uint32_t o = 0; o < out_dim; o++) {
        require_close(what, actual[o], expected[o],
                      2e-5f * (fabsf(expected[o]) + 1.0f));
    }

    require_ok(ds4_gpu_glm53_matmul_bf16(gout, model, model_bytes, offset,
                                         in_dim, out_dim, gx, rows), what);
    require_ok(ds4_gpu_tensor_read(gout, 0, actual, out_bytes),
               "wide BF16 prefill output read");
    for (uint32_t i = 0; i < rows * out_dim; i++) {
        require_close(what, actual[i], expected[i],
                      2e-5f * (fabsf(expected[i]) + 1.0f));
    }
    ds4_gpu_tensor_free(gx);
    ds4_gpu_tensor_free(gout);
    free(x);
    free(expected);
    free(actual);
}

#ifdef __APPLE__
/*
 * Split-versus-generic indexed decode attention.
 *
 * GLM 5.3 decode runs kernel_glm_attention_indexed_decode_split_group8 once
 * more than 512 rows are selected; the generic kernel is what --quality and
 * every other backend run.  The two score and reduce in different orders, so
 * each is checked against a double-precision reference and they are checked
 * against each other with a tolerance rather than bit for bit.
 *
 * The selection holds what the GLM 5.3 indexer actually emits: rows at and
 * past cache_cap and UINT32_MAX tail sentinels, which both kernels must
 * exclude.  The rows just past cache_cap exist in memory and hold values that
 * would dominate every softmax, so a kernel that skips the bounds test fails
 * this loudly rather than by luck.  The row counts cover one partial block,
 * the 17-, 32-, 16- and 33-block reductions decode can request, the
 * fixed-count 16-block reduce, and a 65-block request the wrapper must refuse.
 */
static void check_split_dsa_attention(uint8_t *model, size_t model_bytes,
                                      uint64_t value_offset) {
    enum {
        SA_HEADS = 16,
        SA_LORA = 512,
        SA_NOPE = 64,
        SA_VALUE = 8,
        SA_MAX_SELECTED = 4096,
        SA_CAP = 4163,          /* > SA_MAX_SELECTED and coprime with 7919 */
        SA_POISON_ROWS = 16,    /* allocated past cache_cap, never to be read */
        SA_ROWS = SA_CAP + SA_POISON_ROWS,
        SA_MAX_BLOCKS = 65,
        SA_Q8_ROW_BYTES = (SA_LORA / 32) * 34,
    };
    static const struct {
        uint32_t n_selected;
        uint32_t block_rows;
        bool     accepted;
    } cases[] = {
        {8,    32,  true},   /* one partial block */
        {513,  32,  true},   /* 17 blocks: the first count decode splits */
        {1024, 32,  true},   /* 32 blocks */
        {2048, 128, true},   /* 16 blocks: the fixed-count reduce */
        {2051, 128, true},   /* 17 blocks: GLM 5.3's selection limit */
        {2051, 64,  true},   /* 33 blocks */
        {4096, 128, true},   /* 32 blocks: the resident dense window */
        {2051, 32,  false},  /* 65 blocks: more than the reduce walks */
    };

    /* Scores need a spread of tens, not a flat softmax, or the running-max
     * rescale in the split kernel is never exercised.  Each row and head
     * carries a multiple of one shared basis vector plus small noise, so
     * scores land in about [-17, 17] with many near-maximal rows. */
    float base[SA_LORA];
    for (uint32_t j = 0; j < SA_LORA; j++) {
        base[j] = (float)((int)((j * 13u) % 17u) - 8) / 8.0f;
    }
    uint16_t *kv_bits = malloc((size_t)SA_ROWS * SA_LORA * sizeof(*kv_bits));
    float *kv = malloc((size_t)SA_ROWS * SA_LORA * sizeof(*kv));
    float *low = malloc((size_t)SA_HEADS * SA_LORA * sizeof(*low));
    float *q = calloc((size_t)SA_HEADS * SA_NOPE, sizeof(*q));
    uint32_t *sel = malloc((size_t)SA_MAX_SELECTED * sizeof(*sel));
    double *ref = malloc((size_t)SA_HEADS * SA_VALUE * sizeof(*ref));
    double *lora = malloc((size_t)SA_LORA * sizeof(*lora));
    float *gen = malloc((size_t)SA_HEADS * SA_VALUE * sizeof(*gen));
    float *spl = malloc((size_t)SA_HEADS * SA_VALUE * sizeof(*spl));
    float *spl2 = malloc((size_t)SA_HEADS * SA_VALUE * sizeof(*spl2));
    float *exact = malloc((size_t)SA_HEADS * SA_VALUE * sizeof(*exact));
    require_ok(kv_bits && kv && low && q && sel && ref && lora &&
               gen && spl && spl2 && exact, "split attention host allocation");
    for (uint32_t row = 0; row < SA_ROWS; row++) {
        const float a = row < SA_CAP
            ? (float)((int)(row % 23u) - 11) / 22.0f
            : 8.0f;   /* poison: would dominate any softmax it leaked into */
        for (uint32_t j = 0; j < SA_LORA; j++) {
            const float noise = row < SA_CAP
                ? (float)((int)((row * 7u + j * 3u + (row ^ j)) % 97u) - 48) / 256.0f
                : 0.0f;
            const uint16_t bits = f32_to_f16(a * base[j] + noise);
            kv_bits[(size_t)row * SA_LORA + j] = bits;
            kv[(size_t)row * SA_LORA + j] = f16_to_f32(bits);
        }
    }
    for (uint32_t h = 0; h < SA_HEADS; h++) {
        for (uint32_t j = 0; j < SA_LORA; j++) {
            low[(size_t)h * SA_LORA + j] =
                (0.5f + (float)h / 16.0f) * base[j] +
                (float)((int)((h * 11u + j * 5u) % 61u) - 30) / 240.0f;
        }
    }
    /* Q8_0 value rows with unit scales, so a dequantized weight is its int8. */
    require_ok(value_offset + (uint64_t)SA_HEADS * SA_VALUE * SA_Q8_ROW_BYTES <= model_bytes,
               "split attention value rows fit the fixture model");
    for (uint32_t h = 0; h < SA_HEADS; h++) {
        for (uint32_t d = 0; d < SA_VALUE; d++) {
            uint8_t *row = model + value_offset +
                (size_t)(h * SA_VALUE + d) * SA_Q8_ROW_BYTES;
            for (uint32_t b = 0; b < SA_LORA / 32u; b++) {
                const uint16_t one = 0x3c00u;
                memcpy(row + b * 34u, &one, sizeof(one));
                int8_t *qs = (int8_t *)(row + b * 34u + 2u);
                for (uint32_t i = 0; i < 32u; i++) {
                    qs[i] = (int8_t)((int)((h * 5u + d * 3u + (b * 32u + i) * 7u) % 15u) - 7);
                }
            }
        }
    }

    ds4_gpu_tensor *heads_gpu = ds4_gpu_tensor_alloc((uint64_t)SA_HEADS * SA_VALUE * sizeof(float));
    ds4_gpu_tensor *partial_lora_gpu = ds4_gpu_tensor_alloc(
        (uint64_t)SA_MAX_BLOCKS * SA_HEADS * SA_LORA * sizeof(float));
    ds4_gpu_tensor *partial_ms_gpu = ds4_gpu_tensor_alloc(
        (uint64_t)SA_MAX_BLOCKS * SA_HEADS * 2u * sizeof(float));
    ds4_gpu_tensor *q_gpu = ds4_gpu_tensor_alloc((uint64_t)SA_HEADS * SA_NOPE * sizeof(float));
    ds4_gpu_tensor *low_gpu = ds4_gpu_tensor_alloc((uint64_t)SA_HEADS * SA_LORA * sizeof(float));
    ds4_gpu_tensor *kv_gpu = ds4_gpu_tensor_alloc((uint64_t)SA_ROWS * SA_LORA * sizeof(uint16_t));
    ds4_gpu_tensor *rope_gpu = ds4_gpu_tensor_alloc(sizeof(float));
    ds4_gpu_tensor *sel_gpu = ds4_gpu_tensor_alloc((uint64_t)SA_MAX_SELECTED * sizeof(uint32_t));
    ds4_gpu_tensor *exact_scores_gpu = ds4_gpu_tensor_alloc(
        (uint64_t)SA_HEADS * SA_MAX_SELECTED * sizeof(float));
    ds4_gpu_tensor *exact_lora_gpu = ds4_gpu_tensor_alloc((uint64_t)SA_HEADS * SA_LORA * sizeof(float));
    ds4_gpu_tensor *exact_denom_gpu = ds4_gpu_tensor_alloc((uint64_t)SA_HEADS * sizeof(float));
    require_ok(heads_gpu && partial_lora_gpu && partial_ms_gpu && q_gpu &&
               low_gpu && kv_gpu && rope_gpu && sel_gpu && exact_scores_gpu &&
               exact_lora_gpu && exact_denom_gpu,
               "split attention GPU allocation");
    require_ok(ds4_gpu_tensor_write(q_gpu, 0, q, (uint64_t)SA_HEADS * SA_NOPE * sizeof(float)) &&
               ds4_gpu_tensor_write(low_gpu, 0, low, (uint64_t)SA_HEADS * SA_LORA * sizeof(float)) &&
               ds4_gpu_tensor_write(kv_gpu, 0, kv_bits, (uint64_t)SA_ROWS * SA_LORA * sizeof(uint16_t)),
               "split attention input write");

    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        const uint32_t n = cases[c].n_selected;
        const uint32_t block_rows = cases[c].block_rows;
        const uint32_t n_blocks = (n + block_rows - 1u) / block_rows;
        char what[96];
        snprintf(what, sizeof(what), "split attention %u rows x %u/block",
                 n, block_rows);

        /* A permutation of valid rows, with the indexer's failure shapes
         * scattered through it: rows at and just past cache_cap, and the
         * UINT32_MAX tail sentinels GLM 5.3's pool expansion emits. */
        for (uint32_t s = 0; s < n; s++) sel[s] = (s * 7919u) % SA_CAP;
        sel[0] = SA_CAP;
        sel[1] = SA_CAP - 1u;
        for (uint32_t s = 50; s < n; s += 97u) sel[s] = SA_CAP + s % 5u;
        for (uint32_t s = n >= 3u ? n - 3u : 0u; s < n; s++) sel[s] = UINT32_MAX;
        require_ok(ds4_gpu_tensor_write(sel_gpu, 0, sel, (uint64_t)n * sizeof(uint32_t)),
                   "split attention selection write");

        /* The reference follows the generic kernel: score valid rows, drop
         * the rest, softmax, weighted lora sum, then the value projection. */
        double ref_scale = 0.0;
        for (uint32_t h = 0; h < SA_HEADS; h++) {
            const float *lh = low + (size_t)h * SA_LORA;
            double max_score = -DBL_MAX;
            for (uint32_t s = 0; s < n; s++) {
                if (sel[s] >= SA_CAP) continue;
                const float *row = kv + (size_t)sel[s] * SA_LORA;
                double dot = 0.0;
                for (uint32_t j = 0; j < SA_LORA; j++) dot += (double)lh[j] * row[j];
                const double score = dot * 0.125;   /* 1/sqrt(SA_NOPE) */
                if (score > max_score) max_score = score;
            }
            double denom = 0.0;
            for (uint32_t j = 0; j < SA_LORA; j++) lora[j] = 0.0;
            for (uint32_t s = 0; s < n; s++) {
                if (sel[s] >= SA_CAP) continue;
                const float *row = kv + (size_t)sel[s] * SA_LORA;
                double dot = 0.0;
                for (uint32_t j = 0; j < SA_LORA; j++) dot += (double)lh[j] * row[j];
                const double w = exp(dot * 0.125 - max_score);
                denom += w;
                for (uint32_t j = 0; j < SA_LORA; j++) lora[j] += w * row[j];
            }
            if (denom < 1e-20) denom = 1e-20;
            for (uint32_t d = 0; d < SA_VALUE; d++) {
                const uint8_t *row = model + value_offset +
                    (size_t)(h * SA_VALUE + d) * SA_Q8_ROW_BYTES;
                double out = 0.0;
                for (uint32_t j = 0; j < SA_LORA; j++) {
                    const int8_t qv = (int8_t)row[(j / 32u) * 34u + 2u + j % 32u];
                    out += (double)qv * (lora[j] / denom);
                }
                ref[h * SA_VALUE + d] = out;
                if (fabs(out) > ref_scale) ref_scale = fabs(out);
            }
        }

        const int split_rc = ds4_gpu_glm_attention_indexed_decode_split_group8_tensor(
            heads_gpu, partial_lora_gpu, partial_ms_gpu, q_gpu, low_gpu,
            kv_gpu, rope_gpu, model, model_bytes, value_offset, sel_gpu, n,
            false, SA_CAP, true, SA_HEADS, SA_LORA, SA_NOPE, 0, SA_VALUE, 0,
            block_rows, n_blocks, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        if (!cases[c].accepted) {
            require_ok(split_rc == 0 && n_blocks > 64u,
                       "split attention refuses more blocks than the reduce walks");
            continue;
        }
        require_ok(split_rc, what);
        require_ok(ds4_gpu_tensor_read(heads_gpu, 0, spl, (uint64_t)SA_HEADS * SA_VALUE * sizeof(float)),
                   "split attention output read");
        require_ok(ds4_gpu_glm_attention_indexed_decode_split_group8_tensor(
            heads_gpu, partial_lora_gpu, partial_ms_gpu, q_gpu, low_gpu,
            kv_gpu, rope_gpu, model, model_bytes, value_offset, sel_gpu, n,
            false, SA_CAP, true, SA_HEADS, SA_LORA, SA_NOPE, 0, SA_VALUE, 0,
            block_rows, n_blocks, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f), what);
        require_ok(ds4_gpu_tensor_read(heads_gpu, 0, spl2, (uint64_t)SA_HEADS * SA_VALUE * sizeof(float)),
                   "split attention repeat read");
        require_ok(ds4_gpu_glm_attention_indexed_decode_tensor(
            heads_gpu, q_gpu, low_gpu, kv_gpu, rope_gpu, model, model_bytes,
            value_offset, sel_gpu, n, SA_CAP, true, SA_HEADS, SA_LORA,
            SA_NOPE, 0, SA_VALUE, 0, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f),
            "generic indexed decode attention");
        require_ok(ds4_gpu_tensor_read(heads_gpu, 0, gen, (uint64_t)SA_HEADS * SA_VALUE * sizeof(float)),
                   "generic attention output read");

        /* The phased exact kernels claim the generic kernel's arithmetic
         * operation for operation, so their output must match it bit for
         * bit -- including the excluded rows and sentinels. */
        require_ok(ds4_gpu_glm_attention_indexed_decode_exact_tensor(
            heads_gpu, exact_scores_gpu, exact_lora_gpu, exact_denom_gpu,
            low_gpu, kv_gpu, model, model_bytes, value_offset, sel_gpu, n,
            SA_CAP, true, SA_HEADS, SA_LORA, SA_NOPE, 0, SA_VALUE),
            "exact indexed decode attention");
        require_ok(ds4_gpu_tensor_read(heads_gpu, 0, exact, (uint64_t)SA_HEADS * SA_VALUE * sizeof(float)),
                   "exact attention output read");
        if (memcmp(exact, gen, (size_t)SA_HEADS * SA_VALUE * sizeof(float)) != 0) {
            double worst = 0.0;
            for (uint32_t i = 0; i < SA_HEADS * SA_VALUE; i++) {
                worst = fmax(worst, fabs((double)exact[i] - (double)gen[i]));
            }
            fprintf(stderr, "%s: exact kernels differ from the generic kernel (max |delta| %.3g)\n",
                    what, worst);
            exit(1);
        }

        double gen_err = 0.0, spl_err = 0.0, pair_err = 0.0;
        for (uint32_t i = 0; i < SA_HEADS * SA_VALUE; i++) {
            if (!isfinite(gen[i]) || !isfinite(spl[i])) {
                fprintf(stderr, "%s: non-finite output at %u\n", what, i);
                exit(1);
            }
            gen_err = fmax(gen_err, fabs((double)gen[i] - ref[i]));
            spl_err = fmax(spl_err, fabs((double)spl[i] - ref[i]));
            pair_err = fmax(pair_err, fabs((double)spl[i] - (double)gen[i]));
        }
        if (memcmp(spl, spl2, (size_t)SA_HEADS * SA_VALUE * sizeof(float)) != 0) {
            fprintf(stderr, "%s: split output changed on repeat\n", what);
            exit(1);
        }
        /* Both kernels accumulate in f32 over up to 2051 rows and 512 lanes;
         * a tiling, block or bounds error moves a result by a large fraction
         * of ref_scale, orders of magnitude past this. */
        const double tol = 1e-4 * ref_scale;
        fprintf(stderr,
                "%s: ref_scale %.3g, generic %.3g, split %.3g, split-vs-generic %.3g (tol %.3g), exact == generic\n",
                what, ref_scale, gen_err, spl_err, pair_err, tol);
        if (gen_err > tol || spl_err > tol || pair_err > tol) {
            fprintf(stderr, "%s: attention diverged\n", what);
            exit(1);
        }
    }

    /* GLM 5.2's selections are always in range and it ran the unchecked
     * variant before GLM 5.3 was admitted; decode now passes false for every
     * GLM model, which is free of numerical consequence only if the two
     * variants perform identical arithmetic on valid rows. */
    for (uint32_t s = 0; s < 2048u; s++) sel[s] = (s * 7919u) % SA_CAP;
    require_ok(ds4_gpu_tensor_write(sel_gpu, 0, sel, 2048u * sizeof(uint32_t)),
               "all-valid selection write");
    for (int assume_valid = 0; assume_valid < 2; assume_valid++) {
        require_ok(ds4_gpu_glm_attention_indexed_decode_split_group8_tensor(
            heads_gpu, partial_lora_gpu, partial_ms_gpu, q_gpu, low_gpu,
            kv_gpu, rope_gpu, model, model_bytes, value_offset, sel_gpu, 2048u,
            assume_valid != 0, SA_CAP, true, SA_HEADS, SA_LORA, SA_NOPE, 0,
            SA_VALUE, 0, 128u, 16u, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f),
            "split attention on an all-valid selection");
        require_ok(ds4_gpu_tensor_read(heads_gpu, 0, assume_valid ? spl2 : spl,
                                       (uint64_t)SA_HEADS * SA_VALUE * sizeof(float)),
                   "all-valid split output read");
    }
    if (memcmp(spl, spl2, (size_t)SA_HEADS * SA_VALUE * sizeof(float)) != 0) {
        fprintf(stderr, "split attention: bounds-checked and unchecked "
                        "variants differ on an all-valid selection\n");
        exit(1);
    }

    ds4_gpu_tensor_free(exact_denom_gpu);
    ds4_gpu_tensor_free(exact_lora_gpu);
    ds4_gpu_tensor_free(exact_scores_gpu);
    ds4_gpu_tensor_free(sel_gpu);
    ds4_gpu_tensor_free(rope_gpu);
    ds4_gpu_tensor_free(kv_gpu);
    ds4_gpu_tensor_free(low_gpu);
    ds4_gpu_tensor_free(q_gpu);
    ds4_gpu_tensor_free(partial_ms_gpu);
    ds4_gpu_tensor_free(partial_lora_gpu);
    ds4_gpu_tensor_free(heads_gpu);
    free(exact);
    free(spl2);
    free(spl);
    free(gen);
    free(lora);
    free(ref);
    free(sel);
    free(q);
    free(low);
    free(kv);
    free(kv_bits);
}
#endif

#ifdef __APPLE__
static void require_prefill_dispatch(uint32_t feature, bool expected,
                                     const char *what) {
    const uint32_t dispatched = ds4_gpu_test_glm53_prefill_take_dispatches();
    require_ok(dispatched == (expected ? feature : 0u), what);
}

/* Exactness oracle for the GLM 5.3 Flash prefill qk-low token tile.
 *
 * kernel_glm_qk_lowrank_q8_0_batch_t<TT> changes only which threadgroup
 * computes which outputs and how many tokens one thread carries; every output
 * keeps the reference kernel's expression, block order and column order.  So
 * each tile must reproduce kernel_glm_qk_lowrank_q8_0_batch bit for bit at the
 * model's shape, including the partial tail tile that a 1596-token chunk and a
 * 33- or 1-token prompt produce.  The dispatch runs through the same selection
 * the graph uses; DS4_METAL_DISABLE_GLM53_PREFILL_QK_LOW picks the reference
 * and DS4_METAL_GLM53_PREFILL_QK_LOW_TILE picks the tile. */
static void check_glm53_qk_lowrank_token_tile(uint8_t *model,
                                              uint64_t model_bytes,
                                              uint64_t kb_offset) {
    enum {
        QL_HEADS = 64,
        QL_KV_LORA = 512,
        QL_QK_NOPE = 256,
        QL_QK_DIM = 256,
        QL_ROW_BYTES = 272,      /* 8 Q8_0 blocks of 34 bytes */
        QL_Q8_0_TYPE = 8,        /* GGUF type code for Q8_0 */
        QL_MAX_TOKENS = 2048,
    };
    static const uint32_t token_counts[] = { 2048u, 1596u, 33u, 1u };
    static const uint32_t tiles[] = { 4u, 8u, 16u };
    /* Scales that exercise the sign and the subnormal half range, where a
     * reassociated product would round differently. */
    static const uint16_t scale_bits[] = {
        0x0001u, 0x8001u, 0x03ffu, 0x83ffu, 0x0000u, 0x8000u,
        0x3c00u, 0xbc00u, 0x1234u, 0x9876u, 0x2c00u, 0xac00u, 0x0400u, 0x8400u,
    };
    const uint64_t weight_bytes =
        (uint64_t)QL_HEADS * QL_KV_LORA * QL_ROW_BYTES;
    require_ok(kb_offset + weight_bytes <= model_bytes,
               "qk-low K_b rows fit the fixture model");

    uint64_t rng = 0x9e3779b97f4a7c15ull;
    for (uint64_t row = 0; row < (uint64_t)QL_HEADS * QL_KV_LORA; row++) {
        uint8_t *dst = model + kb_offset + row * QL_ROW_BYTES;
        for (uint32_t b = 0; b < QL_QK_NOPE / 32u; b++) {
            const uint16_t d =
                scale_bits[(row * 8u + b) % (sizeof(scale_bits) / sizeof(scale_bits[0]))];
            memcpy(dst + b * 34u, &d, sizeof(d));
            int8_t *qs = (int8_t *)(dst + b * 34u + 2u);
            for (uint32_t i = 0; i < 32u; i++) {
                rng = rng * 6364136223846793005ull + 1442695040888963407ull;
                qs[i] = (int8_t)(uint8_t)(rng >> 33);
            }
        }
    }

    const uint64_t q_elems = (uint64_t)QL_MAX_TOKENS * QL_HEADS * QL_QK_DIM;
    const uint64_t out_elems = (uint64_t)QL_MAX_TOKENS * QL_HEADS * QL_KV_LORA;
    float *q_host = malloc(q_elems * sizeof(float));
    float *ref_host = malloc(out_elems * sizeof(float));
    float *tile_host = malloc(out_elems * sizeof(float));
    require_ok(q_host && ref_host && tile_host, "qk-low host allocation");
    for (uint64_t i = 0; i < q_elems; i++) {
        rng = rng * 6364136223846793005ull + 1442695040888963407ull;
        q_host[i] = (float)((int32_t)(uint32_t)(rng >> 32) / 1073741824.0) - 1.0f;
    }

    ds4_gpu_tensor *q_gpu = ds4_gpu_tensor_alloc(q_elems * sizeof(float));
    ds4_gpu_tensor *ref_gpu = ds4_gpu_tensor_alloc(out_elems * sizeof(float));
    ds4_gpu_tensor *tile_gpu = ds4_gpu_tensor_alloc(out_elems * sizeof(float));
    require_ok(q_gpu && ref_gpu && tile_gpu, "qk-low GPU allocation");
    require_ok(ds4_gpu_tensor_write(q_gpu, 0, q_host, q_elems * sizeof(float)),
               "qk-low q write");

    /* The decode tile changes output ownership only. Reuse the signed,
     * subnormal-scale fixture above and verify the active and rollback paths. */
    require_ok(unsetenv("DS4_METAL_DISABLE_M3_ULTRA_GLM53_DECODE") == 0,
               "clear inherited decode rollback");
    for (unsigned arm = 0; arm < 3; arm++) {
        if (arm == 0) setenv("DS4_METAL_DISABLE_GLM53_DECODE_QK_LOW", "1", 1);
        else unsetenv("DS4_METAL_DISABLE_GLM53_DECODE_QK_LOW");
        if (arm == 2) setenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING", "1", 1);
        const uint64_t bytes = (uint64_t)QL_HEADS * QL_KV_LORA * sizeof(float);
        require_ok(ds4_gpu_tensor_fill_f32(tile_gpu, -17.0f, QL_HEADS * QL_KV_LORA),
                   "decode qk-low poison");
        require_ok(ds4_gpu_glm_qk_lowrank_typed_tensor(tile_gpu, q_gpu,
                       model, model_bytes, kb_offset, QL_Q8_0_TYPE,
                       QL_HEADS, QL_KV_LORA, QL_QK_NOPE, QL_QK_DIM),
                   "decode qk-low dispatch");
        require_prefill_dispatch(DS4_GPU_GLM53_DECODE_QK_LOW, arm == 1,
                                 "decode qk-low coverage");
        require_ok(ds4_gpu_tensor_read(tile_gpu, 0, tile_host, bytes),
                   "decode qk-low read");
        if (arm == 0) memcpy(ref_host, tile_host, bytes);
        else require_ok(memcmp(ref_host, tile_host, bytes) == 0,
                        "decode qk-low bitwise output");
    }
    unsetenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING");

    for (size_t c = 0; c < sizeof(token_counts) / sizeof(token_counts[0]); c++) {
        const uint32_t n_tokens = token_counts[c];
        const uint64_t bytes =
            (uint64_t)n_tokens * QL_HEADS * QL_KV_LORA * sizeof(float);
        char what[96];

        require_ok(setenv("DS4_METAL_DISABLE_GLM53_PREFILL_QK_LOW", "1", 1) == 0,
                   "qk-low reference switch");
        snprintf(what, sizeof(what), "qk-low reference at %u tokens", n_tokens);
        require_ok(ds4_gpu_glm_qk_lowrank_typed_batch_tensor(
                       ref_gpu, q_gpu, model, model_bytes, kb_offset,
                       QL_Q8_0_TYPE, n_tokens, QL_HEADS, QL_KV_LORA,
                       QL_QK_NOPE, QL_QK_DIM), what);
        require_ok(ds4_gpu_tensor_read(ref_gpu, 0, ref_host, bytes), what);
        require_prefill_dispatch(DS4_GPU_GLM53_PREFILL_QK_LOW, false, what);
        require_ok(unsetenv("DS4_METAL_DISABLE_GLM53_PREFILL_QK_LOW") == 0,
                   "qk-low reference switch clear");

        const size_t tile_count = sizeof(tiles) / sizeof(tiles[0]);
        for (size_t t = 0; t <= tile_count; t++) {
            const bool rollback = t == tile_count;
            const uint32_t tile = tiles[t % tile_count];
            if (rollback) setenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING", "1", 1);
            char tile_text[8];
            snprintf(tile_text, sizeof(tile_text), "%u", tile);
            require_ok(setenv("DS4_METAL_GLM53_PREFILL_QK_LOW_TILE", tile_text, 1) == 0,
                       "qk-low tile switch");
            snprintf(what, sizeof(what), "qk-low tile %u at %u tokens rollback=%u",
                     tile, n_tokens, rollback);
            /* A quiet NaN in every output first, so a kernel that skips rows
             * fails here rather than matching a stale buffer.  Built from bits
             * because -ffast-math makes the NAN macro undefined. */
            const uint32_t poison_bits = 0x7fc01234u;
            float poison;
            memcpy(&poison, &poison_bits, sizeof(poison));
            require_ok(ds4_gpu_tensor_fill_f32(tile_gpu, poison,
                                               (uint64_t)n_tokens * QL_HEADS * QL_KV_LORA),
                       what);
            require_ok(ds4_gpu_glm_qk_lowrank_typed_batch_tensor(
                           tile_gpu, q_gpu, model, model_bytes, kb_offset,
                           QL_Q8_0_TYPE, n_tokens, QL_HEADS, QL_KV_LORA,
                           QL_QK_NOPE, QL_QK_DIM), what);
            require_ok(ds4_gpu_tensor_read(tile_gpu, 0, tile_host, bytes), what);
            require_prefill_dispatch(DS4_GPU_GLM53_PREFILL_QK_LOW,
                                      !rollback && n_tokens >= tile, what);
            if (memcmp(ref_host, tile_host, (size_t)bytes) != 0) {
                for (uint64_t i = 0; i < bytes / sizeof(float); i++) {
                    if (memcmp(&ref_host[i], &tile_host[i], sizeof(float)) == 0) continue;
                    fprintf(stderr,
                            "%s: output %llu is %.9g, reference %.9g\n",
                            what, (unsigned long long)i,
                            (double)tile_host[i], (double)ref_host[i]);
                    break;
                }
                exit(1);
            }
        }
        require_ok(unsetenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING") == 0,
                   "clear aggregate rollback switch");
        require_ok(unsetenv("DS4_METAL_GLM53_PREFILL_QK_LOW_TILE") == 0,
                   "qk-low tile switch clear");
    }

    ds4_gpu_tensor_free(tile_gpu);
    ds4_gpu_tensor_free(ref_gpu);
    ds4_gpu_tensor_free(q_gpu);
    free(tile_host);
    free(ref_host);
    free(q_host);
}

/* Optional 48-GiB regression for both 32-bit element-offset wrap boundaries.
 * Kept out of the ordinary suite so machines with smaller memory can run it.
 * The last real token must equal a one-token reference, not retain poison or
 * read token zero after a wrapped input offset. */
static void check_glm53_qk_lowrank_large_offsets(uint8_t *model,
                                                uint64_t model_bytes,
                                                uint64_t kb_offset) {
    if (!getenv("DS4_TEST_GLM53_LARGE_QK")) return;
    enum { HEADS = 64, NOPE = 256, LORA = 512, TOKENS = 262145 };
    const uint64_t q_row = (uint64_t)HEADS * NOPE * sizeof(float);
    const uint64_t out_row = (uint64_t)HEADS * LORA * sizeof(float);
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc((uint64_t)TOKENS * q_row);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)TOKENS * out_row);
    ds4_gpu_tensor *ref = ds4_gpu_tensor_alloc(out_row);
    require_ok(q && out && ref, "large qk-low allocations (48 GiB required)");
    ds4_gpu_tensor *q_last = ds4_gpu_tensor_view(q, (TOKENS - 1ull) * q_row, q_row);
    ds4_gpu_tensor *out_last = ds4_gpu_tensor_view(out, (TOKENS - 1ull) * out_row, out_row);
    require_ok(q_last && out_last, "large qk-low tail views");
    require_ok(ds4_gpu_tensor_fill_f32(q, 0.0f, (uint64_t)TOKENS * HEADS * NOPE) &&
               ds4_gpu_tensor_fill_f32(q_last, 1.0f, HEADS * NOPE) &&
               ds4_gpu_tensor_fill_f32(out_last, 123.0f, HEADS * LORA), "large qk-low inputs and poison");
    require_ok(ds4_gpu_glm_qk_lowrank_typed_batch_tensor(ref, q_last,
        model, model_bytes, kb_offset, 8u, 1, HEADS, LORA, NOPE, NOPE), "large qk-low one-token reference");
    require_prefill_dispatch(DS4_GPU_GLM53_PREFILL_QK_LOW, false, "large qk-low reference coverage");
    require_ok(ds4_gpu_glm_qk_lowrank_typed_batch_tensor(out, q,
        model, model_bytes, kb_offset, 8u, TOKENS, HEADS, LORA, NOPE, NOPE), "large qk-low token tile");
    require_prefill_dispatch(DS4_GPU_GLM53_PREFILL_QK_LOW, true, "large qk-low tile coverage");
    float expected[HEADS * LORA], actual[HEADS * LORA];
    require_ok(ds4_gpu_tensor_read(ref, 0, expected, out_row) &&
               ds4_gpu_tensor_read(out_last, 0, actual, out_row), "large qk-low readback");
    require_ok(memcmp(expected, actual, out_row) == 0, "large qk-low tail is bit-identical");
    ds4_gpu_tensor_free(out_last); ds4_gpu_tensor_free(q_last);
    ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(q);
    puts("GLM qk-low 64-bit offset regression: PASS");
}

/* Exactness oracle for the GLM 5.3 Flash indexed prefill attention head width.
 *
 * kernel_glm_attention_indexed_batch_lora_group16_vec_valid_fullheads carries
 * two heads per simdgroup, so a token stages its selected rows four times
 * instead of eight.  Each head keeps the one-head kernel's row order, its four
 * dot(float4) terms, its simd_sum tree and its online-softmax update, so all
 * 512 outputs of every head must match bit for bit.
 * DS4_METAL_GLM53_PREFILL_INDEXED_ATTN_HEADS_PER_SG picks the width, and
 * DS4_METAL_DISABLE_GLM53_PREFILL_INDEXED_ATTN pins main's one-head kernel. */
static void check_glm53_indexed_attention_head_width(void) {
    enum {
        IA_HEADS = 64,
        IA_LORA = 512,
        IA_NOPE = 256,
        IA_CACHE_CAP = 4096,
        IA_TOKENS = 8,
        IA_MAX_SELECTED = 2051,
    };
    /* 2051 is the model's selection limit; the rest land on a partial trailing
     * 16-row staging block, which is where a head-width bug would show. */
    static const uint32_t selected_counts[] = { 2051u, 512u, 33u, 16u, 1u };

    const uint64_t q_elems = (uint64_t)IA_TOKENS * IA_HEADS * IA_NOPE;
    const uint64_t low_elems = (uint64_t)IA_TOKENS * IA_HEADS * IA_LORA;
    const uint64_t cache_elems = (uint64_t)IA_CACHE_CAP * IA_LORA;
    const uint64_t sel_elems = (uint64_t)IA_TOKENS * IA_MAX_SELECTED;

    float *q_host = malloc(q_elems * sizeof(float));
    float *low_host = malloc(low_elems * sizeof(float));
    uint16_t *cache_host = malloc(cache_elems * sizeof(uint16_t));
    uint32_t *sel_host = malloc(sel_elems * sizeof(uint32_t));
    float *ref_host = malloc(low_elems * sizeof(float));
    float *dual_host = malloc(low_elems * sizeof(float));
    require_ok(q_host && low_host && cache_host && sel_host && ref_host && dual_host,
               "indexed attention host allocation");

    uint64_t rng = 0xda3e39cb94b95bdbull;
#define IA_NEXT_UNIT() ( \
    rng = rng * 6364136223846793005ull + 1442695040888963407ull, \
    (float)((int32_t)(uint32_t)(rng >> 32) / 1073741824.0) - 1.0f)
    for (uint64_t i = 0; i < q_elems; i++) q_host[i] = IA_NEXT_UNIT();
    for (uint64_t i = 0; i < low_elems; i++) low_host[i] = IA_NEXT_UNIT();
    /* Half values kept in the normal range so the truncating encoder above is
     * exact and the fixture round-trips. */
    for (uint64_t i = 0; i < cache_elems; i++) {
        const float unit = IA_NEXT_UNIT();
        cache_host[i] = f32_to_f16(unit >= 0.0f ? 0.0625f + unit : -0.0625f + unit);
    }
    /* Every selected row must be in cache range: this kernel family is the
     * "valid rows" instantiation and does not re-check them. */
    for (uint64_t i = 0; i < sel_elems; i++) {
        rng = rng * 6364136223846793005ull + 1442695040888963407ull;
        sel_host[i] = (uint32_t)((rng >> 33) % (uint64_t)IA_CACHE_CAP);
    }

    ds4_gpu_tensor *q_gpu = ds4_gpu_tensor_alloc(q_elems * sizeof(float));
    ds4_gpu_tensor *low_gpu = ds4_gpu_tensor_alloc(low_elems * sizeof(float));
    ds4_gpu_tensor *cache_gpu = ds4_gpu_tensor_alloc(cache_elems * sizeof(uint16_t));
    ds4_gpu_tensor *rope_gpu = ds4_gpu_tensor_alloc(sizeof(float));
    ds4_gpu_tensor *sel_gpu = ds4_gpu_tensor_alloc(sel_elems * sizeof(uint32_t));
    ds4_gpu_tensor *ref_gpu = ds4_gpu_tensor_alloc(low_elems * sizeof(float));
    ds4_gpu_tensor *dual_gpu = ds4_gpu_tensor_alloc(low_elems * sizeof(float));
    require_ok(q_gpu && low_gpu && cache_gpu && rope_gpu && sel_gpu && ref_gpu && dual_gpu,
               "indexed attention GPU allocation");
    require_ok(ds4_gpu_tensor_write(q_gpu, 0, q_host, q_elems * sizeof(float)) &&
               ds4_gpu_tensor_write(low_gpu, 0, low_host, low_elems * sizeof(float)) &&
               ds4_gpu_tensor_write(cache_gpu, 0, cache_host, cache_elems * sizeof(uint16_t)) &&
               ds4_gpu_tensor_write(sel_gpu, 0, sel_host, sel_elems * sizeof(uint32_t)),
               "indexed attention input write");

    for (size_t c = 0; c < sizeof(selected_counts) / sizeof(selected_counts[0]); c++) {
        const uint32_t n_selected = selected_counts[c];
        const uint64_t bytes = low_elems * sizeof(float);
        char what[96];

        require_ok(setenv("DS4_METAL_GLM53_PREFILL_INDEXED_ATTN_HEADS_PER_SG", "1", 1) == 0,
                   "indexed attention width switch");
        snprintf(what, sizeof(what), "indexed attention one head at %u rows", n_selected);
        require_ok(ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
                       ref_gpu, q_gpu, low_gpu, cache_gpu, rope_gpu, sel_gpu,
                       IA_TOKENS, n_selected, IA_CACHE_CAP, true, IA_HEADS,
                       IA_LORA, IA_NOPE, 0u, 0u,
                       10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f), what);
        require_ok(ds4_gpu_tensor_read(ref_gpu, 0, ref_host, bytes), what);
        require_prefill_dispatch(DS4_GPU_GLM53_PREFILL_INDEXED_ATTN, false, what);

        require_ok(setenv("DS4_METAL_GLM53_PREFILL_INDEXED_ATTN_HEADS_PER_SG", "2", 1) == 0,
                   "indexed attention width switch");
        snprintf(what, sizeof(what), "indexed attention two heads at %u rows", n_selected);
        const uint32_t poison_bits = 0x7fc01234u;
        float poison;
        memcpy(&poison, &poison_bits, sizeof(poison));
        require_ok(ds4_gpu_tensor_fill_f32(dual_gpu, poison, low_elems), what);
        require_ok(ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
                       dual_gpu, q_gpu, low_gpu, cache_gpu, rope_gpu, sel_gpu,
                       IA_TOKENS, n_selected, IA_CACHE_CAP, true, IA_HEADS,
                       IA_LORA, IA_NOPE, 0u, 0u,
                       10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f), what);
        require_ok(ds4_gpu_tensor_read(dual_gpu, 0, dual_host, bytes), what);
        require_prefill_dispatch(DS4_GPU_GLM53_PREFILL_INDEXED_ATTN, true, what);
        if (memcmp(ref_host, dual_host, (size_t)bytes) != 0) {
            for (uint64_t i = 0; i < low_elems; i++) {
                if (memcmp(&ref_host[i], &dual_host[i], sizeof(float)) == 0) continue;
                fprintf(stderr,
                        "%s: token %llu head %llu lane element %llu is %.9g, one-head %.9g\n",
                        what,
                        (unsigned long long)(i / (IA_HEADS * IA_LORA)),
                        (unsigned long long)((i / IA_LORA) % IA_HEADS),
                        (unsigned long long)(i % IA_LORA),
                        (double)dual_host[i], (double)ref_host[i]);
                break;
            }
            exit(1);
        }
    }
    require_ok(unsetenv("DS4_METAL_GLM53_PREFILL_INDEXED_ATTN_HEADS_PER_SG") == 0,
               "indexed attention width switch clear");
#undef IA_NEXT_UNIT

    ds4_gpu_tensor_free(dual_gpu);
    ds4_gpu_tensor_free(ref_gpu);
    ds4_gpu_tensor_free(sel_gpu);
    ds4_gpu_tensor_free(rope_gpu);
    ds4_gpu_tensor_free(cache_gpu);
    ds4_gpu_tensor_free(low_gpu);
    ds4_gpu_tensor_free(q_gpu);
    free(dual_host);
    free(ref_host);
    free(sel_host);
    free(cache_host);
    free(low_host);
    free(q_host);
}

/* Invalid selected IDs must be skipped without changing the order of valid
 * rows. Compare the guarded API with the same selection compacted through the
 * valid API, including the original RoPE and partial-head specializations. */
static void check_glm53_indexed_attention_invalid_rows(void) {
    enum { CAP = 64, MAX_HEADS = 64, LORA = 512, MAX_Q = 320, VALID = 16 };
    float q[MAX_HEADS * MAX_Q], low[MAX_HEADS * LORA];
    uint16_t cache[CAP * LORA], rope[CAP * 64];
    uint32_t compact[VALID], masked[2 * VALID];
    float expected[MAX_HEADS * LORA], actual[MAX_HEADS * LORA];
    for (unsigned i = 0; i < MAX_HEADS * MAX_Q; i++) q[i] = 0.01f * ((int)(i % 17) - 8);
    for (unsigned i = 0; i < MAX_HEADS * LORA; i++) low[i] = 0.02f * ((int)(i % 19) - 9);
    for (unsigned i = 0; i < CAP * LORA; i++) cache[i] = f32_to_f16(0.125f * ((int)(i % 13) - 6));
    for (unsigned i = 0; i < CAP * 64; i++) rope[i] = f32_to_f16(0.125f * ((int)(i % 7) - 3));
    for (unsigned i = 0; i < VALID; i++) {
        compact[i] = (i * 7) % CAP;
        masked[2 * i] = compact[i];
        masked[2 * i + 1] = i % 2 ? UINT32_MAX : CAP + i;
    }
    ds4_gpu_tensor *gq = ds4_gpu_tensor_alloc(sizeof(q));
    ds4_gpu_tensor *glow = ds4_gpu_tensor_alloc(sizeof(low));
    ds4_gpu_tensor *gcache = ds4_gpu_tensor_alloc(sizeof(cache));
    ds4_gpu_tensor *grope = ds4_gpu_tensor_alloc(sizeof(rope));
    ds4_gpu_tensor *gcompact = ds4_gpu_tensor_alloc(sizeof(compact));
    ds4_gpu_tensor *gmasked = ds4_gpu_tensor_alloc(sizeof(masked));
    ds4_gpu_tensor *gout = ds4_gpu_tensor_alloc(sizeof(actual));
    require_ok(gq && glow && gcache && grope && gcompact && gmasked && gout, "invalid attention allocations");
    require_ok(ds4_gpu_tensor_write(gq, 0, q, sizeof(q)) &&
               ds4_gpu_tensor_write(glow, 0, low, sizeof(low)) &&
               ds4_gpu_tensor_write(gcache, 0, cache, sizeof(cache)) &&
               ds4_gpu_tensor_write(grope, 0, rope, sizeof(rope)) &&
               ds4_gpu_tensor_write(gcompact, 0, compact, sizeof(compact)) &&
               ds4_gpu_tensor_write(gmasked, 0, masked, sizeof(masked)), "invalid attention uploads");
    for (unsigned h = 0; h < 2; h++) for (unsigned r = 0; r < 2; r++) {
        const uint32_t heads = h ? 7 : 64, rot = r ? 64 : 0;
        const uint64_t bytes = (uint64_t)heads * LORA * sizeof(float);
        require_ok(ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
            gout, gq, glow, gcache, grope, gcompact, 1, VALID, CAP, true,
            heads, LORA, 256, rot, 4096, 10000.0f, 1.0f, 1.0f, 1.0f, 32.0f, 1.0f), "compact attention reference");
        require_ok(ds4_gpu_tensor_read(gout, 0, expected, bytes), "compact attention read");
        require_prefill_dispatch(DS4_GPU_GLM53_PREFILL_INDEXED_ATTN, heads == 64 && rot == 0, "compact attention coverage");
        require_ok(ds4_gpu_tensor_fill_f32(gout, 123.0f, heads * LORA), "poison masked output");
        require_ok(ds4_gpu_glm_attention_indexed_batch_lora_tensor(
            gout, gq, glow, gcache, grope, gmasked, 1, 2 * VALID, CAP, true,
            heads, LORA, 256, rot, 4096, 10000.0f, 1.0f, 1.0f, 1.0f, 32.0f, 1.0f), "masked attention");
        require_ok(ds4_gpu_tensor_read(gout, 0, actual, bytes), "masked attention read");
        require_prefill_dispatch(DS4_GPU_GLM53_PREFILL_INDEXED_ATTN, false, "masked attention fallback");
        require_ok(memcmp(expected, actual, bytes) == 0, "invalid rows preserve exact valid-row attention");
    }
    ds4_gpu_tensor_free(gout); ds4_gpu_tensor_free(gmasked); ds4_gpu_tensor_free(gcompact);
    ds4_gpu_tensor_free(grope); ds4_gpu_tensor_free(gcache); ds4_gpu_tensor_free(glow); ds4_gpu_tensor_free(gq);
}

/* Exactness oracle for the Q4_K routed-expert tail cull.
 *
 * kernel_mul_mm_id_q4_K_{f32,f16}_tail_cull differ from the kernels beside
 * them only in that the SIMDgroup pair owning routed rows 16..31 skips its
 * MMA and store when the expert's final 32-row tile holds 16 rows or fewer.
 * Those outputs are padding rows nothing reads, so both the f16 mid and the
 * summed f32 output must be byte-identical.  The per-expert row counts below
 * cover every final-tile size that matters: exact multiples of 32, 16 or
 * fewer, and 17 or more.
 *
 * Test mode forces the synthetic shape through the cull and records dispatch
 * coverage, so unsupported/default-off devices cannot silently compare the
 * reference with itself. */
static void check_glm53_routed_moe_tail_cull(uint8_t *model,
                                             uint64_t model_bytes,
                                             uint64_t gate_offset,
                                             uint64_t up_offset,
                                             uint64_t down_offset) {
    enum {
        MOE_EXPERTS = 36,
        MOE_USED = 8,
        MOE_DIM = 256,
        MOE_TOKENS = 384,
        MOE_Q4_K_ROW_BYTES = 144,          /* one 256-element Q4_K block */
        MOE_Q4_K_TYPE = 12,                /* GGUF type code for Q4_K */
        MOE_EXPERT_BYTES = MOE_DIM * MOE_Q4_K_ROW_BYTES,
    };
    uint32_t target_rows[MOE_EXPERTS];
    for (uint32_t i = 0; i < 32u; i++) target_rows[i] = 64u + i;
    target_rows[32] = 0u;  /* empty expert */
    target_rows[33] = target_rows[34] = 256u;
    target_rows[35] = 16u; /* total: 384 tokens * 8 distinct experts */
    static const uint16_t scale_bits[] = {
        0x2c00u, 0xac00u, 0x3400u, 0xb400u, 0x1c00u, 0x9c00u, 0x3800u, 0x2400u,
        0x0001u, 0x8001u, 0x03ffu, 0x83ffu, 0x0000u, 0x8000u,
    };
    const uint64_t matrix_bytes = (uint64_t)MOE_EXPERTS * MOE_EXPERT_BYTES;
    require_ok(down_offset + matrix_bytes <= model_bytes,
               "routed MoE expert weights fit the fixture model");

    uint64_t rng = 0xc3a5c85c97cb3127ull;
#define MOE_NEXT_BYTE() ( \
    rng = rng * 6364136223846793005ull + 1442695040888963407ull, \
    (uint8_t)(rng >> 33))
    const uint64_t offsets[3] = { gate_offset, up_offset, down_offset };
    for (int m = 0; m < 3; m++) {
        for (uint32_t row = 0; row < MOE_EXPERTS * MOE_DIM; row++) {
            uint8_t *dst = model + offsets[m] + (uint64_t)row * MOE_Q4_K_ROW_BYTES;
            const uint16_t d = scale_bits[(row + (uint32_t)m) % (sizeof(scale_bits) / sizeof(scale_bits[0]))];
            const uint16_t dmin = scale_bits[(row + (uint32_t)m + 3u) % (sizeof(scale_bits) / sizeof(scale_bits[0]))];
            memcpy(dst + 0, &d, sizeof(d));
            memcpy(dst + 2, &dmin, sizeof(dmin));
            for (uint32_t i = 4; i < MOE_Q4_K_ROW_BYTES; i++) dst[i] = MOE_NEXT_BYTE();
        }
    }

    const uint64_t x_elems = (uint64_t)MOE_TOKENS * MOE_DIM;
    const uint64_t route_elems = (uint64_t)MOE_TOKENS * MOE_USED;
    const uint64_t mid_elems = route_elems * MOE_DIM;
    const uint64_t out_elems = (uint64_t)MOE_TOKENS * MOE_DIM;

    float *x_host = malloc(x_elems * sizeof(float));
    int32_t *sel_host = malloc(route_elems * sizeof(int32_t));
    float *w_host = malloc(route_elems * sizeof(float));
    float *mid_ref = malloc(mid_elems * sizeof(float));
    float *mid_cull = malloc(mid_elems * sizeof(float));
    float *out_ref = malloc(out_elems * sizeof(float));
    float *out_cull = malloc(out_elems * sizeof(float));
    require_ok(x_host && sel_host && w_host && mid_ref && mid_cull && out_ref && out_cull,
               "routed MoE host allocation");
    for (uint64_t i = 0; i < x_elems; i++) {
        rng = rng * 6364136223846793005ull + 1442695040888963407ull;
        x_host[i] = (float)((int32_t)(uint32_t)(rng >> 32) / 1073741824.0) - 1.0f;
    }
    for (uint64_t i = 0; i < route_elems; i++) {
        rng = rng * 6364136223846793005ull + 1442695040888963407ull;
        w_host[i] = 0.05f + (float)(rng >> 40) / 8388608.0f;
    }
    /* Hand every token the eight experts with the most rows still owed, which
     * realizes target_rows exactly and keeps a token's experts distinct. */
    uint32_t remaining[MOE_EXPERTS];
    memcpy(remaining, target_rows, sizeof(remaining));
    for (uint32_t t = 0; t < MOE_TOKENS; t++) {
        bool taken[MOE_EXPERTS] = { false };
        for (uint32_t s = 0; s < MOE_USED; s++) {
            uint32_t best = MOE_EXPERTS;
            for (uint32_t e = 0; e < MOE_EXPERTS; e++) {
                if (taken[e]) continue;
                if (best == MOE_EXPERTS || remaining[e] > remaining[best]) best = e;
            }
            require_ok(best < MOE_EXPERTS && remaining[best] > 0,
                       "routed MoE route construction");
            taken[best] = true;
            remaining[best]--;
            sel_host[(uint64_t)t * MOE_USED + s] = (int32_t)best;
        }
    }
#undef MOE_NEXT_BYTE

    ds4_gpu_tensor *x_gpu = ds4_gpu_tensor_alloc(x_elems * sizeof(float));
    ds4_gpu_tensor *sel_gpu = ds4_gpu_tensor_alloc(route_elems * sizeof(int32_t));
    ds4_gpu_tensor *w_gpu = ds4_gpu_tensor_alloc(route_elems * sizeof(float));
    ds4_gpu_tensor *mid_gpu = ds4_gpu_tensor_alloc(mid_elems * sizeof(float));
    ds4_gpu_tensor *out_gpu = ds4_gpu_tensor_alloc(out_elems * sizeof(float));
    require_ok(x_gpu && sel_gpu && w_gpu && mid_gpu && out_gpu,
               "routed MoE GPU allocation");
    require_ok(ds4_gpu_tensor_write(x_gpu, 0, x_host, x_elems * sizeof(float)) &&
               ds4_gpu_tensor_write(sel_gpu, 0, sel_host, route_elems * sizeof(int32_t)) &&
               ds4_gpu_tensor_write(w_gpu, 0, w_host, route_elems * sizeof(float)),
               "routed MoE input write");

    const uint32_t poison_bits = 0x7fc01234u;
    float poison;
    memcpy(&poison, &poison_bits, sizeof(poison));
    for (int cull = 0; cull < 2; cull++) {
        const char *what = cull ? "routed MoE tail cull" : "routed MoE reference";
        if (cull) {
            require_ok(unsetenv("DS4_METAL_DISABLE_GLM53_PREFILL_MOE_TAIL_CULL") == 0, what);
        } else {
            require_ok(setenv("DS4_METAL_DISABLE_GLM53_PREFILL_MOE_TAIL_CULL", "1", 1) == 0, what);
        }
        require_ok(ds4_gpu_tensor_fill_f32(mid_gpu, poison, mid_elems) &&
                   ds4_gpu_tensor_fill_f32(out_gpu, poison, out_elems), what);
        require_ok(ds4_gpu_glm_routed_moe_batch_tensor(
                       out_gpu, mid_gpu, model, model_bytes,
                       gate_offset, up_offset, down_offset,
                       MOE_Q4_K_TYPE, MOE_Q4_K_TYPE, MOE_Q4_K_TYPE,
                       MOE_EXPERT_BYTES, MOE_Q4_K_ROW_BYTES,
                       MOE_EXPERT_BYTES, MOE_Q4_K_ROW_BYTES,
                       MOE_EXPERT_BYTES, MOE_Q4_K_ROW_BYTES,
                       MOE_DIM, MOE_DIM, MOE_DIM,
                       sel_gpu, w_gpu, MOE_EXPERTS, MOE_USED,
                       10.0f, 0u, x_gpu, MOE_TOKENS,
                       MOE_USED * MOE_DIM, true), what);
        require_prefill_dispatch(DS4_GPU_GLM53_PREFILL_MOE_TAIL_CULL,
                                  cull != 0, what);
        require_ok(ds4_gpu_tensor_read(mid_gpu, 0, cull ? mid_cull : mid_ref,
                                       mid_elems * sizeof(float)) &&
                   ds4_gpu_tensor_read(out_gpu, 0, cull ? out_cull : out_ref,
                                       out_elems * sizeof(float)),
                   what);
    }
    require_ok(unsetenv("DS4_METAL_DISABLE_GLM53_PREFILL_MOE_TAIL_CULL") == 0,
               "routed MoE tail cull switch clear");

    /* The f16 mid occupies the first half of the f32 mid buffer. */
    if (memcmp(mid_ref, mid_cull, (size_t)(mid_elems * sizeof(uint16_t))) != 0 ||
        memcmp(out_ref, out_cull, (size_t)(out_elems * sizeof(float))) != 0) {
        for (uint64_t i = 0; i < out_elems; i++) {
            if (memcmp(&out_ref[i], &out_cull[i], sizeof(float)) == 0) continue;
            fprintf(stderr,
                    "routed MoE tail cull: output %llu is %.9g, reference %.9g\n",
                    (unsigned long long)i, (double)out_cull[i], (double)out_ref[i]);
            break;
        }
        fprintf(stderr, "routed MoE tail cull is not bit-identical\n");
        exit(1);
    }

    ds4_gpu_tensor_free(out_gpu);
    ds4_gpu_tensor_free(mid_gpu);
    ds4_gpu_tensor_free(w_gpu);
    ds4_gpu_tensor_free(sel_gpu);
    ds4_gpu_tensor_free(x_gpu);
    free(out_cull);
    free(out_ref);
    free(mid_cull);
    free(mid_ref);
    free(w_host);
    free(sel_host);
    free(x_host);
}

/* Exactness oracle for the blocked GLM 5.3 KDA prepare kernel.
 *
 * kernel_glm53_kda_prefill_prepare_blocked splits the serial kernel's token
 * loop across (block, head) threadgroups and carries the three-row causal
 * convolution history in registers instead of shifting it through the device
 * conv state.  Every value keeps the serial kernel's expression and order, so
 * the normalized q and k, the silu'd v, the decay gate, the recurrence output
 * and the outgoing conv and recurrent states must all match bit for bit --
 * including where the window still reaches into the incoming conv state
 * (the first three rows) and on the short trailing block.
 *
 * This overwrites the KDA convolution fixture weights, so it runs after the
 * checks that use them. */
static void check_glm53_kda_prepare_blocked(uint8_t *model,
                                            uint64_t model_bytes,
                                            uint64_t q_conv_offset,
                                            uint64_t k_conv_offset,
                                            uint64_t v_conv_offset,
                                            uint64_t a_log_offset,
                                            uint64_t dt_bias_offset,
                                            uint64_t norm_offset) {
    enum { H = 64, D = 128, P = H * D, MAX_TOKENS = 2048 };
    enum { Q, K, V, GATE, OGATE, BETA, CONV, STATE, OUT, NBUF };
    /* Production head count, partial blocks, and explicit fallback boundaries. */
    static const uint32_t tokens[] = { 2048, 1596, 65, 33, 17, 4, 3, 1 };
    static const struct {
        uint32_t block, values;
        bool last_first, profile, batch, split, rollback;
    } variants[] = {
        {0, 1, false, false, false, false, false}, /* serial reference */
        {4, 2, false, false, false, false, false},
        {16, 2, false, false, false, false, false},
        {32, 2, false, false, false, false, false},
        {64, 4, false, false, false, false, false},
        {32, 2, true, false, false, false, false}, /* deterministic race regression */
        {32, 2, false, true, false, false, false}, /* profiler with an owned CB */
        {32, 2, false, true, true, false, false},  /* profiler preserves caller batch */
        {32, 2, true, false, true, true, false},   /* continue 33 + 32 tokens */
        {32, 2, false, false, false, false, true}, /* aggregate rollback */
    };
    const bool inherited_profile = getenv("DS4_METAL_PROFILE_KDA_PREFILL") != NULL;
    require_ok(norm_offset + D * sizeof(float) <= model_bytes,
               "production KDA weights fit fixture");
    uint64_t rng = 0x2545f4914f6cdd1dull;
#define KP_UNIT() ( \
    rng = rng * 6364136223846793005ull + 1442695040888963407ull, \
    (float)((int32_t)(uint32_t)(rng >> 32) / 1073741824.0) - 1.0f)
    const uint64_t conv_offsets[] = { q_conv_offset, k_conv_offset, v_conv_offset };
    for (unsigned m = 0; m < 3; m++) {
        float *w = (float *)(model + conv_offsets[m]);
        for (unsigned i = 0; i < P * 4u; i++) w[i] = 0.4f * KP_UNIT();
    }
    for (unsigned i = 0; i < P; i++) ((float *)(model + dt_bias_offset))[i] = 0.2f * KP_UNIT();
    for (unsigned i = 0; i < H; i++) ((float *)(model + a_log_offset))[i] = 0.3f * KP_UNIT();
    for (unsigned i = 0; i < D; i++) ((float *)(model + norm_offset))[i] = 1.0f + 0.1f * KP_UNIT();

    uint64_t elements[NBUF];
    float *input[NBUF], *reference[NBUF];
    ds4_gpu_tensor *gpu[NBUF];
    for (unsigned b = 0; b < NBUF; b++) {
        elements[b] = b == BETA ? (uint64_t)MAX_TOKENS * H :
                      b == CONV ? 9u * P : b == STATE ? (uint64_t)P * D :
                      (uint64_t)MAX_TOKENS * P;
        input[b] = malloc(elements[b] * sizeof(float));
        reference[b] = malloc(elements[b] * sizeof(float));
        gpu[b] = ds4_gpu_tensor_alloc(elements[b] * sizeof(float));
        require_ok(input[b] && reference[b] && gpu[b], "KDA oracle allocation");
        for (uint64_t i = 0; i < elements[b]; i++) input[b][i] = KP_UNIT() * (b == STATE ? 0.1f : 1.0f);
    }
#undef KP_UNIT
    float *actual = malloc((uint64_t)MAX_TOKENS * P * sizeof(float));
    require_ok(actual != NULL, "KDA readback allocation");
    static const unsigned compared[] = { Q, K, V, GATE, CONV, STATE, OUT };
    static const char *const names[] = { "q", "k", "v", "decay", "output gate", "beta", "conv state", "recurrent state", "output" };
    const uint32_t poison_bits = 0x7fc01234u;
    float poison;
    memcpy(&poison, &poison_bits, sizeof(poison));
    for (unsigned c = 0; c < sizeof(tokens) / sizeof(tokens[0]); c++) {
        const uint32_t n = tokens[c];
        uint64_t bytes[NBUF];
        for (unsigned b = 0; b < NBUF; b++) {
            bytes[b] = (b == BETA ? (uint64_t)n * H :
                        (b == CONV || b == STATE) ? elements[b] : (uint64_t)n * P) * sizeof(float);
        }
        for (unsigned variant = 0; variant < sizeof(variants) / sizeof(variants[0]); variant++) {
            if (variants[variant].split && n != 65u) continue;
            const uint32_t block = variants[variant].block;
            char what[128], text[16];
            snprintf(what, sizeof(what), "KDA n=%u block=%u values=%u last-first=%u profile=%u batch=%u split=%u rollback=%u",
                     n, block, variants[variant].values, variants[variant].last_first, variants[variant].profile,
                     variants[variant].batch, variants[variant].split, variants[variant].rollback);
            if (variants[variant].rollback) setenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING", "1", 1);
            else unsetenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING");
            if (block == 0u) {
                setenv("DS4_METAL_DISABLE_GLM53_PREFILL_KDA_PREPARE", "1", 1);
                setenv("DS4_METAL_DISABLE_GLM53_PREFILL_KDA_RECURRENCE", "1", 1);
            } else {
                unsetenv("DS4_METAL_DISABLE_GLM53_PREFILL_KDA_PREPARE");
                unsetenv("DS4_METAL_DISABLE_GLM53_PREFILL_KDA_RECURRENCE");
            }
            snprintf(text, sizeof(text), "%u", block);
            setenv("DS4_METAL_GLM53_PREFILL_KDA_PREPARE_BLOCK", text, 1);
            snprintf(text, sizeof(text), "%u", variants[variant].values);
            setenv("DS4_METAL_GLM53_PREFILL_KDA_VALUES_PER_SG", text, 1);
            if (variants[variant].profile) setenv("DS4_METAL_PROFILE_KDA_PREFILL", "1", 1);
            else unsetenv("DS4_METAL_PROFILE_KDA_PREFILL");
            ds4_gpu_test_set_flags(DS4_GPU_TEST_GLM53_PREFILL |
                (variants[variant].last_first ? DS4_GPU_TEST_GLM53_KDA_LAST_BLOCK_FIRST : 0u));
            for (unsigned b = 0; b < OUT; b++) require_ok(ds4_gpu_tensor_write(gpu[b], 0, input[b], bytes[b]), what);
            require_ok(ds4_gpu_tensor_fill_f32(gpu[OUT], poison, (uint64_t)n * P), what);
            if (variants[variant].batch) require_ok(ds4_gpu_begin_commands(), what);
            const unsigned chunks = variants[variant].split ? 2 : 1;
            for (unsigned chunk = 0; chunk < chunks; chunk++) {
                const uint32_t offset = chunk == 0 ? 0 : 33;
                const uint32_t rows = chunks == 1 ? n : chunk == 0 ? 33 : n - 33;
                ds4_gpu_tensor *views[6];
                for (unsigned b = Q; b <= BETA; b++) {
                    const uint64_t stride = (b == BETA ? H : P) * sizeof(float);
                    views[b] = ds4_gpu_tensor_view(gpu[b], (uint64_t)offset * stride, (uint64_t)rows * stride);
                    require_ok(views[b] != NULL, what);
                }
                ds4_gpu_tensor *out = ds4_gpu_tensor_view(gpu[OUT], (uint64_t)offset * P * sizeof(float), (uint64_t)rows * P * sizeof(float));
                require_ok(out != NULL, what);
                require_ok(ds4_gpu_glm53_kda_prefill(out, gpu[CONV], gpu[STATE],
                    views[Q], views[K], views[V], views[GATE], views[BETA], views[OGATE],
                    model, model_bytes, q_conv_offset, k_conv_offset, v_conv_offset,
                    a_log_offset, dt_bias_offset, norm_offset, H, rows, -5.0f, 1e-5f), what);
                ds4_gpu_tensor_free(out);
                for (unsigned b = Q; b <= BETA; b++) ds4_gpu_tensor_free(views[b]);
            }
            require_ok(ds4_gpu_end_commands() == (variants[variant].batch ? 1 : 0), "KDA preserves command-batch ownership");
            const uint32_t largest_chunk = chunks == 1 ? n : 33;
            const uint32_t expected_dispatches = variants[variant].rollback ? 0u :
                (block != 0u && largest_chunk > block ? DS4_GPU_GLM53_PREFILL_KDA_PREPARE : 0u) |
                (block != 0u && largest_chunk >= 32u ? DS4_GPU_GLM53_PREFILL_KDA_RECURRENCE : 0u);
            require_prefill_dispatch(expected_dispatches, true, what);
            for (unsigned j = 0; j < sizeof(compared) / sizeof(compared[0]); j++) {
                const unsigned b = compared[j];
                float *dst = variant == 0 ? reference[b] : actual;
                require_ok(ds4_gpu_tensor_read(gpu[b], 0, dst, bytes[b]), what);
                if (variant == 0 || memcmp(reference[b], actual, bytes[b]) == 0) continue;
                for (uint64_t i = 0; i < bytes[b] / sizeof(float); i++) {
                    if (memcmp(&reference[b][i], &actual[i], sizeof(float)) == 0) continue;
                    fprintf(stderr, "%s: %s[%llu] %.9g != %.9g\n", what, names[b],
                            (unsigned long long)i, actual[i], reference[b][i]);
                    break;
                }
                exit(1);
            }
        }
    }
    unsetenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING");
    unsetenv("DS4_METAL_GLM53_PREFILL_KDA_PREPARE_BLOCK");
    unsetenv("DS4_METAL_GLM53_PREFILL_KDA_VALUES_PER_SG");
    unsetenv("DS4_METAL_DISABLE_GLM53_PREFILL_KDA_PREPARE");
    unsetenv("DS4_METAL_DISABLE_GLM53_PREFILL_KDA_RECURRENCE");
    if (inherited_profile) setenv("DS4_METAL_PROFILE_KDA_PREFILL", "1", 1);
    else unsetenv("DS4_METAL_PROFILE_KDA_PREFILL");
    ds4_gpu_test_set_flags(DS4_GPU_TEST_GLM53_PREFILL);
    free(actual);
    for (unsigned b = 0; b < NBUF; b++) {
        ds4_gpu_tensor_free(gpu[b]);
        free(reference[b]);
        free(input[b]);
    }
}

/* Compare every intermediate as score-row ownership changes, including the
 * indexer's three sentinels and the first count above the tuned limit. */
static void check_glm53_dsa_score_tile(uint8_t *model, uint64_t model_bytes,
                                       uint64_t value_offset) {
    enum { H = 64, L = 512, V = 256, CAP = 2052, ROW_BYTES = 544 };
    require_ok(value_offset + (uint64_t)H * V * ROW_BYTES <= model_bytes,
               "DSA tile fixture range");
    for (uint32_t row = 0; row < H * V; row++) {
        uint8_t *w = model + value_offset + (uint64_t)row * ROW_BYTES;
        for (uint32_t b = 0; b < L / 32; b++) {
            const uint16_t scale = (uint16_t)(0x2000u | ((row & 1u) << 15));
            memcpy(w + b * 34u, &scale, sizeof(scale));
            for (uint32_t i = 0; i < 32; i++)
                w[b * 34u + 2u + i] = (uint8_t)(row * 31u + b * 7u + i * 3u);
        }
    }
    float *low = malloc(H * L * sizeof(float));
    uint16_t *kv = malloc(CAP * L * sizeof(uint16_t));
    uint32_t selected[CAP];
    require_ok(low && kv, "DSA tile host inputs");
    for (uint32_t i = 0; i < H * L; i++)
        low[i] = (float)((int)((i * 1664525u + 1013904223u) % 4097u) - 2048) * 0.0007f;
    for (uint32_t i = 0; i < CAP * L; i++)
        kv[i] = f32_to_f16((float)((int)((i * 1664525u + 1013904223u) % 8191u) - 4095) * 0.0003f);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(H * L * sizeof(float));
    ds4_gpu_tensor *cache = ds4_gpu_tensor_alloc(CAP * L * sizeof(uint16_t));
    ds4_gpu_tensor *ids = ds4_gpu_tensor_alloc(sizeof(selected));
    require_ok(x && cache && ids, "DSA tile input buffers");
    require_ok(ds4_gpu_tensor_write(x, 0, low, H * L * sizeof(float)) &&
               ds4_gpu_tensor_write(cache, 0, kv, CAP * L * sizeof(uint16_t)),
               "DSA tile input writes");
    uint64_t sizes[] = { H * V * sizeof(float), H * CAP * sizeof(float),
                         H * L * sizeof(float), H * sizeof(float) };
    ds4_gpu_tensor *out[4];
    float *reference[4], *actual[4];
    for (unsigned i = 0; i < 4; i++) {
        out[i] = ds4_gpu_tensor_alloc(sizes[i]);
        reference[i] = malloc(sizes[i]); actual[i] = malloc(sizes[i]);
        require_ok(out[i] && reference[i] && actual[i], "DSA tile output allocation");
    }
    const uint32_t counts[] = { 1, 33, 257, 2051, 2052 };
    for (unsigned c = 0; c < sizeof(counts) / sizeof(counts[0]); c++) {
        const uint32_t n = counts[c];
        sizes[1] = (uint64_t)H * n * sizeof(float);
        for (uint32_t i = 0; i < n; i++) selected[i] = (i * 31u) % CAP;
        if (n > 3) for (uint32_t i = n - 3; i < n; i++) selected[i] = UINT32_MAX;
        require_ok(ds4_gpu_tensor_write(ids, 0, selected, n * sizeof(uint32_t)),
                   "DSA tile selections");
        for (unsigned arm = 0; arm < 3; arm++) {
            if (arm == 0) setenv("DS4_METAL_DISABLE_GLM53_DSA_SCORE_TILE", "1", 1);
            else unsetenv("DS4_METAL_DISABLE_GLM53_DSA_SCORE_TILE");
            if (arm == 2) setenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING", "1", 1);
            for (unsigned i = 0; i < 4; i++)
                require_ok(ds4_gpu_tensor_fill_f32(out[i], -17.0f, sizes[i] / sizeof(float)),
                           "DSA tile poison outputs");
            require_ok(ds4_gpu_glm_attention_indexed_decode_exact_typed_tensor(
                           out[0], out[1], out[2], out[3], x, cache,
                           model, model_bytes, value_offset, 8, ids,
                           n, CAP, true, H, L, 256, 0, V), "DSA tile dispatch");
            require_prefill_dispatch(DS4_GPU_GLM53_DSA_SCORE_TILE,
                                     arm == 1 && n <= 2051, "DSA tile coverage");
            for (unsigned i = 0; i < 4; i++) {
                require_ok(ds4_gpu_tensor_read(out[i], 0, actual[i], sizes[i]), "DSA tile read");
                if (arm == 0) memcpy(reference[i], actual[i], sizes[i]);
                else require_ok(memcmp(reference[i], actual[i], sizes[i]) == 0,
                                "DSA tile bitwise intermediate/output");
            }
            unsetenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING");
        }
    }
    for (unsigned i = 0; i < 4; i++) {
        ds4_gpu_tensor_free(out[i]); free(reference[i]); free(actual[i]);
    }
    ds4_gpu_tensor_free(ids); ds4_gpu_tensor_free(cache); ds4_gpu_tensor_free(x);
    free(kv); free(low);
}

static void check_glm53_kda_inputs(uint8_t *model, uint64_t model_bytes,
                                  uint64_t offset) {
    const uint32_t widths[6] = {8192, 8192, 8192, 128, 128, 64};
    uint64_t offsets[6], end = offset;
    ds4_gpu_tensor *out[6];
    float *reference[6];
    for (unsigned slot = 0; slot < 6; slot++) {
        offsets[slot] = end;
        const uint64_t count = (uint64_t)4096u * widths[slot];
        end += count * sizeof(uint16_t);
        require_ok(end <= model_bytes, "KDA input fixture range");
        uint16_t *w = (uint16_t *)(void *)(model + offsets[slot]);
        for (uint64_t i = 0; i < count; i++) {
            const uint32_t h = (uint32_t)(i + slot * 8191u) * 1664525u + 1013904223u;
            w[i] = (uint16_t)((0x3b00u + h % 2048u) | ((h & 1u) << 15));
            if (i % 127u == 0) w[i] = (uint16_t)((h & 0x8000u) | (h % 128u));
        }
        out[slot] = ds4_gpu_tensor_alloc(widths[slot] * sizeof(float));
        reference[slot] = malloc(widths[slot] * sizeof(float));
        require_ok(out[slot] && reference[slot], "KDA input output buffers");
    }
    float input[4096], actual[8192];
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(input));
    require_ok(x != NULL, "KDA input activation buffer");
    for (unsigned fixture = 0; fixture < 4; fixture++) {
        for (uint32_t i = 0; i < 4096; i++) {
            input[i] = (float)((int)((i * 1664525u + 1013904223u) % 4097u) - 2048) * 0.00123f;
            if (fixture == 1) input[i] = (i & 1u) ? -0.0f : 0.0f;
            if (fixture == 2) input[i] *= 100000.0f;
            if (fixture == 3) input[i] = (i & 1u) ? -0.00003f : 0.00003f;
        }
        require_ok(ds4_gpu_tensor_write(x, 0, input, sizeof(input)), "KDA input activation write");
        for (unsigned arm = 0; arm < 4; arm++) {
            if (arm == 2) setenv("DS4_METAL_DISABLE_GLM53_KDA_INPUTS", "1", 1);
            if (arm == 3) setenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING", "1", 1);
            for (unsigned i = 0; i < 6; i++)
                require_ok(ds4_gpu_tensor_fill_f32(out[i], -17.0f, widths[i]), "KDA input poison");
            const int fused = arm != 0 && ds4_gpu_glm53_kda_inputs_bf16(
                out, offsets, model, model_bytes, x);
            require_ok(fused == (arm == 1), "KDA input policy selection");
            if (!fused) {
                for (unsigned i = 0; i < 6; i++)
                    require_ok(ds4_gpu_glm53_matmul_bf16(out[i], model, model_bytes,
                        offsets[i], 4096, widths[i], x, 1), "KDA input separate reference");
            }
            require_prefill_dispatch(DS4_GPU_GLM53_KDA_INPUTS, arm == 1, "KDA input coverage");
            for (unsigned i = 0; i < 6; i++) {
                require_ok(ds4_gpu_tensor_read(out[i], 0, actual, widths[i] * sizeof(float)),
                           "KDA input output read");
                if (arm == 0) memcpy(reference[i], actual, widths[i] * sizeof(float));
                else require_ok(memcmp(reference[i], actual, widths[i] * sizeof(float)) == 0,
                                "KDA input bitwise six projections");
            }
            unsetenv("DS4_METAL_DISABLE_GLM53_KDA_INPUTS");
            unsetenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING");
        }
    }
    const uint64_t valid = offsets[5];
    offsets[5] = model_bytes - 1u;
    require_ok(!ds4_gpu_glm53_kda_inputs_bf16(out, offsets, model, model_bytes, x),
               "KDA input invalid last weight range");
    offsets[5] = valid;
    ds4_gpu_tensor *last = out[5]; out[5] = NULL;
    require_ok(!ds4_gpu_glm53_kda_inputs_bf16(out, offsets, model, model_bytes, x),
               "KDA input missing last output");
    out[5] = last;
    require_prefill_dispatch(DS4_GPU_GLM53_KDA_INPUTS, false, "KDA input invalid range coverage");
    for (unsigned i = 0; i < 6; i++) {
        ds4_gpu_tensor_free(out[i]); free(reference[i]);
    }
    ds4_gpu_tensor_free(x);
}

/* Independent persistent histories catch a scheduling error even when the
 * current normalized output hides it. Also exercise the untuned shapes. */
static void check_glm53_kda_decode_values4(uint8_t *model, uint64_t model_bytes,
                                          uint64_t offset) {
    enum { H = 64, D = 128, P = H * D, Q = 0, K = P * 16,
           V = K + P * 16, A = V + P * 16, DT = A + H * 4,
           NORM = DT + P * 4 };
    require_ok(offset + NORM + D * sizeof(float) <= model_bytes,
               "KDA decode values fixture range");
    float *w = (float *)(void *)(model + offset);
    for (uint32_t i = 0; i < A / 4; i++)
        w[i] = (float)((int)((i * 1664525u + 1013904223u) % 1021u) - 510) * 0.001f;
    for (uint32_t i = 0; i < H; i++) w[A / 4 + i] = -1.5f + (float)(i % 7u) * 0.1f;
    for (uint32_t i = 0; i < P; i++) w[DT / 4 + i] = (float)((int)(i % 17u) - 8) * 0.1f;
    for (uint32_t i = 0; i < D; i++) w[NORM / 4 + i] = 0.8f + (float)(i % 23u) * 0.01f;
    const uint32_t shapes[][2] = {{64, 1}, {64, 2}, {2, 1}};
    for (unsigned shape = 0; shape < sizeof(shapes) / sizeof(shapes[0]); shape++) {
        const uint32_t heads = shapes[shape][0], rows = shapes[shape][1];
        const uint32_t n = heads * D * rows;
        const uint64_t sizes[] = {n * sizeof(float), (uint64_t)n * 9 * sizeof(float),
                                  (uint64_t)n * D * sizeof(float)};
        float *actual = malloc(sizes[2]), *reference = malloc(sizes[2]);
        ds4_gpu_tensor *input[6], *out[3][3];
        require_ok(actual && reference, "KDA decode values host buffers");
        for (unsigned i = 0; i < 6; i++) {
            input[i] = ds4_gpu_tensor_alloc(sizes[0]);
            require_ok(input[i] != NULL, "KDA decode values input buffer");
        }
        for (unsigned b = 0; b < 3; b++) {
            for (uint64_t j = 0; j < sizes[b] / sizeof(float); j++)
                actual[j] = (float)((int)((j * 1664525u + 1013904223u) % 2047u) - 1023) * 0.0005f;
            for (unsigned arm = 0; arm < 3; arm++) {
                out[arm][b] = ds4_gpu_tensor_alloc(sizes[b]);
                require_ok(out[arm][b] != NULL, "KDA decode values output buffer");
                require_ok(ds4_gpu_tensor_write(out[arm][b], 0, actual, sizes[b]),
                           "KDA decode values initial state");
            }
        }
        for (unsigned step = 0; step < 16; step++) {
            for (unsigned i = 0; i < 6; i++) {
                for (uint32_t j = 0; j < n; j++) {
                    const uint32_t h = (j + step * n + i * n * 7u) * 1664525u + 1013904223u;
                    actual[j] = step == 7 ? 0.0f : (float)((int)(h % 8191u) - 4095) * 0.0007f;
                }
                require_ok(ds4_gpu_tensor_write(input[i], 0, actual, sizes[0]),
                           "KDA decode values input write");
            }
            for (unsigned arm = 0; arm < 3; arm++) {
                if (arm == 0) setenv("DS4_METAL_DISABLE_GLM53_DECODE_KDA_VALUES4", "1", 1);
                else unsetenv("DS4_METAL_DISABLE_GLM53_DECODE_KDA_VALUES4");
                if (arm == 2) setenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING", "1", 1);
                require_ok(ds4_gpu_tensor_fill_f32(out[arm][0], -17.0f, n),
                           "KDA decode values poison output");
                require_ok(ds4_gpu_glm53_kda_decode(
                    out[arm][0], out[arm][1], out[arm][2], input[0], input[1], input[2],
                    input[3], input[4], input[5], model, model_bytes, offset + Q, offset + K,
                    offset + V, offset + A, offset + DT, offset + NORM,
                    heads, rows, -5.0f, 1e-6f), "KDA decode values dispatch");
                require_prefill_dispatch(DS4_GPU_GLM53_DECODE_KDA_VALUES4,
                                         arm == 1 && shape == 0, "KDA decode values coverage");
                unsetenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING");
            }
            for (unsigned b = 0; b < 3; b++) {
                require_ok(ds4_gpu_tensor_read(out[0][b], 0, reference, sizes[b]),
                           "KDA decode values reference read");
                for (unsigned arm = 1; arm < 3; arm++) {
                    require_ok(ds4_gpu_tensor_read(out[arm][b], 0, actual, sizes[b]),
                               "KDA decode values candidate read");
                    require_ok(memcmp(reference, actual, sizes[b]) == 0,
                               "KDA decode values bitwise output/history/state");
                }
            }
        }
        for (unsigned i = 0; i < 6; i++) ds4_gpu_tensor_free(input[i]);
        for (unsigned arm = 0; arm < 3; arm++)
            for (unsigned b = 0; b < 3; b++) ds4_gpu_tensor_free(out[arm][b]);
        free(actual); free(reference);
    }
}

static void check_glm53_bf16_short_rows(uint8_t *model, uint64_t model_bytes,
                                        uint64_t weight_offset) {
    enum { IN = 128, OUT = 8192 };
    require_ok(weight_offset + (uint64_t)IN * OUT * 2 <= model_bytes,
               "short BF16 fixture range");
    uint16_t *w = (uint16_t *)(void *)(model + weight_offset);
    for (uint32_t i = 0; i < IN * OUT; i++) {
        const uint32_t h = i * 1664525u + 1013904223u;
        w[i] = (uint16_t)((0x3b00u + h % 2048u) | ((h & 1u) << 15u));
    }
    float input[IN], reference[OUT], actual[OUT];
    for (uint32_t i = 0; i < IN; i++) input[i] = (float)((int)(i % 17u) - 8) * 0.012345f;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(input));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(actual));
    require_ok(x && out, "short BF16 buffers");
    require_ok(ds4_gpu_tensor_write(x, 0, input, sizeof(input)), "short BF16 input");
    require_ok(unsetenv("DS4_METAL_DISABLE_M3_ULTRA_GLM53_BF16_NSG4") == 0,
               "clear inherited BF16 grouping rollback");
    for (unsigned arm = 0; arm < 3; arm++) {
        if (arm == 0) setenv("DS4_METAL_DISABLE_GLM53_BF16_SHORT_ROWS", "1", 1);
        else unsetenv("DS4_METAL_DISABLE_GLM53_BF16_SHORT_ROWS");
        if (arm == 2) setenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING", "1", 1);
        require_ok(ds4_gpu_tensor_fill_f32(out, -17.0f, OUT), "short BF16 poison");
        require_ok(ds4_gpu_glm53_matmul_bf16(out, model, model_bytes,
                       weight_offset, IN, OUT, x, 1), "short BF16 dispatch");
        require_prefill_dispatch(DS4_GPU_GLM53_BF16_SHORT_ROWS, arm == 1,
                                 "short BF16 grouping coverage");
        require_ok(ds4_gpu_tensor_read(out, 0, actual, sizeof(actual)), "short BF16 read");
        if (arm == 0) memcpy(reference, actual, sizeof(actual));
        else require_ok(memcmp(reference, actual, sizeof(actual)) == 0,
                        "short BF16 bitwise output");
    }
    unsetenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING");
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
}

/* Exercise cross-SIMD probability reads and selected-ID publication in the
 * shared GLM router, including the 512-thread GLM-5.3 sort. */
static void check_glm_router_publication(uint8_t *model, size_t model_bytes,
                                         uint64_t bias_offset) {
    enum { MAX_EXPERTS = 512, ROWS = 17, USED = 8 };
    float logits[ROWS * MAX_EXPERTS], probs[ROWS * MAX_EXPERTS];
    float ref_probs[ROWS * MAX_EXPERTS], ref_weights[ROWS * USED];
    int32_t ref_selected[ROWS * USED];
    float weights[ROWS * USED];
    int32_t selected[ROWS * USED];
    float *bias = (float *)(void *)(model + bias_offset);
    ds4_gpu_tensor *in = ds4_gpu_tensor_alloc(sizeof(logits));
    ds4_gpu_tensor *pg = ds4_gpu_tensor_alloc(sizeof(probs));
    ds4_gpu_tensor *wg = ds4_gpu_tensor_alloc(sizeof(weights));
    ds4_gpu_tensor *sg = ds4_gpu_tensor_alloc(sizeof(selected));
    require_ok(in && pg && wg && sg, "GLM router buffers");
    const char *dump_path = getenv("DS4_TEST_GLM_ROUTER_DUMP");
    FILE *dump = dump_path ? fopen(dump_path, "wb") : NULL;
    require_ok(!dump_path || dump, "GLM router dump open");
    const uint32_t widths[] = {256, 288, 512};
    const uint32_t batches[] = {1, ROWS};
    for (unsigned wi = 0; wi < sizeof(widths) / sizeof(widths[0]); wi++) {
        const uint32_t n = widths[wi];
        for (unsigned bi = 0; bi < sizeof(batches) / sizeof(batches[0]); bi++) {
            const uint32_t rows = batches[bi];
            for (unsigned fixture = 0; fixture < 10; fixture++) {
                for (uint32_t e = 0; e < n; e++) {
                    /* Multiples of 37 distribute selected experts across
                     * SIMDgroups; fixture zero checks the lower-ID tie rule. */
                    bias[e] = fixture == 0 ? 0.0f :
                        2.0f * (float)((e * 37u + fixture * 13u) % n);
                    for (uint32_t row = 0; row < rows; row++) {
                        logits[row * n + e] = fixture < 2 ? 0.0f :
                            fixture == 2 ? (float)((int)((e * 17u + row * 11u) % 97u) - 48) * 0.25f :
                            ((e + row) % 3u == 0 ? -80.0f :
                             (e + row) % 3u == 1 ? 80.0f : 0.0f);
                    }
                }
                if (fixture >= 4) {
                    for (uint32_t e = 0; e < n; e++) {
                        uint32_t bits = 0;
                        if (fixture == 4) bits = e % 3u == 0 ? 0xff800000u : e % 3u == 1 ? 0x7f800000u : 0u;
                        if (fixture == 5) bits = 0xff800000u;
                        if (fixture == 6) bits = e == 37u ? 0x7fc12345u : 0u;
                        if (fixture == 7) bits = e == n - 1u ? 0xffc23456u : 0u;
                        if (fixture == 8) bits = e & 1u ? 0x80000000u : 0u;
                        memcpy(&bias[e], &bits, sizeof(bits));
                        if (fixture == 9) {
                            bits = 0xffc23456u;
                            for (uint32_t row = 0; row < rows; row++)
                                memcpy(&logits[row * n + 93u], &bits, sizeof(bits));
                        }
                    }
                }
                require_ok(ds4_gpu_tensor_write(in, 0, logits,
                    (uint64_t)rows * n * sizeof(float)), "GLM router logits");
                for (unsigned arm = 0; arm < 3; arm++) {
                    if (arm == 0) setenv("DS4_METAL_DISABLE_GLM53_ROUTER_TOP8", "1", 1);
                    else unsetenv("DS4_METAL_DISABLE_GLM53_ROUTER_TOP8");
                    if (arm == 2) setenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING", "1", 1);
                    require_ok(ds4_gpu_begin_commands(), "GLM router begin");
                    require_ok(ds4_gpu_tensor_fill_f32(pg, -17.0f, rows * n), "GLM router poison probabilities");
                    require_ok(ds4_gpu_tensor_fill_f32(sg, 0.0f, rows * USED), "GLM router clear selection");
                    require_ok(ds4_gpu_tensor_fill_f32(wg, -17.0f, rows * USED), "GLM router poison weights");
                    const int ok = rows == 1 ?
                        ds4_gpu_glm_router_select_tensor(sg, wg, pg, model,
                            model_bytes, bias_offset, in, n, USED, 2.5f) :
                        ds4_gpu_glm_router_select_batch_tensor(sg, wg, pg, model,
                            model_bytes, bias_offset, in, n, USED, 2.5f, rows);
                    require_ok(ok, "GLM router dispatch");
                    require_prefill_dispatch(DS4_GPU_GLM53_ROUTER_TOP8, n == 288 && arm == 1,
                                             "GLM router selection coverage");
                    require_ok(ds4_gpu_end_commands(), "GLM router end");
                    require_ok(ds4_gpu_tensor_read(pg, 0, probs,
                        (uint64_t)rows * n * sizeof(float)), "GLM router probabilities");
                    require_ok(ds4_gpu_tensor_read(sg, 0, selected,
                        (uint64_t)rows * USED * sizeof(int32_t)), "GLM router selected IDs");
                    require_ok(ds4_gpu_tensor_read(wg, 0, weights,
                        (uint64_t)rows * USED * sizeof(float)), "GLM router weights");
                    if (arm == 0) {
                        memcpy(ref_probs, probs, rows * n * sizeof(float));
                        memcpy(ref_selected, selected, rows * USED * sizeof(int32_t));
                        memcpy(ref_weights, weights, rows * USED * sizeof(float));
                    } else {
                        require_ok(memcmp(ref_probs, probs, rows * n * sizeof(float)) == 0,
                                   "GLM router bitwise probabilities");
                        require_ok(memcmp(ref_selected, selected, rows * USED * sizeof(int32_t)) == 0,
                                   "GLM router bitwise expert order");
                        require_ok(memcmp(ref_weights, weights, rows * USED * sizeof(float)) == 0,
                                   "GLM router bitwise weights");
                    }
                    unsetenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING");
                }
                for (uint32_t row = 0; fixture < 4 && row < rows; row++) {
                    float scores[MAX_EXPERTS];
                    int32_t ids[MAX_EXPERTS];
                    for (uint32_t e = 0; e < n; e++) {
                        const float x = logits[row * n + e];
                        const float z = expf(-fabsf(x));
                        const float expected = x >= 0 ? 1.0f / (1.0f + z) : z / (1.0f + z);
                        require_close("GLM router sigmoid", probs[row * n + e], expected, 1e-7f);
                        scores[e] = probs[row * n + e] + bias[e];
                        ids[e] = (int32_t)e;
                    }
                    for (uint32_t k = 0; k < USED; k++) {
                        uint32_t best = k;
                        for (uint32_t j = k + 1; j < n; j++) {
                            if (scores[ids[j]] > scores[ids[best]] ||
                                (scores[ids[j]] == scores[ids[best]] && ids[j] < ids[best])) best = j;
                        }
                        const int32_t tmp = ids[k]; ids[k] = ids[best]; ids[best] = tmp;
                        require_ok(selected[row * USED + k] == ids[k], "GLM router ordered selection");
                    }
                    volatile float sum = 0.0f;
                    for (uint32_t k = 0; k < USED; k++) sum = sum + probs[row * n + (uint32_t)ids[k]];
                    const float denom = fmaxf(sum, 6.103515625e-5f);
                    for (uint32_t k = 0; k < USED; k++) {
                        const float expected = probs[row * n + (uint32_t)ids[k]] / denom * 2.5f;
                        require_close("GLM router normalization", weights[row * USED + k], expected, 2e-7f);
                        if (fixture < 2) require_close("GLM router exact equal weights", weights[row * USED + k], 0.3125f, 0.0f);
                    }
                }
                if (dump) {
                    require_ok(fwrite(probs, sizeof(float), rows * n, dump) == rows * n, "GLM router dump probabilities");
                    require_ok(fwrite(selected, sizeof(int32_t), rows * USED, dump) == rows * USED, "GLM router dump IDs");
                    require_ok(fwrite(weights, sizeof(float), rows * USED, dump) == rows * USED, "GLM router dump weights");
                }
            }
        }
    }
    if (dump) require_ok(fclose(dump) == 0, "GLM router dump close");
    ds4_gpu_tensor_free(sg);
    ds4_gpu_tensor_free(wg);
    ds4_gpu_tensor_free(pg);
    ds4_gpu_tensor_free(in);
}

#endif /* __APPLE__: these oracles exercise Metal-specific dispatch switches. */

int main(void) {
    enum {
        D = 128,
        HEADS = 2,
        PROJECTION = HEADS * D,
        TOKENS = 17,
        Q_CONV_OFFSET = 0,
        K_CONV_OFFSET = 4096,
        V_CONV_OFFSET = 8192,
        A_LOG_OFFSET = 12288,
        DT_BIAS_OFFSET = 16384,
        NORM_OFFSET = 20480,
        POOL_NORM_OFFSET = 22528,
        POOL_BIAS_OFFSET = 24576,
        POOL_APE_OFFSET = 28672,
        BF16_OFFSET = 32768,
        BF16_IN = 64,
        BF16_OUT = 64,
        BF16_ROWS = 16,
        Q4_OFFSET = 49152,
        Q4_IN = 256,
        Q4_OUT = 37,
        Q4_ROWS = 3,
        Q8_OFFSET = 60000,
        /* Real GLM 5.3 widths: 4096 is kda_{q,k,v}, and 512/1024 the
         * low-rank gate projections. */
        WIDE512_OFFSET = 65536,   WIDE512_IN = 512,   WIDE512_OUT = 4,
        WIDE1024_OFFSET = 73728,  WIDE1024_IN = 1024, WIDE1024_OUT = 4,
        WIDE4096_OFFSET = 90112,  WIDE4096_IN = 4096, WIDE4096_OUT = 2,
        WIDE_ROWS = 3,
        /* Compare each compound HC producer against its unfused projection. */
        HC_N = 16384, HC_MIX = 24, HC_EMBD = 4096, HC_HC = 4,
        HC_F16W_OFFSET  = 131072,   /* HC_N * HC_MIX * 2 = 786432 */
        HC_BF16W_OFFSET = 917504,
        HC_SCALE_OFFSET = 1703936,  /* 3 floats  */
        HC_BASE_OFFSET  = 1703968,  /* 24 floats */
        HC_NORM_OFFSET  = 1704064,  /* 4096 floats */
        /* BF16 matvec + HC-expand epilogue fixture */
        FUSED_W_OFFSET = 1720448,   /* FUSED_IN * FUSED_OUT * 2 = 131072 */
        FUSED_IN = 1024, FUSED_OUT = 64, FUSED_HC = 4,
        /* Q8_0 value rows for the split-vs-generic attention check:
         * 16 heads x 8 values x 544 bytes = 69632 */
        SPLIT_V_OFFSET = 1851520,
        /* GLM 5.3 attn_k_b for the prefill qk-low oracle:
         * 64 heads x 512 rows x 272 bytes = 8912896 */
        QK_LOW_KB_OFFSET = 2097152,
        /* Synthetic Q4_K expert matrices for all tail sizes and an empty expert. */
        MOE_GATE_OFFSET = 11010048,
        MOE_UP_OFFSET   = MOE_GATE_OFFSET + 36 * 256 * 144,
        MOE_DOWN_OFFSET = MOE_UP_OFFSET + 36 * 256 * 144,
        KP_Q_OFFSET = MOE_DOWN_OFFSET + 36 * 256 * 144,
        KP_K_OFFSET = KP_Q_OFFSET + 64 * 128 * 4 * 4,
        KP_V_OFFSET = KP_K_OFFSET + 64 * 128 * 4 * 4,
        KP_A_OFFSET = KP_V_OFFSET + 64 * 128 * 4 * 4,
        KP_DT_OFFSET = KP_A_OFFSET + 64 * 4,
        KP_NORM_OFFSET = KP_DT_OFFSET + 64 * 128 * 4,
#ifdef __APPLE__
        KI_WEIGHT_OFFSET = KP_NORM_OFFSET + 128 * 4,
        MODEL_BYTES = KI_WEIGHT_OFFSET + 4096 * 24896 * 2,
#else
        MODEL_BYTES = KP_NORM_OFFSET + 128 * 4,
#endif
    };

    uint8_t *model = mmap(NULL, MODEL_BYTES, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (model == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    float *q_conv = (float *)(model + Q_CONV_OFFSET);
    float *k_conv = (float *)(model + K_CONV_OFFSET);
    float *v_conv = (float *)(model + V_CONV_OFFSET);
    float *a_log = (float *)(model + A_LOG_OFFSET);
    float *dt_bias = (float *)(model + DT_BIAS_OFFSET);
    float *norm = (float *)(model + NORM_OFFSET);
    for (uint32_t channel = 0; channel < PROJECTION; channel++) {
        q_conv[channel * 4u + 3u] = 1.0f;
        k_conv[channel * 4u + 3u] = 1.0f;
        v_conv[channel * 4u + 3u] = 1.0f;
        dt_bias[channel] = 0.0f;
    }
    for (uint32_t head = 0; head < HEADS; head++) a_log[head] = 0.0f;
    for (uint32_t d = 0; d < D; d++) norm[d] = 1.0f;
    float *pool_norm = (float *)(model + POOL_NORM_OFFSET);
    float *pool_bias = (float *)(model + POOL_BIAS_OFFSET);
    uint16_t *pool_ape = (uint16_t *)(model + POOL_APE_OFFSET);
    for (uint32_t d = 0; d < D; d++) {
        pool_norm[d] = 0.75f + 0.002f * (float)d;
        pool_bias[d] = -0.1f + 0.001f * (float)d;
        for (uint32_t r = 0; r < 4u; r++) {
            pool_ape[r * D + d] = f32_to_bf16(
                0.03f * (float)r - 0.0005f * (float)d);
        }
    }

    require_ok(ds4_gpu_init(), "GPU initialization");
    require_ok(ds4_gpu_set_model_map(model, MODEL_BYTES), "model map registration");

    uint16_t *bf16_weights = (uint16_t *)(model + BF16_OFFSET);
    for (uint32_t o = 0; o < BF16_OUT; o++) {
        for (uint32_t i = 0; i < BF16_IN; i++) {
            const float value = 0.002f * (float)((int)(o % 11u) - 5) +
                                0.001f * (float)((int)(i % 13u) - 6);
            bf16_weights[o * BF16_IN + i] = f32_to_bf16(value);
        }
    }
    float bf16_input[BF16_ROWS * BF16_IN];
    float bf16_expected[BF16_ROWS * BF16_OUT];
    for (uint32_t row = 0; row < BF16_ROWS; row++) {
        for (uint32_t i = 0; i < BF16_IN; i++) {
            bf16_input[row * BF16_IN + i] =
                0.02f * (float)((int)(i % 17u) - 8) + 0.005f * (float)row;
        }
        for (uint32_t o = 0; o < BF16_OUT; o++) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < BF16_IN; i++) {
                sum += bf16_to_f32(bf16_weights[o * BF16_IN + i]) *
                       bf16_input[row * BF16_IN + i];
            }
            bf16_expected[row * BF16_OUT + o] = sum;
        }
    }
    ds4_gpu_tensor *bf16_x = ds4_gpu_tensor_alloc(sizeof(bf16_input));
    ds4_gpu_tensor *bf16_out = ds4_gpu_tensor_alloc(sizeof(bf16_expected));
    require_ok(bf16_x && bf16_out, "BF16 tensor allocation");
    require_ok(ds4_gpu_tensor_write(bf16_x, 0, bf16_input, sizeof(bf16_input)),
               "BF16 input write");
    require_ok(ds4_gpu_glm53_matmul_bf16(
        bf16_out, model, MODEL_BYTES, BF16_OFFSET,
        BF16_IN, BF16_OUT, bf16_x, 1), "BF16 decode matmul");
    float bf16_actual[BF16_ROWS * BF16_OUT];
    require_ok(ds4_gpu_tensor_read(bf16_out, 0, bf16_actual,
                                   BF16_OUT * sizeof(float)),
               "BF16 decode output read");
    for (uint32_t i = 0; i < BF16_OUT; i++)
        require_close("BF16 decode matmul", bf16_actual[i], bf16_expected[i], 2e-6f);
    require_ok(ds4_gpu_glm53_matmul_bf16(
        bf16_out, model, MODEL_BYTES, BF16_OFFSET,
        BF16_IN, BF16_OUT, bf16_x, BF16_ROWS), "BF16 prefill matmul");
    require_ok(ds4_gpu_tensor_read(bf16_out, 0, bf16_actual, sizeof(bf16_actual)),
               "BF16 prefill output read");
    for (uint32_t i = 0; i < BF16_ROWS * BF16_OUT; i++)
        require_close("BF16 prefill matmul", bf16_actual[i], bf16_expected[i], 2e-4f);

    /* BF16_IN above is 64; these cover the widths the model actually runs. */
    check_bf16_matmul(model, MODEL_BYTES, WIDE512_OFFSET, WIDE512_IN,
                      WIDE512_OUT, WIDE_ROWS, "BF16 matmul in_dim=512");
    check_bf16_matmul(model, MODEL_BYTES, WIDE1024_OFFSET, WIDE1024_IN,
                      WIDE1024_OUT, WIDE_ROWS, "BF16 matmul in_dim=1024");
    check_bf16_matmul(model, MODEL_BYTES, WIDE4096_OFFSET, WIDE4096_IN,
                      WIDE4096_OUT, WIDE_ROWS, "BF16 matmul in_dim=4096");

#ifdef __APPLE__
    /*
     * F16 and BF16 have different reference matvec reductions. Each fused
     * producer must match its own four-dispatch chain, including every split
     * weight. Comparing the fused types to each other misses a shared error.
     */
    for (unsigned fixture = 0; fixture < 3; fixture++) {
        static const float exact_both[8] = {
            0.5f, -0.5f, 1.0f, -1.0f, 1.5f, -1.5f, 0.25f, -0.75f
        };
        uint16_t *hc_f16 = (uint16_t *)(model + HC_F16W_OFFSET);
        uint16_t *hc_bf16 = (uint16_t *)(model + HC_BF16W_OFFSET);
        for (uint32_t i = 0; i < (uint32_t)(HC_N * HC_MIX); i++) {
            const uint32_t hash = (i * 1664525u + 1013904223u) ^ (i >> 5u);
            const float w = exact_both[(hash >> 16u) % 8u] *
                (fixture == 0 ? 0.03125f : (i % 3u == 0 ? 0.25f : 0.00390625f));
            union { float f; uint32_t u; } b = { .f = w };
            hc_f16[i] = f32_to_f16(w);
            hc_bf16[i] = (uint16_t)(b.u >> 16);
            /* the encodings must round-trip to the same float, or the
             * comparison below would be measuring the fixture, not the kernel */
            require_close("HC fixture encoding", f16_to_f32(hc_f16[i]),
                          bf16_to_f32(hc_bf16[i]), 0.0f);
        }
        float *hc_scale = (float *)(model + HC_SCALE_OFFSET);
        for (int i = 0; i < 3; i++) hc_scale[i] = 0.5f + 0.25f * (float)i;
        float *hc_base = (float *)(model + HC_BASE_OFFSET);
        for (int i = 0; i < HC_MIX; i++) hc_base[i] = 0.125f * (float)((i % 5) - 2);
        float *hc_norm = (float *)(model + HC_NORM_OFFSET);
        for (int i = 0; i < HC_EMBD; i++) hc_norm[i] = 1.0f + 0.001f * (float)(i % 7);

        float *hc_x = malloc((size_t)HC_N * sizeof(float));
        require_ok(hc_x != NULL, "HC residual allocation");
        for (int i = 0; i < HC_N; i++) {
            const float x = 0.01f * (float)((i % 23) - 11) + 0.002f * (float)(i % 5);
            hc_x[i] = fixture == 2 ? 0.0f :
                x * (fixture == 1 && i % 7 == 0 ? 32.0f : 1.0f);
        }

        ds4_gpu_tensor *hc_res = ds4_gpu_tensor_alloc((size_t)HC_N * sizeof(float));
        require_ok(hc_res != NULL, "HC residual tensor");
        require_ok(ds4_gpu_tensor_write(hc_res, 0, hc_x,
                                        (size_t)HC_N * sizeof(float)),
                   "HC residual write");

        for (int pass = 0; pass < 2; pass++) {
            const uint32_t iters = fixture == 0 ? 1u : 20u;
            ds4_gpu_tensor *flat = ds4_gpu_tensor_alloc(HC_N * sizeof(float));
            ds4_gpu_tensor *ref_mix = ds4_gpu_tensor_alloc(HC_MIX * sizeof(float));
            ds4_gpu_tensor *ref_spl = ds4_gpu_tensor_alloc(HC_MIX * sizeof(float));
            ds4_gpu_tensor *ref_out = ds4_gpu_tensor_alloc(HC_EMBD * sizeof(float));
            ds4_gpu_tensor *ref_nrm = ds4_gpu_tensor_alloc(HC_EMBD * sizeof(float));
            ds4_gpu_tensor *mix = ds4_gpu_tensor_alloc(HC_MIX * sizeof(float));
            ds4_gpu_tensor *spl = ds4_gpu_tensor_alloc(HC_MIX * sizeof(float));
            ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(HC_EMBD * sizeof(float));
            ds4_gpu_tensor *nrm = ds4_gpu_tensor_alloc(HC_EMBD * sizeof(float));
            require_ok(flat && ref_mix && ref_spl && ref_out && ref_nrm &&
                       mix && spl && out && nrm, "HC output tensors");
            require_ok(ds4_gpu_begin_commands(), "HC reference begin");
            require_ok(ds4_gpu_rms_norm_plain_tensor(flat, hc_res, HC_N, 1.0e-6f), "HC reference RMS");
            require_ok(pass == 0 ?
                ds4_gpu_matmul_f16_tensor(ref_mix, model, MODEL_BYTES, HC_F16W_OFFSET, HC_N, HC_MIX, flat, 1) :
                ds4_gpu_glm53_matmul_bf16(ref_mix, model, MODEL_BYTES, HC_BF16W_OFFSET, HC_N, HC_MIX, flat, 1),
                "HC reference projection");
            require_ok(ds4_gpu_hc_split_weighted_sum_tensor(ref_out, ref_spl, ref_mix, hc_res,
                model, MODEL_BYTES, HC_SCALE_OFFSET, HC_BASE_OFFSET, HC_EMBD, HC_HC, iters, 1.0e-3f),
                "HC reference split/collapse");
            require_ok(ds4_gpu_rms_norm_weight_tensor(ref_nrm, ref_out, model, MODEL_BYTES,
                HC_NORM_OFFSET, HC_EMBD, 1.0e-6f), "HC reference weighted RMS");
            /* Reuse the same completion counter in one command stream. */
            for (unsigned repeat = 0; repeat < 3; repeat++) {
                require_ok(ds4_gpu_tensor_fill_f32(spl, -17.0f, HC_MIX),
                           "HC poison split before counter reuse");
                const int rc = pass == 0
                    ? ds4_gpu_hc_rms_norm_mix_split_norm_f16_tensor(
                          mix, out, nrm, spl, hc_res, model, MODEL_BYTES,
                          HC_F16W_OFFSET, HC_SCALE_OFFSET, HC_BASE_OFFSET,
                          HC_NORM_OFFSET, HC_N, HC_MIX, HC_EMBD, HC_HC,
                          iters, 1.0e-6f, 1.0e-3f, 1.0e-6f)
                    : ds4_gpu_hc_rms_norm_mix_split_norm_bf16_tensor(
                          mix, out, nrm, spl, hc_res, model, MODEL_BYTES,
                          HC_BF16W_OFFSET, HC_SCALE_OFFSET, HC_BASE_OFFSET,
                          HC_NORM_OFFSET, HC_N, HC_MIX, HC_EMBD, HC_HC,
                          iters, 1.0e-6f, 1.0e-3f, 1.0e-6f);
                require_ok(rc > 0, pass == 0 ? "HC producer f16" : "HC producer bf16");
            }
            require_ok(ds4_gpu_end_commands(), "HC producer end");
            const ds4_gpu_tensor *actual[] = {mix, spl, out, nrm};
            const ds4_gpu_tensor *reference[] = {ref_mix, ref_spl, ref_out, ref_nrm};
            const uint32_t counts[] = {HC_MIX, HC_MIX, HC_EMBD, HC_EMBD};
            const char *names[] = {"mix", "split", "collapse", "pre-norm"};
            for (unsigned stage = 0; stage < 4; stage++) {
                float a[HC_EMBD], b[HC_EMBD];
                const size_t bytes = counts[stage] * sizeof(float);
                require_ok(ds4_gpu_tensor_read(actual[stage], 0, a, bytes) &&
                           ds4_gpu_tensor_read(reference[stage], 0, b, bytes), "HC readback");
                if (memcmp(a, b, bytes) != 0) {
                    for (uint32_t i = 0; i < counts[stage]; i++) {
                        if (memcmp(a + i, b + i, sizeof(float)) != 0) {
                            fprintf(stderr, "HC %s fixture %u %s[%u]: got %.9g, reference %.9g\n",
                                pass == 0 ? "F16" : "BF16", fixture, names[stage], i, a[i], b[i]);
                            break;
                        }
                    }
                    exit(1);
                }
            }
            ds4_gpu_tensor_free(mix);
            ds4_gpu_tensor_free(spl);
            ds4_gpu_tensor_free(out);
            ds4_gpu_tensor_free(nrm);
            ds4_gpu_tensor_free(flat);
            ds4_gpu_tensor_free(ref_mix);
            ds4_gpu_tensor_free(ref_spl);
            ds4_gpu_tensor_free(ref_out);
            ds4_gpu_tensor_free(ref_nrm);
        }
        ds4_gpu_tensor_free(hc_res);
        free(hc_x);
    }

    /*
     * BF16 matvec with the HC expansion folded into its epilogue must equal
     * the separate matvec followed by ds4_gpu_hc_expand_tensor, exactly.  The
     * fused kernel reuses the same row accumulation and repeats the expand
     * arithmetic in the same operand order, so anything but bit-identical
     * output is a bug -- most likely a stride or an index.
     */
    {
        uint16_t *fw = (uint16_t *)(model + FUSED_W_OFFSET);
        for (uint32_t o = 0; o < FUSED_OUT; o++) {
            for (uint32_t i = 0; i < FUSED_IN; i++) {
                fw[(size_t)o * FUSED_IN + i] = f32_to_bf16(
                    0.003f * (float)((int)((o * 7u + i) % 17u) - 8));
            }
        }
        float fx[FUSED_IN], fres[FUSED_HC * FUSED_OUT];
        float fpost[FUSED_HC], fcomb[FUSED_HC * FUSED_HC];
        for (int i = 0; i < FUSED_IN; i++)
            fx[i] = 0.01f * (float)((i % 19) - 9);
        for (int i = 0; i < FUSED_HC * FUSED_OUT; i++)
            fres[i] = 0.05f * (float)((i % 13) - 6);
        for (int i = 0; i < FUSED_HC; i++) fpost[i] = 0.25f + 0.125f * (float)i;
        for (int i = 0; i < FUSED_HC * FUSED_HC; i++)
            fcomb[i] = 0.1f * (float)((i % 7) - 3);

        ds4_gpu_tensor *tx   = ds4_gpu_tensor_alloc(sizeof(fx));
        ds4_gpu_tensor *tres = ds4_gpu_tensor_alloc(sizeof(fres));
        ds4_gpu_tensor *tpost = ds4_gpu_tensor_alloc(sizeof(fpost));
        ds4_gpu_tensor *tcomb = ds4_gpu_tensor_alloc(sizeof(fcomb));
        ds4_gpu_tensor *out_ref = ds4_gpu_tensor_alloc(FUSED_OUT * sizeof(float));
        ds4_gpu_tensor *hc_ref  = ds4_gpu_tensor_alloc(sizeof(fres));
        ds4_gpu_tensor *out_fus = ds4_gpu_tensor_alloc(FUSED_OUT * sizeof(float));
        ds4_gpu_tensor *hc_fus  = ds4_gpu_tensor_alloc(sizeof(fres));
        require_ok(tx && tres && tpost && tcomb && out_ref && hc_ref &&
                   out_fus && hc_fus, "fused epilogue tensors");
        require_ok(ds4_gpu_tensor_write(tx, 0, fx, sizeof(fx)) &&
                   ds4_gpu_tensor_write(tres, 0, fres, sizeof(fres)) &&
                   ds4_gpu_tensor_write(tpost, 0, fpost, sizeof(fpost)) &&
                   ds4_gpu_tensor_write(tcomb, 0, fcomb, sizeof(fcomb)),
                   "fused epilogue inputs");

        require_ok(ds4_gpu_glm53_matmul_bf16(
                       out_ref, model, MODEL_BYTES, FUSED_W_OFFSET,
                       FUSED_IN, FUSED_OUT, tx, 1),
                   "reference BF16 matvec");
        require_ok(ds4_gpu_hc_expand_tensor(hc_ref, out_ref, tres, tpost, tcomb,
                                            FUSED_OUT, FUSED_HC),
                   "reference HC expand");

        const int fused = ds4_gpu_glm53_matmul_bf16_hc_expand4(
            out_fus, hc_fus, model, MODEL_BYTES, FUSED_W_OFFSET,
            FUSED_IN, FUSED_OUT, tx, tres, tpost, tcomb, FUSED_HC);
        if (fused == 0) {
            fprintf(stderr,
                    "BF16 matvec + HC expand: not available on this device, skipped\n");
        } else {
            float a[FUSED_OUT], b[FUSED_OUT];
            float ha[FUSED_HC * FUSED_OUT], hb[FUSED_HC * FUSED_OUT];
            require_ok(ds4_gpu_tensor_read(out_ref, 0, a, sizeof(a)) &&
                       ds4_gpu_tensor_read(out_fus, 0, b, sizeof(b)) &&
                       ds4_gpu_tensor_read(hc_ref, 0, ha, sizeof(ha)) &&
                       ds4_gpu_tensor_read(hc_fus, 0, hb, sizeof(hb)),
                       "fused epilogue readback");
            for (int i = 0; i < FUSED_OUT; i++)
                require_close("fused epilogue block_out", b[i], a[i], 0.0f);
            for (int i = 0; i < FUSED_HC * FUSED_OUT; i++)
                require_close("fused epilogue hc stream", hb[i], ha[i], 0.0f);
        }
        ds4_gpu_tensor_free(tx);
        ds4_gpu_tensor_free(tres);
        ds4_gpu_tensor_free(tpost);
        ds4_gpu_tensor_free(tcomb);
        ds4_gpu_tensor_free(out_ref);
        ds4_gpu_tensor_free(hc_ref);
        ds4_gpu_tensor_free(out_fus);
        ds4_gpu_tensor_free(hc_fus);
    }
#endif /* __APPLE__: fused HC producers and epilogues are Metal-only. */

#ifdef DS4_ROCM_BUILD
    test_block_q4_K *q4_weights = (test_block_q4_K *)(model + Q4_OFFSET);
    for (uint32_t o = 0; o < Q4_OUT; o++) {
        test_block_q4_K *block = q4_weights + o;
        const uint8_t q = (uint8_t)(1u + o % 15u);
        block->d = 0x3c00u;
        block->dmin = 0u;
        for (uint32_t group = 0; group < 4u; group++) {
            block->scales[group] = 1u;
            block->scales[group + 4u] = 0u;
        }
        for (uint32_t group = 4u; group < 8u; group++)
            block->scales[group + 4u] = 1u;
        for (uint32_t i = 0; i < sizeof(block->qs); i++)
            block->qs[i] = (uint8_t)(q | (q << 4u));
    }
    float q4_input[Q4_ROWS * Q4_IN];
    const float q4_row_values[Q4_ROWS] = {1.0f, -0.5f, 0.25f};
    for (uint32_t row = 0; row < Q4_ROWS; row++)
        for (uint32_t i = 0; i < Q4_IN; i++)
            q4_input[row * Q4_IN + i] = q4_row_values[row];
    ds4_gpu_tensor *q4_x = ds4_gpu_tensor_alloc(sizeof(q4_input));
    ds4_gpu_tensor *q4_out =
        ds4_gpu_tensor_alloc((uint64_t)Q4_ROWS * Q4_OUT * sizeof(float));
    require_ok(q4_x && q4_out, "Q4_K tensor allocation");
    require_ok(ds4_gpu_tensor_write(q4_x, 0, q4_input, sizeof(q4_input)),
               "Q4_K input write");
    require_ok(ds4_gpu_matmul_q4_K_tensor(
        q4_out, model, MODEL_BYTES, Q4_OFFSET,
        Q4_IN, Q4_OUT, q4_x, Q4_ROWS), "Q4_K dense matmul");
    float q4_actual[Q4_ROWS * Q4_OUT];
    require_ok(ds4_gpu_tensor_read(q4_out, 0, q4_actual, sizeof(q4_actual)),
               "Q4_K output read");
    for (uint32_t row = 0; row < Q4_ROWS; row++) {
        for (uint32_t o = 0; o < Q4_OUT; o++) {
            const float expected =
                256.0f * (float)(1u + o % 15u) * q4_row_values[row];
            require_close("Q4_K dense matmul",
                          q4_actual[row * Q4_OUT + o], expected, 1e-3f);
        }
    }
    ds4_gpu_tensor_free(q4_out);
    ds4_gpu_tensor_free(q4_x);
#endif

    enum { COMPACT_LORA = 512, COMPACT_TOKENS = 2, COMPACT_CAP = 4 };
    float compact_norm[COMPACT_TOKENS * COMPACT_LORA];
    float compact_raw[COMPACT_TOKENS * COMPACT_LORA];
    for (uint32_t i = 0; i < COMPACT_TOKENS * COMPACT_LORA; i++) {
        compact_norm[i] = 0.001f * (float)((int)(i % 101u) - 50);
        compact_raw[i] = compact_norm[i] + 1.0f;
    }
    ds4_gpu_tensor *compact_norm_gpu =
        ds4_gpu_tensor_alloc(sizeof(compact_norm));
    ds4_gpu_tensor *compact_raw_gpu =
        ds4_gpu_tensor_alloc(sizeof(compact_raw));
    ds4_gpu_tensor *compact_cache_gpu = ds4_gpu_tensor_alloc(
        (uint64_t)COMPACT_CAP * COMPACT_LORA * sizeof(float));
    ds4_gpu_tensor *zero_rope_cache_gpu = NULL;
#ifndef DS4_ROCM_BUILD
    zero_rope_cache_gpu = ds4_gpu_tensor_alloc(1u);
#endif
    require_ok(compact_norm_gpu && compact_raw_gpu && compact_cache_gpu
#ifndef DS4_ROCM_BUILD
               && zero_rope_cache_gpu
#endif
               ,
               "zero-RoPE compact tensor allocation");
    require_ok(ds4_gpu_tensor_write(compact_norm_gpu, 0, compact_norm,
                                    sizeof(compact_norm)),
               "zero-RoPE compact norm write");
    require_ok(ds4_gpu_tensor_write(compact_raw_gpu, 0, compact_raw,
                                    sizeof(compact_raw)),
               "zero-RoPE compact raw write");
    require_ok(ds4_gpu_glm_store_compact_kv_tensor(
        compact_cache_gpu, zero_rope_cache_gpu,
        compact_norm_gpu, compact_raw_gpu,
        1, COMPACT_TOKENS, COMPACT_CAP,
        COMPACT_LORA, COMPACT_LORA, 0, false),
        "GLM-5.3 zero-RoPE compact KV store");
    float compact_actual[COMPACT_CAP * COMPACT_LORA];
    require_ok(ds4_gpu_tensor_read(
        compact_cache_gpu,
        (uint64_t)COMPACT_LORA * sizeof(float),
        compact_actual,
        sizeof(compact_norm)),
        "zero-RoPE compact cache read");
    for (uint32_t i = 0; i < COMPACT_TOKENS * COMPACT_LORA; i++) {
        require_close("zero-RoPE compact KV", compact_actual[i],
                      compact_norm[i], 0.0f);
    }
    ds4_gpu_tensor_free(zero_rope_cache_gpu);
    ds4_gpu_tensor_free(compact_cache_gpu);
    ds4_gpu_tensor_free(compact_raw_gpu);
    ds4_gpu_tensor_free(compact_norm_gpu);

#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU)
    enum {
        DENSE_ATTN_PREFIX = 128,
        DENSE_ATTN_TOKENS = 256,
        DENSE_ATTN_CAP = DENSE_ATTN_PREFIX + DENSE_ATTN_TOKENS,
        DENSE_ATTN_HEADS = 8,
        DENSE_ATTN_LORA = 512,
    };
    const uint64_t dense_attn_cache_count =
        (uint64_t)DENSE_ATTN_CAP * DENSE_ATTN_LORA;
    const uint64_t dense_attn_q_count =
        (uint64_t)DENSE_ATTN_TOKENS * DENSE_ATTN_HEADS * DENSE_ATTN_LORA;
    float *dense_attn_cache =
        malloc((size_t)dense_attn_cache_count * sizeof(*dense_attn_cache));
    float *dense_attn_q =
        malloc((size_t)dense_attn_q_count * sizeof(*dense_attn_q));
    float *dense_attn_reference =
        malloc((size_t)dense_attn_q_count * sizeof(*dense_attn_reference));
    float *dense_attn_actual =
        malloc((size_t)dense_attn_q_count * sizeof(*dense_attn_actual));
    require_ok(dense_attn_cache && dense_attn_q && dense_attn_reference &&
               dense_attn_actual, "dense compact attention host allocation");
    for (uint64_t i = 0; i < dense_attn_cache_count; i++) {
        dense_attn_cache[i] =
            0.004f * (float)((int)(i % 101u) - 50);
    }
    for (uint64_t i = 0; i < dense_attn_q_count; i++) {
        dense_attn_q[i] =
            0.003f * (float)((int)(i % 127u) - 63);
    }

    ds4_gpu_tensor *dense_attn_cache_input_gpu = ds4_gpu_tensor_alloc(
        dense_attn_cache_count * sizeof(float));
    ds4_gpu_tensor *dense_attn_cache_gpu = ds4_gpu_tensor_alloc(
        dense_attn_cache_count * sizeof(uint16_t));
    ds4_gpu_tensor *dense_attn_rope_gpu = ds4_gpu_tensor_alloc(1u);
    ds4_gpu_tensor *dense_attn_q_gpu = ds4_gpu_tensor_alloc(
        dense_attn_q_count * sizeof(float));
    ds4_gpu_tensor *dense_attn_dummy_q_gpu = ds4_gpu_tensor_alloc(
        dense_attn_q_count * sizeof(float));
    ds4_gpu_tensor *dense_attn_reference_gpu = ds4_gpu_tensor_alloc(
        dense_attn_q_count * sizeof(float));
    ds4_gpu_tensor *dense_attn_actual_gpu = ds4_gpu_tensor_alloc(
        dense_attn_q_count * sizeof(float));
    require_ok(dense_attn_cache_input_gpu && dense_attn_cache_gpu &&
               dense_attn_rope_gpu && dense_attn_q_gpu &&
               dense_attn_dummy_q_gpu && dense_attn_reference_gpu &&
               dense_attn_actual_gpu,
               "dense compact attention GPU allocation");
    require_ok(ds4_gpu_tensor_write(
        dense_attn_cache_input_gpu, 0, dense_attn_cache,
        dense_attn_cache_count * sizeof(float)),
        "dense compact attention cache write");
    require_ok(ds4_gpu_tensor_write(
        dense_attn_q_gpu, 0, dense_attn_q,
        dense_attn_q_count * sizeof(float)),
        "dense compact attention Q write");
    require_ok(ds4_gpu_tensor_fill_f32(
        dense_attn_dummy_q_gpu, 0.0f, dense_attn_q_count),
        "dense compact attention dummy Q clear");
    require_ok(ds4_gpu_glm_store_compact_kv_tensor(
        dense_attn_cache_gpu, dense_attn_rope_gpu,
        dense_attn_cache_input_gpu, dense_attn_cache_input_gpu,
        0, DENSE_ATTN_CAP, DENSE_ATTN_CAP,
        DENSE_ATTN_LORA, DENSE_ATTN_LORA, 0, true),
        "dense compact attention F16 cache store");
    require_ok(ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
        dense_attn_reference_gpu, dense_attn_dummy_q_gpu,
        dense_attn_q_gpu, dense_attn_cache_gpu, dense_attn_rope_gpu,
        DENSE_ATTN_TOKENS, DENSE_ATTN_PREFIX, DENSE_ATTN_CAP,
        DENSE_ATTN_CAP, true, DENSE_ATTN_HEADS, DENSE_ATTN_LORA,
        DENSE_ATTN_LORA, 0, 0, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f),
        "dense compact attention scalar reference");
    require_ok(ds4_gpu_glm_attention_dense_compact_lora_causal_tensor(
        dense_attn_actual_gpu, dense_attn_q_gpu, dense_attn_cache_gpu,
        DENSE_ATTN_PREFIX, DENSE_ATTN_TOKENS, DENSE_ATTN_CAP,
        DENSE_ATTN_CAP, true, DENSE_ATTN_HEADS, DENSE_ATTN_LORA,
        DENSE_ATTN_LORA),
        "dense compact attention GEMM path");
    require_ok(ds4_gpu_tensor_read(
        dense_attn_reference_gpu, 0, dense_attn_reference,
        dense_attn_q_count * sizeof(float)),
        "dense compact attention reference read");
    require_ok(ds4_gpu_tensor_read(
        dense_attn_actual_gpu, 0, dense_attn_actual,
        dense_attn_q_count * sizeof(float)),
        "dense compact attention output read");
    double dense_attn_sq_error = 0.0;
    float dense_attn_max_error = 0.0f;
    for (uint64_t i = 0; i < dense_attn_q_count; i++) {
        const float error = fabsf(
            dense_attn_actual[i] - dense_attn_reference[i]);
        dense_attn_max_error = fmaxf(dense_attn_max_error, error);
        dense_attn_sq_error += (double)error * (double)error;
    }
    const double dense_attn_rms_error =
        sqrt(dense_attn_sq_error / (double)dense_attn_q_count);
    if (dense_attn_max_error > 5e-4f || dense_attn_rms_error > 1e-4) {
        fprintf(stderr,
                "dense compact attention diverged from scalar reference "
                "(max %.9g, RMS %.9g)\n",
                dense_attn_max_error, dense_attn_rms_error);
        return 1;
    }
    ds4_gpu_tensor_free(dense_attn_actual_gpu);
    ds4_gpu_tensor_free(dense_attn_reference_gpu);
    ds4_gpu_tensor_free(dense_attn_dummy_q_gpu);
    ds4_gpu_tensor_free(dense_attn_q_gpu);
    ds4_gpu_tensor_free(dense_attn_rope_gpu);
    ds4_gpu_tensor_free(dense_attn_cache_gpu);
    ds4_gpu_tensor_free(dense_attn_cache_input_gpu);
    free(dense_attn_actual);
    free(dense_attn_reference);
    free(dense_attn_q);
    free(dense_attn_cache);

#endif

#ifdef __APPLE__
    enum {
        F32_ATTN_TOKENS = 2,
        F32_ATTN_HEADS = 8,
        F32_ATTN_LORA = 512,
        F32_ATTN_NOPE = 64,
    };
    const uint64_t f32_attn_lora_count =
        (uint64_t)F32_ATTN_TOKENS * F32_ATTN_HEADS * F32_ATTN_LORA;
    const uint64_t f32_attn_q_count =
        (uint64_t)F32_ATTN_TOKENS * F32_ATTN_HEADS * F32_ATTN_NOPE;
    float *f32_attn_low = calloc((size_t)f32_attn_lora_count, sizeof(float));
    float *f32_attn_q = calloc((size_t)f32_attn_q_count, sizeof(float));
    float *f32_attn_cache = malloc(
        (size_t)F32_ATTN_TOKENS * F32_ATTN_LORA * sizeof(float));
    float *f32_attn_actual = malloc((size_t)f32_attn_lora_count * sizeof(float));
    require_ok(f32_attn_low && f32_attn_q && f32_attn_cache && f32_attn_actual,
               "FP32 causal attention host allocation");
    for (uint32_t row = 0; row < F32_ATTN_TOKENS; row++) {
        const float value = 1.0f + 2.0f * (float)row;
        for (uint32_t i = 0; i < F32_ATTN_LORA; i++) {
            f32_attn_cache[(uint64_t)row * F32_ATTN_LORA + i] = value;
        }
    }

    ds4_gpu_tensor *f32_attn_out_gpu =
        ds4_gpu_tensor_alloc(f32_attn_lora_count * sizeof(float));
    ds4_gpu_tensor *f32_attn_low_gpu =
        ds4_gpu_tensor_alloc(f32_attn_lora_count * sizeof(float));
    ds4_gpu_tensor *f32_attn_q_gpu =
        ds4_gpu_tensor_alloc(f32_attn_q_count * sizeof(float));
    ds4_gpu_tensor *f32_attn_cache_gpu = ds4_gpu_tensor_alloc(
        (uint64_t)F32_ATTN_TOKENS * F32_ATTN_LORA * sizeof(float));
    ds4_gpu_tensor *f32_attn_rope_gpu = ds4_gpu_tensor_alloc(sizeof(float));
    require_ok(f32_attn_out_gpu && f32_attn_low_gpu && f32_attn_q_gpu &&
               f32_attn_cache_gpu && f32_attn_rope_gpu,
               "FP32 causal attention GPU allocation");
    require_ok(ds4_gpu_tensor_write(f32_attn_low_gpu, 0, f32_attn_low,
                                    f32_attn_lora_count * sizeof(float)),
               "FP32 causal attention low-rank Q write");
    require_ok(ds4_gpu_tensor_write(f32_attn_q_gpu, 0, f32_attn_q,
                                    f32_attn_q_count * sizeof(float)),
               "FP32 causal attention Q write");
    require_ok(ds4_gpu_tensor_write(
                   f32_attn_cache_gpu, 0, f32_attn_cache,
                   (uint64_t)F32_ATTN_TOKENS * F32_ATTN_LORA * sizeof(float)),
               "FP32 causal attention cache write");

    require_ok(ds4_gpu_glm_attention_dense_compact_lora_causal_tensor(
                   f32_attn_out_gpu,
                   f32_attn_low_gpu,
                   f32_attn_cache_gpu,
                   0,
                   F32_ATTN_TOKENS,
                   F32_ATTN_TOKENS,
                   F32_ATTN_TOKENS,
                   false,
                   F32_ATTN_HEADS,
                   F32_ATTN_LORA,
                   F32_ATTN_NOPE),
               "FP32 dense compact causal attention");
    require_ok(ds4_gpu_tensor_read(f32_attn_out_gpu, 0, f32_attn_actual,
                                   f32_attn_lora_count * sizeof(float)),
               "FP32 dense compact causal attention read");
    for (uint32_t token = 0; token < F32_ATTN_TOKENS; token++) {
        const float expected = token == 0 ? 1.0f : 2.0f;
        for (uint64_t i = (uint64_t)token * F32_ATTN_HEADS * F32_ATTN_LORA;
             i < (uint64_t)(token + 1u) * F32_ATTN_HEADS * F32_ATTN_LORA;
             i++) {
            require_close("FP32 dense compact causal attention",
                          f32_attn_actual[i], expected, 1e-4f);
        }
    }

    require_ok(ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
                   f32_attn_out_gpu,
                   f32_attn_q_gpu,
                   f32_attn_low_gpu,
                   f32_attn_cache_gpu,
                   f32_attn_rope_gpu,
                   F32_ATTN_TOKENS,
                   0,
                   F32_ATTN_TOKENS,
                   F32_ATTN_TOKENS,
                   false,
                   F32_ATTN_HEADS,
                   F32_ATTN_LORA,
                   F32_ATTN_NOPE,
                   0,
                   0,
                   0.0f,
                   0.0f,
                   0.0f,
                   1.0f,
                   0.0f,
                   0.0f),
               "FP32 indexed causal attention");
    require_ok(ds4_gpu_tensor_read(f32_attn_out_gpu, 0, f32_attn_actual,
                                   f32_attn_lora_count * sizeof(float)),
               "FP32 indexed causal attention read");
    for (uint32_t token = 0; token < F32_ATTN_TOKENS; token++) {
        const float expected = token == 0 ? 1.0f : 2.0f;
        for (uint64_t i = (uint64_t)token * F32_ATTN_HEADS * F32_ATTN_LORA;
             i < (uint64_t)(token + 1u) * F32_ATTN_HEADS * F32_ATTN_LORA;
             i++) {
            require_close("FP32 indexed causal attention",
                          f32_attn_actual[i], expected, 1e-4f);
        }
    }

    ds4_gpu_tensor_free(f32_attn_rope_gpu);
    ds4_gpu_tensor_free(f32_attn_cache_gpu);
    ds4_gpu_tensor_free(f32_attn_q_gpu);
    ds4_gpu_tensor_free(f32_attn_low_gpu);
    ds4_gpu_tensor_free(f32_attn_out_gpu);
    free(f32_attn_actual);
    free(f32_attn_cache);
    free(f32_attn_q);
    free(f32_attn_low);

    check_split_dsa_attention(model, MODEL_BYTES, SPLIT_V_OFFSET);
#endif

#ifdef DS4_ROCM_BUILD
    enum {
        ATTN_LORA = 32,
        ATTN_NOPE = 1,
        ATTN_HEADS = 1,
        ATTN_VALUE = 1,
        ATTN_SELECTED = 2,
        ATTN_CAP = 3,
    };
    test_block_q8_0 *value_weight =
        (test_block_q8_0 *)(model + Q8_OFFSET);
    value_weight->d = 0x3c00u;
    for (uint32_t i = 0; i < ATTN_LORA; i++) value_weight->qs[i] = 1;
    float attn_q[ATTN_HEADS * ATTN_NOPE] = {0.0f};
    float attn_low[ATTN_HEADS * ATTN_LORA] = {0.0f};
    float attn_cache[ATTN_CAP * ATTN_LORA];
    for (uint32_t row = 0; row < ATTN_CAP; row++) {
        for (uint32_t i = 0; i < ATTN_LORA; i++) {
            attn_cache[row * ATTN_LORA + i] = 1.0f + 2.0f * (float)row;
        }
    }
    int32_t attn_selected[ATTN_SELECTED] = {0, 1};
    ds4_gpu_tensor *attn_heads_gpu =
        ds4_gpu_tensor_alloc(ATTN_HEADS * ATTN_VALUE * sizeof(float));
    ds4_gpu_tensor *attn_q_gpu = ds4_gpu_tensor_alloc(sizeof(attn_q));
    ds4_gpu_tensor *attn_low_gpu = ds4_gpu_tensor_alloc(sizeof(attn_low));
    ds4_gpu_tensor *attn_cache_gpu = ds4_gpu_tensor_alloc(sizeof(attn_cache));
    ds4_gpu_tensor *attn_selected_gpu =
        ds4_gpu_tensor_alloc(sizeof(attn_selected));
    require_ok(attn_heads_gpu && attn_q_gpu && attn_low_gpu &&
               attn_cache_gpu && attn_selected_gpu,
               "zero-RoPE indexed attention allocation");
    require_ok(ds4_gpu_tensor_write(attn_q_gpu, 0, attn_q, sizeof(attn_q)),
               "zero-RoPE attention Q write");
    require_ok(ds4_gpu_tensor_write(attn_low_gpu, 0, attn_low,
                                    sizeof(attn_low)),
               "zero-RoPE attention low-rank Q write");
    require_ok(ds4_gpu_tensor_write(attn_cache_gpu, 0, attn_cache,
                                    sizeof(attn_cache)),
               "zero-RoPE attention cache write");
    require_ok(ds4_gpu_tensor_write(attn_selected_gpu, 0, attn_selected,
                                    sizeof(attn_selected)),
               "zero-RoPE attention selection write");
    require_ok(ds4_gpu_glm_attention_indexed_decode_tensor(
        attn_heads_gpu, attn_q_gpu, attn_low_gpu, attn_cache_gpu, NULL,
        model, MODEL_BYTES, Q8_OFFSET, attn_selected_gpu,
        ATTN_SELECTED, ATTN_CAP, false, ATTN_HEADS, ATTN_LORA,
        ATTN_NOPE, 0, ATTN_VALUE, 0, 0.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 0.0f), "GLM-5.3 zero-RoPE indexed decode attention");
    float attn_actual = 0.0f;
    require_ok(ds4_gpu_tensor_read(attn_heads_gpu, 0, &attn_actual,
                                    sizeof(attn_actual)),
               "zero-RoPE attention output read");
    require_close("zero-RoPE indexed attention", attn_actual, 64.0f, 1e-4f);
    ds4_gpu_tensor_free(attn_selected_gpu);
    ds4_gpu_tensor_free(attn_cache_gpu);
    ds4_gpu_tensor_free(attn_low_gpu);
    ds4_gpu_tensor_free(attn_q_gpu);
    ds4_gpu_tensor_free(attn_heads_gpu);

    enum {
        BATCH_ATTN_TOKENS = 128,
        BATCH_ATTN_HEADS = 64,
        BATCH_ATTN_LORA = 32,
        BATCH_ATTN_NOPE = 1,
        BATCH_ATTN_REPEATS = 16,
    };
    const uint64_t batch_q_count =
        (uint64_t)BATCH_ATTN_TOKENS * BATCH_ATTN_HEADS * BATCH_ATTN_NOPE;
    const uint64_t batch_lora_count =
        (uint64_t)BATCH_ATTN_TOKENS * BATCH_ATTN_HEADS * BATCH_ATTN_LORA;
    const uint64_t batch_cache_count =
        (uint64_t)BATCH_ATTN_TOKENS * BATCH_ATTN_LORA;
    float *batch_q = calloc((size_t)batch_q_count, sizeof(*batch_q));
    float *batch_low = malloc((size_t)batch_lora_count * sizeof(*batch_low));
    float *batch_cache = malloc((size_t)batch_cache_count * sizeof(*batch_cache));
    float *batch_expected = malloc((size_t)batch_lora_count * sizeof(*batch_expected));
    float *batch_actual = malloc((size_t)batch_lora_count * sizeof(*batch_actual));
    require_ok(batch_q && batch_low && batch_cache &&
               batch_expected && batch_actual,
               "indexed attention determinism host allocation");
    for (uint64_t i = 0; i < batch_lora_count; i++) {
        batch_low[i] = 0.001f * (float)((int)(i % 127u) - 63);
    }
    for (uint64_t i = 0; i < batch_cache_count; i++) {
        batch_cache[i] = 0.002f * (float)((int)(i % 113u) - 56);
    }
    ds4_gpu_tensor *batch_out_gpu =
        ds4_gpu_tensor_alloc(batch_lora_count * sizeof(float));
    ds4_gpu_tensor *batch_q_gpu =
        ds4_gpu_tensor_alloc(batch_q_count * sizeof(float));
    ds4_gpu_tensor *batch_low_gpu =
        ds4_gpu_tensor_alloc(batch_lora_count * sizeof(float));
    ds4_gpu_tensor *batch_cache_gpu =
        ds4_gpu_tensor_alloc(batch_cache_count * sizeof(float));
    require_ok(batch_out_gpu && batch_q_gpu && batch_low_gpu && batch_cache_gpu,
               "indexed attention determinism GPU allocation");
    require_ok(ds4_gpu_tensor_write(batch_q_gpu, 0, batch_q,
                                    batch_q_count * sizeof(float)),
               "indexed attention determinism Q write");
    require_ok(ds4_gpu_tensor_write(batch_low_gpu, 0, batch_low,
                                    batch_lora_count * sizeof(float)),
               "indexed attention determinism low-rank Q write");
    require_ok(ds4_gpu_tensor_write(batch_cache_gpu, 0, batch_cache,
                                    batch_cache_count * sizeof(float)),
               "indexed attention determinism cache write");
    for (uint32_t repeat = 0; repeat < BATCH_ATTN_REPEATS; repeat++) {
        require_ok(ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
            batch_out_gpu, batch_q_gpu, batch_low_gpu, batch_cache_gpu, NULL,
            BATCH_ATTN_TOKENS, 0, BATCH_ATTN_TOKENS, BATCH_ATTN_TOKENS,
            false, BATCH_ATTN_HEADS, BATCH_ATTN_LORA, BATCH_ATTN_NOPE, 0,
            0, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f),
            "indexed attention determinism launch");
        require_ok(ds4_gpu_tensor_read(batch_out_gpu, 0, batch_actual,
                                       batch_lora_count * sizeof(float)),
                   "indexed attention determinism output read");
        if (repeat == 0) {
            memcpy(batch_expected, batch_actual,
                   (size_t)batch_lora_count * sizeof(float));
        } else if (memcmp(batch_expected, batch_actual,
                          (size_t)batch_lora_count * sizeof(float)) != 0) {
            fprintf(stderr,
                    "indexed attention changed on repeat %u\n", repeat);
            return 1;
        }
    }
    ds4_gpu_tensor_free(batch_cache_gpu);
    ds4_gpu_tensor_free(batch_low_gpu);
    ds4_gpu_tensor_free(batch_q_gpu);
    ds4_gpu_tensor_free(batch_out_gpu);
    free(batch_actual);
    free(batch_expected);
    free(batch_cache);
    free(batch_low);
    free(batch_q);
#endif

    enum { POOL = 4, POOL_TOKENS = 11, POOL_CAP = 16, POOL_COUNT = 4 };
    float pool_raw[POOL_TOKENS * D];
    float pool_gate_values[POOL_TOKENS * D];
    for (uint32_t t = 0; t < POOL_TOKENS; t++) {
        for (uint32_t d = 0; d < D; d++) {
            pool_raw[t * D + d] =
                0.01f * (float)t + 0.002f * (float)((int)(d % 19u) - 9);
            pool_gate_values[t * D + d] =
                -0.2f + 0.07f * (float)t - 0.001f * (float)(d % 23u);
        }
    }
    ds4_gpu_tensor *pool_cache =
        ds4_gpu_tensor_alloc((uint64_t)POOL_COUNT * D * sizeof(float));
    const uint64_t pool_tail_bytes =
        (uint64_t)POOL * D * sizeof(float);
    ds4_gpu_tensor *pool_tail_k =
        ds4_gpu_tensor_alloc(2u * pool_tail_bytes);
    ds4_gpu_tensor *pool_tail_gate =
        ds4_gpu_tensor_view(pool_tail_k, pool_tail_bytes, pool_tail_bytes);
    ds4_gpu_tensor *pool_raw_gpu =
        ds4_gpu_tensor_alloc(8u * D * sizeof(float));
    ds4_gpu_tensor *pool_gate_gpu =
        ds4_gpu_tensor_alloc(8u * D * sizeof(float));
    require_ok(pool_cache && pool_tail_k && pool_tail_gate &&
               pool_raw_gpu && pool_gate_gpu, "pool tensor allocation");
    require_ok(ds4_gpu_tensor_fill_f32(pool_cache, 0.0f,
                                       (uint64_t)POOL_COUNT * D),
               "pool cache clear");
    require_ok(ds4_gpu_tensor_fill_f32(pool_tail_k, 0.0f,
                                       (uint64_t)POOL * D),
               "pool K tail clear");
    require_ok(ds4_gpu_tensor_fill_f32(pool_tail_gate, 0.0f,
                                       (uint64_t)POOL * D),
               "pool gate tail clear");
    const uint32_t pool_chunks[] = {3, 8};
    uint32_t pool_pos = 0;
    for (uint32_t c = 0;
         c < sizeof(pool_chunks) / sizeof(pool_chunks[0]); c++) {
        const uint32_t rows = pool_chunks[c];
        require_ok(ds4_gpu_tensor_write(pool_raw_gpu, 0,
                                        pool_raw + (uint64_t)pool_pos * D,
                                        (uint64_t)rows * D * sizeof(float)),
                   "pool raw write");
        require_ok(ds4_gpu_tensor_write(pool_gate_gpu, 0,
                                        pool_gate_values + (uint64_t)pool_pos * D,
                                        (uint64_t)rows * D * sizeof(float)),
                   "pool gate write");
        require_ok(ds4_gpu_glm53_indexer_pool_update_tensor(
            pool_cache, pool_tail_k, pool_tail_gate,
            pool_raw_gpu, pool_gate_gpu,
            model, MODEL_BYTES, POOL_NORM_OFFSET, POOL_BIAS_OFFSET, POOL_APE_OFFSET,
            pool_pos, rows, POOL_CAP, D, POOL, 1e-6f, false),
            "GLM-5.3 indexer pool update");
        pool_pos += rows;
    }
    float pool_actual[POOL_COUNT * D];
    require_ok(ds4_gpu_tensor_read(pool_cache, 0, pool_actual,
                                   sizeof(pool_actual)), "pool cache read");
    for (uint32_t p = 0; p < 2u; p++) {
        float means[POOL], invs[POOL];
        for (uint32_t r = 0; r < POOL; r++) {
            const float *row = pool_raw + (uint64_t)(p * POOL + r) * D;
            float sum = 0.0f;
            for (uint32_t d = 0; d < D; d++) sum += row[d];
            means[r] = sum / (float)D;
            float ss = 0.0f;
            for (uint32_t d = 0; d < D; d++) {
                const float delta = row[d] - means[r];
                ss += delta * delta;
            }
            invs[r] = 1.0f / sqrtf(ss / (float)D + 1e-6f);
        }
        for (uint32_t d = 0; d < D; d++) {
            float logits[POOL], max_logit = -FLT_MAX, denom = 0.0f;
            for (uint32_t r = 0; r < POOL; r++) {
                logits[r] = pool_gate_values[(uint64_t)(p * POOL + r) * D + d] +
                    bf16_to_f32(pool_ape[r * D + d]);
                if (logits[r] > max_logit) max_logit = logits[r];
            }
            for (uint32_t r = 0; r < POOL; r++) {
                logits[r] = expf(logits[r] - max_logit);
                denom += logits[r];
            }
            float expected_pool = 0.0f;
            for (uint32_t r = 0; r < POOL; r++) {
                const float value =
                    (pool_raw[(uint64_t)(p * POOL + r) * D + d] - means[r]) *
                    invs[r] * pool_norm[d] + pool_bias[d];
                expected_pool += logits[r] / denom * value;
            }
            require_close("GLM-5.3 pool", pool_actual[p * D + d],
                          expected_pool, 2e-5f);
        }
    }

    enum { SCORE_ROWS = 3, SCORE_TOKENS = 5, SCORE_POS0 = 4 };
    float score_q[SCORE_TOKENS * HEADS * D];
    float score_weights[SCORE_TOKENS * HEADS];
    float score_cache[SCORE_ROWS * D];
    for (uint32_t t = 0; t < SCORE_TOKENS; t++) {
        for (uint32_t h = 0; h < HEADS; h++) {
            score_weights[t * HEADS + h] =
                0.25f + 0.1f * (float)t - 0.05f * (float)h;
            for (uint32_t d = 0; d < D; d++) {
                score_q[((uint64_t)t * HEADS + h) * D + d] =
                    0.01f * (float)(t + 1u) +
                    0.02f * (float)h +
                    0.0001f * (float)d;
            }
        }
    }
    for (uint32_t row = 0; row < SCORE_ROWS; row++) {
        for (uint32_t d = 0; d < D; d++) {
            score_cache[row * D + d] =
                0.03f * (float)(row + 1u) - 0.0002f * (float)d;
        }
    }
    ds4_gpu_tensor *score_q_gpu = ds4_gpu_tensor_alloc(sizeof(score_q));
    ds4_gpu_tensor *score_weights_gpu =
        ds4_gpu_tensor_alloc(sizeof(score_weights));
    ds4_gpu_tensor *score_cache_gpu = ds4_gpu_tensor_alloc(sizeof(score_cache));
    ds4_gpu_tensor *scores_gpu = ds4_gpu_tensor_alloc(
        (uint64_t)SCORE_TOKENS * SCORE_ROWS * sizeof(float));
    require_ok(score_q_gpu && score_weights_gpu && score_cache_gpu && scores_gpu,
               "grouped scorer tensor allocation");
    require_ok(ds4_gpu_tensor_write(score_q_gpu, 0, score_q, sizeof(score_q)),
               "grouped scorer Q write");
    require_ok(ds4_gpu_tensor_write(score_weights_gpu, 0, score_weights,
                                    sizeof(score_weights)),
               "grouped scorer weights write");
    require_ok(ds4_gpu_tensor_write(score_cache_gpu, 0, score_cache,
                                    sizeof(score_cache)),
               "grouped scorer cache write");
    const float score_scale = 0.125f;
    require_ok(ds4_gpu_glm53_indexer_scores_batch_tensor(
        scores_gpu, score_q_gpu, score_weights_gpu, score_cache_gpu,
        SCORE_ROWS, SCORE_TOKENS, SCORE_POS0, POOL,
        HEADS, D, score_scale, false), "GLM-5.3 grouped indexer scores");
    float scores_actual[SCORE_TOKENS * SCORE_ROWS];
    require_ok(ds4_gpu_tensor_read(scores_gpu, 0, scores_actual,
                                   sizeof(scores_actual)),
               "grouped scorer output read");
    for (uint32_t t = 0; t < SCORE_TOKENS; t++) {
        const uint32_t visible = (SCORE_POS0 + t + 1u) / POOL;
        for (uint32_t row = 0; row < SCORE_ROWS; row++) {
            const float actual_score = scores_actual[t * SCORE_ROWS + row];
            if (row >= visible) {
                if (!isinf(actual_score) || actual_score >= 0.0f) {
                    fprintf(stderr,
                            "grouped scorer row %u token %u should be hidden\n",
                            row, t);
                    return 1;
                }
                continue;
            }
            float expected_score = 0.0f;
            for (uint32_t h = 0; h < HEADS; h++) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < D; d++) {
                    dot += score_q[((uint64_t)t * HEADS + h) * D + d] *
                           score_cache[row * D + d];
                }
                expected_score += score_weights[t * HEADS + h] * dot;
            }
            require_close("GLM-5.3 grouped indexer score", actual_score,
                          expected_score * score_scale, 2e-5f);
        }
    }
    ds4_gpu_tensor_free(scores_gpu);
    ds4_gpu_tensor_free(score_cache_gpu);
    ds4_gpu_tensor_free(score_weights_gpu);
    ds4_gpu_tensor_free(score_q_gpu);

    enum { SELECTED_POOLS = 512, INDEX_TOPK = 2048, SELECT_ROWS = 5,
           SELECT_WIDTH = 2051 };
    uint32_t pool_ids[SELECT_ROWS * SELECTED_POOLS];
    for (uint32_t t = 0; t < SELECT_ROWS; t++) {
        for (uint32_t i = 0; i < SELECTED_POOLS; i++) {
            pool_ids[t * SELECTED_POOLS + i] = SELECTED_POOLS - 1u - i;
        }
    }
    ds4_gpu_tensor *pool_ids_gpu = ds4_gpu_tensor_alloc(sizeof(pool_ids));
    ds4_gpu_tensor *raw_ids_gpu =
        ds4_gpu_tensor_alloc((uint64_t)SELECT_ROWS * SELECT_WIDTH * sizeof(uint32_t));
    require_ok(pool_ids_gpu && raw_ids_gpu, "pool selection tensor allocation");
    require_ok(ds4_gpu_tensor_write(pool_ids_gpu, 0, pool_ids, sizeof(pool_ids)),
               "pool selection write");
    require_ok(ds4_gpu_glm53_expand_pool_selection_tensor(
        raw_ids_gpu, pool_ids_gpu, SELECT_ROWS, INDEX_TOPK,
        SELECTED_POOLS, INDEX_TOPK, POOL, SELECT_WIDTH),
        "pool selection expansion");
    uint32_t raw_ids[SELECT_ROWS * SELECT_WIDTH];
    require_ok(ds4_gpu_tensor_read(raw_ids_gpu, 0, raw_ids, sizeof(raw_ids)),
               "pool selection read");
    for (uint32_t t = 0; t < SELECT_ROWS; t++) {
        for (uint32_t i = 0; i < INDEX_TOPK; i++) {
            const uint32_t p = pool_ids[t * SELECTED_POOLS + i / POOL];
            if (raw_ids[t * SELECT_WIDTH + i] != p * POOL + i % POOL) {
                fprintf(stderr, "pool expansion mismatch at row %u slot %u\n", t, i);
                return 1;
            }
        }
        const uint32_t visible = INDEX_TOPK + t + 1u;
        const uint32_t tail_count = visible % POOL;
        for (uint32_t i = 0; i < POOL - 1u; i++) {
            const uint32_t expected_id =
                i < tail_count ? visible - tail_count + i : UINT32_MAX;
            if (raw_ids[t * SELECT_WIDTH + INDEX_TOPK + i] != expected_id) {
                fprintf(stderr, "pool tail mismatch at row %u slot %u\n", t, i);
                return 1;
            }
        }
    }
    ds4_gpu_tensor_free(raw_ids_gpu);
    ds4_gpu_tensor_free(pool_ids_gpu);
    ds4_gpu_tensor_free(pool_gate_gpu);
    ds4_gpu_tensor_free(pool_raw_gpu);
    ds4_gpu_tensor_free(pool_tail_gate);
    ds4_gpu_tensor_free(pool_tail_k);
    ds4_gpu_tensor_free(pool_cache);

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(PROJECTION * sizeof(float));
    ds4_gpu_tensor *k = ds4_gpu_tensor_alloc(PROJECTION * sizeof(float));
    ds4_gpu_tensor *v = ds4_gpu_tensor_alloc(PROJECTION * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(PROJECTION * sizeof(float));
    ds4_gpu_tensor *beta = ds4_gpu_tensor_alloc(HEADS * sizeof(float));
    ds4_gpu_tensor *output_gate = ds4_gpu_tensor_alloc(PROJECTION * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(PROJECTION * sizeof(float));
    ds4_gpu_tensor *conv = ds4_gpu_tensor_alloc(9u * PROJECTION * sizeof(float));
    ds4_gpu_tensor *state = ds4_gpu_tensor_alloc((uint64_t)HEADS * D * D * sizeof(float));
    require_ok(q && k && v && gate && beta && output_gate && out && conv && state,
               "decode tensor allocation");

    float ones[PROJECTION], zeros[PROJECTION], beta_zero[HEADS];
    for (uint32_t i = 0; i < PROJECTION; i++) {
        ones[i] = 1.0f;
        zeros[i] = 0.0f;
    }
    for (uint32_t i = 0; i < HEADS; i++) beta_zero[i] = 0.0f;
    require_ok(ds4_gpu_tensor_write(q, 0, ones, sizeof(ones)), "Q write");
    require_ok(ds4_gpu_tensor_write(k, 0, ones, sizeof(ones)), "K write");
    require_ok(ds4_gpu_tensor_write(v, 0, ones, sizeof(ones)), "V write");
    require_ok(ds4_gpu_tensor_write(gate, 0, zeros, sizeof(zeros)), "gate write");
    require_ok(ds4_gpu_tensor_write(output_gate, 0, zeros, sizeof(zeros)), "output gate write");
    require_ok(ds4_gpu_tensor_write(beta, 0, beta_zero, sizeof(beta_zero)), "beta write");
    require_ok(ds4_gpu_tensor_fill_f32(conv, 0.0f, 9u * PROJECTION), "conv clear");
    require_ok(ds4_gpu_tensor_fill_f32(state, 0.0f, (uint64_t)HEADS * D * D), "state clear");
    require_ok(ds4_gpu_glm53_kda_decode(
        out, conv, state, q, k, v, gate, beta, output_gate,
        model, MODEL_BYTES, Q_CONV_OFFSET, K_CONV_OFFSET, V_CONV_OFFSET,
        A_LOG_OFFSET, DT_BIAS_OFFSET, NORM_OFFSET,
        HEADS, 1, -5.0f, 1e-5f), "KDA decode");
    float actual[PROJECTION];
    require_ok(ds4_gpu_tensor_read(out, 0, actual, sizeof(actual)), "output read");
    const float silu_one = 1.0f / (1.0f + expf(-1.0f));
    const float raw = 0.5f * silu_one / sqrtf((float)D);
    const float expected = 0.5f * raw / sqrtf(raw * raw + 1e-5f);
    for (uint32_t i = 0; i < PROJECTION; i++)
        require_close("KDA decode", actual[i], expected, 2e-5f);

    float qs[TOKENS * PROJECTION], ks[TOKENS * PROJECTION];
    float vs[TOKENS * PROJECTION], gates[TOKENS * PROJECTION];
    float output_gates[TOKENS * PROJECTION], betas[TOKENS * HEADS];
    for (uint32_t t = 0; t < TOKENS; t++) {
        for (uint32_t h = 0; h < HEADS; h++)
            betas[t * HEADS + h] = -0.25f + 0.2f * (float)t + 0.1f * (float)h;
        for (uint32_t d = 0; d < PROJECTION; d++) {
            const uint32_t i = t * PROJECTION + d;
            qs[i] = 0.1f + 0.002f * (float)(d % 17u) + 0.03f * (float)t;
            ks[i] = -0.08f + 0.001f * (float)(d % 23u) + 0.02f * (float)t;
            vs[i] = 0.05f - 0.0015f * (float)(d % 13u) + 0.04f * (float)t;
            gates[i] = -0.2f + 0.003f * (float)(d % 11u);
            output_gates[i] = 0.15f - 0.002f * (float)(d % 7u);
        }
    }

    require_ok(ds4_gpu_tensor_fill_f32(conv, 0.0f, 9u * PROJECTION), "decode conv reset");
    require_ok(ds4_gpu_tensor_fill_f32(state, 0.0f, (uint64_t)HEADS * D * D), "decode state reset");
    float decode_outputs[TOKENS * PROJECTION];
    for (uint32_t t = 0; t < TOKENS; t++) {
        const uint32_t off = t * PROJECTION;
        require_ok(ds4_gpu_tensor_write(q, 0, qs + off, PROJECTION * sizeof(float)), "decode Q write");
        require_ok(ds4_gpu_tensor_write(k, 0, ks + off, PROJECTION * sizeof(float)), "decode K write");
        require_ok(ds4_gpu_tensor_write(v, 0, vs + off, PROJECTION * sizeof(float)), "decode V write");
        require_ok(ds4_gpu_tensor_write(gate, 0, gates + off, PROJECTION * sizeof(float)), "decode gate write");
        require_ok(ds4_gpu_tensor_write(output_gate, 0, output_gates + off, PROJECTION * sizeof(float)), "decode output gate write");
        require_ok(ds4_gpu_tensor_write(beta, 0, betas + t * HEADS, HEADS * sizeof(float)), "decode beta write");
        require_ok(ds4_gpu_glm53_kda_decode(
            out, conv, state, q, k, v, gate, beta, output_gate,
            model, MODEL_BYTES, Q_CONV_OFFSET, K_CONV_OFFSET, V_CONV_OFFSET,
            A_LOG_OFFSET, DT_BIAS_OFFSET, NORM_OFFSET,
            HEADS, 1, -5.0f, 1e-5f), "consistency decode");
        require_ok(ds4_gpu_tensor_read(out, 0, decode_outputs + off,
                                       PROJECTION * sizeof(float)), "decode output read");
    }

    ds4_gpu_tensor *pq = ds4_gpu_tensor_alloc(sizeof(qs));
    ds4_gpu_tensor *pk = ds4_gpu_tensor_alloc(sizeof(ks));
    ds4_gpu_tensor *pv = ds4_gpu_tensor_alloc(sizeof(vs));
    ds4_gpu_tensor *pg = ds4_gpu_tensor_alloc(sizeof(gates));
    ds4_gpu_tensor *poutput_gate = ds4_gpu_tensor_alloc(sizeof(output_gates));
    ds4_gpu_tensor *pbeta = ds4_gpu_tensor_alloc(sizeof(betas));
    ds4_gpu_tensor *pout = ds4_gpu_tensor_alloc(sizeof(qs));
    ds4_gpu_tensor *pconv = ds4_gpu_tensor_alloc(9u * PROJECTION * sizeof(float));
    ds4_gpu_tensor *pstate = ds4_gpu_tensor_alloc((uint64_t)HEADS * D * D * sizeof(float));
    require_ok(pq && pk && pv && pg && poutput_gate && pbeta && pout && pconv && pstate,
               "prefill tensor allocation");
    require_ok(ds4_gpu_tensor_write(pq, 0, qs, sizeof(qs)), "prefill Q write");
    require_ok(ds4_gpu_tensor_write(pk, 0, ks, sizeof(ks)), "prefill K write");
    require_ok(ds4_gpu_tensor_write(pv, 0, vs, sizeof(vs)), "prefill V write");
    require_ok(ds4_gpu_tensor_write(pg, 0, gates, sizeof(gates)), "prefill gate write");
    require_ok(ds4_gpu_tensor_write(poutput_gate, 0, output_gates, sizeof(output_gates)), "prefill output gate write");
    require_ok(ds4_gpu_tensor_write(pbeta, 0, betas, sizeof(betas)), "prefill beta write");
    require_ok(ds4_gpu_tensor_fill_f32(pconv, 0.0f, 9u * PROJECTION), "prefill conv clear");
    require_ok(ds4_gpu_tensor_fill_f32(pstate, 0.0f, (uint64_t)HEADS * D * D), "prefill state clear");
    require_ok(ds4_gpu_glm53_kda_prefill(
        pout, pconv, pstate, pq, pk, pv, pg, pbeta, poutput_gate,
        model, MODEL_BYTES, Q_CONV_OFFSET, K_CONV_OFFSET, V_CONV_OFFSET,
        A_LOG_OFFSET, DT_BIAS_OFFSET, NORM_OFFSET,
        HEADS, TOKENS, -5.0f, 1e-5f), "KDA prefill");
    float prefill_outputs[TOKENS * PROJECTION];
    require_ok(ds4_gpu_tensor_read(pout, 0, prefill_outputs, sizeof(prefill_outputs)),
               "prefill output read");
    for (uint32_t i = 0; i < TOKENS * PROJECTION; i++)
        require_close("KDA prefill/decode", prefill_outputs[i], decode_outputs[i], 5e-5f);

    ds4_gpu_tensor_free(pstate);
    ds4_gpu_tensor_free(pconv);
    ds4_gpu_tensor_free(pout);
    ds4_gpu_tensor_free(pbeta);
    ds4_gpu_tensor_free(poutput_gate);
    ds4_gpu_tensor_free(pg);
    ds4_gpu_tensor_free(pv);
    ds4_gpu_tensor_free(pk);
    ds4_gpu_tensor_free(pq);
    ds4_gpu_tensor_free(state);
    ds4_gpu_tensor_free(conv);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(output_gate);
    ds4_gpu_tensor_free(beta);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(bf16_out);
    ds4_gpu_tensor_free(bf16_x);
#ifdef __APPLE__
    /* Never silently compare the reference with itself because the parent
     * process has disabled tuning. Each oracle asserts dispatch coverage. */
    require_ok(unsetenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING") == 0,
               "clear inherited aggregate tuning switch for kernel oracles");
    require_ok(unsetenv("DS4_METAL_DISABLE_GLM53_PREFILL_INDEXED_ATTN") == 0,
               "clear inherited attention tuning switch for kernel oracles");
    ds4_gpu_test_set_flags(DS4_GPU_TEST_GLM53_PREFILL);
    ds4_gpu_test_glm53_prefill_take_dispatches();
    check_glm53_qk_lowrank_token_tile(model, MODEL_BYTES, QK_LOW_KB_OFFSET);
    check_glm53_qk_lowrank_large_offsets(model, MODEL_BYTES, QK_LOW_KB_OFFSET);
    check_glm53_bf16_short_rows(model, MODEL_BYTES, QK_LOW_KB_OFFSET);
    check_glm53_dsa_score_tile(model, MODEL_BYTES, QK_LOW_KB_OFFSET);
    check_glm53_kda_decode_values4(model, MODEL_BYTES, QK_LOW_KB_OFFSET);
    check_glm53_kda_inputs(model, MODEL_BYTES, KI_WEIGHT_OFFSET);
    check_glm53_indexed_attention_head_width();
    check_glm53_indexed_attention_invalid_rows();
    check_glm53_routed_moe_tail_cull(model, MODEL_BYTES, MOE_GATE_OFFSET,
                                     MOE_UP_OFFSET, MOE_DOWN_OFFSET);
    check_glm53_kda_prepare_blocked(model, MODEL_BYTES, KP_Q_OFFSET,
                                    KP_K_OFFSET, KP_V_OFFSET, KP_A_OFFSET,
                                    KP_DT_OFFSET, KP_NORM_OFFSET);
    check_glm_router_publication(model, MODEL_BYTES, POOL_BIAS_OFFSET);
    ds4_gpu_test_set_flags(0);
#endif
    ds4_gpu_cleanup();
    munmap(model, MODEL_BYTES);
    puts("GLM-5.3 KDA GPU tests: PASS");
    return 0;
}
