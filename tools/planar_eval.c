/*
 * planar_eval.c - offline quality evaluator for Planar3 KV-cache rows.
 *
 * The tool either generates synthetic 512-dim rows or reads dumped rows from a
 * binary file, applies Planar3 quantize/dequantize, and reports row-level,
 * attention-score, and softmax-output drift metrics.
 *
 * Binary input format:
 *   uint32_t nrows
 *   uint32_t n_per_row, must be 512
 *   float32 rows[nrows][512]
 */

#include "ds4_planar_quant.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM 512
#define DEFAULT_ROWS 10000
#define DEFAULT_QUERIES 8
#define TOPK_MAX 10

static uint64_t pcg_state_;

static inline uint32_t pcg32(void) {
    uint64_t s = pcg_state_;
    pcg_state_ = s * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t x = (uint32_t)(((s >> 18) ^ s) >> 27);
    int r = (int)(s >> 59);
    return (x >> r) | (x << ((-r) & 31));
}

static inline float randf01(void) {
    return (float)((double)pcg32() / 4294967296.0);
}

static inline float randf_uniform(void) {
    return randf01() * 2.0f - 1.0f;
}

static inline float randf_normal(void) {
    const float u1 = (float)(((double)pcg32() + 1.0) / 4294967297.0);
    const float u2 = randf01();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}

static bool checked_mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}

static bool parse_int_arg(const char *s, const char *opt, int minv, int maxv, int *out) {
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || !end || *end || v < minv || v > maxv) {
        fprintf(stderr, "planar_eval: invalid value for %s: %s\n", opt, s);
        return false;
    }
    *out = (int)v;
    return true;
}

static bool parse_u64_arg(const char *s, const char *opt, uint64_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || !end || *end) {
        fprintf(stderr, "planar_eval: invalid value for %s: %s\n", opt, s);
        return false;
    }
    *out = (uint64_t)v;
    return true;
}

static float *alloc_rows(int nrows) {
    size_t count, bytes;
    if (!checked_mul_size((size_t)nrows, DIM, &count) ||
        !checked_mul_size(count, sizeof(float), &bytes)) {
        return NULL;
    }
    return (float *)malloc(bytes);
}

static int load_input_file(const char *path, float **data_out, int *nrows_out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "planar_eval: cannot open input file %s: %s\n",
                path, strerror(errno));
        return 0;
    }

    uint32_t hdr_nrows = 0;
    uint32_t hdr_dim = 0;
    if (fread(&hdr_nrows, sizeof(hdr_nrows), 1, f) != 1 ||
        fread(&hdr_dim, sizeof(hdr_dim), 1, f) != 1) {
        fprintf(stderr, "planar_eval: failed to read header from %s\n", path);
        fclose(f);
        return 0;
    }
    if (hdr_dim != DIM) {
        fprintf(stderr, "planar_eval: input file has n_per_row=%u, expected %d\n",
                hdr_dim, DIM);
        fclose(f);
        return 0;
    }
    if (hdr_nrows == 0 || hdr_nrows > (uint32_t)INT_MAX) {
        fprintf(stderr, "planar_eval: input file has invalid nrows=%u\n", hdr_nrows);
        fclose(f);
        return 0;
    }

    int nrows = (int)hdr_nrows;
    float *data = alloc_rows(nrows);
    if (!data) {
        fprintf(stderr, "planar_eval: failed to allocate %d rows x %d dims\n",
                nrows, DIM);
        fclose(f);
        return 0;
    }

    const size_t total = (size_t)nrows * DIM;
    if (fread(data, sizeof(float), total, f) != total) {
        fprintf(stderr, "planar_eval: failed to read %zu float32 values from %s\n",
                total, path);
        free(data);
        fclose(f);
        return 0;
    }

    fclose(f);
    *data_out = data;
    *nrows_out = nrows;
    return 1;
}

static void gen_random_normal(float *buf, int nrows, uint64_t seed) {
    pcg_state_ = seed;
    for (size_t i = 0, n = (size_t)nrows * DIM; i < n; i++)
        buf[i] = randf_normal();
}

static void gen_random_uniform(float *buf, int nrows, uint64_t seed) {
    pcg_state_ = seed;
    for (size_t i = 0, n = (size_t)nrows * DIM; i < n; i++)
        buf[i] = randf_uniform();
}

static void gen_sparse(float *buf, int nrows, uint64_t seed) {
    pcg_state_ = seed;
    for (size_t i = 0, n = (size_t)nrows * DIM; i < n; i++)
        buf[i] = (pcg32() % 10 == 0) ? randf_normal() : 0.0f;
}

static void gen_ds4_like(float *buf, int nrows, uint64_t seed) {
    pcg_state_ = seed;
    for (int r = 0; r < nrows; r++) {
        float *row = buf + (size_t)r * DIM;
        float base[DIM];
        memset(base, 0, sizeof(base));

        const int nwaves = 3 + (int)(pcg32() % 3);
        for (int w = 0; w < nwaves; w++) {
            const float freq  = 0.01f + randf01() * 0.1f;
            const float amp   = 0.5f + randf01() * 0.5f;
            const float phase = randf01() * 6.283185307179586f;
            for (int j = 0; j < DIM; j++)
                base[j] += amp * sinf(6.283185307179586f * freq * (float)j + phase);
        }

        float ms = 0.0f;
        for (int j = 0; j < DIM; j++) ms += base[j] * base[j];
        const float rms = sqrtf(ms / (float)DIM + 1e-6f);
        const float learned_norm = 0.9f + randf01() * 0.2f;
        for (int j = 0; j < DIM; j++)
            row[j] = base[j] / rms * learned_norm;

        for (int j = 0; j < DIM; j += 2) {
            if ((pcg32() >> 16) & 1u) {
                row[j] = -row[j];
                row[j + 1] = -row[j + 1];
            }
        }

        for (int j = 0; j < DIM; j++)
            row[j] += randf_normal() * 0.01f;
    }
}

static int cmp_float(const void *a, const void *b) {
    const float fa = *(const float *)a;
    const float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static float percentile_sorted(const float *arr, int n, float pct) {
    const float idx = (float)(n - 1) * pct;
    const int lo = (int)idx;
    const int hi = lo + 1;
    if (hi >= n) return arr[n - 1];
    const float frac = idx - (float)lo;
    return arr[lo] * (1.0f - frac) + arr[hi] * frac;
}

static float vec_cosine(const float *a, const float *b, int n) {
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    const double denom = sqrt(na) * sqrt(nb);
    return denom > 1e-20 ? (float)(dot / denom) : 0.0f;
}

static float rel_l2_diff(const float *ref, const float *got, int n) {
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < n; i++) {
        const double d = (double)ref[i] - got[i];
        num += d * d;
        den += (double)ref[i] * ref[i];
    }
    return den > 1e-20 ? (float)(sqrt(num) / sqrt(den)) : 0.0f;
}

static void fill_query(float *query) {
    double ss = 0.0;
    for (int j = 0; j < DIM; j++) {
        query[j] = randf_normal();
        ss += (double)query[j] * query[j];
    }
    const float scale = ss > 1e-20 ? (float)(sqrt((double)DIM / ss)) : 1.0f;
    for (int j = 0; j < DIM; j++) query[j] *= scale;
}

static void dot_scores(const float *rows, int nrows, const float *query, float *scores) {
    const float attn_scale = 1.0f / sqrtf((float)DIM);
    for (int r = 0; r < nrows; r++) {
        const float *row = rows + (size_t)r * DIM;
        float s = 0.0f;
        for (int j = 0; j < DIM; j++) s += query[j] * row[j];
        scores[r] = s * attn_scale;
    }
}

static double pearson_corr(const float *a, const float *b, int n) {
    double ma = 0.0;
    double mb = 0.0;
    for (int i = 0; i < n; i++) {
        ma += a[i];
        mb += b[i];
    }
    ma /= n;
    mb /= n;

    double cov = 0.0;
    double va = 0.0;
    double vb = 0.0;
    for (int i = 0; i < n; i++) {
        const double da = a[i] - ma;
        const double db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    const double denom = sqrt(va * vb);
    return denom > 1e-20 ? cov / denom : 0.0;
}

static void topk_indices(const float *scores, int n, int k, int *idx) {
    for (int i = 0; i < k; i++) idx[i] = -1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            if (idx[j] < 0 || scores[i] > scores[idx[j]]) {
                for (int m = k - 1; m > j; m--) idx[m] = idx[m - 1];
                idx[j] = i;
                break;
            }
        }
    }
}

static int topk_overlap(const int *a, const int *b, int k) {
    int overlap = 0;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            if (a[i] == b[j]) {
                overlap++;
                break;
            }
        }
    }
    return overlap;
}

static void softmax_output(const float *scores, const float *rows, int nrows, float *out) {
    float max_score = scores[0];
    for (int r = 1; r < nrows; r++) {
        if (scores[r] > max_score) max_score = scores[r];
    }

    memset(out, 0, DIM * sizeof(float));
    double denom = 0.0;
    for (int r = 0; r < nrows; r++) {
        const float w = expf(scores[r] - max_score);
        const float *row = rows + (size_t)r * DIM;
        denom += w;
        for (int j = 0; j < DIM; j++) out[j] += w * row[j];
    }

    if (denom > 0.0) {
        const float inv = (float)(1.0 / denom);
        for (int j = 0; j < DIM; j++) out[j] *= inv;
    }
}

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --rows N          Number of rows to generate (default: %d)\n", DEFAULT_ROWS);
    printf("  --queries N       Number of random attention probes (default: %d)\n", DEFAULT_QUERIES);
    printf("  --mode MODE       random_normal, random_uniform, sparse, ds4_like, ds4_realistic\n");
    printf("                     default: random_normal; ds4_realistic is an alias for ds4_like\n");
    printf("  --input FILE      Load rows from binary file instead of generating\n");
    printf("                     format: uint32 nrows, uint32 n_per_row, float32 data\n");
    printf("  --seed N          PRNG seed (default: 42)\n");
    printf("  --help            Show this help\n");
}

int main(int argc, char **argv) {
    int nrows = DEFAULT_ROWS;
    int nqueries = DEFAULT_QUERIES;
    const char *mode_str = "random_normal";
    const char *input_file = NULL;
    uint64_t seed = 42;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--rows")) {
            if (++i >= argc || !parse_int_arg(argv[i], "--rows", 1, INT_MAX / DIM, &nrows))
                return 1;
        } else if (!strcmp(argv[i], "--queries")) {
            if (++i >= argc || !parse_int_arg(argv[i], "--queries", 1, 1000000, &nqueries))
                return 1;
        } else if (!strcmp(argv[i], "--mode")) {
            if (++i >= argc) {
                fprintf(stderr, "planar_eval: --mode requires an argument\n");
                return 1;
            }
            mode_str = argv[i];
        } else if (!strcmp(argv[i], "--input")) {
            if (++i >= argc) {
                fprintf(stderr, "planar_eval: --input requires an argument\n");
                return 1;
            }
            input_file = argv[i];
        } else if (!strcmp(argv[i], "--seed")) {
            if (++i >= argc || !parse_u64_arg(argv[i], "--seed", &seed))
                return 1;
        } else if (!strcmp(argv[i], "--help")) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "planar_eval: unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    float *data_orig = NULL;
    if (input_file) {
        if (!load_input_file(input_file, &data_orig, &nrows)) return 1;
    } else {
        data_orig = alloc_rows(nrows);
        if (!data_orig) {
            fprintf(stderr, "planar_eval: failed to allocate %d rows x %d dims\n",
                    nrows, DIM);
            return 1;
        }
        if (!strcmp(mode_str, "random_normal")) {
            gen_random_normal(data_orig, nrows, seed);
        } else if (!strcmp(mode_str, "random_uniform")) {
            gen_random_uniform(data_orig, nrows, seed);
        } else if (!strcmp(mode_str, "sparse")) {
            gen_sparse(data_orig, nrows, seed);
        } else if (!strcmp(mode_str, "ds4_like") || !strcmp(mode_str, "ds4_realistic")) {
            gen_ds4_like(data_orig, nrows, seed);
        } else {
            fprintf(stderr, "planar_eval: unknown mode: %s\n", mode_str);
            print_usage(argv[0]);
            free(data_orig);
            return 1;
        }
    }

    float *data_recon = alloc_rows(nrows);
    ds4_row_planar3 *compressed = (ds4_row_planar3 *)malloc((size_t)nrows * sizeof(ds4_row_planar3));
    float *cosine_arr = (float *)malloc((size_t)nrows * sizeof(float));
    float *mse_arr = (float *)malloc((size_t)nrows * sizeof(float));
    float *maxerr_arr = (float *)malloc((size_t)nrows * sizeof(float));
    float *relnorm_arr = (float *)malloc((size_t)nrows * sizeof(float));
    float *score_orig = (float *)malloc((size_t)nrows * sizeof(float));
    float *score_recon = (float *)malloc((size_t)nrows * sizeof(float));
    if (!data_recon || !compressed || !cosine_arr || !mse_arr ||
        !maxerr_arr || !relnorm_arr || !score_orig || !score_recon) {
        fprintf(stderr, "planar_eval: allocation failure for %d rows\n", nrows);
        free(data_orig);
        free(data_recon);
        free(compressed);
        free(cosine_arr);
        free(mse_arr);
        free(maxerr_arr);
        free(relnorm_arr);
        free(score_orig);
        free(score_recon);
        return 1;
    }

    const size_t bytes_written =
        ds4_planar3_quantize(data_orig, compressed, (size_t)nrows, DIM);
    ds4_planar3_dequantize(compressed, data_recon, (size_t)nrows, DIM);

    double cos_mean = 0.0;
    double mse_mean = 0.0;
    double maxerr_mean = 0.0;
    double relnorm_mean = 0.0;
    for (int r = 0; r < nrows; r++) {
        const float *orig = data_orig + (size_t)r * DIM;
        const float *rec = data_recon + (size_t)r * DIM;

        cosine_arr[r] = ds4_planar3_roundtrip_cosine(orig, (float *)rec, DIM);
        mse_arr[r] = ds4_planar3_roundtrip_mse(orig, (float *)rec, DIM);

        float maxerr = 0.0f;
        float n1 = 0.0f;
        float n2 = 0.0f;
        for (int j = 0; j < DIM; j++) {
            const float d = fabsf(orig[j] - rec[j]);
            if (d > maxerr) maxerr = d;
            n1 += orig[j] * orig[j];
            n2 += rec[j] * rec[j];
        }
        maxerr_arr[r] = maxerr;
        n1 = sqrtf(n1);
        n2 = sqrtf(n2);
        relnorm_arr[r] = n1 > 1e-10f ? fabsf(n1 - n2) / n1 : 0.0f;

        cos_mean += cosine_arr[r];
        mse_mean += mse_arr[r];
        maxerr_mean += maxerr_arr[r];
        relnorm_mean += relnorm_arr[r];
    }
    cos_mean /= nrows;
    mse_mean /= nrows;
    maxerr_mean /= nrows;
    relnorm_mean /= nrows;

    qsort(cosine_arr, (size_t)nrows, sizeof(float), cmp_float);
    qsort(mse_arr, (size_t)nrows, sizeof(float), cmp_float);
    qsort(maxerr_arr, (size_t)nrows, sizeof(float), cmp_float);
    qsort(relnorm_arr, (size_t)nrows, sizeof(float), cmp_float);

    double corr_sum = 0.0;
    double score_abs_sum = 0.0;
    double score_sq_sum = 0.0;
    float score_max_diff = 0.0f;
    int top1_agree = 0;
    int topk_overlap_sum = 0;
    double vonly_cos_sum = 0.0;
    double vonly_rel_l2_sum = 0.0;
    double full_cos_sum = 0.0;
    double full_rel_l2_sum = 0.0;
    const int topk = nrows < TOPK_MAX ? nrows : TOPK_MAX;

    float query[DIM];
    float out_orig[DIM];
    float out_vonly[DIM];
    float out_full[DIM];
    int top_orig[TOPK_MAX];
    int top_recon[TOPK_MAX];

    pcg_state_ = seed ^ 0x9E3779B97F4A7C15ULL;
    for (int q = 0; q < nqueries; q++) {
        fill_query(query);
        dot_scores(data_orig, nrows, query, score_orig);
        dot_scores(data_recon, nrows, query, score_recon);

        corr_sum += pearson_corr(score_orig, score_recon, nrows);
        for (int r = 0; r < nrows; r++) {
            const float d = fabsf(score_orig[r] - score_recon[r]);
            score_abs_sum += d;
            score_sq_sum += (double)d * d;
            if (d > score_max_diff) score_max_diff = d;
        }

        topk_indices(score_orig, nrows, topk, top_orig);
        topk_indices(score_recon, nrows, topk, top_recon);
        if (top_orig[0] == top_recon[0]) top1_agree++;
        topk_overlap_sum += topk_overlap(top_orig, top_recon, topk);

        softmax_output(score_orig, data_orig, nrows, out_orig);
        softmax_output(score_orig, data_recon, nrows, out_vonly);
        softmax_output(score_recon, data_recon, nrows, out_full);
        vonly_cos_sum += vec_cosine(out_orig, out_vonly, DIM);
        vonly_rel_l2_sum += rel_l2_diff(out_orig, out_vonly, DIM);
        full_cos_sum += vec_cosine(out_orig, out_full, DIM);
        full_rel_l2_sum += rel_l2_diff(out_orig, out_full, DIM);
    }

    const double score_count = (double)nrows * (double)nqueries;
    const double score_mae = score_abs_sum / score_count;
    const double score_rmse = sqrt(score_sq_sum / score_count);

    printf("\n=== Planar3 Quality Evaluation ===\n");
    printf("Source: %s | Mode: %s | Rows: %d | Dim: %d | Queries: %d | Seed: %llu\n",
           input_file ? input_file : "synthetic",
           input_file ? "input" : mode_str,
           nrows, DIM, nqueries, (unsigned long long)seed);
    if (!input_file && (!strcmp(mode_str, "ds4_like") || !strcmp(mode_str, "ds4_realistic"))) {
        printf("Note: ds4_like is synthetic. Use --input with dumped compressed-KV rows for real evidence.\n");
    }

    printf("\nRow roundtrip cosine:\n");
    printf("  min=%.4f mean=%.4f median=%.4f p99=%.4f max=%.4f\n",
           cosine_arr[0], (float)cos_mean,
           percentile_sorted(cosine_arr, nrows, 0.5f),
           percentile_sorted(cosine_arr, nrows, 0.99f),
           cosine_arr[nrows - 1]);

    printf("\nRow roundtrip MSE per element:\n");
    printf("  min=%.3e mean=%.3e median=%.3e p99=%.3e max=%.3e\n",
           mse_arr[0], (float)mse_mean,
           percentile_sorted(mse_arr, nrows, 0.5f),
           percentile_sorted(mse_arr, nrows, 0.99f),
           mse_arr[nrows - 1]);

    printf("\nMax element error:\n");
    printf("  min=%.4f mean=%.4f median=%.4f p99=%.4f max=%.4f\n",
           maxerr_arr[0], (float)maxerr_mean,
           percentile_sorted(maxerr_arr, nrows, 0.5f),
           percentile_sorted(maxerr_arr, nrows, 0.99f),
           maxerr_arr[nrows - 1]);

    printf("\nRelative norm error:\n");
    printf("  min=%.3e mean=%.3e median=%.3e p99=%.3e max=%.3e\n",
           relnorm_arr[0], (float)relnorm_mean,
           percentile_sorted(relnorm_arr, nrows, 0.5f),
           percentile_sorted(relnorm_arr, nrows, 0.99f),
           relnorm_arr[nrows - 1]);

    printf("\nAttention score drift (%d random queries):\n", nqueries);
    printf("  corr_mean=%.4f mae=%.4f rmse=%.4f max_diff=%.4f\n",
           (float)(corr_sum / nqueries),
           (float)score_mae,
           (float)score_rmse,
           score_max_diff);
    printf("  top1_agree=%d/%d top%d_overlap=%.2f/%d\n",
           top1_agree, nqueries, topk,
           (float)topk_overlap_sum / (float)nqueries, topk);

    printf("\nSoftmax output drift:\n");
    printf("  V-only:   cos_mean=%.4f rel_l2_mean=%.4f\n",
           (float)(vonly_cos_sum / nqueries),
           (float)(vonly_rel_l2_sum / nqueries));
    printf("  K+V path: cos_mean=%.4f rel_l2_mean=%.4f\n",
           (float)(full_cos_sum / nqueries),
           (float)(full_rel_l2_sum / nqueries));

    const size_t compressed_bytes = sizeof(ds4_row_planar3);
    const size_t fp16_bytes = (size_t)DIM * sizeof(uint16_t);
    printf("\nCompression: %zu bytes/row (%.2fx vs FP16 %zu bytes), total=%zu bytes\n",
           compressed_bytes,
           (double)fp16_bytes / (double)compressed_bytes,
           fp16_bytes,
           bytes_written);

    free(data_orig);
    free(data_recon);
    free(compressed);
    free(cosine_arr);
    free(mse_arr);
    free(maxerr_arr);
    free(relnorm_arr);
    free(score_orig);
    free(score_recon);
    return 0;
}
