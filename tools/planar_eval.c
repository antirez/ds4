/*
 * planar_eval.c — Offline quality evaluator for Planar3 KV-cache compression.
 *
 * Standalone CLI: generates or loads KV-cache-like 512-dim vectors,
 * applies Planar3 quantize/dequantize, reports comprehensive quality metrics.
 *
 * Build:
 *   cc -O2 -Wall -Wextra -std=c99 -I. -o tools/planar_eval tools/planar_eval.c -lm
 *
 * Usage:
 *   ./tools/planar_eval --mode ds4_realistic --rows 10000 --seed 42
 */

/* Direct include for standalone tool — no separate compilation needed */
#include "ds4_planar_quant.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- PCG-style PRNG (matches planar_quant.c) ---- */

static uint64_t pcg_state_;

static inline uint32_t pcg32(void) {
    uint64_t s = pcg_state_;
    pcg_state_ = s * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t x = (uint32_t)(((s >> 18) ^ s) >> 27);
    int r = (int)(s >> 59);
    return (x >> r) | (x << ((-r) & 31));
}

static inline float randf_normal(void) {
    /* Box-Muller transform */
    float u1 = (pcg32() + 1.0f) / 4294967297.0f;
    float u2 = (float)pcg32() / 4294967296.0f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}

static inline float randf_uniform(void) {
    return (float)pcg32() / 2147483648.0f - 1.0f; /* [-1, 1) */
}

/* ---- Data generation modes ---- */

#define DIM 512

static void gen_random_normal(float *buf, int nrows, uint64_t seed) {
    pcg_state_ = seed;
    for (int i = 0; i < nrows * DIM; i++)
        buf[i] = randf_normal();
}

static void gen_random_uniform(float *buf, int nrows, uint64_t seed) {
    pcg_state_ = seed;
    for (int i = 0; i < nrows * DIM; i++)
        buf[i] = randf_uniform();
}

static void gen_sparse(float *buf, int nrows, uint64_t seed) {
    pcg_state_ = seed;
    for (int i = 0; i < nrows * DIM; i++) {
        if (pcg32() % 10 == 0)
            buf[i] = randf_normal();
        else
            buf[i] = 0.0f;
    }
}

static void gen_ds4_realistic(float *buf, int nrows, uint64_t seed) {
    pcg_state_ = seed;
    for (int r = 0; r < nrows; r++) {
        float *row = buf + r * DIM;
        int nwaves = 3 + (int)(pcg32() % 3); /* 3-5 sine waves */
        float base[512];
        memset(base, 0, sizeof(base));
        for (int w = 0; w < nwaves; w++) {
            float freq  = 0.01f + (float)pcg32() / 4294967296.0f * 0.1f;
            float amp   = 0.5f + (float)pcg32() / 4294967296.0f * 0.5f;
            float phase = (float)pcg32() / 4294967296.0f * 6.2831853f;
            for (int j = 0; j < DIM; j++)
                base[j] += amp * sinf(6.2831853f * freq * j + phase);
        }
        /* RMSNorm */
        float ms = 0.0f;
        for (int j = 0; j < DIM; j++) ms += base[j] * base[j];
        ms /= (float)DIM;
        float rms = sqrtf(ms + 1e-6f);
        float learned_norm = 0.9f + (float)pcg32() / 2147483648.0f * 0.2f; /* ~1.0 */
        for (int j = 0; j < DIM; j++)
            row[j] = base[j] / rms * learned_norm;

        /* Position-dependent sign flips (RoPE-like rotation effect) */
        for (int j = 0; j < DIM; j += 2) {
            if ((pcg32() >> 16) & 1) {
                row[j]     = -row[j];
                row[j + 1] = -row[j + 1];
            }
        }

        /* Small Gaussian noise */
        for (int j = 0; j < DIM; j++)
            row[j] += randf_normal() * 0.01f;
    }
}

/* ---- Sorting helper ---- */

static int cmp_float(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    if (fa < fb) return -1;
    if (fa > fb) return  1;
    return 0;
}

static float percentile_sorted(const float *arr, int n, float pct) {
    float idx = (float)(n - 1) * pct;
    int lo = (int)idx;
    int hi = lo + 1;
    if (hi >= n) return arr[n - 1];
    float frac = idx - (float)lo;
    return arr[lo] * (1.0f - frac) + arr[hi] * frac;
}

/* ---- Main ---- */

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --rows N          Number of rows to generate (default: 10000)\n");
    printf("  --mode MODE       Data distribution: random_normal, random_uniform, sparse, ds4_realistic (default: random_normal)\n");
    printf("  --input FILE      Load rows from binary file instead of generating\n");
    printf("                     (format: uint32 nrows, uint32 n_per_row, then nrows*n_per_row float32)\n");
    printf("  --seed N          PRNG seed (default: 42)\n");
    printf("  --help            Show this help\n");
}

int main(int argc, char **argv) {
    int nrows = 10000;
    const char *mode_str = "random_normal";
    const char *input_file = NULL;
    uint64_t seed = 42;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) {
            nrows = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode_str = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_file = argv[++i];
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    /* Allocate data */
    float *data_orig  = NULL;  /* original data */
    float *data_recon = NULL;  /* reconstructed data */

    if (input_file) {
        FILE *f = fopen(input_file, "rb");
        if (!f) { fprintf(stderr, "Cannot open input file: %s\n", input_file); return 1; }
        uint32_t hdr_nrows, hdr_dim;
        if (fread(&hdr_nrows, 4, 1, f) != 1 || fread(&hdr_dim, 4, 1, f) != 1) {
            fprintf(stderr, "Failed to read header from %s\n", input_file);
            fclose(f);
            return 1;
        }
        if (hdr_dim != DIM) {
            fprintf(stderr, "Input file has n_per_row=%u, expected %d\n", hdr_dim, DIM);
            fclose(f);
            return 1;
        }
        fclose(f);
        nrows = (int)hdr_nrows;
    }

    data_orig  = (float *)malloc((size_t)nrows * DIM * sizeof(float));
    data_recon = (float *)malloc((size_t)nrows * DIM * sizeof(float));
    if (!data_orig || !data_recon) {
        fprintf(stderr, "Failed to allocate %d rows x %d dims\n", nrows, DIM);
        return 1;
    }

    /* Generate or load data */
    if (input_file) {
        FILE *f = fopen(input_file, "rb");
        uint32_t tmp;
        if (fread(&tmp, 4, 1, f) != 1) { fclose(f); return 1; }
        if (fread(&tmp, 4, 1, f) != 1) { fclose(f); return 1; }
        size_t total = (size_t)nrows * DIM;
        if (fread(data_orig, sizeof(float), total, f) != total) {
            fprintf(stderr, "Failed to read data from %s\n", input_file);
            fclose(f);
            return 1;
        }
        fclose(f);
    } else if (strcmp(mode_str, "random_normal") == 0) {
        gen_random_normal(data_orig, nrows, seed);
    } else if (strcmp(mode_str, "random_uniform") == 0) {
        gen_random_uniform(data_orig, nrows, seed);
    } else if (strcmp(mode_str, "sparse") == 0) {
        gen_sparse(data_orig, nrows, seed);
    } else if (strcmp(mode_str, "ds4_realistic") == 0) {
        gen_ds4_realistic(data_orig, nrows, seed);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode_str);
        print_usage(argv[0]);
        return 1;
    }

    /* Allocate compressed buffer */
    void *compressed = malloc((size_t)nrows * sizeof(ds4_row_planar3));
    if (!compressed) {
        fprintf(stderr, "Failed to allocate compressed buffer\n");
        return 1;
    }

    /* Quantize all rows */
    ds4_planar3_quantize(data_orig, compressed, (size_t)nrows, DIM);

    /* Dequantize all rows */
    ds4_planar3_dequantize(compressed, data_recon, (size_t)nrows, DIM);

    /* ---- Per-row metrics ---- */
    float *cosine_arr   = (float *)malloc(nrows * sizeof(float));
    float *mse_arr      = (float *)malloc(nrows * sizeof(float));
    float *maxerr_arr   = (float *)malloc(nrows * sizeof(float));
    float *relnorm_arr  = (float *)malloc(nrows * sizeof(float));

    for (int r = 0; r < nrows; r++) {
        float *orig = data_orig  + r * DIM;
        float *rec  = data_recon + r * DIM;

        /* Cosine similarity — API takes non-const second ptr */
        cosine_arr[r] = ds4_planar3_roundtrip_cosine(orig, rec, DIM);

        /* MSE per element */
        mse_arr[r] = ds4_planar3_roundtrip_mse(orig, rec, DIM);

        /* Max element error */
        float mx = 0.0f;
        for (int j = 0; j < DIM; j++) {
            float d = fabsf(orig[j] - rec[j]);
            if (d > mx) mx = d;
        }
        maxerr_arr[r] = mx;

        /* Relative norm error */
        float n1 = 0.0f, n2 = 0.0f;
        for (int j = 0; j < DIM; j++) {
            n1 += orig[j] * orig[j];
            n2 += rec[j]  * rec[j];
        }
        n1 = sqrtf(n1);
        n2 = sqrtf(n2);
        relnorm_arr[r] = (n1 > 1e-10f) ? fabsf(n1 - n2) / n1 : 0.0f;
    }

    /* Compute means before sorting */
    double cos_mean = 0.0;
    double mse_mean = 0.0;
    double maxerr_mean = 0.0;
    double relnorm_mean = 0.0;
    for (int r = 0; r < nrows; r++) {
        cos_mean     += cosine_arr[r];
        mse_mean     += mse_arr[r];
        maxerr_mean  += maxerr_arr[r];
        relnorm_mean += relnorm_arr[r];
    }
    cos_mean     /= nrows;
    mse_mean     /= nrows;
    maxerr_mean  /= nrows;
    relnorm_mean /= nrows;

    /* Sort for median / P99 */
    qsort(cosine_arr,  nrows, sizeof(float), cmp_float);
    qsort(mse_arr,     nrows, sizeof(float), cmp_float);
    qsort(maxerr_arr,  nrows, sizeof(float), cmp_float);
    qsort(relnorm_arr, nrows, sizeof(float), cmp_float);

    /* ---- Attention score drift ---- */
    /* Generate a random query vector */
    float query[512];
    pcg_state_ = seed ^ 0xDEADBEEFULL;
    for (int j = 0; j < DIM; j++)
        query[j] = randf_normal();

    float *score_orig  = (float *)malloc(nrows * sizeof(float));
    float *score_recon = (float *)malloc(nrows * sizeof(float));

    for (int r = 0; r < nrows; r++) {
        const float *orig = data_orig  + r * DIM;
        const float *rec  = data_recon + r * DIM;
        float so = 0.0f, sr = 0.0f;
        for (int j = 0; j < DIM; j++) {
            so += query[j] * orig[j];
            sr += query[j] * rec[j];
        }
        score_orig[r]  = so;
        score_recon[r] = sr;
    }

    /* Pearson correlation */
    double mean_so = 0.0, mean_sr = 0.0;
    for (int r = 0; r < nrows; r++) {
        mean_so += score_orig[r];
        mean_sr += score_recon[r];
    }
    mean_so /= nrows;
    mean_sr /= nrows;

    double cov = 0.0, var_o = 0.0, var_r = 0.0;
    for (int r = 0; r < nrows; r++) {
        double dso = score_orig[r]  - mean_so;
        double dsr = score_recon[r] - mean_sr;
        cov   += dso * dsr;
        var_o += dso * dso;
        var_r += dsr * dsr;
    }
    double p_denom = sqrt(var_o * var_r);
    double pearson = (p_denom > 1e-12) ? cov / p_denom : 0.0;

    /* Max absolute score difference */
    float max_score_diff = 0.0f;
    for (int r = 0; r < nrows; r++) {
        float d = fabsf(score_orig[r] - score_recon[r]);
        if (d > max_score_diff) max_score_diff = d;
    }

    /* Top-1 agreement */
    int top1_orig  = 0;
    int top1_recon = 0;
    for (int r = 1; r < nrows; r++) {
        if (score_orig[r]  > score_orig[top1_orig])  top1_orig  = r;
        if (score_recon[r] > score_recon[top1_recon]) top1_recon = r;
    }
    int top1_agree = (top1_orig == top1_recon) ? 1 : 0;

    /* ---- Print report ---- */
    printf("\n=== Planar3 Quality Evaluation ===\n");
    printf("Mode: %s | Rows: %d | Dim: %d | Seed: %llu\n",
           mode_str, nrows, DIM, (unsigned long long)seed);

    printf("\nCosine similarity:\n");
    printf("  min=%.4f mean=%.4f median=%.4f max=%.4f p99=%.4f\n",
           cosine_arr[0], (float)cos_mean,
           percentile_sorted(cosine_arr, nrows, 0.5f),
           cosine_arr[nrows - 1],
           percentile_sorted(cosine_arr, nrows, 0.99f));

    printf("\nMSE (per element):\n");
    printf("  min=%.3e mean=%.3e median=%.3e max=%.3e p99=%.3e\n",
           mse_arr[0], (float)mse_mean,
           percentile_sorted(mse_arr, nrows, 0.5f),
           mse_arr[nrows - 1],
           percentile_sorted(mse_arr, nrows, 0.99f));

    printf("\nMax element error:\n");
    printf("  min=%.4f mean=%.4f median=%.4f max=%.4f\n",
           maxerr_arr[0], (float)maxerr_mean,
           percentile_sorted(maxerr_arr, nrows, 0.5f),
           maxerr_arr[nrows - 1]);

    printf("\nRelative norm error:\n");
    printf("  min=%.3e mean=%.3e max=%.3e\n",
           relnorm_arr[0], (float)relnorm_mean,
           relnorm_arr[nrows - 1]);

    printf("\nAttention score drift (single random query):\n");
    printf("  score_corr=%.4f max_diff=%.4f top1_agree=%s\n",
           (float)pearson, max_score_diff,
           top1_agree ? "yes" : "no");

    size_t compressed_bytes = sizeof(ds4_row_planar3); /* per row */
    size_t fp16_bytes = (size_t)DIM * 2; /* per row */
    double ratio = (double)fp16_bytes / (double)compressed_bytes;

    printf("\nCompression: %zu bytes/row (%.2fx vs FP16 %zu bytes)\n",
           compressed_bytes, ratio, fp16_bytes);

    /* Cleanup */
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
