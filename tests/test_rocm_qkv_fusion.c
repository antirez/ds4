/* Correctness coverage for the ROCm DeepSeek V4 fused Q/KV RMSNorm + KV RoPE.
 *
 * The reference path deliberately uses the existing two launches:
 * ds4_gpu_dsv4_qkv_rms_norm_rows_tensor() followed by
 * ds4_gpu_rope_tail_tensor().  The fused result must be bit-identical for both
 * Q and KV, including multi-row, inverse, and YaRN-style compressed settings.
 */

#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

enum {
    Q_N = 1024u,
    /* Large enough to cover the fused kernel's 256-pair boundary and the
     * public entry point's greater-than-256-pair host fallback. */
    KV_N = 768u,
    MAX_ROWS = 7u,
};

typedef struct {
    const char *name;
    uint32_t rows;
    uint32_t kv_n_head;
    uint32_t kv_head_dim;
    uint32_t n_rot;
    uint32_t pos0;
    uint32_t n_ctx_orig;
    bool inverse;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
} fusion_case;

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static uint32_t mix32(uint32_t x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

static uint32_t float_bits(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int compare_exact(const char *case_name,
                         const char *tensor_name,
                         const float *reference,
                         const float *fused,
                         size_t count) {
    enum { MAX_REPORTED_MISMATCHES = 8 };
    size_t mismatches = 0;
    float max_abs = 0.0f;
    size_t first = 0;
    for (size_t i = 0; i < count; i++) {
        if (float_bits(reference[i]) == float_bits(fused[i])) continue;
        const float diff = fabsf(reference[i] - fused[i]);
        if (mismatches == 0u) first = i;
        if (diff > max_abs) max_abs = diff;
        mismatches++;
    }
    if (mismatches != 0u) {
        fprintf(stderr,
                "ROCm QKV fusion %s %s mismatch: %zu/%zu, first=%zu "
                "ref=%a (0x%08x) fused=%a (0x%08x), max_abs=%g\n",
                case_name,
                tensor_name,
                mismatches,
                count,
                first,
                reference[first],
                float_bits(reference[first]),
                fused[first],
                float_bits(fused[first]),
                max_abs);
        size_t reported = 0;
        for (size_t i = 0;
             i < count && reported < MAX_REPORTED_MISMATCHES;
             i++) {
            if (float_bits(reference[i]) == float_bits(fused[i])) continue;
            fprintf(stderr,
                    "  mismatch[%zu]: index=%zu ref=%a (0x%08x) "
                    "fused=%a (0x%08x)\n",
                    reported,
                    i,
                    reference[i],
                    float_bits(reference[i]),
                    fused[i],
                    float_bits(fused[i]));
            reported++;
        }
        return 0;
    }
    return 1;
}

static void fill_inputs(float *q,
                        float *kv,
                        uint32_t rows,
                        uint32_t case_index) {
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t i = 0; i < Q_N; i++) {
            const uint32_t h = mix32(i + 1u + row * 0x9e3779b9u +
                                     case_index * 0x85ebca6bu);
            const int32_t centered = (int32_t)(h % 4093u) - 2046;
            q[(uint64_t)row * Q_N + i] = (float)centered / 1024.0f;
        }
        for (uint32_t i = 0; i < KV_N; i++) {
            const uint32_t h = mix32(i + 17u + row * 0xc2b2ae35u +
                                     case_index * 0x27d4eb2du);
            const int32_t centered = (int32_t)(h % 2039u) - 1019;
            kv[(uint64_t)row * KV_N + i] = (float)centered / 768.0f;
        }

        /* Deterministic nonuniform magnitudes exercise every reduction row. */
        q[(uint64_t)row * Q_N + (13u + row * 97u) % Q_N] =
            (row & 1u) ? -7.25f : 7.5f;
        kv[(uint64_t)row * KV_N + (29u + row * 53u) % KV_N] =
            (row & 1u) ? 5.75f : -6.0f;
    }
}

static int run_case(const fusion_case *tc,
                    uint32_t case_index,
                    const void *model,
                    uint64_t model_size,
                    uint64_t q_weight_offset,
                    uint64_t kv_weight_offset) {
    const uint64_t q_count = (uint64_t)tc->rows * Q_N;
    const uint64_t kv_count = (uint64_t)tc->rows * KV_N;
    const uint64_t q_bytes = q_count * sizeof(float);
    const uint64_t kv_bytes = kv_count * sizeof(float);
    float *q_host = (float *)malloc((size_t)q_bytes);
    float *kv_host = (float *)malloc((size_t)kv_bytes);
    float *q_reference = (float *)malloc((size_t)q_bytes);
    float *q_fused = (float *)malloc((size_t)q_bytes);
    float *kv_reference = (float *)malloc((size_t)kv_bytes);
    float *kv_fused = (float *)malloc((size_t)kv_bytes);
    ds4_gpu_tensor *q = NULL;
    ds4_gpu_tensor *kv = NULL;
    ds4_gpu_tensor *q_ref_tensor = NULL;
    ds4_gpu_tensor *kv_ref_tensor = NULL;
    ds4_gpu_tensor *q_fused_tensor = NULL;
    ds4_gpu_tensor *kv_fused_tensor = NULL;
    int ok = q_host && kv_host && q_reference && q_fused &&
             kv_reference && kv_fused;

    if (ok) {
        fill_inputs(q_host, kv_host, tc->rows, case_index);
        q = ds4_gpu_tensor_alloc(q_bytes);
        kv = ds4_gpu_tensor_alloc(kv_bytes);
        q_ref_tensor = ds4_gpu_tensor_alloc(q_bytes);
        kv_ref_tensor = ds4_gpu_tensor_alloc(kv_bytes);
        q_fused_tensor = ds4_gpu_tensor_alloc(q_bytes);
        kv_fused_tensor = ds4_gpu_tensor_alloc(kv_bytes);
        ok = q && kv && q_ref_tensor && kv_ref_tensor &&
             q_fused_tensor && kv_fused_tensor;
    }
    if (ok) {
        ok = ds4_gpu_tensor_write(q, 0u, q_host, q_bytes) != 0 &&
             ds4_gpu_tensor_write(kv, 0u, kv_host, kv_bytes) != 0;
    }

    int begun = 0;
    if (ok) {
        begun = ds4_gpu_begin_commands();
        ok = begun != 0;
        if (ok) {
            ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
                     q_ref_tensor, q, model, model_size,
                     q_weight_offset, Q_N,
                     kv_ref_tensor, kv, kv_weight_offset, KV_N,
                     tc->rows, 1.0e-6f) != 0;
        }
        if (ok && tc->n_rot != 0u) {
            ok = ds4_gpu_rope_tail_tensor(
                     kv_ref_tensor,
                     tc->rows,
                     tc->kv_n_head,
                     tc->kv_head_dim,
                     tc->n_rot,
                     tc->pos0,
                     tc->n_ctx_orig,
                     tc->inverse,
                     tc->freq_base,
                     tc->freq_scale,
                     tc->ext_factor,
                     tc->attn_factor,
                     tc->beta_fast,
                     tc->beta_slow) != 0;
        }
        if (begun && ds4_gpu_end_commands() == 0) ok = 0;
    }

    begun = 0;
    if (ok) {
        begun = ds4_gpu_begin_commands();
        ok = begun != 0;
        if (ok) {
            ok = ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
                     q_fused_tensor, q, model, model_size,
                     q_weight_offset, Q_N,
                     kv_fused_tensor, kv, kv_weight_offset, KV_N,
                     tc->rows,
                     tc->kv_n_head,
                     tc->kv_head_dim,
                     tc->n_rot,
                     tc->pos0,
                     tc->n_ctx_orig,
                     tc->inverse,
                     tc->freq_base,
                     tc->freq_scale,
                     tc->ext_factor,
                     tc->attn_factor,
                     tc->beta_fast,
                     tc->beta_slow,
                     1.0e-6f) != 0;
        }
        if (begun && ds4_gpu_end_commands() == 0) ok = 0;
    }

    if (ok) {
        ok = ds4_gpu_tensor_read(
                 q_ref_tensor, 0u, q_reference, q_bytes) != 0 &&
             ds4_gpu_tensor_read(
                 q_fused_tensor, 0u, q_fused, q_bytes) != 0 &&
             ds4_gpu_tensor_read(
                 kv_ref_tensor, 0u, kv_reference, kv_bytes) != 0 &&
             ds4_gpu_tensor_read(
                 kv_fused_tensor, 0u, kv_fused, kv_bytes) != 0;
    }
    if (ok) {
        const int q_ok = compare_exact(
            tc->name, "Q", q_reference, q_fused, (size_t)q_count);
        const int kv_ok = compare_exact(
            tc->name, "KV", kv_reference, kv_fused, (size_t)kv_count);
        ok = q_ok && kv_ok;
    }

    fprintf(stderr,
            "ROCm QKV fusion %-22s rows=%u heads=%u head_dim=%u "
            "rot=%u pos=%u ctx_orig=%u inverse=%u base=%a scale=%a "
            "ext=%a attn=%a beta_fast=%a beta_slow=%a: %s\n",
            tc->name,
            tc->rows,
            tc->kv_n_head,
            tc->kv_head_dim,
            tc->n_rot,
            tc->pos0,
            tc->n_ctx_orig,
            tc->inverse ? 1u : 0u,
            tc->freq_base,
            tc->freq_scale,
            tc->ext_factor,
            tc->attn_factor,
            tc->beta_fast,
            tc->beta_slow,
            ok ? "PASS" : "FAIL");

    ds4_gpu_tensor_free(kv_fused_tensor);
    ds4_gpu_tensor_free(q_fused_tensor);
    ds4_gpu_tensor_free(kv_ref_tensor);
    ds4_gpu_tensor_free(q_ref_tensor);
    ds4_gpu_tensor_free(kv);
    ds4_gpu_tensor_free(q);
    free(kv_fused);
    free(kv_reference);
    free(q_fused);
    free(q_reference);
    free(kv_host);
    free(q_host);
    return ok;
}

static int run_invalid_shape_cases(const void *model,
                                   uint64_t model_size,
                                   uint64_t q_weight_offset,
                                   uint64_t kv_weight_offset) {
    const uint32_t rows = 2u;
    const uint64_t q_bytes = (uint64_t)rows * Q_N * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)rows * KV_N * sizeof(float);
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(kv_bytes);
    ds4_gpu_tensor *q_out = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *kv_out = ds4_gpu_tensor_alloc(kv_bytes);
    ds4_gpu_tensor *short_q_out = ds4_gpu_tensor_alloc(q_bytes - sizeof(float));
    int ok = q && kv && q_out && kv_out && short_q_out;

#define EXPECT_REJECT(label, call) do { \
        const int accepted = (call); \
        if (accepted) { \
            fprintf(stderr, "ROCm QKV fusion invalid case accepted: %s\n", label); \
            ok = 0; \
        } \
    } while (0)

    if (ok) {
        EXPECT_REJECT("KV head product mismatch",
            ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
                q_out, q, model, model_size, q_weight_offset, Q_N,
                kv_out, kv, kv_weight_offset, KV_N, rows,
                3u, 170u, 64u, 17u, 0u, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f));
        EXPECT_REJECT("odd rotation width",
            ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
                q_out, q, model, model_size, q_weight_offset, Q_N,
                kv_out, kv, kv_weight_offset, KV_N, rows,
                1u, KV_N, 63u, 17u, 0u, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f));
        EXPECT_REJECT("rotation wider than head",
            ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
                q_out, q, model, model_size, q_weight_offset, Q_N,
                kv_out, kv, kv_weight_offset, KV_N, rows,
                1u, KV_N, KV_N + 2u, 17u, 0u, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f));
        EXPECT_REJECT("zero head count",
            ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
                q_out, q, model, model_size, q_weight_offset, Q_N,
                kv_out, kv, kv_weight_offset, KV_N, rows,
                0u, KV_N, 64u, 17u, 0u, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f));
        EXPECT_REJECT("zero Q width",
            ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
                q_out, q, model, model_size, q_weight_offset, 0u,
                kv_out, kv, kv_weight_offset, KV_N, rows,
                1u, KV_N, 64u, 17u, 0u, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f));
        EXPECT_REJECT("undersized Q output",
            ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
                short_q_out, q, model, model_size, q_weight_offset, Q_N,
                kv_out, kv, kv_weight_offset, KV_N, rows,
                1u, KV_N, 64u, 17u, 0u, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f));
        EXPECT_REJECT("Q weight range overflow",
            ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
                q_out, q, model, model_size, model_size - sizeof(float), Q_N,
                kv_out, kv, kv_weight_offset, KV_N, rows,
                1u, KV_N, 64u, 17u, 0u, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f));

        const int zero_rows_ok =
            ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
                q_out, q, model, model_size, q_weight_offset, Q_N,
                kv_out, kv, kv_weight_offset, KV_N, 0u,
                1u, KV_N, 64u, 17u, 0u, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
        if (!zero_rows_ok) {
            fprintf(stderr, "ROCm QKV fusion rejected valid zero-row no-op\n");
            ok = 0;
        }
    }

#undef EXPECT_REJECT

    /* Drain any launch if a rejection regresses before releasing tensors. */
    if (ds4_gpu_end_commands() == 0) ok = 0;
    ds4_gpu_tensor_free(short_q_out);
    ds4_gpu_tensor_free(kv_out);
    ds4_gpu_tensor_free(q_out);
    ds4_gpu_tensor_free(kv);
    ds4_gpu_tensor_free(q);
    fprintf(stderr, "ROCm QKV fusion invalid shapes: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

int main(void) {
    const float compressed_scale = 1.0f / 16.0f;
    const float compressed_attn =
        1.0f / (1.0f + 0.1f * logf(1.0f / compressed_scale));
    const fusion_case cases[] = {
        { "dense-single", 1u, 1u, KV_N, 64u, 17u, 0u, false,
          10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f },
        { "dense-inverse-multi", 4u, 2u, KV_N / 2u, 64u, 4093u, 0u, true,
          10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f },
        { "dense-high-position", 2u, 1u, KV_N, 64u, 70000u, 0u, false,
          10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f },
        { "scaled-no-yarn", 2u, 1u, KV_N, 64u, 70000u, 0u, false,
          10000.0f, compressed_scale, 0.0f, 1.0f, 32.0f, 1.0f },
        { "ext-unity-scale", 2u, 1u, KV_N, 64u, 70000u, 65536u, false,
          10000.0f, 1.0f, 1.0f, 1.0f, 32.0f, 1.0f },
        /* corr0 is clamped to zero, so pair zero's ramp is necessarily one.
         * Oversized equal betas make corr1 negative and every later pair's
         * ramp exactly zero, the closest realizable uniform-zero control. */
        { "yarn-zero-ramp", 2u, 1u, KV_N, 64u, 70000u, 65536u, false,
          10000.0f, compressed_scale, 1.0f, compressed_attn,
          65536.0f, 65536.0f },
        { "yarn-low-position", 2u, 1u, KV_N, 64u, 17u, 65536u, false,
          10000.0f, compressed_scale, 1.0f, compressed_attn, 32.0f, 1.0f },
        { "yarn-multi", 3u, 1u, KV_N, 64u, 65520u, 65536u, false,
          10000.0f, compressed_scale, 1.0f, compressed_attn, 32.0f, 1.0f },
        { "yarn-inverse-multi", 7u, 2u, KV_N / 2u, 64u, 70000u, 65536u, true,
          10000.0f, compressed_scale, 1.0f, compressed_attn, 32.0f, 1.0f },
        /* One 512-wide rotated tail is exactly 256 pairs, so this case must
         * remain on the fused launch.  The following 514-wide tail is 257
         * pairs and therefore exercises the two-launch host fallback. */
        { "rope-pairs-256-boundary", 3u, 1u, KV_N, 512u, 65520u, 65536u, false,
          10000.0f, compressed_scale, 1.0f, compressed_attn, 32.0f, 1.0f },
        { "rope-pairs-257-fallback", 3u, 1u, KV_N, 514u, 65520u, 65536u, false,
          10000.0f, compressed_scale, 1.0f, compressed_attn, 32.0f, 1.0f },
        { "norm-only", 2u, 4u, KV_N / 4u, 0u, 0u, 0u, false,
          10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f },
    };
    const uint64_t q_weight_bytes = (uint64_t)Q_N * sizeof(float);
    const uint64_t kv_weight_bytes = (uint64_t)KV_N * sizeof(float);
    const uint64_t q_weight_offset = 0u;
    const uint64_t kv_weight_offset = align_up(q_weight_bytes, 4096u);
    const uint64_t model_size =
        align_up(kv_weight_offset + kv_weight_bytes, 4096u);
    FILE *model_file = NULL;
    void *model = MAP_FAILED;
    int initialized = 0;
    int ok = 1;

    model_file = tmpfile();
    if (model_file &&
        ftruncate(fileno(model_file), (off_t)model_size) == 0) {
        model = mmap(NULL, (size_t)model_size, PROT_READ | PROT_WRITE,
                     MAP_SHARED, fileno(model_file), 0);
    }
    if (!model_file || model == MAP_FAILED) {
        fprintf(stderr, "ROCm QKV fusion model-map allocation failed\n");
        ok = 0;
        goto cleanup;
    }
    memset(model, 0, (size_t)model_size);
    float *q_weight = (float *)((uint8_t *)model + q_weight_offset);
    float *kv_weight = (float *)((uint8_t *)model + kv_weight_offset);
    for (uint32_t i = 0; i < Q_N; i++) {
        q_weight[i] = 0.75f + (float)((int32_t)(i % 29u) - 14) / 128.0f;
    }
    for (uint32_t i = 0; i < KV_N; i++) {
        kv_weight[i] = 0.875f + (float)((int32_t)(i % 23u) - 11) / 96.0f;
    }
    if (msync(model, (size_t)model_size, MS_SYNC) != 0) {
        fprintf(stderr, "ROCm QKV fusion model-map sync failed\n");
        ok = 0;
        goto cleanup;
    }

    ok = ds4_gpu_init();
    initialized = ok;
    if (!ok) {
        fprintf(stderr, "ROCm QKV fusion ds4_gpu_init failed\n");
        goto cleanup;
    }
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    const uint64_t model_offsets[] = {
        q_weight_offset, kv_weight_offset,
    };
    const uint64_t model_sizes[] = {
        q_weight_bytes, kv_weight_bytes,
    };
    ok = ds4_gpu_set_model_map(model, model_size) &&
         ds4_gpu_set_model_fd(fileno(model_file)) &&
         ds4_gpu_set_model_map_spans(
             model, model_size, model_offsets, model_sizes,
             sizeof(model_offsets) / sizeof(model_offsets[0]),
             q_weight_bytes);
    if (!ok) {
        fprintf(stderr, "ROCm QKV fusion model cache setup failed\n");
        goto cleanup;
    }

    for (uint32_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (!run_case(&cases[i], i, model, model_size,
                      q_weight_offset, kv_weight_offset)) {
            ok = 0;
        }
    }
    if (!run_invalid_shape_cases(model, model_size,
                                 q_weight_offset, kv_weight_offset)) {
        ok = 0;
    }

cleanup:
    if (initialized) {
        ds4_gpu_set_model_fd(-1);
        ds4_gpu_cleanup();
    }
    if (model != MAP_FAILED) munmap(model, (size_t)model_size);
    if (model_file) fclose(model_file);
    fprintf(stderr, "ROCm fused QKV RMSNorm + KV RoPE: %s\n",
            ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
