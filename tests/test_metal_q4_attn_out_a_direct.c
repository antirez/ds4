#define _DARWIN_C_SOURCE

/* Production-shape, GGUF-free oracle for the pre-M5 Metal Q4_K attention
 * output-A fixed-route specialization.  The established routed kernel is the
 * bitwise baseline; output-B deliberately stays small to keep this focused
 * test's resident memory footprint prudent. */

#include "ds4_gpu.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

#ifdef __APPLE__

enum {
    Q4_K_TYPE = 12u,
    Q8_0_TYPE = 8u,
    QK_K = 256u,
    QK8_0 = 32u,
    GROUP_DIM = 4096u,
    RANK = 1024u,
    N_GROUPS = 8u,
    LOW_DIM = N_GROUPS * RANK,
    OUT_DIM = 64u,
    ACTIVE_ROWS = 513u,
    REJECT_ROWS = 511u,
    ALLOC_ROWS = ACTIVE_ROWS + 1u,
    GUARD_ELEMENTS = 64u,
};

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2u];
} block_q4_K;

typedef struct {
    uint16_t d;
    int8_t qs[QK8_0];
} block_q8_0;

static const char *k_disable =
    "DS4_METAL_DISABLE_Q4_ATTN_OUT_A_DIRECT";
static const char *k_require =
    "DS4_METAL_REQUIRE_Q4_ATTN_OUT_A_DIRECT";
static const char *k_disable_f16_rhs =
    "DS4_METAL_DISABLE_Q4_ATTN_OUT_B_F16_RHS";
static const char *k_require_f16_rhs =
    "DS4_METAL_REQUIRE_Q4_ATTN_OUT_B_F16_RHS";

static void fail(const char *what) {
    fprintf(stderr, "Metal Q4 output-A direct oracle FAIL: %s\n", what);
    exit(1);
}

#define CHECK(expr, what) do { if (!(expr)) fail(what); } while (0)

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static uint64_t hash_bytes(const void *raw, uint64_t bytes) {
    const uint8_t *p = raw;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (uint64_t i = 0; i < bytes; i++) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void pack_scales(uint8_t packed[12],
                        const uint8_t scale[8],
                        const uint8_t minimum[8]) {
    memset(packed, 0, 12u);
    for (uint32_t group = 0; group < 4u; group++) {
        packed[group] = scale[group] & 63u;
        packed[group + 4u] = minimum[group] & 63u;
    }
    for (uint32_t group = 4u; group < 8u; group++) {
        packed[group + 4u] = (scale[group] & 15u) |
                             ((minimum[group] & 15u) << 4u);
        packed[group - 4u] |= (scale[group] >> 4u) << 6u;
        packed[group] |= (minimum[group] >> 4u) << 6u;
    }
}

static void fill_q4_matrix(void *raw, uint32_t in_dim,
                           uint32_t rows, uint32_t salt) {
    CHECK(sizeof(block_q4_K) == 144u, "unexpected Q4_K block size");
    CHECK((in_dim % QK_K) == 0u, "unaligned Q4_K fixture");
    const uint32_t blocks_per_row = in_dim / QK_K;
    block_q4_K *matrix = raw;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t block = 0; block < blocks_per_row; block++) {
            block_q4_K *b = matrix +
                (uint64_t)row * blocks_per_row + block;
            const uint32_t key = salt + row * 1009u + block * 313u +
                                 (row ^ (block * 17u));
            uint8_t scale[8];
            uint8_t minimum[8];
            for (uint32_t group = 0; group < 8u; group++) {
                scale[group] = (uint8_t)((key + group * 7u) % 64u);
                minimum[group] =
                    (uint8_t)((key / 3u + group * 5u) % 64u);
            }
            pack_scales(b->scales, scale, minimum);
            for (uint32_t i = 0; i < QK_K / 2u; i++) {
                b->qs[i] =
                    (uint8_t)(key + i * 37u + (i >> 2u) * 11u);
            }
            /* Exact binary half scales keep the fixture deterministic. */
            b->d = 0x2400u;
            b->dmin = 0x1c00u;
        }
    }
}

static void fill_q8_matrix(void *raw, uint32_t in_dim,
                           uint32_t rows, uint32_t salt) {
    CHECK(sizeof(block_q8_0) == 34u, "unexpected Q8_0 block size");
    CHECK((in_dim % QK8_0) == 0u, "unaligned Q8_0 fixture");
    const uint32_t blocks_per_row = in_dim / QK8_0;
    block_q8_0 *matrix = raw;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t block = 0; block < blocks_per_row; block++) {
            block_q8_0 *b = matrix +
                (uint64_t)row * blocks_per_row + block;
            const uint32_t key = salt + row * 977u + block * 101u;
            b->d = 0x2000u;
            for (uint32_t i = 0; i < QK8_0; i++) {
                b->qs[i] = (int8_t)((int)((key + i * 29u) % 127u) - 63);
            }
        }
    }
}

static void fill_heads(float *heads) {
    for (uint32_t row = 0; row < ALLOC_ROWS; row++) {
        for (uint32_t i = 0; i < N_GROUPS * GROUP_DIM; i++) {
            const uint32_t key =
                i * 41u + row * 271u + ((i >> 2u) ^ (row * 19u));
            heads[(uint64_t)row * N_GROUPS * GROUP_DIM + i] =
                (float)((int)(key % 255u) - 127) / 512.0f;
        }
    }
}

static void poison(float *values, uint64_t count, uint32_t base) {
    for (uint64_t i = 0; i < count; i++) {
        const uint32_t bits = base + (uint32_t)(i & 0xffffu);
        memcpy(&values[i], &bits, sizeof(bits));
    }
}

static uint64_t count_poison_mismatches(const float *actual,
                                        uint64_t begin,
                                        uint64_t end,
                                        uint32_t base) {
    uint64_t mismatches = 0;
    for (uint64_t i = begin; i < end; i++) {
        uint32_t bits = 0;
        memcpy(&bits, &actual[i], sizeof(bits));
        if (bits != base + (uint32_t)(i & 0xffffu)) mismatches++;
    }
    return mismatches;
}

static uint64_t count_bit_mismatches(const float *reference,
                                     const float *actual,
                                     uint64_t count,
                                     uint64_t *first) {
    uint64_t mismatches = 0;
    *first = UINT64_MAX;
    for (uint64_t i = 0; i < count; i++) {
        if (memcmp(&reference[i], &actual[i], sizeof(float)) != 0) {
            if (*first == UINT64_MAX) *first = i;
            mismatches++;
        }
    }
    return mismatches;
}

static void upload_poison(ds4_gpu_tensor *low,
                          ds4_gpu_tensor *out,
                          float *low_host,
                          float *out_host,
                          uint64_t low_count,
                          uint64_t out_count,
                          uint32_t low_poison,
                          uint32_t out_poison) {
    poison(low_host, low_count, low_poison);
    poison(out_host, out_count, out_poison);
    CHECK(ds4_gpu_tensor_write(low, 0, low_host,
                               low_count * sizeof(float)) != 0,
          "low poison upload");
    CHECK(ds4_gpu_tensor_write(out, 0, out_host,
                               out_count * sizeof(float)) != 0,
          "out poison upload");
}

static void read_outputs(ds4_gpu_tensor *low,
                         ds4_gpu_tensor *out,
                         float *low_host,
                         float *out_host,
                         uint64_t low_count,
                         uint64_t out_count) {
    CHECK(ds4_gpu_tensor_read(low, 0, low_host,
                              low_count * sizeof(float)) != 0,
          "low read");
    CHECK(ds4_gpu_tensor_read(out, 0, out_host,
                              out_count * sizeof(float)) != 0,
          "out read");
}

static void check_inputs_immutable(ds4_gpu_tensor *heads,
                                   float *heads_host,
                                   uint64_t heads_bytes,
                                   uint64_t heads_hash,
                                   const void *model,
                                   uint64_t model_bytes,
                                   uint64_t model_hash) {
    CHECK(ds4_gpu_tensor_read(heads, 0, heads_host, heads_bytes) != 0,
          "heads immutability read");
    CHECK(hash_bytes(heads_host, heads_bytes) == heads_hash,
          "heads were modified");
    CHECK(hash_bytes(model, model_bytes) == model_hash,
          "model weights were modified");
}

int main(void) {
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t row_a_bytes =
        (GROUP_DIM / QK_K) * sizeof(block_q4_K);
    const uint64_t out_a_bytes = (uint64_t)LOW_DIM * row_a_bytes;
    const uint64_t out_b_offset = align_up(out_a_bytes, page);
    const uint64_t row_b_bytes =
        (LOW_DIM / QK8_0) * sizeof(block_q8_0);
    const uint64_t out_b_bytes = (uint64_t)OUT_DIM * row_b_bytes;
    const uint64_t model_bytes =
        align_up(out_b_offset + out_b_bytes, page);
    const uint64_t heads_payload_count =
        (uint64_t)ALLOC_ROWS * N_GROUPS * GROUP_DIM;
    const uint64_t low_payload_count = (uint64_t)ALLOC_ROWS * LOW_DIM;
    const uint64_t out_payload_count = (uint64_t)ALLOC_ROWS * OUT_DIM;
    const uint64_t heads_storage_count =
        GUARD_ELEMENTS + heads_payload_count + GUARD_ELEMENTS;
    const uint64_t low_storage_count =
        GUARD_ELEMENTS + low_payload_count + GUARD_ELEMENTS;
    const uint64_t out_storage_count =
        GUARD_ELEMENTS + out_payload_count + GUARD_ELEMENTS;
    const uint64_t active_low_count = (uint64_t)ACTIVE_ROWS * LOW_DIM;
    const uint64_t active_out_count = (uint64_t)ACTIVE_ROWS * OUT_DIM;
    const uint64_t heads_payload_bytes =
        heads_payload_count * sizeof(float);
    const uint64_t heads_storage_bytes =
        heads_storage_count * sizeof(float);

    CHECK(unsetenv(k_disable) == 0, "clear direct disable env");
    CHECK(unsetenv(k_require) == 0, "clear direct require env");
    CHECK(unsetenv(k_disable_f16_rhs) == 0,
          "clear output-B F16 disable env");
    CHECK(unsetenv(k_require_f16_rhs) == 0,
          "clear output-B F16 require env");

    CHECK(ds4_gpu_init() != 0, "Metal init");
    if (!ds4_gpu_device_is_pre_m5_apple_silicon()) {
        fprintf(stderr,
                "Metal Q4 output-A direct oracle SKIP: requires Apple M1--M4\n");
        ds4_gpu_cleanup();
        return 0;
    }

    void *model = NULL;
    CHECK(posix_memalign(&model, (size_t)page, (size_t)model_bytes) == 0,
          "model allocation");
    memset(model, 0, (size_t)model_bytes);
    fill_q4_matrix(model, GROUP_DIM, LOW_DIM, 211u);
    fill_q8_matrix((uint8_t *)model + out_b_offset,
                   LOW_DIM, OUT_DIM, 307u);
    const uint64_t model_hash = hash_bytes(model, model_bytes);

    float *heads_host = malloc((size_t)heads_storage_bytes);
    float *low_host = malloc((size_t)low_storage_count * sizeof(float));
    float *out_host = malloc((size_t)out_storage_count * sizeof(float));
    float *baseline_low = malloc((size_t)active_low_count * sizeof(float));
    float *baseline_out = malloc((size_t)active_out_count * sizeof(float));
    CHECK(heads_host && low_host && out_host && baseline_low && baseline_out,
          "host tensor allocation");
    poison(heads_host, heads_storage_count, 0x7fc00000u);
    fill_heads(heads_host + GUARD_ELEMENTS);
    const uint64_t heads_hash =
        hash_bytes(heads_host, heads_storage_bytes);

    ds4_gpu_tensor *heads_base =
        ds4_gpu_tensor_alloc(heads_storage_bytes);
    ds4_gpu_tensor *low_base =
        ds4_gpu_tensor_alloc(low_storage_count * sizeof(float));
    ds4_gpu_tensor *out_base =
        ds4_gpu_tensor_alloc(out_storage_count * sizeof(float));
    CHECK(heads_base && low_base && out_base,
          "guarded Metal base allocation");
    ds4_gpu_tensor *heads = ds4_gpu_tensor_view(
        heads_base, GUARD_ELEMENTS * sizeof(float), heads_payload_bytes);
    ds4_gpu_tensor *low = ds4_gpu_tensor_view(
        low_base, GUARD_ELEMENTS * sizeof(float),
        low_payload_count * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_view(
        out_base, GUARD_ELEMENTS * sizeof(float),
        out_payload_count * sizeof(float));
    CHECK(heads && low && out, "guarded Metal tensor views");
    CHECK(ds4_gpu_tensor_write(heads_base, 0, heads_host,
                               heads_storage_bytes) != 0,
          "heads upload");

    /* Quality mode also disables the Metal4 cooperative path.  SSD mode is
     * intentional: this is the exact production dispatch policy under test. */
    ds4_gpu_set_quality(true);
    ds4_gpu_set_ssd_streaming(true);
    CHECK(ds4_gpu_set_model_map(model, model_bytes) != 0, "model map");

    const uint32_t low_poison = 0x7fc10000u;
    const uint32_t out_poison = 0x7fc20000u;
    upload_poison(low_base, out_base, low_host, out_host,
                  low_storage_count, out_storage_count,
                  low_poison, out_poison);
    CHECK(setenv(k_disable, "1", 1) == 0, "select routed baseline");
    CHECK(setenv(k_require, "1", 1) == 0,
          "require direct for kill-switch preflight");
    CHECK(ds4_gpu_attention_output_q4_K_batch_tensor(
              out, low, NULL, NULL, model, model_bytes,
              0, out_b_offset, Q8_0_TYPE, GROUP_DIM, RANK,
              N_GROUPS, OUT_DIM, heads, ACTIVE_ROWS) == -1,
          "direct disable must win over REQUIRE");
    read_outputs(low_base, out_base, low_host, out_host,
                 low_storage_count, out_storage_count);
    CHECK(count_poison_mismatches(low_host, 0, low_storage_count,
                                  low_poison) == 0,
          "direct kill-switch preflight modified low");
    CHECK(count_poison_mismatches(out_host, 0, out_storage_count,
                                  out_poison) == 0,
          "direct kill-switch preflight modified out");
    CHECK(unsetenv(k_require) == 0, "clear REQUIRE for baseline");
    CHECK(ds4_gpu_attention_output_q4_K_batch_tensor(
              out, low, NULL, NULL, model, model_bytes,
              0, out_b_offset, Q8_0_TYPE, GROUP_DIM, RANK,
              N_GROUPS, OUT_DIM, heads, ACTIVE_ROWS) == 1,
          "routed baseline dispatch");
    read_outputs(low_base, out_base, low_host, out_host,
                 low_storage_count, out_storage_count);
    CHECK(count_poison_mismatches(
              low_host, GUARD_ELEMENTS,
              GUARD_ELEMENTS + active_low_count, low_poison) ==
              active_low_count,
          "baseline did not overwrite every active low value");
    CHECK(count_poison_mismatches(
              out_host, GUARD_ELEMENTS,
              GUARD_ELEMENTS + active_out_count, out_poison) ==
              active_out_count,
          "baseline did not overwrite every active out value");
    CHECK(count_poison_mismatches(low_host, 0, GUARD_ELEMENTS,
                                  low_poison) == 0,
          "baseline low prefix canary");
    CHECK(count_poison_mismatches(
              low_host, GUARD_ELEMENTS + active_low_count,
              GUARD_ELEMENTS + low_payload_count,
              low_poison) == 0,
          "baseline low tail canary");
    CHECK(count_poison_mismatches(
              low_host, GUARD_ELEMENTS + low_payload_count,
              low_storage_count, low_poison) == 0,
          "baseline low suffix canary");
    CHECK(count_poison_mismatches(out_host, 0, GUARD_ELEMENTS,
                                  out_poison) == 0,
          "baseline out prefix canary");
    CHECK(count_poison_mismatches(
              out_host, GUARD_ELEMENTS + active_out_count,
              GUARD_ELEMENTS + out_payload_count,
              out_poison) == 0,
          "baseline out tail canary");
    CHECK(count_poison_mismatches(
              out_host, GUARD_ELEMENTS + out_payload_count,
              out_storage_count, out_poison) == 0,
          "baseline out suffix canary");
    memcpy(baseline_low, low_host + GUARD_ELEMENTS,
           (size_t)active_low_count * sizeof(float));
    memcpy(baseline_out, out_host + GUARD_ELEMENTS,
           (size_t)active_out_count * sizeof(float));
    check_inputs_immutable(heads_base, heads_host, heads_storage_bytes,
                           heads_hash,
                           model, model_bytes, model_hash);

    upload_poison(low_base, out_base, low_host, out_host,
                  low_storage_count, out_storage_count,
                  low_poison, out_poison);
    CHECK(unsetenv(k_disable) == 0, "enable direct candidate");
    CHECK(setenv(k_require, "1", 1) == 0, "require direct candidate");
    CHECK(ds4_gpu_attention_output_q4_K_batch_tensor(
              out, low, NULL, NULL, model, model_bytes,
              0, out_b_offset, Q8_0_TYPE, GROUP_DIM, RANK,
              N_GROUPS, OUT_DIM, heads, ACTIVE_ROWS) == 1,
          "direct candidate dispatch");
    read_outputs(low_base, out_base, low_host, out_host,
                 low_storage_count, out_storage_count);
    CHECK(count_poison_mismatches(
              low_host, GUARD_ELEMENTS,
              GUARD_ELEMENTS + active_low_count, low_poison) ==
              active_low_count,
          "direct candidate did not overwrite every active low value");
    CHECK(count_poison_mismatches(
              out_host, GUARD_ELEMENTS,
              GUARD_ELEMENTS + active_out_count, out_poison) ==
              active_out_count,
          "direct candidate did not overwrite every active out value");

    uint64_t first_low = UINT64_MAX;
    uint64_t first_out = UINT64_MAX;
    const uint64_t low_mismatch = count_bit_mismatches(
        baseline_low, low_host + GUARD_ELEMENTS,
        active_low_count, &first_low);
    const uint64_t out_mismatch = count_bit_mismatches(
        baseline_out, out_host + GUARD_ELEMENTS,
        active_out_count, &first_out);
    const uint64_t low_prefix_mismatch = count_poison_mismatches(
        low_host, 0, GUARD_ELEMENTS, low_poison);
    const uint64_t low_tail_mismatch = count_poison_mismatches(
        low_host, GUARD_ELEMENTS + active_low_count,
        GUARD_ELEMENTS + low_payload_count, low_poison);
    const uint64_t low_suffix_mismatch = count_poison_mismatches(
        low_host, GUARD_ELEMENTS + low_payload_count,
        low_storage_count, low_poison);
    const uint64_t out_prefix_mismatch = count_poison_mismatches(
        out_host, 0, GUARD_ELEMENTS, out_poison);
    const uint64_t out_tail_mismatch = count_poison_mismatches(
        out_host, GUARD_ELEMENTS + active_out_count,
        GUARD_ELEMENTS + out_payload_count, out_poison);
    const uint64_t out_suffix_mismatch = count_poison_mismatches(
        out_host, GUARD_ELEMENTS + out_payload_count,
        out_storage_count, out_poison);
    fprintf(stderr,
            "Metal Q4 output-A direct N=%u low=%llu/%llu out=%llu/%llu "
            "low_guard=%llu/%llu/%llu out_guard=%llu/%llu/%llu\n",
            ACTIVE_ROWS,
            (unsigned long long)low_mismatch,
            (unsigned long long)active_low_count,
            (unsigned long long)out_mismatch,
            (unsigned long long)active_out_count,
            (unsigned long long)low_prefix_mismatch,
            (unsigned long long)low_tail_mismatch,
            (unsigned long long)low_suffix_mismatch,
            (unsigned long long)out_prefix_mismatch,
            (unsigned long long)out_tail_mismatch,
            (unsigned long long)out_suffix_mismatch);
    if (low_mismatch != 0) {
        fprintf(stderr, "  first low mismatch index=%llu\n",
                (unsigned long long)first_low);
    }
    if (out_mismatch != 0) {
        fprintf(stderr, "  first out mismatch index=%llu\n",
                (unsigned long long)first_out);
    }
    CHECK(low_mismatch == 0, "direct low bitwise mismatch");
    CHECK(out_mismatch == 0, "direct final output bitwise mismatch");
    CHECK(low_prefix_mismatch == 0, "direct low prefix canary");
    CHECK(low_tail_mismatch == 0, "direct low tail canary");
    CHECK(low_suffix_mismatch == 0, "direct low suffix canary");
    CHECK(out_prefix_mismatch == 0, "direct out prefix canary");
    CHECK(out_tail_mismatch == 0, "direct out tail canary");
    CHECK(out_suffix_mismatch == 0, "direct out suffix canary");
    check_inputs_immutable(heads_base, heads_host, heads_storage_bytes,
                           heads_hash,
                           model, model_bytes, model_hash);

    /* REQUIRE must fail closed before encoding anything just below the
     * production threshold.  The entire low/out allocations are poison here,
     * so this also detects partial work from an ineligible dispatch. */
    const uint32_t reject_low_poison = 0x7fc30000u;
    const uint32_t reject_out_poison = 0x7fc40000u;
    upload_poison(low_base, out_base, low_host, out_host,
                  low_storage_count, out_storage_count,
                  reject_low_poison, reject_out_poison);
    CHECK(ds4_gpu_attention_output_q4_K_batch_tensor(
              out, low, NULL, NULL, model, model_bytes,
              0, out_b_offset, Q8_0_TYPE, GROUP_DIM, RANK,
              N_GROUPS, OUT_DIM, heads, REJECT_ROWS) == -1,
          "N=511 REQUIRE must fail closed");
    read_outputs(low_base, out_base, low_host, out_host,
                 low_storage_count, out_storage_count);
    CHECK(count_poison_mismatches(low_host, 0, low_storage_count,
                                  reject_low_poison) == 0,
          "N=511 REQUIRE modified low");
    CHECK(count_poison_mismatches(out_host, 0, out_storage_count,
                                  reject_out_poison) == 0,
          "N=511 REQUIRE modified out");
    check_inputs_immutable(heads_base, heads_host, heads_storage_bytes,
                           heads_hash,
                           model, model_bytes, model_hash);

    CHECK(unsetenv(k_require) == 0, "clear direct REQUIRE");
    CHECK(unsetenv(k_disable) == 0, "clear direct disable");
    ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_set_quality(false);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(low);
    ds4_gpu_tensor_free(heads);
    ds4_gpu_tensor_free(out_base);
    ds4_gpu_tensor_free(low_base);
    ds4_gpu_tensor_free(heads_base);
    ds4_gpu_cleanup();
    free(baseline_out);
    free(baseline_low);
    free(out_host);
    free(low_host);
    free(heads_host);
    free(model);
    fprintf(stderr,
            "Metal Q4 output-A direct oracle PASS N=513 bitwise=1 "
            "tail=1 immutable=1 reject_N511=1\n");
    return 0;
}

#else

int main(void) {
    fprintf(stderr, "Metal Q4 output-A direct oracle SKIP: non-Apple host\n");
    return 0;
}

#endif
