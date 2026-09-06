#define _DARWIN_C_SOURCE

/* Synthetic, GGUF-free bitwise oracle for the M1--M4 SSD-prefill Q4_K
 * attention-output token-tiled kernel. */

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
    QK_K = 256u,
    /* Multiblock geometry: attention-A spans 16 Q4_K blocks per row and
     * output-B spans four, exercising every ix lane and repeated ib steps. */
    GROUP_DIM = 4096u,
    RANK = 512u,
    N_GROUPS = 2u,
    LOW_DIM = N_GROUPS * RANK,
    OUT_DIM = 67u,
    MAX_ROWS = 31u,
    ALLOC_ROWS = 33u,
};

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2u];
} block_q4_K;

static const char *k_enable =
    "DS4_METAL_ENABLE_Q4_SSD_PREFILL_ATTN_OUT_EXACTN";
static const char *k_disable =
    "DS4_METAL_DISABLE_Q4_SSD_PREFILL_ATTN_OUT_EXACTN";
static const char *k_require =
    "DS4_METAL_REQUIRE_Q4_SSD_PREFILL_ATTN_OUT_EXACTN";
static const char *k_disable_scale_meta =
    "DS4_METAL_DISABLE_Q4_SSD_PREFILL_ATTN_OUT_SCALE_META";
static const char *k_require_scale_meta =
    "DS4_METAL_REQUIRE_Q4_SSD_PREFILL_ATTN_OUT_SCALE_META";
static const char *k_disable_classic = "DS4_METAL_DISABLE_Q4_MV_CLASSIC";
static const char *k_disable_f16_rhs =
    "DS4_METAL_DISABLE_Q4_ATTN_OUT_B_F16_RHS";
static const char *k_require_f16_rhs =
    "DS4_METAL_REQUIRE_Q4_ATTN_OUT_B_F16_RHS";

static void fail(const char *what) {
    fprintf(stderr, "Metal Q4 SSD-prefill exact-N oracle FAIL: %s\n", what);
    exit(1);
}

#define CHECK(expr, what) do { if (!(expr)) fail(what); } while (0)

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
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

static void fill_q4_matrix(void *raw,
                           uint32_t in_dim,
                           uint32_t rows,
                           uint32_t salt) {
    CHECK(sizeof(block_q4_K) == 144u, "unexpected Q4_K block size");
    CHECK((in_dim % QK_K) == 0u, "unaligned Q4_K fixture");
    const uint32_t blocks_per_row = in_dim / QK_K;
    block_q4_K *matrix = raw;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t block = 0; block < blocks_per_row; block++) {
            block_q4_K *b = matrix +
                (uint64_t)row * blocks_per_row + block;
            const uint32_t key =
                salt + row * 1009u + block * 313u +
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
            /* Exact binary scales: 2^-5 and 2^-7. */
            b->d = 0x2800u;
            b->dmin = 0x2000u;
        }
    }
}

static void poison(float *values, uint64_t count, uint32_t base) {
    for (uint64_t i = 0; i < count; i++) {
        const uint32_t bits = base + (uint32_t)(i & 0xffffu);
        memcpy(&values[i], &bits, sizeof(bits));
    }
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

int main(void) {
    static const uint32_t exact_rows[] = {6u, 8u, 9u, 16u, 21u, 30u, 31u};
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t row_a_bytes = (GROUP_DIM / QK_K) * sizeof(block_q4_K);
    const uint64_t out_a_bytes =
        (uint64_t)N_GROUPS * RANK * row_a_bytes;
    const uint64_t out_b_offset = align_up(out_a_bytes, page);
    const uint64_t row_b_bytes = (LOW_DIM / QK_K) * sizeof(block_q4_K);
    const uint64_t out_b_bytes = (uint64_t)OUT_DIM * row_b_bytes;
    const uint64_t model_bytes =
        align_up(out_b_offset + out_b_bytes, page);
    const uint64_t heads_row_bytes =
        (uint64_t)N_GROUPS * GROUP_DIM * sizeof(float);
    const uint64_t low_row_bytes = (uint64_t)LOW_DIM * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)OUT_DIM * sizeof(float);
    const uint64_t heads_count = (uint64_t)ALLOC_ROWS * N_GROUPS * GROUP_DIM;
    const uint64_t low_count = (uint64_t)ALLOC_ROWS * LOW_DIM;
    const uint64_t out_count = (uint64_t)ALLOC_ROWS * OUT_DIM;

    CHECK(unsetenv(k_enable) == 0, "clear enable env");
    CHECK(unsetenv(k_disable) == 0, "clear disable env");
    CHECK(unsetenv(k_require) == 0, "clear require env");
    CHECK(unsetenv(k_disable_scale_meta) == 0,
          "clear scale-meta disable env");
    CHECK(unsetenv(k_require_scale_meta) == 0,
          "clear scale-meta require env");
    CHECK(unsetenv(k_disable_classic) == 0, "clear classic kill env");
    CHECK(unsetenv(k_disable_f16_rhs) == 0, "clear F16 RHS disable env");
    CHECK(unsetenv(k_require_f16_rhs) == 0, "clear F16 RHS require env");

    CHECK(ds4_gpu_init() != 0, "Metal init");
    if (!ds4_gpu_device_is_pre_m5_apple_silicon()) {
        fprintf(stderr,
                "Metal Q4 SSD-prefill exact-N oracle SKIP: requires Apple M1--M4\n");
        ds4_gpu_cleanup();
        return 0;
    }

    void *model = NULL;
    CHECK(posix_memalign(&model, (size_t)page, (size_t)model_bytes) == 0,
          "model allocation");
    memset(model, 0, (size_t)model_bytes);
    fill_q4_matrix(model, GROUP_DIM, N_GROUPS * RANK, 211u);
    fill_q4_matrix((uint8_t *)model + out_b_offset,
                   LOW_DIM, OUT_DIM, 307u);

    float *heads_host = malloc((size_t)heads_count * sizeof(float));
    float *reference_low_host = calloc((size_t)low_count, sizeof(float));
    float *reference_out_host = calloc((size_t)out_count, sizeof(float));
    float *candidate_low_host = malloc((size_t)low_count * sizeof(float));
    float *candidate_out_host = malloc((size_t)out_count * sizeof(float));
    CHECK(heads_host && reference_low_host && reference_out_host &&
          candidate_low_host && candidate_out_host, "host tensors");

    for (uint32_t row = 0; row < ALLOC_ROWS; row++) {
        for (uint32_t i = 0; i < N_GROUPS * GROUP_DIM; i++) {
            const uint32_t key =
                i * 41u + row * 271u + ((i >> 2u) ^ (row * 19u));
            heads_host[(uint64_t)row * N_GROUPS * GROUP_DIM + i] =
                (float)((int)(key % 257u) - 128) / 137.0f;
        }
    }

    ds4_gpu_tensor *heads =
        ds4_gpu_tensor_alloc(heads_count * sizeof(float));
    ds4_gpu_tensor *reference_low =
        ds4_gpu_tensor_alloc(low_count * sizeof(float));
    ds4_gpu_tensor *reference_out =
        ds4_gpu_tensor_alloc(out_count * sizeof(float));
    ds4_gpu_tensor *candidate_low =
        ds4_gpu_tensor_alloc(low_count * sizeof(float));
    ds4_gpu_tensor *candidate_out =
        ds4_gpu_tensor_alloc(out_count * sizeof(float));
    CHECK(heads && reference_low && reference_out && candidate_low &&
          candidate_out, "Metal tensors");
    CHECK(ds4_gpu_tensor_write(heads, 0, heads_host,
                               heads_count * sizeof(float)) != 0,
          "heads upload");
    CHECK(ds4_gpu_tensor_write(reference_low, 0, reference_low_host,
                               low_count * sizeof(float)) != 0,
          "reference low clear");
    CHECK(ds4_gpu_tensor_write(reference_out, 0, reference_out_host,
                               out_count * sizeof(float)) != 0,
          "reference out clear");
    CHECK(ds4_gpu_set_model_map(model, model_bytes) != 0, "model map");
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);

    /* Canonical oracle: one row at a time through the established classic
     * Q4_K attention-A slice and output matvec entry points. */
    for (uint32_t row = 0; row < MAX_ROWS; row++) {
        ds4_gpu_tensor *heads_row = ds4_gpu_tensor_view(
            heads, (uint64_t)row * heads_row_bytes, heads_row_bytes);
        ds4_gpu_tensor *low_row = ds4_gpu_tensor_view(
            reference_low, (uint64_t)row * low_row_bytes, low_row_bytes);
        ds4_gpu_tensor *out_row = ds4_gpu_tensor_view(
            reference_out, (uint64_t)row * out_row_bytes, out_row_bytes);
        CHECK(heads_row && low_row && out_row, "reference views");
        CHECK(ds4_gpu_attention_output_low_q4_K_slice_tensor(
                  low_row, model, model_bytes, 0,
                  GROUP_DIM, RANK, 0, N_GROUPS, heads_row, 0) != 0,
              "reference low projection");
        CHECK(ds4_gpu_matmul_quant_tensor(
                  out_row, model, model_bytes, out_b_offset, Q4_K_TYPE,
                  LOW_DIM, OUT_DIM, low_row, 1u) != 0,
              "reference output projection");
        ds4_gpu_tensor_free(out_row);
        ds4_gpu_tensor_free(low_row);
        ds4_gpu_tensor_free(heads_row);
    }
    CHECK(ds4_gpu_tensor_read(reference_low, 0, reference_low_host,
                              low_count * sizeof(float)) != 0,
          "reference low read");
    CHECK(ds4_gpu_tensor_read(reference_out, 0, reference_out_host,
                              out_count * sizeof(float)) != 0,
          "reference out read");

    ds4_gpu_set_ssd_streaming(true);

    /* Default-off is observable through the production wrapper. */
    CHECK(ds4_gpu_attention_output_q4_K_batch_tensor(
              candidate_out, candidate_low, NULL, NULL, model, model_bytes,
              0, out_b_offset, Q4_K_TYPE, GROUP_DIM, RANK,
              N_GROUPS, OUT_DIM, heads, 6u) == 0,
          "default-off gate");
    CHECK(setenv(k_enable, "1", 1) == 0, "enable candidate");

    for (uint32_t scale_variant = 0; scale_variant < 2u; scale_variant++) {
        const bool use_scale_meta = scale_variant == 0u;
        if (use_scale_meta) {
            CHECK(unsetenv(k_disable_scale_meta) == 0,
                  "enable shared scale metadata");
            CHECK(setenv(k_require_scale_meta, "1", 1) == 0,
                  "require shared scale metadata");
        } else {
            CHECK(unsetenv(k_require_scale_meta) == 0,
                  "clear shared scale metadata requirement");
            CHECK(setenv(k_disable_scale_meta, "1", 1) == 0,
                  "select legacy scale unpack");
        }
        for (uint32_t case_i = 0;
             case_i < sizeof(exact_rows) / sizeof(exact_rows[0]);
             case_i++) {
            const uint32_t n_rows = exact_rows[case_i];
            poison(candidate_low_host, low_count, 0x7fc10000u);
            poison(candidate_out_host, out_count, 0x7fc20000u);
            CHECK(ds4_gpu_tensor_write(candidate_low, 0, candidate_low_host,
                                       low_count * sizeof(float)) != 0,
                  "candidate low poison");
            CHECK(ds4_gpu_tensor_write(candidate_out, 0, candidate_out_host,
                                       out_count * sizeof(float)) != 0,
                  "candidate out poison");

            /* Exercise the production wrapper delegation, not only the direct
             * test entry point. Scratch arguments are unused by this path. */
            CHECK(ds4_gpu_attention_output_q4_K_batch_tensor(
                      candidate_out, candidate_low, NULL, NULL,
                      model, model_bytes, 0, out_b_offset, Q4_K_TYPE,
                      GROUP_DIM, RANK, N_GROUPS, OUT_DIM, heads, n_rows) == 1,
                  "candidate dispatch");
            CHECK(ds4_gpu_tensor_read(candidate_low, 0, candidate_low_host,
                                      low_count * sizeof(float)) != 0,
                  "candidate low read");
            CHECK(ds4_gpu_tensor_read(candidate_out, 0, candidate_out_host,
                                      out_count * sizeof(float)) != 0,
                  "candidate out read");

            uint64_t first_low = UINT64_MAX;
            uint64_t first_out = UINT64_MAX;
            const uint64_t compared_low = (uint64_t)n_rows * LOW_DIM;
            const uint64_t compared_out = (uint64_t)n_rows * OUT_DIM;
            const uint64_t low_mismatch = count_bit_mismatches(
                reference_low_host, candidate_low_host,
                compared_low, &first_low);
            const uint64_t out_mismatch = count_bit_mismatches(
                reference_out_host, candidate_out_host,
                compared_out, &first_out);
            const uint64_t low_canary = count_poison_mismatches(
                candidate_low_host, compared_low, low_count, 0x7fc10000u);
            const uint64_t out_canary = count_poison_mismatches(
                candidate_out_host, compared_out, out_count, 0x7fc20000u);
            fprintf(stderr,
                    "Metal Q4 SSD-prefill exact-N=%u scale_meta=%s "
                    "low=%llu/%llu "
                    "out=%llu/%llu low_canary=%llu out_canary=%llu\n",
                    n_rows,
                    use_scale_meta ? "shared" : "legacy",
                    (unsigned long long)low_mismatch,
                    (unsigned long long)compared_low,
                    (unsigned long long)out_mismatch,
                    (unsigned long long)compared_out,
                    (unsigned long long)low_canary,
                    (unsigned long long)out_canary);
            if (low_mismatch != 0) {
                uint32_t reference_bits = 0;
                uint32_t candidate_bits = 0;
                memcpy(&reference_bits, &reference_low_host[first_low],
                       sizeof(reference_bits));
                memcpy(&candidate_bits, &candidate_low_host[first_low],
                       sizeof(candidate_bits));
                fprintf(stderr,
                        "  first low mismatch index=%llu "
                        "reference=%a (0x%08x) candidate=%a (0x%08x)\n",
                        (unsigned long long)first_low,
                        reference_low_host[first_low], reference_bits,
                        candidate_low_host[first_low], candidate_bits);
            }
            if (out_mismatch != 0) {
                fprintf(stderr, "  first out mismatch index=%llu\n",
                        (unsigned long long)first_out);
            }
            CHECK(low_mismatch == 0, "low projection bitwise mismatch");
            CHECK(out_mismatch == 0, "output projection bitwise mismatch");
            CHECK(low_canary == 0, "low tail canary");
            CHECK(out_canary == 0, "output tail canary");
        }
    }

    /* At N=32 both output-B variants execute the same legacy M64xN32xK32
     * schedule.  The candidate differs only by materializing the staging cast
     * once into scratch, so the complete wrapper output must remain bitwise
     * identical.  Keep one extra output row and guards around the F16 scratch
     * to catch either output or conversion overruns. */
    ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_set_quality(true);
    enum { F16_RHS_ROWS = 32u, F16_RHS_GUARD = 64u };
    const uint64_t f16_rhs_payload =
        (uint64_t)F16_RHS_ROWS * LOW_DIM;
    const uint64_t f16_rhs_storage =
        F16_RHS_GUARD + f16_rhs_payload + F16_RHS_GUARD;
    uint16_t *f16_rhs_host = malloc(
        (size_t)f16_rhs_storage * sizeof(uint16_t));
    ds4_gpu_tensor *f16_rhs_base = ds4_gpu_tensor_alloc(
        f16_rhs_storage * sizeof(uint16_t));
    ds4_gpu_tensor *f16_rhs = ds4_gpu_tensor_view(
        f16_rhs_base,
        F16_RHS_GUARD * sizeof(uint16_t),
        f16_rhs_payload * sizeof(uint16_t));
    CHECK(f16_rhs_host && f16_rhs_base && f16_rhs,
          "F16 RHS scratch allocation");
    for (uint64_t i = 0; i < f16_rhs_storage; i++) {
        f16_rhs_host[i] =
            (uint16_t)(0x7e00u | (uint16_t)(i & 0x1ffu));
    }
    CHECK(ds4_gpu_tensor_write(
              f16_rhs_base, 0, f16_rhs_host,
              f16_rhs_storage * sizeof(uint16_t)) != 0,
          "F16 RHS scratch poison");

    poison(reference_low_host, low_count, 0x7fc30000u);
    poison(reference_out_host, out_count, 0x7fc40000u);
    poison(candidate_low_host, low_count, 0x7fc30000u);
    poison(candidate_out_host, out_count, 0x7fc40000u);
    CHECK(ds4_gpu_tensor_write(reference_low, 0, reference_low_host,
                               low_count * sizeof(float)) != 0,
          "F16 RHS baseline low poison");
    CHECK(ds4_gpu_tensor_write(reference_out, 0, reference_out_host,
                               out_count * sizeof(float)) != 0,
          "F16 RHS baseline out poison");
    CHECK(ds4_gpu_tensor_write(candidate_low, 0, candidate_low_host,
                               low_count * sizeof(float)) != 0,
          "F16 RHS candidate low poison");
    CHECK(ds4_gpu_tensor_write(candidate_out, 0, candidate_out_host,
                               out_count * sizeof(float)) != 0,
          "F16 RHS candidate out poison");

    CHECK(ds4_gpu_attention_output_q4_K_batch_tensor(
              reference_out, reference_low, NULL, NULL,
              model, model_bytes, 0, out_b_offset, Q4_K_TYPE,
              GROUP_DIM, RANK, N_GROUPS, OUT_DIM, heads,
              F16_RHS_ROWS) == 1,
          "F16 RHS baseline dispatch");
    CHECK(ds4_gpu_attention_output_q4_K_batch_tensor(
              candidate_out, candidate_low, f16_rhs, NULL,
              model, model_bytes, 0, out_b_offset, Q4_K_TYPE,
              GROUP_DIM, RANK, N_GROUPS, OUT_DIM, heads,
              F16_RHS_ROWS) == 1,
          "F16 RHS default-on candidate dispatch");
    CHECK(ds4_gpu_tensor_read(reference_low, 0, reference_low_host,
                              low_count * sizeof(float)) != 0,
          "F16 RHS baseline low read");
    CHECK(ds4_gpu_tensor_read(reference_out, 0, reference_out_host,
                              out_count * sizeof(float)) != 0,
          "F16 RHS baseline out read");
    CHECK(ds4_gpu_tensor_read(candidate_low, 0, candidate_low_host,
                              low_count * sizeof(float)) != 0,
          "F16 RHS candidate low read");
    CHECK(ds4_gpu_tensor_read(candidate_out, 0, candidate_out_host,
                              out_count * sizeof(float)) != 0,
          "F16 RHS candidate out read");
    CHECK(ds4_gpu_tensor_read(
              f16_rhs_base, 0, f16_rhs_host,
              f16_rhs_storage * sizeof(uint16_t)) != 0,
          "F16 RHS scratch read");

    uint64_t first_low = UINT64_MAX;
    uint64_t first_out = UINT64_MAX;
    const uint64_t f16_low_mismatch = count_bit_mismatches(
        reference_low_host, candidate_low_host,
        (uint64_t)F16_RHS_ROWS * LOW_DIM, &first_low);
    const uint64_t f16_out_mismatch = count_bit_mismatches(
        reference_out_host, candidate_out_host,
        (uint64_t)F16_RHS_ROWS * OUT_DIM, &first_out);
    const uint64_t baseline_low_tail = count_poison_mismatches(
        reference_low_host, (uint64_t)F16_RHS_ROWS * LOW_DIM,
        low_count, 0x7fc30000u);
    const uint64_t candidate_low_tail = count_poison_mismatches(
        candidate_low_host, (uint64_t)F16_RHS_ROWS * LOW_DIM,
        low_count, 0x7fc30000u);
    const uint64_t baseline_out_tail = count_poison_mismatches(
        reference_out_host, (uint64_t)F16_RHS_ROWS * OUT_DIM,
        out_count, 0x7fc40000u);
    const uint64_t candidate_out_tail = count_poison_mismatches(
        candidate_out_host, (uint64_t)F16_RHS_ROWS * OUT_DIM,
        out_count, 0x7fc40000u);
    uint64_t f16_guard_mismatch = 0;
    uint64_t f16_payload_poison = 0;
    for (uint64_t i = 0; i < F16_RHS_GUARD; i++) {
        const uint16_t expected =
            (uint16_t)(0x7e00u | (uint16_t)(i & 0x1ffu));
        if (f16_rhs_host[i] != expected) f16_guard_mismatch++;
    }
    for (uint64_t i = F16_RHS_GUARD;
         i < F16_RHS_GUARD + f16_rhs_payload; i++) {
        const uint16_t poison_value =
            (uint16_t)(0x7e00u | (uint16_t)(i & 0x1ffu));
        if (f16_rhs_host[i] == poison_value) f16_payload_poison++;
    }
    for (uint64_t i = F16_RHS_GUARD + f16_rhs_payload;
         i < f16_rhs_storage; i++) {
        const uint16_t expected =
            (uint16_t)(0x7e00u | (uint16_t)(i & 0x1ffu));
        if (f16_rhs_host[i] != expected) f16_guard_mismatch++;
    }
    fprintf(stderr,
            "Metal Q4 attention output-B F16 RHS N=32 "
            "low=%llu out=%llu low_tail=%llu/%llu "
            "out_tail=%llu/%llu scratch_guard=%llu payload_poison=%llu\n",
            (unsigned long long)f16_low_mismatch,
            (unsigned long long)f16_out_mismatch,
            (unsigned long long)baseline_low_tail,
            (unsigned long long)candidate_low_tail,
            (unsigned long long)baseline_out_tail,
            (unsigned long long)candidate_out_tail,
            (unsigned long long)f16_guard_mismatch,
            (unsigned long long)f16_payload_poison);
    CHECK(f16_low_mismatch == 0, "F16 RHS low bitwise mismatch");
    CHECK(f16_out_mismatch == 0, "F16 RHS output bitwise mismatch");
    CHECK(baseline_low_tail == 0 && candidate_low_tail == 0,
          "F16 RHS low tail canary");
    CHECK(baseline_out_tail == 0 && candidate_out_tail == 0,
          "F16 RHS output tail canary");
    CHECK(f16_guard_mismatch == 0, "F16 RHS scratch canary");
    CHECK(f16_payload_poison == 0, "F16 RHS default-on materialization");

    /* Flash uses an output dimension divisible by 64, which selects the
     * direct full-tile store and its smaller threadgroup allocation.  Reuse
     * the first 64 rows of the fixture so both legacy specializations remain
     * covered without growing the model allocation. */
    enum { F16_RHS_FULL_TILE_OUT_DIM = 64u };
    const uint64_t f16_full_tile_count =
        (uint64_t)F16_RHS_ROWS * F16_RHS_FULL_TILE_OUT_DIM;
    poison(reference_out_host, out_count, 0x7fc50000u);
    poison(candidate_out_host, out_count, 0x7fc50000u);
    CHECK(ds4_gpu_tensor_write(reference_out, 0, reference_out_host,
                               out_count * sizeof(float)) != 0,
          "F16 RHS full-tile baseline poison");
    CHECK(ds4_gpu_tensor_write(candidate_out, 0, candidate_out_host,
                               out_count * sizeof(float)) != 0,
          "F16 RHS full-tile candidate poison");
    CHECK(ds4_gpu_attention_output_q4_K_batch_tensor(
              reference_out, reference_low, NULL, NULL,
              model, model_bytes, 0, out_b_offset, Q4_K_TYPE,
              GROUP_DIM, RANK, N_GROUPS, F16_RHS_FULL_TILE_OUT_DIM, heads,
              F16_RHS_ROWS) == 1,
          "F16 RHS full-tile baseline dispatch");
    CHECK(ds4_gpu_attention_output_q4_K_batch_tensor(
              candidate_out, candidate_low, f16_rhs, NULL,
              model, model_bytes, 0, out_b_offset, Q4_K_TYPE,
              GROUP_DIM, RANK, N_GROUPS, F16_RHS_FULL_TILE_OUT_DIM, heads,
              F16_RHS_ROWS) == 1,
          "F16 RHS full-tile candidate dispatch");
    CHECK(ds4_gpu_tensor_read(reference_out, 0, reference_out_host,
                              out_count * sizeof(float)) != 0,
          "F16 RHS full-tile baseline read");
    CHECK(ds4_gpu_tensor_read(candidate_out, 0, candidate_out_host,
                              out_count * sizeof(float)) != 0,
          "F16 RHS full-tile candidate read");
    uint64_t first_full_tile = UINT64_MAX;
    const uint64_t f16_full_tile_mismatch = count_bit_mismatches(
        reference_out_host, candidate_out_host,
        f16_full_tile_count, &first_full_tile);
    const uint64_t baseline_full_tile_tail = count_poison_mismatches(
        reference_out_host, f16_full_tile_count,
        out_count, 0x7fc50000u);
    const uint64_t candidate_full_tile_tail = count_poison_mismatches(
        candidate_out_host, f16_full_tile_count,
        out_count, 0x7fc50000u);
    fprintf(stderr,
            "Metal Q4 attention output-B F16 RHS full tile N=32 "
            "out=%llu tail=%llu/%llu\n",
            (unsigned long long)f16_full_tile_mismatch,
            (unsigned long long)baseline_full_tile_tail,
            (unsigned long long)candidate_full_tile_tail);
    CHECK(f16_full_tile_mismatch == 0,
          "F16 RHS full-tile output bitwise mismatch");
    CHECK(baseline_full_tile_tail == 0 && candidate_full_tile_tail == 0,
          "F16 RHS full-tile output tail canary");

    /* Restore the boundary-specialized reference consumed by the SSD
     * fallback comparison below. */
    poison(reference_out_host, out_count, 0x7fc40000u);
    CHECK(ds4_gpu_tensor_write(reference_out, 0, reference_out_host,
                               out_count * sizeof(float)) != 0,
          "F16 RHS boundary reference poison restore");
    CHECK(ds4_gpu_attention_output_q4_K_batch_tensor(
              reference_out, reference_low, NULL, NULL,
              model, model_bytes, 0, out_b_offset, Q4_K_TYPE,
              GROUP_DIM, RANK, N_GROUPS, OUT_DIM, heads,
              F16_RHS_ROWS) == 1,
          "F16 RHS boundary reference restore dispatch");
    CHECK(ds4_gpu_tensor_read(reference_out, 0, reference_out_host,
                              out_count * sizeof(float)) != 0,
          "F16 RHS boundary reference restore read");

    CHECK(setenv(k_require_f16_rhs, "1", 1) == 0,
          "require F16 RHS candidate");
    CHECK(ds4_gpu_attention_output_q4_K_batch_tensor(
              candidate_out, candidate_low, f16_rhs, NULL,
              model, model_bytes, 0, out_b_offset, Q4_K_TYPE,
              GROUP_DIM, RANK, N_GROUPS, OUT_DIM, heads,
              F16_RHS_ROWS) == 1,
          "required F16 RHS candidate dispatch");
    CHECK(ds4_gpu_attention_output_q4_K_batch_tensor(
              candidate_out, candidate_low, f16_rhs, NULL,
              model, model_bytes, 0, out_b_offset, Q4_K_TYPE,
              GROUP_DIM, RANK, N_GROUPS, OUT_DIM, heads, 31u) == -1,
          "required F16 RHS rejects N below one MM tile");
    CHECK(setenv(k_disable_f16_rhs, "1", 1) == 0,
          "disable required F16 RHS candidate");
    CHECK(ds4_gpu_attention_output_q4_K_batch_tensor(
              candidate_out, candidate_low, f16_rhs, NULL,
              model, model_bytes, 0, out_b_offset, Q4_K_TYPE,
              GROUP_DIM, RANK, N_GROUPS, OUT_DIM, heads,
              F16_RHS_ROWS) == -1,
          "F16 RHS disable wins over REQUIRE");
    CHECK(unsetenv(k_disable_f16_rhs) == 0,
          "clear F16 RHS disable env");
    ds4_gpu_set_ssd_streaming(true);
    CHECK(ds4_gpu_attention_output_q4_K_batch_tensor(
              candidate_out, candidate_low, f16_rhs, NULL,
              model, model_bytes, 0, out_b_offset, Q4_K_TYPE,
              GROUP_DIM, RANK, N_GROUPS, OUT_DIM, heads,
              F16_RHS_ROWS) == -1,
          "required F16 RHS rejects SSD streaming");
    CHECK(unsetenv(k_require_f16_rhs) == 0,
          "clear F16 RHS require env");

    /* In ordinary SSD mode the wrapper must use output-B's established F32
     * path, return success, and leave the F16 scratch completely untouched. */
    for (uint64_t i = 0; i < f16_rhs_storage; i++) {
        f16_rhs_host[i] =
            (uint16_t)(0x7e00u | (uint16_t)(i & 0x1ffu));
    }
    poison(candidate_low_host, low_count, 0x7fc30000u);
    poison(candidate_out_host, out_count, 0x7fc40000u);
    CHECK(ds4_gpu_tensor_write(
              f16_rhs_base, 0, f16_rhs_host,
              f16_rhs_storage * sizeof(uint16_t)) != 0,
          "SSD fallback scratch poison");
    CHECK(ds4_gpu_tensor_write(candidate_low, 0, candidate_low_host,
                               low_count * sizeof(float)) != 0,
          "SSD fallback low poison");
    CHECK(ds4_gpu_tensor_write(candidate_out, 0, candidate_out_host,
                               out_count * sizeof(float)) != 0,
          "SSD fallback out poison");
    CHECK(ds4_gpu_attention_output_q4_K_batch_tensor(
              candidate_out, candidate_low, f16_rhs, NULL,
              model, model_bytes, 0, out_b_offset, Q4_K_TYPE,
              GROUP_DIM, RANK, N_GROUPS, OUT_DIM, heads,
              F16_RHS_ROWS) == 1,
          "SSD fallback F32 dispatch");
    CHECK(ds4_gpu_tensor_read(candidate_low, 0, candidate_low_host,
                              low_count * sizeof(float)) != 0,
          "SSD fallback low read");
    CHECK(ds4_gpu_tensor_read(candidate_out, 0, candidate_out_host,
                              out_count * sizeof(float)) != 0,
          "SSD fallback out read");
    CHECK(ds4_gpu_tensor_read(
              f16_rhs_base, 0, f16_rhs_host,
              f16_rhs_storage * sizeof(uint16_t)) != 0,
          "SSD fallback scratch read");
    const uint64_t ssd_low_mismatch = count_bit_mismatches(
        reference_low_host, candidate_low_host,
        (uint64_t)F16_RHS_ROWS * LOW_DIM, &first_low);
    const uint64_t ssd_out_mismatch = count_bit_mismatches(
        reference_out_host, candidate_out_host,
        (uint64_t)F16_RHS_ROWS * OUT_DIM, &first_out);
    uint64_t ssd_scratch_mismatch = 0;
    for (uint64_t i = 0; i < f16_rhs_storage; i++) {
        const uint16_t expected =
            (uint16_t)(0x7e00u | (uint16_t)(i & 0x1ffu));
        if (f16_rhs_host[i] != expected) ssd_scratch_mismatch++;
    }
    fprintf(stderr,
            "Metal Q4 attention output-B SSD fallback N=32 "
            "low=%llu out=%llu scratch=%llu\n",
            (unsigned long long)ssd_low_mismatch,
            (unsigned long long)ssd_out_mismatch,
            (unsigned long long)ssd_scratch_mismatch);
    CHECK(ssd_low_mismatch == 0, "SSD fallback low bitwise mismatch");
    CHECK(ssd_out_mismatch == 0, "SSD fallback output bitwise mismatch");
    CHECK(ssd_scratch_mismatch == 0, "SSD fallback touched F16 scratch");
    ds4_gpu_set_quality(false);
    ds4_gpu_tensor_free(f16_rhs);
    ds4_gpu_tensor_free(f16_rhs_base);
    free(f16_rhs_host);

    CHECK(unsetenv(k_disable_scale_meta) == 0,
          "restore shared scale metadata");
    CHECK(unsetenv(k_require_scale_meta) == 0,
          "clear shared scale metadata requirement");
    CHECK(setenv(k_require_scale_meta, "1", 1) == 0,
          "set scale-meta REQUIRE for kill-switch check");
    CHECK(setenv(k_disable_scale_meta, "1", 1) == 0,
          "set scale-meta disable for kill-switch check");
    CHECK(ds4_gpu_attention_output_q4_K_batch_tensor(
              candidate_out, candidate_low, NULL, NULL, model, model_bytes,
              0, out_b_offset, Q4_K_TYPE, GROUP_DIM, RANK,
              N_GROUPS, OUT_DIM, heads, 21u) == -1,
          "scale-meta disable wins over REQUIRE");
    CHECK(unsetenv(k_disable_scale_meta) == 0,
          "clear scale-meta disable after kill-switch check");
    CHECK(unsetenv(k_require_scale_meta) == 0,
          "clear scale-meta REQUIRE after kill-switch check");

    /* REQUIRE implies enable.  Both explicit kill switches win and must
     * return -1 instead of allowing a false-green row fallback. */
    CHECK(unsetenv(k_enable) == 0, "clear enable before REQUIRE");
    CHECK(setenv(k_require, "1", 1) == 0, "set REQUIRE");
    CHECK(ds4_gpu_attention_output_q4_K_batch_tensor(
              candidate_out, candidate_low, NULL, NULL, model, model_bytes,
              0, out_b_offset, Q4_K_TYPE, GROUP_DIM, RANK,
              N_GROUPS, OUT_DIM, heads, 21u) == 1,
          "REQUIRE implies enable");
    CHECK(setenv(k_disable, "1", 1) == 0, "set exact-N disable");
    CHECK(ds4_gpu_attention_output_q4_K_batch_tensor(
              candidate_out, candidate_low, NULL, NULL, model, model_bytes,
              0, out_b_offset, Q4_K_TYPE, GROUP_DIM, RANK,
              N_GROUPS, OUT_DIM, heads, 21u) == -1,
          "disable wins over REQUIRE");
    CHECK(unsetenv(k_disable) == 0, "clear exact-N disable");
    CHECK(setenv(k_disable_classic, "1", 1) == 0, "set classic disable");
    CHECK(ds4_gpu_attention_output_q4_K_batch_tensor(
              candidate_out, candidate_low, NULL, NULL, model, model_bytes,
              0, out_b_offset, Q4_K_TYPE, GROUP_DIM, RANK,
              N_GROUPS, OUT_DIM, heads, 21u) == -1,
          "classic kill wins over REQUIRE");

    ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_tensor_free(candidate_out);
    ds4_gpu_tensor_free(candidate_low);
    ds4_gpu_tensor_free(reference_out);
    ds4_gpu_tensor_free(reference_low);
    ds4_gpu_tensor_free(heads);
    ds4_gpu_cleanup();
    free(candidate_out_host);
    free(candidate_low_host);
    free(reference_out_host);
    free(reference_low_host);
    free(heads_host);
    free(model);
    fprintf(stderr,
            "Metal Q4 SSD-prefill exact-N oracle PASS rows=6,8,9,16,21,30,31 "
            "scale_meta=shared,legacy bitwise=1 canary=1 gates=1\n");
    return 0;
}

#else

int main(void) {
    fprintf(stderr, "Metal Q4 SSD-prefill exact-N oracle SKIP: non-Apple host\n");
    return 0;
}

#endif
