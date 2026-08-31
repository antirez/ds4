#include "ds4_qwen4.h"
#include "ds4_image.h"

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static int failures;

#define CHECK(expr) do {                                                       \
    if (!(expr)) {                                                             \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__,     \
                #expr);                                                        \
        failures++;                                                            \
    }                                                                          \
} while (0)

static void store_u16_le(uint8_t *p, uint16_t value) {
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
}

static void store_u32_le(uint8_t *p, uint32_t value) {
    for (unsigned i = 0; i < 4; i++) p[i] = (uint8_t)(value >> (8u * i));
}

static void store_u64_le(uint8_t *p, uint64_t value) {
    for (unsigned i = 0; i < 8; i++) p[i] = (uint8_t)(value >> (8u * i));
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static bool write_all(int fd, const void *data, size_t size) {
    const uint8_t *p = data;
    while (size != 0) {
        const ssize_t wrote = write(fd, p, size);
        if (wrote <= 0) return false;
        p += (size_t)wrote;
        size -= (size_t)wrote;
    }
    return true;
}

typedef struct {
    uint8_t *data;
    size_t capacity;
    size_t used;
    bool ok;
} test_buffer;

static void test_buffer_bytes(test_buffer *buffer,
                              const void *data,
                              size_t size) {
    if (!buffer->ok || size > buffer->capacity - buffer->used) {
        buffer->ok = false;
        return;
    }
    memcpy(buffer->data + buffer->used, data, size);
    buffer->used += size;
}

static void test_buffer_u32(test_buffer *buffer, uint32_t value) {
    uint8_t bytes[4];
    store_u32_le(bytes, value);
    test_buffer_bytes(buffer, bytes, sizeof(bytes));
}

static void test_buffer_u64(test_buffer *buffer, uint64_t value) {
    uint8_t bytes[8];
    store_u64_le(bytes, value);
    test_buffer_bytes(buffer, bytes, sizeof(bytes));
}

static void test_buffer_string(test_buffer *buffer, const char *value) {
    const size_t size = strlen(value);
    test_buffer_u64(buffer, size);
    test_buffer_bytes(buffer, value, size);
}

static void test_buffer_kv_u32(test_buffer *buffer,
                               const char *key,
                               uint32_t value) {
    test_buffer_string(buffer, key);
    test_buffer_u32(buffer, 4u);
    test_buffer_u32(buffer, value);
}

static void test_buffer_kv_string(test_buffer *buffer,
                                  const char *key,
                                  const char *value) {
    test_buffer_string(buffer, key);
    test_buffer_u32(buffer, 8u);
    test_buffer_string(buffer, value);
}

static void test_buffer_tensor(test_buffer *buffer,
                               const char *name,
                               uint32_t dimensions,
                               const uint64_t *shape,
                               uint32_t qtype,
                               uint64_t offset) {
    test_buffer_string(buffer, name);
    test_buffer_u32(buffer, dimensions);
    for (uint32_t i = 0; i < dimensions; i++)
        test_buffer_u64(buffer, shape[i]);
    test_buffer_u32(buffer, qtype);
    test_buffer_u64(buffer, offset);
}

static void test_buffer_align(test_buffer *buffer, size_t alignment) {
    const size_t aligned = (buffer->used + alignment - 1u) & ~(alignment - 1u);
    if (aligned > buffer->capacity) {
        buffer->ok = false;
        return;
    }
    memset(buffer->data + buffer->used, 0, aligned - buffer->used);
    buffer->used = aligned;
}

static size_t build_ple_gguf(uint8_t *file,
                             size_t capacity,
                             uint32_t pack_version,
                             uint64_t *weight_offset_out) {
    enum {
        ALIGNMENT = 32,
        ROWS = 2,
        DIM = DS4_QWEN4_PLE_ROW_DIM,
        BLOCKS = DIM / 32,
        ROW_BYTES = BLOCKS * 20,
    };
    const uint64_t weight_relative = 0u;
    const uint64_t multipliers_relative = 224u;
    const uint64_t offsets_relative = 256u;
    const uint64_t vocab_relative = 384u;
    test_buffer buffer = {
        .data = file,
        .capacity = capacity,
        .used = 0u,
        .ok = true,
    };
    test_buffer_bytes(&buffer, "GGUF", 4u);
    test_buffer_u32(&buffer, 3u);
    test_buffer_u64(&buffer, 4u);
    test_buffer_u64(&buffer, 7u);
    test_buffer_kv_string(&buffer, "general.architecture",
                          DS4_QWEN4_PLE_ARCHITECTURE);
    test_buffer_kv_u32(&buffer, "general.alignment", ALIGNMENT);
    test_buffer_kv_u32(&buffer, "ds4.pack.version",
                       pack_version);
    test_buffer_kv_string(
        &buffer, "ds4.pack.id", "0123456789abcdef0123456789abcdef");
    test_buffer_kv_string(&buffer, "general.source.revision", "deadbeef");
    test_buffer_kv_string(&buffer, "ds4.pack.artifact", "ple");
    test_buffer_kv_string(&buffer, "ds4.pack.quant.ple", "Q4_1");
    const uint64_t weight_shape[] = {DIM, ROWS};
    const uint64_t multipliers_shape[] = {3u};
    const uint64_t heads_shape[] = {DS4_QWEN4_NGRAM_HEADS};
    test_buffer_tensor(&buffer, "ple.weight", 2u, weight_shape, 3u,
                       weight_relative);
    test_buffer_tensor(&buffer, "ple.layer_multipliers", 1u,
                       multipliers_shape, 27u, multipliers_relative);
    test_buffer_tensor(&buffer, "ple.ngram_heads_offsets", 1u,
                       heads_shape, 27u, offsets_relative);
    test_buffer_tensor(&buffer, "ple.ngram_heads_vocab_sizes", 1u,
                       heads_shape, 27u, vocab_relative);
    test_buffer_align(&buffer, ALIGNMENT);
    if (!buffer.ok || buffer.used > capacity - 512u) return 0u;
    const size_t data_offset = buffer.used;
    memset(file + data_offset, 0, 512u);
    for (uint32_t row = 0; row < ROWS; row++) {
        uint8_t *row_data = file + data_offset + row * ROW_BYTES;
        for (uint32_t block = 0; block < BLOCKS; block++) {
            uint8_t *q4 = row_data + block * 20u;
            store_u16_le(q4, row == 0u ? UINT16_C(0x3800) : 0u);
            store_u16_le(q4 + 2u,
                         row == 0u ? UINT16_C(0x3c00) : UINT16_C(0x4500));
            for (uint32_t i = 0; i < 16u; i++)
                q4[4u + i] = row == 0u ? (uint8_t)(i | (i << 4))
                                        : UINT8_C(0x33);
        }
    }
    ds4_qwen4_ngram_hash hash;
    if (!ds4_qwen4_ngram_hash_init(
            &hash, DS4_QWEN4_VOCAB, DS4_QWEN4_NGRAM_SIZE,
            DS4_QWEN4_HEADS_PER_NGRAM, DS4_QWEN4_NGRAM_VOCAB_BASE,
            DS4_QWEN4_NGRAM_VOCAB_DIVISOR, DS4_QWEN4_NGRAM_SEED, 0u,
            DS4_QWEN4_NGRAM_EOS)) return 0u;
    for (uint32_t i = 0; i < DS4_QWEN4_NGRAM_SIZE; i++)
        store_u64_le(file + data_offset + multipliers_relative +
                         (uint64_t)i * sizeof(int64_t),
                     (uint64_t)hash.multipliers[i]);
    for (uint32_t i = 0; i < DS4_QWEN4_NGRAM_HEADS; i++) {
        store_u64_le(file + data_offset + offsets_relative +
                         (uint64_t)i * sizeof(int64_t),
                     (uint64_t)hash.offsets[i]);
        store_u64_le(file + data_offset + vocab_relative +
                         (uint64_t)i * sizeof(int64_t),
                     (uint64_t)hash.vocab[i]);
    }
    if (weight_offset_out) *weight_offset_out = data_offset;
    return data_offset + 512u;
}

static void test_prefill_policy(void) {
    char error[128];
    ds4_qwen4_prefill_mode mode = DS4_QWEN4_PREFILL_AUTO;
    CHECK(ds4_qwen4_parse_prefill_mode("auto", &mode, error, sizeof(error)));
    CHECK(mode == DS4_QWEN4_PREFILL_AUTO);
    CHECK(ds4_qwen4_parse_prefill_mode("8192", &mode, error, sizeof(error)));
    CHECK(mode == DS4_QWEN4_PREFILL_8192);
    CHECK(!ds4_qwen4_parse_prefill_mode("1024", &mode, error, sizeof(error)));
    CHECK(strstr(error, "auto, 2048, 4096, or 8192") != NULL);

    const uint64_t scratch2k = ds4_qwen4_prefill_scratch_bytes(2048);
    const uint64_t scratch4k = ds4_qwen4_prefill_scratch_bytes(4096);
    const uint64_t scratch8k = ds4_qwen4_prefill_scratch_bytes(8192);
    CHECK(scratch2k > 256ull * 1024ull * 1024ull);
    CHECK(scratch2k < scratch4k && scratch4k < scratch8k);
    CHECK(ds4_qwen4_prefill_scratch_bytes(1024) == UINT64_MAX);

    ds4_qwen4_prefill_admission admission = {0};
    CHECK(!ds4_qwen4_admit_prefill(
              DS4_QWEN4_PREFILL_AUTO, true, NULL, scratch2k - 1u,
              &admission, error, sizeof(error)));
    CHECK(admission.admitted_cap == 0u);
    CHECK(strstr(error, "at least 2048-token scratch") != NULL);
    CHECK(ds4_qwen4_admit_prefill(
              DS4_QWEN4_PREFILL_AUTO, true, NULL, scratch2k,
              &admission, error, sizeof(error)));
    CHECK(admission.admitted_cap == 2048u);
    CHECK(admission.scratch_required == scratch2k);
    CHECK(ds4_qwen4_admit_prefill(
              DS4_QWEN4_PREFILL_AUTO, true, NULL, scratch8k - 1u,
              &admission, error, sizeof(error)));
    CHECK(admission.admitted_cap == 2048u);
    CHECK(ds4_qwen4_admit_prefill(
              DS4_QWEN4_PREFILL_AUTO, true, NULL, scratch8k,
              &admission, error, sizeof(error)));
    CHECK(admission.admitted_cap == 8192u);
    CHECK(admission.scratch_required == scratch8k);
    CHECK(ds4_qwen4_admit_prefill(
              DS4_QWEN4_PREFILL_AUTO, false, "missing exact Q8", scratch8k,
              &admission, error, sizeof(error)));
    CHECK(admission.admitted_cap == 2048u);
    CHECK(!ds4_qwen4_admit_prefill(
              DS4_QWEN4_PREFILL_8192, false, "missing exact Q8", scratch8k,
              &admission, error, sizeof(error)));
    CHECK(strstr(error, "missing exact Q8") != NULL);

    const uint64_t mtp_per_token =
        (uint64_t)DS4_QWEN4_HC_COUNT * DS4_QWEN4_HIDDEN * sizeof(float);
    const uint64_t mtp_scratch2k = scratch2k + 2048u * mtp_per_token;
    const uint64_t mtp_scratch8k = scratch8k + 8192u * mtp_per_token;
    CHECK(ds4_qwen4_admit_prefill_with_extra(
              DS4_QWEN4_PREFILL_AUTO, true, NULL, mtp_scratch8k - 1u,
              mtp_per_token, &admission, error, sizeof(error)));
    CHECK(admission.admitted_cap == 2048u);
    CHECK(admission.scratch_required == mtp_scratch2k);
    CHECK(ds4_qwen4_admit_prefill_with_extra(
              DS4_QWEN4_PREFILL_AUTO, true, NULL, mtp_scratch8k,
              mtp_per_token, &admission, error, sizeof(error)));
    CHECK(admission.admitted_cap == 8192u);
    CHECK(admission.scratch_required == mtp_scratch8k);

    const char *reason = NULL;
    CHECK(ds4_qwen4_select_prefill_chunk(DS4_QWEN4_PREFILL_AUTO, true,
                                         8192, 8192, 0, &reason) == 8192);
    CHECK(!strcmp(reason, "cold Qwen native path"));
    CHECK(ds4_qwen4_select_prefill_chunk(DS4_QWEN4_PREFILL_AUTO, false,
                                         8192, 8192, 0, &reason) == 2048);
    CHECK(!strcmp(reason, "mandatory Qwen fast path incomplete"));
    CHECK(ds4_qwen4_select_prefill_chunk(DS4_QWEN4_PREFILL_AUTO, true,
                                         8192, 16384, 2048, &reason) == 2048);
    CHECK(!strcmp(reason, "resuming prefix cache"));
    CHECK(ds4_qwen4_select_prefill_chunk(DS4_QWEN4_PREFILL_AUTO, true,
                                         8192, 8191, 0, &reason) == 2048);
    CHECK(!strcmp(reason, "uncached suffix shorter than 8192"));
    CHECK(ds4_qwen4_select_prefill_chunk(DS4_QWEN4_PREFILL_AUTO, true,
                                         2048, 8192, 0, &reason) == 2048);
    CHECK(!strcmp(reason, "pre-admitted scratch cap"));
    CHECK(ds4_qwen4_select_prefill_chunk(DS4_QWEN4_PREFILL_4096, false,
                                         4096, 1, 1, &reason) == 4096);
}

static void test_pack_profiles(void) {
    char error[256] = {0};
    ds4_qwen4_pack_profile profile = DS4_QWEN4_PACK_PROFILE_INVALID;

    CHECK(ds4_qwen4_pack_profile_from_version(
        DS4_QWEN4_PACK_VERSION_Q4, &profile));
    CHECK(profile == DS4_QWEN4_PACK_PROFILE_Q4_K);
    CHECK(!strcmp(ds4_qwen4_pack_profile_name(profile), "q4_k"));
    CHECK(!strcmp(ds4_qwen4_pack_manifest_file(profile),
                  DS4_QWEN4_PACK_MANIFEST_FILE));
    CHECK(!strcmp(ds4_qwen4_pack_routed_name(profile), "Q4_K"));
    CHECK(ds4_qwen4_pack_routed_metadata_valid(
        profile, "Q4_K", NULL, NULL, error, sizeof(error)));
    CHECK(!ds4_qwen4_pack_routed_metadata_valid(
        profile, "mixed-q2", NULL, NULL, error, sizeof(error)));
    CHECK(strstr(error, "expected Q4_K") != NULL);
    CHECK(ds4_qwen4_pack_routed_qtypes_valid(
        profile, DS4_QWEN4_QTYPE_Q4_K, DS4_QWEN4_QTYPE_Q4_K,
        DS4_QWEN4_QTYPE_Q4_K, error, sizeof(error)));
    CHECK(!ds4_qwen4_pack_routed_qtypes_valid(
        profile, DS4_QWEN4_QTYPE_IQ2_XXS, DS4_QWEN4_QTYPE_IQ2_XXS,
        DS4_QWEN4_QTYPE_Q2_K, error, sizeof(error)));
    CHECK(strstr(error, "expected 12/12/12") != NULL);
    CHECK(ds4_qwen4_pack_fast_required_mask(profile) ==
          (DS4_QWEN4_FAST_Q4_METAL_REQUIRED_MASK |
           DS4_QWEN4_FAST_PLE_STAGER));

    CHECK(ds4_qwen4_pack_profile_from_version(
        DS4_QWEN4_PACK_VERSION_Q2, &profile));
    CHECK(profile == DS4_QWEN4_PACK_PROFILE_IQ2_XXS_Q2_K);
    CHECK(!strcmp(ds4_qwen4_pack_profile_name(profile),
                  "iq2_xxs_gate_up_q2_k_down"));
    CHECK(!strcmp(ds4_qwen4_pack_manifest_file(profile),
                  DS4_QWEN4_PACK_Q2_MANIFEST_FILE));
    CHECK(!strcmp(ds4_qwen4_pack_routed_name(profile), "mixed-q2"));
    CHECK(!strcmp(ds4_qwen4_pack_routed_gate_up_name(profile), "IQ2_XXS"));
    CHECK(!strcmp(ds4_qwen4_pack_routed_down_name(profile), "Q2_K"));
    CHECK(ds4_qwen4_pack_routed_metadata_valid(
        profile, "mixed-q2", "IQ2_XXS", "Q2_K",
        error, sizeof(error)));
    CHECK(!ds4_qwen4_pack_routed_metadata_valid(
        profile, "mixed-q2", "IQ2_XXS", "Q4_K",
        error, sizeof(error)));
    CHECK(strstr(error, "expected IQ2_XXS/Q2_K") != NULL);
    CHECK(!ds4_qwen4_pack_routed_metadata_valid(
        profile, "mixed-q2", NULL, "Q2_K", error, sizeof(error)));
    CHECK(ds4_qwen4_pack_routed_qtypes_valid(
        profile, DS4_QWEN4_QTYPE_IQ2_XXS, DS4_QWEN4_QTYPE_IQ2_XXS,
        DS4_QWEN4_QTYPE_Q2_K, error, sizeof(error)));
    CHECK(!ds4_qwen4_pack_routed_qtypes_valid(
        profile, DS4_QWEN4_QTYPE_IQ2_XXS, DS4_QWEN4_QTYPE_IQ2_XXS,
        DS4_QWEN4_QTYPE_Q4_K, error, sizeof(error)));
    CHECK(strstr(error, "expected 16/16/10") != NULL);
    CHECK(ds4_qwen4_pack_fast_required_mask(profile) ==
          (DS4_QWEN4_FAST_Q2_METAL_REQUIRED_MASK |
           DS4_QWEN4_FAST_PLE_STAGER));

    CHECK(!ds4_qwen4_pack_profile_from_version(2u, &profile));
    CHECK(profile == DS4_QWEN4_PACK_PROFILE_INVALID);
    CHECK(ds4_qwen4_pack_fast_required_mask(profile) == 0u);
}

static void test_mrope_positions(void) {
    enum { CAPACITY = 16, PROMPT = 12 };
    const ds4_qwen4_mrope_span images[] = {
        {.token_start = 2, .token_count = 4,
         .grid_width = 4, .grid_height = 4},
        {.token_start = 8, .token_count = 2,
         .grid_width = 4, .grid_height = 2},
    };
    int32_t positions[3 * CAPACITY];
    int32_t delta = 0;
    CHECK(ds4_qwen4_build_mrope_positions(
        positions, CAPACITY, PROMPT, images, 2, &delta));
    CHECK(delta == -2);
    static const int32_t expected[3][CAPACITY] = {
        {0,1,2,2,2,2,4,5,6,6,8,9,10,11,12,13},
        {0,1,2,2,3,3,4,5,6,6,8,9,10,11,12,13},
        {0,1,2,3,2,3,4,5,6,7,8,9,10,11,12,13},
    };
    CHECK(!memcmp(positions, expected, sizeof(expected)));

    ds4_qwen4_mrope_span invalid = images[0];
    invalid.token_count = 3;
    CHECK(!ds4_qwen4_build_mrope_positions(
        positions, CAPACITY, PROMPT, &invalid, 1, &delta));
    invalid = images[0];
    invalid.grid_width = 3;
    CHECK(!ds4_qwen4_build_mrope_positions(
        positions, CAPACITY, PROMPT, &invalid, 1, &delta));
    CHECK(!ds4_qwen4_build_mrope_positions(
        positions, 4, PROMPT, images, 2, &delta));
}

static float normalized_u8(uint8_t value) {
    return (float)value * (2.0f / 255.0f) - 1.0f;
}

static void test_qwen4_image_preprocess(void) {
    enum { SIDE = 32, PATCH_VALUES = 3 * 2 * 16 * 16 };
    ds4_image image = {
        .width = SIDE,
        .height = SIDE,
        .rgb = malloc((size_t)SIDE * SIDE * 3u),
    };
    CHECK(image.rgb != NULL);
    if (!image.rgb) return;
    for (uint32_t y = 0; y < SIDE; y++) {
        for (uint32_t x = 0; x < SIDE; x++) {
            uint8_t *pixel = image.rgb + ((size_t)y * SIDE + x) * 3u;
            pixel[0] = (uint8_t)x;
            pixel[1] = (uint8_t)y;
            pixel[2] = (uint8_t)(x + y);
        }
    }

    ds4_image_patches patches;
    char error[128];
    CHECK(ds4_image_preprocess_qwen4(
        &patches, &image, SIDE * SIDE, SIDE * SIDE,
        error, sizeof(error)));
    CHECK(patches.content_width == SIDE && patches.content_height == SIDE);
    CHECK(patches.padded_width == SIDE && patches.padded_height == SIDE);
    CHECK(patches.grid_width == 2 && patches.grid_height == 2);
    CHECK(patches.patch_count == 4 && patches.image_token_count == 1);
    if (patches.patches) {
        const float *p0 = patches.patches;
        const float *p1 = patches.patches + PATCH_VALUES;
        const float *p2 = patches.patches + 2 * PATCH_VALUES;
        const float *p3 = patches.patches + 3 * PATCH_VALUES;
        CHECK(fabsf(p0[0] - normalized_u8(0)) < 1e-7f);
        CHECK(fabsf(p0[15] - normalized_u8(15)) < 1e-7f);
        CHECK(fabsf(p0[16 * 16] - p0[0]) < 1e-7f);
        CHECK(fabsf(p0[2 * 16 * 16] - normalized_u8(0)) < 1e-7f);
        CHECK(fabsf(p1[0] - normalized_u8(16)) < 1e-7f);
        CHECK(fabsf(p2[2 * 16 * 16] - normalized_u8(16)) < 1e-7f);
        CHECK(fabsf(p3[4 * 16 * 16] - normalized_u8(32)) < 1e-7f);
    }
    ds4_image_patches_free(&patches);
    free(image.rgb);
}

static void test_ngram_hash(void) {
    ds4_qwen4_ngram_hash hash;
    CHECK(ds4_qwen4_ngram_hash_init(&hash, DS4_QWEN4_VOCAB,
                                     DS4_QWEN4_NGRAM_SIZE,
                                     DS4_QWEN4_HEADS_PER_NGRAM,
                                     DS4_QWEN4_NGRAM_VOCAB_BASE,
                                     DS4_QWEN4_NGRAM_VOCAB_DIVISOR,
                                     DS4_QWEN4_NGRAM_SEED, 0,
                                     DS4_QWEN4_NGRAM_EOS));
    CHECK(hash.multipliers[0] == INT64_C(23703573157769));
    CHECK(hash.multipliers[1] == INT64_C(20109073645365));
    CHECK(hash.multipliers[2] == INT64_C(8052911324071));
    CHECK(hash.vocab[0] == INT64_C(20000003));
    CHECK(hash.vocab[15] == INT64_C(20000171));
    CHECK(hash.offsets[15] == INT64_C(300001275));
    CHECK(hash.total_rows == UINT64_C(320001536));

    const uint32_t previous[] = {
        DS4_QWEN4_NGRAM_EOS, DS4_QWEN4_NGRAM_EOS
    };
    const uint32_t tokens[] = {5, 7, DS4_QWEN4_NGRAM_EOS, 9, 11};
    int64_t rows[5 * DS4_QWEN4_NGRAM_HEADS];
    const int64_t expected[5][DS4_QWEN4_NGRAM_HEADS] = {
        {15389869,39778609,55713969,62213332,88817728,118483999,133731511,155458159,179763390,197956758,205378969,220499474,242466248,265658744,293662119,315720898},
        {12441580,26378836,53347667,75104214,99467174,114254887,126436461,156012011,169119442,187827161,214803956,239809754,242938905,266427765,294337448,314484167},
        {10204458,27984170,41283776,68842151,85621153,118821647,129504214,158727320,176298516,181690702,206665473,238343128,252151767,267018740,285543023,319927855},
        {18043673,37626835,51159316,78294604,94015356,106720349,136526052,144330141,176817901,186368539,203707490,230017629,247662678,266533413,293096193,307951937},
        {10041117,28960672,48420531,71664411,83016360,106800418,122476460,150044571,163654473,184259024,206781966,224776026,248853488,273290488,294849492,303242927},
    };
    CHECK(ds4_qwen4_ngram_row_ids(&hash, previous, 2, tokens, 5,
                                   rows, sizeof(rows) / sizeof(rows[0])));
    CHECK(!memcmp(rows, expected, sizeof(expected)));

    /* Splitting verification immediately before or after EOS must produce
     * the same PLE rows as one uninterrupted target pass. */
    const uint32_t before_eos[] = {5, 7};
    const uint32_t from_eos[] = {DS4_QWEN4_NGRAM_EOS, 9, 11};
    int64_t split_from_eos[3 * DS4_QWEN4_NGRAM_HEADS];
    CHECK(ds4_qwen4_ngram_row_ids(
        &hash, before_eos, 2, from_eos, 3, split_from_eos,
        sizeof(split_from_eos) / sizeof(split_from_eos[0])));
    CHECK(!memcmp(split_from_eos, rows + 2 * DS4_QWEN4_NGRAM_HEADS,
                  sizeof(split_from_eos)));

    const uint32_t after_eos_history[] = {7, DS4_QWEN4_NGRAM_EOS};
    const uint32_t after_eos[] = {9, 11};
    int64_t split_after_eos[2 * DS4_QWEN4_NGRAM_HEADS];
    CHECK(ds4_qwen4_ngram_row_ids(
        &hash, after_eos_history, 2, after_eos, 2, split_after_eos,
        sizeof(split_after_eos) / sizeof(split_after_eos[0])));
    CHECK(!memcmp(split_after_eos, rows + 3 * DS4_QWEN4_NGRAM_HEADS,
                  sizeof(split_after_eos)));
}

static void test_ple_table(void) {
    uint8_t file[4096];
    memset(file, 0, sizeof(file));
    uint64_t weight_offset = 0;
    const size_t file_size =
        build_ple_gguf(file, sizeof(file), DS4_QWEN4_PACK_VERSION_Q4,
                       &weight_offset);
    CHECK(file_size != 0u);
    if (file_size == 0u) return;

    char path[128];
    snprintf(path, sizeof(path), "/tmp/ds4-qwen4-ple-%ld.bin", (long)getpid());
    int fd = open(path, O_CREAT | O_EXCL | O_WRONLY, 0600);
    CHECK(fd >= 0);
    if (fd < 0) return;
    CHECK(write_all(fd, file, file_size));
    CHECK(close(fd) == 0);

    char error[256];
    char digest[65];
    uint64_t file_bytes = 0;
    CHECK(ds4_qwen4_sha256_file(path, digest, &file_bytes,
                                error, sizeof(error)) == 0);
    ds4_qwen4_pack_artifact artifact = {
        .kind = DS4_QWEN4_ARTIFACT_PLE,
        .bytes = file_bytes,
    };
    memcpy(artifact.sha256, digest, sizeof(artifact.sha256));
    ds4_qwen4_ple_table table = {.fd = -1};
    CHECK(ds4_qwen4_pack_validate_ple_open(
              &artifact, path, &table, error, sizeof(error)) == 0);
    CHECK(table.rows == 2);
    CHECK(table.dim == DS4_QWEN4_PLE_ROW_DIM);
    CHECK(table.qtype == 3u);
    CHECK(table.block_size == 32u);
    CHECK(table.blocks_per_row == 5u);
    CHECK(table.row_bytes == 100u);
    CHECK(table.pack_version == DS4_QWEN4_PACK_VERSION_Q4);
    CHECK(!strcmp(table.pack_id, "0123456789abcdef0123456789abcdef"));
    CHECK(!strcmp(table.source_revision, "deadbeef"));
    CHECK(table.data_offset == weight_offset);
    CHECK(table.identity_device != 0 || table.identity_inode != 0);
    CHECK(!strcmp(table.sha256, digest));
    CHECK(table.validation_nocache_requested);
    ds4_qwen4_ple_page_plan page_plan;
    const int64_t plan_rows[] = {1, 0, 1};
    CHECK(ds4_qwen4_ple_plan_pages(
        &table, plan_rows, 3, 16, &page_plan, error, sizeof(error)));
    CHECK(page_plan.page_size == 16);
    CHECK(page_plan.count == 13);
    if (page_plan.count == 13) {
        for (uint32_t i = 0; i < 13u; i++)
            CHECK(page_plan.pages[i] == weight_offset / 16u + i);
    }
    ds4_qwen4_ple_page_plan_free(&page_plan);
#if defined(__APPLE__)
    ds4_qwen4_ple_residency residency;
    CHECK(ds4_qwen4_ple_measure_residency(
        &table, plan_rows, 3, &residency, error, sizeof(error)));
    CHECK(residency.target_pages >= 1);
    CHECK(residency.resident_target_pages + residency.cold_target_pages ==
          residency.target_pages);
    CHECK(residency.identity_stable);
#endif
    const int64_t invalid_plan_row[] = {2};
    CHECK(!ds4_qwen4_ple_plan_pages(
        &table, invalid_plan_row, 1, 16, &page_plan,
        error, sizeof(error)));
    ds4_qwen4_ple_page_plan_free(&page_plan);
    float row[DS4_QWEN4_PLE_ROW_DIM];
    CHECK(ds4_qwen4_ple_row_f32(
        &table, 0, row, DS4_QWEN4_PLE_ROW_DIM));
    for (uint32_t i = 0; i < DS4_QWEN4_PLE_ROW_DIM; i++)
        CHECK(fabsf(row[i] - (1.0f + 0.5f * (float)(i % 16u))) < 1.0e-6f);
    const int64_t indices[] = {1, 0};
    uint16_t gathered[2 * DS4_QWEN4_PLE_ROW_DIM];
    CHECK(ds4_qwen4_ple_gather_bf16(
        &table, indices, 2, gathered,
        sizeof(gathered) / sizeof(gathered[0])));
    for (uint32_t i = 0; i < DS4_QWEN4_PLE_ROW_DIM; i++)
        CHECK(fabsf(bf16_to_f32(gathered[i]) - 5.0f) < 1.0e-6f);
    for (uint32_t i = 0; i < DS4_QWEN4_PLE_ROW_DIM; i++)
        CHECK(fabsf(bf16_to_f32(gathered[DS4_QWEN4_PLE_ROW_DIM + i]) -
                    (1.0f + 0.5f * (float)(i % 16u))) < 1.0e-6f);
    ds4_qwen4_ple_stager stager;
    CHECK(setenv("DS4_QWEN4_PLE_GATHER", "pread", 1) == 0);
    CHECK(ds4_qwen4_ple_stager_init(&stager, &table, 2,
                                     error, sizeof(error)) == 0);
    const int64_t stage0_rows[] = {0, 1};
    const int64_t stage1_rows[] = {1, 0};
    CHECK(ds4_qwen4_ple_stager_submit(&stager, 0, stage0_rows, 2));
    CHECK(ds4_qwen4_ple_stager_submit(&stager, 1, stage1_rows, 2));
    const uint16_t *stage_data = NULL;
    size_t stage_count = 0;
    CHECK(ds4_qwen4_ple_stager_wait(&stager, 1, &stage_data, &stage_count));
    CHECK(stage_count == 2u * DS4_QWEN4_PLE_ROW_DIM);
    for (uint32_t i = 0; i < DS4_QWEN4_PLE_ROW_DIM; i++)
        CHECK(fabsf(bf16_to_f32(stage_data[i]) - 5.0f) < 1.0e-6f);
    CHECK(ds4_qwen4_ple_stager_wait(&stager, 0, &stage_data, &stage_count));
    CHECK(stage_count == 2u * DS4_QWEN4_PLE_ROW_DIM);
    for (uint32_t i = 0; i < DS4_QWEN4_PLE_ROW_DIM; i++)
        CHECK(fabsf(bf16_to_f32(stage_data[i]) -
                    (1.0f + 0.5f * (float)(i % 16u))) < 1.0e-6f);
    ds4_qwen4_ple_stager_destroy(&stager);
    CHECK(unsetenv("DS4_QWEN4_PLE_GATHER") == 0);
    CHECK(!ds4_qwen4_ple_row_f32(
        &table, 2, row, DS4_QWEN4_PLE_ROW_DIM));
    ds4_qwen4_ple_table_close(&table);

    ds4_qwen4_pack_artifact bad = artifact;
    bad.sha256[0] = bad.sha256[0] == '0' ? '1' : '0';
    CHECK(ds4_qwen4_pack_validate_ple_open(
              &bad, path, &table, error, sizeof(error)) != 0);
    CHECK(table.fd == -1 && table.map == NULL);
    /* Failure always leaves a close-safe output, including callers that do
     * unconditional cleanup. */
    ds4_qwen4_ple_table_close(&table);
    CHECK(table.fd == -1 && table.map == NULL);
    memset(&table, 0xa5, sizeof(table));
    CHECK(ds4_qwen4_pack_validate_ple_open(
              NULL, path, &table, error, sizeof(error)) != 0);
    CHECK(table.fd == -1 && table.map == NULL);
    ds4_qwen4_ple_table_close(&table);
    CHECK(ds4_qwen4_ple_table_open(
              &table, "/tmp/ds4-qwen4-no-such-ple", error,
              sizeof(error)) != 0);
    ds4_qwen4_ple_table_close(&table);
    CHECK(table.fd == -1 && table.map == NULL);
    CHECK(ds4_qwen4_ple_table_open(
              &table, path, error, sizeof(error)) == 0);
    ds4_qwen4_ple_table_close(&table);

    /* The GGUF auxiliary vectors are executable hash geometry, not advisory
     * metadata.  Reject a file whose offsets still fit but whose values no
     * longer match the runtime row-ID function. */
    int corrupt_aux = open(path, O_RDWR);
    uint8_t aux_byte = 0u;
    CHECK(corrupt_aux >= 0);
    CHECK(corrupt_aux >= 0 &&
          lseek(corrupt_aux, (off_t)(weight_offset + 224u), SEEK_SET) >= 0);
    CHECK(corrupt_aux >= 0 && read(corrupt_aux, &aux_byte, 1u) == 1);
    aux_byte ^= 1u;
    CHECK(corrupt_aux >= 0 &&
          lseek(corrupt_aux, (off_t)(weight_offset + 224u), SEEK_SET) >= 0);
    CHECK(corrupt_aux >= 0 && write_all(corrupt_aux, &aux_byte, 1u));
    if (corrupt_aux >= 0) CHECK(close(corrupt_aux) == 0);
    CHECK(ds4_qwen4_ple_table_open(
              &table, path, error, sizeof(error)) != 0);
    CHECK(table.fd == -1 && table.map == NULL);
    ds4_qwen4_ple_table_close(&table);
    CHECK(unlink(path) == 0);
}

static void test_ple_v4_header(void) {
    uint8_t file[4096] = {0};
    const size_t file_size = build_ple_gguf(
        file, sizeof(file), DS4_QWEN4_PACK_VERSION_Q2, NULL);
    CHECK(file_size != 0u);
    if (file_size == 0u) return;

    char path[128];
    snprintf(path, sizeof(path), "/tmp/ds4-qwen4-ple-v4-%ld.bin",
             (long)getpid());
    int fd = open(path, O_CREAT | O_EXCL | O_WRONLY, 0600);
    CHECK(fd >= 0 && write_all(fd, file, file_size));
    if (fd >= 0) CHECK(close(fd) == 0);

    char error[256], digest[65];
    uint64_t bytes = 0u;
    CHECK(ds4_qwen4_sha256_file(path, digest, &bytes,
                                error, sizeof(error)) == 0);
    ds4_qwen4_pack_artifact artifact = {
        .kind = DS4_QWEN4_ARTIFACT_PLE,
        .bytes = bytes,
    };
    memcpy(artifact.sha256, digest, sizeof(artifact.sha256));
    ds4_qwen4_ple_table table = {.fd = -1};
    CHECK(ds4_qwen4_pack_validate_ple_open(
              &artifact, path, &table, error, sizeof(error)) == 0);
    CHECK(table.rows == 2u && table.row_bytes == 100u);
    CHECK(table.pack_version == DS4_QWEN4_PACK_VERSION_Q2);
    ds4_qwen4_ple_table_close(&table);
    CHECK(unlink(path) == 0);
}

static void test_ple_mmap_stager(void) {
    const long system_page = sysconf(_SC_PAGESIZE);
    CHECK(system_page > 0);
    if (system_page <= 0) return;
    const size_t page = (size_t)system_page;
    CHECK(page > 128u && page <= SIZE_MAX / 4u);
    if (page <= 128u || page > SIZE_MAX / 4u) return;
    const size_t file_size = page * 4u;
    uint8_t *file = calloc(file_size, 1u);
    CHECK(file != NULL);
    if (!file) return;

    enum { ROWS = 4, DIM = 32, ROW_BYTES = 20 };
    const uint64_t weight_offset = page - 8u;
    for (uint32_t row = 0; row < ROWS; row++) {
        uint8_t *block = file + weight_offset + row * ROW_BYTES;
        store_u16_le(block,
                     row & 1u ? UINT16_C(0x3800) : UINT16_C(0x3c00));
        store_u16_le(block + 2u,
                     row & 1u ? UINT16_C(0xbc00) : UINT16_C(0x3800));
        for (uint32_t i = 0; i < 16u; i++) {
            const uint8_t low = (uint8_t)((row * 5u + i * 7u) & 15u);
            const uint8_t high =
                (uint8_t)((row * 5u + (i + 16u) * 7u) & 15u);
            block[4u + i] = (uint8_t)(low | (high << 4));
        }
    }

    char path[160];
    snprintf(path, sizeof(path), "/tmp/ds4-qwen4-ple-boundary-%ld.bin",
             (long)getpid());
    const int fd = open(path, O_CREAT | O_EXCL | O_RDWR, 0600);
    CHECK(fd >= 0);
    if (fd < 0) {
        free(file);
        return;
    }
    CHECK(write_all(fd, file, file_size));
    free(file);
    uint8_t *map = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    CHECK(map != MAP_FAILED);
    if (map == MAP_FAILED) {
        CHECK(close(fd) == 0);
        CHECK(unlink(path) == 0);
        return;
    }

    ds4_qwen4_ple_table table = {
        .fd = fd,
        .map = map,
        .size = file_size,
        .rows = ROWS,
        .dim = DIM,
        .qtype = 3,
        .block_size = 32,
        .blocks_per_row = 1,
        .row_bytes = ROW_BYTES,
        .data_offset = weight_offset,
    };
    const int64_t rows[] = {3, 0, 2, 3, 1, 0, 1, 2};
    enum { GATHER_ROWS = sizeof(rows) / sizeof(rows[0]) };
    uint16_t reference[GATHER_ROWS * DIM] = {0};
    uint16_t pread_result[GATHER_ROWS * DIM] = {0};
    CHECK(ds4_qwen4_ple_gather_bf16(
        &table, rows, GATHER_ROWS, reference,
        sizeof(reference) / sizeof(reference[0])));

    char error[256] = {0};
    ds4_qwen4_ple_stager stager;
    ds4_qwen4_ple_stager_stats stats;
    CHECK(unsetenv("DS4_QWEN4_PLE_GATHER") == 0);
    CHECK(ds4_qwen4_ple_stager_init(
        &stager, &table, GATHER_ROWS, error, sizeof(error)) == 0);
    CHECK(ds4_qwen4_ple_stager_snapshot(&stager, &stats));
    CHECK(stats.mode == DS4_QWEN4_PLE_GATHER_PREAD);
    CHECK(!stats.random_advice_requested);
    CHECK(!stats.random_advice_succeeded);
    CHECK(ds4_qwen4_ple_stager_submit(&stager, 0, rows, GATHER_ROWS));
    const uint16_t *stage_data = NULL;
    size_t stage_count = 0;
    CHECK(ds4_qwen4_ple_stager_wait(
        &stager, 0, &stage_data, &stage_count));
    CHECK(stage_count == GATHER_ROWS * DIM);
    if (stage_data && stage_count == GATHER_ROWS * DIM)
        memcpy(pread_result, stage_data, sizeof(pread_result));
    CHECK(!memcmp(reference, pread_result, sizeof(reference)));
    ds4_qwen4_ple_stager_destroy(&stager);

    CHECK(setenv("DS4_QWEN4_PLE_GATHER", "mmap", 1) == 0);
    CHECK(ds4_qwen4_ple_stager_init(
        &stager, &table, GATHER_ROWS, error, sizeof(error)) == 0);
    CHECK(ds4_qwen4_ple_stager_snapshot(&stager, &stats));
    CHECK(stats.mode == DS4_QWEN4_PLE_GATHER_MMAP);
    CHECK(stats.random_advice_requested);
    CHECK(ds4_qwen4_ple_stager_submit(&stager, 0, rows, GATHER_ROWS));
    const int64_t alternate_rows[] = {0, 3};
    CHECK(ds4_qwen4_ple_stager_submit(&stager, 1, alternate_rows, 2));
    CHECK(ds4_qwen4_ple_stager_wait(
        &stager, 0, &stage_data, &stage_count));
    CHECK(stage_count == GATHER_ROWS * DIM);
    CHECK(stage_data != NULL);
    if (stage_data)
        CHECK(!memcmp(reference, stage_data, sizeof(reference)));
    CHECK(ds4_qwen4_ple_stager_wait(
        &stager, 1, &stage_data, &stage_count));
    uint16_t alternate_reference[2 * DIM] = {0};
    CHECK(ds4_qwen4_ple_gather_bf16(
        &table, alternate_rows, 2, alternate_reference,
        sizeof(alternate_reference) / sizeof(alternate_reference[0])));
    CHECK(stage_count == 2u * DIM);
    CHECK(stage_data != NULL);
    if (stage_data)
        CHECK(!memcmp(alternate_reference, stage_data,
                      sizeof(alternate_reference)));

    const int64_t invalid_negative[] = {-1};
    const int64_t invalid_high[] = {ROWS};
    uint16_t invalid_output[DIM] = {0};
    CHECK(!ds4_qwen4_ple_gather_bf16(
        &table, invalid_negative, 1, invalid_output, DIM));
    CHECK(!ds4_qwen4_ple_gather_bf16(
        &table, invalid_high, 1, invalid_output, DIM));
    CHECK(ds4_qwen4_ple_stager_submit(&stager, 0, invalid_negative, 1));
    CHECK(!ds4_qwen4_ple_stager_wait(
        &stager, 0, &stage_data, &stage_count));
    CHECK(ds4_qwen4_ple_stager_submit(&stager, 0, invalid_high, 1));
    CHECK(!ds4_qwen4_ple_stager_wait(
        &stager, 0, &stage_data, &stage_count));
    CHECK(ds4_qwen4_ple_stager_submit(&stager, 0, alternate_rows, 2));
    CHECK(ds4_qwen4_ple_stager_wait(
        &stager, 0, &stage_data, &stage_count));
    CHECK(stage_count == 2u * DIM);
    CHECK(stage_data != NULL);
    if (stage_data)
        CHECK(!memcmp(alternate_reference, stage_data,
                      sizeof(alternate_reference)));
    CHECK(ds4_qwen4_ple_stager_snapshot(&stager, &stats));
    CHECK(stats.gather_count == 5u);
    CHECK(stats.gather_rows == GATHER_ROWS + 6u);
    CHECK(stats.gather_seconds >= 0.0);
    CHECK(stats.wait_seconds >= 0.0);
    ds4_qwen4_ple_stager_destroy(&stager);

    /* Destroying with both slots queued is the stager's cancellation and
     * shutdown boundary: workers either finish their immutable reads or exit
     * without publishing a partial buffer. */
    CHECK(ds4_qwen4_ple_stager_init(
        &stager, &table, GATHER_ROWS, error, sizeof(error)) == 0);
    CHECK(ds4_qwen4_ple_stager_submit(&stager, 0, rows, GATHER_ROWS));
    CHECK(ds4_qwen4_ple_stager_submit(&stager, 1, rows, GATHER_ROWS));
    ds4_qwen4_ple_stager_destroy(&stager);
    CHECK(stager.impl == NULL);

    CHECK(setenv("DS4_QWEN4_PLE_GATHER", "invalid", 1) == 0);
    memset(&stager, 0xa5, sizeof(stager));
    CHECK(ds4_qwen4_ple_stager_init(
        &stager, &table, GATHER_ROWS, error, sizeof(error)) != 0);
    CHECK(stager.impl == NULL);
    CHECK(strstr(error, "expected pread or mmap") != NULL);
    CHECK(unsetenv("DS4_QWEN4_PLE_GATHER") == 0);

    CHECK(munmap(map, file_size) == 0);
    CHECK(close(fd) == 0);
    CHECK(unlink(path) == 0);
}

enum {
    QWEN4_TEST_LEGACY_BASE_COUNT = 4,
    QWEN4_TEST_PLE_FILE = 4,
    QWEN4_TEST_VISION_FILE = 5,
    QWEN4_TEST_MTP_FILE = 6,
    QWEN4_TEST_CANONICAL_BASE_FILE = 7,
    QWEN4_TEST_FILE_COUNT = 8,
};

static const char QWEN4_TEST_TENSORS_JSON[] =
    "{\"language_model.model.embed_tokens.weight\":{"
    "\"artifact\":\"Qwen3.8-Flash-Next-Q4KExperts-BF16Emb-BF16Control-"
    "Q8GDN-Q8QSA-Q8Shared-Q8Out.gguf\","
    "\"logical_shape\":[248320,2560],"
    "\"physical_shape\":[248320,2560],"
    "\"qtype\":\"BF16\","
    "\"sha256\":\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\","
    "\"source\":\"model.language_model.embed_tokens.weight\"}}";

static const char QWEN4_TEST_TENSORS_SHA256[] =
    "14d266a3a7a883a42b184f2926071a8ecd5c1c4b9dba8ea9b47bbcf02cb1b33e";

static bool write_pack_manifest_with_tensors(
        const char *manifest_path,
        const char *const names[QWEN4_TEST_FILE_COUNT],
        char digests[QWEN4_TEST_FILE_COUNT][65],
        const uint64_t sizes[QWEN4_TEST_FILE_COUNT],
        uint32_t version,
        bool include_optional,
        const char *tensors_json,
        const char *tensor_digest) {
    const bool legacy = version == 1u;
    if (version < 1u || version > DS4_QWEN4_PACK_LATEST_VERSION ||
        !tensors_json || !tensor_digest) return false;
    const uint32_t base_count = legacy ? QWEN4_TEST_LEGACY_BASE_COUNT : 1u;
    FILE *fp = fopen(manifest_path, "wb");
    if (!fp) return false;
    bool ok = fprintf(
        fp,
        "{\"architecture\":\"qwen4-exp\",\"artifacts\":[") >= 0;
    for (uint32_t shard = 0; ok && shard < base_count; shard++) {
        const uint32_t file = legacy ? shard : QWEN4_TEST_CANONICAL_BASE_FILE;
        if (legacy) {
            ok = fprintf(
                fp,
                "%s{\"bytes\":%llu,\"kind\":\"base\",\"path\":\"%s\","
                "\"sha256\":\"%s\",\"shard_index\":%u}",
                shard == 0u ? "" : ",",
                (unsigned long long)sizes[file], names[file], digests[file],
                shard) >= 0;
        } else {
            /* Canonical v3 manifests omit shard_index for the sole base. */
            ok = fprintf(
                fp,
                "{\"bytes\":%llu,\"kind\":\"base\",\"path\":\"%s\","
                "\"sha256\":\"%s\"}",
                (unsigned long long)sizes[file], names[file], digests[file]) >= 0;
        }
    }
    ok = ok && fprintf(
        fp,
        ",{\"bytes\":%llu,\"kind\":\"ple\",\"path\":\"%s\",\"sha256\":\"%s\"}",
        (unsigned long long)sizes[QWEN4_TEST_PLE_FILE],
        names[QWEN4_TEST_PLE_FILE], digests[QWEN4_TEST_PLE_FILE]) >= 0;
    if (include_optional) {
        ok = ok && fprintf(
            fp,
            ",{\"bytes\":%llu,\"kind\":\"vision\",\"path\":\"%s\",\"sha256\":\"%s\"}"
            ",{\"bytes\":%llu,\"kind\":\"mtp\",\"path\":\"%s\",\"sha256\":\"%s\"}",
            (unsigned long long)sizes[QWEN4_TEST_VISION_FILE],
            names[QWEN4_TEST_VISION_FILE], digests[QWEN4_TEST_VISION_FILE],
            (unsigned long long)sizes[QWEN4_TEST_MTP_FILE],
            names[QWEN4_TEST_MTP_FILE], digests[QWEN4_TEST_MTP_FILE]) >= 0;
    }
    ok = ok && fprintf(
        fp,
        "],\"geometry\":{\"tensors\":{\"decoy\":{"
        "\"text\":\"escaped \\\" brace }\"}}},"
        "\"pack_id\":\"0123456789abcdef0123456789abcdef\","
        "\"quantization\":{\"control\":{\"qtype\":\"source\"}},"
        "\"schema\":\"ds4.qwen4.fast-pack\","
        "\"source\":{\"repository\":\"Qwen/Qwen3.8-Flash-Next\",\"revision\":\"deadbeef\"},"
        "\"tensor_manifest_sha256\":\"%s\",\"tensors\":%s,"
        "\"version\":%u}\n", tensor_digest, tensors_json, version) >= 0;
    return fclose(fp) == 0 && ok;
}

static bool write_pack_manifest(
        const char *manifest_path,
        const char *const names[QWEN4_TEST_FILE_COUNT],
        char digests[QWEN4_TEST_FILE_COUNT][65],
        const uint64_t sizes[QWEN4_TEST_FILE_COUNT],
        uint32_t version,
        bool include_optional) {
    return write_pack_manifest_with_tensors(
        manifest_path, names, digests, sizes, version, include_optional,
        QWEN4_TEST_TENSORS_JSON, QWEN4_TEST_TENSORS_SHA256);
}

static void test_artifact_tensor_counts(void) {
    CHECK(DS4_QWEN4_VISION_TENSOR_COUNT == 333u);
    CHECK(ds4_qwen4_artifact_tensor_count_valid(
        DS4_QWEN4_ARTIFACT_BASE, DS4_QWEN4_BASE_TENSOR_COUNT));
    CHECK(!ds4_qwen4_artifact_tensor_count_valid(
        DS4_QWEN4_ARTIFACT_BASE, DS4_QWEN4_BASE_TENSOR_COUNT + 1u));
    CHECK(ds4_qwen4_artifact_tensor_count_valid(
        DS4_QWEN4_ARTIFACT_MTP, DS4_QWEN4_MTP_TENSOR_COUNT));
    CHECK(!ds4_qwen4_artifact_tensor_count_valid(
        DS4_QWEN4_ARTIFACT_MTP, DS4_QWEN4_MTP_TENSOR_COUNT + 1u));
    CHECK(ds4_qwen4_artifact_tensor_count_valid(
        DS4_QWEN4_ARTIFACT_PLE, DS4_QWEN4_PLE_TENSOR_COUNT));
    CHECK(ds4_qwen4_artifact_tensor_count_valid(
        DS4_QWEN4_ARTIFACT_VISION, DS4_QWEN4_VISION_TENSOR_COUNT));
    CHECK(!ds4_qwen4_artifact_tensor_count_valid(
        DS4_QWEN4_ARTIFACT_VISION,
        DS4_QWEN4_VISION_TENSOR_COUNT + 1u));
}

static void test_pack_manifest(void) {
    char dir[] = "/tmp/ds4-qwen4-manifest-XXXXXX";
    char *created = mkdtemp(dir);
    CHECK(created != NULL);
    if (!created) return;
    static const char *names[QWEN4_TEST_FILE_COUNT] = {
        "qwen3.8-flash-next-q4-00001-of-00004.gguf",
        "qwen3.8-flash-next-q4-00002-of-00004.gguf",
        "qwen3.8-flash-next-q4-00003-of-00004.gguf",
        "qwen3.8-flash-next-q4-00004-of-00004.gguf",
        "Qwen3.8-Flash-Next-PLE-Q4_1.gguf",
        "qwen3.8-flash-next-q4-vision.gguf",
        "qwen3.8-flash-next-q4-mtp.gguf",
        ("Qwen3.8-Flash-Next-Q4KExperts-BF16Emb-BF16Control-Q8GDN-"
         "Q8QSA-Q8Shared-Q8Out.gguf"),
    };
    char paths[QWEN4_TEST_FILE_COUNT][256];
    char digests[QWEN4_TEST_FILE_COUNT][65];
    uint64_t sizes[QWEN4_TEST_FILE_COUNT];
    char error[256];
    for (uint32_t i = 0; i < QWEN4_TEST_FILE_COUNT; i++) {
        snprintf(paths[i], sizeof(paths[i]), "%s/%s", dir, names[i]);
        int fd = open(paths[i], O_CREAT | O_EXCL | O_WRONLY, 0600);
        CHECK(fd >= 0);
        char payload[3] = {(char)('a' + i), (char)('0' + i), '\n'};
        CHECK(fd >= 0 && write_all(fd, payload, sizeof(payload)));
        if (fd >= 0) CHECK(close(fd) == 0);
        CHECK(ds4_qwen4_sha256_file(paths[i], digests[i], &sizes[i],
                                    error, sizeof(error)) == 0);
        CHECK(sizes[i] == sizeof(payload));
    }

    char abc_path[256], abc_digest[65];
    snprintf(abc_path, sizeof(abc_path), "%s/abc", dir);
    int abc = open(abc_path, O_CREAT | O_EXCL | O_WRONLY, 0600);
    CHECK(abc >= 0 && write_all(abc, "abc", 3));
    if (abc >= 0) CHECK(close(abc) == 0);
    CHECK(ds4_qwen4_sha256_file(abc_path, abc_digest, NULL,
                                error, sizeof(error)) == 0);
    CHECK(!strcmp(abc_digest,
                  "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));

    char manifest_path[256];
    snprintf(manifest_path, sizeof(manifest_path), "%s/%s", dir,
             DS4_QWEN4_PACK_MANIFEST_FILE);

    /* Canonical v3 packs contain one base GGUF and may omit shard_index. */
    CHECK(write_pack_manifest(manifest_path, names, digests, sizes,
                              DS4_QWEN4_PACK_VERSION_Q4, true));
    ds4_qwen4_pack_manifest canonical;
    CHECK(ds4_qwen4_pack_manifest_load(&canonical, manifest_path,
                                        error, sizeof(error)) == 0);
    CHECK(canonical.version == DS4_QWEN4_PACK_VERSION_Q4);
    CHECK(canonical.base_shard_count == DS4_QWEN4_PACK_BASE_SHARDS);
    CHECK(canonical.base_shard_count == 1u);
    CHECK(canonical.artifact_count == 4u);
    CHECK(!strcmp(canonical.source_revision, "deadbeef"));
    CHECK(!strcmp(canonical.tensor_manifest_sha256,
                  QWEN4_TEST_TENSORS_SHA256));
    CHECK(canonical.profile == DS4_QWEN4_PACK_PROFILE_Q4_K);

    /* v4 keeps the authenticated artifact topology but selects the fixed
     * IQ2_XXS gate/up + Q2_K down profile. */
    CHECK(write_pack_manifest(manifest_path, names, digests, sizes,
                              DS4_QWEN4_PACK_VERSION_Q2, true));
    ds4_qwen4_pack_manifest q2;
    CHECK(ds4_qwen4_pack_manifest_load(&q2, manifest_path,
                                        error, sizeof(error)) == 0);
    CHECK(q2.version == DS4_QWEN4_PACK_VERSION_Q2);
    CHECK(q2.profile == DS4_QWEN4_PACK_PROFILE_IQ2_XXS_Q2_K);
    CHECK(q2.base_shard_count == 1u && q2.artifact_count == 4u);

    char changed_tensors[sizeof(QWEN4_TEST_TENSORS_JSON)];
    memcpy(changed_tensors, QWEN4_TEST_TENSORS_JSON,
           sizeof(changed_tensors));
    char *changed_qtype = strstr(changed_tensors, "\"qtype\":\"BF16\"");
    CHECK(changed_qtype != NULL);
    if (changed_qtype) changed_qtype[9] = 'X';
    CHECK(write_pack_manifest_with_tensors(
        manifest_path, names, digests, sizes,
        DS4_QWEN4_PACK_VERSION_Q4, true,
        changed_tensors, QWEN4_TEST_TENSORS_SHA256));
    ds4_qwen4_pack_manifest invalid_tensor_manifest;
    CHECK(ds4_qwen4_pack_manifest_load(
              &invalid_tensor_manifest, manifest_path,
              error, sizeof(error)) != 0);
    CHECK(strstr(error, "tensor manifest SHA-256 mismatch") != NULL);
    CHECK(invalid_tensor_manifest.version == 0u &&
          invalid_tensor_manifest.artifact_count == 0u);

    char truncated_tensors[sizeof(QWEN4_TEST_TENSORS_JSON)];
    memcpy(truncated_tensors, QWEN4_TEST_TENSORS_JSON,
           sizeof(truncated_tensors));
    truncated_tensors[sizeof(truncated_tensors) - 2u] = '\0';
    CHECK(write_pack_manifest_with_tensors(
        manifest_path, names, digests, sizes,
        DS4_QWEN4_PACK_VERSION_Q4, true,
        truncated_tensors, QWEN4_TEST_TENSORS_SHA256));
    CHECK(ds4_qwen4_pack_manifest_load(
              &invalid_tensor_manifest, manifest_path,
              error, sizeof(error)) != 0);
    CHECK(strstr(error, "malformed top-level tensors object") != NULL);
    CHECK(invalid_tensor_manifest.version == 0u &&
          invalid_tensor_manifest.artifact_count == 0u);

    CHECK(write_pack_manifest(manifest_path, names, digests, sizes,
                              DS4_QWEN4_PACK_VERSION_Q4, true));
    uint32_t validation_mask = 0u;
    CHECK(ds4_qwen4_pack_validation_mask(
              &canonical, false, false, &validation_mask,
              error, sizeof(error)) == 0);
    CHECK(validation_mask == UINT32_C(0x03));
    CHECK(ds4_qwen4_pack_validation_mask(
              &canonical, true, true, &validation_mask,
              error, sizeof(error)) == 0);
    CHECK(validation_mask == UINT32_C(0x0f));
    const ds4_qwen4_pack_artifact *base = ds4_qwen4_pack_find_artifact(
        &canonical, DS4_QWEN4_ARTIFACT_BASE, 0u);
    CHECK(base != NULL &&
          !strcmp(base->path, names[QWEN4_TEST_CANONICAL_BASE_FILE]));
    CHECK(ds4_qwen4_pack_find_artifact(
              &canonical, DS4_QWEN4_ARTIFACT_BASE, 1u) == NULL);
    CHECK(ds4_qwen4_pack_validate_artifact(
              base, paths[QWEN4_TEST_CANONICAL_BASE_FILE],
              error, sizeof(error)) == 0);
    CHECK(ds4_qwen4_pack_validate_artifact(
              &canonical.artifacts[1], paths[QWEN4_TEST_PLE_FILE],
              error, sizeof(error)) == 0);

    int corrupt = open(paths[QWEN4_TEST_CANONICAL_BASE_FILE],
                       O_WRONLY | O_APPEND);
    CHECK(corrupt >= 0 && write_all(corrupt, "x", 1));
    if (corrupt >= 0) CHECK(close(corrupt) == 0);
    CHECK(ds4_qwen4_pack_validate_artifact(
              base, paths[QWEN4_TEST_CANONICAL_BASE_FILE],
              error, sizeof(error)) != 0);
    int repair = open(paths[QWEN4_TEST_CANONICAL_BASE_FILE],
                      O_WRONLY | O_TRUNC);
    const char repaired_payload[3] = {'h', '7', '\n'};
    CHECK(repair >= 0 && write_all(repair, repaired_payload,
                                   sizeof(repaired_payload)));
    if (repair >= 0) CHECK(close(repair) == 0);
    CHECK(ds4_qwen4_pack_validate_artifact(
              base, paths[QWEN4_TEST_CANONICAL_BASE_FILE],
              error, sizeof(error)) == 0);

    const ds4_qwen4_pack_artifact *vision = ds4_qwen4_pack_find_artifact(
        &canonical, DS4_QWEN4_ARTIFACT_VISION, 0u);
    const ds4_qwen4_pack_artifact *mtp = ds4_qwen4_pack_find_artifact(
        &canonical, DS4_QWEN4_ARTIFACT_MTP, 0u);
    CHECK(vision != NULL && mtp != NULL);
    CHECK(ds4_qwen4_pack_validate_artifact(
              vision, paths[QWEN4_TEST_VISION_FILE],
              error, sizeof(error)) == 0);
    CHECK(ds4_qwen4_pack_validate_artifact(
              mtp, paths[QWEN4_TEST_MTP_FILE],
              error, sizeof(error)) == 0);
    CHECK(ds4_qwen4_pack_validate_artifact(
              mtp, paths[QWEN4_TEST_VISION_FILE],
              error, sizeof(error)) != 0);

    /* Legacy v2 single-base packs and v1 four-shard packs are deliberately
     * rejected: the runtime admits only the standard-block v3 contract. */
    CHECK(write_pack_manifest(manifest_path, names, digests, sizes,
                              2u, true));
    ds4_qwen4_pack_manifest legacy;
    CHECK(ds4_qwen4_pack_manifest_load(&legacy, manifest_path,
                                        error, sizeof(error)) != 0);
    CHECK(legacy.version == 0u && legacy.artifact_count == 0u);
    CHECK(write_pack_manifest(manifest_path, names, digests, sizes,
                              1u, true));
    CHECK(ds4_qwen4_pack_manifest_load(&legacy, manifest_path,
                                        error, sizeof(error)) != 0);
    CHECK(legacy.version == 0u && legacy.artifact_count == 0u);

    CHECK(unlink(paths[QWEN4_TEST_MTP_FILE]) == 0);
    CHECK(ds4_qwen4_pack_validate_artifact(
              mtp, paths[QWEN4_TEST_MTP_FILE],
              error, sizeof(error)) != 0);

    /* Optional manifest entries are not selected for text-only startup, so
     * absent vision/MTP files cannot fail validation. */
    CHECK(unlink(paths[QWEN4_TEST_VISION_FILE]) == 0);
    CHECK(ds4_qwen4_pack_validation_mask(
              &canonical, false, false, &validation_mask,
              error, sizeof(error)) == 0);
    CHECK(validation_mask == UINT32_C(0x03));

    /* v3 may omit optional artifacts entirely; requesting a corresponding
     * CLI sidecar must still fail selection. */
    CHECK(write_pack_manifest(manifest_path, names, digests, sizes,
                              DS4_QWEN4_PACK_VERSION_Q4, false));
    ds4_qwen4_pack_manifest canonical_text;
    CHECK(ds4_qwen4_pack_manifest_load(&canonical_text, manifest_path,
                                        error, sizeof(error)) == 0);
    CHECK(canonical_text.base_shard_count == 1u);
    CHECK(canonical_text.artifact_count == 2u);
    CHECK(ds4_qwen4_pack_validation_mask(
              &canonical_text, false, false, &validation_mask,
              error, sizeof(error)) == 0);
    CHECK(validation_mask == UINT32_C(0x03));
    CHECK(ds4_qwen4_pack_validation_mask(
              &canonical_text, true, false, &validation_mask,
              error, sizeof(error)) != 0);
    CHECK(validation_mask == 0u);
    CHECK(strstr(error, "--vision") != NULL);
    CHECK(ds4_qwen4_pack_validation_mask(
              &canonical_text, false, true, &validation_mask,
              error, sizeof(error)) != 0);
    CHECK(validation_mask == 0u);
    CHECK(strstr(error, "--mtp-model") != NULL);

    CHECK(unlink(manifest_path) == 0);
    CHECK(unlink(abc_path) == 0);
    for (uint32_t i = 0; i <= QWEN4_TEST_PLE_FILE; i++)
        CHECK(unlink(paths[i]) == 0);
    CHECK(unlink(paths[QWEN4_TEST_CANONICAL_BASE_FILE]) == 0);
    CHECK(rmdir(dir) == 0);
}

int main(void) {
    test_pack_profiles();
    test_prefill_policy();
    test_mrope_positions();
    test_qwen4_image_preprocess();
    test_ngram_hash();
    test_ple_table();
    test_ple_v4_header();
    test_ple_mmap_stager();
    test_artifact_tensor_counts();
    test_pack_manifest();
    if (failures != 0) {
        fprintf(stderr, "qwen4 host tests: %d failure(s)\n", failures);
        return 1;
    }
    puts("qwen4 host tests: ok");
    return 0;
}
