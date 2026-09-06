#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

/* Primitive oracle for the ds4.c leading-indexer-query slice. This does not
 * call the private graph admission gate: full-model same-binary tests must
 * separately cover that gate and compressor/cache continuation state. */
enum {
    MAX_ROWS = 97, Q_RANK = 1024, EMBED = 4096, HEADS = 64,
    INDEX_DIM = 128, QUERY_WIDTH = HEADS * INDEX_DIM,
    ATTN_DIM = 512, ATTN_WIDTH = HEADS * ATTN_DIM,
    TOP_K = 512, RAW_CAP = 256, WINDOW = 128, GUARD = 16,
};
static const uint32_t poison = UINT32_C(0x7fc12345);

typedef struct {
    ds4_gpu_tensor *storage, *data;
    size_t words;
} guarded_tensor;

static bool allocate_tensor(guarded_tensor *t, size_t words) {
    t->words = words;
    t->storage = ds4_gpu_tensor_alloc((words + 2u * GUARD) * sizeof(uint32_t));
    t->data = t->storage ? ds4_gpu_tensor_view(t->storage,
        GUARD * sizeof(uint32_t), words * sizeof(uint32_t)) : NULL;
    return t->data != NULL;
}

static void free_tensor(guarded_tensor *t) {
    ds4_gpu_tensor_free(t->data);
    ds4_gpu_tensor_free(t->storage);
}

static bool reset_tensor(const guarded_tensor *t, uint32_t *scratch) {
    for (size_t i = 0; i < t->words + 2u * GUARD; ++i) scratch[i] = poison;
    return ds4_gpu_tensor_write(t->storage, 0, scratch,
        (t->words + 2u * GUARD) * sizeof(uint32_t)) != 0;
}

static ds4_gpu_tensor *row_view(const guarded_tensor *t,
                               uint32_t first, uint32_t rows, uint32_t width) {
    return ds4_gpu_tensor_view(t->data, (uint64_t)first * width * sizeof(float),
                              (uint64_t)rows * width * sizeof(float));
}

/* Compare exact active words, poison before the sliced rows and after the
 * logical shape, plus storage guards. Exponent tests survive -ffast-math. */
static bool check_stage(const char *name, const guarded_tensor *t,
                        uint32_t *scratch, uint32_t *reference,
                        uint32_t rows, uint32_t width, uint32_t first,
                        bool candidate, bool finite, bool identity_prefix) {
    if (!ds4_gpu_tensor_read(t->storage, 0, scratch,
            (t->words + 2u * GUARD) * sizeof(uint32_t))) return false;
    const size_t live = (size_t)rows * width;
    const size_t skip = candidate ? (size_t)first * width : 0;
    for (size_t i = 0; i < t->words + 2u * GUARD; ++i) {
        const bool data = i >= GUARD && i < GUARD + live;
        const size_t j = i >= GUARD ? i - GUARD : 0;
        uint32_t expected = poison;
        bool exact = !data || j < skip;
        if (data && j < skip && identity_prefix) expected = (uint32_t)(j % width);
        if (data && j >= skip && candidate) {
            expected = reference[j];
            exact = true;
        }
        if (exact && scratch[i] != expected) {
            fprintf(stderr, "%s word=%zu expected=%08x actual=%08x\n",
                    name, i, expected, scratch[i]);
            return false;
        }
        if (data && j >= skip && finite &&
            (scratch[i] & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
            fprintf(stderr, "%s nonfinite word=%zu bits=%08x\n", name, i, scratch[i]);
            return false;
        }
    }
    if (!candidate) memcpy(reference, scratch + GUARD, live * sizeof(uint32_t));
    return true;
}

static float pattern(size_t i, unsigned salt, float scale) {
    return (float)((int)((i * 29u + salt) % 127u) - 63) * scale;
}

int main(void) {
    if (!ds4_gpu_init()) return 1;
    if (!ds4_gpu_device_is_pre_m5_apple_silicon()) {
        fprintf(stderr, "leading-indexer-query oracle requires pre-M5 Apple Silicon\n");
        ds4_gpu_cleanup();
        return 77;
    }
    ds4_gpu_set_quality(false);
    const long system_page = sysconf(_SC_PAGESIZE);
    const size_t page = system_page > 0 ? (size_t)system_page : 0;
    const size_t query_offset = page;
    const size_t weight_offset = query_offset + (size_t)Q_RANK * QUERY_WIDTH * 2u;
    const size_t model_bytes = weight_offset + (size_t)EMBED * HEADS * 2u;
    const size_t max_words = (size_t)MAX_ROWS * ATTN_WIDTH;
    void *model = NULL;
    float *input = malloc(max_words * sizeof(float));
    uint32_t *scratch = malloc((max_words + 2u * GUARD) * sizeof(uint32_t));
    const size_t stage_words[] = {
        (size_t)MAX_ROWS * QUERY_WIDTH, (size_t)MAX_ROWS * QUERY_WIDTH,
        (size_t)MAX_ROWS * QUERY_WIDTH, (size_t)MAX_ROWS * HEADS,
        (size_t)MAX_ROWS * 537u, (size_t)MAX_ROWS * TOP_K,
        (size_t)MAX_ROWS * TOP_K,
        (size_t)MAX_ROWS * ATTN_WIDTH,
    };
    uint32_t *reference[8] = {NULL};
    guarded_tensor source_q = {0}, source_weights = {0}, query = {0}, weights = {0};
    guarded_tensor scores = {0}, selected = {0}, sorted = {0}, attn = {0};
    ds4_gpu_tensor *index_cache = NULL, *attn_q = NULL, *raw = NULL, *comp = NULL;
    bool ok = page != 0 && input && scratch &&
        posix_memalign(&model, page, model_bytes) == 0;
    for (unsigned i = 0; ok && i < 8; ++i) {
        reference[i] = malloc(stage_words[i] * sizeof(uint32_t));
        ok = reference[i] != NULL;
    }
    if (ok) ok = allocate_tensor(&source_q, (size_t)MAX_ROWS * Q_RANK) &&
        allocate_tensor(&source_weights, (size_t)MAX_ROWS * EMBED) &&
        allocate_tensor(&query, stage_words[0]) &&
        allocate_tensor(&weights, stage_words[3]) &&
        allocate_tensor(&scores, stage_words[4]) &&
        allocate_tensor(&selected, stage_words[5]) &&
        allocate_tensor(&sorted, stage_words[5]) &&
        allocate_tensor(&attn, stage_words[7]);
    if (!ok) goto cleanup;
    memset(model, 0, model_bytes);
    for (unsigned h = 0; h < HEADS; ++h)
        ((float *)model)[h] = (float)((int)(h % 3u) - 1) * 80.0f;
    _Float16 *q_matrix = (_Float16 *)((char *)model + query_offset);
    _Float16 *w_matrix = (_Float16 *)((char *)model + weight_offset);
    for (size_t i = 0; i < (size_t)Q_RANK * QUERY_WIDTH; ++i)
        q_matrix[i] = (_Float16)pattern(i, 17, 1.0f / 2048.0f);
    for (size_t i = 0; i < (size_t)EMBED * HEADS; ++i)
        w_matrix[i] = (_Float16)pattern(i, 43, 1.0f / 4096.0f);
    ok = ds4_gpu_set_model_map(model, model_bytes) != 0;
    const guarded_tensor *sources[] = {&source_q, &source_weights};
    for (unsigned s = 0; ok && s < 2; ++s) {
        for (size_t i = 0; i < sources[s]->words; ++i)
            input[i] = pattern(i, 11u + 31u * s, 1.0f / 128.0f);
        ok = reset_tensor(sources[s], scratch) &&
            ds4_gpu_tensor_write(sources[s]->data, 0, input,
                                  sources[s]->words * sizeof(float));
    }
    index_cache = ds4_gpu_tensor_alloc(537u * INDEX_DIM * sizeof(float));
    attn_q = ds4_gpu_tensor_alloc(max_words * sizeof(float));
    raw = ds4_gpu_tensor_alloc(RAW_CAP * ATTN_DIM * sizeof(float));
    comp = ds4_gpu_tensor_alloc(537u * ATTN_DIM * sizeof(_Float16));
    ok = ok && index_cache && attn_q && raw && comp;
    if (!ok) goto cleanup;
    for (size_t i = 0; i < 537u * INDEX_DIM; ++i) input[i] = pattern(i, 53, 1.0f / 128.0f);
    ok = ds4_gpu_tensor_write(index_cache, 0, input, 537u * INDEX_DIM * sizeof(float)) &&
        ds4_gpu_dsv4_indexer_qat_tensor(index_cache, 537u, INDEX_DIM);
    for (size_t i = 0; i < max_words; ++i) input[i] = pattern(i, 59, 1.0f / 256.0f);
    ok = ok && ds4_gpu_tensor_write(attn_q, 0, input, max_words * sizeof(float));
    for (size_t i = 0; i < RAW_CAP * ATTN_DIM; ++i) input[i] = pattern(i, 61, 1.0f / 128.0f);
    ok = ok && ds4_gpu_tensor_write(raw, 0, input, RAW_CAP * ATTN_DIM * sizeof(float));
    _Float16 *half_input = (_Float16 *)input;
    for (size_t i = 0; i < 537u * ATTN_DIM; ++i)
        half_input[i] = (_Float16)pattern(i, 67, 1.0f / 128.0f);
    ok = ok && ds4_gpu_tensor_write(comp, 0, half_input, 537u * ATTN_DIM * sizeof(_Float16));

    /* Cached origins make the real 2051 boundary GPU-small. Include the last
     * all-visible row, a 32-row minimum tail, nonmultiple32 tails, a smaller
     * cut, and a no-cut case. All raw spans wrap the 256-row physical ring. */
    const struct { uint32_t pos0, rows, expected_cut; } shapes[] = {
        {1987, 96, 64}, {1987, 97, 64}, {2019, 64, 32}, {2051, 96, 0},
    };
    for (unsigned shape = 0; ok && shape < sizeof(shapes) / sizeof(shapes[0]); ++shape) {
        const uint32_t pos0 = shapes[shape].pos0, rows = shapes[shape].rows;
        const uint32_t n_comp = (pos0 + rows) / 4u;
        uint32_t cut = pos0 < 2051u ? 2051u - pos0 : 0u;
        if (cut > rows - 32u) cut = rows - 32u;
        cut &= ~31u;
        ok = cut == shapes[shape].expected_cut && n_comp > TOP_K;
        for (unsigned variant = 0; ok && variant < 2; ++variant) {
            const bool candidate = variant != 0;
            const uint32_t first = candidate ? cut : 0u, tail = rows - first;
            const guarded_tensor *outputs[] = {&query, &weights, &scores, &selected, &sorted, &attn};
            for (unsigned i = 0; ok && i < 6; ++i) ok = reset_tensor(outputs[i], scratch);
            ds4_gpu_tensor *views[] = {
                row_view(&source_q, first, tail, Q_RANK),
                row_view(&source_weights, first, tail, EMBED),
                row_view(&query, first, tail, QUERY_WIDTH),
                row_view(&weights, first, tail, HEADS),
                row_view(&scores, first, tail, n_comp),
                row_view(&selected, first, tail, TOP_K),
                first ? row_view(&selected, 0, first, TOP_K) : NULL,
            };
            for (unsigned i = 0; i < 6; ++i) ok = ok && views[i];
            if (first) ok = ok && views[6] &&
                ds4_gpu_glm_fill_selected_range_batch_tensor(views[6], first, TOP_K - 1u, TOP_K, 0);
            if (ok) ok = ds4_gpu_matmul_f16_tensor(views[2], model, model_bytes,
                query_offset, Q_RANK, QUERY_WIDTH, views[0], tail) &&
                check_stage("query projection", &query, scratch, reference[0],
                            rows, QUERY_WIDTH, first, candidate, true, false);
            if (ok) ok = ds4_gpu_rope_tail_tensor(views[2], tail, HEADS, INDEX_DIM,
                64, pos0 + first, 65536, false, 160000.0f, 0.0625f, 1.0f,
                1.0f / (1.0f + 0.1f * logf(16.0f)), 32.0f, 1.0f) &&
                check_stage("query RoPE", &query, scratch, reference[1],
                            rows, QUERY_WIDTH, first, candidate, true, false);
            if (ok) ok = ds4_gpu_dsv4_indexer_qat_tensor(views[2], tail * HEADS, INDEX_DIM) &&
                check_stage("query QAT", &query, scratch, reference[2],
                            rows, QUERY_WIDTH, first, candidate, true, false);
            if (ok) ok = ds4_gpu_matmul_f16_tensor(views[3], model, model_bytes,
                weight_offset, EMBED, HEADS, views[1], tail) &&
                check_stage("head weights", &weights, scratch, reference[3],
                            rows, HEADS, first, candidate, true, false);
            if (ok) ok = ds4_gpu_indexer_scores_decode_batch_tensor(views[4], views[2],
                views[3], index_cache, n_comp, tail, pos0 + first, HEADS, INDEX_DIM,
                4, 1.0f / sqrtf((float)QUERY_WIDTH)) &&
                check_stage("scores", &scores, scratch, reference[4],
                            rows, n_comp, first, candidate, false, false);
            for (uint32_t r = first; ok && r < rows; ++r) {
                const uint32_t visible = (pos0 + r + 1u) / 4u;
                for (uint32_t c = 0; ok && c < n_comp; ++c) {
                    const uint32_t bits = scratch[GUARD + (size_t)r * n_comp + c];
                    ok = c < visible ?
                        (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000) :
                        bits == UINT32_C(0xff800000);
                    if (!ok) fprintf(stderr, "score mask row=%u col=%u bits=%08x\n", r, c, bits);
                }
            }
            if (ok) ok = ds4_gpu_indexer_topk_tensor(views[5], views[4], n_comp, tail, TOP_K) &&
                check_stage("top-k", &selected, scratch, reference[5],
                            rows, TOP_K, first, candidate, false, true);
            for (size_t i = 0; ok && i < (size_t)rows * TOP_K; ++i) {
                ok = scratch[GUARD + i] < n_comp;
                if (!ok) fprintf(stderr, "top-k invalid ID word=%zu id=%u n_comp=%u\n",
                                 i, scratch[GUARD + i], n_comp);
            }
            if (ok) ok = ds4_gpu_sort_i32_rows_asc_tensor(sorted.data, selected.data, TOP_K, rows) &&
                check_stage("sorted top-k", &sorted, scratch, reference[6],
                            rows, TOP_K, first, candidate, false, true);
            for (uint32_t r = 0; ok && r < rows; ++r) {
                for (uint32_t c = 0; ok && c < TOP_K; ++c) {
                    const size_t i = GUARD + (size_t)r * TOP_K + c;
                    ok = scratch[i] < n_comp && (c == 0 || scratch[i - 1u] < scratch[i]);
                    if (!ok) fprintf(stderr, "sorted top-k invalid/duplicate ID row=%u col=%u id=%u\n",
                                     r, c, scratch[i]);
                }
            }
            /* The old prefix may contain different future IDs. Its visible
             * IDs must be precisely 0..visible-1 in chronological order. */
            for (uint32_t r = 0; ok && r < cut; ++r) {
                const uint32_t visible = (pos0 + r + 1u) / 4u;
                for (uint32_t c = 0; ok && c < visible; ++c)
                    ok = scratch[GUARD + (size_t)r * TOP_K + c] == c;
            }
            if (ok) ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(attn.data,
                model, model_bytes, 0, attn_q, raw, comp, 1, selected.data, rows,
                pos0, WINDOW + rows, RAW_CAP, (pos0 - WINDOW) % RAW_CAP,
                n_comp, TOP_K, WINDOW, 4, HEADS, ATTN_DIM) &&
                check_stage("indexed attention", &attn, scratch, reference[7],
                            rows, ATTN_WIDTH, 0, candidate, true, false);
            for (unsigned i = 0; i < 7; ++i) ds4_gpu_tensor_free(views[i]);
            if (!ok) fprintf(stderr, "leading-indexer-query failed pos0=%u rows=%u cut=%u variant=%u\n",
                             pos0, rows, cut, variant);
        }
    }
    if (ok) fprintf(stderr, "leading-indexer-query: 4 shapes; projection/RoPE/QAT/weights/scores/top-k/attention bit-identical; output guards intact\n");

cleanup:
    free_tensor(&source_q); free_tensor(&source_weights); free_tensor(&query); free_tensor(&weights);
    free_tensor(&scores); free_tensor(&selected); free_tensor(&sorted); free_tensor(&attn);
    ds4_gpu_tensor_free(index_cache); ds4_gpu_tensor_free(attn_q);
    ds4_gpu_tensor_free(raw); ds4_gpu_tensor_free(comp);
    ds4_gpu_cleanup();
    for (unsigned i = 0; i < 8; ++i) free(reference[i]);
    free(model); free(input); free(scratch);
    return ok ? 0 : 1;
}
