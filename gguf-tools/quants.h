#ifndef DS4_QUANTS_H
#define DS4_QUANTS_H

/*
 * Narrow quantization API used by the DS4 GGUF writer.
 *
 * The enum values intentionally match GGUF/GGML type IDs so template metadata
 * can be copied without translation.  Only the formats used by the DS4 Flash
 * quantization recipes are implemented as output targets.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define DS4Q_MAX_DIMS 4

typedef enum {
    DS4Q_TYPE_F32     = 0,
    DS4Q_TYPE_F16     = 1,
    DS4Q_TYPE_Q4_0    = 2,
    DS4Q_TYPE_Q4_1    = 3,
    DS4Q_TYPE_Q5_0    = 6,
    DS4Q_TYPE_Q5_1    = 7,
    DS4Q_TYPE_Q8_0    = 8,
    DS4Q_TYPE_Q8_1    = 9,
    DS4Q_TYPE_Q2_K    = 10,
    DS4Q_TYPE_Q3_K    = 11,
    DS4Q_TYPE_Q4_K    = 12,
    DS4Q_TYPE_Q5_K    = 13,
    DS4Q_TYPE_Q6_K    = 14,
    DS4Q_TYPE_Q8_K    = 15,
    DS4Q_TYPE_IQ2_XXS = 16,
    DS4Q_TYPE_IQ2_XS  = 17,
    DS4Q_TYPE_IQ3_XXS = 18,
    DS4Q_TYPE_IQ1_S   = 19,
    DS4Q_TYPE_IQ4_NL  = 20,
    DS4Q_TYPE_IQ3_S   = 21,
    DS4Q_TYPE_IQ2_S   = 22,
    DS4Q_TYPE_IQ4_XS  = 23,
    DS4Q_TYPE_I8      = 24,
    DS4Q_TYPE_I16     = 25,
    DS4Q_TYPE_I32     = 26,
    DS4Q_TYPE_I64     = 27,
    DS4Q_TYPE_F64     = 28,
    DS4Q_TYPE_IQ1_M   = 29,
    DS4Q_TYPE_BF16    = 30,
    DS4Q_TYPE_TQ1_0   = 34,
    DS4Q_TYPE_TQ2_0   = 35,
    DS4Q_TYPE_MXFP4   = 39,
    DS4Q_TYPE_NVFP4   = 40,
    DS4Q_TYPE_Q1_0    = 41,
    DS4Q_TYPE_COUNT   = 42,
} ds4q_type;

static inline size_t ds4q_pad(size_t x, size_t n) {
    return ((x + n - 1) / n) * n;
}

const char *ds4q_type_name(ds4q_type type);
bool ds4q_can_quantize(ds4q_type type);
int64_t ds4q_block_size(ds4q_type type);
size_t ds4q_row_size(ds4q_type type, int64_t ne);
bool ds4q_requires_imatrix(ds4q_type type);
void ds4q_quantize_init(ds4q_type type);
size_t ds4q_quantize_chunk(ds4q_type type, const float *src, void *dst,
                           int64_t start, int64_t nrows, int64_t ncols,
                           const float *imatrix);

float ds4q_f16_to_f32(uint16_t bits);
float ds4q_bf16_to_f32(uint16_t bits);
void ds4q_f32_to_f16_row(const float *src, uint16_t *dst, int64_t n);
void ds4q_f32_to_bf16_row(const float *src, uint16_t *dst, int64_t n);

#endif
