#ifndef DS4_PLANAR_QUANT_H
#define DS4_PLANAR_QUANT_H

#include <stdint.h>
#include <stddef.h>

/*
 * PlanarQuant: KV cache compression via 2D Givens rotation + Lloyd-Max.
 *
 * Adapted from planar-llama (experolk/planar-llama) ggml-planar-quant.c,
 * modified for ds4's head_dim=512 and standalone use (no ggml dependency).
 *
 * Block layout (per 128-dim block):
 *   norm:   uint16_t (FP16)     — 2 bytes
 *   qs:     uint8_t[32]         — 2-bit quantized indices per element, 32 bytes
 *   signs:  uint8_t[16]         — 1-bit QJL signs per element, 16 bytes
 *   Total: 50 bytes per 128-dim block
 *
 * For ds4 head_dim=512: 4 blocks per row = 200 bytes.
 * Compared to FP16 (1024 bytes): 5.12x compression.
 *
 * Reference: ParaMind2025 PlanarQuant, RotorQuant (arXiv:2403.xxxxx)
 */

#define DS4_PLANAR3_BLOCK_DIM  128
#define DS4_PLANAR3_BLOCKS_512 4    /* 512 / 128 */

/* One 128-dim block = 50 bytes. */
typedef struct {
    uint16_t norm;                       /* FP16 group norm */
    uint8_t  qs[DS4_PLANAR3_BLOCK_DIM / 4];    /* 2-bit indices: 32 bytes */
    uint8_t  signs[DS4_PLANAR3_BLOCK_DIM / 8]; /* 1-bit QJL signs: 16 bytes */
} ds4_block_planar3;

/* One 512-dim row = 4 blocks = 200 bytes. */
typedef struct {
    ds4_block_planar3 blocks[DS4_PLANAR3_BLOCKS_512];
} ds4_row_planar3;

/* Quantize a 512-dim FP32 row into Planar3 format.
 * src must have 512 floats, dst must point to sizeof(ds4_row_planar3) bytes.
 * Returns the compressed size in bytes. */
size_t ds4_planar3_quantize_row(const float *src, ds4_row_planar3 *dst);

/* Dequantize a Planar3 512-dim row back to FP32.
 * dst must have space for 512 floats. */
void ds4_planar3_dequantize_row(const ds4_row_planar3 *src, float *dst);

/* Quantize nrows rows. Returns total compressed bytes. */
size_t ds4_planar3_quantize(const float *src, void *dst,
                             size_t nrows, size_t n_per_row);

/* Dequantize nrows rows. */
void ds4_planar3_dequantize(const void *src, float *dst,
                              size_t nrows, size_t n_per_row);

/* Compute roundtrip quality metrics for one row.
 * Returns cosine similarity between original and reconstructed vectors. */
float ds4_planar3_roundtrip_cosine(const float *original, float *reconstructed,
                                    size_t dim);

/* Compute MSE between original and reconstructed. */
float ds4_planar3_roundtrip_mse(const float *original, float *reconstructed,
                                  size_t dim);

#endif /* DS4_PLANAR_QUANT_H */
