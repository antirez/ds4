#ifndef DS4_IMAGE_H
#define DS4_IMAGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_IMAGE_MAX_ENCODED_BYTES (64u * 1024u * 1024u)
#define DS4_IMAGE_MAX_DIMENSION 16384u
#define DS4_IMAGE_MAX_PIXELS (64u * 1024u * 1024u)

typedef struct {
    uint32_t width;
    uint32_t height;
    uint8_t *rgb;
    uint8_t fingerprint[32];
} ds4_image;

typedef struct {
    uint32_t content_width;
    uint32_t content_height;
    uint32_t padded_width;
    uint32_t padded_height;
    uint32_t grid_width;
    uint32_t grid_height;
    uint32_t patch_count;
    uint32_t image_token_count;
    float *patches;
} ds4_image_patches;

typedef struct {
    uint32_t content_width;
    uint32_t content_height;
    uint32_t padded_width;
    uint32_t padded_height;
    uint32_t grid_width;
    uint32_t grid_height;
    uint32_t llm_grid_width;
    uint32_t llm_grid_height;
    uint32_t patch_count;
    float *patches;
} ds4_deepseek4_image_patches;

typedef enum {
    DS4_DEEPSEEK4_IMAGE_START = 0,
    DS4_DEEPSEEK4_IMAGE_PAD = 1,
    DS4_DEEPSEEK4_IMAGE = 2,
    DS4_DEEPSEEK4_IMAGE_NEWLINE = 3,
    DS4_DEEPSEEK4_IMAGE_END = 4,
} ds4_deepseek4_image_token_type;

typedef struct {
    uint32_t token_count;
    uint32_t image_count;
    uint8_t *types;
    uint32_t *perm;
} ds4_deepseek4_image_layout;

int ds4_image_decode_memory(
        ds4_image     *out,
        const uint8_t *encoded,
        size_t         encoded_len,
        char          *error,
        size_t         error_cap);

int ds4_image_decode_file(
        ds4_image *out,
        const char *path,
        char *error,
        size_t error_cap);

void ds4_image_free(ds4_image *image);

int ds4_image_preprocess_glm53(
        ds4_image_patches *out,
        const ds4_image   *image,
        uint32_t           min_image_tokens,
        uint32_t           max_image_tokens,
        char              *error,
        size_t             error_cap);

void ds4_image_patches_free(ds4_image_patches *patches);

int ds4_image_preprocess_deepseek4(
        ds4_deepseek4_image_patches *out,
        const ds4_image             *image,
        char                        *error,
        size_t                       error_cap);

int ds4_deepseek4_image_layout_build(
        ds4_deepseek4_image_layout *out,
        uint32_t                    grid_height,
        uint32_t                    grid_width,
        uint32_t                    start_pos,
        char                       *error,
        size_t                      error_cap);

void ds4_deepseek4_image_patches_free(ds4_deepseek4_image_patches *patches);
void ds4_deepseek4_image_layout_free(ds4_deepseek4_image_layout *layout);

/* Find the next DeepSeek image block at or after *cursor. A block includes
 * the alignment PAD tokens immediately before IMAGE_START. Returns 1 when a
 * block was found, 0 at end of input, and -1 for malformed synthetic tokens. */
int ds4_deepseek4_next_image_span(
        const int *tokens,
        uint32_t   token_count,
        uint32_t   vocab_size,
        uint32_t  *cursor,
        uint32_t  *block_start,
        uint32_t  *image_start,
        uint32_t  *image_end);

/* Choose a prefill chunk without splitting a DeepSeek image block. */
int ds4_deepseek4_prefill_chunk(
        const int *tokens,
        uint32_t   token_count,
        uint32_t   vocab_size,
        uint32_t   pos0,
        uint32_t   cap,
        uint32_t  *chunk);

/* Fill [lo, hi] absolute raw-key bounds for every query row. */
int ds4_deepseek4_attention_bounds(
        const int *tokens,
        uint32_t   token_count,
        uint32_t   vocab_size,
        uint32_t   pos0,
        uint32_t   n_raw,
        uint32_t   window,
        uint32_t  *bounds);

#ifdef __cplusplus
}
#endif

#endif
