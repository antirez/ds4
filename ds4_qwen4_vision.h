#ifndef DS4_QWEN4_VISION_H
#define DS4_QWEN4_VISION_H

#include <stdint.h>

#define DS4_QWEN4_VISION_LAYERS 27
typedef struct {
    uint64_t ln1_w, ln1_b, qkv_w, qkv_b, out_w, out_b, ln2_w, ln2_b, up_w, up_b, down_w, down_b;
    uint32_t qkv_type, out_type, up_type, down_type;
} ds4_qwen4_vision_layer_weights;
typedef struct {
    uint64_t patch_w0, patch_w1, patch_b, pos_embd, post_ln_w, post_ln_b, mm0_w, mm0_b, mm2_w, mm2_b;
    uint32_t mm0_type, mm2_type, patch_type;
    uint32_t n_embd, n_ff, n_head, n_patch, n_merge, n_pos_side, n_out;
    float eps;
    ds4_qwen4_vision_layer_weights layer[DS4_QWEN4_VISION_LAYERS];
} ds4_qwen4_vision_weights;

#endif
