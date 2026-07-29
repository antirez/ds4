#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "ds4.h"
#include "ds4_gpu.h"
#include "gguf-tools/quants.h"
#include "k3_metal.h"

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t f32_to_f16(float value) {
    const __fp16 half = (__fp16)value;
    uint16_t bits;
    memcpy(&bits, &half, sizeof(bits));
    return bits;
}

static float f16_to_f32(uint16_t value) {
    __fp16 half;
    memcpy(&half, &value, sizeof(half));
    return (float)half;
}

static float e8m0_to_f32(uint8_t value) {
    return exp2f((float)((int)value - 127));
}

static float e4m3_to_f32(uint8_t value) {
    const int sign = value >> 7;
    const int exponent = (value >> 3) & 15;
    const int mantissa = value & 7;
    const float a = exponent == 0
        ? (float)mantissa * 0x1p-9f
        : (1.0f + (float)mantissa * 0.125f) *
          exp2f((float)(exponent - 7));
    return sign ? -a : a;
}

static float e2m1_to_f32(uint8_t value) {
    static const float table[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
       -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f,
    };
    return table[value & 15u];
}

static void require_close(
        const char *name,
        float actual,
        float expected,
        float tolerance) {
    if (!isfinite(actual) || fabsf(actual - expected) > tolerance) {
        fprintf(stderr,
                "%s mismatch: actual %.9g expected %.9g tolerance %.9g\n",
                name, actual, expected, tolerance);
        exit(1);
    }
}

static void require_ok(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "%s failed\n", what);
        exit(1);
    }
}

int main(void) {
    enum {
        IN = 32,
        OUT = 8,
        MX_OUT = 4,
        EXPERTS = 8,
        TOP_K = 4,
        KDA_DIM = 128,
        MM_ROWS = 9,
        MAP_BYTES = 2 * 1024 * 1024,
    };
    enum {
        MAT_OFF = 0,
        NORM_OFF = 4096,
        GATE_PACKED_OFF = 8192,
        GATE_SCALE_OFF = 12288,
        UP_PACKED_OFF = 16384,
        UP_SCALE_OFF = 20480,
        DOWN_PACKED_OFF = 24576,
        DOWN_SCALE_OFF = 28672,
        ROUTER_BIAS_OFF = 32768,
        ATTN_NORM_OFF = 36864,
        ATTN_PROJ_OFF = 40960,
        KDA_Q_CONV_OFF = 45056,
        KDA_K_CONV_OFF = 47104,
        KDA_V_CONV_OFF = 49152,
        KDA_A_LOG_OFF = 51200,
        KDA_DT_BIAS_OFF = 53248,
        KDA_NORM_OFF = 55296,
        MLA_KV_B_OFF = 65536,
        GGUF_MXFP4_OFF = 458752,
        GGUF_MXFP4_UP_OFF = 462848,
        GGUF_MXFP4_DOWN_OFF = 466944,
        GGUF_IQ2_GATE_OFF = 524288,
        GGUF_IQ2_UP_OFF = 819200,
        GGUF_Q2_DOWN_OFF = 1114112,
    };
    enum {
        GGUF_MXFP4_IN = 256,
        GGUF_MXFP4_OUT = 8,
        GGUF_MXFP4_ROWS = 9,
        GGUF_MXFP4_ROW_BYTES = GGUF_MXFP4_IN / 32 * 17,
    };
    enum {
        GGUF_MOE_DIM = 256,
        GGUF_MOE_EXPERTS = 16,
        GGUF_MOE_IQ2_ROW_BYTES = 66,
        GGUF_MOE_Q2_ROW_BYTES = 84,
        GGUF_MOE_IQ2_EXPERT_BYTES =
            GGUF_MOE_DIM * GGUF_MOE_IQ2_ROW_BYTES,
        GGUF_MOE_Q2_EXPERT_BYTES =
            GGUF_MOE_DIM * GGUF_MOE_Q2_ROW_BYTES,
    };

    uint8_t *model = mmap(NULL, MAP_BYTES, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (model == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    uint16_t *matrix = (uint16_t *)(model + MAT_OFF);
    for (uint32_t o = 0; o < OUT; o++) {
        for (uint32_t k = 0; k < IN; k++) {
            matrix[o * IN + k] =
                f32_to_bf16(((int)(o + k) % 9 - 4) * 0.125f);
        }
    }
    uint16_t *norm_weight = (uint16_t *)(model + NORM_OFF);
    uint16_t *attn_norm = (uint16_t *)(model + ATTN_NORM_OFF);
    uint16_t *attn_proj = (uint16_t *)(model + ATTN_PROJ_OFF);
    for (uint32_t k = 0; k < IN; k++) {
        norm_weight[k] = f32_to_bf16(0.75f + 0.01f * (float)k);
        attn_norm[k] = f32_to_bf16(1.0f);
        attn_proj[k] = f32_to_bf16(1.0f);
    }

    uint8_t *gate_packed = model + GATE_PACKED_OFF;
    uint8_t *gate_scale = model + GATE_SCALE_OFF;
    uint8_t *up_packed = model + UP_PACKED_OFF;
    uint8_t *up_scale = model + UP_SCALE_OFF;
    uint8_t *down_packed = model + DOWN_PACKED_OFF;
    uint8_t *down_scale = model + DOWN_SCALE_OFF;
    for (uint32_t o = 0; o < MX_OUT; o++) {
        const uint8_t gate_code = (uint8_t)(1u + o);
        const uint8_t up_code = (uint8_t)(5u - o);
        for (uint32_t k = 0; k < IN / 2; k++) {
            gate_packed[o * (IN / 2) + k] =
                (uint8_t)(gate_code | (gate_code << 4));
            up_packed[o * (IN / 2) + k] =
                (uint8_t)(up_code | (up_code << 4));
            down_packed[o * (IN / 2) + k] =
                (uint8_t)(2u | (2u << 4));
        }
        gate_scale[o] = 127u;
        up_scale[o] = 127u;
        down_scale[o] = 127u;
    }

    float *router_bias = (float *)(model + ROUTER_BIAS_OFF);
    for (uint32_t e = 0; e < EXPERTS; e++) {
        router_bias[e] = e == 6u ? 0.2f : 0.0f;
    }
    float *kda_q_conv = (float *)(model + KDA_Q_CONV_OFF);
    float *kda_k_conv = (float *)(model + KDA_K_CONV_OFF);
    float *kda_v_conv = (float *)(model + KDA_V_CONV_OFF);
    float *kda_a_log = (float *)(model + KDA_A_LOG_OFF);
    float *kda_dt_bias = (float *)(model + KDA_DT_BIAS_OFF);
    float *kda_norm = (float *)(model + KDA_NORM_OFF);
    for (uint32_t d = 0; d < KDA_DIM * 4u; d++) {
        kda_q_conv[d] = 0.0f;
        kda_k_conv[d] = 0.0f;
        kda_v_conv[d] = 0.0f;
    }
    for (uint32_t d = 0; d < KDA_DIM; d++) {
        kda_q_conv[d * 4u + 3u] = 1.0f;
        kda_k_conv[d * 4u + 3u] = 1.0f;
        kda_v_conv[d * 4u + 3u] = 1.0f;
        kda_dt_bias[d] = 0.0f;
        kda_norm[d] = 1.0f;
    }
    kda_a_log[0] = 0.0f;
    uint16_t *mla_kv_b = (uint16_t *)(model + MLA_KV_B_OFF);
    mla_kv_b[0] = f32_to_bf16(1.0f);
    for (uint32_t value = 0; value < 128u; value++) {
        mla_kv_b[((128u + value) * 512u)] = f32_to_bf16(1.0f);
    }
    const uint32_t mxfp4_offsets[] = {
        GGUF_MXFP4_OFF,
        GGUF_MXFP4_UP_OFF,
        GGUF_MXFP4_DOWN_OFF,
    };
    for (uint32_t matrix_index = 0; matrix_index < 3u; matrix_index++) {
        for (uint32_t o = 0; o < GGUF_MXFP4_OUT; o++) {
            uint8_t *row = model + mxfp4_offsets[matrix_index] +
                o * GGUF_MXFP4_ROW_BYTES;
            const uint8_t low = (uint8_t)(
                matrix_index == 0u ? 1u + o % 6u :
                matrix_index == 1u ? 6u - o % 6u :
                                     2u + o % 5u);
            const uint8_t high = (uint8_t)(low + 8u);
            for (uint32_t block = 0; block < GGUF_MXFP4_IN / 32u; block++) {
                uint8_t *packed = row + block * 17u;
                packed[0] = 127u;
                for (uint32_t i = 0; i < 16u; i++) {
                    packed[1u + i] = (uint8_t)(low | (high << 4));
                }
            }
        }
    }
    float *moe_matrix = malloc(
        (size_t)GGUF_MOE_DIM * GGUF_MOE_DIM * sizeof(float));
    float *moe_importance = malloc(GGUF_MOE_DIM * sizeof(float));
    require_ok(moe_matrix && moe_importance,
               "GGUF routed quantization allocation");
    for (uint32_t k = 0; k < GGUF_MOE_DIM; k++) {
        moe_importance[k] = 1.0f + (float)(k % 11u) * 0.03125f;
    }
    ds4q_quantize_init(DS4Q_TYPE_IQ2_XXS);
    for (uint32_t expert = 0; expert < GGUF_MOE_EXPERTS; expert++) {
        for (uint32_t o = 0; o < GGUF_MOE_DIM; o++) {
            for (uint32_t k = 0; k < GGUF_MOE_DIM; k++) {
                moe_matrix[o * GGUF_MOE_DIM + k] =
                    ((int)((expert * 5u + o * 3u + k * 7u) % 29u) - 14) *
                    0.0078125f;
            }
        }
        require_ok(ds4q_quantize_chunk(
            DS4Q_TYPE_IQ2_XXS,
            moe_matrix,
            model + GGUF_IQ2_GATE_OFF +
                expert * GGUF_MOE_IQ2_EXPERT_BYTES,
            0,
            GGUF_MOE_DIM,
            GGUF_MOE_DIM,
            moe_importance) == GGUF_MOE_IQ2_EXPERT_BYTES,
            "GGUF routed gate quantization");
        for (uint32_t o = 0; o < GGUF_MOE_DIM; o++) {
            for (uint32_t k = 0; k < GGUF_MOE_DIM; k++) {
                moe_matrix[o * GGUF_MOE_DIM + k] =
                    ((int)((expert * 11u + o * 7u + k * 2u) % 31u) - 15) *
                    0.009765625f;
            }
        }
        require_ok(ds4q_quantize_chunk(
            DS4Q_TYPE_IQ2_XXS,
            moe_matrix,
            model + GGUF_IQ2_UP_OFF +
                expert * GGUF_MOE_IQ2_EXPERT_BYTES,
            0,
            GGUF_MOE_DIM,
            GGUF_MOE_DIM,
            moe_importance) == GGUF_MOE_IQ2_EXPERT_BYTES,
            "GGUF routed up quantization");
    }
    memset(model + GGUF_Q2_DOWN_OFF, 0,
           (size_t)GGUF_MOE_EXPERTS * GGUF_MOE_Q2_EXPERT_BYTES);
    free(moe_importance);
    free(moe_matrix);

    require_ok(ds4_gpu_init(), "Metal initialization");
    require_ok(ds4_gpu_set_model_map(model, MAP_BYTES), "model map");

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(IN * sizeof(float));
    ds4_gpu_tensor *bf16_in =
        ds4_gpu_tensor_alloc(IN * sizeof(uint16_t));
    ds4_gpu_tensor *bf16_out =
        ds4_gpu_tensor_alloc(IN * sizeof(float));
    ds4_gpu_tensor *mm_out = ds4_gpu_tensor_alloc(OUT * sizeof(float));
    ds4_gpu_tensor *q8_weights = ds4_gpu_tensor_alloc(OUT * 34u);
    ds4_gpu_tensor *mm_batch_x =
        ds4_gpu_tensor_alloc(MM_ROWS * IN * sizeof(float));
    ds4_gpu_tensor *mm_batch_out =
        ds4_gpu_tensor_alloc(MM_ROWS * OUT * sizeof(float));
    ds4_gpu_tensor *gguf_mxfp4_x =
        ds4_gpu_tensor_alloc(
            GGUF_MXFP4_ROWS * GGUF_MXFP4_IN * sizeof(float));
    ds4_gpu_tensor *gguf_mxfp4_out =
        ds4_gpu_tensor_alloc(
            GGUF_MXFP4_ROWS * GGUF_MXFP4_OUT * sizeof(float));
    ds4_gpu_tensor *gguf_mxfp8_values =
        ds4_gpu_tensor_alloc(GGUF_MXFP4_IN);
    ds4_gpu_tensor *gguf_mxfp8_scales =
        ds4_gpu_tensor_alloc(GGUF_MXFP4_IN / 32u);
    ds4_gpu_tensor *gguf_moe_x =
        ds4_gpu_tensor_alloc(GGUF_MOE_DIM * sizeof(float));
    ds4_gpu_tensor *gguf_moe_gate =
        ds4_gpu_tensor_alloc(
            GGUF_MOE_EXPERTS * GGUF_MOE_DIM * sizeof(float));
    ds4_gpu_tensor *gguf_moe_up =
        ds4_gpu_tensor_alloc(
            GGUF_MOE_EXPERTS * GGUF_MOE_DIM * sizeof(float));
    ds4_gpu_tensor *gguf_moe_mid =
        ds4_gpu_tensor_alloc(
            GGUF_MOE_EXPERTS * GGUF_MOE_DIM * sizeof(float));
    ds4_gpu_tensor *gguf_moe_expert_out =
        ds4_gpu_tensor_alloc(
            GGUF_MOE_EXPERTS * GGUF_MOE_DIM * sizeof(float));
    ds4_gpu_tensor *gguf_moe_out =
        ds4_gpu_tensor_alloc(GGUF_MOE_DIM * sizeof(float));
    ds4_gpu_tensor *gguf_moe_selected =
        ds4_gpu_tensor_alloc(GGUF_MOE_EXPERTS * sizeof(int32_t));
    ds4_gpu_tensor *gguf_moe_weights =
        ds4_gpu_tensor_alloc(GGUF_MOE_EXPERTS * sizeof(float));
    ds4_gpu_tensor *norm_out = ds4_gpu_tensor_alloc(IN * sizeof(float));
    ds4_gpu_tensor *situ_gate = ds4_gpu_tensor_alloc(IN * sizeof(float));
    ds4_gpu_tensor *situ_up = ds4_gpu_tensor_alloc(IN * sizeof(float));
    ds4_gpu_tensor *situ_out = ds4_gpu_tensor_alloc(IN * sizeof(float));
    ds4_gpu_tensor *mx_values = ds4_gpu_tensor_alloc(IN);
    ds4_gpu_tensor *mx_scales = ds4_gpu_tensor_alloc(IN / 32);
    ds4_gpu_tensor *mx_gate = ds4_gpu_tensor_alloc(MX_OUT * sizeof(float));
    ds4_gpu_tensor *mx_up = ds4_gpu_tensor_alloc(MX_OUT * sizeof(float));
    ds4_gpu_tensor *mx_mid = ds4_gpu_tensor_alloc(MX_OUT * sizeof(float));
    ds4_gpu_tensor *down_values = ds4_gpu_tensor_alloc(MX_OUT);
    ds4_gpu_tensor *down_scales = ds4_gpu_tensor_alloc(1);
    ds4_gpu_tensor *down_out = ds4_gpu_tensor_alloc(MX_OUT * sizeof(float));
    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(EXPERTS * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(TOP_K * sizeof(int32_t));
    ds4_gpu_tensor *selected_weights =
        ds4_gpu_tensor_alloc(TOP_K * sizeof(float));
    ds4_gpu_tensor *prefix = ds4_gpu_tensor_alloc(IN * sizeof(float));
    ds4_gpu_tensor *delta = ds4_gpu_tensor_alloc(IN * sizeof(float));
    ds4_gpu_tensor *blocks = ds4_gpu_tensor_alloc(2 * IN * sizeof(float));
    ds4_gpu_tensor *attn_out = ds4_gpu_tensor_alloc(IN * sizeof(float));
    ds4_gpu_tensor *kda_q = ds4_gpu_tensor_alloc(KDA_DIM * sizeof(float));
    ds4_gpu_tensor *kda_k = ds4_gpu_tensor_alloc(KDA_DIM * sizeof(float));
    ds4_gpu_tensor *kda_v = ds4_gpu_tensor_alloc(KDA_DIM * sizeof(float));
    ds4_gpu_tensor *kda_gate =
        ds4_gpu_tensor_alloc(KDA_DIM * sizeof(float));
    ds4_gpu_tensor *kda_beta = ds4_gpu_tensor_alloc(sizeof(float));
    ds4_gpu_tensor *kda_output_gate =
        ds4_gpu_tensor_alloc(KDA_DIM * sizeof(float));
    ds4_gpu_tensor *kda_conv_state =
        ds4_gpu_tensor_alloc(3u * 3u * KDA_DIM * sizeof(float));
    ds4_gpu_tensor *kda_state =
        ds4_gpu_tensor_alloc(KDA_DIM * KDA_DIM * sizeof(float));
    ds4_gpu_tensor *kda_out =
        ds4_gpu_tensor_alloc(KDA_DIM * sizeof(float));
    ds4_gpu_tensor *mla_q = ds4_gpu_tensor_alloc(192u * sizeof(float));
    ds4_gpu_tensor *mla_kv = ds4_gpu_tensor_alloc(576u * sizeof(float));
    ds4_gpu_tensor *mla_gate = ds4_gpu_tensor_alloc(128u * sizeof(float));
    ds4_gpu_tensor *mla_cache =
        ds4_gpu_tensor_alloc(4u * 576u * sizeof(uint16_t));
    ds4_gpu_tensor *mla_q_latent =
        ds4_gpu_tensor_alloc(512u * sizeof(float));
    ds4_gpu_tensor *mla_partial =
        ds4_gpu_tensor_alloc(512u * sizeof(float));
    ds4_gpu_tensor *mla_stats =
        ds4_gpu_tensor_alloc(2u * sizeof(float));
    ds4_gpu_tensor *mla_out = ds4_gpu_tensor_alloc(128u * sizeof(float));
    if (!x || !bf16_in || !bf16_out || !mm_out || !q8_weights ||
        !mm_batch_x || !mm_batch_out ||
        !gguf_mxfp4_x || !gguf_mxfp4_out ||
        !gguf_mxfp8_values || !gguf_mxfp8_scales || !norm_out ||
        !gguf_moe_x || !gguf_moe_gate || !gguf_moe_up ||
        !gguf_moe_mid || !gguf_moe_expert_out || !gguf_moe_out ||
        !gguf_moe_selected || !gguf_moe_weights ||
        !situ_gate || !situ_up || !situ_out || !mx_values || !mx_scales ||
        !mx_gate || !mx_up || !mx_mid || !down_values || !down_scales ||
        !down_out || !logits || !selected || !selected_weights || !prefix ||
        !delta || !blocks || !attn_out || !kda_q || !kda_k || !kda_v ||
        !kda_gate || !kda_beta || !kda_output_gate || !kda_conv_state ||
        !kda_state || !kda_out || !mla_q || !mla_kv || !mla_gate ||
        !mla_cache || !mla_q_latent || !mla_partial || !mla_stats ||
        !mla_out) {
        fprintf(stderr, "GPU tensor allocation failed\n");
        return 1;
    }

    float input[IN];
    for (uint32_t k = 0; k < IN; k++) {
        input[k] = ((int)(k % 11u) - 5) * 0.1875f;
    }
    require_ok(ds4_gpu_tensor_write(x, 0, input, sizeof(input)), "input write");
    require_ok(ds4_gpu_begin_commands(), "ordered zero begin");
    require_ok(k3_gpu_zero_f32_ordered(x, IN), "ordered zero encode");
    require_ok(ds4_gpu_end_commands(), "ordered zero end");
    float zeroed[IN];
    require_ok(ds4_gpu_tensor_read(x, 0, zeroed, sizeof(zeroed)),
               "ordered zero read");
    for (uint32_t k = 0; k < IN; k++) {
        require_close("ordered zero", zeroed[k], 0.0f, 0.0f);
    }
    require_ok(ds4_gpu_tensor_write(x, 0, input, sizeof(input)),
               "input restore");

    uint16_t bf16_input[IN];
    for (uint32_t k = 0; k < IN; k++) {
        bf16_input[k] = f32_to_bf16(input[k]);
    }
    require_ok(ds4_gpu_tensor_write(
        bf16_in, 0, bf16_input, sizeof(bf16_input)), "BF16 expand write");
    require_ok(k3_gpu_expand_bf16(
        bf16_out, bf16_in, IN), "BF16 expansion");
    float bf16_actual[IN];
    require_ok(ds4_gpu_tensor_read(
        bf16_out, 0, bf16_actual, sizeof(bf16_actual)),
        "BF16 expansion read");
    for (uint32_t k = 0; k < IN; k++) {
        require_close(
            "BF16 expansion", bf16_actual[k],
            bf16_to_f32(bf16_input[k]), 0.0f);
    }

    require_ok(k3_gpu_matmul_bf16(
        mm_out, model, MAP_BYTES, MAT_OFF, IN, OUT, x, 1), "BF16 matmul");
    float mm_actual[OUT];
    require_ok(ds4_gpu_tensor_read(mm_out, 0, mm_actual, sizeof(mm_actual)),
               "matmul read");
    for (uint32_t o = 0; o < OUT; o++) {
        float expected = 0.0f;
        for (uint32_t k = 0; k < IN; k++) {
            expected += input[k] * bf16_to_f32(matrix[o * IN + k]);
        }
        require_close("BF16 matmul", mm_actual[o], expected, 2e-4f);
    }
    require_ok(ds4_gpu_release_model_views(),
               "release ordinary model views");
    const uint64_t persistent_offsets[] = {MAT_OFF};
    const uint64_t persistent_sizes[] = {65536};
    require_ok(k3_gpu_pin_model_spans(
        model,
        MAP_BYTES,
        persistent_offsets,
        persistent_sizes,
        1), "persistent K3 model span");
    require_ok(k3_gpu_matmul_bf16(
        mm_out, model, MAP_BYTES, MAT_OFF, IN, OUT, x, 1),
        "persistent-span BF16 matmul");
    require_ok(ds4_gpu_tensor_read(
        mm_out, 0, mm_actual, sizeof(mm_actual)),
        "persistent-span matmul read");
    for (uint32_t o = 0; o < OUT; o++) {
        float expected = 0.0f;
        for (uint32_t k = 0; k < IN; k++) {
            expected += input[k] * bf16_to_f32(matrix[o * IN + k]);
        }
        require_close(
            "persistent-span BF16 matmul",
            mm_actual[o],
            expected,
            2e-4f);
    }
    k3_gpu_clear_static_residency();
    require_ok(ds4_gpu_set_model_map(model, MAP_BYTES),
               "restore model map");

    uint8_t q8_data[OUT * 34u];
    for (uint32_t o = 0; o < OUT; o++) {
        float amax = 0.0f;
        for (uint32_t k = 0; k < IN; k++) {
            const float absolute =
                fabsf(bf16_to_f32(matrix[o * IN + k]));
            if (absolute > amax) amax = absolute;
        }
        const float scale = amax / 127.0f;
        const float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
        const uint16_t half_scale = f32_to_f16(scale);
        memcpy(q8_data + o * 34u, &half_scale, sizeof(half_scale));
        int8_t *quants = (int8_t *)(q8_data + o * 34u + 2u);
        for (uint32_t k = 0; k < IN; k++) {
            quants[k] = (int8_t)roundf(
                bf16_to_f32(matrix[o * IN + k]) * inverse);
        }
    }
    require_ok(ds4_gpu_tensor_write(
        q8_weights, 0, q8_data, sizeof(q8_data)), "Q8 weight write");
    require_ok(k3_gpu_matmul_q8_0_tensor(
        mm_out, q8_weights, 0, IN, OUT, x, 1), "cached Q8 matmul");
    require_ok(ds4_gpu_tensor_read(
        mm_out, 0, mm_actual, sizeof(mm_actual)), "cached Q8 matmul read");
    for (uint32_t o = 0; o < OUT; o++) {
        uint16_t scale_bits;
        memcpy(&scale_bits, q8_data + o * 34u, sizeof(scale_bits));
        const float scale = f16_to_f32(scale_bits);
        const int8_t *quants = (const int8_t *)(q8_data + o * 34u + 2u);
        float expected = 0.0f;
        for (uint32_t k = 0; k < IN; k++) {
            expected += input[k] * scale * quants[k];
        }
        require_close("cached Q8 matmul", mm_actual[o], expected, 3e-4f);
    }

    float mm_batch_input[MM_ROWS * IN];
    for (uint32_t r = 0; r < MM_ROWS; r++) {
        for (uint32_t k = 0; k < IN; k++) {
            mm_batch_input[r * IN + k] =
                ((int)((r * 7u + k * 3u) % 23u) - 11) * 0.071f;
        }
    }

    float gguf_mxfp4_input[GGUF_MXFP4_ROWS * GGUF_MXFP4_IN];
    for (uint32_t row = 0; row < GGUF_MXFP4_ROWS; row++) {
        for (uint32_t k = 0; k < GGUF_MXFP4_IN; k++) {
            gguf_mxfp4_input[row * GGUF_MXFP4_IN + k] =
                ((int)((row * 13u + k * 7u) % 31u) - 15) * 0.03125f;
        }
    }
    require_ok(ds4_gpu_tensor_write(
        gguf_mxfp4_x, 0, gguf_mxfp4_input,
        sizeof(gguf_mxfp4_input)), "GGUF MXFP4 input write");
    for (uint32_t test_rows = 1; test_rows <= GGUF_MXFP4_ROWS;
         test_rows += GGUF_MXFP4_ROWS - 1u) {
        require_ok(ds4_gpu_matmul_quant_tensor(
            gguf_mxfp4_out,
            model,
            MAP_BYTES,
            GGUF_MXFP4_OFF,
            DS4_GPU_GGML_TYPE_MXFP4,
            GGUF_MXFP4_IN,
            GGUF_MXFP4_OUT,
            gguf_mxfp4_x,
            test_rows), "GGUF MXFP4 matmul");
        float actual[GGUF_MXFP4_ROWS * GGUF_MXFP4_OUT];
        require_ok(ds4_gpu_tensor_read(
            gguf_mxfp4_out, 0, actual,
            (size_t)test_rows * GGUF_MXFP4_OUT * sizeof(float)),
            "GGUF MXFP4 output read");
        for (uint32_t row = 0; row < test_rows; row++) {
            for (uint32_t o = 0; o < GGUF_MXFP4_OUT; o++) {
                const float low = e2m1_to_f32(1u + o % 6u);
                const float high = e2m1_to_f32(9u + o % 6u);
                float expected = 0.0f;
                for (uint32_t block = 0;
                     block < GGUF_MXFP4_IN / 32u; block++) {
                    for (uint32_t i = 0; i < 16u; i++) {
                        expected +=
                            gguf_mxfp4_input[
                                row * GGUF_MXFP4_IN +
                                block * 32u + i] * low;
                        expected +=
                            gguf_mxfp4_input[
                                row * GGUF_MXFP4_IN +
                                block * 32u + 16u + i] * high;
                    }
                }
                require_close(
                    "GGUF MXFP4 matmul",
                    actual[row * GGUF_MXFP4_OUT + o],
                    expected,
                    test_rows == 1u ? 5e-4f : 8e-3f);
            }
        }
    }
    require_ok(k3_gpu_gguf_mxfp4_pair_situ_f32(
        gguf_mxfp4_out,
        model,
        MAP_BYTES,
        GGUF_MXFP4_OFF,
        GGUF_MXFP4_UP_OFF,
        gguf_mxfp4_x,
        GGUF_MXFP4_IN,
        GGUF_MXFP4_OUT,
        4.0f,
        25.0f), "GGUF MXFP4/F32 pair SiTU");
    float gguf_pair_f32_actual[GGUF_MXFP4_OUT];
    require_ok(ds4_gpu_tensor_read(
        gguf_mxfp4_out,
        0,
        gguf_pair_f32_actual,
        sizeof(gguf_pair_f32_actual)),
        "GGUF MXFP4/F32 pair read");
    for (uint32_t o = 0; o < GGUF_MXFP4_OUT; o++) {
        const float gate_low = e2m1_to_f32(1u + o % 6u);
        const float gate_high = e2m1_to_f32(9u + o % 6u);
        const float up_low = e2m1_to_f32(6u - o % 6u);
        const float up_high = e2m1_to_f32(14u - o % 6u);
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        for (uint32_t block = 0;
             block < GGUF_MXFP4_IN / 32u;
             block++) {
            for (uint32_t i = 0; i < 16u; i++) {
                const float x_low =
                    gguf_mxfp4_input[block * 32u + i];
                const float x_high =
                    gguf_mxfp4_input[block * 32u + 16u + i];
                gate_sum += x_low * gate_low;
                gate_sum += x_high * gate_high;
                up_sum += x_low * up_low;
                up_sum += x_high * up_high;
            }
        }
        const float expected =
            4.0f * tanhf(gate_sum / 4.0f) /
            (1.0f + expf(-gate_sum)) *
            (25.0f * tanhf(up_sum / 25.0f));
        require_close(
            "GGUF MXFP4/F32 pair SiTU",
            gguf_pair_f32_actual[o],
            expected,
            1e-3f);
    }
    require_ok(k3_gpu_mxfp8_quantize(
        gguf_mxfp8_values,
        gguf_mxfp8_scales,
        gguf_mxfp4_x,
        GGUF_MXFP4_IN,
        1u), "GGUF shared MXFP8 quantization");
    uint8_t gguf_a8_values[GGUF_MXFP4_IN];
    uint8_t gguf_a8_scales[GGUF_MXFP4_IN / 32u];
    require_ok(ds4_gpu_tensor_read(
        gguf_mxfp8_values, 0, gguf_a8_values, sizeof(gguf_a8_values)),
        "GGUF shared MXFP8 values read");
    require_ok(ds4_gpu_tensor_read(
        gguf_mxfp8_scales, 0, gguf_a8_scales, sizeof(gguf_a8_scales)),
        "GGUF shared MXFP8 scales read");
    require_ok(k3_gpu_gguf_mxfp4_pair_situ(
        gguf_mxfp4_out,
        model,
        MAP_BYTES,
        GGUF_MXFP4_OFF,
        GGUF_MXFP4_UP_OFF,
        gguf_mxfp8_values,
        gguf_mxfp8_scales,
        GGUF_MXFP4_IN,
        GGUF_MXFP4_OUT,
        1u,
        4.0f,
        25.0f), "GGUF MXFP4/A8 pair SiTU");
    float gguf_shared_actual[GGUF_MXFP4_OUT];
    require_ok(ds4_gpu_tensor_read(
        gguf_mxfp4_out,
        0,
        gguf_shared_actual,
        sizeof(gguf_shared_actual)),
        "GGUF MXFP4/A8 pair read");
    for (uint32_t o = 0; o < GGUF_MXFP4_OUT; o++) {
        const uint8_t gate_low = (uint8_t)(1u + o % 6u);
        const uint8_t gate_high = (uint8_t)(gate_low + 8u);
        const uint8_t up_low = (uint8_t)(6u - o % 6u);
        const uint8_t up_high = (uint8_t)(up_low + 8u);
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        for (uint32_t k = 0; k < GGUF_MXFP4_IN; k++) {
            const float xq =
                e4m3_to_f32(gguf_a8_values[k]) *
                e8m0_to_f32(gguf_a8_scales[k / 32u]);
            gate_sum += xq * e2m1_to_f32(
                k % 32u < 16u ? gate_low : gate_high);
            up_sum += xq * e2m1_to_f32(
                k % 32u < 16u ? up_low : up_high);
        }
        const float expected =
            4.0f * tanhf(gate_sum / 4.0f) /
            (1.0f + expf(-gate_sum)) *
            (25.0f * tanhf(up_sum / 25.0f));
        require_close(
            "GGUF MXFP4/A8 pair SiTU",
            gguf_shared_actual[o],
            expected,
            2e-3f);
    }
    require_ok(k3_gpu_gguf_mxfp4_down(
        gguf_mxfp4_out,
        model,
        MAP_BYTES,
        GGUF_MXFP4_DOWN_OFF,
        gguf_mxfp8_values,
        gguf_mxfp8_scales,
        GGUF_MXFP4_IN,
        GGUF_MXFP4_OUT,
        1u), "GGUF MXFP4/A8 down");
    require_ok(ds4_gpu_tensor_read(
        gguf_mxfp4_out,
        0,
        gguf_shared_actual,
        sizeof(gguf_shared_actual)),
        "GGUF MXFP4/A8 down read");
    for (uint32_t o = 0; o < GGUF_MXFP4_OUT; o++) {
        const uint8_t low = (uint8_t)(2u + o % 5u);
        const uint8_t high = (uint8_t)(low + 8u);
        float expected = 0.0f;
        for (uint32_t k = 0; k < GGUF_MXFP4_IN; k++) {
            const float xq =
                e4m3_to_f32(gguf_a8_values[k]) *
                e8m0_to_f32(gguf_a8_scales[k / 32u]);
            expected += xq * e2m1_to_f32(
                k % 32u < 16u ? low : high);
        }
        require_close(
            "GGUF MXFP4/A8 down",
            gguf_shared_actual[o],
            expected,
            2e-3f);
    }

    float gguf_moe_input[GGUF_MOE_DIM];
    int32_t gguf_moe_ids[GGUF_MOE_EXPERTS];
    float gguf_moe_route_weights[GGUF_MOE_EXPERTS];
    float route_sum = 0.0f;
    for (uint32_t i = 0; i < GGUF_MOE_EXPERTS; i++) {
        gguf_moe_input[i] = 0.0f;
        gguf_moe_ids[i] = (int32_t)(GGUF_MOE_EXPERTS - 1u - i);
        gguf_moe_route_weights[i] = (float)(i + 1u);
        route_sum += gguf_moe_route_weights[i];
    }
    for (uint32_t k = GGUF_MOE_EXPERTS; k < GGUF_MOE_DIM; k++) {
        gguf_moe_input[k] =
            ((int)((k * 17u) % 37u) - 18) * 0.015625f;
    }
    for (uint32_t i = 0; i < GGUF_MOE_EXPERTS; i++) {
        gguf_moe_route_weights[i] /= route_sum;
    }
    require_ok(ds4_gpu_tensor_write(
        gguf_moe_x, 0, gguf_moe_input, sizeof(gguf_moe_input)),
        "GGUF routed input write");
    require_ok(ds4_gpu_tensor_write(
        gguf_moe_selected, 0, gguf_moe_ids, sizeof(gguf_moe_ids)),
        "GGUF routed ids write");
    require_ok(ds4_gpu_tensor_write(
        gguf_moe_weights, 0,
        gguf_moe_route_weights, sizeof(gguf_moe_route_weights)),
        "GGUF routed weights write");
    require_ok(setenv("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION",
                      "1", 1) == 0,
               "disable routed pair fusion");
    require_ok(ds4_gpu_k3_routed_moe_one_tensor(
        gguf_moe_out,
        gguf_moe_gate,
        gguf_moe_up,
        gguf_moe_mid,
        gguf_moe_expert_out,
        model,
        MAP_BYTES,
        GGUF_IQ2_GATE_OFF,
        GGUF_IQ2_UP_OFF,
        GGUF_Q2_DOWN_OFF,
        GGUF_MOE_IQ2_EXPERT_BYTES,
        GGUF_MOE_IQ2_ROW_BYTES,
        GGUF_MOE_Q2_EXPERT_BYTES,
        GGUF_MOE_Q2_ROW_BYTES,
        GGUF_MOE_DIM,
        GGUF_MOE_DIM,
        GGUF_MOE_DIM,
        gguf_moe_selected,
        gguf_moe_weights,
        GGUF_MOE_EXPERTS,
        GGUF_MOE_EXPERTS,
        gguf_moe_x,
        1u,
        true), "GGUF K3 routed MoE reference");
    float gguf_moe_gate_values[GGUF_MOE_EXPERTS * GGUF_MOE_DIM];
    float gguf_moe_up_values[GGUF_MOE_EXPERTS * GGUF_MOE_DIM];
    float gguf_moe_mid_values[GGUF_MOE_EXPERTS * GGUF_MOE_DIM];
    float gguf_moe_output[GGUF_MOE_DIM];
    require_ok(ds4_gpu_tensor_read(
        gguf_moe_gate, 0,
        gguf_moe_gate_values, sizeof(gguf_moe_gate_values)),
        "GGUF routed gate read");
    require_ok(ds4_gpu_tensor_read(
        gguf_moe_up, 0,
        gguf_moe_up_values, sizeof(gguf_moe_up_values)),
        "GGUF routed up read");
    require_ok(ds4_gpu_tensor_read(
        gguf_moe_mid, 0,
        gguf_moe_mid_values, sizeof(gguf_moe_mid_values)),
        "GGUF routed mid read");
    require_ok(ds4_gpu_tensor_read(
        gguf_moe_out, 0,
        gguf_moe_output, sizeof(gguf_moe_output)),
        "GGUF routed output read");
    uint32_t nonzero_mid = 0;
    for (uint32_t row = 0; row < GGUF_MOE_EXPERTS; row++) {
        for (uint32_t k = 0; k < GGUF_MOE_DIM; k++) {
            const size_t index = (size_t)row * GGUF_MOE_DIM + k;
            const float gate_value = gguf_moe_gate_values[index];
            const float up_value = gguf_moe_up_values[index];
            const float expected =
                4.0f * tanhf(gate_value / 4.0f) /
                (1.0f + expf(-gate_value)) *
                25.0f * tanhf(up_value / 25.0f) *
                gguf_moe_route_weights[row];
            if (fabsf(gguf_moe_mid_values[index] - expected) > 4e-5f) {
                fprintf(stderr,
                        "GGUF routed SiTU row=%u col=%u gate=%g up=%g weight=%g\n",
                        row, k, gate_value, up_value,
                        gguf_moe_route_weights[row]);
            }
            require_close(
                "GGUF routed SiTU",
                gguf_moe_mid_values[index], expected, 4e-5f);
            nonzero_mid += fabsf(gguf_moe_mid_values[index]) > 1e-6f;
        }
    }
    require_ok(nonzero_mid != 0, "GGUF routed nonzero SiTU");
    for (uint32_t k = 0; k < GGUF_MOE_DIM; k++) {
        require_close(
            "GGUF routed zero down",
            gguf_moe_output[k], 0.0f, 1e-6f);
    }
    require_ok(unsetenv("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") == 0,
               "enable routed pair fusion");
    require_ok(ds4_gpu_k3_routed_moe_one_tensor(
        gguf_moe_out,
        gguf_moe_gate,
        gguf_moe_up,
        gguf_moe_mid,
        gguf_moe_expert_out,
        model,
        MAP_BYTES,
        GGUF_IQ2_GATE_OFF,
        GGUF_IQ2_UP_OFF,
        GGUF_Q2_DOWN_OFF,
        GGUF_MOE_IQ2_EXPERT_BYTES,
        GGUF_MOE_IQ2_ROW_BYTES,
        GGUF_MOE_Q2_EXPERT_BYTES,
        GGUF_MOE_Q2_ROW_BYTES,
        GGUF_MOE_DIM,
        GGUF_MOE_DIM,
        GGUF_MOE_DIM,
        gguf_moe_selected,
        gguf_moe_weights,
        GGUF_MOE_EXPERTS,
        GGUF_MOE_EXPERTS,
        gguf_moe_x,
        1u,
        true), "GGUF K3 routed MoE fused");
    float gguf_moe_fused_mid[GGUF_MOE_EXPERTS * GGUF_MOE_DIM];
    require_ok(ds4_gpu_tensor_read(
        gguf_moe_mid, 0,
        gguf_moe_fused_mid, sizeof(gguf_moe_fused_mid)),
        "GGUF routed fused mid read");
    for (size_t i = 0;
         i < (size_t)GGUF_MOE_EXPERTS * GGUF_MOE_DIM;
         i++) {
        require_close("GGUF routed fused SiTU",
                      gguf_moe_fused_mid[i],
                      gguf_moe_mid_values[i],
                      4e-5f);
    }

    require_ok(ds4_gpu_tensor_write(
        mm_batch_x, 0, mm_batch_input, sizeof(mm_batch_input)),
        "BF16 batch input write");
    require_ok(k3_gpu_matmul_bf16(
        mm_batch_out, model, MAP_BYTES, MAT_OFF, IN, OUT,
        mm_batch_x, MM_ROWS), "BF16 batch matmul");
    float mm_batch_actual[MM_ROWS * OUT];
    require_ok(ds4_gpu_tensor_read(
        mm_batch_out, 0, mm_batch_actual, sizeof(mm_batch_actual)),
        "BF16 batch matmul read");
    for (uint32_t r = 0; r < MM_ROWS; r++) {
        for (uint32_t o = 0; o < OUT; o++) {
            float expected = 0.0f;
            for (uint32_t k = 0; k < IN; k++) {
                expected += mm_batch_input[r * IN + k] *
                    bf16_to_f32(matrix[o * IN + k]);
            }
            require_close("BF16 batch matmul",
                          mm_batch_actual[r * OUT + o], expected, 3e-3f);
        }
    }
    require_ok(k3_gpu_matmul_q8_0_tensor(
        mm_batch_out, q8_weights, 0, IN, OUT,
        mm_batch_x, MM_ROWS), "cached Q8 batch matmul");
    require_ok(ds4_gpu_tensor_read(
        mm_batch_out, 0, mm_batch_actual, sizeof(mm_batch_actual)),
        "cached Q8 batch matmul read");
    for (uint32_t r = 0; r < MM_ROWS; r++) {
        for (uint32_t o = 0; o < OUT; o++) {
            uint16_t scale_bits;
            memcpy(&scale_bits, q8_data + o * 34u, sizeof(scale_bits));
            const float scale = f16_to_f32(scale_bits);
            const int8_t *quants =
                (const int8_t *)(q8_data + o * 34u + 2u);
            float expected = 0.0f;
            for (uint32_t k = 0; k < IN; k++) {
                expected += mm_batch_input[r * IN + k] *
                    scale * quants[k];
            }
            require_close("cached Q8 batch matmul",
                          mm_batch_actual[r * OUT + o], expected, 4e-3f);
        }
    }

    require_ok(k3_gpu_rms_norm_bf16(
        norm_out, x, model, MAP_BYTES, NORM_OFF, IN, 1, 1e-6f),
        "BF16 RMS norm");
    float norm_actual[IN];
    require_ok(ds4_gpu_tensor_read(
        norm_out, 0, norm_actual, sizeof(norm_actual)), "norm read");
    float sumsq = 0.0f;
    for (uint32_t k = 0; k < IN; k++) sumsq += input[k] * input[k];
    const float norm_scale = 1.0f / sqrtf(sumsq / IN + 1e-6f);
    for (uint32_t k = 0; k < IN; k++) {
        require_close("BF16 RMS norm", norm_actual[k],
                      input[k] * norm_scale * bf16_to_f32(norm_weight[k]),
                      2e-5f);
    }

    float gate_input[IN], up_input[IN];
    for (uint32_t k = 0; k < IN; k++) {
        gate_input[k] = ((int)k - 16) * 0.5f;
        up_input[k] = ((int)k - 8) * 1.25f;
    }
    require_ok(ds4_gpu_tensor_write(
        situ_gate, 0, gate_input, sizeof(gate_input)), "SiTU gate write");
    require_ok(ds4_gpu_tensor_write(
        situ_up, 0, up_input, sizeof(up_input)), "SiTU up write");
    require_ok(k3_gpu_situ(
        situ_out, situ_gate, situ_up, IN, 1, 4.0f, 25.0f), "SiTU");
    float situ_actual[IN];
    require_ok(ds4_gpu_tensor_read(
        situ_out, 0, situ_actual, sizeof(situ_actual)), "SiTU read");
    for (uint32_t k = 0; k < IN; k++) {
        const float expected =
            4.0f * tanhf(gate_input[k] / 4.0f) /
            (1.0f + expf(-gate_input[k])) *
            25.0f * tanhf(up_input[k] / 25.0f);
        require_close("SiTU", situ_actual[k], expected, 3e-5f);
    }

    float fp8_boundary[IN] = {0};
    fp8_boundary[0] = 448.0f;
    fp8_boundary[1] = 15.0f / 1024.0f;
    fp8_boundary[2] = 1.0f / 1024.0f;
    fp8_boundary[3] = 1.0f / 512.0f;
    fp8_boundary[4] = 3.0f / 1024.0f;
    fp8_boundary[5] = 5.0f / 1024.0f;
    fp8_boundary[6] = 1.0f;
    fp8_boundary[7] = 1.0625f;
    fp8_boundary[8] = 1.1875f;
    fp8_boundary[9] = -1.1875f;
    require_ok(ds4_gpu_tensor_write(
        x, 0, fp8_boundary, sizeof(fp8_boundary)), "FP8 boundary write");
    require_ok(k3_gpu_mxfp8_quantize(
        mx_values, mx_scales, x, IN, 1), "MXFP8 boundary quantization");
    uint8_t boundary_values[IN], boundary_scale;
    require_ok(ds4_gpu_tensor_read(
        mx_values, 0, boundary_values, sizeof(boundary_values)),
        "MXFP8 boundary value read");
    require_ok(ds4_gpu_tensor_read(
        mx_scales, 0, &boundary_scale, sizeof(boundary_scale)),
        "MXFP8 boundary scale read");
    const uint8_t expected_boundary[10] = {
        126u, 8u, 0u, 1u, 2u, 2u, 56u, 56u, 58u, 186u,
    };
    if (boundary_scale != 127u) {
        fprintf(stderr, "MXFP8 boundary scale: got %u expected 127\n",
                boundary_scale);
        return 1;
    }
    for (uint32_t k = 0; k < 10u; k++) {
        if (boundary_values[k] != expected_boundary[k]) {
            fprintf(stderr,
                    "MXFP8 boundary %u: got %u expected %u\n",
                    k, boundary_values[k], expected_boundary[k]);
            return 1;
        }
    }
    require_ok(ds4_gpu_tensor_write(
        x, 0, input, sizeof(input)), "MXFP8 input restore");
    require_ok(k3_gpu_mxfp8_quantize(
        mx_values, mx_scales, x, IN, 1), "MXFP8 quantization");
    uint8_t qvalues[IN], qscale;
    require_ok(ds4_gpu_tensor_read(mx_values, 0, qvalues, sizeof(qvalues)),
               "MXFP8 value read");
    require_ok(ds4_gpu_tensor_read(mx_scales, 0, &qscale, sizeof(qscale)),
               "MXFP8 scale read");
    float dequant[IN];
    for (uint32_t k = 0; k < IN; k++) {
        dequant[k] = e4m3_to_f32(qvalues[k]) * e8m0_to_f32(qscale);
        require_close("MXFP8", dequant[k], input[k], 0.065f);
    }

    require_ok(k3_gpu_mxfp4_pair_situ(
        mx_gate, mx_up, mx_mid, model, MAP_BYTES,
        GATE_PACKED_OFF, GATE_SCALE_OFF,
        UP_PACKED_OFF, UP_SCALE_OFF,
        mx_values, mx_scales, IN, MX_OUT, 1, 4.0f, 25.0f),
        "MXFP4 pair");
    float gate_actual[MX_OUT], up_actual[MX_OUT], mid_actual[MX_OUT];
    require_ok(ds4_gpu_tensor_read(
        mx_gate, 0, gate_actual, sizeof(gate_actual)), "MX gate read");
    require_ok(ds4_gpu_tensor_read(
        mx_up, 0, up_actual, sizeof(up_actual)), "MX up read");
    require_ok(ds4_gpu_tensor_read(
        mx_mid, 0, mid_actual, sizeof(mid_actual)), "MX mid read");
    for (uint32_t o = 0; o < MX_OUT; o++) {
        float expected_gate = 0.0f, expected_up = 0.0f;
        for (uint32_t k = 0; k < IN; k++) {
            expected_gate += dequant[k] * e2m1_to_f32(1u + o);
            expected_up += dequant[k] * e2m1_to_f32(5u - o);
        }
        const float expected_mid =
            4.0f * tanhf(expected_gate / 4.0f) /
            (1.0f + expf(-expected_gate)) *
            25.0f * tanhf(expected_up / 25.0f);
        require_close("MXFP4 gate", gate_actual[o], expected_gate, 2e-4f);
        require_close("MXFP4 up", up_actual[o], expected_up, 2e-4f);
        require_close("MXFP4 SiTU", mid_actual[o], expected_mid, 2e-4f);
    }

    ds4_gpu_tensor *cached_weights = ds4_gpu_tensor_alloc(MAP_BYTES);
    require_ok(cached_weights != NULL, "cached MXFP4 weight allocation");
    require_ok(ds4_gpu_tensor_write(
        cached_weights, 0, model, MAP_BYTES), "cached MXFP4 weight write");
    require_ok(k3_gpu_mxfp4_pair_situ_tensor(
        mx_gate, mx_up, mx_mid, cached_weights,
        GATE_PACKED_OFF, GATE_SCALE_OFF,
        UP_PACKED_OFF, UP_SCALE_OFF,
        mx_values, mx_scales, IN, MX_OUT, 1, 4.0f, 25.0f),
        "cached MXFP4 pair");
    require_ok(ds4_gpu_tensor_read(
        mx_gate, 0, gate_actual, sizeof(gate_actual)),
        "cached MX gate read");
    require_ok(ds4_gpu_tensor_read(
        mx_up, 0, up_actual, sizeof(up_actual)),
        "cached MX up read");
    require_ok(ds4_gpu_tensor_read(
        mx_mid, 0, mid_actual, sizeof(mid_actual)),
        "cached MX mid read");
    for (uint32_t o = 0; o < MX_OUT; o++) {
        float expected_gate = 0.0f, expected_up = 0.0f;
        for (uint32_t k = 0; k < IN; k++) {
            expected_gate += dequant[k] * e2m1_to_f32(1u + o);
            expected_up += dequant[k] * e2m1_to_f32(5u - o);
        }
        const float expected_mid =
            4.0f * tanhf(expected_gate / 4.0f) /
            (1.0f + expf(-expected_gate)) *
            25.0f * tanhf(expected_up / 25.0f);
        require_close(
            "cached MXFP4 gate", gate_actual[o], expected_gate, 2e-4f);
        require_close(
            "cached MXFP4 up", up_actual[o], expected_up, 2e-4f);
        require_close(
            "cached MXFP4 SiTU", mid_actual[o], expected_mid, 2e-4f);
    }

    require_ok(k3_gpu_mxfp4_down(
        down_out, model, MAP_BYTES, DOWN_PACKED_OFF, DOWN_SCALE_OFF,
        mx_values, mx_scales, IN, MX_OUT, 1, 0.25f, 0),
        "MXFP4 down");
    float down_actual[MX_OUT];
    require_ok(ds4_gpu_tensor_read(
        down_out, 0, down_actual, sizeof(down_actual)), "MX down read");
    float down_expected = 0.0f;
    for (uint32_t k = 0; k < IN; k++) down_expected += dequant[k];
    down_expected *= 0.25f;
    for (uint32_t o = 0; o < MX_OUT; o++) {
        require_close("MXFP4 down", down_actual[o], down_expected, 2e-4f);
    }
    require_ok(k3_gpu_mxfp4_down_tensor(
        down_out, cached_weights, DOWN_PACKED_OFF, DOWN_SCALE_OFF,
        mx_values, mx_scales, IN, MX_OUT, 1, 0.25f, 0),
        "cached MXFP4 down");
    require_ok(ds4_gpu_tensor_read(
        down_out, 0, down_actual, sizeof(down_actual)),
        "cached MX down read");
    for (uint32_t o = 0; o < MX_OUT; o++) {
        require_close(
            "cached MXFP4 down", down_actual[o], down_expected, 2e-4f);
    }

    require_ok(k3_gpu_mxfp8_quantize(
        down_values, down_scales, mx_mid, MX_OUT, 1) == 0,
        "MXFP8 rejects a non-32 width");

    float router_logits[EXPERTS] = {
        -1.0f, 0.2f, 2.0f, -0.5f, 0.8f, 1.5f, 0.1f, 1.0f,
    };
    require_ok(ds4_gpu_tensor_write(
        logits, 0, router_logits, sizeof(router_logits)), "logit write");
    require_ok(k3_gpu_router_topk(
        selected, selected_weights, logits, model, MAP_BYTES,
        ROUTER_BIAS_OFF, 1, EXPERTS, TOP_K), "router top-k");
    int32_t ids[TOP_K];
    float route_weights[TOP_K];
    require_ok(ds4_gpu_tensor_read(selected, 0, ids, sizeof(ids)),
               "router id read");
    require_ok(ds4_gpu_tensor_read(
        selected_weights, 0, route_weights, sizeof(route_weights)),
        "router weight read");
    const int32_t expected_ids[TOP_K] = {2, 5, 7, 6};
    float weight_sum = 0.0f;
    for (uint32_t i = 0; i < TOP_K; i++) {
        if (ids[i] != expected_ids[i]) {
            fprintf(stderr, "router id %u: got %d expected %d\n",
                    i, ids[i], expected_ids[i]);
            return 1;
        }
        weight_sum += route_weights[i];
    }
    require_close("router normalized sum", weight_sum, 1.0f, 1e-6f);

    const float tied_router_logits[EXPERTS] = {
        2.0f, 2.0f, 1.0f, 1.0f, -10.0f, -10.0f, -10.0f, -10.0f,
    };
    require_ok(ds4_gpu_tensor_write(
        logits, 0, tied_router_logits, sizeof(tied_router_logits)),
        "tied logit write");
    require_ok(k3_gpu_router_topk(
        selected, selected_weights, logits, model, MAP_BYTES,
        ROUTER_BIAS_OFF, 1, EXPERTS, TOP_K), "tied router top-k");
    require_ok(ds4_gpu_tensor_read(selected, 0, ids, sizeof(ids)),
               "tied router id read");
    const int32_t tied_expected_ids[TOP_K] = {0, 1, 2, 3};
    for (uint32_t i = 0; i < TOP_K; i++) {
        if (ids[i] != tied_expected_ids[i]) {
            fprintf(stderr,
                    "tied router id %u: got %d expected %d\n",
                    i, ids[i], tied_expected_ids[i]);
            return 1;
        }
    }

    float prefix_input[IN], delta_input[IN], block_input[2 * IN];
    for (uint32_t k = 0; k < IN; k++) {
        prefix_input[k] = 0.5f + 0.01f * (float)k;
        delta_input[k] = 0.125f;
        block_input[k] = -0.25f + 0.005f * (float)k;
        block_input[IN + k] = 0.0f;
    }
    require_ok(ds4_gpu_tensor_write(
        prefix, 0, prefix_input, sizeof(prefix_input)), "prefix write");
    require_ok(ds4_gpu_tensor_write(
        delta, 0, delta_input, sizeof(delta_input)), "delta write");
    require_ok(ds4_gpu_tensor_write(
        blocks, 0, block_input, sizeof(block_input)), "blocks write");
    require_ok(k3_gpu_attn_res(
        attn_out, prefix, delta, blocks, model, MAP_BYTES,
        ATTN_NORM_OFF, ATTN_PROJ_OFF, UINT64_MAX,
        IN, 1, 2, 1, -1, 0, 1e-6f, 0.0f), "AttnRes");
    float attn_actual[IN];
    require_ok(ds4_gpu_tensor_read(
        attn_out, 0, attn_actual, sizeof(attn_actual)), "AttnRes read");
    float rounded_prefix[IN];
    float block_ss = 0.0f, block_dot = 0.0f;
    float prefix_ss = 0.0f, prefix_dot = 0.0f;
    for (uint32_t k = 0; k < IN; k++) {
        rounded_prefix[k] =
            bf16_to_f32(f32_to_bf16(prefix_input[k] + delta_input[k]));
        block_ss += block_input[k] * block_input[k];
        block_dot += block_input[k];
        prefix_ss += rounded_prefix[k] * rounded_prefix[k];
        prefix_dot += rounded_prefix[k];
    }
    const float block_logit =
        block_dot / sqrtf(block_ss / IN + 1e-6f);
    const float prefix_logit =
        prefix_dot / sqrtf(prefix_ss / IN + 1e-6f);
    const float max_logit = fmaxf(block_logit, prefix_logit);
    const float block_exp = expf(block_logit - max_logit);
    const float prefix_exp = expf(prefix_logit - max_logit);
    const float block_mix = block_exp / (block_exp + prefix_exp);
    const float prefix_mix = prefix_exp / (block_exp + prefix_exp);
    for (uint32_t k = 0; k < IN; k++) {
        require_close("AttnRes", attn_actual[k],
                      block_mix * block_input[k] +
                      prefix_mix * rounded_prefix[k], 2e-5f);
    }
    require_ok(ds4_gpu_tensor_write(
        prefix, 0, prefix_input, sizeof(prefix_input)),
        "prefix-only write");
    require_ok(k3_gpu_attn_res(
        attn_out, prefix, NULL, blocks, model, MAP_BYTES,
        ATTN_NORM_OFF, ATTN_PROJ_OFF, UINT64_MAX,
        IN, 1, 2, 0, -1, 0, 1e-6f, 0.0f), "prefix-only AttnRes");
    require_ok(ds4_gpu_tensor_read(
        attn_out, 0, attn_actual, sizeof(attn_actual)),
        "prefix-only AttnRes read");
    for (uint32_t k = 0; k < IN; k++) {
        require_close("prefix-only AttnRes", attn_actual[k],
                      bf16_to_f32(f32_to_bf16(prefix_input[k])), 0.0f);
    }
    require_ok(ds4_gpu_tensor_write(
        prefix, 0, prefix_input, sizeof(prefix_input)),
        "read-only prefix write");
    require_ok(k3_gpu_attn_res(
        attn_out, prefix, NULL, blocks, model, MAP_BYTES,
        ATTN_NORM_OFF, ATTN_PROJ_OFF, UINT64_MAX,
        IN, 1, 2, 0, -1, 1, 1e-6f, 0.0f),
        "read-only AttnRes");
    require_ok(ds4_gpu_tensor_read(
        prefix, 0, attn_actual, sizeof(attn_actual)),
        "read-only prefix read");
    for (uint32_t k = 0; k < IN; k++) {
        require_close("read-only prefix", attn_actual[k],
                      prefix_input[k], 0.0f);
    }

    float kda_ones[KDA_DIM], kda_zero[KDA_DIM], zero_beta = 0.0f;
    for (uint32_t d = 0; d < KDA_DIM; d++) {
        kda_ones[d] = 1.0f;
        kda_zero[d] = 0.0f;
    }
    require_ok(ds4_gpu_tensor_write(
        kda_q, 0, kda_ones, sizeof(kda_ones)), "KDA Q write");
    require_ok(ds4_gpu_tensor_write(
        kda_k, 0, kda_ones, sizeof(kda_ones)), "KDA K write");
    require_ok(ds4_gpu_tensor_write(
        kda_v, 0, kda_ones, sizeof(kda_ones)), "KDA V write");
    require_ok(ds4_gpu_tensor_write(
        kda_gate, 0, kda_zero, sizeof(kda_zero)), "KDA gate write");
    require_ok(ds4_gpu_tensor_write(
        kda_output_gate, 0, kda_zero, sizeof(kda_zero)),
        "KDA output gate write");
    require_ok(ds4_gpu_tensor_write(
        kda_beta, 0, &zero_beta, sizeof(zero_beta)), "KDA beta write");
    require_ok(ds4_gpu_tensor_fill_f32(
        kda_conv_state, 0.0f, 3u * 3u * KDA_DIM), "KDA conv state clear");
    require_ok(ds4_gpu_tensor_fill_f32(
        kda_state, 0.0f, KDA_DIM * KDA_DIM), "KDA state clear");
    require_ok(k3_gpu_kda_decode(
        kda_out, kda_conv_state, kda_state,
        kda_q, kda_k, kda_v, kda_gate, kda_beta, kda_output_gate,
        model, MAP_BYTES,
        KDA_Q_CONV_OFF, KDA_K_CONV_OFF, KDA_V_CONV_OFF,
        KDA_A_LOG_OFF, KDA_DT_BIAS_OFF, KDA_NORM_OFF,
        1, 1, -5.0f, 1e-5f), "KDA decode");
    float kda_actual[KDA_DIM];
    require_ok(ds4_gpu_tensor_read(
        kda_out, 0, kda_actual, sizeof(kda_actual)), "KDA output read");
    const float silu_one = 1.0f / (1.0f + expf(-1.0f));
    const float raw_kda = 0.5f * silu_one / sqrtf((float)KDA_DIM);
    const float expected_kda =
        0.5f * raw_kda / sqrtf(raw_kda * raw_kda + 1e-5f);
    for (uint32_t d = 0; d < KDA_DIM; d++) {
        require_close("KDA decode", kda_actual[d], expected_kda, 2e-5f);
    }

    float mla_q_input[192] = {0};
    float mla_kv_input[576] = {0};
    float mla_gate_input[128] = {0};
    uint16_t mla_cache_input[4 * 576] = {0};
    mla_q_input[0] = 1.0f;
    mla_kv_input[0] = 2.0f;
    mla_cache_input[0] = f32_to_bf16(1.0f);
    require_ok(ds4_gpu_tensor_write(
        mla_q, 0, mla_q_input, sizeof(mla_q_input)), "MLA Q write");
    require_ok(ds4_gpu_tensor_write(
        mla_kv, 0, mla_kv_input, sizeof(mla_kv_input)), "MLA KV write");
    require_ok(ds4_gpu_tensor_write(
        mla_gate, 0, mla_gate_input, sizeof(mla_gate_input)),
        "MLA gate write");
    require_ok(ds4_gpu_tensor_write(
        mla_cache, 0, mla_cache_input, sizeof(mla_cache_input)),
        "MLA cache write");
    require_ok(k3_gpu_mla_decode(
        mla_out, mla_cache, mla_q_latent, mla_partial, mla_stats,
        mla_q, mla_kv, mla_gate,
        model, MAP_BYTES, MLA_KV_B_OFF,
        1, 1, 4, 1), "MLA decode");
    float mla_actual[128];
    require_ok(ds4_gpu_tensor_read(
        mla_out, 0, mla_actual, sizeof(mla_actual)), "MLA output read");
    const float mla_s0 = 1.0f / sqrtf(192.0f);
    const float mla_s1 = 2.0f / sqrtf(192.0f);
    const float mla_max = fmaxf(mla_s0, mla_s1);
    const float mla_e0 = expf(mla_s0 - mla_max);
    const float mla_e1 = expf(mla_s1 - mla_max);
    const float mla_expected =
        0.5f * (mla_e0 + 2.0f * mla_e1) / (mla_e0 + mla_e1);
    for (uint32_t d = 0; d < 128u; d++) {
        require_close("MLA decode", mla_actual[d], mla_expected, 2e-5f);
    }

    /*
     * The prompt-specialized kernels must produce the same outputs and final
     * state as token-at-a-time generation. This catches causal/state-layout
     * drift between the two independently optimized paths.
     */
    enum { CONSISTENCY_TOKENS = 2 };
    float consistency_q[CONSISTENCY_TOKENS * KDA_DIM];
    float consistency_k[CONSISTENCY_TOKENS * KDA_DIM];
    float consistency_v[CONSISTENCY_TOKENS * KDA_DIM];
    float consistency_gate[CONSISTENCY_TOKENS * KDA_DIM];
    float consistency_output_gate[CONSISTENCY_TOKENS * KDA_DIM];
    float consistency_beta[CONSISTENCY_TOKENS];
    for (uint32_t t = 0; t < CONSISTENCY_TOKENS; t++) {
        consistency_beta[t] = -0.25f + 0.5f * (float)t;
        for (uint32_t d = 0; d < KDA_DIM; d++) {
            const uint32_t i = t * KDA_DIM + d;
            consistency_q[i] = 0.1f + 0.002f * (float)(d % 17u) +
                               0.03f * (float)t;
            consistency_k[i] = -0.08f + 0.001f * (float)(d % 23u) +
                               0.02f * (float)t;
            consistency_v[i] = 0.05f - 0.0015f * (float)(d % 13u) +
                               0.04f * (float)t;
            consistency_gate[i] = -0.2f + 0.003f * (float)(d % 11u);
            consistency_output_gate[i] =
                0.15f - 0.002f * (float)(d % 7u);
        }
    }
    ds4_gpu_tensor *kda_prefill_q =
        ds4_gpu_tensor_alloc(sizeof(consistency_q));
    ds4_gpu_tensor *kda_prefill_k =
        ds4_gpu_tensor_alloc(sizeof(consistency_k));
    ds4_gpu_tensor *kda_prefill_v =
        ds4_gpu_tensor_alloc(sizeof(consistency_v));
    ds4_gpu_tensor *kda_prefill_gate =
        ds4_gpu_tensor_alloc(sizeof(consistency_gate));
    ds4_gpu_tensor *kda_prefill_output_gate =
        ds4_gpu_tensor_alloc(sizeof(consistency_output_gate));
    ds4_gpu_tensor *kda_prefill_beta =
        ds4_gpu_tensor_alloc(sizeof(consistency_beta));
    ds4_gpu_tensor *kda_prefill_conv =
        ds4_gpu_tensor_alloc(3u * 3u * KDA_DIM * sizeof(float));
    ds4_gpu_tensor *kda_prefill_state =
        ds4_gpu_tensor_alloc(KDA_DIM * KDA_DIM * sizeof(float));
    ds4_gpu_tensor *kda_prefill_out =
        ds4_gpu_tensor_alloc(sizeof(consistency_q));
    require_ok(kda_prefill_q && kda_prefill_k && kda_prefill_v &&
               kda_prefill_gate && kda_prefill_output_gate &&
               kda_prefill_beta && kda_prefill_conv &&
               kda_prefill_state && kda_prefill_out,
               "KDA prefill consistency allocation");
    require_ok(ds4_gpu_tensor_fill_f32(
        kda_conv_state, 0.0f, 3u * 3u * KDA_DIM), "KDA decode conv reset");
    require_ok(ds4_gpu_tensor_fill_f32(
        kda_state, 0.0f, KDA_DIM * KDA_DIM), "KDA decode state reset");
    float kda_decode_outputs[CONSISTENCY_TOKENS * KDA_DIM];
    for (uint32_t t = 0; t < CONSISTENCY_TOKENS; t++) {
        const uint32_t off = t * KDA_DIM;
        require_ok(ds4_gpu_tensor_write(
            kda_q, 0, consistency_q + off, KDA_DIM * sizeof(float)),
            "KDA consistency Q write");
        require_ok(ds4_gpu_tensor_write(
            kda_k, 0, consistency_k + off, KDA_DIM * sizeof(float)),
            "KDA consistency K write");
        require_ok(ds4_gpu_tensor_write(
            kda_v, 0, consistency_v + off, KDA_DIM * sizeof(float)),
            "KDA consistency V write");
        require_ok(ds4_gpu_tensor_write(
            kda_gate, 0, consistency_gate + off, KDA_DIM * sizeof(float)),
            "KDA consistency gate write");
        require_ok(ds4_gpu_tensor_write(
            kda_output_gate, 0, consistency_output_gate + off,
            KDA_DIM * sizeof(float)), "KDA consistency output gate write");
        require_ok(ds4_gpu_tensor_write(
            kda_beta, 0, consistency_beta + t, sizeof(float)),
            "KDA consistency beta write");
        require_ok(k3_gpu_kda_decode(
            kda_out, kda_conv_state, kda_state,
            kda_q, kda_k, kda_v, kda_gate, kda_beta, kda_output_gate,
            model, MAP_BYTES,
            KDA_Q_CONV_OFF, KDA_K_CONV_OFF, KDA_V_CONV_OFF,
            KDA_A_LOG_OFF, KDA_DT_BIAS_OFF, KDA_NORM_OFF,
            1, 1, -5.0f, 1e-5f), "KDA consistency decode");
        require_ok(ds4_gpu_tensor_read(
            kda_out, 0, kda_decode_outputs + off,
            KDA_DIM * sizeof(float)), "KDA consistency decode read");
    }
    require_ok(ds4_gpu_tensor_write(
        kda_prefill_q, 0, consistency_q, sizeof(consistency_q)),
        "KDA prefill Q write");
    require_ok(ds4_gpu_tensor_write(
        kda_prefill_k, 0, consistency_k, sizeof(consistency_k)),
        "KDA prefill K write");
    require_ok(ds4_gpu_tensor_write(
        kda_prefill_v, 0, consistency_v, sizeof(consistency_v)),
        "KDA prefill V write");
    require_ok(ds4_gpu_tensor_write(
        kda_prefill_gate, 0, consistency_gate, sizeof(consistency_gate)),
        "KDA prefill gate write");
    require_ok(ds4_gpu_tensor_write(
        kda_prefill_output_gate, 0, consistency_output_gate,
        sizeof(consistency_output_gate)), "KDA prefill output gate write");
    require_ok(ds4_gpu_tensor_write(
        kda_prefill_beta, 0, consistency_beta, sizeof(consistency_beta)),
        "KDA prefill beta write");
    require_ok(ds4_gpu_tensor_fill_f32(
        kda_prefill_conv, 0.0f, 3u * 3u * KDA_DIM),
        "KDA prefill conv reset");
    require_ok(ds4_gpu_tensor_fill_f32(
        kda_prefill_state, 0.0f, KDA_DIM * KDA_DIM),
        "KDA prefill state reset");
    require_ok(k3_gpu_kda_prefill(
        kda_prefill_out, kda_prefill_conv, kda_prefill_state,
        kda_prefill_q, kda_prefill_k, kda_prefill_v, kda_prefill_gate,
        kda_prefill_beta, kda_prefill_output_gate,
        model, MAP_BYTES,
        KDA_Q_CONV_OFF, KDA_K_CONV_OFF, KDA_V_CONV_OFF,
        KDA_A_LOG_OFF, KDA_DT_BIAS_OFF, KDA_NORM_OFF,
        1, CONSISTENCY_TOKENS, -5.0f, 1e-5f),
        "KDA consistency prefill");
    float kda_prefill_outputs[CONSISTENCY_TOKENS * KDA_DIM];
    require_ok(ds4_gpu_tensor_read(
        kda_prefill_out, 0, kda_prefill_outputs,
        sizeof(kda_prefill_outputs)), "KDA consistency prefill read");
    for (uint32_t i = 0; i < CONSISTENCY_TOKENS * KDA_DIM; i++) {
        require_close("KDA prefill/decode", kda_prefill_outputs[i],
                      kda_decode_outputs[i], 5e-5f);
    }

    float mla_consistency_q[CONSISTENCY_TOKENS * 192] = {0};
    float mla_consistency_kv[CONSISTENCY_TOKENS * 576] = {0};
    float mla_consistency_gate[CONSISTENCY_TOKENS * 128] = {0};
    mla_consistency_q[0] = 0.7f;
    mla_consistency_q[192] = 1.1f;
    mla_consistency_kv[0] = 0.6f;
    mla_consistency_kv[576] = 1.4f;
    ds4_gpu_tensor *mla_prefill_q =
        ds4_gpu_tensor_alloc(sizeof(mla_consistency_q));
    ds4_gpu_tensor *mla_prefill_kv =
        ds4_gpu_tensor_alloc(sizeof(mla_consistency_kv));
    ds4_gpu_tensor *mla_prefill_gate =
        ds4_gpu_tensor_alloc(sizeof(mla_consistency_gate));
    ds4_gpu_tensor *mla_prefill_cache =
        ds4_gpu_tensor_alloc(4u * 576u * sizeof(uint16_t));
    ds4_gpu_tensor *mla_prefill_latent =
        ds4_gpu_tensor_alloc(CONSISTENCY_TOKENS * 512u * sizeof(float));
    ds4_gpu_tensor *mla_prefill_out =
        ds4_gpu_tensor_alloc(CONSISTENCY_TOKENS * 128u * sizeof(float));
    require_ok(mla_prefill_q && mla_prefill_kv && mla_prefill_gate &&
               mla_prefill_cache && mla_prefill_latent && mla_prefill_out,
               "MLA prefill consistency allocation");
    require_ok(ds4_gpu_tensor_fill_f32(
        mla_cache, 0.0f, 4u * 576u / 2u), "MLA decode cache reset");
    float mla_decode_outputs[CONSISTENCY_TOKENS * 128];
    for (uint32_t t = 0; t < CONSISTENCY_TOKENS; t++) {
        require_ok(ds4_gpu_tensor_write(
            mla_q, 0, mla_consistency_q + t * 192u,
            192u * sizeof(float)), "MLA consistency Q write");
        require_ok(ds4_gpu_tensor_write(
            mla_kv, 0, mla_consistency_kv + t * 576u,
            576u * sizeof(float)), "MLA consistency KV write");
        require_ok(ds4_gpu_tensor_write(
            mla_gate, 0, mla_consistency_gate + t * 128u,
            128u * sizeof(float)), "MLA consistency gate write");
        require_ok(k3_gpu_mla_decode(
            mla_out, mla_cache, mla_q_latent, mla_partial, mla_stats,
            mla_q, mla_kv, mla_gate,
            model, MAP_BYTES, MLA_KV_B_OFF,
            1, t, 4, 1), "MLA consistency decode");
        require_ok(ds4_gpu_tensor_read(
            mla_out, 0, mla_decode_outputs + t * 128u,
            128u * sizeof(float)), "MLA consistency decode read");
    }
    require_ok(ds4_gpu_tensor_write(
        mla_prefill_q, 0, mla_consistency_q, sizeof(mla_consistency_q)),
        "MLA prefill Q write");
    require_ok(ds4_gpu_tensor_write(
        mla_prefill_kv, 0, mla_consistency_kv, sizeof(mla_consistency_kv)),
        "MLA prefill KV write");
    require_ok(ds4_gpu_tensor_write(
        mla_prefill_gate, 0, mla_consistency_gate,
        sizeof(mla_consistency_gate)), "MLA prefill gate write");
    require_ok(ds4_gpu_tensor_fill_f32(
        mla_prefill_cache, 0.0f, 4u * 576u / 2u),
        "MLA prefill cache reset");
    require_ok(k3_gpu_mla_prefill(
        mla_prefill_out, mla_prefill_cache, mla_prefill_latent,
        mla_prefill_q, mla_prefill_kv, mla_prefill_gate,
        model, MAP_BYTES, MLA_KV_B_OFF,
        1, 0, CONSISTENCY_TOKENS, 4),
        "MLA consistency prefill");
    float mla_prefill_outputs[CONSISTENCY_TOKENS * 128];
    require_ok(ds4_gpu_tensor_read(
        mla_prefill_out, 0, mla_prefill_outputs,
        sizeof(mla_prefill_outputs)), "MLA consistency prefill read");
    for (uint32_t i = 0; i < CONSISTENCY_TOKENS * 128u; i++) {
        require_close("MLA prefill/decode", mla_prefill_outputs[i],
                      mla_decode_outputs[i], 5e-5f);
    }

    ds4_gpu_tensor_free(mla_prefill_out);
    ds4_gpu_tensor_free(mla_prefill_latent);
    ds4_gpu_tensor_free(mla_prefill_cache);
    ds4_gpu_tensor_free(mla_prefill_gate);
    ds4_gpu_tensor_free(mla_prefill_kv);
    ds4_gpu_tensor_free(mla_prefill_q);
    ds4_gpu_tensor_free(kda_prefill_out);
    ds4_gpu_tensor_free(kda_prefill_state);
    ds4_gpu_tensor_free(kda_prefill_conv);
    ds4_gpu_tensor_free(kda_prefill_beta);
    ds4_gpu_tensor_free(kda_prefill_output_gate);
    ds4_gpu_tensor_free(kda_prefill_gate);
    ds4_gpu_tensor_free(kda_prefill_v);
    ds4_gpu_tensor_free(kda_prefill_k);
    ds4_gpu_tensor_free(kda_prefill_q);

    ds4_gpu_tensor_free(mla_out);
    ds4_gpu_tensor_free(mla_stats);
    ds4_gpu_tensor_free(mla_partial);
    ds4_gpu_tensor_free(mla_q_latent);
    ds4_gpu_tensor_free(mla_cache);
    ds4_gpu_tensor_free(mla_gate);
    ds4_gpu_tensor_free(mla_kv);
    ds4_gpu_tensor_free(mla_q);
    ds4_gpu_tensor_free(kda_out);
    ds4_gpu_tensor_free(kda_state);
    ds4_gpu_tensor_free(kda_conv_state);
    ds4_gpu_tensor_free(kda_output_gate);
    ds4_gpu_tensor_free(kda_beta);
    ds4_gpu_tensor_free(kda_gate);
    ds4_gpu_tensor_free(kda_v);
    ds4_gpu_tensor_free(kda_k);
    ds4_gpu_tensor_free(kda_q);
    ds4_gpu_tensor_free(attn_out);
    ds4_gpu_tensor_free(blocks);
    ds4_gpu_tensor_free(delta);
    ds4_gpu_tensor_free(prefix);
    ds4_gpu_tensor_free(selected_weights);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(down_out);
    ds4_gpu_tensor_free(down_scales);
    ds4_gpu_tensor_free(down_values);
    ds4_gpu_tensor_free(mx_mid);
    ds4_gpu_tensor_free(mx_up);
    ds4_gpu_tensor_free(mx_gate);
    ds4_gpu_tensor_free(cached_weights);
    ds4_gpu_tensor_free(mx_scales);
    ds4_gpu_tensor_free(mx_values);
    ds4_gpu_tensor_free(situ_out);
    ds4_gpu_tensor_free(situ_up);
    ds4_gpu_tensor_free(situ_gate);
    ds4_gpu_tensor_free(norm_out);
    ds4_gpu_tensor_free(gguf_moe_weights);
    ds4_gpu_tensor_free(gguf_moe_selected);
    ds4_gpu_tensor_free(gguf_moe_out);
    ds4_gpu_tensor_free(gguf_moe_expert_out);
    ds4_gpu_tensor_free(gguf_moe_mid);
    ds4_gpu_tensor_free(gguf_moe_up);
    ds4_gpu_tensor_free(gguf_moe_gate);
    ds4_gpu_tensor_free(gguf_moe_x);
    ds4_gpu_tensor_free(gguf_mxfp8_scales);
    ds4_gpu_tensor_free(gguf_mxfp8_values);
    ds4_gpu_tensor_free(gguf_mxfp4_out);
    ds4_gpu_tensor_free(gguf_mxfp4_x);
    ds4_gpu_tensor_free(mm_batch_out);
    ds4_gpu_tensor_free(mm_batch_x);
    ds4_gpu_tensor_free(mm_out);
    ds4_gpu_tensor_free(q8_weights);
    ds4_gpu_tensor_free(bf16_out);
    ds4_gpu_tensor_free(bf16_in);
    ds4_gpu_tensor_free(x);
    require_ok(ds4_gpu_release_model_views(), "model view release");
    require_ok(ds4_gpu_set_model_map(model, MAP_BYTES), "model view remap");
    require_ok(ds4_gpu_release_model_views(), "remapped model view release");
    ds4_gpu_cleanup();
    munmap(model, MAP_BYTES);
    puts("test_k3_metal: ok");
    return 0;
}
