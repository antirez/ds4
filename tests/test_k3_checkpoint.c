#define _POSIX_C_SOURCE 200809L

#include "ds4_gpu.h"
#include "k3_metal.h"
#include "k3_model.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
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

static float e8m0_to_f32(uint8_t value) {
    return exp2f((float)((int)value - 127));
}

static float e4m3_to_f32(uint8_t value) {
    const int sign = value >> 7;
    const int exponent = (value >> 3) & 15;
    const int mantissa = value & 7;
    const float magnitude = exponent == 0
        ? (float)mantissa * 0x1p-9f
        : (1.0f + (float)mantissa * 0.125f) *
          exp2f((float)(exponent - 7));
    return sign ? -magnitude : magnitude;
}

static float e2m1_to_f32(uint8_t value) {
    static const float table[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
       -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f,
    };
    return table[value & 15u];
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static float checkpoint_mxfp4_dot(
        const uint8_t *base,
        const k3_st_tensor *packed,
        const k3_st_tensor *scales,
        uint32_t output,
        uint32_t in_dim,
        const uint8_t *x_values,
        const uint8_t *x_scales) {
    const uint8_t *weight =
        base + packed->offset + (uint64_t)output * (in_dim / 2u);
    const uint8_t *weight_scales =
        base + scales->offset + (uint64_t)output * (in_dim / 32u);
    float result = 0.0f;
    for (uint32_t k = 0; k < in_dim; k++) {
        const uint8_t byte = weight[k >> 1];
        const uint8_t nibble = (k & 1u) ? byte >> 4 : byte & 15u;
        const uint32_t group = k / 32u;
        result += e2m1_to_f32(nibble) *
                  e8m0_to_f32(weight_scales[group]) *
                  e4m3_to_f32(x_values[k]) *
                  e8m0_to_f32(x_scales[group]);
    }
    return result;
}

static int close_enough(float actual, float expected) {
    const float tolerance = 2e-4f * fmaxf(1.0f, fabsf(expected));
    return isfinite(actual) && fabsf(actual - expected) <= tolerance;
}

static int pread_tensor_into(
        int fd,
        const k3_st_tensor *source,
        ds4_gpu_tensor *destination,
        uint64_t destination_offset) {
    if (fd < 0 || !source || !destination ||
        destination_offset > ds4_gpu_tensor_bytes(destination) ||
        source->size >
            ds4_gpu_tensor_bytes(destination) - destination_offset) {
        return 0;
    }
    uint8_t *dst = ds4_gpu_tensor_contents(destination);
    if (!dst || source->offset > INT64_MAX ||
        source->size > (uint64_t)INT64_MAX - source->offset) {
        return 0;
    }
    uint64_t done = 0;
    while (done < source->size) {
        const size_t want = source->size - done > (uint64_t)SSIZE_MAX
            ? (size_t)SSIZE_MAX
            : (size_t)(source->size - done);
        ssize_t n;
        do {
            n = pread(fd,
                      dst + destination_offset + done,
                      want,
                      (off_t)(source->offset + done));
        } while (n < 0 && errno == EINTR);
        if (n <= 0) return 0;
        done += (uint64_t)n;
    }
    return 1;
}

static int validate_downloaded_layers(
        const k3_model *model,
        const char *directory,
        char *error,
        size_t error_cap) {
    uint32_t validated = 0;
    for (uint32_t index = 0; index < K3_N_LAYER; index++) {
        char path[4096];
        const int n = snprintf(
            path, sizeof(path),
            "%s/model-%05u-of-000096.safetensors",
            directory, index + 1u);
        if (n <= 0 || (size_t)n >= sizeof(path)) {
            fprintf(stderr, "K3 checkpoint path is too long\n");
            return 0;
        }
        if (access(path, R_OK) != 0) continue;
        k3_layer *layer = NULL;
        if (!k3_model_open_layer(
                model, index, &layer, error, error_cap)) {
            fprintf(stderr, "%s\n", error);
            return 0;
        }
        k3_layer_close(layer);
        validated++;
    }
    if (validated == 0) {
        fprintf(stderr, "no downloaded K3 text layers were found\n");
        return 0;
    }
    printf("test_k3_checkpoint: ok (%u downloaded layer descriptors)\n",
           validated);
    return 1;
}

int main(int argc, char **argv) {
    const char *directory = argc > 1 ? argv[1] : "models/Kimi-K3";
    char error[512] = {0};
    k3_model *model = NULL;
    k3_layer *layer = NULL;
    k3_layer *moe_layer = NULL;
    k3_global_weights globals = {0};
    ds4_gpu_tensor *x = NULL;
    ds4_gpu_tensor *normalized = NULL;
    ds4_gpu_tensor *q = NULL;
    ds4_gpu_tensor *expert_x = NULL;
    ds4_gpu_tensor *expert_x_values = NULL;
    ds4_gpu_tensor *expert_x_scales = NULL;
    ds4_gpu_tensor *expert_gate = NULL;
    ds4_gpu_tensor *expert_up = NULL;
    ds4_gpu_tensor *expert_mid = NULL;
    ds4_gpu_tensor *expert_mid_values = NULL;
    ds4_gpu_tensor *expert_mid_scales = NULL;
    ds4_gpu_tensor *expert_out = NULL;
    ds4_gpu_tensor *expert_cached_weights = NULL;
    ds4_gpu_tensor *global_token = NULL;
    ds4_gpu_tensor *global_embedding = NULL;
    ds4_gpu_tensor *global_hidden = NULL;
    ds4_gpu_tensor *global_norm = NULL;
    ds4_gpu_tensor *global_logits = NULL;
    float *input = NULL;
    float *output = NULL;
    float *expert_input = NULL;
    float *gate_output = NULL;
    float *up_output = NULL;
    float *mid_output = NULL;
    float *expert_output = NULL;
    uint8_t *expert_x_q = NULL;
    uint8_t *expert_x_s = NULL;
    uint8_t *expert_mid_q = NULL;
    uint8_t *expert_mid_s = NULL;
    float *global_input = NULL;
    float *global_normalized = NULL;
    int ok = 0;

    if (!k3_model_open(&model, directory, error, sizeof(error)) ||
        !validate_downloaded_layers(
            model, directory, error, sizeof(error)) ||
        !k3_model_open_layer(model, 0, &layer, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        goto done;
    }
    ds4_gpu_set_ssd_streaming(true);
    if (!ds4_gpu_init() ||
        !ds4_gpu_set_model_map_range(
            k3_st_map_base(layer->shard),
            k3_st_file_size(layer->shard),
            layer->static_offset,
            layer->static_size,
            512ull * 1024ull * 1024ull)) {
        fprintf(stderr, "could not map real K3 layer 0\n");
        goto done;
    }
    x = ds4_gpu_tensor_alloc((uint64_t)K3_N_EMBD * sizeof(float));
    normalized =
        ds4_gpu_tensor_alloc((uint64_t)K3_N_EMBD * sizeof(float));
    q = ds4_gpu_tensor_alloc((uint64_t)K3_KDA_PROJ * sizeof(float));
    input = malloc((size_t)K3_N_EMBD * sizeof(float));
    output = malloc((size_t)K3_KDA_PROJ * sizeof(float));
    if (!x || !normalized || !q || !input || !output) {
        fprintf(stderr, "checkpoint smoke allocation failed\n");
        goto done;
    }
    for (uint32_t i = 0; i < K3_N_EMBD; i++) {
        input[i] = ((int)(i % 31u) - 15) * 0.001f;
    }
    if (!ds4_gpu_tensor_write(
            x, 0, input, (uint64_t)K3_N_EMBD * sizeof(float)) ||
        !k3_gpu_rms_norm_bf16(
            normalized, x,
            k3_st_map_base(layer->shard),
            k3_st_file_size(layer->shard),
            layer->input_norm->offset,
            K3_N_EMBD, 1, 1e-5f) ||
        !k3_gpu_matmul_bf16(
            q,
            k3_st_map_base(layer->shard),
            k3_st_file_size(layer->shard),
            layer->q_proj->offset,
            K3_N_EMBD, K3_KDA_PROJ,
            normalized, 1) ||
        !ds4_gpu_tensor_read(
            q, 0, output,
            (uint64_t)K3_KDA_PROJ * sizeof(float))) {
        fprintf(stderr, "real K3 layer 0 projection failed\n");
        goto done;
    }
    double sumsq = 0.0;
    for (uint32_t i = 0; i < K3_KDA_PROJ; i++) {
        if (!isfinite(output[i])) {
            fprintf(stderr, "real K3 projection produced a non-finite value\n");
            goto done;
        }
        sumsq += (double)output[i] * output[i];
    }
    if (!(sumsq > 0.0)) {
        fprintf(stderr, "real K3 projection is unexpectedly zero\n");
        goto done;
    }
    printf("test_k3_checkpoint: ok (layer 0 q RMS %.7g)\n",
           sqrt(sumsq / K3_KDA_PROJ));

    if (!k3_model_open_layer(
            model, 1, &moe_layer, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        goto done;
    }
    const k3_expert_weights *expert = &moe_layer->experts[0];
    const k3_st_tensor *expert_tensors[] = {
        expert->w1_packed,
        expert->w1_scale,
        expert->w2_packed,
        expert->w2_scale,
        expert->w3_packed,
        expert->w3_scale,
    };
    uint64_t moe_offsets[7] = {moe_layer->static_offset};
    uint64_t moe_sizes[7] = {moe_layer->static_size};
    uint64_t max_moe_tensor = moe_layer->static_size;
    for (uint32_t i = 0; i < 6; i++) {
        moe_offsets[i + 1u] = expert_tensors[i]->offset;
        moe_sizes[i + 1u] = expert_tensors[i]->size;
        if (expert_tensors[i]->size > max_moe_tensor) {
            max_moe_tensor = expert_tensors[i]->size;
        }
    }
    if (!ds4_gpu_set_model_fd_for_map(
            k3_st_fd(moe_layer->shard),
            k3_st_map_base(moe_layer->shard)) ||
        !ds4_gpu_set_model_map_spans(
            k3_st_map_base(moe_layer->shard),
            k3_st_file_size(moe_layer->shard),
            moe_offsets,
            moe_sizes,
            7,
            max_moe_tensor)) {
        fprintf(stderr, "could not map real K3 streamed layer 1\n");
        goto done;
    }

    expert_x = ds4_gpu_tensor_alloc(
        (uint64_t)K3_ROUTED_HIDDEN * sizeof(float));
    expert_x_values = ds4_gpu_tensor_alloc(K3_ROUTED_HIDDEN);
    expert_x_scales = ds4_gpu_tensor_alloc(K3_ROUTED_HIDDEN / 32u);
    expert_gate = ds4_gpu_tensor_alloc(
        (uint64_t)K3_EXPERT_FF * sizeof(float));
    expert_up = ds4_gpu_tensor_alloc(
        (uint64_t)K3_EXPERT_FF * sizeof(float));
    expert_mid = ds4_gpu_tensor_alloc(
        (uint64_t)K3_EXPERT_FF * sizeof(float));
    expert_mid_values = ds4_gpu_tensor_alloc(K3_EXPERT_FF);
    expert_mid_scales = ds4_gpu_tensor_alloc(K3_EXPERT_FF / 32u);
    expert_out = ds4_gpu_tensor_alloc(
        (uint64_t)K3_ROUTED_HIDDEN * sizeof(float));
    expert_input = malloc((size_t)K3_ROUTED_HIDDEN * sizeof(float));
    gate_output = malloc((size_t)K3_EXPERT_FF * sizeof(float));
    up_output = malloc((size_t)K3_EXPERT_FF * sizeof(float));
    mid_output = malloc((size_t)K3_EXPERT_FF * sizeof(float));
    expert_output = malloc((size_t)K3_ROUTED_HIDDEN * sizeof(float));
    expert_x_q = malloc(K3_ROUTED_HIDDEN);
    expert_x_s = malloc(K3_ROUTED_HIDDEN / 32u);
    expert_mid_q = malloc(K3_EXPERT_FF);
    expert_mid_s = malloc(K3_EXPERT_FF / 32u);
    if (!expert_x || !expert_x_values || !expert_x_scales ||
        !expert_gate || !expert_up || !expert_mid ||
        !expert_mid_values || !expert_mid_scales || !expert_out ||
        !expert_input || !gate_output || !up_output || !mid_output ||
        !expert_output || !expert_x_q || !expert_x_s ||
        !expert_mid_q || !expert_mid_s) {
        fprintf(stderr, "checkpoint expert smoke allocation failed\n");
        goto done;
    }
    if (!k3_gpu_matmul_bf16(
            expert_x,
            k3_st_map_base(moe_layer->shard),
            k3_st_file_size(moe_layer->shard),
            moe_layer->routed_down->offset,
            K3_N_EMBD, K3_ROUTED_HIDDEN,
            x, 1) ||
        !ds4_gpu_tensor_read(
            expert_x, 0, expert_input,
            (uint64_t)K3_ROUTED_HIDDEN * sizeof(float))) {
        fprintf(stderr, "real K3 routed latent projection failed\n");
        goto done;
    }
    double routed_sumsq = 0.0;
    for (uint32_t i = 0; i < K3_ROUTED_HIDDEN; i++) {
        if (!isfinite(expert_input[i])) {
            fprintf(stderr, "real K3 routed latent is non-finite\n");
            goto done;
        }
        routed_sumsq += (double)expert_input[i] * expert_input[i];
    }
    if (!(routed_sumsq > 0.0)) {
        fprintf(stderr, "real K3 routed latent is unexpectedly zero\n");
        goto done;
    }
    for (uint32_t i = 0; i < K3_ROUTED_HIDDEN; i++) {
        expert_input[i] = ((int)(i % 67u) - 33) * 0.0025f;
    }
    if (!ds4_gpu_tensor_write(
            expert_x, 0, expert_input,
            (uint64_t)K3_ROUTED_HIDDEN * sizeof(float)) ||
        !k3_gpu_mxfp8_quantize(
            expert_x_values, expert_x_scales, expert_x,
            K3_ROUTED_HIDDEN, 1) ||
        !k3_gpu_mxfp4_pair_situ(
            expert_gate, expert_up, expert_mid,
            k3_st_map_base(moe_layer->shard),
            k3_st_file_size(moe_layer->shard),
            expert->w1_packed->offset, expert->w1_scale->offset,
            expert->w3_packed->offset, expert->w3_scale->offset,
            expert_x_values, expert_x_scales,
            K3_ROUTED_HIDDEN, K3_EXPERT_FF, 1, 4.0f, 25.0f) ||
        !k3_gpu_mxfp8_quantize(
            expert_mid_values, expert_mid_scales, expert_mid,
            K3_EXPERT_FF, 1) ||
        !k3_gpu_mxfp4_down(
            expert_out,
            k3_st_map_base(moe_layer->shard),
            k3_st_file_size(moe_layer->shard),
            expert->w2_packed->offset, expert->w2_scale->offset,
            expert_mid_values, expert_mid_scales,
            K3_EXPERT_FF, K3_ROUTED_HIDDEN, 1, 1.0f, 0) ||
        !ds4_gpu_tensor_read(
            expert_x_values, 0, expert_x_q, K3_ROUTED_HIDDEN) ||
        !ds4_gpu_tensor_read(
            expert_x_scales, 0, expert_x_s, K3_ROUTED_HIDDEN / 32u) ||
        !ds4_gpu_tensor_read(
            expert_gate, 0, gate_output,
            (uint64_t)K3_EXPERT_FF * sizeof(float)) ||
        !ds4_gpu_tensor_read(
            expert_up, 0, up_output,
            (uint64_t)K3_EXPERT_FF * sizeof(float)) ||
        !ds4_gpu_tensor_read(
            expert_mid, 0, mid_output,
            (uint64_t)K3_EXPERT_FF * sizeof(float)) ||
        !ds4_gpu_tensor_read(
            expert_mid_values, 0, expert_mid_q, K3_EXPERT_FF) ||
        !ds4_gpu_tensor_read(
            expert_mid_scales, 0, expert_mid_s, K3_EXPERT_FF / 32u) ||
        !ds4_gpu_tensor_read(
            expert_out, 0, expert_output,
            (uint64_t)K3_ROUTED_HIDDEN * sizeof(float))) {
        fprintf(stderr, "real K3 layer 1 expert execution failed\n");
        goto done;
    }
    static const uint32_t gate_checks[] = {0, 17, 1024, 3071};
    const uint8_t *base = k3_st_map_base(moe_layer->shard);
    for (uint32_t i = 0;
         i < sizeof(gate_checks) / sizeof(gate_checks[0]); i++) {
        const uint32_t row = gate_checks[i];
        const float expected_gate = checkpoint_mxfp4_dot(
            base, expert->w1_packed, expert->w1_scale, row,
            K3_ROUTED_HIDDEN, expert_x_q, expert_x_s);
        const float expected_up = checkpoint_mxfp4_dot(
            base, expert->w3_packed, expert->w3_scale, row,
            K3_ROUTED_HIDDEN, expert_x_q, expert_x_s);
        const float expected_mid =
            4.0f * tanhf(expected_gate / 4.0f) /
            (1.0f + expf(-expected_gate)) *
            25.0f * tanhf(expected_up / 25.0f);
        if (!close_enough(gate_output[row], expected_gate) ||
            !close_enough(up_output[row], expected_up) ||
            !close_enough(mid_output[row], expected_mid)) {
            fprintf(stderr,
                    "real K3 expert gate/up mismatch at output %u\n", row);
            goto done;
        }
    }
    static const uint32_t down_checks[] = {0, 31, 2048, 3583};
    for (uint32_t i = 0;
         i < sizeof(down_checks) / sizeof(down_checks[0]); i++) {
        const uint32_t row = down_checks[i];
        const float expected = checkpoint_mxfp4_dot(
            base, expert->w2_packed, expert->w2_scale, row,
            K3_EXPERT_FF, expert_mid_q, expert_mid_s);
        if (!close_enough(expert_output[row], expected)) {
            fprintf(stderr,
                    "real K3 expert down mismatch at output %u: "
                    "actual %.9g expected %.9g\n",
                    row, expert_output[row], expected);
            goto done;
        }
    }

    uint64_t cached_offsets[6];
    uint64_t cached_bytes = 0;
    for (uint32_t i = 0; i < 6; i++) {
        cached_offsets[i] = cached_bytes;
        if (expert_tensors[i]->size > UINT64_MAX - cached_bytes) {
            fprintf(stderr, "real K3 compact expert size overflow\n");
            goto done;
        }
        cached_bytes += expert_tensors[i]->size;
    }
    expert_cached_weights = ds4_gpu_tensor_alloc(cached_bytes);
    if (!expert_cached_weights) {
        fprintf(stderr, "real K3 compact expert allocation failed\n");
        goto done;
    }
    for (uint32_t i = 0; i < 6; i++) {
        if (!pread_tensor_into(
                k3_st_fd(moe_layer->shard),
                expert_tensors[i],
                expert_cached_weights,
                cached_offsets[i])) {
            fprintf(stderr,
                    "real K3 compact expert tensor %u read failed\n", i);
            goto done;
        }
    }
    if (!k3_gpu_mxfp4_pair_situ_tensor(
            expert_gate, expert_up, expert_mid,
            expert_cached_weights,
            cached_offsets[0], cached_offsets[1],
            cached_offsets[4], cached_offsets[5],
            expert_x_values, expert_x_scales,
            K3_ROUTED_HIDDEN, K3_EXPERT_FF, 1, 4.0f, 25.0f) ||
        !k3_gpu_mxfp8_quantize(
            expert_mid_values, expert_mid_scales, expert_mid,
            K3_EXPERT_FF, 1) ||
        !k3_gpu_mxfp4_down_tensor(
            expert_out,
            expert_cached_weights,
            cached_offsets[2], cached_offsets[3],
            expert_mid_values, expert_mid_scales,
            K3_EXPERT_FF, K3_ROUTED_HIDDEN, 1, 1.0f, 0) ||
        !ds4_gpu_tensor_read(
            expert_gate, 0, gate_output,
            (uint64_t)K3_EXPERT_FF * sizeof(float)) ||
        !ds4_gpu_tensor_read(
            expert_up, 0, up_output,
            (uint64_t)K3_EXPERT_FF * sizeof(float)) ||
        !ds4_gpu_tensor_read(
            expert_mid, 0, mid_output,
            (uint64_t)K3_EXPERT_FF * sizeof(float)) ||
        !ds4_gpu_tensor_read(
            expert_mid_values, 0, expert_mid_q, K3_EXPERT_FF) ||
        !ds4_gpu_tensor_read(
            expert_mid_scales, 0, expert_mid_s, K3_EXPERT_FF / 32u) ||
        !ds4_gpu_tensor_read(
            expert_out, 0, expert_output,
            (uint64_t)K3_ROUTED_HIDDEN * sizeof(float))) {
        fprintf(stderr, "real K3 compact expert execution failed\n");
        goto done;
    }
    for (uint32_t i = 0;
         i < sizeof(gate_checks) / sizeof(gate_checks[0]); i++) {
        const uint32_t row = gate_checks[i];
        const float expected_gate = checkpoint_mxfp4_dot(
            base, expert->w1_packed, expert->w1_scale, row,
            K3_ROUTED_HIDDEN, expert_x_q, expert_x_s);
        const float expected_up = checkpoint_mxfp4_dot(
            base, expert->w3_packed, expert->w3_scale, row,
            K3_ROUTED_HIDDEN, expert_x_q, expert_x_s);
        const float expected_mid =
            4.0f * tanhf(expected_gate / 4.0f) /
            (1.0f + expf(-expected_gate)) *
            25.0f * tanhf(expected_up / 25.0f);
        if (!close_enough(gate_output[row], expected_gate) ||
            !close_enough(up_output[row], expected_up) ||
            !close_enough(mid_output[row], expected_mid)) {
            fprintf(stderr,
                    "real K3 compact expert gate/up mismatch at output %u\n",
                    row);
            goto done;
        }
    }
    for (uint32_t i = 0;
         i < sizeof(down_checks) / sizeof(down_checks[0]); i++) {
        const uint32_t row = down_checks[i];
        const float expected = checkpoint_mxfp4_dot(
            base, expert->w2_packed, expert->w2_scale, row,
            K3_EXPERT_FF, expert_mid_q, expert_mid_s);
        if (!close_enough(expert_output[row], expected)) {
            fprintf(stderr,
                    "real K3 compact expert down mismatch at output %u: "
                    "actual %.9g expected %.9g\n",
                    row, expert_output[row], expected);
            goto done;
        }
    }
    printf("test_k3_checkpoint: ok "
           "(layer 1 expert 0 MXFP4/A8 mapped and compact)\n");

    if (!k3_model_open_globals(
            model, &globals, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        goto done;
    }
    global_token = ds4_gpu_tensor_alloc(sizeof(int32_t));
    global_embedding = ds4_gpu_tensor_alloc(
        (uint64_t)K3_N_EMBD * sizeof(float));
    global_hidden = ds4_gpu_tensor_alloc(
        (uint64_t)K3_N_EMBD * sizeof(float));
    global_norm = ds4_gpu_tensor_alloc(
        (uint64_t)K3_N_EMBD * sizeof(float));
    global_logits = ds4_gpu_tensor_alloc(4u * sizeof(float));
    global_input = malloc((size_t)K3_N_EMBD * sizeof(float));
    global_normalized = malloc((size_t)K3_N_EMBD * sizeof(float));
    if (!global_token || !global_embedding || !global_hidden ||
        !global_norm || !global_logits || !global_input ||
        !global_normalized) {
        fprintf(stderr, "checkpoint global smoke allocation failed\n");
        goto done;
    }
    const int32_t token_zero = 0;
    if (!ds4_gpu_set_model_fd_for_map(
            k3_st_fd(globals.shard), k3_st_map_base(globals.shard)) ||
        !ds4_gpu_set_model_map_range(
            k3_st_map_base(globals.shard),
            k3_st_file_size(globals.shard),
            globals.token_embedding->offset,
            globals.token_embedding->size,
            globals.token_embedding->size) ||
        !ds4_gpu_tensor_write(
            global_token, 0, &token_zero, sizeof(token_zero)) ||
        !k3_gpu_embedding_bf16(
            global_embedding,
            k3_st_map_base(globals.shard),
            k3_st_file_size(globals.shard),
            globals.token_embedding->offset,
            global_token, 1, K3_N_EMBD, 1) ||
        !ds4_gpu_tensor_read(
            global_embedding, 0, global_input,
            (uint64_t)K3_N_EMBD * sizeof(float))) {
        fprintf(stderr, "real K3 token embedding failed\n");
        goto done;
    }
    const uint16_t *embedding =
        (const uint16_t *)((const uint8_t *)k3_st_map_base(globals.shard) +
                           globals.token_embedding->offset);
    static const uint32_t embedding_checks[] = {0, 37, 4096, 7167};
    for (uint32_t i = 0;
         i < sizeof(embedding_checks) / sizeof(embedding_checks[0]); i++) {
        const uint32_t d = embedding_checks[i];
        if (global_input[d] != bf16_to_f32(embedding[d])) {
            fprintf(stderr, "real K3 embedding mismatch at dimension %u\n", d);
            goto done;
        }
    }

    const uint64_t global_offsets[2] = {
        globals.norm->offset,
        globals.lm_head->offset,
    };
    const uint64_t global_sizes[2] = {
        globals.norm->size,
        globals.lm_head->size,
    };
    for (uint32_t i = 0; i < K3_N_EMBD; i++) {
        global_input[i] = ((int)(i % 71u) - 35) * 0.001f;
    }
    if (!ds4_gpu_set_model_map_spans(
            k3_st_map_base(globals.shard),
            k3_st_file_size(globals.shard),
            global_offsets, global_sizes, 2, global_sizes[1]) ||
        !ds4_gpu_tensor_write(
            global_hidden, 0, global_input,
            (uint64_t)K3_N_EMBD * sizeof(float)) ||
        !k3_gpu_rms_norm_bf16(
            global_norm, global_hidden,
            k3_st_map_base(globals.shard),
            k3_st_file_size(globals.shard),
            globals.norm->offset, K3_N_EMBD, 1, 1e-5f) ||
        !k3_gpu_matmul_bf16(
            global_logits,
            k3_st_map_base(globals.shard),
            k3_st_file_size(globals.shard),
            globals.lm_head->offset,
            K3_N_EMBD, 4, global_norm, 1) ||
        !ds4_gpu_tensor_read(
            global_norm, 0, global_normalized,
            (uint64_t)K3_N_EMBD * sizeof(float))) {
        fprintf(stderr, "real K3 final norm/head execution failed\n");
        goto done;
    }
    float actual_logits[4];
    if (!ds4_gpu_tensor_read(
            global_logits, 0, actual_logits, sizeof(actual_logits))) {
        fprintf(stderr, "real K3 logits read failed\n");
        goto done;
    }
    const uint16_t *lm_head =
        (const uint16_t *)((const uint8_t *)k3_st_map_base(globals.shard) +
                           globals.lm_head->offset);
    for (uint32_t row = 0; row < 4; row++) {
        float expected = 0.0f;
        for (uint32_t d = 0; d < K3_N_EMBD; d++) {
            expected += global_normalized[d] *
                        bf16_to_f32(lm_head[(uint64_t)row * K3_N_EMBD + d]);
        }
        if (!close_enough(actual_logits[row], expected)) {
            fprintf(stderr,
                    "real K3 LM head mismatch at row %u: "
                    "actual %.9g expected %.9g\n",
                    row, actual_logits[row], expected);
            goto done;
        }
    }
    printf("test_k3_checkpoint: ok (embedding/final norm/LM head)\n");
    ok = 1;

done:
    ds4_gpu_tensor_free(expert_cached_weights);
    ds4_gpu_tensor_free(global_logits);
    ds4_gpu_tensor_free(global_norm);
    ds4_gpu_tensor_free(global_hidden);
    ds4_gpu_tensor_free(global_embedding);
    ds4_gpu_tensor_free(global_token);
    ds4_gpu_tensor_free(expert_out);
    ds4_gpu_tensor_free(expert_mid_scales);
    ds4_gpu_tensor_free(expert_mid_values);
    ds4_gpu_tensor_free(expert_mid);
    ds4_gpu_tensor_free(expert_up);
    ds4_gpu_tensor_free(expert_gate);
    ds4_gpu_tensor_free(expert_x_scales);
    ds4_gpu_tensor_free(expert_x_values);
    ds4_gpu_tensor_free(expert_x);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(normalized);
    ds4_gpu_tensor_free(x);
    free(global_normalized);
    free(global_input);
    free(expert_mid_s);
    free(expert_mid_q);
    free(expert_x_s);
    free(expert_x_q);
    free(expert_output);
    free(mid_output);
    free(up_output);
    free(gate_output);
    free(expert_input);
    free(output);
    free(input);
    ds4_gpu_cleanup();
    k3_global_weights_close(&globals);
    k3_layer_close(moe_layer);
    k3_layer_close(layer);
    k3_model_close(model);
    return ok ? 0 : 1;
}
