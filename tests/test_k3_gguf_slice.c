#define _POSIX_C_SOURCE 200809L

#include "k3_infer.h"

#include <math.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static void fail(const char *message, const char *detail) {
    fprintf(stderr, "test_k3_gguf_slice: %s%s%s\n",
            message, detail && detail[0] ? ": " : "",
            detail && detail[0] ? detail : "");
    exit(1);
}

typedef struct {
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t rows;
    uint32_t calls[4];
    int invalid;
} hidden_hook_state;

static void count_hidden_hook(void *ud,
                              const char *stream,
                              uint32_t layer,
                              uint32_t pos0,
                              ds4_gpu_tensor *tensor,
                              uint64_t n_f32) {
    hidden_hook_state *state = ud;
    static const char *names[] = {
        "hc_attn_pre", "attn_out", "hc_ffn_pre", "ffn_out",
    };
    const uint32_t expected_rows = pos0 == 0u ? state->rows : 1u;
    if (!tensor ||
        layer < state->layer_start ||
        layer > state->layer_end ||
        n_f32 != (uint64_t)expected_rows * K3_N_EMBD) {
        state->invalid = 1;
        return;
    }
    for (uint32_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        if (!strcmp(stream, names[i])) {
            state->calls[i]++;
            return;
        }
    }
    state->invalid = 1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s K3.gguf\n", argv[0]);
        return 2;
    }

    const int pipeline_test = getenv("K3_TEST_PIPELINE") != NULL;
    const int static_test = getenv("K3_TEST_STATIC") != NULL;
    const int resident_test =
        pipeline_test || getenv("K3_TEST_RESIDENT") != NULL;
    uint32_t stream_layers = 1u;
    const char *stream_layers_env = getenv("K3_TEST_STREAM_LAYERS");
    if (stream_layers_env && stream_layers_env[0]) {
        char *end = NULL;
        errno = 0;
        const unsigned long value =
            strtoul(stream_layers_env, &end, 10);
        if (errno || end == stream_layers_env || *end != '\0' ||
            value < 2u || value > 4u || resident_test) {
            fail("invalid K3_TEST_STREAM_LAYERS", stream_layers_env);
        }
        stream_layers = (uint32_t)value;
    }
    uint32_t prefill_rows = 1u;
    const char *prefill_env = getenv("K3_TEST_PREFILL_ROWS");
    if (prefill_env && prefill_env[0]) {
        char *end = NULL;
        errno = 0;
        const unsigned long value = strtoul(prefill_env, &end, 10);
        if (errno || end == prefill_env || *end != '\0' ||
            value < 2u || value > 128u) {
            fail("invalid K3_TEST_PREFILL_ROWS", prefill_env);
        }
        prefill_rows = (uint32_t)value;
    }
    uint32_t layer_start = 1u;
    const char *layer_env = getenv("K3_TEST_LAYER");
    if (layer_env && layer_env[0]) {
        char *end = NULL;
        errno = 0;
        const unsigned long value = strtoul(layer_env, &end, 10);
        if (errno || end == layer_env || *end != '\0' ||
            value >= K3_N_LAYER || pipeline_test) {
            fail("invalid K3_TEST_LAYER", layer_env);
        }
        layer_start = (uint32_t)value;
    }
    const uint32_t local_layers =
        pipeline_test ? 2u : stream_layers;
    if (local_layers > K3_N_LAYER - layer_start) {
        fail("K3 test layer range exceeds the model", NULL);
    }
    const uint32_t layer_end =
        layer_start + local_layers - 1u;
    uint32_t steps = resident_test ? 8u : 2u;
    const char *steps_env = getenv("K3_TEST_STEPS");
    if (steps_env && steps_env[0]) {
        char *end = NULL;
        errno = 0;
        const unsigned long value = strtoul(steps_env, &end, 10);
        if (errno || end == steps_env || *end != '\0' ||
            value < 1u || value > 4096u) {
            fail("invalid K3_TEST_STEPS", steps_env);
        }
        steps = (uint32_t)value;
    }
    if (setenv("K3_GGUF_RESIDENT", resident_test ? "1" : "0", 1) != 0 ||
        (!resident_test && !static_test &&
         setenv("K3_RESIDENT_STATIC", "0", 1) != 0) ||
        (static_test &&
         (setenv("K3_RESIDENT_STATIC", "1", 1) != 0 ||
          setenv("K3_RESIDENT_STATIC_GIB", "16", 1) != 0)) ||
        setenv("K3_STATIC_STAGING", "0", 1) != 0) {
        fail("could not configure slice test", NULL);
    }
    uint32_t context_capacity = steps;
    if (context_capacity < prefill_rows + 1u) {
        context_capacity = prefill_rows + 1u;
    }
    const k3_context_config config = {
        .model_directory = argv[1],
        .context_capacity = context_capacity,
        .max_prefill_tokens = prefill_rows,
        .layer_slice = true,
        .layer_start = layer_start,
        .layer_end = layer_end,
        .has_output = false,
    };
    char err[512] = {0};
    k3_context *ctx = NULL;
    if (!k3_context_create(&ctx, &config, err, sizeof(err))) {
        fail("context creation failed", err);
    }
    hidden_hook_state hook = {
        .layer_start = layer_start,
        .layer_end = layer_end,
        .rows = prefill_rows,
    };
    k3_context_set_hidden_hook(ctx, count_hidden_hook, &hook);

    const size_t boundary_values =
        (size_t)prefill_rows * K3_BOUNDARY_F32_VALUES;
    float *input = malloc(boundary_values * sizeof(float));
    float *output = malloc(boundary_values * sizeof(float));
    if (!input || !output) fail("boundary allocation failed", NULL);
    for (size_t i = 0; i < boundary_values; i++) {
        input[i] = 0.02f * sinf((float)i * 0.013f) +
                   0.01f * cosf((float)i * 0.007f);
    }

    int32_t tokens[128];
    for (uint32_t i = 0; i < prefill_rows; i++) {
        tokens[i] = 1000 + (int32_t)i;
    }
    float *boundary_in = layer_start == 0u ? NULL : input;
    float *boundary_out = output;
    float *result = NULL;
    if (prefill_rows > 1u) {
        if (!k3_context_eval_slice(
                ctx, tokens, prefill_rows, 0u, layer_start, layer_end,
                boundary_in, boundary_out, false, NULL,
                err, sizeof(err))) {
            fail("layer batch prefill evaluation failed", err);
        }
        result = output +
            (size_t)(prefill_rows - 1u) * K3_BOUNDARY_F32_VALUES;
        if (!k3_context_eval_slice(
                ctx, tokens, 1u, prefill_rows, layer_start, layer_end,
                layer_start == 0u ? NULL : result, input, false, NULL,
                err, sizeof(err))) {
            fail("layer post-prefill generation evaluation failed", err);
        }
        result = input;
    } else {
        for (uint32_t position = 0; position < steps; position++) {
            if (!k3_context_eval_slice(
                    ctx, tokens, 1, position, layer_start, layer_end,
                    boundary_in, boundary_out, false, NULL,
                    err, sizeof(err))) {
                fail(position == 0u
                         ? "layer prefill evaluation failed"
                         : "layer generation evaluation failed",
                     err);
            }
            result = boundary_out;
            if (layer_start != 0u) {
                float *next_out = boundary_in;
                boundary_in = boundary_out;
                boundary_out = next_out;
            }
        }
    }
    double checksum = 0.0;
    uint64_t hash = 1469598103934665603ull;
    for (uint32_t i = 0; i < K3_BOUNDARY_F32_VALUES; i++) {
        if (!isfinite(result[i])) {
            fail("non-finite boundary output", NULL);
        }
        checksum += fabs((double)result[i]);
        uint32_t bits = 0;
        memcpy(&bits, &result[i], sizeof(bits));
        hash ^= bits;
        hash *= 1099511628211ull;
    }
    if (!(checksum > 1.0)) fail("boundary output is unexpectedly zero", NULL);
    const uint32_t evaluations = prefill_rows > 1u ? 2u : steps;
    const uint32_t expected_hook_calls = local_layers * evaluations;
    for (uint32_t i = 0; i < sizeof(hook.calls) / sizeof(hook.calls[0]); i++) {
        if (hook.calls[i] != expected_hook_calls) {
            fail("hidden hook call count mismatch", NULL);
        }
    }
    if (hook.invalid) fail("hidden hook received an invalid stage", NULL);
    k3_context_set_hidden_hook(ctx, NULL, NULL);

    const char *dump_path = getenv("K3_TEST_DUMP");
    if (dump_path && dump_path[0]) {
        FILE *fp = fopen(dump_path, "wb");
        int ok = fp != NULL;
        if (ok) {
            ok =
                fwrite(result, sizeof(*result),
                       K3_BOUNDARY_F32_VALUES, fp) ==
                    K3_BOUNDARY_F32_VALUES &&
                fflush(fp) == 0;
            if (fclose(fp) != 0) ok = 0;
        }
        if (!ok) {
            fail("could not write K3_TEST_DUMP", dump_path);
        }
    }

    const char *compare_path = getenv("K3_TEST_COMPARE");
    if (compare_path && compare_path[0]) {
        float *reference =
            malloc((size_t)K3_BOUNDARY_F32_VALUES * sizeof(*reference));
        FILE *fp = fopen(compare_path, "rb");
        int ok = reference != NULL && fp != NULL;
        if (ok) {
            ok =
                fread(reference, sizeof(*reference),
                      K3_BOUNDARY_F32_VALUES, fp) ==
                    K3_BOUNDARY_F32_VALUES &&
                fgetc(fp) == EOF;
            if (fclose(fp) != 0) ok = 0;
        } else if (fp) {
            fclose(fp);
        }
        if (!ok) {
            free(reference);
            fail("could not read K3_TEST_COMPARE", compare_path);
        }
        double squared_error = 0.0;
        double squared_reference = 0.0;
        double squared_result = 0.0;
        double dot = 0.0;
        double max_error = 0.0;
        for (uint32_t i = 0; i < K3_BOUNDARY_F32_VALUES; i++) {
            const double actual = result[i];
            const double expected = reference[i];
            const double error = actual - expected;
            squared_error += error * error;
            squared_reference += expected * expected;
            squared_result += actual * actual;
            dot += actual * expected;
            if (fabs(error) > max_error) max_error = fabs(error);
        }
        const double rel_rmse =
            sqrt(squared_error / squared_reference);
        const double cosine =
            dot / sqrt(squared_result * squared_reference);
        printf("test_k3_gguf_slice: comparison rel_rmse=%.9f "
               "cosine=%.9f max_abs=%.9f\n",
               rel_rmse, cosine, max_error);
        free(reference);
    }

    free(output);
    free(input);
    k3_context_destroy(ctx);
    printf("test_k3_gguf_slice: ok (checksum %.12e, hash %016llx)\n",
           checksum, (unsigned long long)hash);
    return 0;
}
