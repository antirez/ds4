#define _DARWIN_C_SOURCE

/* Runtime oracle for the Metal Q4_K q_a/KV prefill pair.
 *
 * The model is a page-aligned synthetic mapping containing only the two
 * production projection matrices.  It deliberately installs two disjoint
 * model spans while SSD mode is enabled, so the test exercises the same model
 * view contract as streaming inference without reading a GGUF or SSD.
 */

#include "ds4_gpu.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __APPLE__

/* Standalone Metal tests link the backend without ds4.c. */
bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

enum {
    Q4_K_TYPE = 12u,
    QK_K = 256u,
    IN_DIM = 4096u,
    OUT0_DIM = 1024u,
    OUT1_DIM = 512u,
    MAX_TOKENS = 128u,
    TEST_STREAMS = 2u,
};

static const uint32_t k_tokens[] = {32u, 64u, 96u, 128u};
static const float k_poison = -12345.25f;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2u];
} block_q4_K;

_Static_assert(sizeof(block_q4_K) == 144u, "Q4_K ABI");

typedef struct {
    void *model;
    uint64_t model_size;
    uint64_t row_bytes;
    uint64_t weight_offset[2];
    uint64_t weight_bytes[2];
    ds4_gpu_tensor *x[TEST_STREAMS];
    ds4_gpu_tensor *baseline[TEST_STREAMS][2];
    ds4_gpu_tensor *candidate[TEST_STREAMS][2];
    float *x_host[TEST_STREAMS];
} fixture;

static void fail(const char *what) {
    fprintf(stderr, "Metal Q4 prefill pair runtime FAIL: %s\n", what);
    exit(1);
}

static void set_flag(const char *name, bool enabled) {
    const int rc = enabled ? setenv(name, "1", 1) : unsetenv(name);
    if (rc != 0) fail("environment update");
}

static void set_pair_controls(bool enabled, bool disabled, bool required) {
    set_flag("DS4_METAL_ENABLE_Q4_PREFILL_PAIR_F16_RHS", enabled);
    set_flag("DS4_METAL_DISABLE_Q4_PREFILL_PAIR_F16_RHS", disabled);
    set_flag("DS4_METAL_REQUIRE_Q4_PREFILL_PAIR_F16_RHS", required);
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static uint32_t lcg(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static void fill_q4(block_q4_K *blocks, uint64_t count, uint32_t seed) {
    for (uint64_t i = 0; i < count; i++) {
        blocks[i].d = (uint16_t)(0x2400u | (lcg(&seed) & 0x03ffu));
        blocks[i].dmin =
            (uint16_t)(0x1c00u | (lcg(&seed) & 0x03ffu));
        for (uint32_t j = 0; j < sizeof(blocks[i].scales); j++) {
            blocks[i].scales[j] = (uint8_t)(lcg(&seed) >> 24u);
        }
        for (uint32_t j = 0; j < sizeof(blocks[i].qs); j++) {
            blocks[i].qs[j] = (uint8_t)(lcg(&seed) >> 24u);
        }
    }
}

static void fill_x(float *x, uint32_t stream) {
    for (uint64_t i = 0; i < (uint64_t)MAX_TOKENS * IN_DIM; i++) {
        const int32_t v = (int32_t)(
            (i * 29u + stream * 47u + ((uint64_t)stream ^ i) * 3u) %
            255u) - 127;
        x[i] = (float)v / 128.0f;
    }
}

static uint64_t checksum(const void *ptr, uint64_t bytes) {
    const uint8_t *p = ptr;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (uint64_t i = 0; i < bytes; i++) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t output_count(uint32_t output) {
    return (uint64_t)MAX_TOKENS *
           (output == 0u ? OUT0_DIM : OUT1_DIM);
}

static uint64_t active_output_count(uint32_t output, uint32_t n_tokens) {
    return (uint64_t)n_tokens *
           (output == 0u ? OUT0_DIM : OUT1_DIM);
}

static void fixture_init(fixture *f) {
    memset(f, 0, sizeof(*f));
    f->row_bytes = (IN_DIM / QK_K) * sizeof(block_q4_K);
    const uint64_t page = (uint64_t)getpagesize();
    f->weight_offset[0] = 0u;
    f->weight_bytes[0] = (uint64_t)OUT0_DIM * f->row_bytes;
    f->weight_bytes[1] = (uint64_t)OUT1_DIM * f->row_bytes;
    f->weight_offset[1] = align_up(f->weight_bytes[0] + page, page);

    f->model_size = align_up(f->weight_offset[1] + f->weight_bytes[1],
                             page);
    if (posix_memalign(&f->model, (size_t)page, (size_t)f->model_size) != 0) {
        fail("synthetic model allocation");
    }
    memset(f->model, 0, (size_t)f->model_size);
    fill_q4((block_q4_K *)((uint8_t *)f->model + f->weight_offset[0]),
            f->weight_bytes[0] / sizeof(block_q4_K), 0x41c64e6du);
    fill_q4((block_q4_K *)((uint8_t *)f->model + f->weight_offset[1]),
            f->weight_bytes[1] / sizeof(block_q4_K), 0x9e3779b9u);

    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(true);
    const uint64_t offsets[2] = {
        f->weight_offset[0], f->weight_offset[1],
    };
    const uint64_t sizes[2] = {
        f->weight_bytes[0], f->weight_bytes[1],
    };
    if (!ds4_gpu_set_model_map_spans(f->model, f->model_size,
                                     offsets, sizes, 2u,
                                     f->weight_bytes[0])) {
        fail("SSD-style model spans");
    }

    const uint64_t x_count = (uint64_t)MAX_TOKENS * IN_DIM;
    for (uint32_t stream = 0; stream < TEST_STREAMS; stream++) {
        f->x_host[stream] = malloc((size_t)x_count * sizeof(float));
        if (!f->x_host[stream]) fail("host activation allocation");
        fill_x(f->x_host[stream], stream);
        f->x[stream] = ds4_gpu_tensor_alloc(x_count * sizeof(float));
        if (!f->x[stream] ||
            !ds4_gpu_tensor_write(f->x[stream], 0u, f->x_host[stream],
                                  x_count * sizeof(float))) {
            fail("activation tensor");
        }
        for (uint32_t output = 0; output < 2u; output++) {
            const uint64_t bytes = output_count(output) * sizeof(float);
            f->baseline[stream][output] = ds4_gpu_tensor_alloc(bytes);
            f->candidate[stream][output] = ds4_gpu_tensor_alloc(bytes);
            if (!f->baseline[stream][output] ||
                !f->candidate[stream][output]) {
                fail("output tensor allocation");
            }
        }
    }
    fprintf(stderr,
            "Metal Q4 prefill pair runtime: synthetic=%.2f MiB spans=2 "
            "ssd=1 max_tokens=%u\n",
            (double)f->model_size / 1048576.0, MAX_TOKENS);
}

static void fixture_destroy(fixture *f) {
    ds4_gpu_set_stream(0);
    (void)ds4_gpu_synchronize();
    for (uint32_t stream = 0; stream < TEST_STREAMS; stream++) {
        for (uint32_t output = 0; output < 2u; output++) {
            ds4_gpu_tensor_free(f->candidate[stream][output]);
            ds4_gpu_tensor_free(f->baseline[stream][output]);
        }
        ds4_gpu_tensor_free(f->x[stream]);
        free(f->x_host[stream]);
    }
    ds4_gpu_cleanup();
    free(f->model);
    memset(f, 0, sizeof(*f));
}

static void poison_outputs(fixture *f, uint32_t stream) {
    for (uint32_t output = 0; output < 2u; output++) {
        const uint64_t count = output_count(output);
        if (!ds4_gpu_tensor_fill_f32(f->baseline[stream][output],
                                     k_poison, count) ||
            !ds4_gpu_tensor_fill_f32(f->candidate[stream][output],
                                     k_poison, count)) {
            fail("output poison");
        }
    }
}

static int run_baseline(fixture *f, uint32_t stream, uint32_t n_tokens) {
    ds4_gpu_set_stream((int)stream);
    if (!ds4_gpu_begin_commands()) return 0;
    int ok = ds4_gpu_matmul_quant_tensor(
        f->baseline[stream][0], f->model, f->model_size,
        f->weight_offset[0], Q4_K_TYPE, IN_DIM, OUT0_DIM,
        f->x[stream], n_tokens);
    if (ok) {
        ok = ds4_gpu_matmul_quant_tensor(
            f->baseline[stream][1], f->model, f->model_size,
            f->weight_offset[1], Q4_K_TYPE, IN_DIM, OUT1_DIM,
            f->x[stream], n_tokens);
    }
    const int ended = ds4_gpu_end_commands();
    return ok && ended;
}

static int run_candidate(fixture *f, uint32_t stream, uint32_t n_tokens) {
    ds4_gpu_set_stream((int)stream);
    if (!ds4_gpu_begin_commands()) return 0;
    const int rc = ds4_gpu_matmul_q4_K_pair_tensor(
        f->candidate[stream][0], f->candidate[stream][1],
        f->model, f->model_size,
        f->weight_offset[0], f->weight_offset[1],
        IN_DIM, OUT0_DIM, OUT1_DIM, f->x[stream], n_tokens);
    const int ended = ds4_gpu_end_commands();
    return rc == 1 && ended;
}

static void check_poison_suffix(const float *values, uint64_t begin,
                                uint64_t count, const char *label) {
    for (uint64_t i = begin; i < count; i++) {
        if (memcmp(&values[i], &k_poison, sizeof(k_poison)) != 0) {
            fprintf(stderr,
                    "Metal Q4 prefill pair suffix write label=%s index=%llu\n",
                    label, (unsigned long long)i);
            fail("output suffix canary");
        }
    }
}

static void check_tensor_all_poison(const ds4_gpu_tensor *tensor,
                                    uint64_t count, const char *label) {
    float *values = malloc((size_t)count * sizeof(float));
    if (!values ||
        !ds4_gpu_tensor_read(tensor, 0u, values, count * sizeof(float))) {
        fail("poison readback");
    }
    check_poison_suffix(values, 0u, count, label);
    free(values);
}

static void check_outputs(fixture *f, uint32_t stream, uint32_t n_tokens) {
    for (uint32_t output = 0; output < 2u; output++) {
        const uint64_t count = output_count(output);
        const uint64_t active = active_output_count(output, n_tokens);
        float *base = malloc((size_t)count * sizeof(float));
        float *candidate = malloc((size_t)count * sizeof(float));
        if (!base || !candidate) fail("host output allocation");
        if (!ds4_gpu_tensor_read(f->baseline[stream][output], 0u, base,
                                 count * sizeof(float)) ||
            !ds4_gpu_tensor_read(f->candidate[stream][output], 0u, candidate,
                                 count * sizeof(float))) {
            fail("output readback");
        }
        if (memcmp(base, candidate, active * sizeof(float)) != 0) {
            uint64_t first = 0u;
            while (first < active &&
                   memcmp(&base[first], &candidate[first],
                          sizeof(float)) == 0) {
                first++;
            }
            fprintf(stderr,
                    "Metal Q4 prefill pair mismatch N=%u stream=%u output=%u "
                    "first=%llu base=%g candidate=%g\n",
                    n_tokens, stream, output, (unsigned long long)first,
                    first < active ? base[first] : 0.0f,
                    first < active ? candidate[first] : 0.0f);
            fail("bitwise parity");
        }
        check_poison_suffix(base, active, count, "baseline");
        check_poison_suffix(candidate, active, count, "candidate");
        fprintf(stderr,
                "Metal Q4 prefill pair N=%u stream=%u output=%u "
                "checksum=%016llx bitwise=1 canary=1\n",
                n_tokens, stream, output,
                (unsigned long long)checksum(base, active * sizeof(float)));
        free(candidate);
        free(base);
    }
}

static void check_x_unchanged(fixture *f, uint32_t stream) {
    const uint64_t count = (uint64_t)MAX_TOKENS * IN_DIM;
    float *actual = malloc((size_t)count * sizeof(float));
    if (!actual) fail("activation readback allocation");
    if (!ds4_gpu_tensor_read(f->x[stream], 0u, actual,
                             count * sizeof(float)) ||
        memcmp(actual, f->x_host[stream], count * sizeof(float)) != 0) {
        fail("activation modified");
    }
    free(actual);
}

static void test_shapes(fixture *f, bool explicit_enable) {
    set_pair_controls(explicit_enable, false, explicit_enable);
    for (uint32_t i = 0; i < sizeof(k_tokens) / sizeof(k_tokens[0]); i++) {
        const uint32_t n_tokens = k_tokens[i];
        poison_outputs(f, 0u);
        if (!run_baseline(f, 0u, n_tokens)) fail("baseline projection");
        if (!run_candidate(f, 0u, n_tokens)) fail("required pair projection");
        check_outputs(f, 0u, n_tokens);
        check_x_unchanged(f, 0u);
    }
}

static void test_default_scope(fixture *f) {
    set_pair_controls(false, false, false);
    poison_outputs(f, 0u);
    const uint32_t shapes[][3] = {
        {IN_DIM, OUT1_DIM, OUT1_DIM}, {256u, OUT0_DIM, OUT1_DIM},
    };
    for (size_t i = 0; i < sizeof(shapes)/sizeof(shapes[0]); i++) {
        const int rc = ds4_gpu_matmul_q4_K_pair_tensor(
            f->candidate[0][0], f->candidate[0][1], f->model, f->model_size,
            f->weight_offset[0], f->weight_offset[1],
            shapes[i][0], shapes[i][1], shapes[i][2], f->x[0], 32u);
        if (rc != 0) fail("default enabled an unmeasured shape");
    }
    const char *disable_flags[] = {
        "DS4_METAL_ENABLE_Q4_PREFILL_PAIR_F16_RHS",
        "DS4_METAL_DISABLE_Q4_PREFILL_PAIR_F16_RHS",
        "DS4_METAL_DISABLE_Q4_DENSE_PAIR",
    };
    for (size_t i = 0; i < sizeof(disable_flags)/sizeof(disable_flags[0]); i++) {
        if (setenv(disable_flags[i], i == 0 ? "0" : "1", 1)) fail("disable flag");
        const int rc = ds4_gpu_matmul_q4_K_pair_tensor(
            f->candidate[0][0], f->candidate[0][1], f->model, f->model_size,
            f->weight_offset[0], f->weight_offset[1], IN_DIM,
            OUT0_DIM, OUT1_DIM, f->x[0], 32u);
        if (rc != 0) fail("default pair ignored an explicit opt-out");
        if (unsetenv(disable_flags[i])) fail("restore disable flag");
    }
    check_tensor_all_poison(f->candidate[0][0], output_count(0u), "default-scope-out0");
    check_tensor_all_poison(f->candidate[0][1], output_count(1u), "default-scope-out1");
    check_x_unchanged(f, 0u);
    fprintf(stderr, "Metal Q4 prefill pair default scope and opt-outs: PASS\n");
}

static void test_alias_rejection(fixture *f) {
    const uint32_t n_tokens = 32u;
    const uint64_t alias_bytes =
        active_output_count(0u, n_tokens) * sizeof(float);
    ds4_gpu_tensor *alias =
        ds4_gpu_tensor_view(f->x[0], 0u, alias_bytes);
    if (!alias) fail("alias tensor view");
    poison_outputs(f, 0u);
    set_pair_controls(true, false, false);
    const int rc = ds4_gpu_matmul_q4_K_pair_tensor(
        alias, f->candidate[0][1], f->model, f->model_size,
        f->weight_offset[0], f->weight_offset[1],
        IN_DIM, OUT0_DIM, OUT1_DIM, f->x[0], n_tokens);
    ds4_gpu_tensor_free(alias);
    if (rc != 0) fail("alias was not rejected with fallback status");
    check_x_unchanged(f, 0u);

    check_tensor_all_poison(f->candidate[0][1], output_count(1u),
                            "alias-reject");
    fprintf(stderr, "Metal Q4 prefill pair alias rejection: PASS\n");
}

/* The optimized tile changes neither the admitted token range nor the
 * output alignment contract. Rejection must leave caller buffers untouched. */
static void test_ineligible_shapes(fixture *f) {
    const uint32_t cases[][3] = {
        {9u, OUT0_DIM, OUT1_DIM}, {31u, OUT0_DIM, OUT1_DIM},
        {33u, OUT0_DIM, OUT1_DIM}, {127u, OUT0_DIM, OUT1_DIM},
        {129u, OUT0_DIM, OUT1_DIM}, {32u, OUT0_DIM-1u, OUT1_DIM},
        {32u, OUT0_DIM, OUT1_DIM-1u},
    };
    set_pair_controls(true, false, false);
    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); i++) {
        poison_outputs(f, 0u);
        const int rc = ds4_gpu_matmul_q4_K_pair_tensor(
            f->candidate[0][0], f->candidate[0][1], f->model, f->model_size,
            f->weight_offset[0], f->weight_offset[1], IN_DIM,
            cases[i][1], cases[i][2], f->x[0], cases[i][0]);
        if (rc != 0) fail("ineligible pair did not fall back");
        check_tensor_all_poison(f->candidate[0][0], output_count(0u), "shape-out0");
        check_tensor_all_poison(f->candidate[0][1], output_count(1u), "shape-out1");
        check_x_unchanged(f, 0u);
    }
    fprintf(stderr, "Metal Q4 prefill pair token/alignment boundaries: PASS\n");
}

static void test_required_failure_is_local(fixture *f) {
    const uint32_t n_tokens = 32u;
    const uint64_t active = active_output_count(0u, n_tokens);
    const uint64_t count = output_count(0u);
    float *reference = malloc((size_t)active * sizeof(float));
    float *actual = malloc((size_t)count * sizeof(float));
    if (!reference || !actual) fail("REQUIRE oracle allocation");

    set_pair_controls(false, false, false);
    if (!ds4_gpu_matmul_quant_tensor(
            f->baseline[0][0], f->model, f->model_size,
            f->weight_offset[0], Q4_K_TYPE, IN_DIM, OUT0_DIM,
            f->x[0], n_tokens) ||
        !ds4_gpu_tensor_read(f->baseline[0][0], 0u, reference,
                             active * sizeof(float))) {
        fail("REQUIRE reference matmul");
    }

    poison_outputs(f, 0u);
    set_pair_controls(true, true, true);
    const int rc = ds4_gpu_matmul_q4_K_pair_tensor(
        f->candidate[0][0], f->candidate[0][1],
        f->model, f->model_size,
        f->weight_offset[0], f->weight_offset[1],
        IN_DIM, OUT0_DIM, OUT1_DIM, f->x[0], n_tokens);
    if (rc != -1) {
        fprintf(stderr,
                "Metal Q4 prefill pair REQUIRE negative returned %d, expected -1\n",
                rc);
        fail("REQUIRE negative status");
    }
    check_tensor_all_poison(f->candidate[0][0], output_count(0u),
                            "require-negative-out0");
    check_tensor_all_poison(f->candidate[0][1], output_count(1u),
                            "require-negative-out1");

    set_pair_controls(false, false, false);
    if (!ds4_gpu_matmul_quant_tensor(
            f->baseline[0][0], f->model, f->model_size,
            f->weight_offset[0], Q4_K_TYPE, IN_DIM, OUT0_DIM,
            f->x[0], n_tokens)) {
        fail("REQUIRE failure contaminated next quant matmul");
    }
    if (!ds4_gpu_tensor_read(f->baseline[0][0], 0u, actual,
                             count * sizeof(float)) ||
        memcmp(actual, reference, active * sizeof(float)) != 0) {
        fail("REQUIRE failure changed next quant matmul");
    }
    check_poison_suffix(actual, active, count, "require-next-matmul");
    free(actual);
    free(reference);
    fprintf(stderr,
            "Metal Q4 prefill pair REQUIRE negative: PASS "
            "rc=-1 next_matmul_bitwise=1\n");
}

static void test_two_stream_async(fixture *f) {
    static const uint32_t stream_tokens[TEST_STREAMS] = {96u, 128u};
    set_pair_controls(true, false, true);
    for (uint32_t stream = 0; stream < TEST_STREAMS; stream++) {
        poison_outputs(f, stream);
        if (!run_baseline(f, stream, stream_tokens[stream])) {
            fail("async baseline");
        }
    }

    uint32_t submitted = 0u;
    for (uint32_t stream = 0; stream < TEST_STREAMS; stream++) {
        ds4_gpu_set_stream((int)stream);
        if (!ds4_gpu_begin_commands()) fail("async begin");
        const int rc = ds4_gpu_matmul_q4_K_pair_tensor(
            f->candidate[stream][0], f->candidate[stream][1],
            f->model, f->model_size,
            f->weight_offset[0], f->weight_offset[1],
            IN_DIM, OUT0_DIM, OUT1_DIM, f->x[stream],
            stream_tokens[stream]);
        if (rc != 1 || !ds4_gpu_end_commands_async()) {
            fail("async pair submission");
        }
        submitted++;
    }
    for (uint32_t stream = 0; stream < submitted; stream++) {
        if (!ds4_gpu_wait_stream((int)stream)) fail("async stream wait");
    }
    ds4_gpu_set_stream(0);
    for (uint32_t stream = 0; stream < TEST_STREAMS; stream++) {
        check_outputs(f, stream, stream_tokens[stream]);
        check_x_unchanged(f, stream);
    }
    fprintf(stderr,
            "Metal Q4 prefill pair async streams: PASS streams=2 "
            "tokens=96,128 scratch_isolation=bitwise\n");
}

int main(void) {
    set_pair_controls(false, false, false);
    set_flag("DS4_METAL_DISABLE_Q4_DENSE_PAIR", false);
    set_flag("DS4_METAL_DISABLE_CONTIG_F32_F16_COPY", false);

    if (!ds4_gpu_init()) {
        fprintf(stderr,
                "test_metal_q4_prefill_pair: SKIP: no Metal device\n");
        return 0;
    }
    if (!ds4_gpu_device_is_pre_m5_apple_silicon()) {
        fprintf(stderr,
                "test_metal_q4_prefill_pair: SKIP: requires Apple M1-M4\n");
        ds4_gpu_cleanup();
        return 0;
    }

    fixture f;
    fixture_init(&f);
    test_shapes(&f, false);
    test_shapes(&f, true);
    test_default_scope(&f);
    test_alias_rejection(&f);
    test_ineligible_shapes(&f);
    test_required_failure_is_local(&f);
    test_two_stream_async(&f);
    fixture_destroy(&f);
    fprintf(stderr,
            "test_metal_q4_prefill_pair PASS tokens=32,64,96,128 "
            "bitwise=1 require=1 alias=1 ssd_spans=1 streams=2 "
            "unretained=%s\n",
            getenv("DS4_METAL_UNRETAINED_COMMAND_BUFFERS") ? "on" : "off");
    return 0;
}

#else

int main(void) {
    fprintf(stderr,
            "test_metal_q4_prefill_pair: SKIP: Metal requires macOS\n");
    return 0;
}

#endif
