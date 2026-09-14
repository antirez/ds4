#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    QK_K = 256,
    Q4_BLOCK_BYTES = 144,
    THREADS_PER_GROUP = 128,
    THREADGROUP_MEMORY_BYTES = 8192,
    GUARD_BYTES = 256,
    MAX_TOKENS = 65,
    DEFAULT_IN_DIM = 4096,
    DEFAULT_OUT_DIM = 1024,
    DEFAULT_SAMPLES = 12,
    DEFAULT_WARMUP_DISPATCHES = 16,
};

static const uint32_t k_guard = 0x7fc12345u;
static const uint32_t k_baseline_poison = 0x7fc0b001u;
static const uint32_t k_candidate_poison = 0x7fc0c001u;
static const uint32_t k_token_cases[] = {
    9u, 16u, 17u, 31u, 33u, 47u, 63u, 65u,
};

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2];
} block_q4_K_host;

typedef struct {
    int32_t ne00;
    int32_t ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne0;
    int32_t ne1;
    int16_t r2;
    int16_t r3;
} mul_mm_args;

_Static_assert(sizeof(block_q4_K_host) == Q4_BLOCK_BYTES,
               "Q4_K host fixture must match GGUF/Metal layout");
_Static_assert(sizeof(mul_mm_args) == 88,
               "Metal mul_mm argument ABI changed");

typedef struct {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t sets;
    uint32_t dispatches;
    uint32_t warmup_dispatches;
    uint32_t samples;
    bool sets_explicit;
    bool dispatches_explicit;
} bench_config;

typedef enum {
    ARM_BASELINE,
    ARM_CANDIDATE,
} bench_arm;

typedef struct {
    __strong id<MTLDevice> device;
    __strong id<MTLCommandQueue> queue;
    __strong id<MTLComputePipelineState> baseline_pipeline;
    __strong id<MTLComputePipelineState> candidate_pipeline;
    __strong id<MTLBuffer> weights;
    __strong id<MTLBuffer> x;
    __strong id<MTLBuffer> baseline_out;
    __strong id<MTLBuffer> candidate_out;
    uint8_t *x_snapshot;
    NSUInteger weight_set_bytes;
    NSUInteger output_stride;
    NSUInteger x_offset;
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t sets;
} fixture;

static void usage(FILE *fp, const char *argv0) {
    fprintf(fp,
            "usage: %s [options]\n"
            "\n"
            "Kernel-only A/B of the production Metal Q4_K generic-MM tail "
            "SIMDgroup cull.\n"
            "\n"
            "  --in-dim N       K dimension (default: %u)\n"
            "  --out-dim N      M dimension (default: %u)\n"
            "  --sets N         rotating resident weight sets (default: auto)\n"
            "  --dispatches N   dispatches per sample (default: auto)\n"
            "  --warmup N       dispatches per warmup arm (default: %u)\n"
            "  --samples N      samples per arm, multiple of 4 (default: %u)\n"
            "  -h, --help       show this help\n"
            "\n"
            "The default shape is the Flash Q-A projection 4096->1024. "
            "Use --in-dim 1024 --out-dim 32768 for attn_q_b.\n"
            "Set DS4_SOURCE_ROOT when running outside the repository root.\n",
            argv0, DEFAULT_IN_DIM, DEFAULT_OUT_DIM,
            DEFAULT_WARMUP_DISPATCHES, DEFAULT_SAMPLES);
}

static uint32_t parse_u32(const char *text, const char *option,
                          uint32_t minimum) {
    char *end = NULL;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || !text[0] || !end || *end ||
        value < minimum || value > UINT32_MAX) {
        fprintf(stderr, "metal-q4-mm-tail-cull-bench: invalid %s: %s\n",
                option, text);
        exit(2);
    }
    return (uint32_t)value;
}

static const char *need_arg(int *index, int argc, char **argv) {
    if (*index + 1 >= argc) {
        fprintf(stderr, "metal-q4-mm-tail-cull-bench: %s needs a value\n",
                argv[*index]);
        exit(2);
    }
    return argv[++*index];
}

static bench_config parse_options(int argc, char **argv) {
    bench_config config = {
        .in_dim = DEFAULT_IN_DIM,
        .out_dim = DEFAULT_OUT_DIM,
        .sets = 0u,
        .dispatches = 0u,
        .warmup_dispatches = DEFAULT_WARMUP_DISPATCHES,
        .samples = DEFAULT_SAMPLES,
    };
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(stdout, argv[0]);
            exit(0);
        } else if (!strcmp(argv[i], "--in-dim")) {
            const char *option = argv[i];
            config.in_dim =
                parse_u32(need_arg(&i, argc, argv), option, QK_K);
        } else if (!strcmp(argv[i], "--out-dim")) {
            const char *option = argv[i];
            config.out_dim =
                parse_u32(need_arg(&i, argc, argv), option, 64u);
        } else if (!strcmp(argv[i], "--sets")) {
            const char *option = argv[i];
            config.sets = parse_u32(need_arg(&i, argc, argv), option, 1u);
            config.sets_explicit = true;
        } else if (!strcmp(argv[i], "--dispatches")) {
            const char *option = argv[i];
            config.dispatches =
                parse_u32(need_arg(&i, argc, argv), option, 1u);
            config.dispatches_explicit = true;
        } else if (!strcmp(argv[i], "--warmup")) {
            const char *option = argv[i];
            config.warmup_dispatches =
                parse_u32(need_arg(&i, argc, argv), option, 0u);
        } else if (!strcmp(argv[i], "--samples")) {
            const char *option = argv[i];
            config.samples =
                parse_u32(need_arg(&i, argc, argv), option, 4u);
        } else {
            fprintf(stderr,
                    "metal-q4-mm-tail-cull-bench: unknown option: %s\n",
                    argv[i]);
            usage(stderr, argv[0]);
            exit(2);
        }
    }
    if ((config.in_dim % QK_K) != 0u ||
        (config.out_dim % 64u) != 0u) {
        fprintf(stderr,
                "metal-q4-mm-tail-cull-bench: in-dim must be divisible by "
                "%u and out-dim by 64\n", QK_K);
        exit(2);
    }
    if ((config.samples % 4u) != 0u) {
        fprintf(stderr,
                "metal-q4-mm-tail-cull-bench: --samples must be divisible "
                "by 4 for equal ABBA/BAAB cycles\n");
        exit(2);
    }
    return config;
}

static bool checked_mul(NSUInteger a, NSUInteger b, NSUInteger *out) {
    if (a != 0u && b > NSUIntegerMax / a) return false;
    *out = a * b;
    return true;
}

static NSUInteger align_up(NSUInteger value, NSUInteger alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static NSString *metal_prelude(void) {
    return @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "#define MAX(x, y) ((x) > (y) ? (x) : (y))\n"
            "#define MIN(x, y) ((x) < (y) ? (x) : (y))\n"
            "#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }\n"
            "#define QK8_0 32\n"
            "#ifndef QK_K\n#define QK_K 256\n#endif\n"
            "#define N_SIMDWIDTH 32\n"
            "#define N_R0_Q8_0 2\n"
            "#define N_SG_Q8_0 4\n"
            "#define FC_MUL_MV 600\n"
            "#define FC_MUL_MM 700\n"
            "#define FC_BIN 1300\n"
            "#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"
            "#define M_PI_F 3.14159265358979323846f\n"
            "enum ds4_sort_order { DS4_SORT_ORDER_ASC, DS4_SORT_ORDER_DESC };\n"
            "struct block_q8_0 { half d; int8_t qs[QK8_0]; };\n"
            "struct block_q8_K { float d; int8_t qs[QK_K]; "
            "int16_t bsums[QK_K / 16]; };\n";
}

static NSString *load_metal_source(void) {
    static const char *paths[] = {
        "metal/activations.metal",
        "metal/flash_attn.metal",
        "metal/dense.metal",
        "metal/moe.metal",
        "metal/dsv4_hc.metal",
        "metal/unary.metal",
        "metal/dsv4_kv.metal",
        "metal/dsv4_rope.metal",
        "metal/dsv4_misc.metal",
        "metal/argsort.metal",
        "metal/cpy.metal",
        "metal/concat.metal",
        "metal/get_rows.metal",
        "metal/sum_rows.metal",
        "metal/softmax.metal",
        "metal/repeat.metal",
        "metal/glu.metal",
        "metal/norm.metal",
        "metal/bin.metal",
        "metal/set_rows.metal",
    };
    const char *root_env = getenv("DS4_SOURCE_ROOT");
    NSString *root = root_env && root_env[0]
        ? [NSString stringWithUTF8String:root_env]
        : @".";
    NSMutableString *source = [NSMutableString stringWithString:metal_prelude()];
    for (size_t i = 0; i < sizeof(paths) / sizeof(paths[0]); i++) {
        NSString *relative = [NSString stringWithUTF8String:paths[i]];
        NSString *path = [root stringByAppendingPathComponent:relative];
        NSError *error = nil;
        NSString *part = [NSString stringWithContentsOfFile:path
                                                   encoding:NSUTF8StringEncoding
                                                      error:&error];
        if (!part) {
            fprintf(stderr,
                    "metal-q4-mm-tail-cull-bench: cannot read %s: %s\n",
                    [path fileSystemRepresentation],
                    [[error localizedDescription] UTF8String]);
            return nil;
        }
        [source appendFormat:@"\n// appended %@\n%@\n", relative, part];
    }
    return source;
}

static id<MTLComputePipelineState> make_mm_pipeline(
        id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
    bool bc_inp = false;
    bool bc_out = true;
    MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
    [values setConstantValue:&bc_inp type:MTLDataTypeBool atIndex:700];
    [values setConstantValue:&bc_out type:MTLDataTypeBool atIndex:701];
    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name
                                            constantValues:values
                                                     error:&error];
    if (!function) {
        fprintf(stderr, "metal-q4-mm-tail-cull-bench: function %s: %s\n",
                [name UTF8String], [[error localizedDescription] UTF8String]);
        return nil;
    }
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        fprintf(stderr, "metal-q4-mm-tail-cull-bench: pipeline %s: %s\n",
                [name UTF8String], [[error localizedDescription] UTF8String]);
    }
    return pipeline;
}

static uint32_t lcg_next(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static void fill_q4(void *storage, NSUInteger bytes) {
    block_q4_K_host *blocks = storage;
    const NSUInteger count = bytes / sizeof(*blocks);
    uint32_t state = 0x41c64e6du;
    for (NSUInteger b = 0; b < count; b++) {
        blocks[b].d = (uint16_t)(0x2400u | (lcg_next(&state) & 0x03ffu));
        blocks[b].dmin =
            (uint16_t)(0x1c00u | (lcg_next(&state) & 0x03ffu));
        for (size_t i = 0; i < sizeof(blocks[b].scales); i++) {
            blocks[b].scales[i] = (uint8_t)lcg_next(&state);
        }
        for (size_t i = 0; i < sizeof(blocks[b].qs); i++) {
            blocks[b].qs[i] = (uint8_t)lcg_next(&state);
        }
    }
}

static void fill_guard(id<MTLBuffer> buffer) {
    uint32_t *words = buffer.contents;
    for (NSUInteger i = 0; i < buffer.length / sizeof(*words); i++) {
        words[i] = k_guard;
    }
}

static void fill_output_payload(id<MTLBuffer> buffer, NSUInteger stride,
                                uint32_t sets, NSUInteger payload_bytes,
                                uint32_t poison) {
    for (uint32_t set = 0; set < sets; set++) {
        uint32_t *payload = (uint32_t *)((uint8_t *)buffer.contents +
            (NSUInteger)set * stride + GUARD_BYTES);
        for (NSUInteger offset = 0; offset < payload_bytes;
             offset += sizeof(*payload)) {
            payload[offset / sizeof(*payload)] = poison;
        }
    }
}

static void fill_activation(float *values, NSUInteger count) {
    for (NSUInteger i = 0; i < count; i++) {
        const int32_t centered = (int32_t)((i * 37u + i / 11u) % 257u) - 128;
        values[i] = (float)centered / 96.0f;
    }
}

static id<MTLBuffer> alloc_buffer(id<MTLDevice> device, NSUInteger bytes,
                                  NSString *label) {
    id<MTLBuffer> buffer =
        [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!buffer) {
        fprintf(stderr,
                "metal-q4-mm-tail-cull-bench: allocation failed for %s "
                "(%llu bytes)\n",
                [label UTF8String], (unsigned long long)bytes);
        return nil;
    }
    buffer.label = label;
    return buffer;
}

static bool init_fixture(fixture *f, bench_config *config) {
    f->device = MTLCreateSystemDefaultDevice();
    if (!f->device) {
        fprintf(stderr, "metal-q4-mm-tail-cull-bench: no Metal device\n");
        return false;
    }
    f->queue = [f->device newCommandQueue];
    NSString *source = load_metal_source();
    if (!f->queue || !source) return false;

    NSError *error = nil;
    id<MTLLibrary> library =
        [f->device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        fprintf(stderr,
                "metal-q4-mm-tail-cull-bench: Metal compile failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return false;
    }
    f->baseline_pipeline = make_mm_pipeline(
        f->device, library, @"kernel_mul_mm_q4_K_f32");
    f->candidate_pipeline = make_mm_pipeline(
        f->device, library, @"kernel_mul_mm_q4_K_f32_tail_cull");
    if (!f->baseline_pipeline || !f->candidate_pipeline) return false;
    if (f->baseline_pipeline.threadExecutionWidth != 32u ||
        f->candidate_pipeline.threadExecutionWidth != 32u ||
        f->baseline_pipeline.maxTotalThreadsPerThreadgroup < THREADS_PER_GROUP ||
        f->candidate_pipeline.maxTotalThreadsPerThreadgroup < THREADS_PER_GROUP) {
        fprintf(stderr,
                "metal-q4-mm-tail-cull-bench: unexpected pipeline geometry\n");
        return false;
    }

    f->in_dim = config->in_dim;
    f->out_dim = config->out_dim;
    const NSUInteger row_bytes =
        (NSUInteger)(f->in_dim / QK_K) * Q4_BLOCK_BYTES;
    if (!checked_mul(row_bytes, f->out_dim, &f->weight_set_bytes)) {
        fprintf(stderr, "metal-q4-mm-tail-cull-bench: weight size overflow\n");
        return false;
    }
    if (!config->sets_explicit) {
        const NSUInteger target = 144u * 1024u * 1024u;
        NSUInteger sets = (target + f->weight_set_bytes - 1u) /
                          f->weight_set_bytes;
        if (sets < 1u) sets = 1u;
        if (sets > 64u) sets = 64u;
        config->sets = (uint32_t)sets;
    }
    if (!config->dispatches_explicit) {
        const NSUInteger target = 512u * 1024u * 1024u;
        NSUInteger dispatches =
            (target + f->weight_set_bytes - 1u) / f->weight_set_bytes;
        if (dispatches < 8u) dispatches = 8u;
        if (dispatches > 256u) dispatches = 256u;
        config->dispatches = (uint32_t)dispatches;
    }
    f->sets = config->sets;

    NSUInteger max_output_bytes = 0u;
    if (!checked_mul((NSUInteger)MAX_TOKENS, f->out_dim,
                     &max_output_bytes) ||
        !checked_mul(max_output_bytes, sizeof(float), &max_output_bytes)) {
        fprintf(stderr, "metal-q4-mm-tail-cull-bench: output size overflow\n");
        return false;
    }
    f->output_stride =
        align_up(GUARD_BYTES + max_output_bytes + GUARD_BYTES, 256u);
    f->x_offset = GUARD_BYTES;

    NSUInteger weight_bytes = 0u;
    NSUInteger output_bytes = 0u;
    NSUInteger x_payload_bytes = 0u;
    if (!checked_mul(f->weight_set_bytes, f->sets, &weight_bytes) ||
        !checked_mul(f->output_stride, f->sets, &output_bytes) ||
        !checked_mul((NSUInteger)MAX_TOKENS, f->in_dim, &x_payload_bytes) ||
        !checked_mul(x_payload_bytes, sizeof(float), &x_payload_bytes)) {
        fprintf(stderr, "metal-q4-mm-tail-cull-bench: fixture size overflow\n");
        return false;
    }
    f->weights = alloc_buffer(f->device, weight_bytes, @"q4-resident-weights");
    f->x = alloc_buffer(f->device,
                        GUARD_BYTES + x_payload_bytes + GUARD_BYTES,
                        @"prefill-activation");
    f->baseline_out =
        alloc_buffer(f->device, output_bytes, @"q4-baseline-output");
    f->candidate_out =
        alloc_buffer(f->device, output_bytes, @"q4-candidate-output");
    if (!f->weights || !f->x || !f->baseline_out || !f->candidate_out) {
        return false;
    }

    fill_q4(f->weights.contents, f->weights.length);
    fill_guard(f->x);
    fill_activation((float *)((uint8_t *)f->x.contents + f->x_offset),
                    (NSUInteger)MAX_TOKENS * f->in_dim);
    fill_guard(f->baseline_out);
    fill_guard(f->candidate_out);
    f->x_snapshot = malloc(f->x.length);
    if (!f->x_snapshot) {
        fprintf(stderr,
                "metal-q4-mm-tail-cull-bench: input snapshot allocation failed\n");
        return false;
    }
    memcpy(f->x_snapshot, f->x.contents, f->x.length);
    return true;
}

static mul_mm_args make_args(const fixture *f, uint32_t n_tokens) {
    const uint64_t row_bytes =
        (uint64_t)(f->in_dim / QK_K) * Q4_BLOCK_BYTES;
    return (mul_mm_args) {
        .ne00 = (int32_t)f->in_dim,
        .ne02 = 1,
        .nb01 = row_bytes,
        .nb02 = row_bytes * f->out_dim,
        .nb03 = row_bytes * f->out_dim,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = (uint64_t)f->in_dim * sizeof(float),
        .nb12 = (uint64_t)f->in_dim * n_tokens * sizeof(float),
        .nb13 = (uint64_t)f->in_dim * n_tokens * sizeof(float),
        .ne0 = (int32_t)f->out_dim,
        .ne1 = (int32_t)n_tokens,
        .r2 = 1,
        .r3 = 1,
    };
}

static void encode_dispatch(fixture *f, id<MTLComputeCommandEncoder> encoder,
                            bench_arm arm, uint32_t n_tokens,
                            uint32_t set) {
    const mul_mm_args args = make_args(f, n_tokens);
    id<MTLBuffer> output = arm == ARM_BASELINE
        ? f->baseline_out : f->candidate_out;
    [encoder setComputePipelineState:arm == ARM_BASELINE
        ? f->baseline_pipeline : f->candidate_pipeline];
    [encoder setBytes:&args length:sizeof(args) atIndex:0];
    [encoder setBuffer:f->weights
                 offset:(NSUInteger)set * f->weight_set_bytes
                atIndex:1];
    [encoder setBuffer:f->x offset:f->x_offset atIndex:2];
    [encoder setBuffer:output
                 offset:(NSUInteger)set * f->output_stride + GUARD_BYTES
                atIndex:3];
    [encoder setThreadgroupMemoryLength:THREADGROUP_MEMORY_BYTES atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake((n_tokens + 31u) / 32u,
                                               f->out_dim / 64u, 1u)
          threadsPerThreadgroup:MTLSizeMake(THREADS_PER_GROUP, 1u, 1u)];
}

static bool finish_command_buffer(id<MTLCommandBuffer> cb, const char *label) {
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusCompleted) return true;
    fprintf(stderr, "metal-q4-mm-tail-cull-bench: %s failed: %s\n",
            label, cb.error ? [[cb.error localizedDescription] UTF8String]
                            : "unknown Metal error");
    return false;
}

static bool check_guard_words(const uint8_t *bytes, NSUInteger begin,
                              NSUInteger end, const char *label,
                              uint32_t set, uint32_t n_tokens) {
    for (NSUInteger offset = begin; offset < end; offset += sizeof(uint32_t)) {
        uint32_t actual = 0u;
        memcpy(&actual, bytes + offset, sizeof(actual));
        if (actual != k_guard) {
            fprintf(stderr,
                    "metal-q4-mm-tail-cull-bench: %s canary changed "
                    "N=%u set=%u offset=%llu\n",
                    label, n_tokens, set, (unsigned long long)offset);
            return false;
        }
    }
    return true;
}

static bool check_outputs(fixture *f, uint32_t n_tokens) {
    const NSUInteger payload_bytes =
        (NSUInteger)n_tokens * f->out_dim * sizeof(float);
    const uint8_t *baseline = f->baseline_out.contents;
    const uint8_t *candidate = f->candidate_out.contents;
    for (uint32_t set = 0; set < f->sets; set++) {
        const NSUInteger record = (NSUInteger)set * f->output_stride;
        const NSUInteger payload = record + GUARD_BYTES;
        const NSUInteger payload_end = payload + payload_bytes;
        if (memcmp(baseline + payload, candidate + payload, payload_bytes) != 0) {
            fprintf(stderr,
                    "metal-q4-mm-tail-cull-bench: bitwise mismatch "
                    "N=%u set=%u\n", n_tokens, set);
            return false;
        }
        if (!check_guard_words(baseline, record, payload,
                               "baseline prefix", set, n_tokens) ||
            !check_guard_words(candidate, record, payload,
                               "candidate prefix", set, n_tokens) ||
            !check_guard_words(baseline, payload_end,
                               record + f->output_stride,
                               "baseline suffix", set, n_tokens) ||
            !check_guard_words(candidate, payload_end,
                               record + f->output_stride,
                               "candidate suffix", set, n_tokens)) {
            return false;
        }
    }
    if (memcmp(f->x.contents, f->x_snapshot, f->x.length) != 0) {
        fprintf(stderr,
                "metal-q4-mm-tail-cull-bench: input was modified at N=%u\n",
                n_tokens);
        return false;
    }
    return true;
}

static bool run_oracle_case(fixture *f, uint32_t n_tokens) {
    fill_guard(f->baseline_out);
    fill_guard(f->candidate_out);
    const NSUInteger payload_bytes =
        (NSUInteger)n_tokens * f->out_dim * sizeof(float);
    fill_output_payload(f->baseline_out, f->output_stride, f->sets,
                        payload_bytes, k_baseline_poison);
    fill_output_payload(f->candidate_out, f->output_stride, f->sets,
                        payload_bytes, k_candidate_poison);
    id<MTLCommandBuffer> cb = [f->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
    for (uint32_t set = 0; set < f->sets; set++) {
        encode_dispatch(f, encoder, ARM_BASELINE, n_tokens, set);
    }
    for (uint32_t set = 0; set < f->sets; set++) {
        encode_dispatch(f, encoder, ARM_CANDIDATE, n_tokens, set);
    }
    [encoder endEncoding];
    if (!finish_command_buffer(cb, "oracle")) return false;
    return check_outputs(f, n_tokens);
}

static bool run_workload(fixture *f, bench_arm arm, uint32_t n_tokens,
                         uint32_t dispatches, uint32_t start_set,
                         double *gpu_seconds) {
    id<MTLCommandBuffer> cb = [f->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
    for (uint32_t i = 0; i < dispatches; i++) {
        encode_dispatch(f, encoder, arm, n_tokens,
                        (start_set + i) % f->sets);
    }
    [encoder endEncoding];
    if (!finish_command_buffer(cb, arm == ARM_BASELINE
                               ? "baseline workload" : "candidate workload")) {
        return false;
    }
    const double elapsed = cb.GPUEndTime - cb.GPUStartTime;
    if (!(elapsed > 0.0)) {
        fprintf(stderr,
                "metal-q4-mm-tail-cull-bench: GPU timestamps unavailable\n");
        return false;
    }
    *gpu_seconds = elapsed;
    return true;
}

static int compare_double(const void *lhs, const void *rhs) {
    const double a = *(const double *)lhs;
    const double b = *(const double *)rhs;
    return (a > b) - (a < b);
}

static double percentile(const double *sorted, uint32_t count, double p) {
    const double position = p * (double)(count - 1u);
    const uint32_t lo = (uint32_t)position;
    const uint32_t hi = lo + 1u < count ? lo + 1u : lo;
    const double fraction = position - (double)lo;
    return sorted[lo] + (sorted[hi] - sorted[lo]) * fraction;
}

static bool run_benchmark_case(fixture *f, const bench_config *config,
                               uint32_t n_tokens) {
    double ignored = 0.0;
    if (config->warmup_dispatches != 0u) {
        static const bench_arm warm_order[] = {
            ARM_BASELINE, ARM_CANDIDATE,
            ARM_CANDIDATE, ARM_BASELINE,
        };
        for (size_t i = 0; i < sizeof(warm_order) / sizeof(warm_order[0]); i++) {
            if (!run_workload(f, warm_order[i], n_tokens,
                              config->warmup_dispatches,
                              (uint32_t)(i * config->warmup_dispatches) % f->sets,
                              &ignored)) {
                return false;
            }
        }
    }

    double *baseline = calloc(config->samples, sizeof(*baseline));
    double *candidate = calloc(config->samples, sizeof(*candidate));
    double *baseline_sorted = calloc(config->samples, sizeof(*baseline_sorted));
    double *candidate_sorted = calloc(config->samples, sizeof(*candidate_sorted));
    if (!baseline || !candidate || !baseline_sorted || !candidate_sorted) {
        fprintf(stderr,
                "metal-q4-mm-tail-cull-bench: sample allocation failed\n");
        free(baseline);
        free(candidate);
        free(baseline_sorted);
        free(candidate_sorted);
        return false;
    }

    uint32_t baseline_count = 0u;
    uint32_t candidate_count = 0u;
    bool ok = true;
    for (uint32_t cycle = 0; ok && cycle < config->samples / 2u; cycle++) {
        static const bench_arm abba[] = {
            ARM_BASELINE, ARM_CANDIDATE,
            ARM_CANDIDATE, ARM_BASELINE,
        };
        static const bench_arm baab[] = {
            ARM_CANDIDATE, ARM_BASELINE,
            ARM_BASELINE, ARM_CANDIDATE,
        };
        const bench_arm *order = (cycle & 1u) ? baab : abba;
        for (uint32_t position = 0; ok && position < 4u; position++) {
            const bench_arm arm = order[position];
            const uint32_t arm_index = arm == ARM_BASELINE
                ? baseline_count : candidate_count;
            const uint32_t start_set =
                (uint32_t)((uint64_t)arm_index * config->dispatches % f->sets);
            double elapsed = 0.0;
            ok = run_workload(f, arm, n_tokens, config->dispatches,
                              start_set, &elapsed);
            if (ok && arm == ARM_BASELINE) baseline[baseline_count++] = elapsed;
            if (ok && arm == ARM_CANDIDATE) candidate[candidate_count++] = elapsed;
        }
    }
    if (ok && (baseline_count != config->samples ||
               candidate_count != config->samples)) {
        ok = false;
    }
    if (ok) ok = check_outputs(f, n_tokens);

    if (ok) {
        memcpy(baseline_sorted, baseline,
               config->samples * sizeof(*baseline_sorted));
        memcpy(candidate_sorted, candidate,
               config->samples * sizeof(*candidate_sorted));
        qsort(baseline_sorted, config->samples,
              sizeof(*baseline_sorted), compare_double);
        qsort(candidate_sorted, config->samples,
              sizeof(*candidate_sorted), compare_double);
        const double scale = 1.0e6 / config->dispatches;
        const double baseline_median =
            percentile(baseline_sorted, config->samples, 0.50) * scale;
        const double baseline_p95 =
            percentile(baseline_sorted, config->samples, 0.95) * scale;
        const double candidate_median =
            percentile(candidate_sorted, config->samples, 0.50) * scale;
        const double candidate_p95 =
            percentile(candidate_sorted, config->samples, 0.95) * scale;
        double log_speedup = 0.0;
        for (uint32_t i = 0; i < config->samples; i++) {
            log_speedup += log(baseline[i] / candidate[i]);
        }
        const double paired_speedup = exp(log_speedup / config->samples);
        const double reduction =
            (baseline_median - candidate_median) * 100.0 / baseline_median;
        printf("  N=%-2u baseline %.3f us [p95 %.3f]  "
               "candidate %.3f us [p95 %.3f]  "
               "median %.3fx (%+.2f%%), paired-gmean %.3fx\n",
               n_tokens, baseline_median, baseline_p95,
               candidate_median, candidate_p95,
               baseline_median / candidate_median, reduction,
               paired_speedup);
    }

    free(baseline);
    free(candidate);
    free(baseline_sorted);
    free(candidate_sorted);
    return ok;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        bench_config config = parse_options(argc, argv);
        fixture f = {0};
        if (!init_fixture(&f, &config)) return 1;

        printf("Metal Q4_K generic-MM tail-cull kernel-only A/B\n");
        printf("  shape: %u -> %u, resident anonymous Q4_K weights\n",
               config.in_dim, config.out_dim);
        printf("  working set: %.1f MiB across %u rotating sets\n",
               (double)f.weights.length / (1024.0 * 1024.0), f.sets);
        printf("  design: ABBA/BAAB, %u dispatches/sample, "
               "%u samples/arm, GPU timestamps\n",
               config.dispatches, config.samples);
        for (size_t i = 0;
             i < sizeof(k_token_cases) / sizeof(k_token_cases[0]); i++) {
            if (!run_oracle_case(&f, k_token_cases[i])) return 1;
            if (!run_benchmark_case(&f, &config, k_token_cases[i])) return 1;
        }
        fprintf(stderr,
                "Metal Q4_K MM tail-cull oracle: PASS "
                "(N=9,16,17,31,33,47,63,65; bit-exact; "
                "distinct payload poison; canaries intact)\n");
        printf("  correctness: bit-exact outputs; input/output canaries intact\n");
        printf("  scope: no GGUF, mmap, model runtime, SSD I/O, uploads, "
               "readback, or CPU wall timing in measured command buffers\n");
        free(f.x_snapshot);
        return 0;
    }
}
