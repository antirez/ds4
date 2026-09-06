#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Resident, production-shape comparison for Q4 attention output-A.  The
 * measured command buffers contain only Metal dispatches; fixture creation,
 * map construction for the routed-only arm, and all validation stay outside
 * the timed region. */
enum {
    K_DIM = 4096,
    M_DIM = 1024,
    GROUPS = 8,
    QK_K = 256,
    Q4_BLOCK_BYTES = 144,
    THREADS_PER_GROUP = 128,
    THREADGROUP_MEMORY_BYTES = 8192,
    GUARD_BYTES = 4096,
    MIN_TOKENS = 512,
    MAX_TOKENS = 4096,
    MAX_TOKEN_CASES = 16,
    MAX_SAMPLES = 64,
    MAX_WARMUP = 32,
    DEFAULT_SAMPLES = 8,
    DEFAULT_WARMUP = 2,
};

static const uint32_t k_guard = 0x7fc12345u;
static const uint32_t k_poison_current = 0x7fc0a001u;
static const uint32_t k_poison_routed = 0x7fc0b001u;
static const uint32_t k_poison_direct = 0x7fc0c001u;
static const uint32_t k_default_tokens[] = {512u, 1024u, 2048u, 4096u};

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2];
} block_q4_K_host;

typedef struct {
    int32_t ne02;
    int32_t ne10;
    int32_t ne11;
    uint64_t nb11;
    uint64_t nb12;
    int32_t ne21;
    int32_t ne20;
    uint64_t nb21;
} map_args;

typedef struct {
    int32_t ne00;
    int32_t ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne11;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne20;
    int32_t ne21;
    int32_t ne0;
    int32_t ne1;
    int16_t r2;
    int16_t r3;
    int32_t tp_rank;
    int32_t tp_world;
    int32_t tp_expert_base;
} mm_args;

_Static_assert(sizeof(block_q4_K_host) == Q4_BLOCK_BYTES,
               "Q4_K host fixture must match the Metal ABI");
_Static_assert(sizeof(map_args) == 48, "map argument ABI changed");
_Static_assert(sizeof(mm_args) == 104, "routed-MM argument ABI changed");

typedef struct {
    NSUInteger tpe_bytes;
    NSUInteger hids_bytes;
    NSUInteger work_offset;
    NSUInteger total_bytes;
    NSUInteger work_cap;
} map_layout;

typedef struct {
    uint32_t tokens[MAX_TOKEN_CASES];
    uint32_t token_count;
    uint32_t samples;
    uint32_t warmup;
} bench_config;

typedef enum {
    ARM_CURRENT,
    ARM_ROUTED_ONLY,
    ARM_DIRECT,
} bench_arm;

typedef struct {
    __strong id<MTLDevice> device;
    __strong id<MTLCommandQueue> queue;
    __strong id<MTLComputePipelineState> map_pipeline;
    __strong id<MTLComputePipelineState> routed_pipeline;
    __strong id<MTLComputePipelineState> direct_pipeline;
    __strong id<MTLBuffer> weights;
    NSUInteger weights_bytes;
    uint64_t weights_hash;
} fixture;

typedef struct {
    __strong id<MTLBuffer> heads;
    __strong id<MTLBuffer> current_out;
    __strong id<MTLBuffer> routed_out;
    __strong id<MTLBuffer> direct_out;
    __strong id<MTLBuffer> ids;
    __strong id<MTLBuffer> current_map;
    __strong id<MTLBuffer> prebuilt_map;
    NSUInteger heads_bytes;
    NSUInteger output_bytes;
    NSUInteger ids_bytes;
    map_layout layout;
    map_args map;
    mm_args mm;
    uint64_t heads_hash;
    uint64_t ids_hash;
    uint64_t prebuilt_map_hash;
    uint32_t tokens;
} case_buffers;

typedef struct {
    double first[MAX_SAMPLES];
    double second[MAX_SAMPLES];
} pair_samples;

static void usage(FILE *fp, const char *argv0) {
    fprintf(fp,
            "usage: %s [options]\n"
            "\n"
            "Resident Metal Q4_K attention output-A kernel comparison.\n"
            "\n"
            "  --n LIST       comma-separated token counts, %u..%u\n"
            "                 (default: 512,1024,2048,4096)\n"
            "  --samples N    samples per arm, multiple of 4, max %u\n"
            "                 (default: %u)\n"
            "  --warmup N     warmup dispatches per arm, max %u\n"
            "                 (default: %u)\n"
            "  -h, --help     show this help\n"
            "\n"
            "The fixed production geometry is 4096 -> 1024 across 8 groups.\n"
            "Set DS4_SOURCE_ROOT when running outside the repository root.\n",
            argv0, MIN_TOKENS, MAX_TOKENS, MAX_SAMPLES, DEFAULT_SAMPLES,
            MAX_WARMUP, DEFAULT_WARMUP);
}

static const char *need_arg(int *index, int argc, char **argv) {
    if (*index + 1 >= argc) {
        fprintf(stderr,
                "metal-q4-attn-out-a-direct-bench: %s needs a value\n",
                argv[*index]);
        exit(2);
    }
    return argv[++*index];
}

static uint32_t parse_u32(const char *text, const char *option,
                          uint32_t minimum, uint32_t maximum) {
    char *end = NULL;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || !text[0] || !end || *end ||
        value < minimum || value > maximum) {
        fprintf(stderr,
                "metal-q4-attn-out-a-direct-bench: invalid %s: %s\n",
                option, text);
        exit(2);
    }
    return (uint32_t)value;
}

static void parse_token_list(bench_config *config, const char *text) {
    const char *cursor = text;
    config->token_count = 0u;
    while (*cursor) {
        char *end = NULL;
        errno = 0;
        const unsigned long long value = strtoull(cursor, &end, 10);
        if (errno != 0 || end == cursor || value < MIN_TOKENS ||
            value > MAX_TOKENS || (*end != ',' && *end != '\0') ||
            config->token_count == MAX_TOKEN_CASES) {
            fprintf(stderr,
                    "metal-q4-attn-out-a-direct-bench: invalid --n list: %s\n",
                    text);
            exit(2);
        }
        for (uint32_t i = 0; i < config->token_count; i++) {
            if (config->tokens[i] == (uint32_t)value) {
                fprintf(stderr,
                        "metal-q4-attn-out-a-direct-bench: duplicate N=%llu\n",
                        value);
                exit(2);
            }
        }
        config->tokens[config->token_count++] = (uint32_t)value;
        if (*end == '\0') break;
        cursor = end + 1;
        if (!*cursor) {
            fprintf(stderr,
                    "metal-q4-attn-out-a-direct-bench: invalid --n list: %s\n",
                    text);
            exit(2);
        }
    }
    if (config->token_count == 0u) {
        fprintf(stderr,
                "metal-q4-attn-out-a-direct-bench: --n list is empty\n");
        exit(2);
    }
}

static bench_config parse_options(int argc, char **argv) {
    bench_config config = {
        .samples = DEFAULT_SAMPLES,
        .warmup = DEFAULT_WARMUP,
    };
    config.token_count = (uint32_t)(sizeof(k_default_tokens) /
                                     sizeof(k_default_tokens[0]));
    memcpy(config.tokens, k_default_tokens, sizeof(k_default_tokens));

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(stdout, argv[0]);
            exit(0);
        } else if (!strcmp(argv[i], "--n")) {
            parse_token_list(&config, need_arg(&i, argc, argv));
        } else if (!strcmp(argv[i], "--samples")) {
            const char *option = argv[i];
            config.samples = parse_u32(need_arg(&i, argc, argv), option,
                                       4u, MAX_SAMPLES);
        } else if (!strcmp(argv[i], "--warmup")) {
            const char *option = argv[i];
            config.warmup = parse_u32(need_arg(&i, argc, argv), option,
                                      0u, MAX_WARMUP);
        } else {
            fprintf(stderr,
                    "metal-q4-attn-out-a-direct-bench: unknown option: %s\n",
                    argv[i]);
            usage(stderr, argv[0]);
            exit(2);
        }
    }
    if ((config.samples % 4u) != 0u) {
        fprintf(stderr,
                "metal-q4-attn-out-a-direct-bench: --samples must be "
                "divisible by 4 for balanced ABBA/BAAB blocks\n");
        exit(2);
    }
    return config;
}

static bool checked_add(NSUInteger a, NSUInteger b, NSUInteger *out) {
    if (b > NSUIntegerMax - a) return false;
    *out = a + b;
    return true;
}

static bool checked_mul(NSUInteger a, NSUInteger b, NSUInteger *out) {
    if (a != 0u && b > NSUIntegerMax / a) return false;
    *out = a * b;
    return true;
}

static NSUInteger align_up(NSUInteger value, NSUInteger alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static uint32_t lcg_next(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static uint64_t hash_bytes(const void *raw, NSUInteger bytes) {
    const uint8_t *data = raw;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (NSUInteger i = 0; i < bytes; i++) {
        hash ^= data[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static bool is_pre_m5_apple_silicon_name(const char *name) {
    return name && !strncmp(name, "Apple M", 7) &&
           name[7] >= '1' && name[7] <= '4' &&
           (name[8] == '\0' || name[8] == ' ');
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
        : [[NSFileManager defaultManager] currentDirectoryPath];
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
                    "metal-q4-attn-out-a-direct-bench: cannot read %s: %s\n",
                    [path fileSystemRepresentation],
                    [[error localizedDescription] UTF8String]);
            return nil;
        }
        [source appendFormat:@"\n// appended %@\n%@\n", relative, part];
    }
    return source;
}

static id<MTLComputePipelineState> make_pipeline(
        id<MTLDevice> device, id<MTLLibrary> library, NSString *name,
        bool routed_constants) {
    NSError *error = nil;
    id<MTLFunction> function = nil;
    if (routed_constants) {
        bool bc_inp = false;
        MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
        [values setConstantValue:&bc_inp
                            type:MTLDataTypeBool
                         atIndex:700];
        function = [library newFunctionWithName:name
                                  constantValues:values
                                           error:&error];
    } else {
        function = [library newFunctionWithName:name];
    }
    if (!function) {
        fprintf(stderr,
                "metal-q4-attn-out-a-direct-bench: function %s: %s\n",
                [name UTF8String],
                error ? [[error localizedDescription] UTF8String] : "missing");
        return nil;
    }
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        fprintf(stderr,
                "metal-q4-attn-out-a-direct-bench: pipeline %s: %s\n",
                [name UTF8String], [[error localizedDescription] UTF8String]);
    }
    return pipeline;
}

static id<MTLBuffer> guarded_buffer(id<MTLDevice> device,
                                    NSUInteger payload_bytes,
                                    NSString *label) {
    NSUInteger total = 0u;
    if (!checked_add(payload_bytes, 2u * GUARD_BYTES, &total)) return nil;
    id<MTLBuffer> buffer =
        [device newBufferWithLength:total options:MTLResourceStorageModeShared];
    if (!buffer) {
        fprintf(stderr,
                "metal-q4-attn-out-a-direct-bench: allocation failed for "
                "%s (%llu bytes)\n",
                [label UTF8String], (unsigned long long)total);
        return nil;
    }
    buffer.label = label;
    uint32_t *words = buffer.contents;
    for (NSUInteger i = 0; i < total / sizeof(*words); i++) {
        words[i] = k_guard;
    }
    return buffer;
}

static void *payload(id<MTLBuffer> buffer) {
    return (uint8_t *)buffer.contents + GUARD_BYTES;
}

static bool check_canary(id<MTLBuffer> buffer, NSUInteger payload_bytes,
                         const char *label, uint32_t tokens) {
    const uint32_t *words = buffer.contents;
    const NSUInteger suffix = (GUARD_BYTES + payload_bytes) / sizeof(*words);
    for (NSUInteger i = 0; i < GUARD_BYTES / sizeof(*words); i++) {
        if (words[i] != k_guard || words[suffix + i] != k_guard) {
            fprintf(stderr,
                    "metal-q4-attn-out-a-direct-bench: %s canary changed "
                    "N=%u word=%llu\n",
                    label, tokens, (unsigned long long)i);
            return false;
        }
    }
    return true;
}

static void fill_weights(fixture *f) {
    block_q4_K_host *blocks = payload(f->weights);
    const NSUInteger count = f->weights_bytes / sizeof(*blocks);
    uint32_t state = 0x31415926u;
    for (NSUInteger block = 0; block < count; block++) {
        blocks[block].d =
            (uint16_t)(0x2800u | (lcg_next(&state) & 0x01ffu));
        blocks[block].dmin =
            (uint16_t)(0x2000u | (lcg_next(&state) & 0x01ffu));
        for (size_t i = 0; i < sizeof(blocks[block].scales); i++) {
            blocks[block].scales[i] = (uint8_t)(lcg_next(&state) >> 24);
        }
        for (size_t i = 0; i < sizeof(blocks[block].qs); i++) {
            blocks[block].qs[i] = (uint8_t)(lcg_next(&state) >> 24);
        }
    }
}

static bool init_fixture(fixture *f) {
    f->device = MTLCreateSystemDefaultDevice();
    if (!f->device) {
        fprintf(stderr,
                "metal-q4-attn-out-a-direct-bench: no Metal device\n");
        return false;
    }
    if (!is_pre_m5_apple_silicon_name([f->device.name UTF8String])) {
        fprintf(stderr,
                "metal-q4-attn-out-a-direct-bench: production route requires "
                "Apple M1-M4 (device: %s)\n",
                [f->device.name UTF8String]);
        return false;
    }
    f->queue = [f->device newCommandQueue];
    NSString *source = load_metal_source();
    if (!f->queue || !source) return false;

    NSError *error = nil;
    MTLCompileOptions *options = [MTLCompileOptions new];
    id<MTLLibrary> library =
        [f->device newLibraryWithSource:source options:options error:&error];
    if (!library) {
        fprintf(stderr,
                "metal-q4-attn-out-a-direct-bench: Metal compile failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return false;
    }
    f->map_pipeline = make_pipeline(
        f->device, library, @"kernel_mul_mm_id_map0_ne20_8", false);
    f->routed_pipeline = make_pipeline(
        f->device, library, @"kernel_mul_mm_id_q4_K_f32", true);
    f->direct_pipeline = make_pipeline(
        f->device, library,
        @"kernel_attn_out_low_q4_K_legacy_direct", false);
    if (!f->map_pipeline || !f->routed_pipeline || !f->direct_pipeline ||
        f->map_pipeline.maxTotalThreadsPerThreadgroup < GROUPS ||
        f->routed_pipeline.threadExecutionWidth != 32u ||
        f->direct_pipeline.threadExecutionWidth != 32u ||
        f->routed_pipeline.maxTotalThreadsPerThreadgroup < THREADS_PER_GROUP ||
        f->direct_pipeline.maxTotalThreadsPerThreadgroup < THREADS_PER_GROUP ||
        f->routed_pipeline.staticThreadgroupMemoryLength +
                THREADGROUP_MEMORY_BYTES > f->device.maxThreadgroupMemoryLength ||
        f->direct_pipeline.staticThreadgroupMemoryLength +
                THREADGROUP_MEMORY_BYTES > f->device.maxThreadgroupMemoryLength) {
        fprintf(stderr,
                "metal-q4-attn-out-a-direct-bench: unexpected pipeline "
                "geometry or threadgroup memory limit\n");
        return false;
    }

    const NSUInteger row_bytes =
        (NSUInteger)(K_DIM / QK_K) * Q4_BLOCK_BYTES;
    NSUInteger rows = 0u;
    if (!checked_mul((NSUInteger)GROUPS, M_DIM, &rows) ||
        !checked_mul(rows, row_bytes, &f->weights_bytes)) {
        fprintf(stderr,
                "metal-q4-attn-out-a-direct-bench: weight size overflow\n");
        return false;
    }
    f->weights = guarded_buffer(f->device, f->weights_bytes,
                                @"Q4 output-A weights");
    if (!f->weights) return false;
    fill_weights(f);
    f->weights_hash = hash_bytes(payload(f->weights), f->weights_bytes);
    return true;
}

static map_layout make_map_layout(uint32_t tokens) {
    const NSUInteger pair_rows = (NSUInteger)tokens * GROUPS;
    const NSUInteger tpe_bytes = GROUPS * 2u * sizeof(uint32_t);
    const NSUInteger hids_bytes = pair_rows * sizeof(int32_t);
    const NSUInteger work_offset = align_up(tpe_bytes + hids_bytes, 8u);
    const NSUInteger work_cap =
        (pair_rows + 31u * GROUPS + 31u) / 32u;
    return (map_layout) {
        .tpe_bytes = tpe_bytes,
        .hids_bytes = hids_bytes,
        .work_offset = work_offset,
        .total_bytes = work_offset + 8u + work_cap * 2u * sizeof(uint32_t),
        .work_cap = work_cap,
    };
}

static void fill_heads(float *values, NSUInteger count) {
    for (NSUInteger i = 0; i < count; i++) {
        const uint32_t bits =
            (uint32_t)(i * 1103515245u + 12345u + (i >> 7u));
        values[i] = ((float)((int32_t)((bits >> 16u) & 0x7ffu) - 1024)) /
                    1024.0f;
    }
}

static void fill_ids(int32_t *ids, uint32_t tokens) {
    for (uint32_t token = 0; token < tokens; token++) {
        for (uint32_t group = 0; group < GROUPS; group++) {
            ids[(NSUInteger)token * GROUPS + group] = (int32_t)group;
        }
    }
}

static bool init_case(case_buffers *c, fixture *f, uint32_t tokens) {
    c->tokens = tokens;
    c->layout = make_map_layout(tokens);
    NSUInteger head_values = 0u;
    NSUInteger output_values = 0u;
    if (!checked_mul((NSUInteger)tokens, GROUPS * K_DIM, &head_values) ||
        !checked_mul(head_values, sizeof(float), &c->heads_bytes) ||
        !checked_mul((NSUInteger)tokens, GROUPS * M_DIM, &output_values) ||
        !checked_mul(output_values, sizeof(float), &c->output_bytes) ||
        !checked_mul((NSUInteger)tokens, GROUPS * sizeof(int32_t),
                     &c->ids_bytes)) {
        fprintf(stderr,
                "metal-q4-attn-out-a-direct-bench: N=%u size overflow\n",
                tokens);
        return false;
    }

    c->heads = guarded_buffer(f->device, c->heads_bytes, @"attention heads");
    c->current_out = guarded_buffer(f->device, c->output_bytes,
                                    @"map+routed output");
    c->routed_out = guarded_buffer(f->device, c->output_bytes,
                                   @"routed-only output");
    c->direct_out = guarded_buffer(f->device, c->output_bytes,
                                   @"direct output");
    c->ids = guarded_buffer(f->device, c->ids_bytes, @"fixed route ids");
    c->current_map = guarded_buffer(f->device, c->layout.total_bytes,
                                    @"current route map");
    c->prebuilt_map = guarded_buffer(f->device, c->layout.total_bytes,
                                     @"prebuilt route map");
    if (!c->heads || !c->current_out || !c->routed_out || !c->direct_out ||
        !c->ids || !c->current_map || !c->prebuilt_map) {
        return false;
    }

    fill_heads(payload(c->heads), head_values);
    fill_ids(payload(c->ids), tokens);
    c->heads_hash = hash_bytes(payload(c->heads), c->heads_bytes);
    c->ids_hash = hash_bytes(payload(c->ids), c->ids_bytes);

    const uint64_t row_bytes =
        (uint64_t)(K_DIM / QK_K) * Q4_BLOCK_BYTES;
    c->map = (map_args) {
        .ne02 = GROUPS,
        .ne10 = K_DIM,
        .ne11 = GROUPS,
        .nb11 = (uint64_t)K_DIM * sizeof(float),
        .nb12 = (uint64_t)GROUPS * K_DIM * sizeof(float),
        .ne21 = (int32_t)tokens,
        .ne20 = GROUPS,
        .nb21 = (uint64_t)GROUPS * sizeof(int32_t),
    };
    c->mm = (mm_args) {
        .ne00 = K_DIM,
        .ne02 = GROUPS,
        .nb01 = row_bytes,
        .nb02 = (uint64_t)M_DIM * row_bytes,
        .nb03 = (uint64_t)GROUPS * M_DIM * row_bytes,
        .ne11 = GROUPS,
        .nb10 = sizeof(float),
        .nb11 = (uint64_t)K_DIM * sizeof(float),
        .nb12 = (uint64_t)GROUPS * K_DIM * sizeof(float),
        .nb13 = (uint64_t)tokens * GROUPS * K_DIM * sizeof(float),
        .ne20 = GROUPS,
        .ne21 = (int32_t)tokens,
        .ne0 = M_DIM,
        .ne1 = GROUPS,
        .r2 = 1,
        .r3 = 1,
        .tp_rank = 0,
        .tp_world = 1,
        .tp_expert_base = 0,
    };
    return true;
}

static void encode_map(id<MTLComputeCommandEncoder> encoder,
                       fixture *f, case_buffers *c,
                       id<MTLBuffer> map_buffer) {
    [encoder setComputePipelineState:f->map_pipeline];
    [encoder setBytes:&c->map length:sizeof(c->map) atIndex:0];
    [encoder setBuffer:c->ids offset:GUARD_BYTES atIndex:1];
    [encoder setBuffer:map_buffer offset:GUARD_BYTES atIndex:2];
    [encoder setBuffer:map_buffer
                 offset:GUARD_BYTES + c->layout.tpe_bytes
                atIndex:3];
    [encoder setBuffer:map_buffer
                 offset:GUARD_BYTES + c->layout.work_offset
                atIndex:4];
    const NSUInteger staging = GROUPS * GROUPS * sizeof(uint16_t);
    const NSUInteger scatter = GROUPS * 2u * sizeof(uint32_t);
    [encoder setThreadgroupMemoryLength:
        staging > scatter ? staging : scatter atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
          threadsPerThreadgroup:MTLSizeMake(GROUPS, 1u, 1u)];
}

static void encode_routed(id<MTLComputeCommandEncoder> encoder,
                          fixture *f, case_buffers *c,
                          id<MTLBuffer> map_buffer,
                          id<MTLBuffer> output) {
    [encoder setComputePipelineState:f->routed_pipeline];
    [encoder setBytes:&c->mm length:sizeof(c->mm) atIndex:0];
    [encoder setBuffer:f->weights offset:GUARD_BYTES atIndex:1];
    [encoder setBuffer:c->heads offset:GUARD_BYTES atIndex:2];
    [encoder setBuffer:map_buffer offset:GUARD_BYTES atIndex:3];
    [encoder setBuffer:map_buffer
                 offset:GUARD_BYTES + c->layout.tpe_bytes
                atIndex:4];
    [encoder setBuffer:output offset:GUARD_BYTES atIndex:5];
    [encoder setBuffer:map_buffer
                 offset:GUARD_BYTES + c->layout.work_offset
                atIndex:6];
    [encoder setThreadgroupMemoryLength:THREADGROUP_MEMORY_BYTES atIndex:0];
    [encoder dispatchThreadgroups:
        MTLSizeMake(c->layout.work_cap, (M_DIM + 63u) / 64u, 1u)
          threadsPerThreadgroup:MTLSizeMake(THREADS_PER_GROUP, 1u, 1u)];
}

static void encode_direct(id<MTLComputeCommandEncoder> encoder,
                          fixture *f, case_buffers *c) {
    [encoder setComputePipelineState:f->direct_pipeline];
    [encoder setBytes:&c->mm length:sizeof(c->mm) atIndex:0];
    [encoder setBuffer:f->weights offset:GUARD_BYTES atIndex:1];
    [encoder setBuffer:c->heads offset:GUARD_BYTES atIndex:2];
    [encoder setBuffer:c->direct_out offset:GUARD_BYTES atIndex:3];
    [encoder setThreadgroupMemoryLength:THREADGROUP_MEMORY_BYTES atIndex:0];
    [encoder dispatchThreadgroups:
        MTLSizeMake((c->tokens + 31u) / 32u,
                    (M_DIM + 63u) / 64u, GROUPS)
          threadsPerThreadgroup:MTLSizeMake(THREADS_PER_GROUP, 1u, 1u)];
}

static bool finish_command_buffer(id<MTLCommandBuffer> cb,
                                  const char *label) {
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusCompleted) return true;
    fprintf(stderr,
            "metal-q4-attn-out-a-direct-bench: %s failed: %s\n",
            label, cb.error ? [[cb.error localizedDescription] UTF8String]
                            : "unknown Metal error");
    return false;
}

static bool build_prebuilt_map(fixture *f, case_buffers *c) {
    id<MTLCommandBuffer> cb = [f->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
    encode_map(encoder, f, c, c->prebuilt_map);
    [encoder endEncoding];
    if (!finish_command_buffer(cb, "prebuilt map")) return false;
    c->prebuilt_map_hash =
        hash_bytes(payload(c->prebuilt_map), c->layout.total_bytes);
    return true;
}

static bool run_arm(fixture *f, case_buffers *c, bench_arm arm,
                    double *gpu_seconds) {
    id<MTLCommandBuffer> cb = [f->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
    switch (arm) {
    case ARM_CURRENT:
        encode_map(encoder, f, c, c->current_map);
        encode_routed(encoder, f, c, c->current_map, c->current_out);
        break;
    case ARM_ROUTED_ONLY:
        encode_routed(encoder, f, c, c->prebuilt_map, c->routed_out);
        break;
    case ARM_DIRECT:
        encode_direct(encoder, f, c);
        break;
    }
    [encoder endEncoding];
    const char *label = arm == ARM_CURRENT ? "map+routed" :
                        arm == ARM_ROUTED_ONLY ? "routed-only" : "direct";
    if (!finish_command_buffer(cb, label)) return false;
    const double elapsed = cb.GPUEndTime - cb.GPUStartTime;
    if (!(elapsed > 0.0) || !isfinite(elapsed)) {
        fprintf(stderr,
                "metal-q4-attn-out-a-direct-bench: GPU timestamps "
                "unavailable for %s\n", label);
        return false;
    }
    *gpu_seconds = elapsed;
    return true;
}

static void poison_output(id<MTLBuffer> output, NSUInteger bytes,
                          uint32_t poison) {
    uint32_t *words = payload(output);
    for (NSUInteger i = 0; i < bytes / sizeof(*words); i++) {
        words[i] = poison;
    }
}

static bool check_bitwise_outputs(case_buffers *c) {
    const uint32_t *current = payload(c->current_out);
    const uint32_t *routed = payload(c->routed_out);
    const uint32_t *direct = payload(c->direct_out);
    const NSUInteger count = c->output_bytes / sizeof(uint32_t);
    for (NSUInteger i = 0; i < count; i++) {
        if (current[i] != routed[i] || current[i] != direct[i]) {
            fprintf(stderr,
                    "metal-q4-attn-out-a-direct-bench: bitwise mismatch "
                    "N=%u element=%llu current=%08x routed=%08x direct=%08x\n",
                    c->tokens, (unsigned long long)i,
                    current[i], routed[i], direct[i]);
            return false;
        }
    }
    return true;
}

static bool check_integrity(fixture *f, case_buffers *c) {
    bool ok = true;
    if (hash_bytes(payload(f->weights), f->weights_bytes) != f->weights_hash) {
        fprintf(stderr,
                "metal-q4-attn-out-a-direct-bench: weights changed N=%u\n",
                c->tokens);
        ok = false;
    }
    if (hash_bytes(payload(c->heads), c->heads_bytes) != c->heads_hash) {
        fprintf(stderr,
                "metal-q4-attn-out-a-direct-bench: heads changed N=%u\n",
                c->tokens);
        ok = false;
    }
    if (hash_bytes(payload(c->ids), c->ids_bytes) != c->ids_hash) {
        fprintf(stderr,
                "metal-q4-attn-out-a-direct-bench: ids changed N=%u\n",
                c->tokens);
        ok = false;
    }
    if (hash_bytes(payload(c->prebuilt_map), c->layout.total_bytes) !=
        c->prebuilt_map_hash) {
        fprintf(stderr,
                "metal-q4-attn-out-a-direct-bench: routed-only map changed "
                "N=%u\n", c->tokens);
        ok = false;
    }
    if (hash_bytes(payload(c->current_map), c->layout.total_bytes) !=
        c->prebuilt_map_hash) {
        fprintf(stderr,
                "metal-q4-attn-out-a-direct-bench: rebuilt map differs "
                "N=%u\n", c->tokens);
        ok = false;
    }
    ok = check_canary(f->weights, f->weights_bytes,
                      "weights", c->tokens) && ok;
    ok = check_canary(c->heads, c->heads_bytes,
                      "heads", c->tokens) && ok;
    ok = check_canary(c->current_out, c->output_bytes,
                      "map+routed output", c->tokens) && ok;
    ok = check_canary(c->routed_out, c->output_bytes,
                      "routed-only output", c->tokens) && ok;
    ok = check_canary(c->direct_out, c->output_bytes,
                      "direct output", c->tokens) && ok;
    ok = check_canary(c->ids, c->ids_bytes,
                      "ids", c->tokens) && ok;
    ok = check_canary(c->current_map, c->layout.total_bytes,
                      "current map", c->tokens) && ok;
    ok = check_canary(c->prebuilt_map, c->layout.total_bytes,
                      "prebuilt map", c->tokens) && ok;
    return ok;
}

static bool run_oracle(fixture *f, case_buffers *c) {
    poison_output(c->current_out, c->output_bytes, k_poison_current);
    poison_output(c->routed_out, c->output_bytes, k_poison_routed);
    poison_output(c->direct_out, c->output_bytes, k_poison_direct);
    double ignored = 0.0;
    if (!run_arm(f, c, ARM_CURRENT, &ignored) ||
        !run_arm(f, c, ARM_ROUTED_ONLY, &ignored) ||
        !run_arm(f, c, ARM_DIRECT, &ignored)) {
        return false;
    }
    return check_bitwise_outputs(c) && check_integrity(f, c);
}

static bool warm_up(fixture *f, case_buffers *c, uint32_t warmup) {
    static const bench_arm orders[][3] = {
        {ARM_CURRENT, ARM_ROUTED_ONLY, ARM_DIRECT},
        {ARM_DIRECT, ARM_ROUTED_ONLY, ARM_CURRENT},
        {ARM_ROUTED_ONLY, ARM_CURRENT, ARM_DIRECT},
        {ARM_DIRECT, ARM_CURRENT, ARM_ROUTED_ONLY},
        {ARM_CURRENT, ARM_DIRECT, ARM_ROUTED_ONLY},
        {ARM_ROUTED_ONLY, ARM_DIRECT, ARM_CURRENT},
    };
    double ignored = 0.0;
    for (uint32_t cycle = 0; cycle < warmup; cycle++) {
        const bench_arm *order = orders[cycle %
            (sizeof(orders) / sizeof(orders[0]))];
        for (uint32_t position = 0; position < 3u; position++) {
            if (!run_arm(f, c, order[position], &ignored)) return false;
        }
    }
    return true;
}

static bool run_balanced_pair(fixture *f, case_buffers *c,
                              bench_arm first, bench_arm second,
                              uint32_t samples, pair_samples *result) {
    uint32_t first_count = 0u;
    uint32_t second_count = 0u;
    for (uint32_t cycle = 0; cycle < samples / 2u; cycle++) {
        const bench_arm abba[] = {first, second, second, first};
        const bench_arm baab[] = {second, first, first, second};
        const bench_arm *order = (cycle & 1u) ? baab : abba;
        for (uint32_t position = 0; position < 4u; position++) {
            double elapsed = 0.0;
            if (!run_arm(f, c, order[position], &elapsed)) return false;
            if (order[position] == first) {
                result->first[first_count++] = elapsed;
            } else {
                result->second[second_count++] = elapsed;
            }
        }
    }
    return first_count == samples && second_count == samples;
}

static int compare_double(const void *lhs, const void *rhs) {
    const double a = *(const double *)lhs;
    const double b = *(const double *)rhs;
    return (a > b) - (a < b);
}

static double median(const double *values, uint32_t count) {
    double sorted[MAX_SAMPLES];
    memcpy(sorted, values, count * sizeof(*sorted));
    qsort(sorted, count, sizeof(*sorted), compare_double);
    if (count & 1u) return sorted[count / 2u];
    return 0.5 * (sorted[count / 2u - 1u] + sorted[count / 2u]);
}

static double paired_geomean_ratio(const double *baseline,
                                    const double *candidate,
                                    uint32_t count) {
    double log_sum = 0.0;
    for (uint32_t i = 0; i < count; i++) {
        log_sum += log(baseline[i] / candidate[i]);
    }
    return exp(log_sum / count);
}

static bool run_benchmark_case(fixture *f, const bench_config *config,
                               uint32_t tokens) {
    @autoreleasepool {
        case_buffers c = {0};
        if (!init_case(&c, f, tokens) || !build_prebuilt_map(f, &c) ||
            !run_oracle(f, &c) || !warm_up(f, &c, config->warmup)) {
            return false;
        }

        pair_samples current_direct = {0};
        pair_samples current_routed = {0};
        pair_samples routed_direct = {0};
        if (!run_balanced_pair(f, &c, ARM_CURRENT, ARM_DIRECT,
                               config->samples, &current_direct) ||
            !run_balanced_pair(f, &c, ARM_CURRENT, ARM_ROUTED_ONLY,
                               config->samples, &current_routed) ||
            !run_balanced_pair(f, &c, ARM_ROUTED_ONLY, ARM_DIRECT,
                               config->samples, &routed_direct) ||
            !check_bitwise_outputs(&c) || !check_integrity(f, &c)) {
            return false;
        }

        const double current_ms =
            median(current_direct.first, config->samples) * 1.0e3;
        const double direct_ms =
            median(current_direct.second, config->samples) * 1.0e3;
        const double routed_ms =
            median(current_routed.second, config->samples) * 1.0e3;
        const double direct_ratio = paired_geomean_ratio(
            current_direct.first, current_direct.second, config->samples);
        const double map_ratio = paired_geomean_ratio(
            current_routed.first, current_routed.second, config->samples);
        double map_delta[MAX_SAMPLES];
        for (uint32_t i = 0; i < config->samples; i++) {
            map_delta[i] = (current_routed.first[i] -
                            current_routed.second[i]) * 1.0e6;
        }
        const double map_delta_us = median(map_delta, config->samples);
        const double routed_to_direct = paired_geomean_ratio(
            routed_direct.first, routed_direct.second, config->samples);

        printf("  N=%-4u current(map+routed) %8.3f ms  "
               "routed-only %8.3f ms  direct %8.3f ms\n",
               tokens, current_ms, routed_ms, direct_ms);
        printf("         paired direct gain %6.3fx (%+6.2f%%); "
               "map+dispatch %+8.3f us (%+6.2f%%); "
               "paired direct vs routed %6.3fx (%+6.2f%%); PASS\n",
               direct_ratio, (direct_ratio - 1.0) * 100.0,
               map_delta_us, (map_ratio - 1.0) * 100.0,
               routed_to_direct, (routed_to_direct - 1.0) * 100.0);
        fflush(stdout);
        return true;
    }
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const bench_config config = parse_options(argc, argv);
        fixture f = {0};
        if (!init_fixture(&f)) return 1;

        printf("Metal Q4_K attention output-A direct kernel benchmark\n");
        printf("  device: %s\n", [f.device.name UTF8String]);
        printf("  shape: 4096 -> 1024 x 8 groups; resident anonymous Q4_K "
               "weights (%.1f MiB)\n",
               (double)f.weights_bytes / (1024.0 * 1024.0));
        printf("  design: current map+routed, routed-only prebuilt map, "
               "and fixed-route direct\n");
        printf("  schedule: balanced ABBA/BAAB pair blocks, "
               "%u samples/arm/pair, "
               "%u warmup dispatches/arm, GPU timestamps only\n",
               config.samples, config.warmup);

        for (uint32_t i = 0; i < config.token_count; i++) {
            if (!run_benchmark_case(&f, &config, config.tokens[i])) return 1;
        }
        printf("  correctness: all three outputs bit-identical; weights, "
               "heads, ids, and prebuilt map hashes unchanged; all "
               "prefix/suffix canaries intact\n");
        printf("  scope: no GGUF, SSD I/O, model upload, GPU readback, or "
               "CPU wall timing in measured command buffers\n");
        return 0;
    }
}
