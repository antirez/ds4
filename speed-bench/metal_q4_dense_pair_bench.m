#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    IN_DIM = 4096,
    OUT0_DIM = 1024,
    OUT1_DIM = 512,
    QK_K = 256,
    Q4_BLOCK_BYTES = 144,
    NSG = 2,
    NR0 = 2,
    THREADS_PER_SIMDGROUP = 32,
    GUARD_BYTES = 256,
    DEFAULT_SETS = 64,
    DEFAULT_DISPATCHES = 256,
    DEFAULT_WARMUP_DISPATCHES = 64,
    DEFAULT_SAMPLES = 16,
};

static const uint32_t k_guard = 0x7fc12345u;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2];
} block_q4_K_host;

typedef struct {
    int32_t ne00;
    int32_t ne01;
    int32_t ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne10;
    int32_t ne11;
    int32_t ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne0;
    int32_t ne1;
    int32_t nr0;
    int16_t r2;
    int16_t r3;
} mul_mv_args;

_Static_assert(sizeof(block_q4_K_host) == Q4_BLOCK_BYTES,
               "Q4_K host fixture must match GGUF/Metal layout");
_Static_assert(sizeof(mul_mv_args) == 112,
               "Metal mul_mv argument ABI changed");

typedef struct {
    uint32_t sets;
    uint32_t dispatches;
    uint32_t warmup_dispatches;
    uint32_t samples;
} bench_config;

typedef enum {
    ARM_SEPARATE,
    ARM_PAIR,
} bench_arm;

typedef struct {
    __strong id<MTLDevice> device;
    __strong id<MTLCommandQueue> queue;
    __strong id<MTLComputePipelineState> standalone_pipeline;
    __strong id<MTLComputePipelineState> pair_pipeline;
    __strong id<MTLBuffer> w0;
    __strong id<MTLBuffer> w1;
    __strong id<MTLBuffer> x;
    __strong id<MTLBuffer> separate0;
    __strong id<MTLBuffer> separate1;
    __strong id<MTLBuffer> pair0;
    __strong id<MTLBuffer> pair1;
    mul_mv_args args0;
    mul_mv_args args1;
    NSUInteger w0_set_bytes;
    NSUInteger w1_set_bytes;
    NSUInteger x_offset;
    NSUInteger out0_stride;
    NSUInteger out1_stride;
    uint32_t sets;
} fixture;

static void usage(FILE *fp, const char *argv0) {
    fprintf(fp,
            "usage: %s [options]\n"
            "\n"
            "Kernel-only A/B for production decode Q4_K projections:\n"
            "  4096 -> 1024 and 4096 -> 512, n_tok=1\n"
            "\n"
            "  --sets N          rotating resident weight sets (default: 64)\n"
            "  --dispatches N     logical projection pairs per sample (default: 256)\n"
            "  --warmup N         logical projection pairs per warmup arm (default: 64)\n"
            "  --samples N        samples per arm; must be even (default: 16)\n"
            "  -h, --help         show this help\n"
            "\n"
            "Set DS4_SOURCE_ROOT when running outside the repository root.\n",
            argv0);
}

static uint32_t parse_u32(const char *text, const char *option,
                          uint32_t minimum) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || text[0] == '\0' || !end || *end != '\0' ||
        value < minimum || value > UINT32_MAX) {
        fprintf(stderr, "metal-q4-dense-pair-bench: invalid %s: %s\n",
                option, text);
        exit(2);
    }
    return (uint32_t)value;
}

static const char *need_arg(int *i, int argc, char **argv) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "metal-q4-dense-pair-bench: %s needs a value\n",
                argv[*i]);
        exit(2);
    }
    return argv[++*i];
}

static bench_config parse_options(int argc, char **argv) {
    bench_config cfg = {
        .sets = DEFAULT_SETS,
        .dispatches = DEFAULT_DISPATCHES,
        .warmup_dispatches = DEFAULT_WARMUP_DISPATCHES,
        .samples = DEFAULT_SAMPLES,
    };
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout, argv[0]);
            exit(0);
        } else if (!strcmp(arg, "--sets")) {
            cfg.sets = parse_u32(need_arg(&i, argc, argv), arg, 1);
        } else if (!strcmp(arg, "--dispatches")) {
            cfg.dispatches =
                parse_u32(need_arg(&i, argc, argv), arg, 1);
        } else if (!strcmp(arg, "--warmup")) {
            cfg.warmup_dispatches =
                parse_u32(need_arg(&i, argc, argv), arg, 0);
        } else if (!strcmp(arg, "--samples")) {
            cfg.samples = parse_u32(need_arg(&i, argc, argv), arg, 2);
        } else {
            fprintf(stderr, "metal-q4-dense-pair-bench: unknown option: %s\n",
                    arg);
            usage(stderr, argv[0]);
            exit(2);
        }
    }
    if ((cfg.samples & 1u) != 0u) {
        fprintf(stderr,
                "metal-q4-dense-pair-bench: --samples must be even for "
                "ABBA/BAAB balance\n");
        exit(2);
    }
    return cfg;
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

/* Keep the same concatenation order as ds4_metal.m. The benchmark specializes
 * and dispatches the checked-in production kernels; it carries no kernel copy. */
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
            fprintf(stderr, "metal-q4-dense-pair-bench: cannot read %s: %s\n",
                    [path fileSystemRepresentation],
                    [[error localizedDescription] UTF8String]);
            return nil;
        }
        [source appendFormat:@"\n// appended %@\n%@\n", relative, part];
    }
    return source;
}

static id<MTLComputePipelineState> make_pipeline(
        id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
    int16_t nsg = NSG;
    int16_t nxpsg = 8;
    MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
    [values setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
    [values setConstantValue:&nxpsg type:MTLDataTypeShort atIndex:601];
    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name
                                            constantValues:values
                                                     error:&error];
    if (!function) {
        fprintf(stderr, "metal-q4-dense-pair-bench: function %s: %s\n",
                [name UTF8String], [[error localizedDescription] UTF8String]);
        return nil;
    }
    error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        fprintf(stderr, "metal-q4-dense-pair-bench: pipeline %s: %s\n",
                [name UTF8String], [[error localizedDescription] UTF8String]);
    }
    return pipeline;
}

static uint32_t lcg_next(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static void fill_q4(void *storage, NSUInteger bytes, uint32_t seed) {
    block_q4_K_host *blocks = storage;
    const NSUInteger count = bytes / sizeof(*blocks);
    uint32_t state = seed;
    for (NSUInteger b = 0; b < count; b++) {
        blocks[b].d = (uint16_t)(0x2400u | (lcg_next(&state) & 0x03ffu));
        blocks[b].dmin =
            (uint16_t)(0x1c00u | (lcg_next(&state) & 0x03ffu));
        for (size_t i = 0; i < sizeof(blocks[b].scales); i++) {
            blocks[b].scales[i] = (uint8_t)(lcg_next(&state) >> 24u);
        }
        for (size_t i = 0; i < sizeof(blocks[b].qs); i++) {
            blocks[b].qs[i] = (uint8_t)(lcg_next(&state) >> 24u);
        }
    }
}

static void fill_activation(float *x) {
    uint32_t state = 0x243f6a88u;
    for (uint32_t i = 0; i < IN_DIM; i++) {
        int32_t centered = (int32_t)(lcg_next(&state) & 0xffffu) - 32768;
        x[i] = (float)centered / 32768.0f;
    }
}

static NSUInteger align_up(NSUInteger value, NSUInteger alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static bool checked_product(NSUInteger a, NSUInteger b, NSUInteger *out) {
    if (a != 0 && b > NSUIntegerMax / a) return false;
    *out = a * b;
    return true;
}

static id<MTLBuffer> alloc_buffer(id<MTLDevice> device, NSUInteger bytes,
                                  NSString *label) {
    if (bytes == 0 || bytes > device.maxBufferLength) {
        fprintf(stderr,
                "metal-q4-dense-pair-bench: %s size %.1f MiB exceeds device limit\n",
                [label UTF8String], (double)bytes / (1024.0 * 1024.0));
        return nil;
    }
    id<MTLBuffer> buffer =
        [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    buffer.label = label;
    if (!buffer) {
        fprintf(stderr, "metal-q4-dense-pair-bench: allocation failed for %s\n",
                [label UTF8String]);
    }
    return buffer;
}

static mul_mv_args make_args(uint32_t out_dim) {
    const uint64_t row_bytes = (IN_DIM / QK_K) * Q4_BLOCK_BYTES;
    return (mul_mv_args){
        .ne00 = IN_DIM,
        .ne01 = (int32_t)out_dim,
        .ne02 = 1,
        .nb00 = 1,
        .nb01 = row_bytes,
        .nb02 = row_bytes * out_dim,
        .nb03 = row_bytes * out_dim,
        .ne10 = IN_DIM,
        .ne11 = 1,
        .ne12 = 1,
        .nb10 = sizeof(float),
        .nb11 = IN_DIM * sizeof(float),
        .nb12 = IN_DIM * sizeof(float),
        .nb13 = IN_DIM * sizeof(float),
        .ne0 = (int32_t)out_dim,
        .ne1 = 1,
        .nr0 = NR0,
        .r2 = 1,
        .r3 = 1,
    };
}

static void fill_guard_buffer(id<MTLBuffer> buffer) {
    uint32_t *words = buffer.contents;
    for (NSUInteger i = 0; i < buffer.length / sizeof(*words); i++) {
        words[i] = k_guard;
    }
}

static bool init_fixture(fixture *f, const bench_config *cfg) {
    f->device = MTLCreateSystemDefaultDevice();
    if (!f->device) {
        fprintf(stderr, "metal-q4-dense-pair-bench: no Metal device\n");
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
        fprintf(stderr, "metal-q4-dense-pair-bench: Metal compile failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return false;
    }
    f->standalone_pipeline = make_pipeline(
        f->device, library, @"kernel_mul_mv_q4_K_dense_f32");
    f->pair_pipeline = make_pipeline(
        f->device, library, @"kernel_mul_mv_q4_K_dense_pair_f32");
    if (!f->standalone_pipeline || !f->pair_pipeline) return false;

    f->sets = cfg->sets;
    f->w0_set_bytes = (IN_DIM / QK_K) * Q4_BLOCK_BYTES * OUT0_DIM;
    f->w1_set_bytes = (IN_DIM / QK_K) * Q4_BLOCK_BYTES * OUT1_DIM;
    f->x_offset = GUARD_BYTES;
    f->out0_stride = align_up(
        GUARD_BYTES + OUT0_DIM * sizeof(float) + GUARD_BYTES, 256u);
    f->out1_stride = align_up(
        GUARD_BYTES + OUT1_DIM * sizeof(float) + GUARD_BYTES, 256u);
    f->args0 = make_args(OUT0_DIM);
    f->args1 = make_args(OUT1_DIM);

    NSUInteger w0_bytes = 0, w1_bytes = 0, out0_bytes = 0, out1_bytes = 0;
    if (!checked_product(f->w0_set_bytes, cfg->sets, &w0_bytes) ||
        !checked_product(f->w1_set_bytes, cfg->sets, &w1_bytes) ||
        !checked_product(f->out0_stride, cfg->sets, &out0_bytes) ||
        !checked_product(f->out1_stride, cfg->sets, &out1_bytes)) {
        fprintf(stderr, "metal-q4-dense-pair-bench: requested sizes overflow\n");
        return false;
    }

    f->w0 = alloc_buffer(f->device, w0_bytes, @"q4-w0-resident");
    f->w1 = alloc_buffer(f->device, w1_bytes, @"q4-w1-resident");
    f->x = alloc_buffer(f->device,
        GUARD_BYTES + IN_DIM * sizeof(float) + GUARD_BYTES,
        @"decode-activation");
    f->separate0 = alloc_buffer(f->device, out0_bytes, @"separate-out0");
    f->separate1 = alloc_buffer(f->device, out1_bytes, @"separate-out1");
    f->pair0 = alloc_buffer(f->device, out0_bytes, @"pair-out0");
    f->pair1 = alloc_buffer(f->device, out1_bytes, @"pair-out1");
    if (!f->w0 || !f->w1 || !f->x || !f->separate0 || !f->separate1 ||
        !f->pair0 || !f->pair1) {
        return false;
    }

    fill_q4(f->w0.contents, f->w0.length, 0x41c64e6du);
    fill_q4(f->w1.contents, f->w1.length, 0x9e3779b9u);
    fill_guard_buffer(f->x);
    fill_activation((float *)((uint8_t *)f->x.contents + f->x_offset));
    fill_guard_buffer(f->separate0);
    fill_guard_buffer(f->separate1);
    fill_guard_buffer(f->pair0);
    fill_guard_buffer(f->pair1);
    return true;
}

static void encode_standalone(fixture *f, id<MTLComputeCommandEncoder> enc,
                              uint32_t set) {
    const NSUInteger out0 = (NSUInteger)set * f->out0_stride + GUARD_BYTES;
    const NSUInteger out1 = (NSUInteger)set * f->out1_stride + GUARD_BYTES;
    [enc setComputePipelineState:f->standalone_pipeline];
    [enc setBytes:&f->args0 length:sizeof(f->args0) atIndex:0];
    [enc setBuffer:f->w0 offset:(NSUInteger)set * f->w0_set_bytes atIndex:1];
    [enc setBuffer:f->x offset:f->x_offset atIndex:2];
    [enc setBuffer:f->separate0 offset:out0 atIndex:3];
    [enc setThreadgroupMemoryLength:32 atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((OUT0_DIM + NSG * NR0 - 1) /
                                          (NSG * NR0), 1, 1)
         threadsPerThreadgroup:MTLSizeMake(THREADS_PER_SIMDGROUP, NSG, 1)];

    [enc setBytes:&f->args1 length:sizeof(f->args1) atIndex:0];
    [enc setBuffer:f->w1 offset:(NSUInteger)set * f->w1_set_bytes atIndex:1];
    [enc setBuffer:f->separate1 offset:out1 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((OUT1_DIM + NSG * NR0 - 1) /
                                          (NSG * NR0), 1, 1)
         threadsPerThreadgroup:MTLSizeMake(THREADS_PER_SIMDGROUP, NSG, 1)];
}

static void encode_pair(fixture *f, id<MTLComputeCommandEncoder> enc,
                        uint32_t set) {
    const NSUInteger out0 = (NSUInteger)set * f->out0_stride + GUARD_BYTES;
    const NSUInteger out1 = (NSUInteger)set * f->out1_stride + GUARD_BYTES;
    [enc setComputePipelineState:f->pair_pipeline];
    [enc setBytes:&f->args0 length:sizeof(f->args0) atIndex:0];
    [enc setBytes:&f->args1 length:sizeof(f->args1) atIndex:1];
    [enc setBuffer:f->w0 offset:(NSUInteger)set * f->w0_set_bytes atIndex:2];
    [enc setBuffer:f->w1 offset:(NSUInteger)set * f->w1_set_bytes atIndex:3];
    [enc setBuffer:f->x offset:f->x_offset atIndex:4];
    [enc setBuffer:f->pair0 offset:out0 atIndex:5];
    [enc setBuffer:f->pair1 offset:out1 atIndex:6];
    [enc setThreadgroupMemoryLength:32 atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((OUT0_DIM + NSG * NR0 - 1) /
                                          (NSG * NR0), 1, 1)
         threadsPerThreadgroup:MTLSizeMake(THREADS_PER_SIMDGROUP, NSG, 1)];
}

static bool finish_command_buffer(id<MTLCommandBuffer> cb, const char *label) {
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "metal-q4-dense-pair-bench: %s failed: %s\n", label,
                [[cb.error localizedDescription] UTF8String]);
        return false;
    }
    return true;
}

static bool check_guard_records(id<MTLBuffer> buffer, NSUInteger stride,
                                NSUInteger output_bytes, uint32_t sets,
                                const char *label) {
    const uint32_t *words = buffer.contents;
    for (uint32_t set = 0; set < sets; set++) {
        const NSUInteger record = (NSUInteger)set * stride;
        const NSUInteger output_begin = record + GUARD_BYTES;
        const NSUInteger output_end = output_begin + output_bytes;
        for (NSUInteger off = record; off < output_begin; off += sizeof(*words)) {
            if (words[off / sizeof(*words)] != k_guard) goto bad;
        }
        for (NSUInteger off = output_end; off < record + stride;
             off += sizeof(*words)) {
            if (words[off / sizeof(*words)] != k_guard) goto bad;
        }
    }
    return true;

bad:
    fprintf(stderr, "metal-q4-dense-pair-bench: %s canary changed\n", label);
    return false;
}

static bool check_all_guards(fixture *f) {
    const uint32_t *x = f->x.contents;
    for (NSUInteger off = 0; off < f->x_offset; off += sizeof(*x)) {
        if (x[off / sizeof(*x)] != k_guard) goto x_bad;
    }
    for (NSUInteger off = f->x_offset + IN_DIM * sizeof(float);
         off < f->x.length; off += sizeof(*x)) {
        if (x[off / sizeof(*x)] != k_guard) goto x_bad;
    }
    return check_guard_records(f->separate0, f->out0_stride,
                               OUT0_DIM * sizeof(float), f->sets,
                               "separate out0") &&
           check_guard_records(f->separate1, f->out1_stride,
                               OUT1_DIM * sizeof(float), f->sets,
                               "separate out1") &&
           check_guard_records(f->pair0, f->out0_stride,
                               OUT0_DIM * sizeof(float), f->sets,
                               "pair out0") &&
           check_guard_records(f->pair1, f->out1_stride,
                               OUT1_DIM * sizeof(float), f->sets,
                               "pair out1");

x_bad:
    fprintf(stderr, "metal-q4-dense-pair-bench: activation canary changed\n");
    return false;
}

static bool check_outputs(fixture *f) {
    for (uint32_t set = 0; set < f->sets; set++) {
        const uint8_t *separate0 = (uint8_t *)f->separate0.contents +
            (NSUInteger)set * f->out0_stride + GUARD_BYTES;
        const uint8_t *pair0 = (uint8_t *)f->pair0.contents +
            (NSUInteger)set * f->out0_stride + GUARD_BYTES;
        const uint8_t *separate1 = (uint8_t *)f->separate1.contents +
            (NSUInteger)set * f->out1_stride + GUARD_BYTES;
        const uint8_t *pair1 = (uint8_t *)f->pair1.contents +
            (NSUInteger)set * f->out1_stride + GUARD_BYTES;
        if (memcmp(separate0, pair0, OUT0_DIM * sizeof(float)) != 0 ||
            memcmp(separate1, pair1, OUT1_DIM * sizeof(float)) != 0) {
            fprintf(stderr,
                    "metal-q4-dense-pair-bench: bitwise mismatch at set %u\n",
                    set);
            return false;
        }
    }
    return true;
}

static bool run_oracle(fixture *f) {
    id<MTLCommandBuffer> cb = [f->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    for (uint32_t set = 0; set < f->sets; set++) {
        encode_standalone(f, enc, set);
    }
    for (uint32_t set = 0; set < f->sets; set++) {
        encode_pair(f, enc, set);
    }
    [enc endEncoding];
    if (!finish_command_buffer(cb, "correctness oracle")) return false;
    if (!check_outputs(f) || !check_all_guards(f)) return false;
    fprintf(stderr,
            "Metal Q4_K dense pair oracle: PASS (%u resident sets, bit-exact, canaries intact)\n",
            f->sets);
    return true;
}

static bool run_workload(fixture *f, bench_arm arm, uint32_t dispatches,
                         uint32_t start_set, double *gpu_seconds) {
    id<MTLCommandBuffer> cb = [f->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    for (uint32_t i = 0; i < dispatches; i++) {
        const uint32_t set = (start_set + i) % f->sets;
        if (arm == ARM_SEPARATE) encode_standalone(f, enc, set);
        else encode_pair(f, enc, set);
    }
    [enc endEncoding];
    if (!finish_command_buffer(cb, arm == ARM_SEPARATE ?
                               "separate workload" : "pair workload")) {
        return false;
    }
    const double elapsed = cb.GPUEndTime - cb.GPUStartTime;
    if (!(elapsed > 0.0)) {
        fprintf(stderr,
                "metal-q4-dense-pair-bench: GPU timestamps unavailable\n");
        return false;
    }
    if (gpu_seconds) *gpu_seconds = elapsed;
    return true;
}

static int compare_double(const void *a, const void *b) {
    const double x = *(const double *)a;
    const double y = *(const double *)b;
    return (x > y) - (x < y);
}

static double percentile(const double *sorted, uint32_t n, double p) {
    const double position = p * (double)(n - 1u);
    const uint32_t lo = (uint32_t)position;
    const uint32_t hi = lo + 1u < n ? lo + 1u : lo;
    const double fraction = position - (double)lo;
    return sorted[lo] + (sorted[hi] - sorted[lo]) * fraction;
}

static bool run_benchmark(fixture *f, const bench_config *cfg) {
    double ignored = 0.0;
    if (cfg->warmup_dispatches != 0u) {
        static const bench_arm warm_order[] = {
            ARM_SEPARATE, ARM_PAIR, ARM_PAIR, ARM_SEPARATE,
        };
        for (size_t i = 0; i < sizeof(warm_order) / sizeof(warm_order[0]); i++) {
            if (!run_workload(f, warm_order[i], cfg->warmup_dispatches,
                              (uint32_t)(i * cfg->warmup_dispatches) % f->sets,
                              &ignored)) {
                return false;
            }
        }
    }

    double *separate = calloc(cfg->samples, sizeof(*separate));
    double *pair = calloc(cfg->samples, sizeof(*pair));
    if (!separate || !pair) {
        fprintf(stderr, "metal-q4-dense-pair-bench: sample allocation failed\n");
        free(separate);
        free(pair);
        return false;
    }

    uint32_t separate_count = 0;
    uint32_t pair_count = 0;
    bool ok = true;
    for (uint32_t cycle = 0; ok && cycle < cfg->samples / 2u; cycle++) {
        const bench_arm abba[] = {
            ARM_SEPARATE, ARM_PAIR, ARM_PAIR, ARM_SEPARATE,
        };
        const bench_arm baab[] = {
            ARM_PAIR, ARM_SEPARATE, ARM_SEPARATE, ARM_PAIR,
        };
        const bench_arm *order = (cycle & 1u) ? baab : abba;
        for (uint32_t j = 0; ok && j < 4u; j++) {
            const bench_arm arm = order[j];
            uint32_t arm_index = arm == ARM_SEPARATE ? separate_count : pair_count;
            uint32_t start = (uint32_t)((uint64_t)arm_index * cfg->dispatches %
                                        f->sets);
            double elapsed = 0.0;
            ok = run_workload(f, arm, cfg->dispatches, start, &elapsed);
            if (ok && arm == ARM_SEPARATE) separate[separate_count++] = elapsed;
            if (ok && arm == ARM_PAIR) pair[pair_count++] = elapsed;
        }
    }
    if (ok && (separate_count != cfg->samples || pair_count != cfg->samples)) {
        ok = false;
    }
    if (ok) ok = check_outputs(f) && check_all_guards(f);

    if (ok) {
        qsort(separate, cfg->samples, sizeof(*separate), compare_double);
        qsort(pair, cfg->samples, sizeof(*pair), compare_double);
        const double separate_median = percentile(separate, cfg->samples, 0.50) *
            1.0e6 / cfg->dispatches;
        const double separate_p95 = percentile(separate, cfg->samples, 0.95) *
            1.0e6 / cfg->dispatches;
        const double pair_median = percentile(pair, cfg->samples, 0.50) *
            1.0e6 / cfg->dispatches;
        const double pair_p95 = percentile(pair, cfg->samples, 0.95) *
            1.0e6 / cfg->dispatches;
        const double saved = separate_median - pair_median;
        const double working_set =
            (double)(f->w0.length + f->w1.length) / (1024.0 * 1024.0);

        printf("Metal Q4_K decode kernel-only A/B\n");
        printf("  shape: n_tok=1, 4096->1024 + 4096->512\n");
        printf("  resident anonymous weights: %.1f MiB across %u rotating sets\n",
               working_set, f->sets);
        printf("  design: alternating ABBA/BAAB, %u logical calls/sample, "
               "%u samples/arm, GPU timestamps\n",
               cfg->dispatches, cfg->samples);
        printf("  separate (2 dispatches): median %.3f us, p95 %.3f us\n",
               separate_median, separate_p95);
        printf("  pair     (1 dispatch):   median %.3f us, p95 %.3f us\n",
               pair_median, pair_p95);
        printf("  saved: %.3f us/logical call, speedup %.3fx, reduction %.2f%%\n",
               saved, separate_median / pair_median,
               saved * 100.0 / separate_median);
        printf("  correctness: bit-exact outputs; activation/output canaries intact\n");
        printf("  scope: no GGUF, mmap, model runtime, SSD I/O, or CPU wall timing\n");
    }

    free(separate);
    free(pair);
    return ok;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const bench_config cfg = parse_options(argc, argv);
        fixture f = {0};
        if (!init_fixture(&f, &cfg)) return 1;
        if (!run_oracle(&f)) return 1;
        if (!run_benchmark(&f, &cfg)) return 1;
        return 0;
    }
}
