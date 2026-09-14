// GGUF-free regression tests for the recovered automatic Q8 decode paths.
// Include the host implementation to check private dispatch policy without
// exporting test-only APIs from the production backend.
#include "../ds4_metal.m"
bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
int ds4_deepseek4_attention_bounds(const int *a, uint32_t b, uint32_t c, uint32_t d,
    uint32_t e, uint32_t f, uint32_t *g) {
    (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; return 0;
}
enum { GUARD = 256 };
static const uint32_t poison = 0x7fc12345u;
typedef ds4_gpu_q8_0_matvec_args mv_args;
typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
_Static_assert(sizeof(mv_args) == 112 && sizeof(q8_block) == 34, "Q8 ABI");
static BOOL benchmark_enabled;

static void require(int ok, const char *message) {
    if (!ok) { fprintf(stderr, "SSD kernel test: %s\n", message); exit(1); }
}
static uint32_t random_u32(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}
static void *data(id<MTLBuffer> b) { return (char *)b.contents + GUARD; }
static id<MTLBuffer> buffer(id<MTLDevice> dev, NSUInteger bytes) {
    id<MTLBuffer> b = [dev newBufferWithLength:GUARD * 2 + bytes options:MTLResourceStorageModeShared];
    require(b != nil, "buffer allocation failed");
    memset(b.contents, 0xa5, b.length);
    return b;
}
static id<MTLBuffer> output(id<MTLDevice> dev, NSUInteger count) {
    id<MTLBuffer> b = buffer(dev, count * sizeof(float));
    uint32_t *p = data(b);
    for (NSUInteger i = 0; i < count; ++i) p[i] = poison;
    return b;
}
static uint64_t hash(id<MTLBuffer> b) {
    const uint8_t *p = b.contents;
    uint64_t h = 14695981039346656037ull;
    for (NSUInteger i = 0; i < b.length; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}
static void equal_output(id<MTLBuffer> a, id<MTLBuffer> b, NSUInteger rows,
                         NSUInteger width, NSUInteger stride) {
    require(a.length == b.length, "output lengths differ");
    const uint8_t *ap = a.contents, *bp = b.contents;
    for (NSUInteger i = 0; i < GUARD; ++i)
        require(ap[i] == 0xa5 && bp[i] == 0xa5 &&
                ap[a.length - GUARD + i] == 0xa5 && bp[b.length - GUARD + i] == 0xa5,
                "output guard overwritten");
    const uint32_t *av = data(a), *bv = data(b);
    for (NSUInteger r = 0; r < rows; ++r) {
        for (NSUInteger c = 0; c < stride; ++c) {
            NSUInteger i = r * stride + c;
            if (c < width) {
                // Integer checks remain valid when host is built with fast-math.
                require((av[i] & 0x7f800000u) != 0x7f800000u &&
                        (bv[i] & 0x7f800000u) != 0x7f800000u, "nonfinite/unwritten output");
            } else require(av[i] == poison && bv[i] == poison, "stride padding overwritten");
            if (av[i] != bv[i]) {
                fprintf(stderr, "row=%lu col=%lu old=%08x new=%08x\n",
                        (unsigned long)r, (unsigned long)c, av[i], bv[i]);
                require(0, "output is not bitwise equal");
            }
        }
    }
}
static void finish(id<MTLCommandBuffer> cb) {
    [cb commit]; [cb waitUntilCompleted];
    if (cb.error) fprintf(stderr, "%s\n", cb.error.description.UTF8String);
    require(cb.status == MTLCommandBufferStatusCompleted, "GPU execution failed");
}
static void fill_x(id<MTLBuffer> x, NSUInteger count, uint32_t *seed) {
    float *p = data(x);
    for (NSUInteger i = 0; i < count; ++i)
        p[i] = i % 11 == 0 ? 0.f : ((int)(random_u32(seed) % 8193) - 4096) / 4096.f;
}

static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

// Paired alternating order, two warmups, median GPU time over 12 rounds.
static void bench_pair(id<MTLCommandQueue> queue, const char *name,
                       void (^encode)(id<MTLComputeCommandEncoder>, int)) {
    enum { SAMPLES = 12, REPEATS = 64 };
    double times[2][SAMPLES];
    for (int round = -2; round < SAMPLES; round++) for (int pos = 0; pos < 2; pos++) {
        const int arm = (round & 1) ? 1-pos : pos;
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (int i = 0; i < REPEATS; i++) encode(enc, arm);
        [enc endEncoding]; finish(cb);
        require(cb.GPUEndTime > cb.GPUStartTime, "GPU timestamps unavailable");
        if (round >= 0) times[arm][round] = (cb.GPUEndTime-cb.GPUStartTime)*1.e6/REPEATS;
    }
    double median[2];
    for (int arm = 0; arm < 2; arm++) {
        qsort(times[arm], SAMPLES, sizeof(double), compare_double);
        median[arm] = .5*(times[arm][SAMPLES/2-1]+times[arm][SAMPLES/2]);
    }
    printf("%s: legacy %.3f -> candidate %.3f us (%+.2f%% GPU time).\n",
           name, median[0], median[1], 100.*(median[1]/median[0]-1.));
}
static void test_q8(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                    id<MTLComputePipelineState> oldp, id<MTLComputePipelineState> newp,
                    int nsg, BOOL store, int k, int rows, float clamp) {
    uint32_t seed = 41u + k + rows;
    NSUInteger wb = (NSUInteger)(k / 32) * 34 * rows;
    id<MTLBuffer> gate = buffer(dev, wb), up = buffer(dev, wb), x = buffer(dev, k*4);
    for (int arm = 0; arm < 2; ++arm) {
        q8_block *w = data(arm ? up : gate);
        for (NSUInteger i = 0; i < wb/34; ++i) {
            w[i].d = 0x1800 + (random_u32(&seed) & 0x3ff);
            for (int c = 0; c < 32; ++c) w[i].qs[c] = (int)(random_u32(&seed) >> 24) - 128;
        }
    }
    fill_x(x, k, &seed);
    uint64_t gh = hash(gate), uh = hash(up);
    mv_args args = {.ne00=k, .ne01=rows, .ne02=1, .nb00=34,
        .nb01=(uint64_t)k/32*34, .nb02=wb, .nb03=wb,
        .ne10=k, .ne11=1, .ne12=1, .nb10=4, .nb11=(uint64_t)k*4,
        .nb12=(uint64_t)k*4, .nb13=(uint64_t)k*4, .ne0=rows, .ne1=1,
        .nr0=2, .r2=1, .r3=1};
    id<MTLBuffer> out[2][3];
    for (int arm = 0; arm < 2; ++arm)
        for (int d = 0; d < 3; ++d) out[arm][d] = output(dev, rows);
    id<MTLBuffer> g0 = out[0][0], u0 = out[0][1], m0 = out[0][2];
    id<MTLBuffer> g1 = out[1][0], u1 = out[1][1], m1 = out[1][2];
    void (^encode)(id<MTLComputeCommandEncoder>, int) = ^(id<MTLComputeCommandEncoder> enc, int arm) {
            [enc setComputePipelineState:arm ? newp : oldp];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:gate offset:GUARD atIndex:1];
            [enc setBuffer:up offset:GUARD atIndex:2];
            [enc setBuffer:x offset:GUARD atIndex:3];
            [enc setBuffer:(store ? (arm ? g1 : g0) : (arm ? m1 : m0)) offset:GUARD atIndex:4];
            [enc setBuffer:(store ? (arm ? u1 : u0) : (arm ? m1 : m0)) offset:GUARD atIndex:5];
            [enc setBuffer:(arm ? m1 : m0) offset:GUARD atIndex:6];
            [enc setBytes:&clamp length:sizeof(clamp) atIndex:7];
            [enc setThreadgroupMemoryLength:512 atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(rows/2, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    };
    // Repeated dispatches exercise recycled threadgroup memory. In mid-only
    // mode both unused diagnostic outputs alias mid, just like production.
    for (int repeat = 0; repeat < 5; ++repeat) {
        fill_x(x, k, &seed);
        uint64_t xh = hash(x);
        for (int arm = 0; arm < 2; ++arm)
            for (int d = 0; d < 3; ++d)
                for (int row = 0; row < rows; ++row)
                    ((uint32_t *)data(out[arm][d]))[row] = poison;
        for (int arm = 0; arm < 2; ++arm) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            encode(enc, arm);
            [enc endEncoding]; finish(cb);
        }
        for (int d = store ? 0 : 2; d < 3; ++d)
            equal_output(out[0][d], out[1][d], 1, rows, rows);
        require(hash(x) == xh, "Q8 input mutated");
    }
    require(hash(gate) == gh && hash(up) == uh, "Q8 weight mutated");
    if (benchmark_enabled && k == 4096 && rows == 2048 && clamp > 0) {
        char label[96];
        snprintf(label, sizeof(label), "Q8 shared K=4096 M=2048 NSG=%d store=%d", nsg, store);
        bench_pair(queue, label, encode);
        for (int d = store ? 0 : 2; d < 3; d++) equal_output(out[0][d], out[1][d], 1, rows, rows);
    }
}
typedef struct {
    __strong id<MTLComputePipelineState> p[4]; // old/new single, old/new pair
    __strong id<MTLBuffer> w[2], x, out[4][2];
    mv_args args[2];
    int rows[2], nsg, k;
} projection_fixture;

static void init_projection(projection_fixture *f, id<MTLDevice> dev,
                            id<MTLComputePipelineState> __strong p[4], int nsg,
                            int k, int rows_a, int rows_b, int padding) {
    *f = (projection_fixture){0};
    f->rows[0] = rows_a; f->rows[1] = rows_b; f->nsg = nsg; f->k = k;
    uint32_t seed = 71u + k + rows_a + rows_b + padding;
    for (int arm = 0; arm < 4; ++arm) f->p[arm] = p[arm];
    for (int m = 0; m < 2; ++m) {
        // The legacy single kernel reads two weight rows even on odd tails.
        // Pad only its fixture allocation, not args.ne01 or the output. The
        // paired kernels must independently guard the true row extents.
        NSUInteger stride = (k/32 + padding*(m+1)) * sizeof(q8_block);
        NSUInteger bytes = ((f->rows[m] + 1u) & ~1u) * stride;
        f->w[m] = buffer(dev, bytes);
        q8_block *w = data(f->w[m]);
        for (NSUInteger b = 0; b < bytes / sizeof(*w); ++b) {
            w[b].d = b % 17 ? 0x1800 + (random_u32(&seed) & 0x3ff) : 0;
            for (int i = 0; i < 32; ++i) w[b].qs[i] = (int)(random_u32(&seed) >> 24) - 128;
        }
        f->args[m] = (mv_args){.ne00=k, .ne01=f->rows[m], .ne02=1,
            .nb00=34, .nb01=stride, .nb02=bytes, .nb03=bytes,
            .ne10=k, .ne11=1, .ne12=1, .nb10=4, .nb11=(uint64_t)k*4,
            .nb12=(uint64_t)k*4, .nb13=(uint64_t)k*4,
            .ne0=f->rows[m], .ne1=1, .nr0=2, .r2=1, .r3=1};
        for (int arm = 0; arm < 4; ++arm) f->out[arm][m] = output(dev, f->rows[m]);
    }
    f->x = buffer(dev, k * sizeof(float));
    fill_x(f->x, k, &seed);
}

static void encode_projection(projection_fixture *f, id<MTLComputeCommandEncoder> enc, int arm) {
    [enc setComputePipelineState:f->p[arm]];
    if (arm < 2) {
        [enc setBuffer:f->x offset:GUARD atIndex:2];
        [enc setThreadgroupMemoryLength:256 atIndex:0];
        for (int m = 0; m < 2; ++m) {
            [enc setBytes:&f->args[m] length:sizeof(mv_args) atIndex:0];
            [enc setBuffer:f->w[m] offset:GUARD atIndex:1];
            [enc setBuffer:f->out[arm][m] offset:GUARD atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((f->rows[m]+1)/2, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(32, f->nsg, 1)];
        }
    } else {
        [enc setBytes:&f->args[0] length:sizeof(mv_args) atIndex:0];
        [enc setBytes:&f->args[1] length:sizeof(mv_args) atIndex:1];
        [enc setBuffer:f->w[0] offset:GUARD atIndex:2];
        [enc setBuffer:f->w[1] offset:GUARD atIndex:3];
        [enc setBuffer:f->x offset:GUARD atIndex:4];
        [enc setBuffer:f->out[arm][0] offset:GUARD atIndex:5];
        [enc setBuffer:f->out[arm][1] offset:GUARD atIndex:6];
        [enc setThreadgroupMemoryLength:512 atIndex:0];
        int rows = f->rows[0] > f->rows[1] ? f->rows[0] : f->rows[1];
        [enc dispatchThreadgroups:MTLSizeMake((rows+1)/2, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, f->nsg, 1)];
    }
}

static void compare_projection(projection_fixture *f) {
    for (int m = 0; m < 2; ++m) for (int arm = 1; arm < 4; ++arm)
        equal_output(f->out[0][m], f->out[arm][m], 1, f->rows[m], f->rows[m]);
}

static void test_projection(projection_fixture *f, id<MTLCommandQueue> queue) {
    uint64_t wh[2] = {hash(f->w[0]), hash(f->w[1])};
    uint32_t seed = 513;
    for (int repeat = 0; repeat < 3; ++repeat) {
        fill_x(f->x, f->k, &seed);
        uint64_t xh = hash(f->x);
        for (int arm = 0; arm < 4; ++arm) {
            for (int m = 0; m < 2; ++m) for (int row = 0; row < f->rows[m]; ++row)
                ((uint32_t *)data(f->out[arm][m]))[row] = poison;
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            encode_projection(f, enc, arm);
            [enc endEncoding]; finish(cb);
        }
        compare_projection(f);
        require(hash(f->x) == xh, "Q8 projection input mutated");
    }
    require(hash(f->w[0]) == wh[0] && hash(f->w[1]) == wh[1], "Q8 projection weight mutated");
}



static void bench_projection(projection_fixture *f, id<MTLCommandQueue> queue) {
    enum { SAMPLES = 12, REPEATS = 128 };
    double times[4][SAMPLES];
    // Alternate all four arms and reverse the order every round; two warmups.
    for (int round = -2; round < SAMPLES; ++round) for (int position = 0; position < 4; ++position) {
        int arm = (round & 1) ? 3-position : position;
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (int r = 0; r < REPEATS; ++r) encode_projection(f, enc, arm);
        [enc endEncoding]; finish(cb);
        if (round >= 0) {
            require(cb.GPUEndTime > cb.GPUStartTime, "GPU timestamps unavailable");
            times[arm][round] = (cb.GPUEndTime - cb.GPUStartTime) * 1.e6 / REPEATS;
        }
    }
    double median[4];
    for (int arm = 0; arm < 4; ++arm) {
        qsort(times[arm], SAMPLES, sizeof(double), compare_double);
        median[arm] = (times[arm][SAMPLES/2-1] + times[arm][SAMPLES/2]) * .5;
    }
    compare_projection(f);
    printf("Q8 MV kernel-only K=%d M=%d+%d NSG=%d: separate %.3f -> %.3f us (%+.2f%%), pair %.3f -> %.3f us (%+.2f%% time).\n",
           f->k, f->rows[0], f->rows[1], f->nsg, median[0], median[1],
           100.*(median[1]/median[0]-1.), median[2], median[3], 100.*(median[3]/median[2]-1.));
}


static void test_dispatch_policy(void) {
    unsetenv("DS4_METAL_DISABLE_Q8_MV_SINGLE_BARRIER");
    require(ds4_gpu_q8_decode_mv_pipeline(4, 1024, false) ==
            ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_q8_0_f32_single_barrier", 4), "Q8 default NSG4");
    require(ds4_gpu_q8_decode_mv_pipeline(4, 3, true) ==
            ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_q8_0_f32_pair_single_barrier", 4), "Q8 paired odd rows");
    require(ds4_gpu_q8_decode_mv_pipeline(8, 1024, true) ==
            ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_q8_0_f32_pair", 8), "Q8 NSG8 fallback");
    require(ds4_gpu_q8_decode_mv_pipeline(4, 3, false) ==
            ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_q8_0_f32", 4), "Q8 odd single fallback");
    setenv("DS4_METAL_DISABLE_Q8_MV_SINGLE_BARRIER", "1", 1);
    require(ds4_gpu_q8_decode_mv_pipeline(4, 1024, false) ==
            ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_q8_0_f32", 4), "Q8 explicit rollback");
    unsetenv("DS4_METAL_DISABLE_Q8_MV_SINGLE_BARRIER");
    g_quality_mode = 1;
    require(ds4_gpu_q8_decode_mv_pipeline(4, 1024, true) ==
            ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_q8_0_f32_pair", 4), "Q8 quality fallback");
    g_quality_mode = 0; g_tp_split_world = 2;
    require(ds4_gpu_q8_decode_mv_pipeline(4, 1024, true) ==
            ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_q8_0_f32_pair", 4), "Q8 TP fallback");
    g_tp_split_world = 1;

}
int main(int argc, char **argv) {
    benchmark_enabled = argc == 2 && strcmp(argv[1], "--bench") == 0;
    if (argc > 1 && !benchmark_enabled) return 2;
    @autoreleasepool {
        require(ds4_gpu_init(), "Metal init");
        test_dispatch_policy();
        id<MTLDevice> dev = g_device;
        id<MTLCommandQueue> queue = g_queue;
        int cases = 0;
        for (int nsg = 4; nsg <= 8; nsg += 4) for (int store = 0; store < 2; ++store) {
            NSString *base = store ? @"kernel_dsv4_shared_gate_up_swiglu_q8_0" :
                                     @"kernel_dsv4_shared_mid_swiglu_q8_0";
            id<MTLComputePipelineState> oldp = ds4_gpu_get_mul_mv_pipeline(base.UTF8String, nsg);
            id<MTLComputePipelineState> newp = ds4_gpu_get_mul_mv_pipeline([base stringByAppendingString:@"_single_barrier"].UTF8String, nsg);
            const int shapes[][2] = {{32,2}, {256,32}, {4096,2048}, {7168,32}};
            for (unsigned s = 0; s < sizeof(shapes)/sizeof(shapes[0]); ++s)
                for (int clamp = 0; clamp < 2; ++clamp) {
                    test_q8(dev, queue, oldp, newp, nsg, store, shapes[s][0], shapes[s][1], clamp*4.f);
                    ++cases;
                }
        }
        printf("Q8: 32 bitwise cases passed (4/8 simdgroups, mid-only alias, clamp, repeated dispatch).\n");
        for (int nsg = 4; nsg <= 8; nsg += 4) {
            id<MTLComputePipelineState> p[4] = {
                ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_q8_0_f32", nsg),
                ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_q8_0_f32_single_barrier", nsg),
                ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_q8_0_f32_pair", nsg),
                ds4_gpu_get_mul_mv_pipeline("kernel_mul_mv_q8_0_f32_pair_single_barrier", nsg),
            };
            const int shapes[][3] = {
                {32,2,2}, {32,1,3}, {256,3,1}, {1024,65,33},
                {4096,1024,512}, {4096,512,1024}, {7168,66,34},
                {4096,2048,2048}, {32,65538,65540},
            };
            for (unsigned s = 0; s < sizeof(shapes)/sizeof(shapes[0]); ++s)
                for (int padding = 0; padding <= 2; padding += 2) {
                    projection_fixture f;
                    init_projection(&f, dev, p, nsg, shapes[s][0], shapes[s][1], shapes[s][2], padding);
                    test_projection(&f, queue);
                    ++cases;
                }
            if (benchmark_enabled) {
                projection_fixture f;
                init_projection(&f, dev, p, nsg, 4096, 1024, 512, 0);
                test_projection(&f, queue);
                bench_projection(&f, queue);
                if (nsg != 4) printf("NSG=%d candidate is benchmark-only; production keeps the legacy kernel.\n", nsg);
            }
        }
        printf("Q8 projections: 36 bitwise cases passed (single/pair, unequal/odd rows, independent strides).\n");
        printf("PASS: %d Q8 cases and dispatch policies on %s.\n",
               cases, dev.name.UTF8String);
        ds4_gpu_cleanup();
    }
    return 0;
}
