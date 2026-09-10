/* Native driver for metal_f16_decode_bench.py. Only the Python launcher is a
 * public CLI: it resolves and hashes shader sources and validates parameters. */
#include "../tests/metal_decode_test_support.h"

@interface F16BenchCase : NSObject {
@public
    NSString *name;
    unsigned kind, input, width[2], nr, ratio, qwidth[2], virtual_q8_groups;
    short nsg;
    MV mv, qmv[2];
    Store store[2];
    id<MTLBuffer> activation, weights[4], ape[2], qweights[2];
    id<MTLBuffer> output[2][4], state[2][4], qoutput[2][2];
    id<MTLBuffer> reference[4], reference_state[4], qreference[2];
}
@end
@implementation F16BenchCase
@end

static F16BenchCase *make_case(unsigned index) {
    F16BenchCase *c = [F16BenchCase new];
    c->input = 4096;
    c->nsg = 8;
    c->nr = index == 1 ? 4 : 2;
    c->kind = index < 2 ? 0 : index == 2 ? 1 : 2;
    c->ratio = index < 2 || index == 4 ? 128 : 4;
    c->width[0] = c->ratio == 128 ? 512 : 1024;
    c->width[1] = c->ratio == 128 ? 0 : 256;
    c->qwidth[0] = 2048;
    c->qwidth[1] = 512;
    c->virtual_q8_groups = c->qwidth[0] / 2;
    c->name = @[@"pair_nr2_r128", @"pair_nr4_r128", @"quad_r4",
                @"qkv_quad_r4", @"qkv_pair_r128"][index];
    c->mv = args(c->input, c->width[0], c->nr, c->input * sizeof(_Float16));
    c->activation = floats(c->input);
    const unsigned state_rows = c->ratio == 4 ? 8 : c->ratio;
    for (unsigned pair = 0; pair < 2; pair++) {
        c->store[pair] = (Store){c->width[pair], c->ratio, c->ratio - 1, pair};
        c->ape[pair] = pair == 1 ? halfs(c->ratio * c->width[pair])
                                : floats(c->ratio * c->width[pair]);
        for (unsigned side = 0; side < 2; side++) {
            unsigned i = pair * 2 + side;
            c->weights[i] = halfs(c->width[pair] * c->input);
            c->reference[i] = buffer(c->width[pair] * sizeof(float));
            c->reference_state[i] = floats(c->width[pair] * state_rows);
            for (unsigned arm = 0; arm < 2; arm++) {
                c->output[arm][i] = copy(c->reference[i]);
                c->state[arm][i] = copy(c->reference_state[i]);
            }
        }
    }
    if (c->width[1] == 0) {
        /* The production ratio-128 compound binds unused second-compressor
         * resources to the first compressor. No grid group may touch them. */
        c->ape[1] = c->ape[0];
        for (unsigned side = 0; side < 2; side++) {
            c->weights[2 + side] = c->weights[side];
            c->reference[2 + side] = c->reference[side];
            c->reference_state[2 + side] = c->reference_state[side];
            for (unsigned arm = 0; arm < 2; arm++) {
                c->output[arm][2 + side] = c->output[arm][side];
                c->state[arm][2 + side] = c->state[arm][side];
            }
        }
    }
    if (c->kind == 2) {
        unsigned blocks = c->input / 32;
        for (unsigned i = 0; i < 2; i++) {
            c->qmv[i] = args(c->input, c->qwidth[i], 2, blocks * sizeof(Q8));
            c->qweights[i] = q8s(c->qwidth[i] * blocks);
            c->qreference[i] = buffer(c->qwidth[i] * sizeof(float));
            for (unsigned arm = 0; arm < 2; arm++) c->qoutput[arm][i] = copy(c->qreference[i]);
        }
    }
    return c;
}

static unsigned scratch_bytes(F16BenchCase *c, unsigned planes) {
    return c->kind == 2 ? 1024 : planes * 32 * c->nr * sizeof(float);
}

static void make_reference(F16BenchCase *c) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    for (unsigned pair = 0; pair < 2; pair++) {
        if (!c->width[pair]) continue;
        MV mv = c->mv;
        mv.ne01 = mv.ne0 = c->width[pair];
        plain_pair(cb, mv, c->nsg, c->weights[pair * 2], c->weights[pair * 2 + 1],
                   c->activation, c->reference[pair * 2], c->reference[pair * 2 + 1]);
        state_store(cb, c->store[pair], c->reference[pair * 2], c->reference[pair * 2 + 1],
                    c->ape[pair], c->reference_state[pair * 2], c->reference_state[pair * 2 + 1]);
    }
    if (c->kind == 2) {
        for (unsigned i = 0; i < 2; i++) {
            plain_q8(cb, c->qmv[i], 4, c->qweights[i], c->activation, c->qreference[i]);
        }
    }
    finish(cb);
}

static NSString *kernel_name(F16BenchCase *c) {
    return c->kind == 0 ? @"kernel_mul_mv_f16_f32_pair_compressor_store_4" :
           c->kind == 1 ? @"kernel_mul_mv_f16_f32_quad_compressor_store_4" :
                         @"kernel_dsv4_qkv_pair_quad_compressor_store_q8_0";
}

static id<MTLComputeCommandEncoder> encode(F16BenchCase *c, unsigned arm,
                                           unsigned planes, id<MTLCommandBuffer> cb) {
    id<MTLComputePipelineState> p = pipeline(kernel_name(c), c->nsg);
    if (p.threadExecutionWidth != 32 || p.maxTotalThreadsPerThreadgroup < 32u * c->nsg)
        die("GPU does not support this kernel's SIMD/threadgroup shape");
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:p];
    if (c->kind == 0) {
        [e setBytes:&c->mv length:sizeof(MV) atIndex:0];
        [e setBytes:&c->store[0] length:sizeof(Store) atIndex:1];
        bind(e, c->weights[0], 2); bind(e, c->weights[1], 3); bind(e, c->activation, 4);
        bind(e, c->output[arm][0], 5); bind(e, c->output[arm][1], 6); bind(e, c->ape[0], 7);
        bind(e, c->state[arm][0], 8); bind(e, c->state[arm][1], 9);
    } else if (c->kind == 1) {
        [e setBytes:&c->mv length:sizeof(MV) atIndex:0];
        [e setBytes:&c->store[0] length:sizeof(Store) atIndex:1];
        [e setBytes:&c->store[1] length:sizeof(Store) atIndex:2];
        for (unsigned i = 0; i < 4; i++) bind(e, c->weights[i], 3 + i);
        bind(e, c->activation, 7);
        for (unsigned i = 0; i < 4; i++) bind(e, c->output[arm][i], 8 + i);
        bind(e, c->ape[0], 12); bind(e, c->ape[1], 13);
        for (unsigned i = 0; i < 4; i++) bind(e, c->state[arm][i], 14 + i);
    } else {
        [e setBytes:&c->qmv[0] length:sizeof(MV) atIndex:0];
        [e setBytes:&c->qmv[1] length:sizeof(MV) atIndex:1];
        [e setBytes:&c->mv length:sizeof(MV) atIndex:2];
        [e setBytes:&c->store[0] length:sizeof(Store) atIndex:3];
        [e setBytes:&c->store[1] length:sizeof(Store) atIndex:4];
        [e setBytes:&c->virtual_q8_groups length:sizeof(unsigned) atIndex:5];
        bind(e, c->qweights[0], 6); bind(e, c->qweights[1], 7);
        for (unsigned i = 0; i < 4; i++) bind(e, c->weights[i], 8 + i);
        bind(e, c->activation, 12);
        bind(e, c->qoutput[arm][0], 13); bind(e, c->qoutput[arm][1], 14);
        for (unsigned i = 0; i < 4; i++) bind(e, c->output[arm][i], 15 + i);
        bind(e, c->ape[0], 19); bind(e, c->ape[1], 20);
        for (unsigned i = 0; i < 4; i++) bind(e, c->state[arm][i], 21 + i);
    }
    [e setThreadgroupMemoryLength:scratch_bytes(c, planes) atIndex:0];
    return e;
}

static void validate(F16BenchCase *c, unsigned arm) {
    for (unsigned i = 0; i < (c->width[1] ? 4u : 2u); i++) {
        equal(c->output[arm][i], c->reference[i], "F16 projection");
        equal(c->state[arm][i], c->reference_state[i], "compressor state and untouched slots");
        guards(c->output[arm][i]); guards(c->state[arm][i]); guards(c->weights[i]);
    }
    if (c->kind == 2) for (unsigned i = 0; i < 2; i++) {
        equal(c->qoutput[arm][i], c->qreference[i], "compound Q8 projection");
        guards(c->qoutput[arm][i]); guards(c->qweights[i]);
    }
    guards(c->activation); guards(c->ape[0]); guards(c->ape[1]);
}

static double measure(F16BenchCase *c, unsigned arm, unsigned planes, unsigned repeats) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> e = encode(c, arm, planes, cb);
    unsigned groups = c->width[0] / c->nr + c->width[1] / c->nr;
    if (c->kind == 2) groups += (c->virtual_q8_groups + 1) / 2;
    for (unsigned i = 0; i < repeats; i++) {
        [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(32, c->nsg, 1)];
    }
    [e endEncoding];
    finish(cb);
    double seconds = cb.GPUEndTime - cb.GPUStartTime;
    if (!isfinite(seconds) || seconds <= 0) die("GPU timing is unavailable or invalid");
    validate(c, arm);
    return seconds * 1e6 / repeats;
}

static void write_json(NSDictionary *value, const char *path) {
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:value options:NSJSONWritingPrettyPrinted error:&error];
    if (!data || ![data writeToFile:[NSString stringWithUTF8String:path]
                           options:NSDataWritingAtomic error:&error]) die(error.description.UTF8String);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 11) die("Use speed-bench/metal_f16_decode_bench.py to run this driver");
        unsigned samples = (unsigned)strtoul(argv[5], NULL, 10);
        unsigned repeats = (unsigned)strtoul(argv[6], NULL, 10);
        unsigned warmup = (unsigned)strtoul(argv[7], NULL, 10);
        unsigned planes[2] = {(unsigned)strtoul(argv[9], NULL, 10),
                              (unsigned)strtoul(argv[10], NULL, 10)};
        if (!samples || !repeats || samples > 100000 || repeats > 100000 || warmup > 100000 ||
            planes[0] < 1 || planes[0] > 2 || planes[1] < 1 || planes[1] > 2)
            die("Invalid benchmark parameters");
        bool both = !strcmp(argv[8], "both");
        bool fast_only = !strcmp(argv[8], "fast");
        if (!both && !fast_only && strcmp(argv[8], "strict")) die("Invalid math mode");
        device = MTLCreateSystemDefaultDevice();
        if (!device) die("No Metal GPU is available; run outside a GPU-restricted sandbox");
        queue = [device newCommandQueue];
        if (!queue) die("Could not create the Metal command queue");
        FILE *csv = fopen(argv[3], "wx");
        if (!csv) die("Cannot create output CSV (it may already exist)");
        fprintf(csv, "math,case,phase,round,order,variant,repeats,gpu_us\n");
        NSMutableArray *case_metadata = [NSMutableArray new];
        NSMutableArray *mode_metadata = [NSMutableArray new];
        for (unsigned mode = 0; mode < (both ? 2u : 1u); mode++) {
            bool fast = both ? mode == 1 : fast_only;
            id<MTLLibrary> libraries[2];
            NSMutableDictionary *caches[2];
            MTLCompileOptions *options = [MTLCompileOptions new];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = fast;
#pragma clang diagnostic pop
            for (unsigned arm = 0; arm < 2; arm++) {
                NSError *error = nil;
                NSString *source = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1 + arm]]
                                                             encoding:NSUTF8StringEncoding error:&error];
                if (!source) die(error.description.UTF8String);
                libraries[arm] = [device newLibraryWithSource:source options:options error:&error];
                if (!libraries[arm]) die(error.description.UTF8String);
                caches[arm] = [NSMutableDictionary new];
            }
            [mode_metadata addObject:@{@"math":fast ? @"fast" : @"strict",
                                       @"metal_language_version":@(options.languageVersion)}];
            for (unsigned case_index = 0; case_index < 5; case_index++) {
                @autoreleasepool {
                    seed = 0x19ae548d;
                    F16BenchCase *c = make_case(case_index);
                    library = libraries[0]; pipelines = caches[0];
                    make_reference(c);
                    /* Compile both pipelines and check one real dispatch before
                     * any warmup or sample; neither operation is timed. */
                    for (unsigned arm = 0; arm < 2; arm++) {
                        library = libraries[arm]; pipelines = caches[arm];
                        (void)measure(c, arm, planes[arm], 1);
                    }
                    if (mode == 0) [case_metadata addObject:@{
                        @"name":c->name, @"kernel":kernel_name(c), @"input_dim":@(c->input),
                        @"compressor_widths":@[@(c->width[0]), @(c->width[1])],
                        @"q8_projection_widths":c->kind == 2 ? @[@(c->qwidth[0]), @(c->qwidth[1])] : @[],
                        @"ratio":@(c->ratio), @"position":@(c->store[0].pos),
                        @"ape_types":@[@(c->store[0].ape_type), @(c->store[1].ape_type)],
                        @"rows_per_group":@(c->nr), @"simdgroups":@(c->nsg),
                        @"baseline_threadgroup_bytes":@(scratch_bytes(c, planes[0])),
                        @"current_threadgroup_bytes":@(scratch_bytes(c, planes[1])),
                        @"unused_second_resources_alias_first":@(c->width[1] == 0)
                    }];
                    for (unsigned round = 0; round < warmup + samples; round++) {
                        for (unsigned order = 0; order < 2; order++) {
                            unsigned arm = (order + round + case_index) & 1;
                            library = libraries[arm]; pipelines = caches[arm];
                            double us = measure(c, arm, planes[arm], repeats);
                            fprintf(csv, "%s,%s,%s,%u,%u,%s,%u,%.9f\n", fast ? "fast" : "strict",
                                    c->name.UTF8String, round < warmup ? "warmup" : "sample",
                                    round < warmup ? round : round - warmup, order,
                                    arm ? "current" : "baseline", repeats, us);
                            fflush(csv);
                        }
                    }
                    fprintf(stderr, "%s %s: %u paired samples, bitwise checks PASS\n",
                            fast ? "fast" : "strict", c->name.UTF8String, samples);
                }
            }
        }
        if (fclose(csv)) die("Failed to flush benchmark CSV");
        write_json(@{@"gpu":device.name, @"gpu_registry_id":@(device.registryID),
                     @"gpu_low_power":@(device.lowPower),
                     @"max_threadgroup_memory_bytes":@(device.maxThreadgroupMemoryLength),
                     @"os_version":[NSProcessInfo processInfo].operatingSystemVersionString,
                     @"modes":mode_metadata, @"cases":case_metadata,
                     @"bitwise_validation":@"passed"}, argv[4]);
        return 0;
    }
}
