#ifndef DS4_METAL_INTERNAL_H
#define DS4_METAL_INTERNAL_H

/*
 * Objective-C only.  A narrow, stable view of ds4_metal.m's Metal state for
 * out-of-tree kernel files (fork model families such as ds4_metal_bonsai2.m).
 * Each function is a trampoline onto a static helper in ds4_metal.m, so
 * upstream can rename or restructure its internals freely as long as these
 * few signatures keep working.  Wrappers built on these join the active
 * command batch exactly like the in-tree wrappers do.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include "ds4_gpu.h"

id<MTLDevice> ds4_metal_device(void);
id<MTLLibrary> ds4_metal_library(void);
/* Cached by name; nil (with one diagnostic) if the kernel is absent. */
id<MTLComputePipelineState> ds4_metal_get_pipeline(const char *function_name);
/* Batch-aware: returns the open batch command buffer (owned=0) or a fresh one. */
id<MTLCommandBuffer> ds4_metal_command_buffer(int *owned);
id<MTLComputeCommandEncoder> ds4_metal_compute_encoder(id<MTLCommandBuffer> cb);
void ds4_metal_end_compute_encoder(id<MTLCommandBuffer> cb, id<MTLComputeCommandEncoder> enc);
/* Commits and waits only when owned; returns 1 on success. */
int ds4_metal_finish_command_buffer(id<MTLCommandBuffer> cb, int owned, const char *label);
id<MTLBuffer> ds4_metal_tensor_buffer(const ds4_gpu_tensor *tensor);
NSUInteger ds4_metal_tensor_offset(const ds4_gpu_tensor *tensor);
id<MTLBuffer> ds4_metal_wrap_model_range(const void *model_map, uint64_t model_size,
                                         uint64_t offset, uint64_t len, uint64_t *inner_offset);

/* Whether upstream's GEMMs use Metal 4 tensor ops (M5 and later, unless
 * disabled by DS4_METAL_DISABLE_METAL4 or quality mode). */
int ds4_metal_tensor_ops_available(void);

/* Upstream's F16 x F32 GEMM (n_tok > 8 only) on an F16 weight held in a GPU
 * buffer: out_dim rows of in_dim halves starting at woff. */
int ds4_metal_matmul_f16_buffer(ds4_gpu_tensor *out, id<MTLBuffer> wbuf, NSUInteger woff,
                                uint32_t in_dim, uint32_t out_dim, const ds4_gpu_tensor *x, uint32_t n_tok);

/*
 * Extra Metal source files appended to the single compiled library, as
 * @[@[@"ENV_OVERRIDE_NAME", @"metal/file.metal"], ...].  ds4_metal.m holds a
 * weak default returning nil; a fork file provides the strong definition.
 */
NSArray<NSArray<NSString *> *> *ds4_metal_fork_sources(void);

#endif
