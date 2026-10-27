#ifndef DS4_GPU_COPY_H
#define DS4_GPU_COPY_H

#include <stdint.h>

/* Device copy spans, shared by ds4_gpu.h and the CUDA backend, which does
 * not include ds4_gpu.h.  The tensor stays opaque here. */
#ifndef DS4_GPU_TENSOR_DEFINED
#define DS4_GPU_TENSOR_DEFINED
typedef struct ds4_gpu_tensor ds4_gpu_tensor;
#endif

/* One device copy, with the arguments of ds4_gpu_tensor_copy. */
typedef struct {
    ds4_gpu_tensor       *dst;
    uint64_t              dst_offset;
    const ds4_gpu_tensor *src;
    uint64_t              src_offset;
    uint64_t              bytes;
} ds4_gpu_copy_span;

#define DS4_GPU_COPY_SPANS_MAX 316u

#endif
