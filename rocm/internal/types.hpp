#ifndef ROCWMM_INTERNAL_TYPES_HPP
#define ROCWMM_INTERNAL_TYPES_HPP

#include <hip/hip_runtime.h>

namespace rocwmma {

struct fragment_row_t {
    __device__ inline float* data() { return nullptr; }
};

struct fragment_col_t {
    __device__ inline float* data() { return nullptr; }
};

} // namespace rocwmma

#endif
