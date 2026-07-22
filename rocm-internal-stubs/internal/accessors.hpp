#ifndef ROCWMM_INTERNAL_ACCESSORS_HPP
#define ROCWMM_INTERNAL_ACCESSORS_HPP
#include <hip/hip_runtime.h>
namespace rocwmma {
    template<typename T, int rows, int cols, int ld> struct simple_accessor {
        __device__ inline T* data() { return nullptr; }
    };
}
#endif
