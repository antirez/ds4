// SPDX-License-Identifier: MIT
// Shared host/device geometry and host-testable opt-in decode policy.
#ifndef DS4_ROCM_Q4_DECODE_CUH
#define DS4_ROCM_Q4_DECODE_CUH

#include <stdint.h>
#if defined(__HIPCC__) || defined(__CUDACC__)
#define DS4_Q4_DECODE_INLINE __host__ __device__ __forceinline__
#else
#define DS4_Q4_DECODE_INLINE inline
#endif

namespace ds4_rocm_q4_decode {
enum : uint32_t { threads = 256u, lanes = 4u, rows = 64u,
                  k = 1024u, m = 32768u, blocks = 4u };
enum decision { required_failure = -1, fallback = 0, use = 1 };

DS4_Q4_DECODE_INLINE uint32_t lane(uint32_t tid) { return tid & 3u; }
DS4_Q4_DECODE_INLINE uint32_t row(uint32_t tile, uint32_t tid) {
    return tile * rows + (tid >> 2u);
}
DS4_Q4_DECODE_INLINE uint64_t mask(uint32_t tid, uint32_t wave_size) {
    // Cast BEFORE shifting: the final wave64 group occupies bits 60..63.
    return uint64_t(0x0fu) << ((tid & (wave_size - 1u)) & ~3u);
}

inline bool shape(uint64_t in_dim, uint64_t out_dim, uint64_t n_tok) {
    return in_dim == k && out_dim == m && n_tok >= 1u && n_tok <= 8u;
}
inline decision select(uint64_t in_dim, uint64_t out_dim, uint64_t n_tok,
                       uint32_t wave_size, bool quality, bool enabled,
                       bool disabled, bool required) {
    if (!enabled && !required) return fallback;
    if (disabled || quality || !shape(in_dim, out_dim, n_tok) ||
        (wave_size != 32u && wave_size != 64u))
        return required ? required_failure : fallback;
    return use;
}

// Canonical width-8 reduction first adds the empty lanes 4..7. Do not
// remove that addition: -0 + +0 and signaling NaNs are not identities.
// An explicit device intrinsic prevents -ffast-math from deleting it.
DS4_Q4_DECODE_INLINE float canonical_offset4(float v) {
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
    return __fadd_rn(v, 0.0f);
#elif defined(__CUDA_ARCH__)
    return __fadd_rn(v, 0.0f);
#else
    volatile float zero = 0.0f;
    return v + zero;
#endif
}
} // namespace ds4_rocm_q4_decode

#undef DS4_Q4_DECODE_INLINE
#endif
