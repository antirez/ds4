// SPDX-License-Identifier: MIT
// Bounded F16 activation reuse for the existing Q-B K128 WMMA arithmetic.
#ifndef DS4_ROCM_Q4_ACTIVATION_CUH
#define DS4_ROCM_Q4_ACTIVATION_CUH

#include <stdint.h>
#include <stddef.h>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define DS4_Q4_ACT_INLINE __host__ __device__ __forceinline__
#else
#define DS4_Q4_ACT_INLINE inline
#endif

namespace ds4_rocm_q4_activation {
enum : uint32_t { inner = 1024u, outputs = 32768u, tokens = 64u,
                  columns = 128u, pitch = 144u, copy_elements = 8u };
constexpr uint64_t row_bytes = (inner / 256u) * 144u;
constexpr uint64_t weight_bytes = outputs * row_bytes;
constexpr uint64_t max_bytes = 2048ull * inner * 2u;

inline bool scope(uint32_t n, uint32_t groups, uint32_t k, uint32_t m,
                  uint64_t rb, uint64_t xs, uint64_t xgs, uint64_t os,
                  const void *out, const void *w, const void *x,
                  bool k128, bool resident, bool quality, bool gfx1151) {
    return k128 && resident && !quality && gfx1151 && n >= 256u && n <= 2048u &&
           groups == 1u && k == inner && m == outputs && rb == row_bytes &&
           xs == inner && xgs == 0u && os == outputs && out && w && x &&
           (reinterpret_cast<uintptr_t>(x) & 15u) == 0u;
}

inline bool overlap(const void *a, uint64_t as, const void *b, uint64_t bs) {
    if (!a || !b || !as || !bs) return false;
    const uintptr_t ap = reinterpret_cast<uintptr_t>(a);
    const uintptr_t bp = reinterpret_cast<uintptr_t>(b);
    return ap <= bp ? static_cast<uint64_t>(bp - ap) < as
                    : static_cast<uint64_t>(ap - bp) < bs;
}

// Check the entire existing arena before cuda_tmp_alloc: growth can free it.
// The sidecar has independent ownership; never borrow or overwrite it here.
inline bool scratch_disjoint(const void *scratch, uint64_t bytes,
                             const void *out, const void *w, const void *x,
                             uint32_t n, const void *sidecar, uint64_t sidecar_bytes) {
    return !overlap(scratch, bytes, out, uint64_t(n) * outputs * sizeof(float)) &&
           !overlap(scratch, bytes, w, weight_bytes) &&
           !overlap(scratch, bytes, x, uint64_t(n) * inner * sizeof(float)) &&
           !overlap(scratch, bytes, sidecar, sidecar_bytes);
}

struct alignas(16) words4 { uint32_t words[4]; };

// Both pointers have 16-byte alignment. The tight source rows and padded LDS
// pitch preserve it for every half8 copy; no precision conversion occurs here.
template<typename SourceHalf, typename DestHalf>
DS4_Q4_ACT_INLINE void stage_thread(
        DestHalf *dst, const SourceHalf *src, uint32_t n,
        uint32_t tok0, uint32_t k0, uint32_t tid, uint32_t threads) {
    static_assert(sizeof(SourceHalf) == 2u && sizeof(DestHalf) == 2u,
                  "Q-B activation staging copies F16 object representations");
    for (uint32_t j = tid * copy_elements; j < tokens * columns;
         j += threads * copy_elements) {
        const uint32_t token = j / columns;
        const uint32_t k = j % columns;
        words4 value = {};
        if (tok0 + token < n) {
            __builtin_memcpy(&value, __builtin_assume_aligned(
                src + uint64_t(tok0 + token) * inner + k0 + k, 16), sizeof(value));
        }
        __builtin_memcpy(__builtin_assume_aligned(dst + token * pitch + k, 16),
                         &value, sizeof(value));
    }
}
} // namespace ds4_rocm_q4_activation

#undef DS4_Q4_ACT_INLINE
#endif
