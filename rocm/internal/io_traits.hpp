#ifndef ROCWMM_INTERNAL_IO_TRAITS_HPP
#define ROCWMM_INTERNAL_IO_TRAITS_HPP

#include <hip/hip_runtime.h>

namespace rocwmma {

struct mem_traits_row_major { static constexpr int ld() { return 0; } };
struct mem_traits_col_major { static constexpr int ld() { return 0; } };

} // namespace rocwmma

#endif
