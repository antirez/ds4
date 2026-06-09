/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef ROCWMMA_UTILITY_MATH_IMPL_HPP
#define ROCWMMA_UTILITY_MATH_IMPL_HPP

#include "math.hpp"

namespace rocwmma
{
    // Get the next multiple
    template <typename IntT, typename /*= enable_if_integral_t<IntT>*/>
    ROCWMMA_HOST_DEVICE static inline constexpr IntT next_multiple(const IntT val, const IntT lcm)
    {
        return (val % lcm) ? (val + (lcm - val % lcm)) : val;
    }

    // Computes ceil(numerator/divisor) for integer types.
    template <typename IntT0,
              typename /*= enable_if_integral_t<IntT0>*/,
              typename IntT1,
              typename /*= enable_if_integral_t<IntT1>*/>
    ROCWMMA_HOST_DEVICE static inline constexpr IntT0 ceil_div(const IntT0 numerator,
                                                               const IntT1 divisor)
    {
        return (numerator + divisor - 1u) / divisor;
    }

    template <typename IntT, typename /*= enable_if_integral_t<IntT>*/>
    ROCWMMA_HOST_DEVICE static inline constexpr IntT next_pow2(const IntT val)
    {
        // Precondition: x > 1.
        return val > 1u ? (1u << (32u - __builtin_clz(val - 1u))) : val;
    }

    // Computes if the integer is a power of 2.
    template <typename IntT, typename /*= enable_if_integral_t<IntT>*/>
    ROCWMMA_HOST_DEVICE static inline constexpr bool is_pow2(const IntT val)
    {
        return val > 0u ? (val & (val - 1u)) == 0u : false;
    }

} // namespace rocwmma

#endif // ROCWMMA_UTILITY_MATH_IMPL_HPP
