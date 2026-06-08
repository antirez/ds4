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

#ifndef ROCWMMA_UTILITY_MATH_HPP
#define ROCWMMA_UTILITY_MATH_HPP

#include "../config.hpp"
#include "type_traits.hpp"

namespace rocwmma
{
    // Get the next multiple
    template <typename IntT, typename = enable_if_integral_t<IntT>>
    ROCWMMA_HOST_DEVICE static inline constexpr IntT next_multiple(const IntT val, const IntT lcm);

    // Computes ceil(numerator/divisor) for integer types.
    template <typename IntT0,
              typename = enable_if_integral_t<IntT0>,
              typename IntT1,
              typename = enable_if_integral_t<IntT1>>
    ROCWMMA_HOST_DEVICE static inline constexpr IntT0 ceil_div(const IntT0 numerator,
                                                               const IntT1 divisor);

    template <typename IntT, typename = enable_if_integral_t<IntT>>
    ROCWMMA_HOST_DEVICE static inline constexpr IntT next_pow2(const IntT val);

    // Computes if the integer is a power of 2.
    template <typename IntT, typename = enable_if_integral_t<IntT>>
    ROCWMMA_HOST_DEVICE static inline constexpr bool is_pow2(const IntT val);

    template <uint32_t x>
    struct pow2
    {
        static constexpr uint32_t value = 2 * pow2<x - 1>::value;
    };

    template <>
    struct pow2<1>
    {
        static constexpr uint32_t value = 2;
    };

    template <>
    struct pow2<0>
    {
        static constexpr uint32_t value = 1;
    };

    // Calculate integer Log base 2
    template <uint32_t x>
    struct Log2
    {
        static constexpr uint32_t value = 1 + Log2<(x >> 1)>::value;
        static_assert(x % 2 == 0, "Integer input must be a power of 2");
    };

    template <>
    struct Log2<1>
    {
        static constexpr uint32_t value = 0;
    };

    template <>
    struct Log2<0>
    {
        static constexpr uint32_t value = 0;
    };

    // Create a bitmask of size BitCount, starting from the LSB bit
    template <uint32_t BitCount>
    struct LsbMask;

    template <>
    struct LsbMask<1>
    {
        enum : uint32_t
        {
            value = 0x1
        };
    };

    template <>
    struct LsbMask<0>
    {
        enum : uint32_t
        {
            value = 0x0
        };
    };

    template <uint32_t BitCount>
    struct LsbMask
    {
        enum : uint32_t
        {
            value = LsbMask<1>::value << (BitCount - 1) | LsbMask<BitCount - 1>::value
        };
    };

} // namespace rocwmma

#include "math_impl.hpp"

#endif // ROCWMMA_UTILITY_MATH_HPP
