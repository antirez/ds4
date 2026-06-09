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

#ifndef ROCWMMA_TYPE_TRAITS_HPP
#define ROCWMMA_TYPE_TRAITS_HPP

#if !defined(__HIPCC_RTC__)

#include <cfloat>

#else

#define FLT_EPSILON __FLT_EPSILON__
#define FLT_MAX __FLT_MAX__
#define FLT_MIN __FLT_MIN__
#define HUGE_VALF (__builtin_huge_valf())

#endif // !defined(__HIPCC_RTC__)

#include "types.hpp"

namespace rocwmma
{
    namespace detail
    {
        struct Fp16Bits
        {
            union
            {
                uint16_t  i16;
                float16_t f16;
#if !ROCWMMA_NO_HALF
                hfloat16_t h16;
#endif // !ROCWMMA_NO_HALF
                bfloat16_t b16;
            };
            constexpr Fp16Bits(uint16_t initVal)
                : i16(initVal)
            {
            }
            constexpr Fp16Bits(float16_t initVal)
                : f16(initVal)
            {
            }
#if !ROCWMMA_NO_HALF
            constexpr Fp16Bits(hfloat16_t initVal)
                : h16(initVal)
            {
            }
#endif
            constexpr Fp16Bits(bfloat16_t initVal)
                : b16(initVal)
            {
            }
        };

        struct Fp32Bits
        {
            union
            {
                uint32_t   i32;
                float32_t  f32;
                xfloat32_t xf32;
            };
            constexpr Fp32Bits(uint32_t initVal)
                : i32(initVal)
            {
            }
            constexpr Fp32Bits(float32_t initVal)
                : f32(initVal)
            {
            }
            constexpr Fp32Bits(xfloat32_t initVal)
                : xf32(initVal)
            {
            }
        };

    } // namespace detail
} // namespace rocwmma

#include "utility/numeric_limits.hpp"

#if defined(__HIPCC_RTC__)
#define NUMERIC_LIMITS_NAMESPACE rocwmma::detail
#else
#define NUMERIC_LIMITS_NAMESPACE std
#endif

namespace NUMERIC_LIMITS_NAMESPACE
{
#if defined(__HIPCC_RTC__)
    using uint16_t = rocwmma::uint16_t;
#endif

    ///////////////////////////////////////////////////////////
    ///////////  std::numeric_limits<float16_t>  //////////////
    ///////////////////////////////////////////////////////////

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::float16_t
        numeric_limits<rocwmma::float16_t>::epsilon() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x1400));
        return eps.f16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::float16_t
        numeric_limits<rocwmma::float16_t>::infinity() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7C00));
        return eps.f16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::float16_t
        numeric_limits<rocwmma::float16_t>::lowest() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0xFBFF));
        return eps.f16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::float16_t
        numeric_limits<rocwmma::float16_t>::max() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7BFF));
        return eps.f16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::float16_t
        numeric_limits<rocwmma::float16_t>::min() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x0400));
        return eps.f16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::float16_t
        numeric_limits<rocwmma::float16_t>::quiet_NaN() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7FFF));
        return eps.f16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::float16_t
        numeric_limits<rocwmma::float16_t>::signaling_NaN() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7DFF));
        return eps.f16;
    }

    ///////////////////////////////////////////////////////////
    ///////////  numeric_limits<hfloat16_t>  /////////////
    ///////////////////////////////////////////////////////////
#if !ROCWMMA_NO_HALF
    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::hfloat16_t
        numeric_limits<rocwmma::hfloat16_t>::epsilon() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x1400));
        return eps.h16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::hfloat16_t
        numeric_limits<rocwmma::hfloat16_t>::infinity() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7C00));
        return eps.h16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::hfloat16_t
        numeric_limits<rocwmma::hfloat16_t>::lowest() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0xFBFF));
        return eps.h16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::hfloat16_t
        numeric_limits<rocwmma::hfloat16_t>::max() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7BFF));
        return eps.h16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::hfloat16_t
        numeric_limits<rocwmma::hfloat16_t>::min() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x0400));
        return eps.h16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::hfloat16_t
        numeric_limits<rocwmma::hfloat16_t>::quiet_NaN() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7FFF));
        return eps.h16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::hfloat16_t
        numeric_limits<rocwmma::hfloat16_t>::signaling_NaN() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7DFF));
        return eps.h16;
    }

#endif // !ROCWMMA_NO_HALF

    ///////////////////////////////////////////////////////////
    ///////////  numeric_limits<bfloat16_t>  /////////////
    ///////////////////////////////////////////////////////////

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::bfloat16_t
        numeric_limits<rocwmma::bfloat16_t>::epsilon() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x3C00));
        return eps.b16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::bfloat16_t
        numeric_limits<rocwmma::bfloat16_t>::infinity() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7F80));
        return eps.b16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::bfloat16_t
        numeric_limits<rocwmma::bfloat16_t>::lowest() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0xFF7F));
        return eps.b16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::bfloat16_t
        numeric_limits<rocwmma::bfloat16_t>::max() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7F7F));
        return eps.b16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::bfloat16_t
        numeric_limits<rocwmma::bfloat16_t>::min() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x007F));
        return eps.b16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::bfloat16_t
        numeric_limits<rocwmma::bfloat16_t>::quiet_NaN() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7FC0));
        return eps.b16;
    }

    template <>
    ROCWMMA_HOST_DEVICE constexpr rocwmma::bfloat16_t
        numeric_limits<rocwmma::bfloat16_t>::signaling_NaN() noexcept
    {
        rocwmma::detail::Fp16Bits eps(static_cast<uint16_t>(0x7FC0));
        return eps.b16;
    }

} // namespace rocwmma

namespace rocwmma
{
#if !defined(__HIPCC_RTC__)
    template <typename T, enable_if_t<is_integral<T>::value, int> = 0>
    constexpr auto maxExactInteger() -> decltype(numeric_limits<T>::max())
    {
        return numeric_limits<T>::max();
    }

    template <typename T,
              enable_if_t<is_floating_point<T>::value && numeric_limits<T>::digits, int> = 0>
    constexpr auto maxExactInteger()
        -> conditional_t<is_same<T, float64_t>::value, int64_t, int32_t>
    {
        using RetT = conditional_t<is_same<T, float64_t>::value, int64_t, int32_t>;
        return ((RetT)1 << numeric_limits<T>::digits);
    }

    template <typename T,
              enable_if_t<
#if !ROCWMMA_NO_HALF
                  is_same<T, hfloat16_t>::value ||
#endif // !ROCWMMA_NO_HALF
                      is_same<T, float16_t>::value,
                  int>
              = 0>
    constexpr auto maxExactInteger() -> int32_t
    {
        // f16 mantissa is 10 bits
        return ((int32_t)1 << 11);
    }

    template <typename T, enable_if_t<is_same<T, bfloat16_t>::value, int> = 0>
    constexpr auto maxExactInteger() -> int32_t
    {
        // b16 mantissa is 7 bits
        return ((int32_t)1 << 8);
    }

    template <typename T, enable_if_t<is_same<T, float8_fnuz_t>::value, int> = 0>
    constexpr auto maxExactInteger() -> int32_t
    {
        // f8 mantissa is 3 bits
        return ((int32_t)1 << 4);
    }

    template <typename T, enable_if_t<is_same<T, bfloat8_fnuz_t>::value, int> = 0>
    constexpr auto maxExactInteger() -> int32_t
    {
        // bf8 mantissa is 2 bits
        return ((int32_t)1 << 3);
    }

    template <typename T, enable_if_t<is_same<T, xfloat32_t>::value, int> = 0>
    constexpr auto maxExactInteger() -> int32_t
    {
        // xf32 mantissa is 7 bits
        return ((int32_t)1 << 8);
    }

#endif // !defined(__HIPCC_RTC__)

} // namespace rocwmma

#endif // ROCWMMA_TYPE_TRAITS_HPP
