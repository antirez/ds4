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
#ifndef ROCWMMA_TYPES_EXT_HPP
#define ROCWMMA_TYPES_EXT_HPP

#if !defined(__HIPCC_RTC__)
#include <cmath>
#include <hip/hip_bfloat16.h>
#include <limits>
#include <ostream>
#else
#include "utils.hpp"
#endif // !defined(__HIPCC_RTC__)

#include "type_traits.hpp"
#include "types.hpp"

namespace rocwmma
{

#if !defined(__HIPCC_RTC__)

    ////////////////////////////////////////////////////////////////////////
    ///////////  rocwmma::hfloat16_t host and device conversions  //////////
    ////////////////////////////////////////////////////////////////////////
    template <typename Outgoing,
              typename Incoming,
              enable_if_t<!is_same_v<Incoming, Outgoing>, int> = 0>
    __host__ __device__ inline Outgoing convert(const Incoming& value)
    {
#if !ROCWMMA_NO_HALF
        if constexpr(is_same_v<Outgoing, hfloat16_t>)
        {

#if defined(__HIP_NO_HALF_CONVERSIONS__)
            detail::Fp16Bits fp16(static_cast<float16_t>(value));
            return fp16.h16;
#else
            return static_cast<hfloat16_t>(value);
#endif // defined(__HIP_NO_HALF_CONVERSIONS__)
        }
        else if constexpr(is_same_v<Incoming, hfloat16_t>)
        {

#if defined(__HIP_NO_HALF_CONVERSIONS__)
            detail::Fp16Bits fp16(value);
            return static_cast<Outgoing>(fp16.f16);
#else
            return static_cast<Outgoing>(value);
#endif // defined(__HIP_NO_HALF_CONVERSIONS__)
        }
        else
#endif // !ROCWMMA_NO_HALF
        {
            return static_cast<Outgoing>(value);
        }
    }

    template <typename Outgoing,
              typename Incoming,
              enable_if_t<is_same_v<Incoming, Outgoing>, int> = 0>
    __host__ __device__ inline Outgoing const& convert(const Incoming& value)
    {
        return value;
    }

    ////////////////////////////////////////////////////////////////////
    ///////////  rocwmma::hfloat16_t host & device operators  //////////
    ///////////////////////////////////////////////////////////////////

#if defined(__HIP_NO_HALF_OPERATORS__)
// No operators defined for host or device
#define ROCWMMA_HALF_OP_ATTR ROCWMMA_HOST_DEVICE
#else
// No operators defined just for host
#define ROCWMMA_HALF_OP_ATTR ROCWMMA_HOST
#endif // defined(__HIP_NO_HALF_OPERATORS__)

#if !ROCWMMA_NO_HALF

    ROCWMMA_HALF_OP_ATTR inline bool operator==(const hfloat16_t& x, const hfloat16_t& y)
    {
        auto absDiff = std::fabs(__half2float(x) - __half2float(y));
        auto absAdd  = std::fabs(__half2float(x) + __half2float(y));
        return absDiff <= __half2float(numeric_limits<hfloat16_t>::epsilon()) * absAdd * 2.0f
               || absDiff < __half2float(numeric_limits<hfloat16_t>::min());
    }

    ROCWMMA_HALF_OP_ATTR inline bool operator!=(const hfloat16_t& x, const hfloat16_t& y)
    {
        return !(x == y);
    }

    ROCWMMA_HALF_OP_ATTR inline hfloat16_t operator-(const hfloat16_t& x)
    {
        detail::Fp16Bits fp16(x);
        fp16.i16 ^= 0x8000; // Flip sign
        return fp16.h16;
    }

    ROCWMMA_HALF_OP_ATTR inline hfloat16_t operator+(const hfloat16_t& x, const hfloat16_t& y)
    {
        return convert<hfloat16_t>(convert<float16_t>(x) + convert<float16_t>(y));
    }

    ROCWMMA_HALF_OP_ATTR inline hfloat16_t operator-(const hfloat16_t& x, const hfloat16_t& y)
    {
        return convert<hfloat16_t>(convert<float16_t>(x) - convert<float16_t>(y));
    }

    ROCWMMA_HALF_OP_ATTR inline hfloat16_t operator*(const hfloat16_t& x, const hfloat16_t& y)
    {
        return convert<hfloat16_t>(convert<float16_t>(x) * convert<float16_t>(y));
    }

    ROCWMMA_HALF_OP_ATTR inline hfloat16_t operator/(const hfloat16_t& x, const hfloat16_t& y)
    {
        return convert<hfloat16_t>(convert<float16_t>(x) / convert<float16_t>(y));
    }

    ROCWMMA_HALF_OP_ATTR inline hfloat16_t& operator+=(hfloat16_t& x, const hfloat16_t& y)
    {
        return x = x + y;
    }

    ROCWMMA_HALF_OP_ATTR inline hfloat16_t& operator-=(hfloat16_t& x, const hfloat16_t& y)
    {
        return x = x - y;
    }

    ROCWMMA_HALF_OP_ATTR inline hfloat16_t& operator*=(hfloat16_t& x, const hfloat16_t& y)
    {
        return x = x * y;
    }

    ROCWMMA_HALF_OP_ATTR inline hfloat16_t& operator/=(hfloat16_t& x, const hfloat16_t& y)
    {
        return x = x / y;
    }

#endif // !ROCWMMA_NO_HALF

#endif // !defined(__HIPCC_RTC__)

} // namespace rocwmma

namespace std
{
#if !defined(__HIPCC_RTC__)
    ///////////////////////////////////////////////////////////
    //////////  std::ostream::operator<<(float16_t)  //////////
    ///////////////////////////////////////////////////////////

    inline ostream& operator<<(ostream& stream, rocwmma::float16_t const& val)
    {
        return stream << static_cast<float>(val);
    }

    ///////////////////////////////////////////////////////////
    //////////  std::ostream::operator<<(hfloat16_t)  /////////
    ///////////////////////////////////////////////////////////
#if !ROCWMMA_NO_HALF
    inline ostream& operator<<(ostream& stream, rocwmma::hfloat16_t const& val)
    {
        return stream << __half2float(val);
    }
#endif // !ROCWMMA_NO_HALF

#endif // !defined(__HIPCC_RTC__) && !ROCWMMA_NO_HALF

} // namespace std

#endif // ROCWMMA_TYPES_EXT_HPP
