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

#ifndef ROCWMMA_FLOAT8_HPP
#define ROCWMMA_FLOAT8_HPP

// NOTE:
// We want to define some type traits such as
// is_arithmetic, and is_floating_point.
// Need to define them first before the hip f16 header
// is included, as it instantiates is_floating_point
// in class definitions.
#include "utility/type_traits.hpp"

struct __hip_fp8_e4m3;
struct __hip_fp8_e5m2;
struct __hip_fp8_e4m3_fnuz;
struct __hip_fp8_e5m2_fnuz;

using hip_fp8_e4m3 = __hip_fp8_e4m3;
using hip_fp8_e5m2 = __hip_fp8_e5m2;

using hip_fp8_e4m3_fnuz = __hip_fp8_e4m3_fnuz;
using hip_fp8_e5m2_fnuz = __hip_fp8_e5m2_fnuz;

namespace ROCWMMA_TYPE_TRAITS_IMPL_NAMESPACE
{
    template <>
    struct is_arithmetic<hip_fp8_e4m3> : true_type
    {
    };

    template <>
    struct is_floating_point<hip_fp8_e4m3> : true_type
    {
    };

    template <>
    struct is_signed<hip_fp8_e4m3> : true_type
    {
    };

    template <>
    struct is_arithmetic<hip_fp8_e5m2> : true_type
    {
    };

    template <>
    struct is_floating_point<hip_fp8_e5m2> : true_type
    {
    };

    template <>
    struct is_signed<hip_fp8_e5m2> : true_type
    {
    };

    //////////////////////////////////////////
    ///       FNUZ f8 / bf8  overloads     ///
    //////////////////////////////////////////
    template <>
    struct is_arithmetic<hip_fp8_e4m3_fnuz> : true_type
    {
    };

    template <>
    struct is_floating_point<hip_fp8_e4m3_fnuz> : true_type
    {
    };

    template <>
    struct is_signed<hip_fp8_e4m3_fnuz> : true_type
    {
    };

    template <>
    struct is_arithmetic<hip_fp8_e5m2_fnuz> : true_type
    {
    };

    template <>
    struct is_floating_point<hip_fp8_e5m2_fnuz> : true_type
    {
    };

    template <>
    struct is_signed<hip_fp8_e5m2_fnuz> : true_type
    {
    };

} // namespace ROCWMMA_TYPE_TRAITS_IMPL_NAMESPACE

// Include full implementations for following overrides.
#include "utility/numeric_limits.hpp"
#include <hip/hip_fp8.h>

// From HIP, device visibility of fp8/bf8 is limited to certain devices.
// Host has visibility of all fp8/bf8 types
#if defined(HIP_FP8_TYPE_FNUZ) && HIP_FP8_TYPE_FNUZ

// TODO: once HIP_FP8_TYPE_FNUZ symbol fixed on hipRTC
// 1. Re-enable static assert
// 2. Give ROCWMMA_FP8_FNUZ value of 1

//static_assert((bool)ROCWMMA_ARCH_GFX942 || (bool)ROCWMMA_ARCH_HOST,
//              "fp8_fnuz types only supported on gfx942 arch");

#define ROCWMMA_FP8_FNUZ (ROCWMMA_ARCH_GFX942 || ROCWMMA_ARCH_HOST)
#define ROCWMMA_FP8_FNUZ_VISIBILITY ROCWMMA_HOST_DEVICE
#else
#define ROCWMMA_FP8_FNUZ 0
#define ROCWMMA_FP8_FNUZ_VISIBILITY ROCWMMA_HOST
#endif // defined(HIP_FP8_TYPE_FNUZ) && HIP_FP8_TYPE_FNUZ

#if defined(HIP_FP8_TYPE_OCP) && HIP_FP8_TYPE_OCP

// TODO: once HIP_FP8_TYPE_OCP symbol fixed on hipRTC
// 1. Re-enable static assert
// 2. Give ROCWMMA_FP8 value of 1

//static_assert((bool)ROCWMMA_ARCH_GFX12 || (bool)ROCWMMA_ARCH_HOST,
//              "fp8_fnuz types only supported on gfx12 archs");

#define ROCWMMA_FP8 (ROCWMMA_ARCH_GFX950 || ROCWMMA_ARCH_GFX12 || ROCWMMA_ARCH_HOST)
#define ROCWMMA_FP8_VISIBILITY ROCWMMA_HOST_DEVICE
#else

#define ROCWMMA_FP8 0
#define ROCWMMA_FP8_VISIBILITY ROCWMMA_HOST
#endif // defined(HIP_FP8_TYPE_OCP) && HIP_FP8_TYPE_OCP

#if defined(__HIPCC_RTC__)
#if ROCWMMA_FP8_FNUZ
#define ROCWMMA_ENABLE_FNUZ_HIPRTC 1
#else
#define ROCWMMA_ENABLE_FNUZ_HIPRTC 0
#endif

#if ROCWMMA_FP8_OCP
#define ROCWMMA_ENABLE_OCP_HIPRTC 1
#else
#define ROCWMMA_ENABLE_OCP_HIPRTC 0
#endif
#endif

#if !defined(ROCWMMA_ENABLE_OCP_HIPRTC) || ROCWMMA_ENABLE_OCP_HIPRTC
ROCWMMA_FP8_VISIBILITY constexpr inline hip_fp8_e4m3
    make_hip_fp8_e4m3_from_bits(__hip_fp8_storage_t bits)
{
    static_assert(sizeof(hip_fp8_e4m3) == sizeof(__hip_fp8_storage_t),
                  "Sizes of hip_fp8_e4m3 and __hip_fp8_storage_t are different");
    union
    {
        __hip_fp8_storage_t c8;
        hip_fp8_e4m3        fp8;

    } result{bits};
    return result.fp8;
}

ROCWMMA_FP8_VISIBILITY constexpr inline hip_fp8_e5m2
    make_hip_fp8_e5m2_from_bits(__hip_fp8_storage_t bits)
{
    static_assert(sizeof(hip_fp8_e5m2) == sizeof(__hip_fp8_storage_t),
                  "Sizes of hip_fp8_e5m2 and __hip_fp8_storage_t are different");
    union
    {
        __hip_fp8_storage_t c8;
        hip_fp8_e5m2        fp8;

    } result{bits};
    return result.fp8;
}

#if !defined(__HIPCC_RTC__)

// Special operator overloading
inline std::ostream& operator<<(std::ostream& os, hip_fp8_e4m3 a)
{
    return os << float(a);
}

inline std::ostream& operator<<(std::ostream& os, hip_fp8_e5m2 a)
{
    return os << float(a);
}

#endif // !defined(__HIPCC_RTC__)

// Unary sign inversion
ROCWMMA_FP8_VISIBILITY inline hip_fp8_e4m3 operator-(hip_fp8_e4m3 a)
{
    return make_hip_fp8_e4m3_from_bits(a.__x ^ 0x80);
}

ROCWMMA_FP8_VISIBILITY inline hip_fp8_e5m2 operator-(hip_fp8_e5m2 a)
{
    return make_hip_fp8_e5m2_from_bits(a.__x ^ 0x80);
}

// all + operator overloading with mixed types
// mixed types, always converts to f32, does computation in f32, and returns float
ROCWMMA_FP8_VISIBILITY inline float operator+(const float fa, hip_fp8_e4m3 b)
{
    return (fa + float(b));
}

ROCWMMA_FP8_VISIBILITY inline float operator+(hip_fp8_e4m3 a, const float fb)
{
    return (float(a) + fb);
}

ROCWMMA_FP8_VISIBILITY inline float operator+(hip_fp8_e5m2 a, const float fb)
{
    return (float(a) + fb);
}

ROCWMMA_FP8_VISIBILITY inline float operator+(hip_fp8_e4m3 a, hip_fp8_e5m2 b)
{
    return (float(a) + float(b));
}

ROCWMMA_FP8_VISIBILITY inline float operator+(hip_fp8_e5m2 a, hip_fp8_e4m3 b)
{
    return (float(a) + float(b));
}

ROCWMMA_FP8_VISIBILITY inline hip_fp8_e4m3 operator+(hip_fp8_e4m3 a, hip_fp8_e4m3 b)
{
    return hip_fp8_e4m3(float(a) + float(b));
}

ROCWMMA_FP8_VISIBILITY inline hip_fp8_e5m2 operator+(hip_fp8_e5m2 a, hip_fp8_e5m2 b)
{
    return hip_fp8_e5m2(float(a) + float(b));
}

ROCWMMA_FP8_VISIBILITY inline hip_fp8_e4m3& operator+=(hip_fp8_e4m3& a, hip_fp8_e4m3 b)
{
    return a = hip_fp8_e4m3(float(a) + float(b));
}

ROCWMMA_FP8_VISIBILITY inline hip_fp8_e5m2& operator+=(hip_fp8_e5m2& a, hip_fp8_e5m2 b)
{
    return a = hip_fp8_e5m2(float(a) + float(b));
}

// all - operator overloading with mixed types
// mixed types, always converts to f32, does computation in f32, and returns float
ROCWMMA_FP8_VISIBILITY inline float operator-(const float fa, hip_fp8_e4m3 b)
{
    return (fa - float(b));
}

ROCWMMA_FP8_VISIBILITY inline float operator-(const float fa, hip_fp8_e5m2 b)
{
    return (fa - float(b));
}

ROCWMMA_FP8_VISIBILITY inline float operator-(hip_fp8_e4m3 a, const float fb)
{
    return (float(a) - fb);
}

ROCWMMA_FP8_VISIBILITY inline float operator-(hip_fp8_e5m2 a, const float fb)
{
    return (float(a) - fb);
}

ROCWMMA_FP8_VISIBILITY inline float operator-(hip_fp8_e4m3 a, hip_fp8_e5m2 b)
{
    return (float(a) - float(b));
}

ROCWMMA_FP8_VISIBILITY inline float operator-(hip_fp8_e5m2 a, hip_fp8_e4m3 b)
{
    return (float(a) - float(b));
}

ROCWMMA_FP8_VISIBILITY inline hip_fp8_e4m3 operator-(hip_fp8_e4m3 a, hip_fp8_e4m3 b)
{
    return hip_fp8_e4m3(float(a) - float(b));
}

ROCWMMA_FP8_VISIBILITY inline hip_fp8_e5m2 operator-(hip_fp8_e5m2 a, hip_fp8_e5m2 b)
{
    return hip_fp8_e5m2(float(a) - float(b));
}

ROCWMMA_FP8_VISIBILITY inline hip_fp8_e4m3& operator-=(hip_fp8_e4m3& a, hip_fp8_e4m3 b)
{
    return a = hip_fp8_e4m3(float(a) - float(b));
}

ROCWMMA_FP8_VISIBILITY inline hip_fp8_e5m2& operator-=(hip_fp8_e5m2& a, hip_fp8_e5m2 b)
{
    return a = hip_fp8_e5m2(float(a) - float(b));
}

// overloading multiplication, always returns float,
ROCWMMA_FP8_VISIBILITY inline float operator*(hip_fp8_e4m3 a, hip_fp8_e4m3 b)
{
    return float(a) * float(b);
}

ROCWMMA_FP8_VISIBILITY inline float operator*(float a, hip_fp8_e4m3 b)
{
    return (a * float(b));
}

ROCWMMA_FP8_VISIBILITY inline float operator*(hip_fp8_e4m3 a, float b)
{
    return (float(a) * b);
}

ROCWMMA_FP8_VISIBILITY inline float operator*(int32_t a, hip_fp8_e4m3 b)
{
    return ((float)a * float(b));
}

ROCWMMA_FP8_VISIBILITY inline float operator*(double a, hip_fp8_e4m3 b)
{
    return ((float)a * float(b));
}

ROCWMMA_FP8_VISIBILITY inline float operator*(hip_fp8_e5m2 a, hip_fp8_e5m2 b)
{
    return float(a) * float(b);
}

ROCWMMA_FP8_VISIBILITY inline float operator*(float a, hip_fp8_e5m2 b)
{
    return (a * float(b));
}

ROCWMMA_FP8_VISIBILITY inline float operator*(hip_fp8_e5m2 a, float b)
{
    return (float(a) * b);
}

ROCWMMA_FP8_VISIBILITY inline float operator*(int32_t a, hip_fp8_e5m2 b)
{
    return ((float)a * float(b));
}

ROCWMMA_FP8_VISIBILITY inline float operator*(double a, hip_fp8_e5m2 b)
{
    return ((float)a * float(b));
}

// overloading for mixed f8 and bf8 types
ROCWMMA_FP8_VISIBILITY inline float operator*(hip_fp8_e4m3 a, hip_fp8_e5m2 b)
{
    return float(a) * float(b);
}

ROCWMMA_FP8_VISIBILITY inline float operator*(hip_fp8_e5m2 a, hip_fp8_e4m3 b)
{
    return float(a) * float(b);
}

// overloading division, always returns float,
ROCWMMA_FP8_VISIBILITY inline float operator/(hip_fp8_e4m3 a, hip_fp8_e4m3 b)
{
    return float(a) / float(b);
}

ROCWMMA_FP8_VISIBILITY inline float operator/(float a, hip_fp8_e4m3 b)
{
    return (a / float(b));
}

ROCWMMA_FP8_VISIBILITY inline float operator/(hip_fp8_e4m3 a, float b)
{
    return (float(a) / b);
}

ROCWMMA_FP8_VISIBILITY inline float operator/(int32_t a, hip_fp8_e4m3 b)
{
    return ((float)a / float(b));
}

ROCWMMA_FP8_VISIBILITY inline float operator/(double a, hip_fp8_e4m3 b)
{
    return ((float)a / float(b));
}

ROCWMMA_FP8_VISIBILITY inline float operator/(hip_fp8_e5m2 a, hip_fp8_e5m2 b)
{
    return float(a) / float(b);
}

ROCWMMA_FP8_VISIBILITY inline float operator/(float a, hip_fp8_e5m2 b)
{
    return (a / float(b));
}

ROCWMMA_FP8_VISIBILITY inline float operator/(hip_fp8_e5m2 a, float b)
{
    return (float(a) / b);
}

ROCWMMA_FP8_VISIBILITY inline float operator/(int32_t a, hip_fp8_e5m2 b)
{
    return ((float)a / float(b));
}

ROCWMMA_FP8_VISIBILITY inline float operator/(double a, hip_fp8_e5m2 b)
{
    return ((float)a / float(b));
}

// overloading for mixed f8 and bf8 types
ROCWMMA_FP8_VISIBILITY inline float operator/(hip_fp8_e4m3 a, hip_fp8_e5m2 b)
{
    return float(a) / float(b);
}

ROCWMMA_FP8_VISIBILITY inline float operator/(hip_fp8_e5m2 a, hip_fp8_e4m3 b)
{
    return float(a) / float(b);
}

ROCWMMA_FP8_VISIBILITY inline hip_fp8_e4m3& operator/=(hip_fp8_e4m3& a, hip_fp8_e4m3 b)
{
    return a = hip_fp8_e4m3(float(a) / float(b));
}

ROCWMMA_FP8_VISIBILITY inline hip_fp8_e5m2& operator/=(hip_fp8_e5m2& a, hip_fp8_e5m2 b)
{
    return a = hip_fp8_e5m2(float(a) / float(b));
}

// Comparison operators
ROCWMMA_FP8_VISIBILITY inline bool operator==(hip_fp8_e4m3 a, hip_fp8_e4m3 b)
{
    return (a.__x == b.__x);
}

ROCWMMA_FP8_VISIBILITY inline bool operator==(hip_fp8_e5m2 a, hip_fp8_e5m2 b)
{
    return (a.__x == b.__x);
}

ROCWMMA_FP8_VISIBILITY inline bool operator!=(hip_fp8_e4m3 a, hip_fp8_e4m3 b)
{
    return (a.__x != b.__x);
}

ROCWMMA_FP8_VISIBILITY inline bool operator!=(hip_fp8_e5m2 a, hip_fp8_e5m2 b)
{
    return (a.__x != b.__x);
}

ROCWMMA_FP8_VISIBILITY inline bool operator<(hip_fp8_e4m3 a, hip_fp8_e4m3 b)
{
    return float(a) < float(b);
}

ROCWMMA_FP8_VISIBILITY inline bool operator<(hip_fp8_e5m2 a, hip_fp8_e5m2 b)
{
    return float(a) < float(b);
}

ROCWMMA_FP8_VISIBILITY inline bool operator>(hip_fp8_e4m3 a, hip_fp8_e4m3 b)
{
    return float(a) > float(b);
}

ROCWMMA_FP8_VISIBILITY inline bool operator>(hip_fp8_e5m2 a, hip_fp8_e5m2 b)
{
    return float(a) > float(b);
}

ROCWMMA_FP8_VISIBILITY inline bool operator<=(hip_fp8_e4m3 a, hip_fp8_e4m3 b)
{
    return float(a) <= float(b);
}

ROCWMMA_FP8_VISIBILITY inline bool operator<=(hip_fp8_e5m2 a, hip_fp8_e5m2 b)
{
    return float(a) <= float(b);
}

ROCWMMA_FP8_VISIBILITY inline bool operator>=(hip_fp8_e4m3 a, hip_fp8_e4m3 b)
{
    return float(a) >= float(b);
}

ROCWMMA_FP8_VISIBILITY inline bool operator>=(hip_fp8_e5m2 a, hip_fp8_e5m2 b)
{
    return float(a) >= float(b);
}

// Define numeric limits traits
namespace ROCWMMA_NUMERIC_LIMITS_IMPL_NAMESPACE
{
    template <>
    class numeric_limits<hip_fp8_e4m3>
    {
    public:
        static constexpr bool is_specialized    = true;
        static constexpr bool is_signed         = true;
        static constexpr bool is_integer        = false;
        static constexpr bool is_exact          = false;
        static constexpr bool has_infinity      = false;
        static constexpr bool has_quiet_NaN     = true;
        static constexpr bool has_signaling_NaN = true;
        static constexpr auto has_denorm        = true;
        static constexpr auto has_denorm_loss   = true;
        static constexpr auto round_style       = round_to_nearest;
        static constexpr bool is_iec559         = false;
        static constexpr bool is_bounded        = true;
        static constexpr bool is_modulo         = false;
        static constexpr int  digits            = 4;
        static constexpr int  digits10          = 0;
        static constexpr int  max_digits10      = 3;
        static constexpr int  radix             = 2;
        static constexpr int  min_exponent      = -5;
        static constexpr int  min_exponent10    = -1;
        static constexpr int  max_exponent      = 8;
        static constexpr int  max_exponent10    = 2;
        static constexpr auto traps             = false;
        static constexpr auto tinyness_before   = false;

        static constexpr hip_fp8_e4m3 min()
        {
            return make_hip_fp8_e4m3_from_bits(0x08);
        }
        static constexpr hip_fp8_e4m3 lowest()
        {
            return make_hip_fp8_e4m3_from_bits(0xFE);
        }
        static constexpr hip_fp8_e4m3 max()
        {
            return make_hip_fp8_e4m3_from_bits(0x7E);
        }
        static constexpr hip_fp8_e4m3 epsilon()
        {
            return make_hip_fp8_e4m3_from_bits(0x20);
        }
        static constexpr hip_fp8_e4m3 round_error()
        {
            return make_hip_fp8_e4m3_from_bits(0x30);
        }
        static constexpr hip_fp8_e4m3 quiet_NaN()
        {
            return make_hip_fp8_e4m3_from_bits(0x7F);
        }
        static constexpr hip_fp8_e4m3 signaling_NaN()
        {
            return make_hip_fp8_e4m3_from_bits(0x7F);
        }
        static constexpr hip_fp8_e4m3 denorm_min()
        {
            return make_hip_fp8_e4m3_from_bits(0x01);
        }
    };

    template <>
    class numeric_limits<hip_fp8_e5m2>
    {
    public:
        static constexpr bool is_signed         = true;
        static constexpr bool is_integer        = false;
        static constexpr bool is_specialized    = true;
        static constexpr bool is_exact          = false;
        static constexpr bool has_infinity      = true;
        static constexpr bool has_quiet_NaN     = true;
        static constexpr bool has_signaling_NaN = true;
        static constexpr auto has_denorm        = true;
        static constexpr auto has_denorm_loss   = true;
        static constexpr auto round_style       = round_to_nearest;
        static constexpr bool is_iec559         = false;
        static constexpr bool is_bounded        = true;
        static constexpr bool is_modulo         = false;
        static constexpr int  digits            = 3;
        static constexpr int  digits10          = 0;
        static constexpr int  max_digits10      = 2;
        static constexpr int  radix             = 2;
        static constexpr int  min_exponent      = -13;
        static constexpr int  min_exponent10    = -4;
        static constexpr int  max_exponent      = 16;
        static constexpr int  max_exponent10    = 4;
        static constexpr auto traps             = false;
        static constexpr auto tinyness_before   = false;

        static constexpr hip_fp8_e5m2 min()
        {
            return make_hip_fp8_e5m2_from_bits(0x04);
        }
        static constexpr hip_fp8_e5m2 max()
        {
            return make_hip_fp8_e5m2_from_bits(0x7B);
        }
        static constexpr hip_fp8_e5m2 lowest()
        {
            return make_hip_fp8_e5m2_from_bits(0xFB);
        }
        static constexpr hip_fp8_e5m2 epsilon()
        {
            return make_hip_fp8_e5m2_from_bits(0x34);
        }
        static constexpr hip_fp8_e5m2 round_error()
        {
            return make_hip_fp8_e5m2_from_bits(0x38);
        }
        static constexpr hip_fp8_e5m2 infinity()
        {
            return make_hip_fp8_e5m2_from_bits(0x7C);
        }
        static constexpr hip_fp8_e5m2 quiet_NaN()
        {
            return make_hip_fp8_e5m2_from_bits(0x7F);
        }
        static constexpr hip_fp8_e5m2 signaling_NaN()
        {
            return make_hip_fp8_e5m2_from_bits(0x7F);
        }
        static constexpr hip_fp8_e5m2 denorm_min()
        {
            return make_hip_fp8_e5m2_from_bits(0x01);
        }
    };
}
#endif // ROCWMMA_ENABLE_OCP_HIPRTC
//////////////////////////////////////////
///  FNUZ f8 / bf8 operator overloads  ///
//////////////////////////////////////////

using rocwmma::uint8_t;

#if !defined(ROCWMMA_ENABLE_FNUZ_HIPRTC) || ROCWMMA_ENABLE_FNUZ_HIPRTC
ROCWMMA_FP8_FNUZ_VISIBILITY constexpr inline auto
    make_hip_fp8_e4m3_fnuz_from_bits(__hip_fp8_storage_t bits)
{
    union
    {
        uint8_t           c8;
        hip_fp8_e4m3_fnuz fp8;

    } result{bits};
    return result.fp8;
}

ROCWMMA_FP8_FNUZ_VISIBILITY constexpr inline auto
    make_hip_fp8_e5m2_fnuz_from_bits(__hip_fp8_storage_t bits)
{
    union
    {
        uint8_t           c8;
        hip_fp8_e5m2_fnuz fp8;

    } result{bits};
    return result.fp8;
}

#if !defined(__HIPCC_RTC__)

// Special operator overloading
inline std::ostream& operator<<(std::ostream& os, hip_fp8_e4m3_fnuz a)
{
    return os << float(a);
}

inline std::ostream& operator<<(std::ostream& os, hip_fp8_e5m2_fnuz a)
{
    return os << float(a);
}

#endif // !defined(__HIPCC_RTC__)

// Unary sign inversion
ROCWMMA_FP8_FNUZ_VISIBILITY inline hip_fp8_e4m3_fnuz operator-(hip_fp8_e4m3_fnuz a)
{
    // Case 1: a == 0 -> avoid flipping sign to nan, return 0
    // Case 2: a == nan -> avoid flipping to 0, return nan
    // Else: Flip sign
    return (a.__x == __hip_fp8_storage_t{0}) || (a.__x == __hip_fp8_storage_t{0x80})
               ? a
               : make_hip_fp8_e4m3_fnuz_from_bits(static_cast<uint8_t>(a.__x ^ 0x80));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline hip_fp8_e5m2_fnuz operator-(hip_fp8_e5m2_fnuz a)
{
    // Case 1: a == 0 -> avoid flipping sign to nan, return 0
    // Case 2: a == nan -> avoid flipping to 0, return nan
    // Else: Flip sign
    return (a.__x == __hip_fp8_storage_t{0}) || (a.__x == __hip_fp8_storage_t{0x80})
               ? a
               : make_hip_fp8_e5m2_fnuz_from_bits(static_cast<uint8_t>(a.__x ^ 0x80));
}

// all + operator overloading with mixed types
// mixed types, always converts to f32, does computation in f32, and returns float
ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator+(const float fa, hip_fp8_e4m3_fnuz b)
{
    return (fa + float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator+(hip_fp8_e4m3_fnuz a, const float fb)
{
    return (float(a) + fb);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator+(hip_fp8_e5m2_fnuz a, const float fb)
{
    return (float(a) + fb);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator+(hip_fp8_e4m3_fnuz a, hip_fp8_e5m2_fnuz b)
{
    return (float(a) + float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator+(hip_fp8_e5m2_fnuz a, hip_fp8_e4m3_fnuz b)
{
    return (float(a) + float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline hip_fp8_e4m3_fnuz operator+(hip_fp8_e4m3_fnuz a,
                                                               hip_fp8_e4m3_fnuz b)
{
    return hip_fp8_e4m3_fnuz(float(a) + float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline hip_fp8_e5m2_fnuz operator+(hip_fp8_e5m2_fnuz a,
                                                               hip_fp8_e5m2_fnuz b)
{
    return hip_fp8_e5m2_fnuz(float(a) + float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline hip_fp8_e4m3_fnuz& operator+=(hip_fp8_e4m3_fnuz& a,
                                                                 hip_fp8_e4m3_fnuz  b)
{
    return a = hip_fp8_e4m3_fnuz(float(a) + float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline hip_fp8_e5m2_fnuz& operator+=(hip_fp8_e5m2_fnuz& a,
                                                                 hip_fp8_e5m2_fnuz  b)
{
    return a = hip_fp8_e5m2_fnuz(float(a) + float(b));
}

// all - operator overloading with mixed types
// mixed types, always converts to f32, does computation in f32, and returns float
ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator-(const float fa, hip_fp8_e4m3_fnuz b)
{
    return (fa - float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator-(const float fa, hip_fp8_e5m2_fnuz b)
{
    return (fa - float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator-(hip_fp8_e4m3_fnuz a, const float fb)
{
    return (float(a) - fb);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator-(hip_fp8_e5m2_fnuz a, const float fb)
{
    return (float(a) - fb);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator-(hip_fp8_e4m3_fnuz a, hip_fp8_e5m2_fnuz b)
{
    return (float(a) - float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator-(hip_fp8_e5m2_fnuz a, hip_fp8_e4m3_fnuz b)
{
    return (float(a) - float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline hip_fp8_e4m3_fnuz operator-(hip_fp8_e4m3_fnuz a,
                                                               hip_fp8_e4m3_fnuz b)
{
    return hip_fp8_e4m3_fnuz(float(a) - float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline hip_fp8_e5m2_fnuz operator-(hip_fp8_e5m2_fnuz a,
                                                               hip_fp8_e5m2_fnuz b)
{
    return hip_fp8_e5m2_fnuz(float(a) - float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline hip_fp8_e4m3_fnuz& operator-=(hip_fp8_e4m3_fnuz& a,
                                                                 hip_fp8_e4m3_fnuz  b)
{
    return a = hip_fp8_e4m3_fnuz(float(a) - float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline hip_fp8_e5m2_fnuz& operator-=(hip_fp8_e5m2_fnuz& a,
                                                                 hip_fp8_e5m2_fnuz  b)
{
    return a = hip_fp8_e5m2_fnuz(float(a) - float(b));
}

// overloading multiplication, always returns float,
ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator*(hip_fp8_e4m3_fnuz a, hip_fp8_e4m3_fnuz b)
{
    return float(a) * float(b);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator*(float a, hip_fp8_e4m3_fnuz b)
{
    return (a * float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator*(hip_fp8_e4m3_fnuz a, float b)
{
    return (float(a) * b);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator*(signed int a, hip_fp8_e4m3_fnuz b)
{
    return ((float)a * float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator*(double a, hip_fp8_e4m3_fnuz b)
{
    return ((float)a * float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator*(hip_fp8_e5m2_fnuz a, hip_fp8_e5m2_fnuz b)
{
    return float(a) * float(b);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator*(float a, hip_fp8_e5m2_fnuz b)
{
    return (a * float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator*(hip_fp8_e5m2_fnuz a, float b)
{
    return (float(a) * b);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator*(signed int a, hip_fp8_e5m2_fnuz b)
{
    return ((float)a * float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator*(double a, hip_fp8_e5m2_fnuz b)
{
    return ((float)a * float(b));
}

// overloading for mixed f8 and bf8 types
ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator*(hip_fp8_e4m3_fnuz a, hip_fp8_e5m2_fnuz b)
{
    return float(a) * float(b);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator*(hip_fp8_e5m2_fnuz a, hip_fp8_e4m3_fnuz b)
{
    return float(a) * float(b);
}

// overloading division, always returns float,
ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator/(hip_fp8_e4m3_fnuz a, hip_fp8_e4m3_fnuz b)
{
    return float(a) / float(b);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator/(float a, hip_fp8_e4m3_fnuz b)
{
    return (a / float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator/(hip_fp8_e4m3_fnuz a, float b)
{
    return (float(a) / b);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator/(signed int a, hip_fp8_e4m3_fnuz b)
{
    return ((float)a / float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator/(double a, hip_fp8_e4m3_fnuz b)
{
    return ((float)a / float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator/(hip_fp8_e5m2_fnuz a, hip_fp8_e5m2_fnuz b)
{
    return float(a) / float(b);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator/(float a, hip_fp8_e5m2_fnuz b)
{
    return (a / float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator/(hip_fp8_e5m2_fnuz a, float b)
{
    return (float(a) / b);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator/(signed int a, hip_fp8_e5m2_fnuz b)
{
    return ((float)a / float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator/(double a, hip_fp8_e5m2_fnuz b)
{
    return ((float)a / float(b));
}

// overloading for mixed f8 and bf8 types
ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator/(hip_fp8_e4m3_fnuz a, hip_fp8_e5m2_fnuz b)
{
    return float(a) / float(b);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline float operator/(hip_fp8_e5m2_fnuz a, hip_fp8_e4m3_fnuz b)
{
    return float(a) / float(b);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline hip_fp8_e4m3_fnuz& operator/=(hip_fp8_e4m3_fnuz& a,
                                                                 hip_fp8_e4m3_fnuz  b)
{
    return a = hip_fp8_e4m3_fnuz(float(a) / float(b));
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline hip_fp8_e5m2_fnuz& operator/=(hip_fp8_e5m2_fnuz& a,
                                                                 hip_fp8_e5m2_fnuz  b)
{
    return a = hip_fp8_e5m2_fnuz(float(a) / float(b));
}

// Comparison operators
ROCWMMA_FP8_FNUZ_VISIBILITY inline bool operator==(hip_fp8_e4m3_fnuz a, hip_fp8_e4m3_fnuz b)
{
    return (a.__x == b.__x);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline bool operator==(hip_fp8_e5m2_fnuz a, hip_fp8_e5m2_fnuz b)
{
    return (a.__x == b.__x);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline bool operator!=(hip_fp8_e4m3_fnuz a, hip_fp8_e4m3_fnuz b)
{
    return (a.__x != b.__x);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline bool operator!=(hip_fp8_e5m2_fnuz a, hip_fp8_e5m2_fnuz b)
{
    return (a.__x != b.__x);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline bool operator<(hip_fp8_e4m3_fnuz a, hip_fp8_e4m3_fnuz b)
{
    return float(a) < float(b);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline bool operator<(hip_fp8_e5m2_fnuz a, hip_fp8_e5m2_fnuz b)
{
    return float(a) < float(b);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline bool operator>(hip_fp8_e4m3_fnuz a, hip_fp8_e4m3_fnuz b)
{
    return float(a) > float(b);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline bool operator>(hip_fp8_e5m2_fnuz a, hip_fp8_e5m2_fnuz b)
{
    return float(a) > float(b);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline bool operator<=(hip_fp8_e4m3_fnuz a, hip_fp8_e4m3_fnuz b)
{
    return float(a) <= float(b);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline bool operator<=(hip_fp8_e5m2_fnuz a, hip_fp8_e5m2_fnuz b)
{
    return float(a) <= float(b);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline bool operator>=(hip_fp8_e4m3_fnuz a, hip_fp8_e4m3_fnuz b)
{
    return float(a) >= float(b);
}

ROCWMMA_FP8_FNUZ_VISIBILITY inline bool operator>=(hip_fp8_e5m2_fnuz a, hip_fp8_e5m2_fnuz b)
{
    return float(a) >= float(b);
}

namespace ROCWMMA_NUMERIC_LIMITS_IMPL_NAMESPACE
{
    // Float 8 E4M3
    // @cond
    template <>
    constexpr hip_fp8_e4m3_fnuz numeric_limits<hip_fp8_e4m3_fnuz>::epsilon() noexcept
    {
        return make_hip_fp8_e4m3_fnuz_from_bits(static_cast<uint8_t>(0x28));
    }

    template <>
    constexpr hip_fp8_e4m3_fnuz numeric_limits<hip_fp8_e4m3_fnuz>::infinity() noexcept
    {
        return make_hip_fp8_e4m3_fnuz_from_bits(static_cast<uint8_t>(0x80));
    }

    template <>
    constexpr hip_fp8_e4m3_fnuz numeric_limits<hip_fp8_e4m3_fnuz>::lowest() noexcept
    {
        return make_hip_fp8_e4m3_fnuz_from_bits(static_cast<uint8_t>(0xFF));
    }

    template <>
    constexpr hip_fp8_e4m3_fnuz numeric_limits<hip_fp8_e4m3_fnuz>::max() noexcept
    {
        return make_hip_fp8_e4m3_fnuz_from_bits(static_cast<uint8_t>(0x7F));
    }

    template <>
    constexpr hip_fp8_e4m3_fnuz numeric_limits<hip_fp8_e4m3_fnuz>::min() noexcept
    {
        return make_hip_fp8_e4m3_fnuz_from_bits(static_cast<uint8_t>(0x01));
    }

    template <>
    constexpr hip_fp8_e4m3_fnuz numeric_limits<hip_fp8_e4m3_fnuz>::quiet_NaN() noexcept
    {
        return make_hip_fp8_e4m3_fnuz_from_bits(static_cast<uint8_t>(0x80));
    }

    template <>
    constexpr hip_fp8_e4m3_fnuz numeric_limits<hip_fp8_e4m3_fnuz>::signaling_NaN() noexcept
    {
        return make_hip_fp8_e4m3_fnuz_from_bits(static_cast<uint8_t>(0x80));
    }

    // BFloat8 E5M2
    template <>
    constexpr hip_fp8_e5m2_fnuz numeric_limits<hip_fp8_e5m2_fnuz>::epsilon() noexcept
    {
        return make_hip_fp8_e5m2_fnuz_from_bits(static_cast<uint8_t>(0x38));
    }

    template <>
    constexpr hip_fp8_e5m2_fnuz numeric_limits<hip_fp8_e5m2_fnuz>::infinity() noexcept
    {
        return make_hip_fp8_e5m2_fnuz_from_bits(static_cast<uint8_t>(0x80));
    }

    template <>
    constexpr hip_fp8_e5m2_fnuz numeric_limits<hip_fp8_e5m2_fnuz>::lowest() noexcept
    {
        return make_hip_fp8_e5m2_fnuz_from_bits(static_cast<uint8_t>(0xFF));
    }

    template <>
    constexpr hip_fp8_e5m2_fnuz numeric_limits<hip_fp8_e5m2_fnuz>::max() noexcept
    {
        return make_hip_fp8_e5m2_fnuz_from_bits(static_cast<uint8_t>(0x7F));
    }

    template <>
    constexpr hip_fp8_e5m2_fnuz numeric_limits<hip_fp8_e5m2_fnuz>::min() noexcept
    {
        return make_hip_fp8_e5m2_fnuz_from_bits(static_cast<uint8_t>(0x01));
    }

    template <>
    constexpr hip_fp8_e5m2_fnuz numeric_limits<hip_fp8_e5m2_fnuz>::quiet_NaN() noexcept
    {
        return make_hip_fp8_e5m2_fnuz_from_bits(static_cast<uint8_t>(0x80));
    }

    template <>
    constexpr hip_fp8_e5m2_fnuz numeric_limits<hip_fp8_e5m2_fnuz>::signaling_NaN() noexcept
    {
        return make_hip_fp8_e5m2_fnuz_from_bits(static_cast<uint8_t>(0x80));
    }
    //@endcond

} // namespace ROCWMMA_NUMERIC_LIMITS_IMPL_NAMESPACE
#endif // ROCWMMA_ENABLE_FNUZ_HIPRTC

#endif // ROCWMMA_FLOAT8_HPP
