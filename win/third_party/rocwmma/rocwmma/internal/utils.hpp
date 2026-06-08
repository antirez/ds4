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
#ifndef ROCWMMA_UTILS_HPP
#define ROCWMMA_UTILS_HPP

#include "api_fwd.hpp"
#include "types.hpp"

#include "utility/apply.hpp"
#include "utility/get.hpp"
#include "vector.hpp"

namespace rocwmma
{
    ///////////////////////////////////////////////////////////////////
    ///                 Vector manipulation identities              ///
    ///                                                             ///
    /// Note: performs static unroll                                ///
    ///////////////////////////////////////////////////////////////////

    // Unary swap only considered in 2d vectors.
    template <typename DataT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto swap(non_native_vector_base<DataT, 2> const& v)
    {
        return non_native_vector_base<DataT, 2>{get<1>(v), get<0>(v)};
    }

    ///////////////////////////////////////////////////////////////////
    ///                 Coord2d utility overrides                   ///
    ///                                                             ///
    /// Note: Coord2d MUST be constexpr compatible                  ///
    ///////////////////////////////////////////////////////////////////
    ROCWMMA_HOST_DEVICE constexpr static inline auto make_coord2d(Coord2dDataT x, Coord2dDataT y)
    {
        return Coord2d{x, y};
    }

    ROCWMMA_HOST_DEVICE constexpr static inline auto swap(Coord2d const& p)
    {
        return Coord2d{get<1>(p), get<0>(p)};
    }

    ROCWMMA_HOST_DEVICE constexpr static inline Coord2d operator*(Coord2d const& lhs,
                                                                  Coord2d const& rhs)
    {
        return make_coord2d(get<0>(lhs) * get<0>(rhs), get<1>(lhs) * get<1>(rhs));
    }

    ROCWMMA_HOST_DEVICE constexpr static inline Coord2d operator+(Coord2d const& lhs,
                                                                  Coord2d const& rhs)
    {
        return make_coord2d(get<0>(lhs) + get<0>(rhs), get<1>(lhs) + get<1>(rhs));
    }
} // namespace rocwmma

#if !defined(__HIPCC_RTC__)
///////////////////////////////////////////////////////////
/////////////  std::pair<T, T> extensions  ////////////////
///////////////////////////////////////////////////////////
namespace std
{
    // Add, sub operators
    template <typename T>
    ROCWMMA_HOST_DEVICE constexpr static inline pair<T, T> operator+(pair<T, T> const& lhs,
                                                                     pair<T, T> const& rhs)
    {
        return make_pair(get<0>(lhs) + get<0>(rhs), get<1>(lhs) + get<1>(rhs));
    }

    template <typename T>
    ROCWMMA_HOST_DEVICE constexpr static inline pair<T, T>& operator+=(pair<T, T>&       lhs,
                                                                       pair<T, T> const& rhs)
    {
        get<0>(lhs) += get<0>(rhs);
        get<1>(lhs) += get<1>(rhs);
        return lhs;
    }

    template <typename T>
    ROCWMMA_HOST_DEVICE constexpr static inline pair<T, T> operator*(pair<T, T> const& lhs,
                                                                     pair<T, T> const& rhs)
    {
        return make_pair(get<0>(lhs) * get<0>(rhs), get<1>(lhs) * get<1>(rhs));
    }

    template <typename T>
    ROCWMMA_HOST_DEVICE constexpr static inline pair<T, T>& operator*=(pair<T, T>&       lhs,
                                                                       pair<T, T> const& rhs)
    {
        get<0>(lhs) *= get<0>(rhs);
        get<1>(lhs) *= get<1>(rhs);
        return lhs;
    }

    template <typename T>
    ROCWMMA_HOST_DEVICE constexpr static inline pair<T, T> operator-(pair<T, T> const& lhs,
                                                                     pair<T, T> const& rhs)
    {
        return make_pair(get<0>(lhs) - get<0>(rhs), get<1>(lhs) - get<1>(rhs));
    }

    template <typename T>
    ROCWMMA_HOST_DEVICE constexpr static inline pair<T, T>& operator-=(pair<T, T>&       lhs,
                                                                       pair<T, T> const& rhs)
    {
        get<0>(lhs) -= get<0>(rhs);
        get<1>(lhs) -= get<1>(rhs);
        return lhs;
    }

} // namespace std

#endif // !defined(__HIPCC_RTC__)

namespace rocwmma
{
    // Helper for string representations of types
    template <typename DataT>
    constexpr const char* dataTypeToString();

    ///////////////////////////////////////////////////////////
    ///////////  rocwmma::dataTypeToString overloads  /////////
    ///////////////////////////////////////////////////////////

    template <>
    constexpr const char* dataTypeToString<float8_t>()
    {
        return "f8";
    }

    template <>
    constexpr const char* dataTypeToString<bfloat8_t>()
    {
        return "bf8";
    }

    template <>
    constexpr const char* dataTypeToString<float8_fnuz_t>()
    {
        return "f8(fnuz)";
    }

    template <>
    constexpr const char* dataTypeToString<bfloat8_fnuz_t>()
    {
        return "bf8(fnuz)";
    }

    template <>
    constexpr const char* dataTypeToString<float16_t>()
    {
        return "f16";
    }

#if !ROCWMMA_NO_HALF
    template <>
    constexpr const char* dataTypeToString<hfloat16_t>()
    {
        return "h16";
    }
#endif // !ROCWMMA_NO_HALF

    template <>
    constexpr const char* dataTypeToString<bfloat16_t>()
    {
        return "bf16";
    }

    template <>
    constexpr const char* dataTypeToString<float32_t>()
    {
        return "f32";
    }

    template <>
    constexpr const char* dataTypeToString<xfloat32_t>()
    {
        return "xf32";
    }

    template <>
    constexpr const char* dataTypeToString<float64_t>()
    {
        return "f64";
    }

    template <>
    constexpr const char* dataTypeToString<int8_t>()
    {
        return "i8";
    }

    template <>
    constexpr const char* dataTypeToString<uint8_t>()
    {
        return "u8";
    }

    template <>
    constexpr const char* dataTypeToString<int16_t>()
    {
        return "i16";
    }

    template <>
    constexpr const char* dataTypeToString<uint16_t>()
    {
        return "u16";
    }

    template <>
    constexpr const char* dataTypeToString<int32_t>()
    {
        return "i32";
    }

    template <>
    constexpr const char* dataTypeToString<uint32_t>()
    {
        return "u32";
    }

    template <>
    constexpr const char* dataTypeToString<uint64_t>()
    {
        return "u64";
    }

    template <>
    constexpr const char* dataTypeToString<row_major>()
    {
        return "T";
    }

    template <>
    constexpr const char* dataTypeToString<col_major>()
    {
        return "N";
    }

} // namespace rocwmma

#endif // ROCWMMA_UTILS_HPP
