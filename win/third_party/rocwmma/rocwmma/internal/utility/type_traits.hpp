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

#ifndef ROCWMMA_UTILITY_TYPE_TRAITS_HPP
#define ROCWMMA_UTILITY_TYPE_TRAITS_HPP

#include "type_traits_impl.hpp"

#if defined(__HIPCC_RTC__)

namespace rocwmma
{
    // Use drop-in replacement
    using detail::add_lvalue_reference;
    using detail::add_lvalue_reference_t;
    using detail::add_pointer;
    using detail::add_pointer_t;
    using detail::add_rvalue_reference;
    using detail::add_rvalue_reference_t;
    using detail::bool_constant;
    using detail::conditional;
    using detail::conditional_t;
    using detail::decay;
    using detail::decay_t;
    using detail::enable_if;
    using detail::enable_if_t;
    using detail::false_type;
    using detail::integral_constant;
    using detail::is_arithmetic;
    using detail::is_arithmetic_v;
    using detail::is_array;
    using detail::is_array_v;
    using detail::is_const;
    using detail::is_const_v;
    using detail::is_convertible;
    using detail::is_convertible_v;
    using detail::is_floating_point;
    using detail::is_floating_point_v;
    using detail::is_function;
    using detail::is_function_v;
    using detail::is_integral;
    using detail::is_integral_v;
    using detail::is_lvalue_reference;
    using detail::is_lvalue_reference_v;
    using detail::is_reference;
    using detail::is_reference_v;
    using detail::is_rvalue_reference;
    using detail::is_rvalue_reference_v;
    using detail::is_same;
    using detail::is_same_v;
    using detail::is_signed;
    using detail::is_signed_v;
    using detail::is_standard_layout;
    using detail::is_trivial;
    using detail::is_void;
    using detail::is_void_v;
    using detail::remove_const;
    using detail::remove_const_t;
    using detail::remove_cv;
    using detail::remove_cv_t;
    using detail::remove_extent;
    using detail::remove_extent_t;
    using detail::remove_reference;
    using detail::remove_reference_t;
    using detail::remove_volatile;
    using detail::remove_volatile_t;
    using detail::true_type;

} // namespace rocwmma

#define ROCWMMA_TYPE_TRAITS_IMPL_NAMESPACE rocwmma::detail

#else

#include <type_traits>
namespace rocwmma
{
    // std implementations
    using std::add_lvalue_reference;
    using std::add_lvalue_reference_t;
    using std::add_pointer;
    using std::add_pointer_t;
    using std::add_rvalue_reference;
    using std::add_rvalue_reference_t;
    using std::bool_constant;
    using std::conditional;
    using std::conditional_t;
    using std::decay;
    using std::decay_t;
    using std::enable_if;
    using std::enable_if_t;
    using std::false_type;
    using std::integral_constant;
    using std::is_arithmetic;
    using std::is_arithmetic_v;
    using std::is_array;
    using std::is_array_v;
    using std::is_const;
    using std::is_const_v;
    using std::is_convertible;
    using std::is_convertible_v;
    using std::is_floating_point;
    using std::is_floating_point_v;
    using std::is_function;
    using std::is_function_v;
    using std::is_integral;
    using std::is_integral_v;
    using std::is_lvalue_reference;
    using std::is_lvalue_reference_v;
    using std::is_reference;
    using std::is_reference_v;
    using std::is_rvalue_reference;
    using std::is_rvalue_reference_v;
    using std::is_same;
    using std::is_same_v;
    using std::is_signed;
    using std::is_signed_v;
    using std::is_standard_layout;
    using std::is_trivial;
    using std::is_void;
    using std::is_void_v;
    using std::remove_const;
    using std::remove_const_t;
    using std::remove_cv;
    using std::remove_cv_t;
    using std::remove_extent;
    using std::remove_extent_t;
    using std::remove_reference;
    using std::remove_reference_t;
    using std::remove_volatile;
    using std::remove_volatile_t;
    using std::true_type;

} // namespace rocwmma

#define ROCWMMA_TYPE_TRAITS_IMPL_NAMESPACE std

#endif // defined(__HIPCC_RTC__) || defined(__clang__)

// Define some convenience traits
namespace rocwmma
{
    template <typename T>
    using enable_if_integral_t = enable_if_t<is_integral<T>{}>;

    template <typename T>
    using enable_if_signed_t = enable_if_t<is_signed<T>{}>;

    template <typename T>
    using enable_if_arithmetic_t = enable_if_t<is_arithmetic<T>{}>;

    // Variadic type traits
    using detail::first_type;
    using detail::first_type_t;

    // Short-form integral constant
    template <uint32_t N>
    using I = detail::integral_constant<uint32_t, N>;

} // namespace rocwmma

#endif // ROCWMMA_UTILITY_TYPE_TRAITS_HPP
