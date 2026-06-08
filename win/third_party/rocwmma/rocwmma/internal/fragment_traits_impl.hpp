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
#ifndef ROCWMMA_FRAGMENT_TRAITS_IMPL_HPP
#define ROCWMMA_FRAGMENT_TRAITS_IMPL_HPP

#include "api_fwd.hpp"
#include "layout/layout.hpp"

namespace rocwmma
{
    namespace FragmentTraits_impl
    {
        using LayoutTraits_impl::is_col_major;
        using LayoutTraits_impl::is_row_major;

        template <typename MatrixT>
        struct is_matrix_a : false_type
        {
        };

        template <>
        struct is_matrix_a<matrix_a> : true_type
        {
        };

        template <typename MatrixT>
        static constexpr bool is_matrix_a_v = is_matrix_a<MatrixT>::value;

        template <typename MatrixT>
        struct is_matrix_b : false_type
        {
        };

        template <>
        struct is_matrix_b<matrix_b> : true_type
        {
        };

        template <typename MatrixT>
        static constexpr bool is_matrix_b_v = is_matrix_b<MatrixT>::value;

        template <typename MatrixT>
        struct is_accumulator : false_type
        {
        };

        template <>
        struct is_accumulator<accumulator> : true_type
        {
        };

        template <typename MatrixT>
        static constexpr bool is_accumulator_v = is_accumulator<MatrixT>::value;

    } // namespace FragmentTraits_impl

} // namespace rocwmma

#endif // ROCWMMA_FRAGMENT_TRAITS_IMPL_HPP
