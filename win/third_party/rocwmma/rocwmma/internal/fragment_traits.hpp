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
#ifndef ROCWMMA_FRAGMENT_TRAITS_HPP
#define ROCWMMA_FRAGMENT_TRAITS_HPP

#include "api_fwd.hpp"
#include "fragment_traits_impl.hpp"

namespace rocwmma
{
    template <typename FragT>
    struct fragment_traits;

    template <typename _MatrixT,
              uint32_t _FragM,
              uint32_t _FragN,
              uint32_t _FragK,
              typename _DataT,
              typename _DataLayoutT,
              typename _Scheduler>
    struct fragment_traits<
        fragment<_MatrixT, _FragM, _FragN, _FragK, _DataT, _DataLayoutT, _Scheduler>>
        : public fragment<_MatrixT, _FragM, _FragN, _FragK, _DataT, _DataLayoutT, _Scheduler>::
              Traits
    {
        using MatrixT                   = _MatrixT;
        constexpr static uint32_t FragM = _FragM;
        constexpr static uint32_t FragN = _FragN;
        constexpr static uint32_t FragK = _FragK;
        using DataT                     = _DataT;
        using DataLayoutT               = _DataLayoutT;
        using Scheduler                 = _Scheduler;

        constexpr static bool is_matrix_a    = FragmentTraits_impl::is_matrix_a<MatrixT>::value;
        constexpr static bool is_matrix_b    = FragmentTraits_impl::is_matrix_b<MatrixT>::value;
        constexpr static bool is_accumulator = FragmentTraits_impl::is_accumulator<MatrixT>::value;

        constexpr static bool is_input     = is_matrix_a || is_matrix_b;
        constexpr static bool is_col_major = FragmentTraits_impl::is_col_major<DataLayoutT>::value;
        constexpr static bool is_row_major = FragmentTraits_impl::is_row_major<DataLayoutT>::value;
    };

} // namespace rocwmma

#endif // ROCWMMA_FRAGMENT_TRAITS_HPP
