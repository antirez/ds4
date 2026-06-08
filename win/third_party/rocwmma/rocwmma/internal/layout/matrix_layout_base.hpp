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
#ifndef ROCWMMA_MATRIX_LAYOUT_BASE_HPP
#define ROCWMMA_MATRIX_LAYOUT_BASE_HPP

#include "../utility/sequence.hpp"

namespace rocwmma
{
    namespace MatrixLayout
    {
        // All of the matrix layouts are required to provide interface functions to:
        // - cumulativeOffset(idx): using the MatrixLayout strideCounts and strides, determines
        // the matrix coordinate offset for a given iteration idx.
        // - incrementalOffset(): using the MatrixLayout strideCounts and strides, determines
        // the matrix coordinate step offset for a given iteration idx.
        template <typename MatrixLayout>
        struct MatrixLayoutBase
        {
        protected:
            // Incremental iteration offset
            template <typename Coord1d, typename StrideSpace, typename Strides, size_t... Indices>
            ROCWMMA_DEVICE constexpr static inline auto
                incrementalOffset_impl(Coord1d&&     flatCoord,
                                       StrideSpace&& strideSpace,
                                       Strides&&     strides,
                                       index_sequence<Indices...>);

            template <typename Coord1d, typename StrideSpace, typename Strides>
            ROCWMMA_DEVICE constexpr static inline auto cumulativeOffset_impl(
                Coord1d&& flatCoord, StrideSpace&& strideSpace, Strides&& strides);

        public:
            template <typename Coord1d>
            ROCWMMA_DEVICE constexpr static inline decltype(auto)
                cumulativeOffset(Coord1d&& flatCoord);

            // Incremental iteration offset
            template <typename Coord1d>
            ROCWMMA_DEVICE constexpr static inline decltype(auto)
                incrementalOffset(Coord1d&& flatCoord);
        };

    } // namespace MatrixLayout

} // namespace rocwmma

#include "matrix_layout_base_impl.hpp"

#endif // ROCWMMA_MATRIX_LAYOUT_BASE_HPP
