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
#ifndef ROCWMMA_MATRIX_ORTHO_LAYOUT_HPP
#define ROCWMMA_MATRIX_ORTHO_LAYOUT_HPP

#include "../utility/algorithm.hpp"
#include "../utility/vector.hpp"
#include "layout.hpp"
#include "layout_traits.hpp"
#include "matrix_layout_base.hpp"

namespace rocwmma
{

    // Implementations for the MatrixLayout classes
    namespace MatrixLayout
    {

        template <typename MatrixLayout, typename Enabler = void>
        struct OrthoTraits;

        template <typename MatrixLayout>
        struct OrthoTraits<MatrixLayout, enable_if_t<(bool)MatrixLayout::Traits::BlockDimSegs>>
        {
            // Number of threads per wave
            static constexpr uint32_t WaveSize = MatrixLayout::Traits::WaveSize;

            // Strides (swapped)
            static constexpr uint32_t BlockDimStride_X = MatrixLayout::Traits::BlockDimStride_Y;
            static constexpr uint32_t BlockDimStride_Y = MatrixLayout::Traits::BlockDimStride_X;

            static constexpr uint32_t BlockKStride_X = MatrixLayout::Traits::BlockKStride_Y;
            static constexpr uint32_t BlockKStride_Y = MatrixLayout::Traits::BlockKStride_X;

            static constexpr uint32_t VWStride_X = MatrixLayout::Traits::VWStride_Y;
            static constexpr uint32_t VWStride_Y = MatrixLayout::Traits::VWStride_X;

            // Stride space (same)
            static constexpr uint32_t BlockDimSegs = MatrixLayout::Traits::BlockDimSegs;
            static constexpr uint32_t BlockKSegs   = MatrixLayout::Traits::BlockKSegs;
            static constexpr uint32_t VWSegs       = MatrixLayout::Traits::VWSegs;
        };

        template <typename MatrixLayout>
        struct OrthoTraits<MatrixLayout, enable_if_t<(bool)MatrixLayout::Traits::SplitKSegs>>
        {
            // Number of threads per wave
            static constexpr uint32_t WaveSize = MatrixLayout::Traits::WaveSize;

            // Number of elements each thread will fetch in BlockDim direction
            static constexpr uint32_t DimPerThread = MatrixLayout::Traits::DimPerThread;

            // Number of elements each thread will fetch in BlockK direction
            static constexpr uint32_t KPerThread = MatrixLayout::Traits::KPerThread;

            // Number of elements that each thread is responsible for
            static constexpr uint32_t ElementsPerThread = MatrixLayout::Traits::ElementsPerThread;

            // Swapped strides
            static constexpr uint32_t SplitKStride_X = MatrixLayout::Traits::SplitKStride_Y;
            static constexpr uint32_t SplitKStride_Y = MatrixLayout::Traits::SplitKStride_X;

            static constexpr uint32_t BlockKStride_X = MatrixLayout::Traits::BlockKStride_Y;
            static constexpr uint32_t BlockKStride_Y = MatrixLayout::Traits::BlockKStride_X;

            static constexpr uint32_t VWStride_X = MatrixLayout::Traits::VWStride_Y;
            static constexpr uint32_t VWStride_Y = MatrixLayout::Traits::VWStride_X;

            // Stride Space
            static constexpr uint32_t SplitKSegs = MatrixLayout::Traits::SplitKSegs;
            static constexpr uint32_t BlockKSegs = MatrixLayout::Traits::BlockKSegs;
            static constexpr uint32_t VWSegs     = MatrixLayout::Traits::VWSegs;
        };

        template <typename MatrixLayout>
        struct OrthoImpl : public MatrixLayoutBase<OrthoImpl<MatrixLayout>>
        {
            struct Traits : public OrthoTraits<MatrixLayout>
            {
            };

            ROCWMMA_DEVICE constexpr static inline decltype(auto) strideCounts()
            {
                return MatrixLayout::strideCounts();
            }

            ROCWMMA_DEVICE constexpr static inline decltype(auto) strides()
            {
                constexpr auto t            = MatrixLayout::strides();
                constexpr auto swap_strides = [](auto&&... args) {
                    return make_vector(swap(forward<decay_t<decltype(args)>>(args))...);
                };

                return apply(swap_strides, t);
            }

            ROCWMMA_DEVICE static inline decltype(auto) baseOffset()
            {
                return swap(MatrixLayout::baseOffset());
            }
        };

    } // namespace MatrixLayout

} // namespace rocwmma

#endif // ROCWMMA_MATRIX_ORTHO_LAYOUT_HPP
