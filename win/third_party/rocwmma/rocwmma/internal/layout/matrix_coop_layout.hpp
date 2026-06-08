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
#ifndef ROCWMMA_MATRIX_COOP_LAYOUT_HPP
#define ROCWMMA_MATRIX_COOP_LAYOUT_HPP

#include "layout.hpp"
#include "layout_traits.hpp"
#include "matrix_layout_base.hpp"

namespace rocwmma
{
    // Implementations for the MatrixLayout classes
    namespace MatrixLayout
    {
        // This class is a wrapper to basic matrix layouts which provides modified
        // stride counts and offsets based on how many waves are participating in the
        // original layout.
        // The class provides 2 ways to indicate a WaveCount:
        // - Via non-zero class template arg
        // - Via functional argument in the interface, which will override the static template arg.
        // The class also provides additional functional arguments to take in the current
        // wave id, where appropriate to calculate proper offsets.
        // Note: We should prefer the templated WaveCount where possible to leverage compile-time
        // optimizations as much as possible.
        template <typename MatrixLayout, uint32_t WaveCount>
        struct MatrixCoopLayout
            : private MatrixLayoutBase<MatrixCoopLayout<MatrixLayout, WaveCount>>
        {
        private:
            using MatrixLayoutTraits = layout_traits<MatrixLayout>;
            using Base               = MatrixLayoutBase<MatrixCoopLayout<MatrixLayout, WaveCount>>;

            // Note: In a multi-wave context, layout iterative spaces can be split
            // between waves, with some restrictions. Certain types of layouts
            // (e.g., interleaved) have different splitting requirements such that
            // transpose transforms are still valid in a cooperative mode.
            // Rules of thumb:
            // - Interleaved layouts can only split the largest stride
            // - Non-interleaved layouts can split all but the smallest stride
            ROCWMMA_HOST_DEVICE constexpr static auto fixedSpace();

            ROCWMMA_HOST_DEVICE constexpr static auto splittableSpace();

            // Finds the iterative sub-space for each split
            ROCWMMA_DEVICE constexpr static auto calcSplitStrides(uint32_t splitCount);

        public:
            // Finds a suitable power of 2 divisor for equal distribution among waves
            ROCWMMA_HOST_DEVICE constexpr static uint32_t calcMaxSplits(uint32_t splitCount);

            // Overrides for MatrixLayout interface
            ROCWMMA_DEVICE constexpr static inline auto strideCounts(const uint32_t waveCount
                                                                     = WaveCount);

            ROCWMMA_DEVICE constexpr static inline auto strides();

            ROCWMMA_DEVICE constexpr static inline auto baseOffset(const int waveIndex,
                                                                   const int waveCount = WaveCount);

            // Overrides for MatrixLayoutBase interface
            template <typename Coord1d>
            ROCWMMA_DEVICE constexpr static inline decltype(auto)
                cumulativeOffset(Coord1d&& flatCoord, const int waveCount = WaveCount);

            // Incremental iteration offset
            template <typename Coord1d>
            ROCWMMA_DEVICE constexpr static inline decltype(auto)
                incrementalOffset(Coord1d&& flatCoord, const int waveCount = WaveCount);
        };

    } // namespace MatrixLayout

} // namespace rocwmma

#include "matrix_coop_layout_impl.hpp"
#include "matrix_coop_layout_traits_impl.hpp"

#endif // ROCWMMA_MATRIX_COOP_LAYOUT_HPP
