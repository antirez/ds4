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
#ifndef ROCWMMA_MATRIX_COOP_LAYOUT_IMPL_HPP
#define ROCWMMA_MATRIX_COOP_LAYOUT_IMPL_HPP

#include "../utility/algorithm.hpp"
#include "../utility/sequence.hpp"
#include "../utility/vector.hpp"
#include "layout.hpp"
#include "layout_traits.hpp"
#include "matrix_layout_base_impl.hpp"

namespace rocwmma
{
    // Implementations for the MatrixLayout classes
    namespace MatrixLayout
    {

// Clean up function definitions
#define MatrixCoopLayout MatrixCoopLayout<MatrixLayout, WaveCount>

        // Note: In a multi-wave context, layout iterative spaces can be split
        // between waves, with some restrictions. Certain types of layouts
        // (e.g., interleaved) have different splitting requirements such that
        // transpose transforms are still valid in a cooperative mode.
        // Rules of thumb:
        // - Interleaved layouts can only split the largest stride
        // - Non-interleaved layouts can split all but the smallest stride
        template <typename MatrixLayout, uint32_t WaveCount>
        ROCWMMA_HOST_DEVICE constexpr /* static */ auto MatrixCoopLayout::fixedSpace()
        {
            constexpr auto StrideSpace = MatrixLayout::strideCounts();

            if constexpr(MatrixLayoutTraits::is_interleaved)
            {
                return pop_front(StrideSpace);
            }
            else
            {
                return make_vector(get_last(StrideSpace));
            }
        }

        template <typename MatrixLayout, uint32_t WaveCount>
        ROCWMMA_HOST_DEVICE constexpr /* static */ auto MatrixCoopLayout::splittableSpace()
        {
            constexpr auto StrideSpace = MatrixLayout::strideCounts();

            if constexpr(MatrixLayoutTraits::is_interleaved)
            {
                return make_vector(get_first(StrideSpace));
            }
            else
            {
                return pop_back(StrideSpace);
            }
        }

        template <typename MatrixLayout, uint32_t WaveCount>
        // Finds a suitable power of 2 divisor for equal distribution among waves
        ROCWMMA_HOST_DEVICE constexpr /* static */ uint32_t
            MatrixCoopLayout::calcMaxSplits(uint32_t splitCount)
        {
            constexpr auto SplittableWorkItems = reduce_mult(splittableSpace());

            return (SplittableWorkItems % splitCount == 0 ? splitCount
                                                          : calcMaxSplits(splitCount / 2u));
        }

        template <typename MatrixLayout, uint32_t WaveCount>
        // Finds the iterative sub-space for each split
        ROCWMMA_DEVICE constexpr /* static */ auto
            MatrixCoopLayout::calcSplitStrides(uint32_t splitCount)
        {
            // Separate stride space
            constexpr auto StrideSpaceF = fixedSpace();
            constexpr auto StrideSpaceS = splittableSpace();

            // Divide splittable space
            constexpr auto SplittableWorkItems = reduce_mult(StrideSpaceS);
            auto           workItemsPerSplit   = max(SplittableWorkItems / splitCount, 1u);

            // Assemble iterative space for each split
            auto strideSpaceR = inflate_coord_left(workItemsPerSplit - 1u, StrideSpaceS) + 1u;
            return vector_cat(strideSpaceR, StrideSpaceF);
        }

        template <typename MatrixLayout, uint32_t WaveCount>
        ROCWMMA_DEVICE constexpr /* static */ inline auto
            MatrixCoopLayout::strideCounts(const uint32_t waveCount /* = WaveCount */)
        {
            // Note: MaxWaves is the actual maximum amount of waves that can participate.
            uint32_t maxWaveCount = calcMaxSplits(waveCount);
            return calcSplitStrides(maxWaveCount);
        }

        template <typename MatrixLayout, uint32_t WaveCount>
        ROCWMMA_DEVICE constexpr /* static */ inline auto MatrixCoopLayout::strides()
        {
            return MatrixLayout::strides();
        }

        template <typename MatrixLayout, uint32_t WaveCount>
        ROCWMMA_DEVICE constexpr /* static */ inline auto
            MatrixCoopLayout::baseOffset(const int waveIndex, const int waveCount /* = WaveCount */)
        {
            constexpr auto StrideSpace = MatrixLayout::strideCounts();
            constexpr auto Strides     = MatrixLayout::strides();

            // Find wave offset in complete fragment
            auto workItemsPerWave  = reduce_mult(strideCounts(waveCount));
            auto currentWaveOffset = reduce_add(
                inflate_coord_left(waveIndex * workItemsPerWave, StrideSpace) * Strides);

            // Combine base and wave coordinate offset
            return MatrixLayout::baseOffset() + currentWaveOffset;
        }

        template <typename MatrixLayout, uint32_t WaveCount>
        template <typename Coord1d>
        ROCWMMA_DEVICE constexpr /* static */ inline decltype(auto)
            MatrixCoopLayout::cumulativeOffset(Coord1d&& flatCoord,
                                               const int waveCount /* = WaveCount */)
        {
            // Forward the waveCount arg
            auto           strideSpace = strideCounts(waveCount);
            constexpr auto Strides     = strides();

            constexpr auto StridesSpaceSize = VecTraits<decay_t<decltype(strideSpace)>>::size();
            constexpr auto StridesSize      = VecTraits<decay_t<decltype(Strides)>>::size();

            static_assert(StridesSpaceSize == StridesSize,
                          "Mismatched stride space and stride vector sizes");

            return Base::cumulativeOffset_impl(forward<Coord1d>(flatCoord), strideSpace, Strides);
        }

        template <typename MatrixLayout, uint32_t WaveCount>
        // Incremental iteration offset
        template <typename Coord1d>
        ROCWMMA_DEVICE constexpr /* static */ inline decltype(auto)
            MatrixCoopLayout::incrementalOffset(Coord1d&& flatCoord,
                                                const int waveCount /* = WaveCount */)
        {
            // Forward the waveCount arg
            auto           strideSpace = strideCounts(waveCount);
            constexpr auto Strides     = strides();

            constexpr auto StridesSpaceSize = VecTraits<decay_t<decltype(strideSpace)>>::size();
            constexpr auto StridesSize      = VecTraits<decay_t<decltype(Strides)>>::size();

            static_assert(StridesSpaceSize == StridesSize,
                          "Mismatched stride space and strides sizes");

            return Base::incrementalOffset_impl(forward<Coord1d>(flatCoord),
                                                strideSpace,
                                                Strides,
                                                make_index_sequence<StridesSize>{});
        }

#undef MatrixCoopLayout

    } // namespace MatrixLayout

} // namespace rocwmma

#endif // ROCWMMA_MATRIX_COOP_LAYOUT_IMPL_HPP
