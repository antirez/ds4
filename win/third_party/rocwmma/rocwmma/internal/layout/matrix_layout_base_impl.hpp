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
#ifndef ROCWMMA_MATRIX_LAYOUT_BASE_IMPL_HPP
#define ROCWMMA_MATRIX_LAYOUT_BASE_IMPL_HPP

#include "../utility/algorithm.hpp"
#include "../utility/vector.hpp"

namespace rocwmma
{
    namespace MatrixLayout
    {

#define MatrixLayoutBase MatrixLayoutBase<MatrixLayout>

        // Incremental iteration offset
        template <typename MatrixLayout>
        template <typename Coord1d, typename StrideSpace, typename Strides, size_t... Indices>
        ROCWMMA_DEVICE constexpr /* static */ inline auto
            MatrixLayoutBase::incrementalOffset_impl(Coord1d&&     flatCoord,
                                                     StrideSpace&& strideSpace,
                                                     Strides&&     strides,
                                                     index_sequence<Indices...>)
        {
            // This will get invoked for each MatrixLayout stride component to determine
            // iterative stride offset contributions
            auto component_offset = [](auto&& flatCoord,
                                       auto&  flatStride,
                                       auto&& component,
                                       auto&& stride) {
                // Ensure component is a contributor
                if(component > 1)
                {
                    // flatStride it how many iterations between a contribution of given component.
                    // The stride of the LAST iteration of any component will reset the offset back
                    // to the component origin.
                    // Any other component stride will advance the iterative offset.
                    auto next = flatCoord + 1;
                    auto nextOffset
                        = next % (component * flatStride) == 0
                              ? (-(component - 1) * stride) // reset stride
                              : (next % flatStride == 0 ? stride // advance stride
                                                        : decay_t<decltype(stride)>{0}); // NOP

                    // scale the flatStride by current component
                    flatStride *= component;
                    return nextOffset;
                }
                else
                {
                    return decay_t<decltype(stride)>{0};
                }
            };

            // Flat stride begins with neighbor elements
            auto flatStride = decay_t<Coord1d>{1};

            // Accumulate the matrix offset contributions of each component to the next iteration.
            // Note: Build strides in reverse order due to layouts are built in reverse order
            return (component_offset(
                        forward<Coord1d>(flatCoord),
                        forward<decltype(flatStride)&>(flatStride),
                        get<sizeof...(Indices) - 1 - Indices>(forward<StrideSpace>(strideSpace)),
                        get<sizeof...(Indices) - 1 - Indices>(forward<Strides>(strides)))
                    + ...);
        }

        template <typename MatrixLayout>
        template <typename Coord1d, typename StrideSpace, typename Strides>
        ROCWMMA_DEVICE constexpr /* static */ inline auto MatrixLayoutBase::cumulativeOffset_impl(
            Coord1d&& flatCoord, StrideSpace&& strideSpace, Strides&& strides)
        {
            // Get the iterative stride coord in the stride space.
            // Note: Using the reverse inflate because layouts generate strideSpace in reverse order.
            auto strideCoord = inflate_coord_left(forward<Coord1d>(flatCoord),
                                                  forward<StrideSpace>(strideSpace));

            // Calculate the final matrix offset by applying the stride coord on the physical strides
            return to_matrix_space(strideCoord, forward<Strides>(strides));
        }

        template <typename MatrixLayout>
        template <typename Coord1d>
        ROCWMMA_DEVICE constexpr /* static */ inline decltype(auto)
            MatrixLayoutBase::cumulativeOffset(Coord1d&& flatCoord)
        {
            // Apply to MatrixLayout
            constexpr auto StrideSpace = MatrixLayout::strideCounts();
            constexpr auto Strides     = MatrixLayout::strides();

            constexpr auto StridesSpaceSize = VecTraits<decay_t<decltype(StrideSpace)>>::size();
            constexpr auto StridesSize      = VecTraits<decay_t<decltype(Strides)>>::size();

            static_assert(StridesSpaceSize == StridesSize,
                          "Mismatched stride space and stride vector sizes");

            return cumulativeOffset_impl(forward<Coord1d>(flatCoord), StrideSpace, Strides);
        }

        template <typename MatrixLayout>
        template <typename Coord1d>
        ROCWMMA_DEVICE constexpr /* static */ inline decltype(auto)
            MatrixLayoutBase::incrementalOffset(Coord1d&& flatCoord)
        {
            // Initialize the MatrixLayout strides and stride space
            constexpr auto StrideSpace = MatrixLayout::strideCounts();
            constexpr auto Strides     = MatrixLayout::strides();

            constexpr auto StridesSpaceSize = VecTraits<decay_t<decltype(StrideSpace)>>::size();
            constexpr auto StridesSize      = VecTraits<decay_t<decltype(Strides)>>::size();

            static_assert(StridesSpaceSize == StridesSize,
                          "Mismatched stride space and stride vector sizes");

            return incrementalOffset_impl(forward<Coord1d>(flatCoord),
                                          StrideSpace,
                                          Strides,
                                          make_index_sequence<StridesSpaceSize>{});
        }

#undef MatrixLayoutBase

    } // namespace MatrixLayout

} // namespace rocwmma

#endif // ROCWMMA_MATRIX_LAYOUT_BASE_IMPL_HPP
