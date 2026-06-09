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
#ifndef ROCWMMA_MMA_IMPL_HPP
#define ROCWMMA_MMA_IMPL_HPP

#include "utility/vector.hpp"

namespace rocwmma
{

#define MMA_TPARAMS_DECL \
template <uint32_t FragM, \
        uint32_t FragN, \
        uint32_t FragK, \
        class MmaImpl, \
        MmaAccumPolicy AccumPolicy>

#define MMA_TPARAMS FragM, FragN, FragK, MmaImpl, AccumPolicy

    MMA_TPARAMS_DECL
    template <typename VecTA, typename VecTB, typename VecTC>
    ROCWMMA_DEVICE inline decltype(auto) Mma<MMA_TPARAMS>::exec_row_major(VecTA&& a, VecTB&& b, VecTC&& accum)
    {
        if constexpr (BlockWiseMmaTraits::is_supported)
        {
            // Iterate through accum blocks in row_major order.
            // FOR each row of C:
            return vector_for_each<AccumRowSize>(
                forward<VecTC>(accum),
                [](auto&& row_c, auto&& row_idx, auto&& input_a, auto&& input_b)
                {
                    // A input is constant per row, for which we can cache
                    constexpr auto rowId = decay_t<decltype(row_idx)>::value;
                    auto itA = makeVectorIterator<BlockSizeA * BlocksK>(forward<VecTA>(input_a)).it(rowId);

                    // FOR each block of C:
                    return vector_for_each<BlockSizeC>(row_c,
                        [](auto&& block_c, auto&& col_idx, auto&& block_a, auto&& input_b)
                        {
                            // B input is constant per col
                            constexpr auto colId = decay_t<decltype(col_idx)>::value;
                            auto itB = makeVectorIterator<BlockSizeB * BlocksK>(forward<VecTB>(input_b)).it(colId);

                            // FOR each K-iteration: invoke Mma
                            return vector_reduce2<BlockSizeA, BlockSizeB>(
                                block_a,
                                *itB,
                                block_c,
                                [](auto&& a, auto&& b, auto&& accum, auto&& idx)
                                {
                                    return BlockWiseMma::exec(a, b, accum);
                                });
                        },
                        *itA,
                        forward<VecTB>(input_b));
                },
                forward<VecTA>(a),
                forward<VecTB>(b));
        }
        else
        {
            // Not supported
            return forward<VecTC>(accum);
        }
    }

    MMA_TPARAMS_DECL
    template <typename VecTA, typename VecTB, typename VecTC>
    ROCWMMA_DEVICE inline decltype(auto) Mma<MMA_TPARAMS>::exec_col_major(VecTA&& a, VecTB&& b, VecTC&& accum)
    {
        if constexpr (BlockWiseMmaTraits::is_supported)
        {
            // Iterate through accum blocks in col_major order.
            // FOR each col of C:
            return vector_for_each<AccumColSize>(
                forward<VecTC>(accum),
                [](auto&& col_c, auto&& col_idx, auto&& input_a, auto&& input_b)
                {
                    // B input is constant per col, for which we can cache
                    constexpr auto colId = decay_t<decltype(col_idx)>::value;
                    auto itB = makeVectorIterator<BlockSizeB * BlocksK>(forward<VecTB>(input_b)).it(colId);

                    // FOR each block of C:
                    return vector_for_each<BlockSizeC>(col_c,
                        [](auto&& block_c, auto&& row_idx, auto&& input_a, auto&& block_b)
                        {
                            // A input is constant per row
                            constexpr auto rowId = decay_t<decltype(row_idx)>::value;
                            auto itA = makeVectorIterator<BlockSizeA * BlocksK>(forward<VecTA>(input_a)).it(rowId);

                            // FOR each K-iteration: invoke Mma
                            return vector_reduce2<BlockSizeA, BlockSizeB>(
                                *itA,
                                block_b,
                                block_c,
                                [](auto&& a, auto&& b, auto&& accum, auto&& idx)
                                {
                                    return BlockWiseMma::exec(a, b, accum);
                                });
                        },
                        forward<VecTA>(input_a),
                        *itB);
                },
                forward<VecTA>(a),
                forward<VecTB>(b));
        }
        else
        {
            // Not supported
            return forward<VecTC>(accum);
        }
    }

    MMA_TPARAMS_DECL
    template <typename VecTA, typename VecTB, typename VecTC>
    ROCWMMA_DEVICE inline decltype(auto) Mma<MMA_TPARAMS>::exec(VecTA&& a, VecTB&& b, VecTC& accum)
    {
        if constexpr (AccumPolicy == MmaAccumPolicy::ROW_MAJOR)
        {
            return exec_row_major(forward<VecTA>(a), forward<VecTB>(b), forward<VecTC>(accum));
        }
        else
        {
            return exec_col_major(forward<VecTA>(a), forward<VecTB>(b), forward<VecTC>(accum));
        }
    }

#undef MMA_TPARAMS_DECL
#undef MMA_TPARAMS

} // namespace rocwmma

#endif // ROCWMMA_MMA_IMPL_HPP
