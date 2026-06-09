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
#ifndef ROCWMMA_IO_SHAPE_HPP
#define ROCWMMA_IO_SHAPE_HPP

#include "types.hpp"

namespace rocwmma
{
    /*! \struct IOShape
 *  \brief Definition of fragment dimensions
 *         in specific matrix context.
 *
 * @tparam MatrixT fragment context
 * @tparam BlockM/N/K block dimensions
 */
    template <typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK>
    struct IOShape;

    template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK>
    struct IOShape<matrix_a, BlockM, BlockN, BlockK>
    {
        enum : uint32_t
        {
            BlockHeight = BlockM,
            BlockWidth  = BlockK,

            BlockDim = BlockM,
            KDim     = BlockK,
        };
    };

    template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK>
    struct IOShape<matrix_b, BlockM, BlockN, BlockK>
    {
        enum : uint32_t
        {
            BlockHeight = BlockK,
            BlockWidth  = BlockN,

            BlockDim = BlockN,
            KDim     = BlockK,
        };
    };

    template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK>
    struct IOShape<accumulator, BlockM, BlockN, BlockK>
    {
        enum : uint32_t
        {
            BlockHeight = BlockM,
            BlockWidth  = BlockN,

            BlockDim = BlockN,
            KDim     = BlockM,
        };
    };

} // namespace rocwmma

#endif // ROCWMMA_IO_SHAPE_HPP
