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
#ifndef ROCWMMA_IO_TRAITS_HPP
#define ROCWMMA_IO_TRAITS_HPP

#include "constants.hpp"
#include "pack_util.hpp"
#include "types.hpp"
#include "utility/math.hpp"

namespace rocwmma
{
    /*
* The following class provides IO meta-data that is used
* to provide static information used in inference and controlling
* certain aspects of ROCWMMA. Static asserts are also performed in this
* class to indicate specific logical issues when implementing IO
* functionality.
*/
    template <uint32_t BlockDim, uint32_t BlockK, typename DataT, uint32_t VectorWidth = 1>
    struct IOTraits
    {
        enum : uint32_t
        {
            // Number of threads to perform I/O operation
            ThreadsPerIO = Constants::AMDGCN_WAVE_SIZE,

            // Total number of elements in a single I/O operation
            ElementsPerIO = ThreadsPerIO * VectorWidth,

            // Number of BlockDim strides per I/O operation
            KPerIO = ceil_div(ElementsPerIO, BlockDim),

            // Total number of elements per for the entire block
            ElementCount = BlockDim * BlockK,

            // Total number of I/O operations needed for the entire block
            IOCount = ceil_div(ElementCount, ElementsPerIO),

            // Per-thread c++ vector storage size required for:
            // Unpacked vector = raw I/O
            // Packed vector = packed raw I/O
            UnpackedSize = ceil_div(ElementCount, ThreadsPerIO),
            PackedSize   = ceil_div((uint32_t)UnpackedSize, (uint32_t)PackTraits<DataT>::PackRatio),

            // Physical number of hardware vregs used to store packed data
            PackedVRegCount = ElementCount * sizeof(DataT) / Constants::AMDGCN_REGISTER_SIZE_BYTES,
            UnpackedVRegCount = PackedVRegCount * PackTraits<DataT>::PackRatio
        };

        static_assert((BlockDim <= ElementsPerIO)
                          ? ((ElementsPerIO % BlockDim) == 0 || (ElementsPerIO % BlockK) == 0)
                          : ((BlockDim % ElementsPerIO) == 0 || (BlockK % ElementsPerIO) == 0),
                      "I/O operation elements not a multiple of BlockDim");
        static_assert((ElementCount % ElementsPerIO) == 0,
                      "I/O element count not divisible into equal operations");
        static_assert((ElementCount % ThreadsPerIO) == 0,
                      "Element count must be divisible by threads per wave");
    };

} // namespace rocwmma

#endif // ROCWMMA_IO_TRAITS_HPP
