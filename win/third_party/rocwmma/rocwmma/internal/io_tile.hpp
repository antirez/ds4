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
#ifndef ROCWMMA_IO_TILE_HPP
#define ROCWMMA_IO_TILE_HPP

#include "constants.hpp"
#include "utility/algorithm.hpp"
#include "utility/math.hpp"

namespace rocwmma
{
    //! @struct IOTile
    //! @brief Definition of fragment tiling quantization.
    //! @tparam FragM Fragment M dimension
    //! @tparam FragN Fragment N dimension
    //! @tparam FragK Fragment K dimension
    //! @tparam DataT data type
    template <uint32_t FragM, uint32_t FragN, uint32_t FragK, typename DataT>
    struct IOTile
    {
    private:
        // Note: ALL of the mma implementations have support for minimum 1 packed register for each input A and B.
        // The goal here is to pad out fragments to sizes that are conducive to mma.
        // FragM and FragN should be multiples of base block sizes (hardware defined), and FragK needs to be
        // a product of the lowest common multiple of the minimum registers required to fit the padded size.
        static constexpr uint32_t MinBlockDim          = 16u;
        static constexpr uint32_t MinElementsPerThread = ceil_div(sizeof(float), sizeof(DataT));
        static constexpr uint32_t MinElementsPerFrag
            = Constants::AMDGCN_WAVE_SIZE * MinElementsPerThread;

        static constexpr uint32_t MinKDimM
            = max(MinElementsPerThread, MinElementsPerFrag / next_multiple(FragM, MinBlockDim));
        static constexpr uint32_t MinKDimN
            = max(MinElementsPerThread, MinElementsPerFrag / next_multiple(FragN, MinBlockDim));
        static constexpr uint32_t MinKDim = max(MinKDimM, MinKDimN);

        static constexpr uint32_t PadBlockM = next_multiple(FragM, MinBlockDim);
        static constexpr uint32_t PadBlockN = next_multiple(FragN, MinBlockDim);
        static constexpr uint32_t PadBlockK = next_multiple(FragK, MinKDim);

        static constexpr bool doPadding = (FragM * FragK % MinElementsPerFrag)
                                          || (FragN * FragK % MinElementsPerFrag)
                                          || (FragM * FragN % MinElementsPerFrag);

    public:
        // Apply padding if necessary
        static constexpr uint32_t BlockM = doPadding ? PadBlockM : FragM;
        static constexpr uint32_t BlockN = doPadding ? PadBlockN : FragN;
        static constexpr uint32_t BlockK = doPadding ? PadBlockK : FragK;
    };

} // namespace rocwmma

#endif // ROCWMMA_IO_TILE_HPP
