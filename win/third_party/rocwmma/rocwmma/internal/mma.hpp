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
#ifndef ROCWMMA_MMA_HPP
#define ROCWMMA_MMA_HPP

#include "config.hpp"
#include "vector.hpp"

namespace rocwmma
{
    template<typename MmaImpl>
    struct MmaTraits;

    enum struct MmaAccumPolicy : uint32_t
    {
        ROW_MAJOR = 0u,
        COL_MAJOR = 1u,
    };

    /*! \class Mma
    *   \brief Driver for the wave-tile Mma operation. Given a backend block-wise mma implementation (e.g., mfma or wmma),
    * this class performs block-wise decomposition to matrix-multiply input fragments of (A: FragM x FragK) x (B: FragK x FragN)
    * and accumulates results into output fragment (C: FragM x FragN).
    *  @tparam FragM Mma fragment M dimension
    *  @tparam FragN Mma fragment K dimension
    *  @tparam FragK Mma fragment M dimension
    *  @tparam MmaImpl The backend wrapper class that will perform block-wise mma op (e.g., mfma or wmma)
    *  @tparam MmaAccumPolicy The block order of the accumulation registers (row major or col major block order)
    */
    template <uint32_t FragM,
            uint32_t FragN,
            uint32_t FragK,
            class MmaImpl,
            MmaAccumPolicy AccumPolicy = MmaAccumPolicy::ROW_MAJOR>
    struct Mma
    {
        using BlockWiseMma = MmaImpl;
        using BlockWiseMmaTraits = MmaTraits<BlockWiseMma>;

        // Block dimensions
        constexpr static uint32_t BlockM = BlockWiseMmaTraits::BlockM;
        constexpr static uint32_t BlockN = BlockWiseMmaTraits::BlockN;
        constexpr static uint32_t BlockK = BlockWiseMmaTraits::BlockK;

        // Block vector dimensions (packed registers as input to impl)
        constexpr static uint32_t BlockSizeA = BlockWiseMmaTraits::BlockSizeA;
        constexpr static uint32_t BlockSizeB = BlockWiseMmaTraits::BlockSizeB;
        constexpr static uint32_t BlockSizeC = BlockWiseMmaTraits::BlockSizeC;

        // Block counts
        constexpr static uint32_t BlocksM = FragM / BlockM;
        constexpr static uint32_t BlocksN = FragN / BlockN;
        constexpr static uint32_t BlocksK = FragK / BlockK;
        constexpr static uint32_t BlocksC = BlocksM * BlocksN;

        // Register grouping size for accum
        constexpr static uint32_t AccumRowSize = BlocksN * BlockSizeC;
        constexpr static uint32_t AccumColSize = BlocksM * BlockSizeC;

        // Sanity checks
        static_assert(FragM >= BlockM, "FragM must be larger than BlockM");
        static_assert(FragN >= BlockN, "FragN must be larger than BlockN");
        static_assert(FragK >= BlockK, "FragK must be larger than BlockK");
        static_assert(FragM % BlockM == 0u, "FragM must be a multiple of BlockM");
        static_assert(FragN % BlockN == 0u, "FragN must be a multiple of BlockN");
        static_assert(FragK % BlockK == 0u, "FragK must be a multiple of BlockK");

    private:

        template <typename VecTA, typename VecTB, typename VecTC>
        ROCWMMA_DEVICE static inline decltype(auto) exec_row_major(VecTA&& a, VecTB&& b, VecTC&& accum);

        template <typename VecTA, typename VecTB, typename VecTC>
        ROCWMMA_DEVICE static inline decltype(auto) exec_col_major(VecTA&& a, VecTB&& b, VecTC&& accum);

    public:

        template <typename VecTA, typename VecTB, typename VecTC>
        ROCWMMA_DEVICE static inline decltype(auto) exec(VecTA&& a, VecTB&& b, VecTC& accum);
    };

} // namespace rocwmma

#include "mma_impl.hpp"

#endif // ROCWMMA_MMA_HPP
