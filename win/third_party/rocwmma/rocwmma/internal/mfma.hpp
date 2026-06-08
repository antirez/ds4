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
#ifndef ROCWMMA_MFMA_HPP
#define ROCWMMA_MFMA_HPP

#include "mma.hpp"
#include "mma_traits.hpp"
#include "mma_selector.hpp"
#include "mfma_impl.hpp"

namespace rocwmma
{
    // Expose MFMA implementation backend
    template<typename InputTA,
             typename InputTB,
             typename ComputeT,
             uint32_t BlockM,
             uint32_t BlockN,
             uint32_t BlockK>
    using Mfma_impl = detail::amdgcn_mfma<InputTA, InputTB, ComputeT, BlockM, BlockN, BlockK>;

    // Create a backend selector class for mfma backend. Given fixed BlockM and BlockN,
    // will try to find the highest BlockK throughput instruction if it exists.
    template<typename InputTA,
             typename InputTB,
             typename ComputeT,
             uint32_t BlockM,
             uint32_t BlockN,
             uint32_t BlockKTest = 64u> // Current max possible K-value for mfma instr (most efficient)
    struct MfmaSelector : public MmaSelector<Mfma_impl, InputTA, InputTB, ComputeT, BlockM, BlockN, BlockKTest>{};

    // Given a fixed frag size and BlockM and BlockN values, the mfma class will select the mfma backend if it exists
    // and then forward it to the Mma base class driver.
    template <uint32_t FragM,
              uint32_t FragN,
              uint32_t FragK,
              typename InputTA,
              typename InputTB,
              typename ComputeT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK = FragK, // Default K throughput search
              MmaAccumPolicy AccumPolicy = MmaAccumPolicy::ROW_MAJOR>
    struct Mfma 
    : public Mma<FragM,
                 FragN,
                 FragK,
                 typename MfmaSelector<InputTA, InputTB, ComputeT, BlockM, BlockN, BlockK>::SelectedOp,
                 AccumPolicy>
    {
        // Op cache
        using SelectedOp = typename MfmaSelector<InputTA, InputTB, ComputeT, BlockM, BlockN, BlockK>::SelectedOp;

        // Driver interface from base class Mma:
        // template <typename VecTA, typename VecTB, typename VecTC>
        // ROCWMMA_DEVICE static inline decltype(auto) exec(VecTA&& a, VecTB&& b, VecTC& accum);
    };

} // namespace rocwmma

#endif // ROCWMMA_MFMA_HPP
