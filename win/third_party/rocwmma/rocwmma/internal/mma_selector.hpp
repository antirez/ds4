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
#ifndef ROCWMMA_MMA_SELECTOR_HPP
#define ROCWMMA_MMA_SELECTOR_HPP

#include "mma_traits.hpp"

namespace rocwmma
{

    // Inputs BlockM and BlockN are expected to be fixed (e.g., determined previously by other means).
    // This class will attempt to find appropriate BlockK and map to a backend if it exists.
    template<template<typename, typename, typename, uint32_t, uint32_t, uint32_t> class Mma_impl,
             typename InputTA,
             typename InputTB,
             typename ComputeT,
             uint32_t BlockM,
             uint32_t BlockN,
             uint32_t BlockKTest = 64u> // Current max possible K-value for backend instr (most efficient)
    struct MmaSelector
    {
        private:
        static_assert((BlockKTest & (BlockKTest - 1)) == 0u, "BlockK must be a power of 2");

        // Candidate operation for the current params
        using CandidateOp = Mma_impl<InputTA, InputTB, ComputeT, BlockM, BlockN, BlockKTest>;
        using CandidateTraits = MmaTraits<CandidateOp>;

        public:
        // If the candidate is supported (e.g., a backend implementation exists), then select it.
        // Otherwise, test another smaller BlockK. If no existing implementations, keep the current
        // candidate.
        using SelectedOp = conditional_t<CandidateTraits::is_supported,
                                        CandidateOp,
                                        typename MmaSelector<Mma_impl, InputTA, InputTB, ComputeT, BlockM, BlockN, BlockKTest / 2u>::SelectedOp>;
    };

    template<template<typename, typename, typename, uint32_t, uint32_t, uint32_t> class Mma_impl,
             typename InputTA,
             typename InputTB,
             typename ComputeT,
             uint32_t BlockM,
             uint32_t BlockN>
    struct MmaSelector<Mma_impl, InputTA, InputTB, ComputeT, BlockM, BlockN, 1u>
    {
        // Mma_impl will just be a pass-through if no instruction is found
        using SelectedOp = Mma_impl<InputTA, InputTB, ComputeT, BlockM, BlockN, 1u>;
    };

} // namespace rocwmma

#endif // ROCWMMA_MMA_SELECTOR_HPP
