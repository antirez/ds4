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
#ifndef ROCWMMA_MMA_TRAITS_HPP
#define ROCWMMA_MMA_TRAITS_HPP

#include "mma_traits_impl.hpp"

namespace rocwmma
{
    template<typename MmaOp>
    struct MmaTraits : public MmaTraits_impl::MmaTraits<MmaOp>
    {
        // The following traits will have to be filled
        // by the mfma and the wmma backends

        // using Impl = amdgcn_mfma/wmma<...>;

        // Operand types
        // using InputTA = ...;
        // using InputTB = ...;
        // using ComputeT = ...;

        // Raw input / output types
        // using ARegsT = ...;
        // using BRegsT = ...;
        // using CRegsT = ...;
        // using DRegsT = ...;

        // Geometric block sizes
        // constexpr static uint32_t BlockM = ...;
        // constexpr static uint32_t BlockN = ...;
        // constexpr static uint32_t BlockK = ...;

        // Vector sizes per block (packed)
        // constexpr static uint32_t BlockSizeA = ...;
        // constexpr static uint32_t BlockSizeB = ...;
        // constexpr static uint32_t BlockSizeC = ...;

        // Backend flags
        // constexpr static bool is_wmma = ...;
        // constexpr static bool is_mfma = ...;
        // constexpr static bool is_supported = ...;
    };

} // namespace rocwmma

#endif // ROCWMMA_MMA_TRAITS_HPP
