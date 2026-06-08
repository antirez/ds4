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
#ifndef ROCWMMA_LAYOUT_TRANSFORMS_IMPL_HPP
#define ROCWMMA_LAYOUT_TRANSFORMS_IMPL_HPP

#include "../../transforms.hpp"
#include "../../utility/vector.hpp"

namespace rocwmma
{
    // Definitions for non-interleaved transforms
    namespace Transforms
    {
        template<uint32_t BlockDim, uint32_t MaxVectorWidth, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) soa_to_aos(VecT&& v)
        {
            return Transforms::SoaToAos<BlockDim, MaxVectorWidth>::exec(forward<VecT>(v));
        }

        template<uint32_t BlockDim, uint32_t MaxVectorWidth, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) aos_to_soa(VecT&& v)
        {
            return Transforms::AosToSoa<BlockDim, MaxVectorWidth>::exec(forward<VecT>(v));
        }

    } // namespace Transforms

} // namespace rocwmma

#endif // ROCWMMA_LAYOUT_TRANSFORMS_IMPL_HPP
