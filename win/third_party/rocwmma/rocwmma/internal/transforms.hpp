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
#ifndef ROCWMMA_TRANSFORMS_HPP
#define ROCWMMA_TRANSFORMS_HPP

#include "transforms_impl.hpp"
#include "vector.hpp"
#include "vector_iterator.hpp"

namespace rocwmma
{
    namespace Transforms
    {
        template <class Op>
        struct Driver
        {
            using Func = Op;

            template <typename DataT, uint32_t VecSize>
            ROCWMMA_DEVICE static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                auto result = VecT<DataT, VecSize>{};

                auto rIt = makeVectorIterator<Func::VecSize>(v).begin();
                auto wIt = makeVectorIterator<Func::VecSize>(result).begin();

#pragma unroll
                for(uint32_t i = 0; i < decltype(rIt)::range(); i++)
                {
                    *wIt = Func::exec(*rIt);
                    rIt++;
                    wIt++;
                }
                return result;
            }
        };

        template <uint32_t BlockDim, uint32_t MaxVW>
        using AosToSoa = Driver<TransformsImpl::Ops::AosToSoa<BlockDim, MaxVW>>;

        template <uint32_t BlockDim, uint32_t MaxVW>
        using SoaToAos = Driver<TransformsImpl::Ops::SoaToAos<BlockDim, MaxVW>>;

    } // namespace Transforms

} // namespace rocwmma

#endif // ROCWMMA_TRANSFORMS_HPP
