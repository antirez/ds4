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
#ifndef ROCWMMA_LAYOUT_TRANSFORMS_INT_WMMA_IMPL_HPP
#define ROCWMMA_LAYOUT_TRANSFORMS_INT_WMMA_IMPL_HPP

#include "../../config.hpp"
#include "../../swizzle.hpp"
#include "../../pack_util.hpp"
#include "../../vector.hpp"
#include "transforms_int_impl.hpp"
#include "transforms_wmma_impl.hpp"

namespace rocwmma
{

namespace Transforms
{
    template <uint32_t InputBlockSize, typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) soa_int_to_wmma_input_gfx11(VecT&& v)
    {
        if constexpr((bool)ROCWMMA_ARCH_GFX11)
        {
            // Create a double-sized buffer
            using VecTraits = VecTraits<decay_t<VecT>>;
            using DataT = typename VecTraits::DataT;
            constexpr uint32_t VecSize = VecTraits::size();
            using BuffT = typename VecTraits::template VecT<DataT, VecSize * 2u>;
            auto buff = BuffT{};

            // For each set of inputs, duplicate them and store back to the
            // buffer in correct order.
            vector_for_each<InputBlockSize>(
                forward<VecT>(v),
                [](auto&& v, auto&& idx, auto&& buff)
                {
                    using Idx = decay_t<decltype(idx)>;
                    auto wIt = makeVectorIterator<InputBlockSize * 2u>(forward<BuffT&>(buff)).it(Idx::value);
                    *wIt = to_wmma_input_gfx11(v);
                },
                forward<BuffT&>(buff));

            // Return the duplicated buffer
            return buff;
        }
        else
        {
            return forward<VecT>(v);
        }
    }

    template <uint32_t DimPerThread, uint32_t InputBlockSize, typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) aos_int_to_wmma_input_gfx11(VecT&& v)
    {
        if constexpr((bool)ROCWMMA_ARCH_GFX11)
        {
            // Step 1: Transform from aos_int -> soa_int
            // Step 2: Transform to wmma_input.
            return soa_int_to_wmma_input_gfx11<InputBlockSize>(
                aos_int_to_soa_int<DimPerThread>(forward<VecT>(v)));
        }
        else
        {
            return forward<VecT>(v);
        }
    }

    template <uint32_t InputBlockSize, typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) wmma_input_gfx11_to_soa_int(VecT&& v)
    {
        if constexpr((bool)ROCWMMA_ARCH_GFX11)
        {
            // Half-size buffer
            using VecTraits = VecTraits<decay_t<VecT>>;
            using DataT = typename VecTraits::DataT;
            constexpr uint32_t VecSize = VecTraits::size();
            using BuffT = typename VecTraits::template VecT<DataT, VecSize / 2u>;
            auto buff = BuffT{};

            // Iterate over duplicate sets from input and discard duplicates
            vector_for_each<InputBlockSize * 2u>(
                forward<VecT>(v),
                [](auto&& v, auto&& idx, auto&& buff)
                {
                    using Idx = decay_t<decltype(idx)>;
                    auto wIt = makeVectorIterator<InputBlockSize>(forward<BuffT&>(buff)).it(Idx::value);
                    *wIt = from_wmma_input_gfx11(v);
                },
                forward<BuffT&>(buff));

            // Return the buffer
            return buff;
        }
        else
        {
            return forward<VecT>(v);
        }
    }

    template <uint32_t KPerThread, uint32_t InputBlockSize, typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) wmma_input_gfx11_to_aos_int(VecT&& v)
    {
        if constexpr((bool)ROCWMMA_ARCH_GFX11)
        {
            // First need to remove duplicate data
            // Then transform from soa_int to aos_int.
            return soa_int_to_aos_int<KPerThread>(
                wmma_input_gfx11_to_soa_int<InputBlockSize>(forward<VecT>(v)));
        }
        else
        {
            return forward<VecT>(v);
        }
    }

    template <typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) mma_acc_int_a_major_to_wmma_acc_gfx11(VecT&& v)
    {
        if constexpr((bool)ROCWMMA_ARCH_GFX11)
        {
            return Transforms::to_wmma_acc_gfx11(forward<VecT>(v));
        }
        else
        {
            return forward<VecT>(v);
        }
    }

    template <typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) wmma_acc_gfx11_to_mma_acc_int_a_major(VecT&& v)
    {
        if constexpr((bool)ROCWMMA_ARCH_GFX11)
        {
            return Transforms::from_wmma_acc_gfx11(forward<VecT>(v));
        }
        else
        {
            return forward<VecT>(v);
        }
    }

    template <uint32_t AccVecSize, uint32_t MmaBlocksA, uint32_t AccMaxVW, uint32_t MmaDim, typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) soa_int_to_wmma_acc_gfx11(VecT&& v)
    {
        if constexpr((bool)ROCWMMA_ARCH_GFX11)
        {
            // Step1: soa_int             -> mma_acc_int_a_major
            // Step2: mma_acc_int_a_major -> wmma_acc_gfx11
            return mma_acc_int_a_major_to_wmma_acc_gfx11(
                soa_int_to_mma_acc_int_a_major<AccVecSize, MmaBlocksA, AccMaxVW, MmaDim>(forward<VecT>(v)));
        }
        else
        {
            return forward<VecT>(v);
        }
    }

    template <uint32_t AccVecSize, uint32_t MmaBlocksA, uint32_t MmaBlocksB, uint32_t AccMaxVW, uint32_t MmaDim, typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) aos_int_to_wmma_acc_gfx11(VecT&& v)
    {
        if constexpr((bool)ROCWMMA_ARCH_GFX11)
        {
            // Step1: aos_int             -> mma_acc_int_a_major
            // Step2: mma_acc_int_a_major -> wmma_acc_gfx11
            return mma_acc_int_a_major_to_wmma_acc_gfx11(
                aos_int_to_mma_acc_int_a_major<AccVecSize, MmaBlocksA, MmaBlocksB, AccMaxVW, MmaDim>(forward<VecT>(v)));
        }
        else
        {
            return forward<VecT>(v);
        }
    }


    template <uint32_t AccVecSize, uint32_t MmaBlocksB, uint32_t AccMaxVW, uint32_t MmaDim, typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) wmma_acc_gfx11_to_soa_int(VecT&& v)
    {
        if constexpr((bool)ROCWMMA_ARCH_GFX11)
        {
            // Step1: wmma_acc_gfx11      -> mma_acc_int_a_major
            // Step2: mma_acc_int_a_major -> soa_int
            return mma_acc_int_a_major_to_soa_int<AccVecSize, MmaBlocksB, AccMaxVW, MmaDim>(
                wmma_acc_gfx11_to_mma_acc_int_a_major(forward<VecT>(v)));
        }
        else
        {
            return forward<VecT>(v);
        }
    }

    template <uint32_t AccVecSize, uint32_t AccMaxVW, uint32_t MmaDim, typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) wmma_acc_gfx11_to_aos_int(VecT&& v)
    {
        if constexpr((bool)ROCWMMA_ARCH_GFX11)
        {
            // Step1: wmma_acc_gfx11      -> mma_acc_int_a_major
            // Step2: mma_acc_int_a_major -> aos_int
            return mma_acc_int_a_major_to_aos_int<AccVecSize, AccMaxVW, MmaDim>(
                wmma_acc_gfx11_to_mma_acc_int_a_major(forward<VecT>(v)));
        }
        else
        {
            return forward<VecT>(v);
        }
    }

} // namespace Transforms

} // namespace rocwmma

#endif // ROCWMMA_LAYOUT_TRANSFORMS_INT_WMMA_IMPL_HPP
