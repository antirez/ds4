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
#ifndef ROCWMMA_LAYOUT_TRANSFORMS_WMMA_IMPL_HPP
#define ROCWMMA_LAYOUT_TRANSFORMS_WMMA_IMPL_HPP

#include "../../config.hpp"
#include "../../swizzle.hpp"
#include "../../pack_util.hpp"
#include "../../vector.hpp"
#include "transforms_int_impl.hpp"

namespace rocwmma
{

namespace Transforms
{
    template<typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) to_wmma_input_gfx11(VecT&& v)
    {
        if constexpr((bool)ROCWMMA_ARCH_GFX11)
        {
            // Swap + concat
            // v is unpacked
            using VecTraits = VecTraits<decay_t<VecT>>;
            using PackUtil = PackUtil<typename VecTraits::DataT>;

            // Swap upper / lower 16's and then concatenate them
            // to make sure we have each K value in each half.
            // GFX11 wmma layout quirk needs the duplication.
            auto packed = PackUtil::pack(v);
            auto swapped = Swizzle::Swap16::exec(packed);
            auto result = PackUtil::unpack(concat(packed, swapped));
            return result; // Return by copy
        }
        else
        {
            return forward<VecT>(v);
        }
    }

    template<typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) from_wmma_input_gfx11(VecT&& v)
    {
        if constexpr((bool)ROCWMMA_ARCH_GFX11)
        {
            // Discard the swapped dups
            return extractLo(v);
        }
        else
        {
            return forward<VecT>(v);
        }
    }

    template<typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) to_wmma_acc_gfx11(VecT&& v)
    {
        if constexpr((bool)ROCWMMA_ARCH_GFX11)
        {
            using VecTraits = VecTraits<decay_t<VecT>>;

            // pad to wmma accumulator on gfx11.
            // f16 -> padded to f32, with data in lower 16
            // f32 -> nop
            using PackUtil = PackUtil<typename VecTraits::DataT>;
            auto accum = PackUtil::unpack(PackUtil::template pad<>(v));
            return accum; // Return by copy
        }
        else
        {
            return forward<VecT>(v);
        }
    }

    template<typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) from_wmma_acc_gfx11(VecT&& v)
    {
        if constexpr((bool)ROCWMMA_ARCH_GFX11)
        {
            using VecTraits = VecTraits<decay_t<VecT>>;

            // unpad from wmma accumulator on gfx11.
            // f16 -> padded to f32 in lower 16
            // f32 -> nop
            using PackUtil = PackUtil<typename VecTraits::DataT>;
            return PackUtil::template unpad<>(PackUtil::pack(v));
        }
        else
        {
            return forward<VecT>(v);
        }
    }

    template <typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) soa_to_wmma_input_gfx11(VecT&& v)
    {
        if constexpr((bool)ROCWMMA_ARCH_GFX11)
        {
            return to_wmma_input_gfx11(forward<VecT>(v));
        }
        else
        {
            return forward<VecT>(v);
        }
    }

    template<uint32_t BlockDim, uint32_t MaxVectorWidth, typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) aos_to_wmma_input_gfx11(VecT&& v)
    {
        if constexpr((bool)ROCWMMA_ARCH_GFX11)
        {
            // Step 1: Transform from aos_int -> soa_int
            // Step 2: Transform to wmma_input.
            return to_wmma_input_gfx11(aos_to_soa<BlockDim, MaxVectorWidth>(forward<VecT>(v)));
        }
        else
        {
            return forward<VecT>(v);
        }
    }

    template <typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) wmma_input_gfx11_to_soa(VecT&& v)
    {
        if constexpr((bool)ROCWMMA_ARCH_GFX11)
        {
            return from_wmma_acc_gfx11(forward<VecT>(v));
        }
        else
        {
            return forward<VecT>(v);
        }
    }

    template<uint32_t BlockDim, uint32_t MaxVectorWidth, typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) wmma_input_gfx11_to_aos(VecT&& v)
    {
        if constexpr((bool)ROCWMMA_ARCH_GFX11)
        {
            // Step 1: Transform from wmma_input_gfx11 -> soa
            // Step 2: Transform from soa -> aos
            return soa_to_aos<BlockDim, MaxVectorWidth>(from_wmma_input_gfx11(forward<VecT>(v)));
        }
        else
        {
            return forward<VecT>(v);
        }
    }

    template <typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) soa_to_wmma_acc_gfx11(VecT&& v)
    {
        if constexpr((bool)ROCWMMA_ARCH_GFX11)
        {
            return to_wmma_acc_gfx11(forward<VecT>(v));
        }
        else
        {
            return forward<VecT>(v);
        }
    }

    template<uint32_t BlockDim, uint32_t MaxVectorWidth, typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) aos_to_wmma_acc_gfx11(VecT&& v)
    {
        if constexpr((bool)ROCWMMA_ARCH_GFX11)
        {
            // Step 1: Transform from aos -> soa
            // Then transform to wmma_input.
            return to_wmma_acc_gfx11(aos_to_soa<BlockDim, MaxVectorWidth>(forward<VecT>(v)));
        }
        else
        {
            return forward<VecT>(v);
        }
    }

    template <typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) wmma_acc_gfx11_to_soa(VecT&& v)
    {
        if constexpr((bool)ROCWMMA_ARCH_GFX11)
        {
            return from_wmma_acc_gfx11(forward<VecT>(v));
        }
        else
        {
            return forward<VecT>(v);
        }
    }

    template<uint32_t BlockDim, uint32_t MaxVectorWidth, typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) wmma_acc_gfx11_to_aos(VecT&& v)
    {
        if constexpr((bool)ROCWMMA_ARCH_GFX11)
        {
            // First need to transform from aos_int -> soa_int
            // Then transform to wmma_input.
            return soa_to_aos<BlockDim, MaxVectorWidth>(from_wmma_acc_gfx11(forward<VecT>(v)));
        }
        else
        {
            return forward<VecT>(v);
        }
    }

} // namespace Transforms

} // namespace rocwmma

#endif // ROCWMMA_LAYOUT_TRANSFORMS_WMMA_IMPL_HPP
