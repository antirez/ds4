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
#ifndef ROCWMMA_PACK_UTIL_HPP
#define ROCWMMA_PACK_UTIL_HPP

#include "types.hpp"
#include "vector.hpp"

namespace rocwmma
{
    /*
* The following class is intended to define the packing traits
* for particular datatypes. We consider that ROCWMMA uses packed
* registers. The pack ratio is how many registers resulting from
* raw IO are packed together while used in ROCWMMA.
*/

    template <typename DataT>
    struct PackTraits;
    // {
    //     enum : uint32_t
    //     {
    //         PackRatio = N
    //     };

    //     using UnpackedT = ...;
    //     using PackedT   = ...;
    // };

    template <typename DataT>
    struct PackUtil
    {
        using Traits = PackTraits<DataT>;

        using PackedT   = typename Traits::PackedT;
        using UnpackedT = typename Traits::UnpackedT;

        using PaddingB32 = union
        {
            PackedT   packed;
            UnpackedT unpacked[Traits::PackRatio];
        };

    private:
        ///
        /// Internal helpers and static loop unroll
        ///
        template <uint32_t PadIdx = 0u, uint32_t GetIdx = 0u, uint32_t VecSize>
        ROCWMMA_DEVICE static inline decltype(auto) padHelper(VecT<UnpackedT, VecSize> const& v);

        template <uint32_t PadIdx = 0u, uint32_t GetIdx = 0u, uint32_t VecSize>
        ROCWMMA_DEVICE static inline decltype(auto) unpadHelper(VecT<PackedT, VecSize> const& v);

        template <uint32_t VecSize>
        ROCWMMA_DEVICE static inline auto& packHelper(VecT<UnpackedT, VecSize> const& v);

        template <uint32_t VecSize>
        ROCWMMA_DEVICE static inline auto& unpackHelper(VecT<PackedT, VecSize> const& v);

        template <uint32_t PadIdx = 0u, uint32_t VecSize, uint32_t... GetIdx>
        ROCWMMA_DEVICE static inline decltype(auto) pad(VecT<UnpackedT, VecSize> const& v,
                                                        detail::SeqT<GetIdx...>);

        template <uint32_t PadIdx = 0u, uint32_t VecSize, uint32_t... GetIdx>
        ROCWMMA_DEVICE static inline decltype(auto) unpad(VecT<PackedT, VecSize> const& v,
                                                          detail::SeqT<GetIdx...>);

    public:
        ///
        /// Packing from UnpackedT to PackedT, ignoring padding
        ///
        template <uint32_t VecSize>
        ROCWMMA_DEVICE static inline decltype(auto) pack(VecT<UnpackedT, VecSize> const& v);

        template <uint32_t VecSize>
        ROCWMMA_DEVICE static inline decltype(auto) unpack(VecT<PackedT, VecSize> const& v);

        ///
        /// Pad each element from UnpackedT to same sized vector of PackedT
        ///
        template <uint32_t PadIdx = 0u, uint32_t VecSize>
        ROCWMMA_DEVICE static inline decltype(auto) pad(VecT<UnpackedT, VecSize> const& v);

        template <uint32_t PadIdx = 0u, uint32_t VecSize>
        ROCWMMA_DEVICE static inline decltype(auto) unpad(VecT<PackedT, VecSize> const& v);

        ///
        /// Pad UnpackedT elements if necessary for full pack to vector of PackedT
        ///
        template <uint32_t VecSize>
        ROCWMMA_DEVICE static inline decltype(auto) paddedPack(VecT<UnpackedT, VecSize> const& v);

        template <uint32_t UnpaddedSize, uint32_t VecSize>
        ROCWMMA_DEVICE static inline decltype(auto) paddedUnpack(VecT<PackedT, VecSize> const& v);
    };

} // namespace rocwmma

#include "pack_util_impl.hpp"

#endif // ROCWMMA_PACK_UTIL_HPP
