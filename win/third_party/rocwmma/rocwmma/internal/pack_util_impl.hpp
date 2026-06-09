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
#ifndef ROCWMMA_PACK_UTIL_IMPL_HPP
#define ROCWMMA_PACK_UTIL_IMPL_HPP

#include "pack_util.hpp"
#include "types.hpp"
#include "utils.hpp"
#include "vector_util.hpp"

namespace rocwmma
{
    template <>
    struct PackTraits<int8_t>
    {
        enum : uint32_t
        {
            PackRatio = 4
        };

        using UnpackedT = int8_t;
        using PackedT   = int32_t;
    };

    template <>
    struct PackTraits<uint8_t>
    {
        enum : uint32_t
        {
            PackRatio = 4
        };

        using UnpackedT = uint8_t;
        using PackedT   = uint32_t;
    };

    template <>
    struct PackTraits<int16_t>
    {
        enum : uint32_t
        {
            PackRatio = 2
        };

        using UnpackedT = int16_t;
        using PackedT   = int32_t;
    };

    template <>
    struct PackTraits<uint16_t>
    {
        enum : uint32_t
        {
            PackRatio = 2
        };

        using UnpackedT = uint16_t;
        using PackedT   = uint32_t;
    };

    template <>
    struct PackTraits<uint32_t>
    {
        enum : uint32_t
        {
            PackRatio = 1
        };

        using UnpackedT = uint32_t;
        using PackedT   = uint32_t;
    };

    template <>
    struct PackTraits<int32_t>
    {
        enum : uint32_t
        {
            PackRatio = 1
        };

        using UnpackedT = int32_t;
        using PackedT   = int32_t;
    };

    template <>
    struct PackTraits<int64_t>
    {
        enum : uint32_t
        {
            PackRatio = 1 // No pack
        };

        using UnpackedT = int64_t;
        using PackedT   = int64_t;
    };

    template <>
    struct PackTraits<float8_t>
    {
        enum : uint32_t
        {
            PackRatio = 4
        };

        using UnpackedT = float8_t;
        using PackedT   = float32_t;
    };

    template <>
    struct PackTraits<bfloat8_t>
    {
        enum : uint32_t
        {
            PackRatio = 4
        };

        using UnpackedT = bfloat8_t;
        using PackedT   = float32_t;
    };

    template <>
    struct PackTraits<float8_fnuz_t>
    {
        enum : uint32_t
        {
            PackRatio = 4
        };

        using UnpackedT = float8_fnuz_t;
        using PackedT   = float32_t;
    };

    template <>
    struct PackTraits<bfloat8_fnuz_t>
    {
        enum : uint32_t
        {
            PackRatio = 4
        };

        using UnpackedT = bfloat8_fnuz_t;
        using PackedT   = float32_t;
    };

    template <>
    struct PackTraits<float16_t>
    {
        enum : uint32_t
        {
            PackRatio = 2 // 2 Elements combine to one
        };

        using UnpackedT = float16_t;
        using PackedT   = float32_t;
    };

#if !ROCWMMA_NO_HALF
    template <>
    struct PackTraits<hfloat16_t>
    {
        enum : uint32_t
        {
            PackRatio = 2 // 2 Elements combine to one
        };

        using UnpackedT = hfloat16_t;
        using PackedT   = float32_t;
    };
#endif // !ROCWMMA_NO_HALF

    template <>
    struct PackTraits<bfloat16_t>
    {
        enum : uint32_t
        {
            PackRatio = 2 // 2 Elements combine to one
        };

        using UnpackedT = bfloat16_t;
        using PackedT   = float32_t;
    };

    template <>
    struct PackTraits<float32_t>
    {
        enum : uint32_t
        {
            PackRatio = 1 // No pack
        };

        using UnpackedT = float32_t;
        using PackedT   = float32_t;
    };

    template <>
    struct PackTraits<xfloat32_t>
    {
        enum : uint32_t
        {
            PackRatio = 1 // No pack
        };

        using UnpackedT = xfloat32_t;
        using PackedT   = float32_t;
    };

    template <>
    struct PackTraits<float64_t>
    {
        enum : uint32_t
        {
            PackRatio = 1 // No pack
        };

        using UnpackedT = float64_t;
        using PackedT   = float64_t;
    };

    template <typename DataT>
    template <uint32_t PadIdx /*= 0u*/, uint32_t GetIdx /*= 0u*/, uint32_t VecSize>
    ROCWMMA_DEVICE /*static*/ inline decltype(auto)
        PackUtil<DataT>::padHelper(VecT<UnpackedT, VecSize> const& v)
    {
        static_assert(PadIdx < Traits::PackRatio, "Invalid pad index selection");
        static_assert(GetIdx < VecSize, "Invalid vector index selection");

        // Case 1: No padding required
        if constexpr(Traits::PackRatio == 1)
        {
            return get<GetIdx>(v);
        }
        // Case 2: Pad out to 32b
        else
        {
            PaddingB32 p;
            p.unpacked[PadIdx] = get<GetIdx>(v);
            return static_cast<PackedT>(p.packed);
        }
    }

    template <typename DataT>
    template <uint32_t PadIdx /*= 0u*/, uint32_t GetIdx /*= 0u*/, uint32_t VecSize>
    ROCWMMA_DEVICE /*static*/ inline decltype(auto)
        PackUtil<DataT>::unpadHelper(VecT<PackedT, VecSize> const& v)
    {
        static_assert(PadIdx < Traits::PackRatio, "Invalid pad index selection");
        static_assert(GetIdx < VecSize, "Invalid vector index selection");

        // Case 1: No padding required
        if constexpr(Traits::PackRatio == 1)
        {
            return get<GetIdx>(v);
        }
        // Case 2: unpad from 32b
        else
        {
            PaddingB32 p;
            p.packed = get<GetIdx>(v);
            return static_cast<UnpackedT>(p.unpacked[PadIdx]);
        }
    }

    template <typename DataT>
    template <uint32_t VecSize>
    ROCWMMA_DEVICE /*static*/ inline auto&
        PackUtil<DataT>::packHelper(VecT<UnpackedT, VecSize> const& v)
    {
        static_assert(VecSize % Traits::PackRatio == 0,
                      "Cannot pack partial b32 vector. Use paddedPack instead.");

        // NOTE: Assumes that there is NO padding...
        using PackedVecT   = VecT<PackedT, VecSize / Traits::PackRatio>;
        using UnpackedVecT = decay_t<decltype(v)>;
        return reinterpret_cast<PackedVecT const&>(v);
    }

    template <typename DataT>
    template <uint32_t VecSize>
    ROCWMMA_DEVICE /*static*/ inline auto&
        PackUtil<DataT>::unpackHelper(VecT<PackedT, VecSize> const& v)
    {
        if constexpr(is_same_v<PackedT, UnpackedT>)
        {
            static_assert(Traits::PackRatio == 1, "Input vector must be packed");
        }

        // NOTE: Assumes that there is NO padding...
        using PackedVecT   = decay_t<decltype(v)>;
        using UnpackedVecT = VecT<UnpackedT, VecSize * Traits::PackRatio>;

        return reinterpret_cast<UnpackedVecT const&>(v);
    }

    template <typename DataT>
    template <uint32_t PadIdx /*= 0u*/, uint32_t VecSize, uint32_t... GetIdx>
    ROCWMMA_DEVICE /*static*/ inline decltype(auto)
        PackUtil<DataT>::pad(VecT<UnpackedT, VecSize> const& v, detail::SeqT<GetIdx...>)
    {
        static_assert(sizeof...(GetIdx) == VecSize, "Unexpected index count");

        // Case 1: No padding required
        if constexpr(Traits::PackRatio == 1)
        {
            return v;
        }
        // Case 2: Padding to 32b for each of the elements
        else
        {
            return VecT<PackedT, VecSize>{padHelper<PadIdx, GetIdx>(v)...};
        }
    }

    template <typename DataT>
    template <uint32_t PadIdx /*= 0u*/, uint32_t VecSize, uint32_t... GetIdx>
    ROCWMMA_DEVICE /*static*/ inline decltype(auto)
        PackUtil<DataT>::unpad(VecT<PackedT, VecSize> const& v, detail::SeqT<GetIdx...>)
    {
        static_assert(sizeof...(GetIdx) == VecSize, "Unexpected index count");

        // Case 1: No padding required
        if constexpr(Traits::PackRatio == 1)
        {
            return v;
        }
        // Case 2: Unpadding from b32 for each of the elements
        else
        {
            return VecT<UnpackedT, VecSize>{unpadHelper<PadIdx, GetIdx>(v)...};
        }
    }

    template <typename DataT>
    template <uint32_t VecSize>
    ROCWMMA_DEVICE /*static*/ inline decltype(auto)
        PackUtil<DataT>::pack(VecT<UnpackedT, VecSize> const& v)
    {
        return packHelper(v);
    }

    template <typename DataT>
    template <uint32_t VecSize>
    ROCWMMA_DEVICE /*static*/ inline decltype(auto)
        PackUtil<DataT>::unpack(VecT<PackedT, VecSize> const& v)
    {
        return unpackHelper(v);
    }

    template <typename DataT>
    template <uint32_t PadIdx /*= 0u*/, uint32_t VecSize>
    ROCWMMA_DEVICE /*static*/ inline decltype(auto)
        PackUtil<DataT>::pad(VecT<UnpackedT, VecSize> const& v)
    {
        return pad<PadIdx>(v, detail::Seq<VecSize>{});
    }

    template <typename DataT>
    template <uint32_t PadIdx /*= 0u*/, uint32_t VecSize>
    ROCWMMA_DEVICE /*static*/ inline decltype(auto)
        PackUtil<DataT>::unpad(VecT<PackedT, VecSize> const& v)
    {
        return unpad<PadIdx>(v, detail::Seq<VecSize>{});
    }

    template <typename DataT>
    template <uint32_t VecSize>
    ROCWMMA_DEVICE /*static*/ inline decltype(auto)
        PackUtil<DataT>::paddedPack(VecT<UnpackedT, VecSize> const& v)
    {
        // No padding
        if constexpr(VecSize % Traits::PackRatio == 0u)
        {
            return packHelper(v);
        }
        // Duplicate the inputs for padding
        else if constexpr((VecSize * 2u) == Traits::PackRatio)
        {
            // Make sure to return by value here as concat produces rval
            return VecT<PackedT, 1u>(packHelper(concat(v, v)));
        }
        // Pad single element data to b32
        else if constexpr(VecSize == 1u)
        {
            return pad(v);
        }
    }

    template <typename DataT>
    template <uint32_t UnpaddedSize, uint32_t VecSize>
    ROCWMMA_DEVICE /*static*/ inline decltype(auto)
        PackUtil<DataT>::paddedUnpack(VecT<PackedT, VecSize> const& v)
    {
        // No padding
        if constexpr(UnpaddedSize % Traits::PackRatio == 0u)
        {
            return unpackHelper(v);
        }
        // Take lower half of vector
        else if constexpr((UnpaddedSize * 2u) == Traits::PackRatio)
        {
            return extractLo(unpackHelper(v));
        }
        // Pad single element data to b32
        else if constexpr(UnpaddedSize == 1u)
        {
            return unpad(v);
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_PACK_UTIL_IMPL_HPP
