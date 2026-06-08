/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCWMMA_VECTOR_UTIL_IMPL_HPP
#define ROCWMMA_VECTOR_UTIL_IMPL_HPP

#include "blend.hpp"
#include "types.hpp"
#include "utility/algorithm.hpp"
#include "utility/type_traits.hpp"
#include "utility/vector.hpp"

namespace rocwmma
{
    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto concat(VecT<DataT, VecSize> const& v0,
                                                       VecT<DataT, VecSize> const& v1)
    {
        auto concat = [](auto&& idx, auto&& v0, auto&& v1) {
            constexpr auto Index = decay_t<decltype(idx)>::value;
            return (Index < VecSize) ? get<Index>(v0) : get<Index - VecSize>(v1);
        };

        return vector_generator<DataT, VecSize * 2u>()(concat, v0, v1);
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto extractLo(VecT<DataT, VecSize> const& v)
    {
        if constexpr(VecSize > 1)
        {
            auto lo = [](auto&& idx, auto&& v) {
                constexpr auto Index = decay_t<decltype(idx)>::value;
                return get<Index>(v);
            };

            return vector_generator<DataT, VecSize / 2u>()(lo, v);
        }
        // Self-forwarding case
        else
        {
            return v;
        }
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto extractHi(VecT<DataT, VecSize> const& v)
    {
        if constexpr(VecSize > 1)
        {
            auto hi = [](auto&& idx, auto&& v) {
                constexpr auto Index = decay_t<decltype(idx)>::value;
                return get<Index + VecSize / 2u>(v);
            };

            return vector_generator<DataT, VecSize / 2u>()(hi, v);
        }
        else
        {
            return v;
        }
    }

} // namespace rocwmma

#include "pack_util.hpp"

namespace rocwmma
{
    template <typename DataT, uint32_t VecSize>
    ROCWMMA_HOST_DEVICE constexpr static inline auto extractEven(VecT<DataT, VecSize> const& v)
    {
        using PackUtil   = PackUtil<DataT>;
        using PackTraits = typename PackUtil::Traits;

        // Special case: Sub-dword data sizes with minimum 2 packed vectors
        // Extract even elements only.
        constexpr auto ElementSize   = sizeof(DataT);
        constexpr auto PackedVecSize = VecSize / PackTraits::PackRatio;
        if constexpr(ElementSize < 4u && PackedVecSize >= 2u)
        {
            auto evens = [](auto&& idx, auto&& v) {
                constexpr auto Index = decay_t<decltype(idx)>::value;
                return (ElementSize == 2u) ? Blend::ExtractWordEven::exec(get<Index * 2u>(v),
                                                                          get<Index * 2u + 1u>(v))
                                           : Blend::ExtractByteEven::exec(get<Index * 2u>(v),
                                                                          get<Index * 2u + 1u>(v));
            };

            // Pack, extract and unpack
            using PackedT = typename PackTraits::PackedT;
            auto packed   = PackUtil::paddedPack(v);
            auto result   = vector_generator<PackedT, PackedVecSize / 2u>()(evens, packed);
            return PackUtil::template paddedUnpack<VecSize / 2u>(result);
        }
        // General case:
        // Re-arrangement of dword+ data sizes isn't super costly and can
        // be achieved with a simple static shuffle.
        else if constexpr(VecSize > 1)
        {
            auto evens = [](auto&& idx, auto&& v) {
                constexpr auto Index = decay_t<decltype(idx)>::value;
                return get<Index * 2>(v);
            };

            return vector_generator<DataT, VecSize / 2u>()(evens, v);
        }
        // Forwarding case: vector size is 1
        else
        {
            return v;
        }
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto extractOdd(VecT<DataT, VecSize> const& v)
    {
        using PackUtil   = PackUtil<DataT>;
        using PackTraits = typename PackUtil::Traits;

        // Special case: Sub-dword data sizes with minimum 2 packed vectors
        // Extract odd elements only.
        constexpr auto ElementSize   = sizeof(DataT);
        constexpr auto PackedVecSize = VecSize / PackTraits::PackRatio;
        if constexpr(ElementSize < 4u && PackedVecSize >= 2u)
        {
            auto odds = [](auto&& idx, auto&& v) {
                constexpr auto Index = decay_t<decltype(idx)>::value;
                return (ElementSize == 2u) ? Blend::ExtractWordOdd::exec(get<Index * 2u>(v),
                                                                         get<Index * 2u + 1u>(v))
                                           : Blend::ExtractByteOdd::exec(get<Index * 2u>(v),
                                                                         get<Index * 2u + 1u>(v));
            };

            // Pack, extract and unpack
            using PackedT = typename PackTraits::PackedT;
            auto packed   = PackUtil::paddedPack(v);
            auto result   = vector_generator<PackedT, PackedVecSize / 2u>()(odds, packed);
            return PackUtil::template paddedUnpack<VecSize / 2u>(result);
        }
        // General case:
        // Re-arrangement of dword+ data sizes isn't super costly and can
        // be achieved with a simple static shuffle.
        else if constexpr(VecSize > 1)
        {
            auto odds = [](auto&& idx, auto&& v) {
                constexpr auto Index = decay_t<decltype(idx)>::value;
                return get<Index * 2 + 1>(v);
            };

            return vector_generator<DataT, VecSize / 2u>()(odds, v);
        }
        // Forwarding case: vector size is 1
        else
        {
            return v;
        }
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto reorderEvenOdd(VecT<DataT, VecSize> const& v)
    {
        using PackUtil   = PackUtil<DataT>;
        using PackTraits = typename PackUtil::Traits;

        // Special case: Sub-dword data sizes, maximum one packed vector.
        // Extract even elements followed by odd elements.
        constexpr auto ElementSize   = sizeof(DataT);
        constexpr auto PackedVecSize = VecSize / PackTraits::PackRatio;
        if constexpr(ElementSize < 4u && PackedVecSize == 1)
        {
            auto evenOdds = [](auto&& idx, auto&& v) {
                constexpr auto Index = decay_t<decltype(idx)>::value;
                return (ElementSize == 2u)
                           ? Blend::ExtractWordEvenOdd::exec(get<Index>(v), get<Index>(v))
                           : Blend::ExtractByteEvenOdd::exec(get<Index>(v), get<Index>(v));
            };

            // Pack, extract and unpack
            using PackedT = typename PackTraits::PackedT;
            auto packed   = PackUtil::paddedPack(v);
            auto result   = vector_generator<PackedT, PackedVecSize>()(evenOdds, packed);
            return PackUtil::template paddedUnpack<VecSize>(result);
        }
        // General case: Concatenate evens and odds
        else if constexpr(VecSize > 1)
        {
            return concat(extractEven(v), extractOdd(v));
        }
        // Forwarding case: return self
        else
        {
            return v;
        }
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto reorderOddEven(VecT<DataT, VecSize> const& v)
    {
        using PackUtil   = PackUtil<DataT>;
        using PackTraits = typename PackUtil::Traits;

        // Special case: Sub-dword data sizes, maximum one packed vector.
        // Optimize data-reorder with cross-lane ops.
        constexpr auto ElementSize   = sizeof(DataT);
        constexpr auto PackedVecSize = VecSize / PackTraits::PackRatio;
        if constexpr(ElementSize < 4u && PackedVecSize <= 1)
        {
            using PackedT = typename PackTraits::PackedT;

            // Exactly one packed vector
            if constexpr(PackedVecSize == 1)
            {
                auto oddEvens = [](auto&& idx, auto&& v) {
                    constexpr auto Index = decay_t<decltype(idx)>::value;
                    return (ElementSize == 2u)
                               ? Blend::ExtractWordOddEven::exec(get<Index>(v), get<Index>(v))
                               : Blend::ExtractByteOddEven::exec(get<Index>(v), get<Index>(v));
                };

                // Pack, extract and unpack
                auto packed = PackUtil::paddedPack(v);
                auto result = vector_generator<PackedT, PackedVecSize>()(oddEvens, packed);
                return PackUtil::template paddedUnpack<VecSize>(result);
            }
            // Corner case: Swap bytes
            else if constexpr(ElementSize == 1 && VecSize == 2)
            {
                auto oddEvens = [](auto&& idx, auto&& v) {
                    // Manually swap bytes
                    using SwapBytes = Blend::Driver<BlendImpl::Ops::PermByte<1u, 0u, 3u, 2u>>;

                    constexpr auto Index = decay_t<decltype(idx)>::value;
                    return SwapBytes::exec(get<Index>(v), get<Index>(v));
                };

                // Pack, extract and unpack
                auto packed = PackUtil::paddedPack(v);
                auto result = vector_generator<PackedT, 1u>()(oddEvens, packed);
                return PackUtil::template paddedUnpack<VecSize>(result);
            }
            // ElementSize = 1, 2 VecSize = 1
            else
            {
                return v;
            }
        }
        // General case: Concatenate evens and odds
        else if constexpr(VecSize > 1)
        {
            return concat(extractOdd(v), extractEven(v));
        }
        // Forwarding case: return self
        else
        {
            return v;
        }
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto zip(VecT<DataT, VecSize> const& v0,
                                                    VecT<DataT, VecSize> const& v1)
    {
        using PackUtil   = PackUtil<DataT>;
        using PackTraits = typename PackUtil::Traits;

        // Special case: Sub-dword data sizes
        // Optimize data-reorder with cross-lane ops.
        constexpr auto ElementSize   = sizeof(DataT);
        constexpr auto PackedVecSize = max(VecSize / PackTraits::PackRatio, 1u);
        if constexpr(ElementSize < 4u)
        {
            auto zip = [](auto&& idx, auto&& v0, auto&& v1) {
                constexpr auto Index = decay_t<decltype(idx)>::value;
                return (ElementSize == 2u) ? Blend::ZipWord::exec(get<Index>(v0), get<Index>(v1))
                                           : Blend::ZipByte::exec(get<Index>(v0), get<Index>(v1));
            };

            // Pack, extract and unpack
            using PackedT = typename PackTraits::PackedT;
            auto packed0  = PackUtil::paddedPack(v0);
            auto packed1  = PackUtil::paddedPack(v1);
            auto result   = vector_generator<PackedT, PackedVecSize>()(zip, packed0, packed1);
            return PackUtil::template paddedUnpack<VecSize>(result);
        }
        else
        {
            auto zip = [](auto&& idx, auto&& v0, auto&& v1) {
                constexpr auto Index = decay_t<decltype(idx)>::value;
                return (Index % 2u == 0u) ? get<Index>(v0) : get<Index>(v1);
            };

            return vector_generator<DataT, VecSize>()(zip, v0, v1);
        }
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto unpackLo(VecT<DataT, VecSize> const& v0,
                                                         VecT<DataT, VecSize> const& v1)
    {
        using PackUtil   = PackUtil<DataT>;
        using PackTraits = typename PackUtil::Traits;

        // Special case: Sub-dword data sizes
        // Optimize data-reorder with cross-lane ops.
        constexpr auto ElementSize   = sizeof(DataT);
        constexpr auto PackedVecSize = max(VecSize / PackTraits::PackRatio, 1u);

        // The optimization should only be applied on a pair of register. So v0 and v1
        // should not be larger than a register
        if constexpr(ElementSize < 4u && PackedVecSize <= 1)
        {
            auto unpackLo = [](auto&& idx, auto&& v0, auto&& v1) {
                constexpr auto Index = decay_t<decltype(idx)>::value;
                return (ElementSize == 2u)
                           ? Blend::UnpackWordLo::exec(get<Index>(v0), get<Index>(v1))
                           : Blend::UnpackByteLo::exec(get<Index>(v0), get<Index>(v1));
            };

            // Pack, extract and unpack
            using PackedT = typename PackTraits::PackedT;
            auto packed0  = PackUtil::paddedPack(v0);
            auto packed1  = PackUtil::paddedPack(v1);
            auto result   = vector_generator<PackedT, PackedVecSize>()(unpackLo, packed0, packed1);
            return PackUtil::template paddedUnpack<VecSize>(result);
        }
        else
        {
            auto unpackLo = [](auto&& idx, auto&& v0, auto&& v1) {
                constexpr auto Index = decay_t<decltype(idx)>::value;
                return (Index % 2u == 0u) ? get<Index / 2u>(v0) : get<Index / 2u>(v1);
            };

            return vector_generator<DataT, VecSize>()(unpackLo, v0, v1);
        }
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto unpackHi(VecT<DataT, VecSize> const& v0,
                                                         VecT<DataT, VecSize> const& v1)
    {
        using PackUtil   = PackUtil<DataT>;
        using PackTraits = typename PackUtil::Traits;

        // Special case: Sub-dword data sizes
        // Optimize data-reorder with cross-lane ops.
        constexpr auto ElementSize   = sizeof(DataT);
        constexpr auto PackedVecSize = VecSize / PackTraits::PackRatio;

        // The optimization should only be applied on a pair of register. So v0 and v1
        // should not be larger than a register
        if constexpr(ElementSize < 4u && PackedVecSize <= 1)
        {
            if constexpr(ElementSize < 2u && PackedVecSize == 0)
            {
                auto unpackHi = [](auto&& idx, auto&& v0, auto&& v1) {
                    constexpr auto Index = decay_t<decltype(idx)>::value;
                    return Blend::UnpackByte3BCast::exec(get<Index>(v0), get<Index>(v1));
                };

                // Pack, extract and unpack
                using PackedT = typename PackTraits::PackedT;
                auto packed0  = PackUtil::paddedPack(v0);
                auto packed1  = PackUtil::paddedPack(v1);
                auto result   = vector_generator<PackedT, 1u>()(unpackHi, packed0, packed1);
                return PackUtil::template paddedUnpack<VecSize>(result);
            }
            else
            {
                auto unpackHi = [](auto&& idx, auto&& v0, auto&& v1) {
                    constexpr auto Index = decay_t<decltype(idx)>::value;
                    return (ElementSize == 2u)
                               ? Blend::UnpackWordHi::exec(get<Index>(v0), get<Index>(v1))
                               : Blend::UnpackByteHi::exec(get<Index>(v0), get<Index>(v1));
                };

                // Pack, extract and unpack
                using PackedT = typename PackTraits::PackedT;
                auto packed0  = PackUtil::paddedPack(v0);
                auto packed1  = PackUtil::paddedPack(v1);
                auto result   = vector_generator<PackedT, 1u>()(unpackHi, packed0, packed1);
                return PackUtil::template paddedUnpack<VecSize>(result);
            }
        }
        else
        {
            auto unpackHi = [](auto&& idx, auto&& v0, auto&& v1) {
                constexpr auto startIdx = VecSize / 2u;
                constexpr auto Index    = decay_t<decltype(idx)>::value;
                return (Index % 2u == 0u) ? get<startIdx + Index / 2u>(v0)
                                          : get<startIdx + Index / 2u>(v1);
            };

            return vector_generator<DataT, VecSize>()(unpackHi, v0, v1);
        }
    }

    // Interleaving index transform
    template <uint32_t GatherSize, uint32_t ElementStride, uint32_t ElementCount>
    struct interleave_idx
    {
        // Uses Number<I> abstraction for indices
        template <typename NumberT>
        constexpr static inline auto exec(NumberT&& Idx)
        {
            // Calculates offsets in interleave transform.
            // If Index > ElementCount, offset cycle will repeat in-place.
            // E.g., interleave<1, 2, 4> with input vector of size 8.
            // ElementCount = 4, therefore offset cycle will repeat every 4 indices.
            // Idx0 = 0  Idx4 = 4  Offset = 0
            // Idx1 = 2  Idx5 = 6  Offset = 2
            // Idx2 = 1  Idx6 = 5  Offset = 1
            // Idx3 = 3  Idx7 = 7  Offset = 3
            constexpr auto Index   = decay_t<decltype(Idx)>::value % ElementCount;
            constexpr auto Offset0 = (Index / GatherSize) * ElementStride % ElementCount;
            constexpr auto Offset1 = Index % GatherSize;
            constexpr auto Offset2
                = (Index * ElementStride) / (ElementCount * GatherSize) * GatherSize;
            constexpr auto Offset3 = decay_t<decltype(Idx)>::value / ElementCount * ElementCount;
            return I<Offset0 + Offset1 + Offset2 + Offset3>{};
        }
    };

    template <typename IdXform>
    struct interleave_idx_traits;

    template <uint32_t GatherSizeIn, uint32_t ElementStrideIn, uint32_t ElementCountIn>
    struct interleave_idx_traits<interleave_idx<GatherSizeIn, ElementStrideIn, ElementCountIn>>
    {
        static constexpr uint32_t GatherSize    = GatherSizeIn;
        static constexpr uint32_t ElementStride = ElementStrideIn;
        static constexpr uint32_t ElementCount  = ElementCountIn;

        // NOP means transform is pass-through (no change)
        static constexpr bool IsNop
            = (GatherSize == ElementStride) || (ElementStride == ElementCount);

        // Sanity check for params
        static constexpr bool IsValid
            = (GatherSize > 0u) && (ElementStride > 0u) && (ElementCount > 0u)
              && (GatherSize <= ElementStride) && (GatherSize <= ElementCount)
              && (ElementStride <= ElementCount) && (ElementStride % GatherSize == 0u)
              && (ElementCount % GatherSize == 0u);
    };

    template <typename... IdXForms>
    constexpr static inline bool test_interleave_idx_nop()
    {
        // Default value for && in fold is 'true', so this
        // also covers the <> case, where nop is assumed.
        return (interleave_idx_traits<IdXForms>::IsNop && ...);
    }

    template <typename... IdXForms>
    constexpr static inline bool test_interleave_idx_valid()
    {
        // Default value for && in fold is 'true', which
        // is not what we want in the <> case - it should be false
        if constexpr(sizeof...(IdXForms) == 0u)
        {
            return false;
        }
        else
        {
            return (interleave_idx_traits<IdXForms>::IsValid && ...);
        }
    }

    // Given a set of interleave id xforms, combine them in the order given
    template <typename... IdXForms>
    struct idx_generator_fwd;

    template <typename IdXForm, typename... IdXForms>
    struct idx_generator_fwd<IdXForm, IdXForms...>
    {
        template <typename IdT>
        constexpr static inline auto exec(IdT&& Idx)
        {
            return IdXForm::exec(idx_generator_fwd<IdXForms...>::exec(forward<IdT>(Idx)));
        }
    };

    template <typename IdXForm>
    struct idx_generator_fwd<IdXForm>
    {
        template <typename IdT>
        constexpr static inline auto exec(IdT&& Idx)
        {
            return IdXForm::exec(forward<IdT>(Idx));
        }
    };

    // Given a set of interleave id xforms, combine them in the reverse of the order given
    template <typename... IdXForms>
    struct idx_generator_bwd;

    template <typename IdXForm, typename... IdXForms>
    struct idx_generator_bwd<IdXForm, IdXForms...>
    {
        template <typename IdT>
        constexpr static inline auto exec(IdT&& Idx)
        {
            return idx_generator_bwd<IdXForms...>::exec(IdXForm::exec(forward<IdT>(Idx)));
        }
    };

    template <typename IdXForm>
    struct idx_generator_bwd<IdXForm>
    {
        template <typename IdT>
        constexpr static inline auto exec(IdT&& Idx)
        {
            return IdXForm::exec(forward<IdT>(Idx));
        }
    };

    template <template <typename...> class IdxGenerator, typename... IdXForms, typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) interleave_internal(VecT&& v0)
    {
        // Sanity check
        static_assert(test_interleave_idx_valid<IdXForms...>(),
                      "Invalid interleave xform provided");

        if constexpr(test_interleave_idx_nop<IdXForms...>())
        {
            // If nop, don't make a copy
            return forward<VecT>(v0);
        }
        else
        {
            // Interleave Index generator from from input transforms
            using GenIdx = IdxGenerator<IdXForms...>;

            // Extract vector traits
            using VecTraits            = VecTraits<decay_t<decltype(v0)>>;
            using DataT                = typename VecTraits::DataT;
            constexpr uint32_t VecSize = VecTraits::size();

            // Apply index generator
            auto idxform = [](auto&& idx, auto&& v0) {
                using Idx = decay_t<decltype(GenIdx::exec(idx))>;
                return get<Idx>(forward<VecT>(v0));
            };

            // Generate a new vector with the xformed indices
            return vector_generator<DataT, VecSize>()(idxform, forward<VecT>(v0));
        }
    }

    template <typename... IdXForms, typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) interleave_internal_fwd(VecT&& v0)
    {
        return interleave_internal<idx_generator_fwd, IdXForms...>(forward<VecT>(v0));
    }

    template <typename... IdXForms, typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) interleave_internal_bwd(VecT&& v0)
    {
        return interleave_internal<idx_generator_bwd, IdXForms...>(forward<VecT>(v0));
    }

    // A permutation of vector indices, given a gather size and a stride
    // Examples:
    //                                     row_major               col_major
    //       [0, 1] => interleave<1, 2, 6>([0, 1, 2, 3, 4, 5])  =  [0, 2, 4, 1, 3, 5]
    //   A = [2, 3]                        col_major               row_major
    //       [4, 5] => interleave<1, 4, 6>([0, 2, 4, 1, 3, 5])  =  [0, 1, 2, 3, 4, 5]
    //
    //       [0, 1]
    //   A = [2, 3] => interleave<2, 4, 6>([0, 1, 2, 3, 4, 5, 6, 7]) = [0, 1, 4, 5, 2, 3, 6, 7]
    //       [4, 5]
    //       [6, 7]
    //
    template <uint32_t GatherSize,
              uint32_t ElementStride,
              uint32_t ElementCount = 0u,
              typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) interleave(VecT&& v0)
    {
        // If unspecified, ElementCount can be derived from the input vector size
        using VecTraits            = VecTraits<decay_t<decltype(v0)>>;
        constexpr uint32_t VecSize = ElementCount > 0u ? ElementCount : VecTraits::size();

        // Build interleaved index generator
        using IdXForm = interleave_idx<GatherSize, ElementStride, VecSize>;
        return interleave_internal_fwd<IdXForm>(forward<VecT>(v0));
    }

    template <typename... IdXForms, typename VecT>
    ROCWMMA_DEVICE constexpr static inline decltype(auto) interleave_combine(VecT&& v0)
    {
        // Build interleaved index generator
        return interleave_internal_fwd<IdXForms...>(forward<VecT>(v0));
    }

} // namespace rocwmma

#endif // ROCWMMA_VECTOR_UTIL_IMPL_HPP
