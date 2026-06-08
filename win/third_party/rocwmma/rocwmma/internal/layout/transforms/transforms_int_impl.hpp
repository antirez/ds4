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
#ifndef ROCWMMA_LAYOUT_TRANSFORMS_INT_IMPL_HPP
#define ROCWMMA_LAYOUT_TRANSFORMS_INT_IMPL_HPP

#include "../../transforms.hpp"
#include "../../utility/vector.hpp"

namespace rocwmma
{
    namespace Transforms
    {
        template <uint32_t KPerThread, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) soa_int_to_aos_int(VecT&& v)
        {
            // interleave<1, KPT, VecSize>
            return interleave<1u, KPerThread>(forward<VecT>(v));
        }

        template <uint32_t DimPerThread, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) aos_int_to_soa_int(VecT&& v)
        {
            // interleave<1, DPT, VecSize>
            constexpr uint32_t GatherSize = 1u;
            return interleave<GatherSize, DimPerThread>(forward<VecT>(v));
        }

        template <uint32_t AccVecSize, uint32_t MmaBlocksA, uint32_t AccMaxVW, uint32_t MmaDim, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto)
            soa_int_to_mma_acc_int_a_major(VecT&& v)
        {
            using VecTraits = VecTraits<decay_t<VecT>>;

            if constexpr((bool)ROCWMMA_ARCH_GFX9)
            {
                if constexpr (MmaDim == 16u)
                {
                    if constexpr(AccMaxVW == 1u)
                    {
                        // First, interleave full vector
                        // interleave<1, MmaBlocksA, VecSize>
                        auto result = interleave<1u, MmaBlocksA>(forward<VecT>(v));

                        // For each subvector of AccVecSize:
                        // unpackLoHi16 + unpackLoHi32
                        return vector_for_each<AccVecSize>(result, [](auto&& v, uint32_t idx) {
                            return unpackLoHi32(unpackLoHi16(v));
                        });
                    }
                    else if constexpr(AccMaxVW == 4u)
                    {
                        // Interleave full vector
                        return interleave<1u, MmaBlocksA>(forward<VecT>(v));
                    }
                }
                else if constexpr(MmaDim == 32u && AccMaxVW == 4u)
                {
                    using interleave_idx0 = interleave_idx<1u, MmaBlocksA, VecTraits::size()>;
                    using interleave_idx1 = interleave_idx<4u, 8u, 16u>;

                    // First perform combined interleave on full vector
                    // interleave<1, MmaBlocksA, VecSize> + interleave<1, 2, AccVecSize>
                    auto result
                        = interleave_combine<interleave_idx0, interleave_idx1>(forward<VecT>(v));

                    // For each subvector of AccVecSize:
                    // unpackLoHi32
                    return vector_for_each<AccVecSize>(result, [](auto&& v, uint32_t idx) {
                        return unpackLoHi32(extractLo(v), extractHi(v));
                    });
                }
            }
            else if constexpr((bool)ROCWMMA_ARCH_GFX11)
            {
                if constexpr (MmaDim == 16u && AccMaxVW == 1u)
                {
                    using interleave_idx0 = interleave_idx<1u, MmaBlocksA, VecTraits::size()>;
                    using interleave_idx1 = interleave_idx<1u, 2u, AccVecSize>;

                    // First perform combined interleave on full vector
                    // interleave<1, MmaBlocksA, VecSize> + interleave<1, 2, AccVecSize>
                    auto result
                        = interleave_combine<interleave_idx0, interleave_idx1>(forward<VecT>(v));

                    // For each subvector of AccVecSize:
                    // unpackLoHi16
                    return vector_for_each<AccVecSize>(result, [](auto&& v, uint32_t idx) {
                        return unpackLoHi16(extractLo(v), extractHi(v));
                    });
                }
            }
            else if constexpr((bool)ROCWMMA_ARCH_GFX12)
            {
                if constexpr(MmaDim == 16u && AccMaxVW == 8u)
                {
                    // Interleave full vector
                    return interleave<1u, MmaBlocksA>(forward<VecT>(v));
                }
            }
            else
            {
                return forward<VecT>(v);
            }
        }

        template <uint32_t AccVecSize, uint32_t MmaBlocksB, uint32_t AccMaxVW, uint32_t MmaDim, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto)
            mma_acc_int_a_major_to_soa_int(VecT&& v)
        {
            using VecTraits = VecTraits<decay_t<VecT>>;

            if constexpr((bool)ROCWMMA_ARCH_GFX9)
            {
                if constexpr(MmaDim == 16u)
                {
                    if constexpr(AccMaxVW == 1u)
                    {
                        // unpackLoHi16 on entire vector
                        // unpackLoHi32 on entire vector
                        // interleave entire vector
                        // Note: make a copy here, due to interleave may return
                        // fwd arg input on nop.
                        auto result = interleave<1u, AccMaxVW * MmaBlocksB>(
                            unpackLoHi32(unpackLoHi16(forward<VecT>(v))));
                        return result;
                    }
                    else if constexpr(AccMaxVW == 4u)
                    {
                        // Interleave full vector
                        return interleave<1u, AccMaxVW * MmaBlocksB>(forward<VecT>(v));
                    }
                }
                else if constexpr (MmaDim == 32u && AccMaxVW == 4u)
                {
                    // For each subvector of AccVecSize:
                    // unpackLoHi32
                    auto result
                        = vector_for_each<AccVecSize>(forward<VecT>(v), [](auto&& v, uint32_t idx) {
                            return unpackLoHi32(extractLo(v), extractHi(v));
                        });

                    // Perform combined interleave on full vector
                    using interleave_idx0 = interleave_idx<4u, 8u, AccVecSize>;
                    using interleave_idx1 = interleave_idx<1u, MmaBlocksB * 16u, VecTraits::size()>;
                    return interleave_combine<interleave_idx0, interleave_idx1>(result);
                }
            }
            else if constexpr((bool)ROCWMMA_ARCH_GFX11)
            {
                if constexpr(MmaDim == 16u && AccMaxVW == 1u)
                {
                    // For each subvector of AccVecSize:
                    // unpackLoHi16
                    auto result
                        = vector_for_each<AccVecSize>(forward<VecT>(v), [](auto&& v, uint32_t idx) {
                            return unpackLoHi16(extractLo(v), extractHi(v));
                        });

                    // Perform combined interleave on full vector
                    using interleave_idx0 = interleave_idx<1u, 4u, AccVecSize>;
                    using interleave_idx1 = interleave_idx<1u, MmaBlocksB * 8u, VecTraits::size()>;
                    return interleave_combine<interleave_idx0, interleave_idx1>(result);
                }
            }
            else if constexpr((bool)ROCWMMA_ARCH_GFX12)
            {
                if constexpr(MmaDim == 16u && AccMaxVW == 8u)
                {
                    // Interleave full vector
                    return interleave<1u, AccMaxVW * MmaBlocksB>(forward<VecT>(v));
                }
            }
            else
            {
                //static_assert(0, "Shouldn't get here");
                return forward<VecT>(v);
            }
        }

        template <uint32_t AccVecSize,
                  uint32_t MmaBlocksA,
                  uint32_t MmaBlocksB,
                  uint32_t AccMaxVW,
                  uint32_t MmaDim,
                  typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto)
            aos_int_to_mma_acc_int_a_major(VecT&& v)
        {
            using VecTraits = VecTraits<decay_t<VecT>>;

            if constexpr((bool)ROCWMMA_ARCH_GFX9)
            {
                if constexpr (MmaDim == 16u)
                {
                    if constexpr(AccMaxVW == 1u)
                    {
                        // First, interleave full vector
                        // interleave<1, MmaBlocksA, VecSize>
                        auto result = interleave<1u, MmaBlocksA * MmaBlocksB>(forward<VecT>(v));

                        // For each subvector of AccVecSize:
                        // unpackLoHi16 + unpackLoHi32
                        return vector_for_each<AccVecSize>(result, [](auto&& v, uint32_t idx) {
                            return unpackLoHi32(unpackLoHi16(v));
                        });
                    }
                    else if constexpr(AccMaxVW == 4u)
                    {
                        // Interleave full vector
                        return interleave<1u, MmaBlocksA * MmaBlocksB>(forward<VecT>(v));
                    }
                }
                else if constexpr(MmaDim == 32u && AccMaxVW == 4u)
                {
                    using interleave_idx0
                        = interleave_idx<1u, MmaBlocksA * MmaBlocksB, VecTraits::size()>;
                    using interleave_idx1 = interleave_idx<4u, 8u, AccVecSize>;

                    // First perform combined interleave on full vector
                    // interleave<1, MmaBlocksA, VecSize> + interleave<1, 2, AccVecSize>
                    auto result
                        = interleave_combine<interleave_idx0, interleave_idx1>(forward<VecT>(v));

                    // For each subvector of AccVecSize:
                    // unpackLoHi32
                    return vector_for_each<AccVecSize>(result, [](auto&& v, uint32_t idx) {
                        return unpackLoHi32(extractLo(v), extractHi(v));
                    });
                }
            }
            else if constexpr((bool)ROCWMMA_ARCH_GFX11)
            {
                if constexpr(MmaDim == 16u && AccMaxVW == 1u)
                {
                    using interleave_idx0
                        = interleave_idx<1u, MmaBlocksA * MmaBlocksB, VecTraits::size()>;
                    using interleave_idx1 = interleave_idx<1u, 2u, AccVecSize>;

                    // First perform combined interleave on full vector
                    // interleave<1, MmaBlocksA, VecSize> + interleave<1, 2, AccVecSize>
                    auto result
                        = interleave_combine<interleave_idx0, interleave_idx1>(forward<VecT>(v));

                    // For each subvector of AccVecSize:
                    // unpackLoHi16
                    return vector_for_each<AccVecSize>(result, [](auto&& v, uint32_t idx) {
                        return unpackLoHi16(extractLo(v), extractHi(v));
                    });
                }
            }
            else if constexpr((bool)ROCWMMA_ARCH_GFX12)
            {
                if constexpr(MmaDim == 16u && AccMaxVW == 8u)
                {
                    // Interleave full vector
                    return interleave<1u, MmaBlocksA * MmaBlocksB>(forward<VecT>(v));
                }
            }
            else
            {
                //static_assert(0, "Shouldn't get here");
                return forward<VecT>(v);
            }
        }

        template <uint32_t AccVecSize, uint32_t AccMaxVW, uint32_t MmaDim, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto)
            mma_acc_int_a_major_to_aos_int(VecT&& v)
        {
            using VecTraits = VecTraits<decay_t<VecT>>;

            if constexpr((bool)ROCWMMA_ARCH_GFX9)
            {
                if constexpr (MmaDim == 16u)
                {
                    if constexpr(AccMaxVW == 1u)
                    {
                        // unpackLoHi16 on entire vector
                        // unpackLoHi32 on entire vector
                        return unpackLoHi32(unpackLoHi16(forward<VecT>(v)));
                    }
                    else if constexpr(AccMaxVW == 4u)
                    {
                        // Interleave full vector
                        return interleave<1u, AccMaxVW>(forward<VecT>(v));
                    }
                }
                else if constexpr (MmaDim == 32u && AccMaxVW == 4u)
                {
                    // For each subvector of AccVecSize:
                    // unpackLoHi32
                    auto result
                        = vector_for_each<AccVecSize>(forward<VecT>(v), [](auto&& v, uint32_t idx) {
                            return unpackLoHi32(extractLo(v), extractHi(v));
                        });

                    // Perform combined interleave on full vector
                    using interleave_idx0 = interleave_idx<4u, 8u, AccVecSize>;
                    using interleave_idx1 = interleave_idx<1u, 16u, VecTraits::size()>;
                    return interleave_combine<interleave_idx0, interleave_idx1>(result);
                }
            }
            else if constexpr((bool)ROCWMMA_ARCH_GFX11)
            {
                if constexpr (MmaDim == 16u && AccMaxVW == 1u)
                {
                    // For each subvector of AccVecSize:
                    // unpackLoHi16
                    auto result
                        = vector_for_each<AccVecSize>(forward<VecT>(v), [](auto&& v, uint32_t idx) {
                            return unpackLoHi16(extractLo(v), extractHi(v));
                        });

                    // Perform combined interleave on full vector
                    using interleave_idx0 = interleave_idx<1u, 4u, AccVecSize>;
                    using interleave_idx1 = interleave_idx<1u, 8u, VecTraits::size()>;
                    return interleave_combine<interleave_idx0, interleave_idx1>(result);
                }
            }
            else if constexpr((bool)ROCWMMA_ARCH_GFX12)
            {
                if constexpr(MmaDim == 16u && AccMaxVW == 8u)
                {
                    // Interleave full vector
                    return interleave<1u, AccMaxVW>(forward<VecT>(v));
                }
            }
            else
            {
                //static_assert(0, "Shouldn't get here");
                return forward<VecT>(v);
            }
        }

    } // namespace Transforms

} // namespace rocwmma

#endif // ROCWMMA_LAYOUT_TRANSFORMS_INT_IMPL_HPP
