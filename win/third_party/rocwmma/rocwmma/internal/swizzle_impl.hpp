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
#ifndef ROCWMMA_SWIZZLE_IMPL_HPP
#define ROCWMMA_SWIZZLE_IMPL_HPP

#include "swizzle.hpp"
#include "utils.hpp"

namespace rocwmma
{
    namespace SwizzleImpl
    {
        // Implementation meta-data
        using CrossLaneOps::OpBase;
        using CrossLaneOps::Properties;

        // Dpp backend
        using Properties::OP_IMPL_SWIZZLE;

        // Functional
        using Properties::OP_ID_BCAST;
        using Properties::OP_ID_FFT;
        using Properties::OP_ID_REVERSE;
        using Properties::OP_ID_ROTATE;
        using Properties::OP_ID_SHUFFLE;
        using Properties::OP_ID_SWAP;

        // Groups
        using Properties::OP_GROUP_SIZE_16;
        using Properties::OP_GROUP_SIZE_2;
        using Properties::OP_GROUP_SIZE_32;
        using Properties::OP_GROUP_SIZE_4;
        using Properties::OP_GROUP_SIZE_8;
        using Properties::OP_GROUP_SIZE_WARP;

        // Detail
        using Properties::OP_DIR_L;
        using Properties::OP_DIR_R;

        namespace Backend
        {
            template <class SwizzleCtrl>
            struct amdgcn_swizzle
            {
                template <typename DataT>
                ROCWMMA_DEVICE static inline DataT exec(DataT input)
                {
                    static_assert(sizeof(DataT) == sizeof(uint32_t),
                                  "The unit of the Swizzle operation should be 32 bits");
                    reinterpret_cast<int32_t&>(input) = __builtin_amdgcn_ds_swizzle(
                        reinterpret_cast<int32_t const&>(input), SwizzleCtrl::opCtrl());
                    return input;
                }
            };

        } // namespace Backend

        namespace Ctrl
        {
            template <uint32_t FftCtrl>
            struct Fft
            {
            private:
                enum Traits : uint32_t
                {
                    // Swizzle mode: 0xe000 = FFT mode
                    // FFT_CTRL: a 5-bit code for fft transform (see ISA details)
                    SWIZZLE_MODE = 0xe000,
                    FFT_CTRL     = FftCtrl & 0x1F,

                    SWIZZLE_CTRL = SWIZZLE_MODE | FFT_CTRL
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::SWIZZLE_CTRL;
                }
            };

            template <uint32_t RotationDir, uint32_t RotationDist, uint32_t GroupSize>
            struct Rotate
            {
            private:
                enum Traits : uint32_t
                {
                    // Swizzle mode: 0xc000 = rotation mode
                    // Rotation dir [10]: 0 = left, 1 = right
                    // Rotation distance [9:5]: rotation distance in element count
                    // Group size mask [4:0]: rotation within specified group size
                    // 0x00 = 32, 0x10 = 16, 0x18 = 8, 0x1C = 4 0x1E = 2, 0x1F = 1
                    SWIZZLE_MODE  = 0xc000,
                    ROTATION_DIR  = (RotationDir & 0x1) << 10,
                    ROTATION_DIST = (RotationDist & 0x1F) << 5,
                    GROUP_SIZE    = ((32u - GroupSize) & 0x1F),

                    SWIZZLE_CTRL = SWIZZLE_MODE | ROTATION_DIR | ROTATION_DIST | GROUP_SIZE
                };

                static_assert(Log2<GroupSize>::value, "GroupSize must be a power of 2");
                static_assert(RotationDir <= 1u, "RotationDir must be either 0: left or 1: right");

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::SWIZZLE_CTRL;
                }
            };

            template <uint32_t Select0, uint32_t Select1, uint32_t Select2, uint32_t Select3>
            struct Shuffle4
            {
            private:
                enum Traits : uint32_t
                {
                    // Swizzle mode: 0x8000 = full data shuffle within thread groups of 4
                    // Valid index selection is 0, 1, 2 or 3
                    // For every group of 4 threads, shuffle element selects:
                    SWIZZLE_MODE = 0x8000,
                    SELECT_0     = (Select0 & 0x3),
                    SELECT_1     = (Select1 & 0x3) << 2,
                    SELECT_2     = (Select2 & 0x3) << 4,
                    SELECT_3     = (Select3 & 0x3) << 6,

                    SWIZZLE_CTRL = SWIZZLE_MODE | SELECT_0 | SELECT_1 | SELECT_2 | SELECT_3
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::SWIZZLE_CTRL;
                }
            };

            template <uint32_t XorMask, uint32_t OrMask, uint32_t AndMask>
            struct Manual
            {
            private:
                enum Traits : uint32_t
                {
                    // Swizzle mode: 0x0000 = manual assignment of xor, or and and masks
                    // Note: Limited data sharing within 32 groups of threads.
                    // XorMask [14:10]
                    // OrMask [9:5]
                    // AndMask [0:4]
                    SWIZZLE_MODE = 0x0000,
                    XOR_MASK     = (XorMask & 0x1F) << 10,
                    OR_MASK      = (OrMask & 0x1F) << 5,
                    AND_MASK     = (AndMask & 0x1F),

                    SWIZZLE_CTRL = SWIZZLE_MODE | XOR_MASK | OR_MASK | AND_MASK
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::SWIZZLE_CTRL;
                }
            };

            template <uint32_t ElementIdx, uint32_t SubGroupSize>
            using BCast = Manual<0x0, ElementIdx, 0x20 - SubGroupSize>;

            template <uint32_t SubGroupSize>
            using Reverse = Manual<SubGroupSize - 0x1, 0x00, 0x1F>;

            template <uint32_t SubGroupSize>
            using Swap = Manual<SubGroupSize, 0x00, 0x1F>;

        } // namespace Ctrl

        namespace OpsBase
        {
            template <uint32_t OpId, uint32_t SubGroupSize>
            using SwzOp = OpBase<OpId, SubGroupSize, OP_IMPL_SWIZZLE>;

            /*! \class BCast
            *  \brief Performs localized broadcast of one element in each sub-group to the entire sub-group.
            *
            * @tparam ElementIdx - element index to broadcast to rest of the sub-group
            */
            template <uint32_t ElementIdx, uint32_t SubGroupSize>
            struct BCast : public SwzOp<OP_ID_BCAST, SubGroupSize>,
                           public Backend::amdgcn_swizzle<Ctrl::BCast<ElementIdx, SubGroupSize>>
            {
                enum : uint32_t
                {
                    ELEMENT_IDX = ElementIdx,
                };

                constexpr static uint32_t elementIdx()
                {
                    return ELEMENT_IDX;
                }
            };

            /*! \class Fft
            *  \brief Supports FFT-like cross-bar transforms
            */
            template <uint32_t SubGroupSize, uint32_t FftCtrl>
            struct Fft : public SwzOp<OP_ID_FFT, SubGroupSize>,
                         public Backend::amdgcn_swizzle<Ctrl::Fft<FftCtrl>>
            {
                enum : uint32_t
                {
                    FFT_CTRL = FftCtrl
                };

                constexpr static uint32_t fftCtrl()
                {
                    return FFT_CTRL;
                }
            };

            /*! \class Reverse
            *  \brief Perform reversal of elements in sub-groups of \p SubGroupSize threads.
            */
            template <uint32_t SubGroupSize>
            struct Reverse : public SwzOp<OP_ID_REVERSE, SubGroupSize>,
                             public Backend::amdgcn_swizzle<Ctrl::Reverse<SubGroupSize>>
            {
            };

            /*! \class Rotate
            *  \brief Perform element-wise rotation in direction \p RotateDir in sub-groups of \p SubGroupSize threads.
            *
            * @tparam RotateDir rotation direction: see Properties
            * @tparam RotateDistance element positions to move in specified direction. Positions wrapped by sub group size.
            */
            template <uint32_t RotateDir, uint32_t RotateDist, uint32_t SubGroupSize>
            struct Rotate
                : public SwzOp<OP_ID_ROTATE, SubGroupSize>,
                  public Backend::amdgcn_swizzle<Ctrl::Rotate<RotateDir, RotateDist, SubGroupSize>>
            {
                enum : uint32_t
                {
                    OP_DIR  = RotateDir,
                    OP_DIST = RotateDist
                };

                constexpr static uint32_t opDir()
                {
                    return OP_DIR;
                }
                constexpr static uint32_t opDist()
                {
                    return OP_DIST;
                }
            };

            template <uint32_t RotateDistance, uint32_t SubGroupSize>
            using RotateR = Rotate<OP_DIR_R, RotateDistance, SubGroupSize>;

            template <uint32_t RotateDistance, uint32_t SubGroupSize>
            using RotateL = Rotate<OP_DIR_L, RotateDistance, SubGroupSize>;

            /*! \class Shuffle
            *  \brief Perform localized shuffling within sub-groups of \p SubGroupSize threads.
            */
            template <uint32_t SubGroupSize, class ShuffleCtrl>
            struct Shuffle : public SwzOp<OP_ID_SHUFFLE, SubGroupSize>,
                             public Backend::amdgcn_swizzle<ShuffleCtrl>
            {
            };

            // Common Shuffle variants
            /*! \class Shuffle4
            *  \brief Shuffle\<N\> Perform localized shuffling within all sub-groups of \em N threads.
            * \em N = group size.
            *
            * @tparam Select0 - index of element to shuffle to index 0
            * @tparam Select1 - index of element to shuffle to index 1
            * @tparam Select2 - index of element to shuffle to index 2
            * @tparam Select3 - index of element to shuffle to index 3
            */
            template <uint32_t Select0, uint32_t Select1, uint32_t Select2, uint32_t Select3>
            struct Shuffle4 : public Shuffle<OP_GROUP_SIZE_4,
                                             Ctrl::Shuffle4<Select0, Select1, Select2, Select3>>
            {
                enum : uint32_t
                {
                    SELECT_0 = Select0,
                    SELECT_1 = Select1,
                    SELECT_2 = Select2,
                    SELECT_3 = Select3,
                };

                constexpr static uint32_t select0()
                {
                    return SELECT_0;
                }
                constexpr static uint32_t select1()
                {
                    return SELECT_1;
                }
                constexpr static uint32_t select2()
                {
                    return SELECT_2;
                }
                constexpr static uint32_t select3()
                {
                    return SELECT_3;
                }
            };

            template <uint32_t Select0, uint32_t Select1>
            struct Shuffle2 : public Shuffle<OP_GROUP_SIZE_2,
                                             Ctrl::Shuffle4<Select0,
                                                            Select1,
                                                            Select0 + OP_GROUP_SIZE_2,
                                                            Select1 + OP_GROUP_SIZE_2>>
            {
                enum : uint32_t
                {
                    SELECT_0 = Select0,
                    SELECT_1 = Select1,
                };

                constexpr static uint32_t select0()
                {
                    return SELECT_0;
                }
                constexpr static uint32_t select1()
                {
                    return SELECT_1;
                }
            };

            /*! \class Swap
            *  \brief Perform swap of neigbouring sub-groups of \p SubGroupSize threads.
            */
            template <uint32_t SubGroupSize>
            struct Swap : public SwzOp<OP_ID_SWAP, SubGroupSize>,
                          public Backend::amdgcn_swizzle<Ctrl::Swap<SubGroupSize>>
            {
            };

        } // namespace OpsBase

        namespace Ops
        {

            /**
             * \ingroup Cross_Lane_Operations
             *
             * @brief Cross-lane operations implemented with the amdgcn_ds_swizzle backend.
             * @{
             */

            // clang-format off

            // BCast variants
            template <uint32_t ElementIdx>
            using BCast32 = OpsBase::BCast<ElementIdx, OP_GROUP_SIZE_32>;

            template <uint32_t ElementIdx>
            using BCast16 = OpsBase::BCast<ElementIdx, OP_GROUP_SIZE_16>;

            template <uint32_t ElementIdx>
            using BCast8 = OpsBase::BCast<ElementIdx, OP_GROUP_SIZE_8>;

            template <uint32_t ElementIdx>
            using BCast4 = OpsBase::BCast<ElementIdx, OP_GROUP_SIZE_4>;

            template <uint32_t ElementIdx>
            using BCast2 = OpsBase::BCast<ElementIdx, OP_GROUP_SIZE_2>;

            // Reverse variants
            using Reverse32 = OpsBase::Reverse<OP_GROUP_SIZE_32>;

            using Reverse16 = OpsBase::Reverse<OP_GROUP_SIZE_16>;

            using Reverse8 = OpsBase::Reverse<OP_GROUP_SIZE_8>;

            using Reverse4 = OpsBase::Reverse<OP_GROUP_SIZE_4>;

            using Reverse2 = OpsBase::Reverse<OP_GROUP_SIZE_2>;

            // RotateL variants
            template <uint32_t RotateDistance>
            using RotateL32 = OpsBase::RotateL<RotateDistance, OP_GROUP_SIZE_32>;

            template <uint32_t RotateDistance>
            using RotateL16 = OpsBase::RotateL<RotateDistance, OP_GROUP_SIZE_16>;

            template <uint32_t RotateDistance>
            using RotateL8 = OpsBase::RotateL<RotateDistance, OP_GROUP_SIZE_8>;

            template <uint32_t RotateDistance>
            using RotateL4 = OpsBase::RotateL<RotateDistance, OP_GROUP_SIZE_4>;

            template <uint32_t RotateDistance>
            using RotateL2 = OpsBase::RotateL<RotateDistance, OP_GROUP_SIZE_2>;

            // RotateR variants
            template <uint32_t RotateDistance>
            using RotateR32 = OpsBase::RotateR<RotateDistance, OP_GROUP_SIZE_32>;

            template <uint32_t RotateDistance>
            using RotateR16 = OpsBase::RotateR<RotateDistance, OP_GROUP_SIZE_16>;

            template <uint32_t RotateDistance>
            using RotateR8 = OpsBase::RotateR<RotateDistance, OP_GROUP_SIZE_8>;

            template <uint32_t RotateDistance>
            using RotateR4 = OpsBase::RotateR<RotateDistance, OP_GROUP_SIZE_4>;

            template <uint32_t RotateDistance>
            using RotateR2 = OpsBase::RotateR<RotateDistance, OP_GROUP_SIZE_2>;

            // Shuffle variants
            template <uint32_t Select0, uint32_t Select1, uint32_t Select2, uint32_t Select3>
            using Shuffle4 = OpsBase::Shuffle4<Select0, Select1, Select2, Select3>;

            template <uint32_t Select0, uint32_t Select1>
            using Shuffle2 = OpsBase::Shuffle2<Select0, Select1>;

            // Swap variants
            using Swap16 = OpsBase::Swap<OP_GROUP_SIZE_16>;

            using Swap8 = OpsBase::Swap<OP_GROUP_SIZE_8>;

            using Swap4 = OpsBase::Swap<OP_GROUP_SIZE_4>;

            using Swap2 = OpsBase::Swap<OP_GROUP_SIZE_2>;

            /*! \class Fft
            *  \brief Supports FFT-like cross-bar transforms
            *
            * @tparam FftCtrl - 5-bit swizzle code (see instruction ISA for layouts)
            */
            template <uint32_t SubGroupSize, uint32_t FftCtrl>
            using Fft = OpsBase::Fft<SubGroupSize, FftCtrl>;
            /** @}*/

            // clang-format on
        } // namespace Ops

    } // namespace SwizzleImpl

} // namespace rocwmma

#endif // ROCWMMA_SWIZZLE_IMPL_HPP
