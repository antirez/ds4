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
#ifndef ROCWMMA_MOVE_DPP_IMPL_HPP
#define ROCWMMA_MOVE_DPP_IMPL_HPP

#include "dpp.hpp"
#include "vector_util.hpp"

namespace rocwmma
{

    namespace DppImpl
    {
        // Implementation meta-data
        using CrossLaneOps::OpBase;
        using CrossLaneOps::Properties;

        // Dpp backend
        using Properties::OP_IMPL_DPP;

        // Functional
        using Properties::OP_ID_BCAST;
        using Properties::OP_ID_BLEND;
        using Properties::OP_ID_MOVE;
        using Properties::OP_ID_REVERSE;
        using Properties::OP_ID_ROTATE;
        using Properties::OP_ID_SHIFT;
        using Properties::OP_ID_SHUFFLE;
        using Properties::OP_ID_SWAP;
        using Properties::OP_ID_WFALL_BCAST;

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
            /*! \class amdgcn_mov_dpp
            *  \brief Implements the DPP backend between 32b elements of two src vectors.
            * DPP has a wide variety of cross-lane manipulations, operating on src0, and
            * applying the result to src1 according to row and bank masks.
            *
            * 'RowMask' is a 4 bit flag. One 'row' is a group of 16 threads.
            * For each 'row' in the register:
            * Select bit hi '1' will write the src0 operation result,
            * Select bit lo '0' will write the src1 values.
            *
            * In Wave32, there are two 'rows' (upper 2 bits ignored).
            * In Wave64, there are four 'rows'
            *
            * 'BankMask' is a 4-bit flag. One 'bank' is a group of 4 threads. There are 4
            * banks in every 'row'. For each 'bank' in every 'row':
            * Select bit hi '1' will write the src0 operation result,
            * Select bit lo '0' will write the src1 values.
            *
            * This allows control over masking data movement from src vectors and results.
            * E.g. RowMask=0x3, BankMask=0x1
            * Writes the operation result to destination elements [0..3], [16..19] => (R0B0, R1B0)
            * The rest of destination elements are filled in with src1 values.
            *
            * @tparam DppCtrl is a generator class to create control code for DPP backend
            * @tparam WriteRowMask is a 4-bit mask for destination row write (w32 only 2 lower bits)
            * @tparam WriteBankMask is a 4-bit mask for destination bank write
            * @tparam BoundCtrl single bit mask, fills 0 on OOB indices.
            * @arg src0 is input vector to dpp
            * @arg src1 is fill values for row/bank mask
            */
            template <class DppCtrl>
            struct amdgcn_mov_dpp
            {
                template <uint32_t WriteRowMask,
                          uint32_t WriteBankMask,
                          bool     BoundCtrl,
                          typename DataT>
                ROCWMMA_DEVICE static inline DataT exec(DataT src0, DataT src1)
                {
                    static_assert(sizeof(DataT) == sizeof(uint32_t),
                                  "The unit of the DPP operation should be 32 bits");
                    reinterpret_cast<int32_t&>(src0) = __builtin_amdgcn_update_dpp(
                        reinterpret_cast<int32_t const&>(src1), // fill value 'prev'
                        reinterpret_cast<int32_t const&>(src0), // Src value
                        DppCtrl::opCtrl(), // DPP control code
                        WriteRowMask, // Mask for affected rows
                        WriteBankMask, // Mask for affected banks
                        BoundCtrl); // Fill in 0 on invalid indices
                    return src0;
                }
            };

        } // namespace backend

        namespace Ctrl
        {

            template <uint32_t Select0, uint32_t Select1, uint32_t Select2, uint32_t Select3>
            struct Shuffle4
            {
            private:
                enum Traits : uint32_t
                {
                    // Valid index selection is 0, 1, 2 or 3
                    // For every group of 4 threads, shuffle element selects:
                    DPP_BASE = 0x0,
                    SELECT_0 = (Select0 & 0x3),
                    SELECT_1 = (Select1 & 0x3) << 2,
                    SELECT_2 = (Select2 & 0x3) << 4,
                    SELECT_3 = (Select3 & 0x3) << 6,

                    DPP_CTRL = DPP_BASE | SELECT_0 | SELECT_1 | SELECT_2 | SELECT_3
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

            // Straight dpp move
            using Move = Shuffle4<0u, 1u, 2u, 3u>;

            // This 'nop' is to signify that the opctrl will be a basic move
            // which is supported on all archs.
            // Still respects masking and bound control flags.
            using NOP = Move;

            template <uint32_t ShiftDir, uint32_t ShiftDist>
            struct RowShift
            {
            private:
                enum Traits : uint32_t
                {
                    // Shift mode base offset: 0x0100 (undefined for ShiftDistance = 0)
                    // Shift dir [4] : 0 = left, 1 = right
                    // Shift distance [3:0]: shift distance in element count
                    DPP_BASE   = 0x0100,
                    SHIFT_DIR  = (ShiftDir & 0x1) << 4,
                    SHIFT_DIST = (ShiftDist & 0xF),

                    DPP_CTRL = DPP_BASE | SHIFT_DIR | SHIFT_DIST
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

            template <uint32_t ShiftDist>
            using RowShiftR = RowShift<OP_DIR_R, ShiftDist>;
            template <uint32_t ShiftDist>
            using RowShiftL = RowShift<OP_DIR_L, ShiftDist>;

            template <uint32_t RotateDist>
            struct RowRotateR
            {
            private:
                enum Traits : uint32_t
                {
                    // Rotate mode base offset: 0x0120
                    // Rotate distance [3:0]: rotate distance in element count
                    DPP_BASE    = 0x0120,
                    ROTATE_DIST = (RotateDist & 0xF),

                    DPP_CTRL = DPP_BASE | ROTATE_DIST
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

            struct RowReverse
            {
            private:
                enum Traits : uint32_t
                {
                    // Reverse mode base offset: 0x0140
                    DPP_CTRL = 0x0140,
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

            struct HalfRowReverse
            {
            private:
                enum Traits : uint32_t
                {
                    // Reverse mode base offset: 0x0140
                    DPP_CTRL = 0x0141,
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

// GPU exclusion for unsupported targets, but assume host is valid
// for testing purposes.
#if !__gfx908__ // + Host

            template <uint32_t ElementIdx>
            struct RowBCast
            {
            private:
                enum Traits : uint32_t
                {
                    // Bcast mode base offset: 0x0150
                    DPP_CTRL = 0x0150 + ElementIdx,
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

#else // __gfx908__

            template <uint32_t ElementIdx>
            struct RowBCast
            {
            private:
                enum Traits : uint32_t
                {
                    DPP_CTRL = NOP::opCtrl(),
                };

            public:
                // clang-format off
                ROCWMMA_UNSUPPORTED_IMPL("amdgcn_dpp_row_bcast is not supported on gfx908")
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
                // clang-format on
            };

#endif // !__gfx908__

// GPU exclusion for unsupported targets, but assume host is valid
// for testing purposes.
#if !ROCWMMA_ARCH_GFX11 && !ROCWMMA_ARCH_GFX12 // + Host

            template <uint32_t ShiftDir>
            struct WaveShift1
            {
            private:
                enum Traits : uint32_t
                {
                    // Shift mode base offset: 0x0130
                    // Shift dir [3] : 0 = left, 1 = right
                    // wave_shift1_L = 0x0130
                    // wave_shift1_R = 0x0138
                    DPP_BASE  = 0x0130,
                    SHIFT_DIR = (ShiftDir & 0x1) << 3,

                    DPP_CTRL = DPP_BASE | SHIFT_DIR
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

            template <uint32_t RotateDir>
            struct WaveRotate1
            {
            private:
                enum Traits : uint32_t
                {
                    // Rotate mode base offset: 0x0134
                    // Rotate dir [3] : 0 = left, 1 = right
                    // wave_rotate1_L = 0x0134
                    // wave_rotate1_R = 0x013c
                    DPP_BASE   = 0x0134,
                    ROTATE_DIR = (RotateDir & 0x1) << 3,

                    DPP_CTRL = DPP_BASE | ROTATE_DIR
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

            struct RowBCast15
            {
            private:
                enum Traits : uint32_t
                {
                    // Bcast mode base offset: 0x0142
                    DPP_CTRL = 0x0142,
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

            struct RowBCast31
            {
            private:
                enum Traits : uint32_t
                {
                    // Bcast mode base offset: 0x0143
                    DPP_CTRL = 0x0143,
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

#else // ROCWMMA_ARCH_GFX11 || ROCWMMA_ARCH_GFX12

            template <uint32_t ShiftDir>
            struct WaveShift1
            {
            private:
                enum Traits : uint32_t
                {
                    DPP_CTRL = NOP::opCtrl(),
                };

            public:
                // clang-format off
                ROCWMMA_UNSUPPORTED_IMPL("amdgcn_dpp_wave_shift1 is not supported on gfx10+")
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
                // clang-format on
            };

            template <uint32_t RotateDir>
            struct WaveRotate1
            {
            private:
                enum Traits : uint32_t
                {
                    DPP_CTRL = NOP::opCtrl(),
                };

            public:
                // clang-format off
                ROCWMMA_UNSUPPORTED_IMPL("amdgcn_dpp_wave_rotate1 is not supported on gfx10+")
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
                // clang-format on
            };

            struct RowBCast15
            {
            private:
                enum Traits : uint32_t
                {
                    DPP_CTRL = NOP::opCtrl(),
                };

            public:
                // clang-format off
                ROCWMMA_UNSUPPORTED_IMPL("amdgcn_dpp_row_bcast15 is not supported on gfx10+")
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
                // clang-format on
            };

            struct RowBCast31
            {
            private:
                enum Traits : uint32_t
                {
                    DPP_CTRL = NOP::opCtrl(),
                };

            public:
                // clang-format off
                ROCWMMA_UNSUPPORTED_IMPL("amdgcn_dpp_row_bcast31 is not supported on gfx10+")
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
                // clang-format on
            };

#endif // !ROCWMMA_ARCH_GFX11 && !ROCWMMA_ARCH_GFX12

            using WaveRotateR1 = WaveRotate1<OP_DIR_R>;
            using WaveRotateL1 = WaveRotate1<OP_DIR_L>;
            using WaveShiftR1  = WaveShift1<OP_DIR_R>;
            using WaveShiftL1  = WaveShift1<OP_DIR_L>;

            // Derivatives
            template <uint32_t RotateDir, uint32_t RotateDistance>
            struct BankRotate
            {
            private:
                enum Traits : uint32_t
                {
                    ROTATE_DISTANCE
                    = (RotateDir == OP_DIR_L ? RotateDistance : 4u - RotateDistance),
                    SELECT_0 = (0u + ROTATE_DISTANCE) % 4u,
                    SELECT_1 = (1u + ROTATE_DISTANCE) % 4u,
                    SELECT_2 = (2u + ROTATE_DISTANCE) % 4u,
                    SELECT_3 = (3u + ROTATE_DISTANCE) % 4u,

                    DPP_CTRL = Shuffle4<SELECT_0, SELECT_1, SELECT_2, SELECT_3>::opCtrl()
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

            template <uint32_t RotateDistance>
            using BankRotateR = BankRotate<OP_DIR_R, RotateDistance>;
            template <uint32_t RotateDistance>
            using BankRotateL = BankRotate<OP_DIR_L, RotateDistance>;

            template <uint32_t RotateDir, uint32_t RotateDistance>
            struct HalfBankRotate
            {
            private:
                enum Traits : uint32_t
                {
                    ROTATE_DISTANCE
                    = (RotateDir == OP_DIR_L ? RotateDistance : 2u - RotateDistance),
                    SELECT_0 = (0u + ROTATE_DISTANCE) % 2u,
                    SELECT_1 = (1u + ROTATE_DISTANCE) % 2u,
                    SELECT_2 = 2u + SELECT_0,
                    SELECT_3 = 2u + SELECT_1,

                    DPP_CTRL = Shuffle4<SELECT_0, SELECT_1, SELECT_2, SELECT_3>::opCtrl()
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::DPP_CTRL;
                }
            };

            template <uint32_t RotateDistance>
            using HalfBankRotateR = HalfBankRotate<OP_DIR_R, RotateDistance>;
            template <uint32_t RotateDistance>
            using HalfBankRotateL = HalfBankRotate<OP_DIR_L, RotateDistance>;

        } // namespace Ctrl

        namespace OpsBase
        {
            /**
             * \ingroup Cross_Lane_Operations
             *
             * @brief Cross-lane operations implemented with the amdgcn_mov_dpp backend.
             * @note In this context:
             * 'row' means sub-group size of 16 elements. Wave64 has 4 rows, Wave32 has 2 rows per register.
             * 'bank' means sub-group size of 4 elements. There are 4 banks per row.
             *
             * DPP (Data Parallel Primitives) backend can implement specific variations of the cross-lane operations
             * so we can fully specialize those here.
             *
             * Here we build out the cross-lane properties specific to DPP, such as the backend (OP_IMPL) and the
             * control code drivers for the backend function call (OP_CTRL). Control code generators are implemented
             * in the DppCtrl namespace.
             *
             * @{
             */

            template <uint32_t OpId, uint32_t SubGroupSize>
            using DppOp = OpBase<OpId, SubGroupSize, OP_IMPL_DPP>;

            /*! \class BCast
            *  \brief Performs localized broadcast of one element in each sub-group to the entire sub-group.
            *
            * @tparam ElementIdx - element index to broadcast to rest of the sub-group
            */

            template <uint32_t ElementIdx, uint32_t SubGroupSize, class BCastCtrl>
            struct BCast : public DppOp<OP_ID_BCAST, SubGroupSize>,
                           public Backend::amdgcn_mov_dpp<BCastCtrl>
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

            /*! \class WFallBCast
            *  \brief Performs broadcast of the last sub-group element to the next sub-group.
            *
            * @tparam SubGroupSize - size of the broadcast blocks.
            */
            template <uint32_t SubGroupSize, class BCastCtrl>
            struct WFallBCast : public DppOp<OP_ID_WFALL_BCAST, SubGroupSize>,
                                public Backend::amdgcn_mov_dpp<BCastCtrl>
            {
            };

            /*! \class Move
            *  \brief Performs a copy of each element.
            *
            * @tparam SubGroupSize - size of the broadcast blocks.
            */
            struct Move : public DppOp<OP_ID_MOVE, OP_GROUP_SIZE_WARP>,
                          public Backend::amdgcn_mov_dpp<Ctrl::Move>
            {
            };

            /*! \class Reverse
            *  \brief Perform reversal of elements in sub-groups of \p SubGroupSize threads.
            */

            template <uint32_t SubGroupSize, class ReverseCtrl>
            struct Reverse : public DppOp<OP_ID_REVERSE, SubGroupSize>,
                             public Backend::amdgcn_mov_dpp<ReverseCtrl>
            {
            };

            /*! \class Rotate
            *  \brief Perform element-wise rotation in direction \p RotateDir in sub-groups of \p SubGroupSize threads.
            *
            * @tparam RotateDir rotation direction: see Properties
            * @tparam RotateDistance element positions to move in specified direction. Positions wrapped by sub group size.
            */
            template <uint32_t RotateDir,
                      uint32_t RotateDist,
                      uint32_t SubGroupSize,
                      class RotateCtrl>
            struct Rotate : public DppOp<OP_ID_ROTATE, SubGroupSize>,
                            public Backend::amdgcn_mov_dpp<RotateCtrl>
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

            template <uint32_t RotateDistance, uint32_t SubGroupSize, class RotateCtrl>
            using RotateR = Rotate<OP_DIR_R, RotateDistance, SubGroupSize, RotateCtrl>;

            template <uint32_t RotateDistance, uint32_t SubGroupSize, class RotateCtrl>
            using RotateL = Rotate<OP_DIR_L, RotateDistance, SubGroupSize, RotateCtrl>;

            /*! \class Shift
            *  \brief Perform element-wise shift in direction \p ShiftDir in sub-groups of \p SubGroupSize threads.
            *
            * @tparam ShiftDir shift direction: see Properties
            * @tparam ShiftDistance element positions to move in specified direction. Positions do not wrap around
            * the sub group size.
            */
            template <uint32_t ShiftDir, uint32_t ShiftDist, uint32_t SubGroupSize, class ShiftCtrl>
            struct Shift : public DppOp<OP_ID_SHIFT, SubGroupSize>,
                           public Backend::amdgcn_mov_dpp<ShiftCtrl>
            {
                enum : uint32_t
                {
                    OP_DIR  = ShiftDir,
                    OP_DIST = ShiftDist
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

            template <uint32_t ShiftDistance, uint32_t SubGroupSize, class ShiftCtrl>
            using ShiftR = Shift<OP_DIR_R, ShiftDistance, SubGroupSize, ShiftCtrl>;

            template <uint32_t ShiftDistance, uint32_t SubGroupSize, class ShiftCtrl>
            using ShiftL = Shift<OP_DIR_L, ShiftDistance, SubGroupSize, ShiftCtrl>;

            /*! \class Shuffle
            *  \brief Perform localized shuffling within sub-groups of \p SubGroupSize threads.
            */
            template <uint32_t SubGroupSize, class ShuffleCtrl>
            struct Shuffle : public DppOp<OP_ID_SHUFFLE, SubGroupSize>,
                             public Backend::amdgcn_mov_dpp<ShuffleCtrl>
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

            /*! \class Shuffle2
            *  \brief Shuffle\<N\> Perform localized shuffling within all sub-groups of \em N threads.
            * \em N = group size.
            *
            * @tparam Select0 - index of element to shuffle to index 0
            * @tparam Select1 - index of element to shuffle to index 1
            */
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
            template <uint32_t SubGroupSize, class SwapCtrl>
            struct Swap : public DppOp<OP_ID_SWAP, SubGroupSize>,
                          public Backend::amdgcn_mov_dpp<SwapCtrl>
            {
            };

            /*! \class Zip
            *  \brief Perform blending of sub-groups of \p SubGroupSize threads between sources.
            */
            template <uint32_t SubGroupSize>
            struct Zip : public DppOp<OP_ID_BLEND, SubGroupSize>,
                         public Backend::amdgcn_mov_dpp<Ctrl::Move>
            {
            };
            /** @}*/

        } // namespace OpsBase

        namespace Ops
        {
            // clang-format off

            // BCast variants
            template <uint32_t ElementIdx>
            using BCast16 = OpsBase::BCast<ElementIdx, OP_GROUP_SIZE_16, Ctrl::RowBCast<ElementIdx>>;

            template <uint32_t ElementIdx>
            using BCast4 = OpsBase::BCast<ElementIdx, OP_GROUP_SIZE_4, Ctrl::Shuffle4<ElementIdx, ElementIdx, ElementIdx, ElementIdx>>;

            template <uint32_t ElementIdx>
            using BCast2 = OpsBase::BCast<ElementIdx, OP_GROUP_SIZE_2, Ctrl::Shuffle4<ElementIdx, ElementIdx, ElementIdx + OP_GROUP_SIZE_2, ElementIdx + OP_GROUP_SIZE_2>>;

            // Special BCast variants:
            // BCast<M>x<N>, where:
            // <M> = subgroup size
            // <N> = element idx
            // NOTE: These functions only broadcast the <N>th element of the current subgroup to the NEXT subgroup
            using BCast16x15 = OpsBase::WFallBCast<OP_GROUP_SIZE_16, Ctrl::RowBCast15>;

            using BCast32x31 = OpsBase::WFallBCast<OP_GROUP_SIZE_32, Ctrl::RowBCast31>;

            // Move variants

            using MaskMove = OpsBase::Move;

            /// Reversal variants
            using Reverse16 = OpsBase::Reverse<OP_GROUP_SIZE_16, Ctrl::RowReverse>;

            using Reverse8 = OpsBase::Reverse<OP_GROUP_SIZE_8, Ctrl::HalfRowReverse>;

            using Reverse4 = OpsBase::Reverse<OP_GROUP_SIZE_4, Ctrl::Shuffle4<0x3, 0x2, 0x1, 0x0>>;

            using Reverse2 = OpsBase::Reverse<OP_GROUP_SIZE_2, Ctrl::Shuffle4<0x1, 0x0, 0x3, 0x2>>;


            /// Rotation variants

            // Rotate the entire wave by 1
            using RotateWaveR1 = OpsBase::RotateR<1u, OP_GROUP_SIZE_WARP, Ctrl::WaveRotateR1>;

            using RotateWaveL1 = OpsBase::RotateL<1u, OP_GROUP_SIZE_WARP, Ctrl::WaveRotateL1>;

            // Rotate in element groups
            template <uint32_t RotateDistance>
            using RotateR16 = OpsBase::RotateR<RotateDistance, OP_GROUP_SIZE_16, Ctrl::RowRotateR<RotateDistance>>;

            template <uint32_t RotateDistance>
            using RotateL4 = OpsBase::RotateL<RotateDistance, OP_GROUP_SIZE_4, Ctrl::BankRotateL<RotateDistance>>;

            template <uint32_t RotateDistance>
            using RotateR4 = OpsBase::RotateR<RotateDistance, OP_GROUP_SIZE_4, Ctrl::BankRotateR<RotateDistance>>;

            template <uint32_t RotateDistance>
            using RotateL2 = OpsBase::RotateL<RotateDistance, OP_GROUP_SIZE_2, Ctrl::HalfBankRotateL<RotateDistance>>;

            template <uint32_t RotateDistance>
            using RotateR2 = OpsBase::RotateR<RotateDistance, OP_GROUP_SIZE_2, Ctrl::HalfBankRotateR<RotateDistance>>;

            /// Shift variants

            // Rotate the entire wave by 1
            using ShiftWaveL1 = OpsBase::ShiftL<1u, OP_GROUP_SIZE_WARP, Ctrl::WaveShiftL1>;

            using ShiftWaveR1 = OpsBase::ShiftR<1u, OP_GROUP_SIZE_WARP, Ctrl::WaveShiftR1>;

            // Rotate in element groups
            template <uint32_t ShiftDistance>
            using ShiftL16 = OpsBase::ShiftL<ShiftDistance, OP_GROUP_SIZE_16, Ctrl::RowShiftL<ShiftDistance>>;

            template <uint32_t ShiftDistance>
            using ShiftR16 = OpsBase::ShiftR<ShiftDistance, OP_GROUP_SIZE_16, Ctrl::RowShiftR<ShiftDistance>>;

            /// Shuffle variants
            template <uint32_t Select0, uint32_t Select1, uint32_t Select2, uint32_t Select3>
            using Shuffle4 = OpsBase::Shuffle4<Select0, Select1, Select2, Select3>;

            template <uint32_t Select0, uint32_t Select1>
            using Shuffle2 = OpsBase::Shuffle2<Select0, Select1>;

            // Swap variants
            using Swap2 = OpsBase::Swap<OP_GROUP_SIZE_2, Ctrl::Shuffle4<0x02, 0x03, 0x00, 0x01>>;

            // Zip variants
            using Zip4 = OpsBase::Zip<OP_GROUP_SIZE_4>;

            using Zip8 = OpsBase::Zip<OP_GROUP_SIZE_8>;

            using Zip16 = OpsBase::Zip<OP_GROUP_SIZE_16>;

            using Zip32 = OpsBase::Zip<OP_GROUP_SIZE_32>;

            // clang-format on

        } // namespace Ops

    } // namespace DppImpl

} // namespace rocwmma

#endif // ROCWMMA_MOVE_DPP_IMPL_HPP
