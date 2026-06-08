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
#ifndef ROCWMMA_BLEND_IMPL_HPP
#define ROCWMMA_BLEND_IMPL_HPP

#include "utility/vector.hpp"
#include "utils.hpp"

namespace rocwmma
{

    namespace BlendImpl
    {
        // Implementation meta-data
        using CrossLaneOps::OpBase;
        using CrossLaneOps::Properties;

        // Dpp backend
        using Properties::OP_IMPL_VBLEND;
        using Properties::OP_IMPL_VPERM;

        // Functional
        using Properties::OP_ID_BLEND;
        using Properties::OP_ID_PERM_BYTE;

        // Groups
        using Properties::OP_GROUP_SIZE_1;
        using Properties::OP_GROUP_SIZE_16;
        using Properties::OP_GROUP_SIZE_2;
        using Properties::OP_GROUP_SIZE_32;
        using Properties::OP_GROUP_SIZE_4;
        using Properties::OP_GROUP_SIZE_8;

        namespace Backend
        {
            /*! \class amdgcn_perm
            *  \brief Implements the element blending backend.
            * This backend provides sub-b32 permutation between two sources. Blending in this
            * context means that ordered pairs of 32b elements (src0, src1) may be permuted to
            * generate a 32b element result for every lane.
            *
            * E.g. Byte-wise permutation of ordered source elements:
            * Result[31:0] = Src0[7:0],Src0[7:0],Src1[15:8],Src1[23:16]
            *
            * @tparam opCtrl is a generator class to create selection control for each byte
            * @arg src0 is the 'lower' reference
            * @arg src1 is the 'upper' reference
            */
            template <class OpCtrl>
            struct amdgcn_perm : OpCtrl
            {
                template <typename DataT>
                ROCWMMA_DEVICE static inline DataT exec(DataT src0, DataT src1)
                {
                    static_assert(sizeof(DataT) == sizeof(uint32_t), "Inputs must be 32 bit");

                    // NOTE: src0 and src1 are flipped here due to spec's select
                    // concatenation of i[3:0] = src1 and i[7:4] = src0 .
                    // amdgcn_blend_byte does the inverse of this to make
                    // the rocWMMA interface more intuitive, with src0 as the lower
                    // 4 bytes and src1 in the upper 4 bytes.
                    reinterpret_cast<uint32_t&>(src0)
                        = __builtin_amdgcn_perm(reinterpret_cast<uint32_t&>(src1),
                                                reinterpret_cast<uint32_t&>(src0),
                                                OpCtrl::opCtrl());
                    return src0;
                }

                // This function takes an opCtrl override to support pre-calculated opCtrl re-use
                template <typename DataT>
                ROCWMMA_DEVICE static inline DataT exec(DataT src0, DataT src1, uint32_t opCtrl)
                {
                    static_assert(sizeof(DataT) == sizeof(uint32_t), "Inputs must be 32 bit");

                    // NOTE: src0 and src1 are flipped here due to spec's select
                    // concatenation of i[3:0] = src1 and i[7:4] = src0 .
                    // amdgcn_blend_byte does the inverse of this to make
                    // the rocWMMA interface more intuitive, with src0 as the lower
                    // 4 bytes and src1 in the upper 4 bytes.
                    reinterpret_cast<uint32_t&>(src0)
                        = __builtin_amdgcn_perm(reinterpret_cast<uint32_t&>(src1),
                                                reinterpret_cast<uint32_t&>(src0),
                                                opCtrl);
                    return src0;
                }
            };

            /*! \class amdgcn_blend
            *  \brief Implements the bitwise blend backend between 32b elements of two src vectors.
            * Blend means bit-wise ordered combination of two source vector elements, in a mutally exclusive
            * fashion, as in either / or, without permutation.
            *
            * E.g. res = src0 <= mask = 0x00000000
            *      res = src1 <= mask = 0xFFFFFFFF
            *      res = select_bits(src0, src1) <= mask 0's take bits of src0 and mask1's take bits of src1
            *
            * @tparam maskCtrl is a generator class to create selection mask for each bit
            * @arg src0 is the 'lower' source (e.g. mask = 0)
            * @arg src1 is the 'upper' source (e.g. mask = 1)
            */
            template <class OpCtrl>
            struct amdgcn_blend : OpCtrl
            {
                template <typename DataT>
                ROCWMMA_DEVICE static inline DataT exec(DataT src0, DataT src1)
                {
                    static_assert(sizeof(DataT) % sizeof(uint32_t) == 0,
                                  "Inputs must be a multiple of 32 bit");
                    uint32_t const maskCtrl = OpCtrl::opCtrl();
                    return maskCtrl ? src1 : src0;
                }

                // This function takes an opCtrl override to support pre-calculated opCtrl re-use
                template <typename DataT>
                ROCWMMA_DEVICE static inline DataT exec(DataT src0, DataT src1, uint32_t maskCtrl)
                {
                    static_assert(sizeof(DataT) == sizeof(uint32_t), "Inputs must be 32 bit");
                    // TODO:: WHY?
                    // reinterpret_cast<uint32_t&>(src0)
                    //     = (reinterpret_cast<uint32_t&>(src1) & maskCtrl)
                    //       | (reinterpret_cast<uint32_t&>(src0) & ~maskCtrl);
                    // return src0;
                    return maskCtrl ? src1 : src0;
                }
            };

        } // namespace backend

        namespace Ctrl
        {
            // Ctrl generator for amdgcn_perm backend
            template <uint32_t Select0, uint32_t Select1, uint32_t Select2, uint32_t Select3>
            struct PermByte
            {
            private:
                enum Traits : uint32_t
                {
                    // Byte blend mode: for each 32b element, select the 8b ordering.
                    // 0u <= select_index < 4u  = src0
                    // 4u <= select_index < 8u  = src1
                    //
                    // dst[7:0]   = SELECT_BYTE_0{ concat(src1[31:0], src0[31:0]) }
                    // dst[15:8]  = SELECT_BYTE_1{ concat(src1[31:0], src0[31:0]) }
                    // dst[24:16] = SELECT_BYTE_2{ concat(src1[31:0], src0[31:0]) }
                    // dst[31:23] = SELECT_BYTE_3{ concat(src1[31:0], src0[31:0]) }
                    BLEND_MODE = 0x0,
                    SELECT_0   = (Select0 & 0xFF),
                    SELECT_1   = (Select1 & 0xFF) << 8,
                    SELECT_2   = (Select2 & 0xFF) << 16,
                    SELECT_3   = (Select3 & 0xFF) << 24,

                    BLEND_CTRL = BLEND_MODE | SELECT_0 | SELECT_1 | SELECT_2 | SELECT_3
                };

            public:
                constexpr static uint32_t opCtrl()
                {
                    return Traits::BLEND_CTRL;
                }
            };

            // Ctrl generator for amdgcn_blend backend
            template <uint32_t GroupSize>
            struct BlendElements
            {
            public:
                constexpr static bool opCtrl()
                {
                    // Just like a zipper, if threadIdx.x in [0, GroupSize - 1], return 0,
                    // if threadIdx.x in [GroupSize, 2 * GroupSize - 1], return 1
                    // and so on.
                    return (threadIdx.x >> Log2<GroupSize>::value) & 0x1;
                }
            };

        } // namespace Ctrl

        namespace Ops
        {
            /*! \class PermByte
            *  \brief Perform byte-wise permute between two sources.
            * Uses the VPerm backend, therefore must have visible opCtrl() function.
            *
            * Adds access element fuctions select0-3
            * Inherits exec() function of the backend with an interface for two input sources.
            */
            template <uint32_t Select0, uint32_t Select1, uint32_t Select2, uint32_t Select3>
            struct PermByte
                : public OpBase<OP_ID_PERM_BYTE, OP_GROUP_SIZE_1, OP_IMPL_VPERM>,
                  Backend::amdgcn_perm<Ctrl::PermByte<Select0, Select1, Select2, Select3>>
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

            /*! \class PermWord
            *  \brief Perform word-wise permute between two sources.
            * Uses the VPerm backend, therefore must have visible opCtrl() function.
            *
            * Adds access element fuctions select0-1
            * Inherits exec() function of the backend with an interface for two input sources.
            */
            template <uint32_t Select0, uint32_t Select1>
            struct PermWord
                : public PermByte<Select0 * 2u, Select0 * 2u + 1u, Select1 * 2u, Select1 * 2u + 1u>
            {
            };

            /*! \class ZipBlend
            *  \brief Perform element-wise blending between two sources.
            * Uses the VBlend backend, therefore must have visible maskCtrl() function.
            *
            * Inherits exec() function of the backend with an interface for two input sources.
            */
            template <uint32_t SubGroupSize>
            struct Zip : public OpBase<OP_ID_BLEND, SubGroupSize, OP_IMPL_VBLEND>,
                         Backend::amdgcn_blend<Ctrl::BlendElements<SubGroupSize>>
            {
            };

            // Blend even bytes from src0 and odd bytes from src1
            using ZipByte = PermByte<0u, 5u, 2u, 7u>;
            using ZipWord = PermWord<0u, 3u>;
            using Zip1    = Zip<OP_GROUP_SIZE_1>;
            using Zip2    = Zip<OP_GROUP_SIZE_2>;
            using Zip4    = Zip<OP_GROUP_SIZE_4>;
            using Zip8    = Zip<OP_GROUP_SIZE_8>;
            using Zip16   = Zip<OP_GROUP_SIZE_16>;
            using Zip32   = Zip<OP_GROUP_SIZE_32>;

            // Blend sub-dword elements in regular ordered patterns
            using UnpackByteLo     = PermByte<0u, 4u, 1u, 5u>;
            using UnpackByteHi     = PermByte<2u, 6u, 3u, 7u>;
            using UnpackWordLo     = PermWord<0u, 2u>;
            using UnpackWordHi     = PermWord<1u, 3u>;
            using UnpackByteLoHi   = PermByte<0u, 6u, 1u, 7u>;
            using UnpackByte3BCast = PermByte<3u, 7u, 3u, 7u>;

            using ExtractByteEven = PermByte<0u, 2u, 4u, 6u>;
            using ExtractByteOdd  = PermByte<1u, 3u, 5u, 7u>;
            using ExtractWordEven = UnpackWordLo;
            using ExtractWordOdd  = UnpackWordHi;

            using ExtractByteEvenOdd = PermByte<0u, 2u, 5u, 7u>;
            using ExtractByteOddEven = PermByte<1u, 3u, 4u, 6u>;
            using ExtractWordEvenOdd = PermWord<0u, 3u>;
            using ExtractWordOddEven = PermWord<1u, 2u>;
        } // namespace Ops

    } // namespace BlendImpl

} // namespace rocwmma

#endif // ROCWMMA_BLEND_IMPL_HPP
