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
#ifndef ROCWMMA_BLEND_HPP
#define ROCWMMA_BLEND_HPP

// clang-format off
#include "cross_lane_ops.hpp"
#include "blend_impl.hpp"
#include "vector.hpp"
// clang-format on

namespace rocwmma
{
    namespace Blend
    {

        template <typename BlendOp>
        struct Driver
        {
            // Sanity checks
            static_assert((BlendOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_VPERM)
                              || (BlendOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_VBLEND),
                          "BlendOp must use vperm or blend backend");
            static_assert((BlendOp::opId() == CrossLaneOps::Properties::OP_ID_BLEND)
                              || (BlendOp::opId() == CrossLaneOps::Properties::OP_ID_PERM_BYTE),
                          "BlendOp is unsupported");

            template <typename DataT>
            ROCWMMA_DEVICE static inline auto exec(DataT const& src0, DataT const& src1)
            {
                // Vectorize to B32.
                // This way we can support B64+ types
                using B32VecT = VecT<uint32_t, sizeof(DataT) / sizeof(uint32_t)>;

                // Ensure that we can vectorize to B32
                static_assert(sizeof(DataT) % sizeof(uint32_t) == 0,
                              "DataT size must be a multiple of B32");
                static_assert(sizeof(B32VecT) == sizeof(DataT), "Unable to vectorize DataT");

                // Forward to vectorized function
                auto result = exec(reinterpret_cast<B32VecT const&>(src0),
                                   reinterpret_cast<B32VecT const&>(src1));

                // Restore result to input type
                return reinterpret_cast<DataT&>(result);
            }

            template <typename DataT, uint32_t VecSize>
            ROCWMMA_DEVICE static inline auto exec(VecT<DataT, VecSize> const& src0,
                                                   DataT const&                src1)
            {
                // Reinterpret vector as B32 so we can support B64+ elements.
                constexpr uint32_t B32VecSize = sizeof(DataT) / sizeof(uint32_t) * VecSize;
                using B32VecT                 = VecT<uint32_t, B32VecSize>;
                using InputVecT               = VecT<DataT, VecSize>;

                // Ensure that we can vectorize src0 to B32
                static_assert(sizeof(InputVecT) % sizeof(uint32_t) == 0,
                              "VecT size must be a multiple of B32");
                static_assert(sizeof(B32VecT) == sizeof(InputVecT),
                              "Unable to vectorize src0 to B32");

                // Reinterpret scalar as B32 vector so we can support B64+ elements.
                constexpr uint32_t B32ScalarVecSize = sizeof(DataT) / sizeof(uint32_t);
                using B32ScalarVecT                 = VecT<uint32_t, B32ScalarVecSize>;

                // Ensure that we can vectorize src1 to B32
                static_assert(sizeof(B32ScalarVecT) % sizeof(uint32_t) == 0,
                              "DataT size must be a multiple of B32");
                static_assert(sizeof(B32ScalarVecT) == sizeof(DataT), "Unable to vectorize src1");

                auto op = [](auto&& idx, auto&& v0, auto&& s, auto&& opCtrl) {
                    // Pair up the b32 vector elements with the appropriate b32 scalar elements.
                    constexpr auto i = decay_t<decltype(idx)>::value;
                    return BlendOp::exec(get<i>(v0), get<i % B32ScalarVecSize>(s), opCtrl);
                };

                // Static unroll with cached opCtrl
                auto result = vector_generator<uint32_t, B32VecSize>()(
                    op,
                    reinterpret_cast<B32VecT const&>(src0),
                    reinterpret_cast<B32ScalarVecT const&>(src1),
                    BlendOp::opCtrl());

                // Restore result to input type
                return reinterpret_cast<InputVecT&>(result);
            }

            template <typename DataT, uint32_t VecSize>
            ROCWMMA_DEVICE static inline auto exec(VecT<DataT, VecSize> const& src0,
                                                   VecT<DataT, VecSize> const& src1)
            {
                // Reinterpret vectors as B32 so we can support B64+ elements.
                constexpr uint32_t B32VecSize = sizeof(DataT) / sizeof(uint32_t) * VecSize;
                using B32VecT                 = VecT<uint32_t, B32VecSize>;
                using InputVecT               = VecT<DataT, VecSize>;
                static_assert(sizeof(InputVecT) % sizeof(uint32_t) == 0,
                              "VecT size must be a multiple of B32");
                static_assert(sizeof(B32VecT) == sizeof(InputVecT),
                              "Unable to vectorize src0 / src1 to B32");

                auto op = [](auto&& idx, auto&& v0, auto&& v1, auto&& opCtrl) {
                    // Pair up the b32 vector elements
                    constexpr auto i = decay_t<decltype(idx)>::value;
                    return BlendOp::exec(get<i>(v0), get<i>(v1), opCtrl);
                };

                // Static unroll with cached opCtrl
                auto result = vector_generator<uint32_t, B32VecSize>()(
                    op,
                    reinterpret_cast<B32VecT const&>(src0),
                    reinterpret_cast<B32VecT const&>(src1),
                    BlendOp::opCtrl());

                // Restore result to input type
                return reinterpret_cast<InputVecT&>(result);
            }
        };

        /// Blend ops interface
        // Func::exec(src0, src1)

        // Zip functions
        // Blend even thread elements from src0 with odd thread elements from src1,
        // just like a zipper. We can reverse the order of the inputs if the opposite is desired.
        // E.g. src0 = [0, 1, 2, 3] src1 = [4, 5, 6, 7]
        // Zip1 = [0, 5, 2, 7]
        // Zip2 = [0, 1, 6, 7]

        /*! \class ZipByte
        *  \brief  Blend class that alternates bytes from src0 and src1
        */
        using ZipByte = Driver<BlendImpl::Ops::ZipByte>;

        /*! \class ZipWord
        *  \brief  Blend class that alternates words (2B) from src0 and src1
        */
        using ZipWord = Driver<BlendImpl::Ops::ZipWord>;

        /*! \class Zip1
        *  \brief  Blend class that alternates thread elements (4B) from src0 and src1.
        */
        using Zip1 = Driver<BlendImpl::Ops::Zip1>;

        /*! \class Zip2
        *  \brief  Blend class that alternates groups of 2 thread elements (8B) from src0 and src1.
        */
        using Zip2 = Driver<BlendImpl::Ops::Zip2>;

        /*! \class Zip4
        *  \brief  Blend class that alternates groups of 4 thread elements (16B) from src0 and src1.
        */
        using Zip4 = Driver<BlendImpl::Ops::Zip4>;

        /*! \class Zip8
        *  \brief  Blend class that alternates groups of 8 thread elements (32B) from src0 and src1.
        */
        using Zip8 = Driver<BlendImpl::Ops::Zip8>;

        /*! \class Zip16
        *  \brief  Blend class that alternates groups of 16 thread elements (64B) from src0 and src1.
        */
        using Zip16 = Driver<BlendImpl::Ops::Zip16>;

        /*! \class Zip32
        *  \brief  Blend class that alternates groups of 32 thread elements (128B) from src0 and src1.
        */
        using Zip32 = Driver<BlendImpl::Ops::Zip32>;

        // Unpack functions
        // The unpack functionality blends elements from the same thread in src0 and src1.
        // Because there is only one result from 2 sources, we must choose to blend either the low, or the
        // high elements per group of inputs.
        // E.g. src0 = [0, 1, 2, 3] src1 = [4, 5, 6, 7]
        // UnpackLo   = [0, 4, 1, 5]
        // UnpackHi   = [2, 6, 3, 7]
        // UnpackLoHi = [0, 4, 2, 6]

        /*! \class UnpackByteLo
        *  \brief  Blend class that unpacks lo bytes from each 4B element of src0 and src1.
        */
        using UnpackByteLo = Driver<BlendImpl::Ops::UnpackByteLo>;

        /*! \class UnpackByteHi
        *  \brief  Blend class that unpacks hi bytes from each 4B element of src0 and src1.
        */
        using UnpackByteHi = Driver<BlendImpl::Ops::UnpackByteHi>;

        /*! \class UnpackWordLo
        *  \brief  Blend class that unpacks lo words from each 4B element of src0 and src1.
        */
        using UnpackWordLo = Driver<BlendImpl::Ops::UnpackWordLo>;

        /*! \class UnpackWordHi
        *  \brief  Blend class that unpacks hi words from each 4B element of src0 and src1.
        */
        using UnpackWordHi = Driver<BlendImpl::Ops::UnpackWordHi>;

        /*! \class UnpackByteLoHi
        *  \brief  Blend class that unpacks two lo bytes followed by two hi bytes from each 4B element of src0 and src1.
        */
        using UnpackByteLoHi = Driver<BlendImpl::Ops::UnpackByteLoHi>;

        /*! \class UnpackByte3BCast
        *  \brief  Blend class that unpacks byte 3 from each 4B element of src0 and src1, which is broadcasted.
        *  e.g. src0[0]   = [0, 1, 2, 3] src1[0] = [4, 5, 6, 7]
        *       result[0] = [3, 7, 3, 7]
        */
        using UnpackByte3BCast = Driver<BlendImpl::Ops::UnpackByte3BCast>;

        // Extract functions
        // These functions extract elements in order from src0 and src1 and concatenate their result together.
        // e.g. src0   = [0, 1, 2, 3] src1 = [4, 5, 6, 7]
        // extract_even      = [0, 2, 4, 6]
        // extract_odd       = [1, 3, 5, 7]
        // extract_even_odd  = [0, 2, 5, 7]
        // extract_odd_even  = [1, 3, 4, 6]

        /*! \class ExtractByteEven
        *  \brief  Blend class that extracts even ordered bytes from each 4B element of src0 and src1 and
        *  concatenates them together.
        */
        using ExtractByteEven = Driver<BlendImpl::Ops::ExtractByteEven>;

        /*! \class ExtractByteOdd
        *  \brief  Blend class that extracts odd ordered bytes from each 4B element of src0 and src1 and
        *  concatenates them together.
        */
        using ExtractByteOdd = Driver<BlendImpl::Ops::ExtractByteOdd>;

        /*! \class ExtractWordEven
        *  \brief  Blend class that extracts even ordered words from each 4B element of src0 and src1 and
        *  concatenates them together.
        */
        using ExtractWordEven = Driver<BlendImpl::Ops::ExtractWordEven>;

        /*! \class ExtractWordOdd
        *  \brief  Blend class that extracts odd ordered words from each 4B element of src0 and src1 and
        *  concatenates them together.
        */
        using ExtractWordOdd = Driver<BlendImpl::Ops::ExtractWordOdd>;

        /*! \class ExtractByteEvenOdd
        *  \brief  Blend class that extracts even ordered bytes src0, odd ordered bytes from src1 and
        *  concatenates them together.
        */
        using ExtractByteEvenOdd = Driver<BlendImpl::Ops::ExtractByteEvenOdd>;

        /*! \class ExtractWordEvenOdd
        *  \brief  Blend class that extracts even ordered words src0, odd ordered words from src1 and
        *  concatenates them together.
        */
        using ExtractWordEvenOdd = Driver<BlendImpl::Ops::ExtractWordEvenOdd>;

        /*! \class ExtractByteOddEven
        *  \brief  Blend class that extracts odd ordered bytes src0, even ordered bytes from src1 and
        *  concatenates them together.
        */
        using ExtractByteOddEven = Driver<BlendImpl::Ops::ExtractByteOddEven>;

        /*! \class ExtractWordOddEven
        *  \brief  Blend class that extracts odd ordered words src0, even ordered words from src1 and
        *  concatenates them together.
        */
        using ExtractWordOddEven = Driver<BlendImpl::Ops::ExtractWordOddEven>;

    } // namespace Blend

} // namespace rocwmma

#endif // ROCWMMA_BLEND_HPP
