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
#ifndef ROCWMMA_SWIZZLE_HPP
#define ROCWMMA_SWIZZLE_HPP

#include "cross_lane_ops.hpp"
#include "swizzle_impl.hpp"
#include "vector.hpp"

namespace rocwmma
{
    namespace Swizzle
    {
        /**
         * \ingroup Cross_Lane_Operations
         * @{
         *
         * @brief Cross-lane operations implemented with the amdgcn_ds_swizzle backend.
         *
         * This function does not use LDS memory, but does use the LDS hardware, therefore it will
         * implicitly require lgkmcnt waits. This function is significantly faster than reading or
         * writing to the LDS memory, but does introduce lgkmcnt dependence for data cohesion.
         * This is not the most performant backend (dpp can be faster), however there is better
         * flexibility with swizzle when necessary. Fft swizzle is also unique to this backend.
         *
         * The swizzle backend offers a set of cross-lane support:
         * [X, Y] = Includes X and Y
         * [X - Y] = X, powers of 2 in between and including Y
         *
         * BCast (Subgroups[2 - 32])
         * Reverse (Subgroups[2 - 32])
         * Rotate (L/R, Subgroups[2 - 32])
         * Shuffle (Subgroups[2, 4])
         * Swap (Subgroups[2 - 16])
         *
         * Fft (FftCtrl [0x00 - 0x1F]) -> See ISA for swizzle fft codes
         *
         * The swizzle backend does not support Shift.
         */

        /*! \class Swizzle
        *  \brief A front-end utility that invokes swizzle operations on input data.
        *
        * @tparam SwizzleOp - fully qualified op class that generates the OP_CTRL code to apply to Dpp function.
        */
        template <typename SwizzleOp>
        struct Driver
        {
        public:
            // Sanity checks
            static_assert(SwizzleOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_SWIZZLE,
                          "SwizzleOp must use swizzle backend");
            static_assert((SwizzleOp::opId() == CrossLaneOps::Properties::OP_ID_ROTATE)
                              || (SwizzleOp::opId() == CrossLaneOps::Properties::OP_ID_SHUFFLE)
                              || (SwizzleOp::opId() == CrossLaneOps::Properties::OP_ID_REVERSE)
                              || (SwizzleOp::opId() == CrossLaneOps::Properties::OP_ID_SWAP)
                              || (SwizzleOp::opId() == CrossLaneOps::Properties::OP_ID_BCAST)
                              || (SwizzleOp::opId() == CrossLaneOps::Properties::OP_ID_FFT),
                          "SwizzleOp is unsupported");

            template <typename DataT>
            ROCWMMA_DEVICE static inline auto exec(DataT const& src)
            {
                // Vectorize to B32.
                // This way we can support B64+ types
                using B32VecT = VecT<uint32_t, sizeof(DataT) / sizeof(uint32_t)>;

                // Ensure that we can vectorize to B32
                static_assert(sizeof(DataT) % sizeof(uint32_t) == 0,
                              "DataT size must be a multiple of B32");
                static_assert(sizeof(B32VecT) == sizeof(DataT), "Unable to vectorize DataT");

                // Forward to vectorized function
                auto result = exec(reinterpret_cast<B32VecT const&>(src));

                // Restore result to input type
                return reinterpret_cast<DataT&>(result);
            }

            template <typename DataT, uint32_t VecSize>
            ROCWMMA_DEVICE static inline auto exec(VecT<DataT, VecSize> const& src)
            {
                constexpr uint32_t B32VecSize = sizeof(src) / sizeof(uint32_t);
                using B32VecT                 = VecT<uint32_t, B32VecSize>;

                // Ensure that we can vectorize to B32
                static_assert(sizeof(DataT) % sizeof(uint32_t) == 0,
                              "DataT size must be a multiple of B32");
                static_assert(sizeof(B32VecT) == sizeof(src), "Unable to vectorize DataT");

                auto op = [](auto&& idx, auto&& v) {
                    constexpr auto i = decay_t<decltype(idx)>::value;
                    return SwizzleOp::exec(get<i>(v));
                };

                auto swizzle_result = vector_generator<uint32_t, B32VecSize>()(
                    op, reinterpret_cast<B32VecT const&>(src));
                return reinterpret_cast<VecT<DataT, VecSize> const&>(swizzle_result);
            }
        };

        /// Swizzle ops interface
        // Func::exec(src0)

        // BCast variants

        /*! \class BCast32
        *  \brief  Swizzle class that broadcasts one thread to all threads in each group of 32
        *  @tparam ElementIdx thread index [0-31]
        */
        template <uint32_t ElementIdx>
        using BCast32 = Driver<SwizzleImpl::Ops::BCast32<ElementIdx>>;

        /*! \class BCast16
        *  \brief  Swizzle class that broadcasts one thread to all threads in each group of 16
        *  @tparam ElementIdx thread index [0-15]
        */
        template <uint32_t ElementIdx>
        using BCast16 = Driver<SwizzleImpl::Ops::BCast16<ElementIdx>>;

        /*! \class BCast8
        *  \brief  Swizzle class that broadcasts one thread to all threads in each group of 8
        *  @tparam ElementIdx thread index [0-7]
        */
        template <uint32_t ElementIdx>
        using BCast8 = Driver<SwizzleImpl::Ops::BCast8<ElementIdx>>;

        /*! \class BCast4
        *  \brief  Swizzle class that broadcasts one thread to all threads in each group of 4
        *  @tparam ElementIdx thread index [0-3]
        */
        template <uint32_t ElementIdx>
        using BCast4 = Driver<SwizzleImpl::Ops::BCast4<ElementIdx>>;

        /*! \class BCast2
        *  \brief  Swizzle class that broadcasts one thread to all threads in each group of 2
        *  @tparam ElementIdx thread index [0-1]
        */
        template <uint32_t ElementIdx>
        using BCast2 = Driver<SwizzleImpl::Ops::BCast2<ElementIdx>>;

        // Reverse variants

        /*! \class Reverse32
        *  \brief  Swizzle class that mirrors all threads in each group of 32
        */
        using Reverse32 = Driver<SwizzleImpl::Ops::Reverse32>;

        /*! \class Reverse16
        *  \brief  Swizzle class that mirrors all threads in each group of 16
        */
        using Reverse16 = Driver<SwizzleImpl::Ops::Reverse16>;

        /*! \class Reverse8
        *  \brief  Swizzle class that mirrors all threads in each group of 8
        */
        using Reverse8 = Driver<SwizzleImpl::Ops::Reverse8>;

        /*! \class Reverse4
        *  \brief  Swizzle class that mirrors all threads in each group of 4
        */
        using Reverse4 = Driver<SwizzleImpl::Ops::Reverse4>;

        /*! \class Reverse2
        *  \brief  Swizzle class that mirrors all threads in each group of 2
        */
        using Reverse2 = Driver<SwizzleImpl::Ops::Reverse2>;

        // Rotate variants

        /*! \class RotateL32
        *  \brief  Swizzle class that rotates all threads to the left in each group of 32
        *  @tparam RotateDistance thread index [0-31]
        */
        template <uint32_t RotateDistance>
        using RotateL32 = Driver<SwizzleImpl::Ops::RotateL32<RotateDistance>>;

        /*! \class RotateL16
        *  \brief  Swizzle class that rotates all threads to the left in each group of 16
        *  @tparam RotateDistance thread index [0-15]
        */
        template <uint32_t RotateDistance>
        using RotateL16 = Driver<SwizzleImpl::Ops::RotateL16<RotateDistance>>;

        /*! \class RotateL8
        *  \brief  Swizzle class that rotates all threads to the left in each group of 8
        *  @tparam RotateDistance thread index [0-7]
        */
        template <uint32_t RotateDistance>
        using RotateL8 = Driver<SwizzleImpl::Ops::RotateL8<RotateDistance>>;

        /*! \class RotateL4
        *  \brief  Swizzle class that rotates all threads to the left in each group of 4
        *  @tparam RotateDistance thread index [0-3]
        */
        template <uint32_t RotateDistance>
        using RotateL4 = Driver<SwizzleImpl::Ops::RotateL4<RotateDistance>>;

        /*! \class RotateL2
        *  \brief  Swizzle class that rotates all threads to the left in each group of 2
        *  @tparam RotateDistance thread index [0-1]
        */
        template <uint32_t RotateDistance>
        using RotateL2 = Driver<SwizzleImpl::Ops::RotateL2<RotateDistance>>;

        /*! \class RotateR32
        *  \brief  Swizzle class that rotates all threads to the right in each group of 32
        *  @tparam RotateDistance thread index [0-31]
        */
        template <uint32_t RotateDistance>
        using RotateR32 = Driver<SwizzleImpl::Ops::RotateR32<RotateDistance>>;

        /*! \class RotateR16
        *  \brief  Swizzle class that rotates all threads to the right in each group of 16
        *  @tparam RotateDistance thread index [0-15]
        */
        template <uint32_t RotateDistance>
        using RotateR16 = Driver<SwizzleImpl::Ops::RotateR16<RotateDistance>>;

        /*! \class RotateR8
        *  \brief  Swizzle class that rotates all threads to the right in each group of 8
        *  @tparam RotateDistance thread index [0-7]
        */
        template <uint32_t RotateDistance>
        using RotateR8 = Driver<SwizzleImpl::Ops::RotateR8<RotateDistance>>;

        /*! \class RotateR4
        *  \brief  Swizzle class that rotates all threads to the right in each group of 4
        *  @tparam RotateDistance thread index [0-3]
        */
        template <uint32_t RotateDistance>
        using RotateR4 = Driver<SwizzleImpl::Ops::RotateR4<RotateDistance>>;

        /*! \class RotateR2
        *  \brief  Swizzle class that rotates all threads to the right in each group of 2
        *  @tparam RotateDistance thread index [0-1]
        */
        template <uint32_t RotateDistance>
        using RotateR2 = Driver<SwizzleImpl::Ops::RotateR2<RotateDistance>>;

        // Shuffle variants

        /*! \class Shuffle4
        *  \brief  Swizzle class that shuffles elements in each group of 4
        *  @tparam Select0 element index [0-3]
        *  @tparam Select1 element index [0-3]
        *  @tparam Select2 element index [0-3]
        *  @tparam Select3 element index [0-3]
        */
        template <uint32_t Select0, uint32_t Select1, uint32_t Select2, uint32_t Select3>
        using Shuffle4 = Driver<SwizzleImpl::Ops::Shuffle4<Select0, Select1, Select2, Select3>>;

        /*! \class Shuffle2
        *  \brief  Swizzle class that shuffles elements in each group of 2
        *  @tparam Select0 element index [0-1]
        *  @tparam Select1 element index [0-1]
        */
        template <uint32_t Select0, uint32_t Select1>
        using Shuffle2 = Driver<SwizzleImpl::Ops::Shuffle2<Select0, Select1>>;

        // Swap variants

        /*! \class Swap16
        *  \brief  Swizzle class that swaps neighbouring groups in sizes of 16
        */
        using Swap16 = Driver<SwizzleImpl::Ops::Swap16>;

        /*! \class Swap8
        *  \brief  Swizzle class that swaps neighbouring groups in sizes of 8
        */
        using Swap8 = Driver<SwizzleImpl::Ops::Swap8>;

        /*! \class Swap4
        *  \brief  Swizzle class that swaps neighbouring groups in sizes of 4
        */
        using Swap4 = Driver<SwizzleImpl::Ops::Swap4>;

        /*! \class Swap2
        *  \brief  Swizzle class that swaps neighbouring groups in sizes of 2
        */
        using Swap2 = Driver<SwizzleImpl::Ops::Swap2>;

        // Fft variants

        /*! \class Fft
        *  \brief  Swizzle class that applies a specific fft transform given by control codes.
        *  For a listing of these codes, consult the ISA documentation for ds_swizzle.
        *  @tparam SubGroupSize affect size of sub-groups [0, 1, 2, 4, 8, 16, 32]
        *  @tparam FftCtrl control code for ds_swizzle fft function (see ISA)
        */
        template <uint32_t SubGroupSize, uint32_t FftCtrl>
        using Fft = Driver<SwizzleImpl::Ops::Fft<SubGroupSize, FftCtrl>>;
        /** @}*/

    } // namespace Swizzle

} // namespace rocwmma

#endif // ROCWMMA_SWIZZLE_HPP
