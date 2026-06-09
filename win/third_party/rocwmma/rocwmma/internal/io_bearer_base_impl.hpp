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
#ifndef ROCWMMA_IO_BEARER_BASE_IMPL_HPP
#define ROCWMMA_IO_BEARER_BASE_IMPL_HPP

#include "layout/layout.hpp"
#include "layout/layout_traits.hpp"
#include "tuple.hpp"
#include "utility/static_for.hpp"
#include "utility/vector.hpp"

namespace rocwmma
{

#define IOBearerBaseTypesDecl             \
    class DataLayout, class MatrixLayout, \
        template <typename, uint32_t>     \
        class BearerPolicy, class BoundsCtrl

#define IOBearerBaseTypesImpl DataLayout, MatrixLayout, BearerPolicy, BoundsCtrl

    // Operate on the given vector with expected partial values.
    // Assumption: the original input vector size is a power of 2.
    // Procedure: Starting from the beginning of the vector, recursively
    // split the input vector into smaller powers of 2 until the
    // transfers of PartialCount can be satisfied.
    // Example:
    // v = VecT<DataT, 8> PartialCount = 7
    // Step1:
    // - Input: v[0-7], PartialCount = 7
    // - TransactionSize = (8 / 2) = 4
    // - Chunks          = (7 / 4) = 1
    // - Xfer v[0-3], Remain v[4-7]
    // - NewPartialCount = 7 - (1 * 4) = 3
    // Step2:
    // - Input: v[4-7], PartialCount = 3
    // - TransactionSize = (4 / 2) = 2
    // - Chunks          = (3 / 2) = 1
    // - Xfer v[4-5], Remain v[6-7]
    // - NewPartialCount = 3 - (1 * 2) = 1
    // Step3:
    // - Input: v[6-7], PartialCount = 1
    // - TransactionSize = (2 / 2) = 1
    // - Chunks          = (1 / 1) = 1
    // - Xfer v[6]
    // - New PartialCount = 1 - (1 * 1) = 0
    // </Done> v[7] is untouched
    template <IOBearerBaseTypesDecl>
    template <uint32_t PartialCount, typename PBufferT, typename DataPtrT>
    ROCWMMA_DEVICE inline auto IOBearerBase<IOBearerBaseTypesImpl>::partial_impl(PBufferT&  buff,
                                                                                 DataPtrT&& dataPtr)
    {
        using VecTraits = VecTraits<decay_t<decltype(buff)>>;
        static_assert(is_pow2(VecTraits::size()), "Buffer size must be a power of 2");

        // Invalid case: pass-through
        if constexpr(PartialCount == 0u || PartialCount > VecTraits::size())
        {
            // Normally this would be a static_assert(PartialCount <= VecTraits::size()),
            // however some runtime dispatching may break the rule
            return;
        }
        // Simple case: PartialCount is exactly the size of the vector
        else if constexpr(PartialCount == VecTraits::size())
        {
            using Bearer = BearerPolicy<DataT, VecTraits::size()>;
            Bearer::exec(buff, dataPtr);
        }
        // General case: split the vector in half and initiate transfer of smaller chunks.
        else
        {
            constexpr uint32_t SplitVecSize = VecTraits::size() / 2u;

            if constexpr(SplitVecSize > 0u)
            {
                constexpr auto PartialIterations = PartialCount / SplitVecSize;

                if constexpr(PartialIterations > 0u)
                {
                    // Iterate over the input vector with split size
                    vector_mutate_for_each<SplitVecSize>(
                        forward<PBufferT>(buff),
                        [](auto&& buff, auto&& idx, auto* dataPtr) {
                            constexpr auto Idx = decay_t<decltype(idx)>::value;
                            if constexpr(Idx < PartialIterations)
                            {
                                using Bearer = BearerPolicy<DataT, SplitVecSize>;
                                Bearer::exec(buff, dataPtr + Idx * SplitVecSize);
                            }
                        },
                        dataPtr);
                }

                // Check if there are more partials to process (e.g., the SplitVecSize is still too granular)
                constexpr int32_t RemainingPartialCount
                    = PartialCount - PartialIterations * SplitVecSize;
                static_assert(RemainingPartialCount >= 0, "Invalid new PartialCount");

                if constexpr(RemainingPartialCount > 0)
                {
                    // Perform the split again on data that has not been passed over
                    partial_impl<RemainingPartialCount>(
                        *makeVectorIterator<SplitVecSize>(buff).it(PartialIterations),
                        dataPtr + PartialIterations * SplitVecSize);
                }
            }
        }
    }

    // Operate on the given vector with expected partial values.
    // Assumption: the original input vector size is a power of 2.
    // Procedure: Starting from the end of the vector, recursively
    // split the input vector into smaller powers of 2 until the
    // BoundCount boundary conditions can be applied.
    // Example:
    // buff = VecT<DataT, 8> BoundCount = 7
    // Step1:
    // - Input: v[0-7], BoundCount = 7
    // - TransactionSize = (8 / 2) = 4
    // - Chunks          = (7 / 4) = 1
    // - ApplyBounds v[4-7], Remain v[0-3]
    // - NewBoundCount = 7 - (1 * 4) = 3
    // Step2:
    // - Input: v[0-3], BoundCount = 3
    // - TransactionSize = (4 / 2) = 2
    // - Chunks          = (3 / 2) = 1
    // - ApplyBounds v[2-3], Remain v[0-1]
    // - NewBoundCount = 3 - (1 * 2) = 1
    // Step3:
    // - Input: v[0-1], BoundCount = 1
    // - TransactionSize = (2 / 2) = 1
    // - Chunks          = (1 / 1) = 1
    // - ApplyBounds v[1]
    // - New BoundCount = 1 - (1 * 1) = 0
    // </Done> v[0] is untouched
    template <IOBearerBaseTypesDecl>
    template <uint32_t BoundCount, typename BBufferT, typename ScalarT>
    ROCWMMA_DEVICE inline auto IOBearerBase<IOBearerBaseTypesImpl>::bounds_impl(BBufferT& buff,
                                                                                ScalarT&& clipVal)
    {
        using VecTraits = VecTraits<decay_t<decltype(buff)>>;
        static_assert(is_pow2(VecTraits::size()), "Buffer size must be a power of 2");

        // Invalid case: pass-through
        if constexpr(BoundCount == 0u || BoundCount > VecTraits::size())
        {
            // Normally this would be a static_assert(BoundCount <= VecTraits::size()),
            // however some runtime dispatching may break the rule
            return;
        }
        // Simple case: BoundCount is exactly the size of the vector
        else if constexpr(BoundCount == VecTraits::size())
        {
            BoundsCtrl::exec(buff, clipVal);
        }
        // General case: split the vector in half and initiate bounds control of smaller chunks.
        else
        {
            // Split vecsize and determine if the split size can be used to fill the next chunk.
            constexpr uint32_t SplitVecSize = VecTraits::size() / 2u;

            if constexpr(SplitVecSize > 0u)
            {
                constexpr auto BoundIterations = BoundCount / SplitVecSize;

                if constexpr(BoundIterations > 0u)
                {
                    // Iterate in reverse over the input vector with split size
                    vector_mutate_for_each<SplitVecSize>(
                        forward<BBufferT>(buff),
                        [](auto&& buff, auto&& idx, auto&& clipVal) {
                            constexpr auto Idx = decay_t<decltype(idx)>::value;
                            if constexpr((VecTraits::size() / SplitVecSize - 1u - Idx)
                                         < BoundIterations)
                            {
                                BoundsCtrl::exec(buff, clipVal);
                            }
                        },
                        clipVal);
                }

                // Check if there are more partials to process (e.g., the SplitVecSize is still too granular)
                constexpr int32_t RemainingBoundCount = BoundCount - BoundIterations * SplitVecSize;
                static_assert(RemainingBoundCount >= 0, "Invalid new BoundCount");

                if constexpr(RemainingBoundCount > 0)
                {
                    // Perform the split again on data that has not been passed over
                    bounds_impl<RemainingBoundCount>(
                        *makeVectorIterator<SplitVecSize>(buff).it(VecTraits::size() / SplitVecSize
                                                                   - 1u - BoundIterations),
                        clipVal);
                }
            }
        }
    }

    // Outer loops = index 0, ..., N-2
    // Inner loop  = index N-1
    // Notes:
    // - Assumption: MatrixLayout provides constexpr strideCounts and strides.
    //   We can then use static unroll to eliminate looping.
    // - Unroll model will use vector_mutate_for_each because the current bearers are implemented
    //   as visitor patterns. They either write to, or read from an input vector object.
    //   The vector needs to be mutable for loads, but the stores will cast as const reference.
    //   vector_mutate_for_each provides a reference to raw vector data it is operating on, which
    //   will satisfy both loading and storing visitation needs.
    template <IOBearerBaseTypesDecl>
    template <size_t Depth /*= 0*/, typename BufferT, typename Coord2d, typename ExternDataT>
    ROCWMMA_DEVICE inline auto IOBearerBase<IOBearerBaseTypesImpl>::unroll_impl(
        BufferT&& buff, Coord2d&& baseOffset2d, ExternDataT* dataPtr, uint32_t ldm)
    {
        // Get the layout strides for the current depth
        constexpr auto StrideSpace = pop_front<Depth>(MatrixLayout::strideCounts());
        constexpr auto Strides     = pop_front<Depth>(MatrixLayout::strides());
        constexpr auto ItCount     = reduce_mult(StrideSpace);

        // Ensure the buffer size is appropriate
        using BufferTraits = VecTraits<decay_t<BufferT>>;
        static_assert(BufferTraits::size() == ItCount * TransactionSize, "Invalid buffer size");

        // Replacement value for boundary violations
        constexpr auto BoundsValue = 0u;

        // Offset stride for current depth
        constexpr auto CurrentStride2d = get_first(Strides);

        // Recurse to the next nested layer
        if constexpr((VecTraits<decay_t<decltype(StrideSpace)>>::size()) > 1u)
        {
            constexpr auto NextStrideSpace  = pop_front(StrideSpace);
            constexpr auto NextBufferStride = reduce_mult(NextStrideSpace) * TransactionSize;

            vector_mutate_for_each<NextBufferStride>(
                forward<BufferT>(buff),
                [](auto&& buff,
                   auto&& idx,
                   auto&& baseOffset2d,
                   auto&& offsetStep2d,
                   auto&& boundsValue,
                   auto*  dataPtr,
                   auto   ldm) {
                    // Current offset from wave tile origin (0, 0)
                    auto currentOffset2d
                        = baseOffset2d + decay_t<decltype(idx)>::value * offsetStep2d;

                    // Check bounds conditions for minimum 1x1 valid area. If the minimum area isn't valid,
                    // there are no partials and entire transaction should be treated as boundary violation.
                    constexpr auto dataSize2d = make_coord2d(1u, 1u);
                    if(!BoundsCtrl::checkBounds(currentOffset2d, dataSize2d))
                    {
                        // Unroll to next layer
                        unroll_impl<Depth + 1u>(buff, currentOffset2d, dataPtr, ldm);
                    }
                    else
                    {
                        // Boundary violation
                        BoundsCtrl::exec(buff, boundsValue);
                    }
                },
                baseOffset2d,
                CurrentStride2d,
                BoundsValue,
                dataPtr,
                ldm);
        }
        // Final depth layer will invoke the memory transactions
        else
        {
            vector_mutate_for_each<TransactionSize>(
                forward<BufferT>(buff),
                [](auto&&   buff,
                   auto&&   idx,
                   auto&&   baseOffset2d,
                   auto&&   offsetStep2d,
                   auto&&   boundsValue,
                   auto*    dataPtr,
                   uint32_t ldm) {
                    // Current offset from wave tile origin (0, 0)
                    auto currentOffset2d
                        = baseOffset2d + decay_t<decltype(idx)>::value * offsetStep2d;

                    // Each load of the bearer is linear.
                    // Expand into 2d offset area to check rectangular bounds
                    constexpr auto dataSize2d = DataLayoutTraits::is_col_major
                                                    ? make_coord2d(TransactionSize, 1u)
                                                    : make_coord2d(1u, TransactionSize);

                    // Cache 1d data address offset
                    uint32_t dataOffset = DataLayout::fromMatrixCoord(currentOffset2d, ldm);

                    // The outer loop layers guarantee that the baseOffset2d is at least within the boundary area,
                    // and has at least one partial (1x1).
                    // Assumptions:
                    // - Partial count < VW
                    // - Partial count + bound count == VW.
                    // - Assume a minimum of 1 valid partial

                    // Note: Outer layer has already tested 1x1 from baseOffset2d, which gives us
                    // 1 valid partial.
                    if constexpr(decay_t<decltype(idx)>::value == 0u && TransactionSize <= 2u)
                    {
                        // Special case of VW == 1u on iteration 0
                        if constexpr(TransactionSize == 1u)
                        {
                            // There are no possible boundary violations
                            // because we already have 1 valid partial!
                            using Bearer = BearerPolicy<DataT, TransactionSize>;
                            Bearer::exec(buff, dataPtr + dataOffset);
                        }
                        // Special case of VW == 2u on iteration 0
                        else if constexpr(TransactionSize == 2u)
                        {
                            // Check bounds conditions.
                            // Low mask means no boundary violation
                            if(!BoundsCtrl::checkBounds(currentOffset2d, dataSize2d))
                            {
                                using Bearer = BearerPolicy<DataT, TransactionSize>;
                                Bearer::exec(buff, dataPtr + dataOffset);
                            }
                            // On high mask treat boundary violations
                            else
                            {
                                // Because a boundary violation has been detected, we now have
                                // at least 1 boundary violation, in addition to at least 1 partial.

                                // Handle partial transaction
                                partial_impl<1u>(buff, dataPtr + dataOffset);

                                // Handle boundary violations
                                bounds_impl<1u>(buff, boundsValue);
                            }
                        }
                    }
                    // General case where we need to calculate partial and bound counts.
                    else
                    {
                        // Check bounds conditions.
                        // Low mask means no boundary violation
                        if(!BoundsCtrl::checkBounds(currentOffset2d, dataSize2d))
                        {
                            using Bearer = BearerPolicy<DataT, TransactionSize>;
                            Bearer::exec(buff, dataPtr + dataOffset);
                        }
                        // On high mask, we must treat boundary violations
                        else
                        {
                            // Bounds violations are possible by either:
                            // 1. Offset step (currentOffset2d falls on boundary)
                            // 2. Bearer transaction (currentOffset2d + size falls on boundary or beyond)
                            // In col major: if diffY is positive, the violation is due to bearer, otherwise the coordinate is fully OOB.
                            // In row major: if diffX is positive, the violation is due to bearer, otherwise the coordinate is fully OOB.
                            // When coordinate is fully OOB, we have 0 partials and all bounds violations.
                            int32_t diffX        = BoundsCtrl::BoundX - get<0>(currentOffset2d);
                            int32_t diffY        = BoundsCtrl::BoundY - get<1>(currentOffset2d);
                            auto    partialCount = DataLayoutTraits::is_col_major
                                                       ? (diffY > 0u ? diffX : 0u)
                                                       : (diffX > 0u ? diffY : 0u);

                            bool processed = false;

                            // Caveat: partialCount is not a constexpr value due to dependency on per-thread baseOffset2d.
                            // Dispatch the partialCount value at runtime to match it with templated PartialCount and BoundCount.
                            static_for<0u, TransactionSize, 1u>(
                                [](auto&& idx,
                                   auto&& buff,
                                   auto&& partialCount,
                                   auto&  processedFlag,
                                   auto&& boundsValue,
                                   auto&& dataPtr,
                                   auto&& dataOffset) {
                                    // Calculate potential partial and bound count match
                                    constexpr auto PartialCount = decay_t<decltype(idx)>::value;
                                    constexpr auto BoundCount   = TransactionSize - PartialCount;

                                    // Test match
                                    if(partialCount == PartialCount && !processedFlag)
                                    {
                                        // Handle partial transactions
                                        partial_impl<PartialCount>(buff, dataPtr + dataOffset);

                                        // Handle boundary violations
                                        // Note: Give a value of 0 if ClipAndReplace is used
                                        bounds_impl<BoundCount>(buff, boundsValue);

                                        // Match is found
                                        processedFlag = true;
                                    }
                                },
                                buff,
                                partialCount,
                                processed,
                                boundsValue,
                                dataPtr,
                                dataOffset);
                        }
                    }
                },
                baseOffset2d,
                CurrentStride2d,
                BoundsValue,
                dataPtr,
                ldm);
        }
    }

    template <IOBearerBaseTypesDecl>
    template <typename BufferT, typename ExternDataT>
    ROCWMMA_DEVICE inline void IOBearerBase<IOBearerBaseTypesImpl>::exec(BufferT&&    buff,
                                                                         ExternDataT* dataPtr,
                                                                         uint32_t     ldm)
    {
        // Current offset from wave tile origin (0, 0)
        auto currentOffset2d = MatrixLayout::baseOffset();

        // Start unrolling from the base offset of matrix layout
        unroll_impl(forward<BufferT>(buff), currentOffset2d, dataPtr, ldm);
    }

#undef IOBearerBaseTypesDecl
#undef IOBearerBaseTypesImpl

} // namespace rocwmma

#endif // ROCWMMA_IO_BEARER_BASE_IMPL_HPP
