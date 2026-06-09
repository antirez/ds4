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
#ifndef ROCWMMA_IO_BEARER_BASE_HPP
#define ROCWMMA_IO_BEARER_BASE_HPP

#include "layout/layout.hpp"
#include "layout/layout_traits.hpp"
#include "utility/forward.hpp"

namespace rocwmma
{
    //! @struct IOBearer
    //! @brief IOBearer is the vehicle that executes BearerPolicy transactions iteratively through the coordinate space
    //! offsets given by the MatrixLayout, while checking and applying bounds controls.
    //! @tparam DataLayout The class that handles the configuration of the 1d data layout.
    //! @tparam MatrixLayout The class that handles the configuration of the 2d iterative space in which the BearerPolicy is applied.
    //! @tparam BearerPolicy Typically represents a memory transaction such as a store or load, described by data type and vector size.
    //! @tparam BoundsCtrl Checks bounding box boundary violations by the BearerPolicy and may apply adjustments to the violating buffer.
    template <class DataLayout,
              class MatrixLayout,
              template <typename, uint32_t>
              class BearerPolicy,
              class BoundsCtrl>
    struct IOBearerBase
    {
    protected:
        // Access traits from Matrix and DataLayouts
        using MatrixLayoutTraits = layout_traits<MatrixLayout>;
        using DataLayoutTraits   = layout_traits<DataLayout>;
        using DataT              = typename MatrixLayoutTraits::DataT;

        // Iterative BearerPolicy traits
        static constexpr uint32_t TransactionSize = MatrixLayoutTraits::VectorWidth;
        static constexpr uint32_t IterationCount  = reduce_mult(MatrixLayout::strideCounts());
        static constexpr uint32_t BuffSize        = TransactionSize * IterationCount;

    public:
        // Full sized buffer for the whole operation
        using BufferT = typename BearerPolicy<DataT, BuffSize>::BufferT;

    protected:
        //! @brief Handle partial transactions from front-to-back.
        //! @tparam PartialCount The number of partial elements in the transaction
        //! @tparam PBufferT The buffer partial transactions will apply to
        //! @tparam DataPtrT The base memory data pointer
        template <uint32_t PartialCount, typename PBufferT, typename DataPtrT>
        ROCWMMA_DEVICE static inline auto partial_impl(PBufferT& buff, DataPtrT&& dataPtr);

        //! @brief Handle bounds violations from back-to-front.
        //! @tparam CoundCount The number of boundary violations in the transaction
        //! @tparam BBufferT The buffer boundary control will apply to
        //! @tparam ScalarT The datatype of a scalar value to apply in the bounds control
        template <uint32_t BoundCount, typename BBufferT, typename ScalarT>
        ROCWMMA_DEVICE static inline auto bounds_impl(BBufferT& buff, ScalarT&& clipVal);

        //! @brief Loop-unroll to cover all transactions described by MatrixLayout strides
        //! @tparam Depth The loop recursion depth (Default 0)
        //! @tparam BufferT The buffer segment for the current recursion depth
        //! @tparam Coord2d The type of the given 2d coordinate object
        //! @tparam ExternDataT The type of the pointer given by the user
        //! @note This class is used for both load / store transactions, so the ExternDataT
        //! is intended to be opaque on the const-ness.
        template <size_t Depth = 0, typename BufferT, typename Coord2d, typename ExternDataT>
        ROCWMMA_DEVICE static inline auto
            unroll_impl(BufferT&& buff, Coord2d&& baseOffset2d, ExternDataT* dataPtr, uint32_t ldm);

    public:
        //! @brief Interface driver for loop-unroll to cover all transactions described by MatrixLayout strides
        //! @tparam BufferT The buffer full buffer segment for all transactions
        //! @tparam ExternDataT The type of the pointer given by the user
        //! @note This class is used for both load / store transactions, so the ExternDataT
        //! is intended to be opaque on the const-ness.
        template <typename BufferT, typename ExternDataT>
        ROCWMMA_DEVICE static inline void exec(BufferT&& buff, ExternDataT* dataPtr, uint32_t ldm);
    };

} // namespace rocwmma

#include "io_bearer_base_impl.hpp"

#endif // ROCWMMA_IO_BEARER_BASE_HPP
