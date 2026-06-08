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
#ifndef ROCWMMA_IO_BEARER_HPP
#define ROCWMMA_IO_BEARER_HPP

#include "io_bearer_base.hpp"
#include "io_scheduler.hpp"

namespace rocwmma
{
    //! @struct CoopIOBearer
    //! @brief CoopIOBearer is the vehicle that executes BearerPolicy transactions iteratively through the coordinate space
    //! offsets given by the MatrixLayout, while checking and applying bounds controls. Additional logic is present to manage
    //! splitting load between waves.
    //! @tparam DataLayout The class that handles the configuration of the 1d data layout.
    //! @tparam CoopMatrixLayout The class that handles the configuration of the 2d iterative space in which the BearerPolicy is applied.
    //! It is a basic MatrixLayout wrapped in the MatrixCoopLayout class to get access to WaveCount and splitting properties.
    //! @tparam BearerPolicy Typically represents a memory transaction such as a store or load, described by data type and vector size.
    //! @tparam BoundsCtrl Checks bounding box boundary violations by the BearerPolicy and may apply adjustments to the violating buffer.
    template <class DataLayout,
              class MatrixLayout,
              template <typename, uint32_t>
              class BearerPolicy,
              class BoundsCtrl,
              class Scheduler>
    struct IOBearer : protected IOBearerBase<DataLayout, MatrixLayout, BearerPolicy, BoundsCtrl>
    {
    private:
        using SchedulerTraits = scheduler_traits<Scheduler>;
        using BaseImpl        = IOBearerBase<DataLayout, MatrixLayout, BearerPolicy, BoundsCtrl>;

        ROCWMMA_DEVICE constexpr static inline auto waveEnabler();

    public:
        using BufferT = typename BaseImpl::BufferT;

        //! @brief Interface driver for loop-unroll to cover all transactions described by MatrixLayout strides
        //! @tparam BufferT The buffer full buffer segment for all transactions
        //! @tparam ExternDataT The type of the pointer given by the user
        //! @note This class is used for both load / store transactions, so the ExternDataT
        //! is intended to be opaque on the const-ness.
        template <typename BufferT, typename ExternDataT>
        ROCWMMA_DEVICE static inline void
            exec(BufferT&& buffer, ExternDataT* dataPtr, uint32_t ldm);
    };

} // namespace rocwmma

#include "io_bearer_impl.hpp"

#endif // ROCWMMA_IO_BEARER_HPP
