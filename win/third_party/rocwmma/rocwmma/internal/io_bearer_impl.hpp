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
#ifndef ROCWMMA_IO_BEARER_IMPL_HPP
#define ROCWMMA_IO_BEARER_IMPL_HPP

#include "io_bearer.hpp"
#include "io_scheduler.hpp"
#include "utility/math.hpp"
#include "vector.hpp"

namespace rocwmma
{

#define IOBearerTypesDecl                 \
    class DataLayout, class MatrixLayout, \
        template <typename, uint32_t>     \
        class BearerPolicy, class BoundsCtrl, class Scheduler

#define IOBearerTypesImpl DataLayout, MatrixLayout, BearerPolicy, BoundsCtrl, Scheduler

    // Determines whether a wave should participate or not in the layout
    template <IOBearerTypesDecl>
    ROCWMMA_DEVICE constexpr /* static */ inline auto IOBearer<IOBearerTypesImpl>::waveEnabler()
    {
        // Note: MaxWaves is the actual maximum amount of waves that can participate.
        constexpr auto maxWaves = MatrixLayout::calcMaxSplits(SchedulerTraits::WaveCount);

        static_assert(maxWaves == SchedulerTraits::WaveCount,
                      "Cannot accommodate Scheduler WaveCount. Try a bigger block size or "
                      "scheduler with fewer waves.");

        // RangeCheck indicates the wave indices coming from the Scheduler may be invalid!
        if constexpr(SchedulerTraits::RangeCheck)
        {
            // Must branch
            if(static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(Scheduler::waveIndex()))
               < maxWaves)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        else
        {
            return true;
        }
    }

    template <IOBearerTypesDecl>
    template <typename BufferT, typename ExternDataT>
    ROCWMMA_DEVICE /* static */ inline void
        IOBearer<IOBearerTypesImpl>::exec(BufferT&& buffer, ExternDataT* dataPtr, uint32_t ldm)
    {
        // Traits
        using SchedulerTraits = scheduler_traits<Scheduler>;

        // Sanity check
        static_assert(SchedulerTraits::is_valid, "Invalid scheduler");

        // Cooperative
        if constexpr(SchedulerTraits::is_cooperative)
        {
            // Wave filter
            if(!waveEnabler())
            {
                return;
            }

            // Current offset from wave tile origin (0, 0)
            auto currentOffset2d = MatrixLayout::baseOffset(Scheduler::waveIndex());

            BaseImpl::unroll_impl(forward<BufferT>(buffer), currentOffset2d, dataPtr, ldm);
        }
        // Non-cooperative
        else
        {
            BaseImpl::exec(forward<BufferT>(buffer), dataPtr, ldm);
        }
    }

#undef IOBearerTypesDecl
#undef IOBearerTypesImpl

} // namespace rocwmma

#endif // ROCWMMA_IO_BEARER_IMPL_HPP
