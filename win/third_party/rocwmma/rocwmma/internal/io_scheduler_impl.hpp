/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc.
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
#ifndef ROCWMMA_IO_SCHEDULER_IMPL_HPP
#define ROCWMMA_IO_SCHEDULER_IMPL_HPP

namespace rocwmma
{
    namespace IOScheduler
    {
        constexpr /* static */ inline auto NonCooperative::waveIndex()
        {
            return 0u;
        }

        constexpr /* static */ inline auto NonCooperative::waveCount()
        {
            return 1u;
        }

        template <uint32_t TBlockX, uint32_t TBlockY>
        constexpr /* static */ inline auto RowMajor2d<TBlockX, TBlockY>::waveIndex()
        {
            return DataSpace::fromMatrixCoord(WaveSpace::localWaveCoord(),
                                              get<1>(WaveSpace::workgroupDim()));
        }

        template <uint32_t TBlockX, uint32_t TBlockY>
        constexpr /* static */ inline auto RowMajor2d<TBlockX, TBlockY>::waveCount()
        {
            return reduce_mult(WaveSpace::workgroupDim());
        }

        template <uint32_t TBlockX, uint32_t TBlockY>
        constexpr /* static */ inline auto ColMajor2d<TBlockX, TBlockY>::waveIndex()
        {
            return DataSpace::fromMatrixCoord(WaveSpace::localWaveCoord(),
                                              get<0>(WaveSpace::workgroupDim()));
        }

        template <uint32_t TBlockX, uint32_t TBlockY>
        constexpr /* static */ inline auto ColMajor2d<TBlockX, TBlockY>::waveCount()
        {
            return reduce_mult(WaveSpace::workgroupDim());
        }

        template <uint32_t TBlockX, uint32_t TBlockY>
        constexpr /* static */ inline auto RowSlice2d<TBlockX, TBlockY>::waveIndex()
        {
            return get<1>(WaveSpace::localWaveCoord());
        }

        template <uint32_t TBlockX, uint32_t TBlockY>
        constexpr /* static */ inline auto RowSlice2d<TBlockX, TBlockY>::waveCount()
        {
            return get<1>(WaveSpace::workgroupDim());
        }

        template <uint32_t TBlockX, uint32_t TBlockY>
        constexpr /* static */ inline auto ColSlice2d<TBlockX, TBlockY>::waveIndex()
        {
            return get<0>(WaveSpace::localWaveCoord());
        }

        template <uint32_t TBlockX, uint32_t TBlockY>
        constexpr /* static */ inline auto ColSlice2d<TBlockX, TBlockY>::waveCount()
        {
            return get<0>(WaveSpace::workgroupDim());
        }

        template <uint32_t TBlockX, uint32_t TBlockY, uint32_t WaveIdx /*= 0u*/>
        constexpr /* static */ inline auto Single<TBlockX, TBlockY, WaveIdx>::waveIndex()
        {
            // Default as row-major id
            auto RIdx = DataSpace::fromMatrixCoord(WaveSpace::localWaveCoord(),
                                                   get<1>(WaveSpace::workgroupDim()));

            // Reset the WaveIdx to origin, any waves > 0 will be clipped
            return static_cast<uint32_t>(RIdx - WaveIdx);
        }

        template <uint32_t TBlockX, uint32_t TBlockY, uint32_t WaveIdx /*= 0u*/>
        constexpr /* static */ inline auto Single<TBlockX, TBlockY, WaveIdx>::waveCount()
        {
            return 1u;
        }

    } // namespace IOScheduler

    namespace SchedulerTraits_impl
    {
        using namespace IOScheduler;

        // Build meta-traits for scheduler objects

        // is_scheduler_valid: must be a known scheduler class
        template <typename Scheduler>
        struct is_scheduler_valid : false_type
        {
        };

        template <>
        struct is_scheduler_valid<NonCooperative> : true_type
        {
        };

        template <uint32_t TBlockX, uint32_t TBlockY>
        struct is_scheduler_valid<RowMajor2d<TBlockX, TBlockY>> : true_type
        {
        };

        template <uint32_t TBlockX, uint32_t TBlockY>
        struct is_scheduler_valid<ColMajor2d<TBlockX, TBlockY>> : true_type
        {
        };

        template <uint32_t TBlockX, uint32_t TBlockY>
        struct is_scheduler_valid<RowSlice2d<TBlockX, TBlockY>> : true_type
        {
        };

        template <uint32_t TBlockX, uint32_t TBlockY>
        struct is_scheduler_valid<ColSlice2d<TBlockX, TBlockY>> : true_type
        {
        };

        template <uint32_t TBlockX, uint32_t TBlockY, uint32_t WaveIdx>
        struct is_scheduler_valid<Single<TBlockX, TBlockY, WaveIdx>> : true_type
        {
        };

        template <typename Scheduler>
        static constexpr bool is_scheduler_valid_v = is_scheduler_valid<Scheduler>::value;

        // is_scheduler_cooperative: multiple waves in the threadblock will participate
        template <typename Scheduler>
        struct is_scheduler_cooperative : false_type
        {
        };

        template <uint32_t TBlockX, uint32_t TBlockY>
        struct is_scheduler_cooperative<RowMajor2d<TBlockX, TBlockY>> : true_type
        {
        };

        template <uint32_t TBlockX, uint32_t TBlockY>
        struct is_scheduler_cooperative<ColMajor2d<TBlockX, TBlockY>> : true_type
        {
        };

        template <uint32_t TBlockX, uint32_t TBlockY>
        struct is_scheduler_cooperative<RowSlice2d<TBlockX, TBlockY>> : true_type
        {
        };

        template <uint32_t TBlockX, uint32_t TBlockY>
        struct is_scheduler_cooperative<ColSlice2d<TBlockX, TBlockY>> : true_type
        {
        };

        template <uint32_t TBlockX, uint32_t TBlockY, uint32_t WaveIdx>
        struct is_scheduler_cooperative<Single<TBlockX, TBlockY, WaveIdx>> : true_type
        {
        };

        template <typename Scheduler>
        static constexpr bool is_scheduler_cooperative_v
            = is_scheduler_cooperative<Scheduler>::value;

    } // namespace SchedulerTraits_impl

} // namespace rocwmma

#endif // ROCWMMA_IO_SCHEDULER_IMPL_HPP
