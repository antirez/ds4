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
#ifndef ROCWMMA_IO_SCHEDULER_HPP
#define ROCWMMA_IO_SCHEDULER_HPP

#include "mapping_util.hpp"

namespace rocwmma
{
    namespace IOScheduler
    {
        //! @struct NonCooperative
        //! @brief No thread-block cooperation; each wave will execute independently
        struct NonCooperative
        {
            constexpr static inline auto waveIndex();
            constexpr static inline auto waveCount();
            constexpr static inline bool RangeCheck = false;
        };

        //! @struct RowMajor2d
        //! @brief All waves are scheduled in row_major order.
        //! E.g. (TBlockX, TBlockY) => 2x2 waves
        //! w0 = (0, 0),  w1 = (0, 1),
        //! w2 = (1, 0),  w3 = (1, 1)
        //! @tparam TBlockX the X dimension of the cooperative thread-block
        //! @tparam TBlockY the Y dimension of the cooperative thread-block
        template <uint32_t TBlockX, uint32_t TBlockY>
        struct RowMajor2d
        {
        private:
            using WaveSpace = detail::WaveSpace<TBlockX, TBlockY>;
            using DataSpace = detail::DataSpace<row_major>;

        public:
            constexpr static inline auto waveIndex();
            constexpr static inline auto waveCount();
            constexpr static inline bool RangeCheck = false;
        };

        //! @struct ColMajor2d
        //! @brief All waves are scheduled in col_major order.
        //! E.g. (TBlockX, TBlockY) = 2x2 waves
        //! w0 = (0, 0), w2 = (0, 1),
        //! w1 = (1, 0), w3 = (1, 1)
        //! @tparam TBlockX the X dimension of the cooperative thread-block
        //! @tparam TBlockY the Y dimension of the cooperative thread-block
        template <uint32_t TBlockX, uint32_t TBlockY>
        struct ColMajor2d
        {
        private:
            using WaveSpace = detail::WaveSpace<TBlockX, TBlockY>;
            using DataSpace = detail::DataSpace<col_major>;

        public:
            constexpr static inline auto waveIndex();
            constexpr static inline auto waveCount();
            constexpr static inline bool RangeCheck = false;
        };

        //! @struct RowSlice2d
        //! @brief Waves are partitioned into rows. Only waves in the same row
        //! participate together.
        //! E.g. (TBlockX, TBlockY) = 2x2 waves
        //! RowSlice0: w0 = (0, 0), w1 = (0, 1)
        //! RowSlice1: w0 = (1, 0), w1 = (1, 1)
        //! @tparam TBlockX the X dimension of the cooperative thread-block
        //! @tparam TBlockY the Y dimension of the cooperative thread-block
        template <uint32_t TBlockX, uint32_t TBlockY>
        struct RowSlice2d
        {
        private:
            using WaveSpace = detail::WaveSpace<TBlockX, TBlockY>;

        public:
            constexpr static inline auto waveIndex();
            constexpr static inline auto waveCount();
            constexpr static inline bool RangeCheck = false;
        };

        //! @struct ColSlice2d
        //! @brief Waves are partitioned into cols. Only waves in the same col
        //! participate together.
        //! E.g. (TBlockX, TBlockY) = 2x2 waves
        //! ColSlice0:     ColSlice1:
        //! w0 = (0, 0),   w0 = (0, 1),
        //! w1 = (1, 0)    w1 = (1, 1)
        //! @tparam TBlockX the X dimension of the cooperative thread-block
        //! @tparam TBlockY the Y dimension of the cooperative thread-block
        template <uint32_t TBlockX, uint32_t TBlockY>
        struct ColSlice2d
        {
        private:
            using WaveSpace = detail::WaveSpace<TBlockX, TBlockY>;

        public:
            constexpr static inline auto waveIndex();
            constexpr static inline auto waveCount();
            constexpr static inline bool RangeCheck = false;
        };

        //! @struct Single
        //! @brief Only a single wave out of the thread block will act
        //! @tparam TBlockX the X dimension of the cooperative thread-block
        //! @tparam TBlockY the Y dimension of the cooperative thread-block
        template <uint32_t TBlockX, uint32_t TBlockY, uint32_t WaveIdx = 0u>
        struct Single
        {
        private:
            using WaveSpace = detail::WaveSpace<TBlockX, TBlockY>;
            using DataSpace = detail::DataSpace<col_major>;

        public:
            constexpr static inline auto waveIndex();
            constexpr static inline auto waveCount();
            constexpr static inline bool RangeCheck = true;
        };

        using Default = NonCooperative;

    } // namespace IOScheduler

} // namespace rocwmma

#include "io_scheduler_impl.hpp"

namespace rocwmma
{
    //! @struct scheduler_traits
    //! @brief A collection of scheduler traits
    //! @tparam Scheduler instance of scheduler class
    template <typename Scheduler>
    struct scheduler_traits
    {
        constexpr static bool is_valid = SchedulerTraits_impl::is_scheduler_valid_v<Scheduler>;
        constexpr static bool is_cooperative
            = SchedulerTraits_impl::is_scheduler_cooperative_v<Scheduler>;
        static constexpr uint32_t WaveCount  = Scheduler::waveCount();
        static constexpr bool     RangeCheck = Scheduler::RangeCheck;
    };

} // namespace rocwmma

#endif // ROCWMMA_IO_SCHEDULER_HPP
