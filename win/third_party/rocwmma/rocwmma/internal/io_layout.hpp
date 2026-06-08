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
#ifndef ROCWMMA_IO_LAYOUT_HPP
#define ROCWMMA_IO_LAYOUT_HPP

#include "api_fwd.hpp"
#include "constants.hpp"
#include "io_scheduler.hpp"
#include "layout/layout.hpp"
#include "layout/matrix_coop_layout.hpp"
#include "types.hpp"
#include "utility/algorithm.hpp"

namespace rocwmma
{
    namespace detail
    {
        template <typename MatrixT,
                  uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayoutT,
                  uint32_t WaveCount = 1u,
                  uint32_t TestWidth
                  = 4u * Constants::AMDGCN_DWORD_SIZE_BYTES / (uint32_t)sizeof(DataT)>
        struct MaxVWSelector
        {
        private:
            // For small block sizes (16, 32):
            // Best to keep MaxVW high and reduce splits amongst waves.
            static constexpr uint32_t WaveCountFactor = (BlockDim <= 32) ? 1u : WaveCount;

            // Total number of elements in a single I/O operation
            static constexpr uint32_t ElementsPerIO
                = Constants::AMDGCN_WAVE_SIZE * TestWidth * WaveCountFactor;

            // Total number of elements for the entire block
            static constexpr uint32_t ElementCount = BlockDim * BlockK;

            // Ensure that for MaxVW:
            // - A minimum of one IO from each wave can fit
            // - A balanced multiple of IOs from each wave
            static constexpr bool ElementCountTest
                = (ElementsPerIO <= ElementCount) && (ElementCount % ElementsPerIO == 0);

            // Layout fitness check:
            // Basic non-interleaved layouts are classified into *OrthoVW (SOA) and *InlineVW (AOS) formats.
            // For any BlockDim/BlockK geometry, we ensure that these layouts come up with the same MaxVW,
            // so that the AOS <-> SOA transforms are possible and valid. The followings tests assure this.
            static constexpr bool BlockKTest = (Constants::AMDGCN_WAVE_SIZE * TestWidth
                                                / min(BlockDim, Constants::AMDGCN_WAVE_SIZE))
                                               <= BlockK;
            static constexpr bool OrthoTest         = TestWidth <= BlockK;
            static constexpr bool InlineTest        = TestWidth <= BlockDim;
            static constexpr bool LayoutFitnessTest = (BlockKTest && OrthoTest && InlineTest);

            // Decide on final MaxVW
            static constexpr uint32_t MaxVectorWidth = (ElementCountTest && LayoutFitnessTest)
                                                           ? TestWidth
                                                           : MaxVWSelector<MatrixT,
                                                                           BlockDim,
                                                                           BlockK,
                                                                           DataT,
                                                                           DataLayoutT,
                                                                           WaveCount,
                                                                           TestWidth / 2>::Result;

        public:
            static constexpr uint32_t Result = MaxVectorWidth;
        };

        // Accumulator case, is architecture specific
        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayoutT,
                  uint32_t WaveCount,
                  uint32_t TestWidth>
        struct MaxVWSelector<accumulator,
                             BlockDim,
                             BlockK,
                             DataT,
                             DataLayoutT,
                             WaveCount,
                             TestWidth>
        {
            static_assert(WaveCount == 1u, "Accumulators are not cooperative");

            constexpr static uint32_t Result
                = (bool)ROCWMMA_ARCH_GFX12
                      ? 8u
                      : ((is_same_v<DataT, float64_t> || (bool)ROCWMMA_ARCH_GFX11) ? 1u : 4u);
        };

        // Fallback case for bad test. Stay safe to VW=1
        template <typename MatrixT,
                  uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayoutT,
                  uint32_t WaveCount>
        struct MaxVWSelector<MatrixT, BlockDim, BlockK, DataT, DataLayoutT, WaveCount, 0u>
        {
            static constexpr uint32_t Result = 1u;
        };

    } // namespace detail

    /*! \struct IOLayout
    *  \brief Definition of VW, MaxVW, data and matrix mapping utilities
    *         in specific matrix context.
    *
    * @tparam MatrixT fragment context
    * @tparam BlockDim Block leading dimension
    * @tparam BlockK Block K-dimension
    * @tparam DataT data type
    * @tparam DataLayoutT in-memory layout as col_major or row_major
    * @tparam Scheduler wave-wise scheduler
    */
    template <typename MatrixT,
              uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              typename Scheduler>
    struct IOLayout;

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              typename Scheduler>
    struct IOLayout<matrix_a, BlockDim, BlockK, DataT, DataLayoutT, Scheduler>
    {
        using SchedulerTraits = scheduler_traits<Scheduler>;

        // For non-interleaved layout, MmaDim is given by BlockDim
        constexpr static uint32_t MmaDim = BlockDim;

        // Large dim affects our layout choice
        constexpr static bool LargeBlockDim = BlockDim > 32u;

        // Determine optimal vector width properties
        constexpr static uint32_t MaxVW = detail::MaxVWSelector<matrix_a,
                                                                BlockDim,
                                                                BlockK,
                                                                DataT,
                                                                DataLayoutT,
                                                                SchedulerTraits::WaveCount>::Result;

        // Note: ColOrthoVW in col_major is only valid for VW = 1
        constexpr static uint32_t VW
            = (is_same_v<DataLayoutT, row_major> || LargeBlockDim) ? MaxVW : 1u;

        // DataLayout selection
        using DataLayout = DataLayout::template Array1d<DataLayoutT>;

        // MatrixLayout selection depends on some factors:
        // - BlockDim
        // - Cooperative wave mode

        // Small dim: ensure our layout mma friendly
        using SmallDimMatrixLayout = MatrixLayout::ColOrthoVW<BlockDim, BlockK, DataT, VW, MaxVW>;

        // Large dim: we can use higher BW layouts, not mma friendly
        using LargeDimMatrixLayout
            = conditional_t<is_same_v<DataLayoutT, col_major>,
                            MatrixLayout::ColInlineVW<BlockDim, BlockK, DataT, VW, MaxVW>,
                            MatrixLayout::ColOrthoVW<BlockDim, BlockK, DataT, VW, MaxVW>>;

        // Select base layout from BlockDim
        using BaseMatrixLayout
            = conditional_t<LargeBlockDim, LargeDimMatrixLayout, SmallDimMatrixLayout>;

        // Select final matrix layout from cooperative mode
        using MatrixLayout = conditional_t<
            SchedulerTraits::is_cooperative,
            MatrixLayout::MatrixCoopLayout<BaseMatrixLayout, SchedulerTraits::WaveCount>,
            BaseMatrixLayout>;

        // Register layout direct to memory storage (load / store)
        using StorageLayout = RegisterLayout::Storage<MatrixLayout, DataLayout>;

        // Register layout required for mma. Expect non-interleaved SOA format.
        // Quirk: gfx11 requires input duplication.
        using MmaLayout = RegisterLayout::MmaInput<MmaDim,
                                                   DataT,
                                                   false,
                                                   (bool)ROCWMMA_ARCH_GFX11
                                                       ? RegisterLayout::Format::WMMA_INPUT_GFX11
                                                       : RegisterLayout::Format::SOA>;

        // Fragments will keep storage register layout.
        using FragmentLayout = StorageLayout;
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              typename Scheduler>
    struct IOLayout<matrix_b, BlockDim, BlockK, DataT, DataLayoutT, Scheduler>
    {
        using SchedulerTraits = scheduler_traits<Scheduler>;

        // For non-interleaved layout, MmaDim is given by BlockDim
        constexpr static uint32_t MmaDim = BlockDim;

        // Large dim affects our layout choice
        constexpr static bool LargeBlockDim = BlockDim > 32u;

        // Determine optimal vector width properties
        constexpr static uint32_t MaxVW = detail::MaxVWSelector<matrix_b,
                                                                BlockDim,
                                                                BlockK,
                                                                DataT,
                                                                DataLayoutT,
                                                                SchedulerTraits::WaveCount>::Result;

        // Note: RowOrthoVW in row_major is only valid for VW = 1
        constexpr static uint32_t VW
            = (is_same_v<DataLayoutT, col_major> || LargeBlockDim) ? MaxVW : 1u;

        // DataLayout selection
        using DataLayout = DataLayout::template Array1d<DataLayoutT>;

        // MatrixLayout selection depends on some factors:
        // - BlockDim
        // - Cooperative wave mode

        // Small dim: ensure our layout mma friendly
        using SmallDimMatrixLayout = MatrixLayout::RowOrthoVW<BlockDim, BlockK, DataT, VW, MaxVW>;

        // Large dim: we can use higher BW layouts, not mma friendly
        using LargeDimMatrixLayout
            = conditional_t<is_same_v<DataLayoutT, row_major>,
                            MatrixLayout::RowInlineVW<BlockDim, BlockK, DataT, VW, MaxVW>,
                            MatrixLayout::RowOrthoVW<BlockDim, BlockK, DataT, VW, MaxVW>>;

        // Select base layout from BlockDim
        using BaseMatrixLayout
            = conditional_t<LargeBlockDim, LargeDimMatrixLayout, SmallDimMatrixLayout>;

        // Select final matrix layout from cooperative mode
        using MatrixLayout = conditional_t<
            SchedulerTraits::is_cooperative,
            MatrixLayout::MatrixCoopLayout<BaseMatrixLayout, SchedulerTraits::WaveCount>,
            BaseMatrixLayout>;

        // Register layout direct to memory storage (load / store)
        using StorageLayout = RegisterLayout::Storage<MatrixLayout, DataLayout>;

        // Register layout required for mma. Expect non-interleaved SOA format.
        // Quirk: gfx11 requires input duplication.
        using MmaLayout = RegisterLayout::MmaInput<MmaDim,
                                                   DataT,
                                                   false,
                                                   (bool)ROCWMMA_ARCH_GFX11
                                                       ? RegisterLayout::Format::WMMA_INPUT_GFX11
                                                       : RegisterLayout::Format::SOA>;

        // Fragments will keep storage register layout.
        using FragmentLayout = StorageLayout;
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              typename Scheduler>
    struct IOLayout<accumulator, BlockDim, BlockK, DataT, DataLayoutT, Scheduler>
    {
        using SchedulerTraits = scheduler_traits<Scheduler>;

        // For non-interleaved layout, MmaDim is given by BlockDim
        constexpr static uint32_t MmaDim = BlockDim;

        // Vector size properties
        constexpr static uint32_t MaxVW = detail::MaxVWSelector<accumulator,
                                                                BlockDim,
                                                                BlockK,
                                                                DataT,
                                                                DataLayoutT,
                                                                SchedulerTraits::WaveCount>::Result;

        // Note: RowOrthoVW in row_major is only valid for VW = 1
        constexpr static uint32_t VW = is_same_v<DataLayoutT, col_major> ? MaxVW : 1u;

        // DataLayout
        using DataLayout = DataLayout::template Array1d<DataLayoutT>;

        // Always mma friendly
        using BaseMatrixLayout = MatrixLayout::RowOrthoVW<BlockDim, BlockK, DataT, VW, MaxVW>;

        // Select final matrix layout from cooperative mode
        using MatrixLayout = conditional_t<
            SchedulerTraits::is_cooperative,
            MatrixLayout::MatrixCoopLayout<BaseMatrixLayout, SchedulerTraits::WaveCount>,
            BaseMatrixLayout>;

        // Register layout direct to memory storage (load / store)
        using StorageLayout = RegisterLayout::Storage<MatrixLayout, DataLayout>;

        // Register layout required for mma. Expect non-interleaved SOA format.
        // Quirk: gfx11 requires padded acc.
        using MmaLayout = RegisterLayout::MmaAcc<MmaDim,
                                                 DataT,
                                                 false,
                                                 (bool)ROCWMMA_ARCH_GFX11
                                                     ? RegisterLayout::Format::WMMA_ACC_GFX11
                                                     : RegisterLayout::Format::SOA>;

        // Fragments will keep storage register layout.
        using FragmentLayout = StorageLayout;
    };

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT, typename Scheduler>
    struct IOLayout<accumulator, BlockDim, BlockK, DataT, void, Scheduler>
    {
        // For non-interleaved layout, MmaDim is given by BlockDim
        constexpr static uint32_t MmaDim = BlockDim;

        // We don't know which storage is needed: no DataLayout
        using StorageLayout = void;

        // Register layout required for mma. Expect non-interleaved SOA format.
        // Quirk: gfx11 requires padded acc.
        using MmaLayout = RegisterLayout::MmaAcc<MmaDim,
                                                 DataT,
                                                 false,
                                                 (bool)ROCWMMA_ARCH_GFX11
                                                     ? RegisterLayout::Format::WMMA_ACC_GFX11
                                                     : RegisterLayout::Format::SOA>;

        // Fragments will assume default mma register layout.
        using FragmentLayout
            = RegisterLayout::MmaAcc<MmaDim, DataT, false, RegisterLayout::Format::SOA>;
    };

    namespace detail
    {
        template <uint32_t BlockDim,
                  typename DataT,
                  uint32_t MmaDim = (bool)ROCWMMA_BLOCK_DIM_32_SUPPORTED ? 32u : 16u>
        struct MmaDimSelector
        {
        private:
            // Smallest valid mma dim for mfma/wmma.
            // Test MmaDim must not exceed BlockDim for valid layout.
            static constexpr uint32_t MinMmaDim  = 16u;
            static constexpr uint32_t TestMmaDim = min(BlockDim, MmaDim);

            // For valid mma sizes, (BlockDim >= 16)
            // Find minimum 16 byte load with MmaDim of 32 or 16
            static constexpr uint32_t MinLargeBytes  = 16u;
            static constexpr uint32_t DimPerThread   = BlockDim / TestMmaDim;
            static constexpr uint32_t BytesPerThread = DimPerThread * sizeof(DataT);
            static constexpr uint32_t MmaDimResult
                = (BytesPerThread < MinLargeBytes ? MinMmaDim : TestMmaDim);

            // For invalid mma sizes (BlockDim < 16), we can have smaller MmaDim to increase VW.
            // Try to balance DimPerThread and KPerThread by aiming to get half BlockDim bytes.
            static constexpr bool     SmallDim      = TestMmaDim < MinMmaDim;
            static constexpr uint32_t MinSmallBytes = BlockDim / 2u * sizeof(DataT);
            static constexpr uint32_t SmallDimResult
                = (BytesPerThread < MinSmallBytes)
                      ? MmaDimSelector<BlockDim, DataT, TestMmaDim / 2u>::Result
                      : TestMmaDim;

        public:
            // TODO: Remove the min with 16u when MmaDim32 is implemented for interleaved layouts
            static constexpr uint32_t Result = SmallDim ? SmallDimResult : min(MmaDimResult, 16u);
        };

        template <uint32_t BlockDim, typename DataT>
        struct MmaDimSelector<BlockDim, DataT, 0u>
        {
            static constexpr uint32_t Result = 1u;
        };

    } // namespace detail

    /*! \struct IOLayoutInt
    *  \brief Definition of VW, MaxVW, data and matrix mapping utilities
    *         in specific matrix context.
    *
    * @tparam MatrixT fragment context
    * @tparam BlockDim Block leading dimension
    * @tparam BlockK Block K-dimension
    * @tparam DataT data type
    * @tparam DataLayoutT in-memory layout as col_major or row_major
    * @tparam Scheduler wave-wise scheduler
    */
    template <typename MatrixT,
              uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              typename Scheduler>
    struct IOLayoutInt;

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              typename Scheduler>
    struct IOLayoutInt<matrix_a, BlockDim, BlockK, DataT, DataLayoutT, Scheduler>
    {
        using SchedulerTraits = scheduler_traits<Scheduler>;

        // SplitK factor for interleaved layouts can be applied to WaveCount
        constexpr static uint32_t SplitK = SchedulerTraits::WaveCount;

        // Select an appropriate MmaDim
        constexpr static uint32_t MmaDim = detail::MmaDimSelector<BlockDim, DataT>::Result;

        // DataLayout
        using DataLayout = DataLayout::template Array1d<DataLayoutT>;

        // Matrix Layouts
        using BaseMatrixLayout
            = conditional_t<is_same_v<DataLayoutT, col_major>,
                            MatrixLayout::ColInlineInt<BlockDim, BlockK, DataT, MmaDim, SplitK>,
                            MatrixLayout::ColOrthoInt<BlockDim, BlockK, DataT, MmaDim, SplitK>>;

        // Select final matrix layout from cooperative mode
        using MatrixLayout = conditional_t<
            SchedulerTraits::is_cooperative,
            MatrixLayout::MatrixCoopLayout<BaseMatrixLayout, SchedulerTraits::WaveCount>,
            BaseMatrixLayout>;

        // Register layout direct to memory storage (load / store)
        using StorageLayout = RegisterLayout::Storage<MatrixLayout, DataLayout>;

        // Register layout required for mma. Expect interleaved SOA format.
        // Quirk: gfx11 requires input duplication.
        using MmaLayout = RegisterLayout::MmaInput<MmaDim,
                                                   DataT,
                                                   true,
                                                   (bool)ROCWMMA_ARCH_GFX11
                                                       ? RegisterLayout::Format::WMMA_INPUT_GFX11
                                                       : RegisterLayout::Format::SOA_INT>;

        // Fragments will keep storage register layout.
        using FragmentLayout = StorageLayout;

        // Vector size properties derived from the matrix layout
        constexpr static uint32_t MaxVW = layout_traits<MatrixLayout>::MaxVectorWidth;
        constexpr static uint32_t VW    = MaxVW;
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              typename Scheduler>
    struct IOLayoutInt<matrix_b, BlockDim, BlockK, DataT, DataLayoutT, Scheduler>
    {
        using SchedulerTraits = scheduler_traits<Scheduler>;

        // SplitK factor for interleaved layouts can be applied to WaveCount
        constexpr static uint32_t SplitK = SchedulerTraits::WaveCount;

        // Select an appropriate MmaDim
        constexpr static uint32_t MmaDim = detail::MmaDimSelector<BlockDim, DataT>::Result;

        // DataLayout
        using DataLayout = DataLayout::template Array1d<DataLayoutT>;

        // Matrix Layouts
        using BaseMatrixLayout
            = conditional_t<is_same_v<DataLayoutT, col_major>,
                            MatrixLayout::RowOrthoInt<BlockDim, BlockK, DataT, MmaDim, SplitK>,
                            MatrixLayout::RowInlineInt<BlockDim, BlockK, DataT, MmaDim, SplitK>>;

        // Select final matrix layout from cooperative mode
        using MatrixLayout = conditional_t<
            SchedulerTraits::is_cooperative,
            MatrixLayout::MatrixCoopLayout<BaseMatrixLayout, SchedulerTraits::WaveCount>,
            BaseMatrixLayout>;

        // Register layout direct to memory storage (load / store)
        using StorageLayout = RegisterLayout::Storage<MatrixLayout, DataLayout>;

        // Register layout required for mma. Expect interleaved SOA format.
        // Quirk: gfx11 requires input duplication.
        using MmaLayout = RegisterLayout::MmaInput<MmaDim,
                                                   DataT,
                                                   true,
                                                   (bool)ROCWMMA_ARCH_GFX11
                                                       ? RegisterLayout::Format::WMMA_INPUT_GFX11
                                                       : RegisterLayout::Format::SOA_INT>;
        // Fragments will keep storage register layout.
        using FragmentLayout = StorageLayout;

        // Vector size properties derived from the matrix layout
        constexpr static uint32_t MaxVW = layout_traits<MatrixLayout>::MaxVectorWidth;
        constexpr static uint32_t VW    = MaxVW;
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              typename Scheduler>
    struct IOLayoutInt<accumulator, BlockDim, BlockK, DataT, DataLayoutT, Scheduler>
    {
        using SchedulerTraits = scheduler_traits<Scheduler>;

        // SplitK factor for interleaved layouts can be applied to WaveCount
        constexpr static uint32_t SplitK = SchedulerTraits::WaveCount;

        // Select an appropriate MmaDim
        constexpr static uint32_t MmaDim = detail::MmaDimSelector<BlockDim, DataT>::Result;

        // DataLayout
        using DataLayout = DataLayout::template Array1d<DataLayoutT>;

        // Matrix Layouts
        using BaseMatrixLayout
            = conditional_t<is_same_v<DataLayoutT, col_major>,
                            MatrixLayout::RowOrthoInt<BlockDim, BlockK, DataT, MmaDim, SplitK>,
                            MatrixLayout::RowInlineInt<BlockDim, BlockK, DataT, MmaDim, SplitK>>;

        // Select final matrix layout from cooperative mode
        using MatrixLayout = conditional_t<
            SchedulerTraits::is_cooperative,
            MatrixLayout::MatrixCoopLayout<BaseMatrixLayout, SchedulerTraits::WaveCount>,
            BaseMatrixLayout>;

        // Register layout direct to memory storage (load / store)
        using StorageLayout = RegisterLayout::Storage<MatrixLayout, DataLayout>;

        // Register layout required for mma. Expect interleaved accum format for multiple blocks.
        // Quirk: gfx11 requires padded mma acc
        using MmaLayout = RegisterLayout::MmaAcc<MmaDim,
                                                 DataT,
                                                 true,
                                                 (bool)ROCWMMA_ARCH_GFX11
                                                     ? RegisterLayout::Format::WMMA_ACC_GFX11
                                                     : RegisterLayout::Format::ACC_INT_A_MAJOR>;

        // Fragments will keep mma register layout.
        using FragmentLayout
            = RegisterLayout::MmaAcc<MmaDim, DataT, true, RegisterLayout::Format::ACC_INT_A_MAJOR>;

        // Vector size properties derived from the matrix layout
        constexpr static uint32_t MaxVW = layout_traits<MatrixLayout>::MaxVectorWidth;
        constexpr static uint32_t VW    = MaxVW;
    };

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT, typename Scheduler>
    struct IOLayoutInt<accumulator, BlockDim, BlockK, DataT, void, Scheduler>
    {
        // Select an appropriate MmaDim
        constexpr static uint32_t MmaDim = detail::MmaDimSelector<BlockDim, DataT>::Result;

        // We don't know which storage is needed: no DataLayout
        using StorageLayout = void;

        // Register layout required for mma. Expect interleaved accum format for multiple blocks.
        // Quirk: gfx11 requires padded mma acc
        using MmaLayout = RegisterLayout::MmaAcc<MmaDim,
                                                 DataT,
                                                 true,
                                                 (bool)ROCWMMA_ARCH_GFX11
                                                     ? RegisterLayout::Format::WMMA_ACC_GFX11
                                                     : RegisterLayout::Format::ACC_INT_A_MAJOR>;

        // Fragments will keep mma interleaved layout.
        using FragmentLayout
            = RegisterLayout::MmaAcc<MmaDim, DataT, true, RegisterLayout::Format::ACC_INT_A_MAJOR>;
    };

} // namespace rocwmma

#endif // ROCWMMA_IO_LAYOUT_HPP
