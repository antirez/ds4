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
#ifndef ROCWMMA_MAPPING_UTIL_IMPL_HPP
#define ROCWMMA_MAPPING_UTIL_IMPL_HPP

#if !defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
#endif

#include "constants.hpp"
#include "mapping_util.hpp"
#include "types.hpp"
#include "utils.hpp"

namespace rocwmma
{

    namespace detail
    {
        // Workgroup configuration
        // Assumption: major thread dimension is X
        // Notation (x, y) = (row, col)
        ROCWMMA_DEVICE inline uint32_t laneId()
        {
            // threadIdx.x % AMDGCN_WAVE_SIZE;
            return threadIdx.x & (Constants::AMDGCN_WAVE_SIZE - 1u);
        }

        ROCWMMA_DEVICE constexpr inline Coord2d waveCount(Coord2d const& threadCount)
        {
            // waveCount.x = threadCount.x / AMDGCN_WAVE_SIZE
            // waveCount.y = threadCount.y
            return make_coord2d(get<0>(threadCount) >> Log2<Constants::AMDGCN_WAVE_SIZE>::value,
                                get<1>(threadCount));
        }

        ROCWMMA_DEVICE constexpr inline Coord2d threadCount(Coord2d const& waveCount)
        {
            // threadCount.x = waveCount.x * AMDGCN_WAVE_SIZE
            // threadCount.y = waveCount.y
            return make_coord2d(get<0>(waveCount) << Log2<Constants::AMDGCN_WAVE_SIZE>::value,
                                get<1>(waveCount));
        }

        /// WaveSpace

        template <uint32_t TBlockX, uint32_t TBlockY>
        ROCWMMA_DEVICE inline uint32_t WaveSpace<TBlockX, TBlockY>::localLaneId()
        {
            return laneId();
        }

        template <uint32_t TBlockX, uint32_t TBlockY>
        ROCWMMA_DEVICE constexpr inline auto WaveSpace<TBlockX, TBlockY>::localWaveCoord()
            -> WaveCoordT
        {
            return waveCount(make_coord2d(static_cast<uint32_t>(threadIdx.x),
                                          static_cast<uint32_t>(threadIdx.y)));
        }

        template <uint32_t TBlockX, uint32_t TBlockY>
        ROCWMMA_DEVICE inline auto WaveSpace<TBlockX, TBlockY>::globalWaveCoord() -> WaveCoordT
        {
            return waveCount(make_coord2d(blockIdx.x * TBlockX + threadIdx.x,
                                          blockIdx.y * TBlockY + threadIdx.y));
        }

        template <>
        ROCWMMA_DEVICE inline auto WaveSpace<0, 0>::globalWaveCoord() -> WaveCoordT
        {
            return waveCount(make_coord2d(blockIdx.x * blockDim.x + threadIdx.x,
                                          blockIdx.y * blockDim.y + threadIdx.y));
        }

        template <uint32_t TBlockX, uint32_t TBlockY>
        ROCWMMA_DEVICE constexpr inline auto WaveSpace<TBlockX, TBlockY>::workgroupCoord()
            -> WorkgroupCoordT
        {
            return make_coord2d(static_cast<uint32_t>(blockIdx.x),
                                static_cast<uint32_t>(blockIdx.y));
        }

        template <uint32_t TBlockX, uint32_t TBlockY>
        template <bool IsConst /* = (TBlockX > 0u && TBlockY > 0u) */,
                  enable_if_t<IsConst>* /* = nullptr */>
        ROCWMMA_DEVICE constexpr inline auto WaveSpace<TBlockX, TBlockY>::workgroupDim()
            -> WorkgroupDimT
        {
            return waveCount(make_coord2d(TBlockX, TBlockY));
        }

        template <uint32_t TBlockX, uint32_t TBlockY>
        template <bool IsConst /* = (TBlockX > 0u && TBlockY > 0u) */,
                  enable_if_t<!IsConst>* /* = nullptr */>
        ROCWMMA_DEVICE inline auto WaveSpace<TBlockX, TBlockY>::workgroupDim() -> WorkgroupDimT
        {
            return waveCount(make_coord2d(blockDim.x, blockDim.y));
        }

        /// MatrixSpace
        template <uint32_t BlockHeight, uint32_t BlockWidth>
        ROCWMMA_DEVICE inline auto
            MatrixSpace<BlockHeight, BlockWidth>::fromBlockCoord(BlockCoordT const& blockCoord)
                -> MatrixCoordT
        {
            // Map block to matrix coordinate space.
            return make_coord2d(get<0>(blockCoord) * BlockHeight, // ROW
                                get<1>(blockCoord) * BlockWidth); // COL
        }

        /// DataSpace
        template <typename DataOrientation>
        ROCWMMA_DEVICE constexpr inline auto
            DataSpace<DataOrientation>::leadingDim(MatrixSizeT const& matrixSize)
        {
            return get<MinorIndex>(matrixSize);
        }

        template <typename DataOrientation>
        ROCWMMA_DEVICE constexpr inline auto
            DataSpace<DataOrientation>::fromMatrixCoord(MatrixCoordT const& matrixCoord,
                                                        uint32_t            leadingDim)
        {
            // 1D data element offset transform
            return get<MajorIndex>(matrixCoord) * leadingDim + get<MinorIndex>(matrixCoord);
        }

    } // namespace detail

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT, typename DataLayout>
    ROCWMMA_DEVICE inline uint32_t MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::laneId()
    {
        return WaveSpace::localLaneId();
    }

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT, typename DataLayout>
    ROCWMMA_DEVICE inline auto MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::waveCoord()
        -> WaveCoordT
    {
        return WaveSpace::localWaveCoord();
    }

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT, typename DataLayout>
    ROCWMMA_DEVICE inline auto MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::blockCoord()
        -> BlockCoordT
    {
        // Map each wave 1 : 1 to global block grid
        return WaveSpace::globalWaveCoord();
    }

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT, typename DataLayout>
    ROCWMMA_DEVICE inline auto
        MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::matrixCoord() -> MatrixCoordT
    {
        return MatrixSpace::fromBlockCoord(blockCoord());
    }

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT, typename DataLayout>
    ROCWMMA_DEVICE inline DataT const*
        MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::dataCoord(DataT const* baseAddr,
                                                                           uint32_t     ldm)
    {
        return baseAddr + DataSpace::fromMatrixCoord(matrixCoord(), ldm);
    }

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT, typename DataLayout>
    ROCWMMA_DEVICE inline DataT*
        MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::dataCoord(DataT*   baseAddr,
                                                                           uint32_t ldm)
    {
        return baseAddr + DataSpace::fromMatrixCoord(matrixCoord(), ldm);
    }

    /// Current workgroup perspective

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT, typename DataLayout>
    ROCWMMA_DEVICE inline auto
        MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::workgroupDim() -> WorkgroupDimT
    {
        return WaveSpace::workgroupDim();
    }

    /// Coordinate override helpers

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT, typename DataLayout>
    ROCWMMA_DEVICE inline auto
        MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::blockCoordM(uint32_t m)
            -> BlockCoordT
    {
        auto coord    = blockCoord();
        get<0>(coord) = m;
        return coord;
    }

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT, typename DataLayout>
    ROCWMMA_DEVICE inline auto
        MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::blockCoordN(uint32_t n)
            -> BlockCoordT
    {
        auto coord    = blockCoord();
        get<1>(coord) = n;
        return coord;
    }

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT, typename DataLayout>
    ROCWMMA_DEVICE inline auto
        MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::matrixCoordM(uint32_t m)
            -> MatrixCoordT
    {
        auto coord    = matrixCoord();
        get<0>(coord) = m;
        return coord;
    }

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT, typename DataLayout>
    ROCWMMA_DEVICE inline auto
        MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::matrixCoordN(uint32_t n)
            -> MatrixCoordT
    {
        auto coord    = matrixCoord();
        get<1>(coord) = n;
        return coord;
    }

    /// Conversion helpers

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT, typename DataLayout>
    ROCWMMA_DEVICE inline auto MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::matrixCoord(
        BlockCoordT const& blockCoord) -> MatrixCoordT
    {
        return MatrixSpace::fromBlockCoord(forward<BlockCoordT const>(blockCoord));
    }

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT, typename DataLayout>
    ROCWMMA_DEVICE inline uint32_t
        MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::dataOffset(
            MatrixCoordT const& matrixCoord, uint32_t ldm)
    {
        return DataSpace::fromMatrixCoord(forward<MatrixCoordT const>(matrixCoord), ldm);
    }

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT, typename DataLayout>
    ROCWMMA_DEVICE inline DataT const*
        MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::dataCoord(
            DataT const* baseAddr, MatrixCoordT const& matrixCoord, uint32_t ldm)
    {
        return baseAddr + DataSpace::fromMatrixCoord(forward<MatrixCoordT const>(matrixCoord), ldm);
    }

    template <uint32_t BlockHeight, uint32_t BlockWidth, typename DataT, typename DataLayout>
    ROCWMMA_DEVICE inline DataT* MappingUtil<BlockHeight, BlockWidth, DataT, DataLayout>::dataCoord(
        DataT* baseAddr, MatrixCoordT const& matrixCoord, uint32_t ldm)
    {
        return baseAddr + DataSpace::fromMatrixCoord(forward<MatrixCoordT const>(matrixCoord), ldm);
    }

} // namespace rocwmma

#endif // ROCWMMA_MAPPING_UTIL_IMPL_HPP
