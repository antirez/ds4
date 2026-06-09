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
#ifndef ROCWMMA_MATRIX_LAYOUT_IMPL_HPP
#define ROCWMMA_MATRIX_LAYOUT_IMPL_HPP

#include "../utility/algorithm.hpp"
#include "../utility/vector.hpp"
#include "layout.hpp"
#include "layout_traits.hpp"
#include "matrix_layout_base.hpp"
#include "matrix_ortho_layout.hpp"

namespace rocwmma
{

    // Implementations for the MatrixLayout classes
    namespace MatrixLayout
    {
        /* Pattern that maps threads contiguously to matrix columns and assumes
        * that VW will be mapped orthogonally to the column.
        * This pattern considers VW up to MaxVW, BlockDim <= 64 and BlockDim > 64.
        *
        * Iterative thread stride cycles (same for all threads):
        *   Fill MaxVW => Fill BlockK => Fill BlockDim
        *
        * Example:
        *  BlockDim = 128   BlockK = 16
        *  MaxVW = 4       VW = 1
        *
        *  BlockDim Stride Count = 2, BlockDimStride = (64, 0)
        *  BlockK   Stride Count = 4, BlockKStride   = (0,  4)
        *  VW       Stride Count = 4, VWStride       = (0,  1)
        *
        *  Stride mapping (BlockDim, BlockK, VW)
        *  C_n = Matrix column
        *  i_n = cumulative iteration
        *
        *   kDim --------->
        *                     VW Stride
        *   BlockDim          |--1--|
        *   |                 |-- BlockK Stride = 4 --|
        *   |                 i0(0,0,0)   i2(0,0,2)   i4(0,1,0)   i6(0,1,2)         i14(0,3,2)
        *   |            --   v_____ _____v_____ _____v_____ _____v_____ _____      v_____  _____
        *   v            |    |     |     |     |     |     |     |     |     |     |     ||     |
        *                |    |     |     |     |     |     |     |     |     |     |     ||     |
        *       BlockDim 64   | C0  |  C1 |  C2 |  C3 |  C4 |  C5 | C6  | C7  | ... | C14 || C15 |
        *        Stride  |    |     |     |     |     |     |     |     |     |     |     ||     |
        *                --   |_____|_____|_____|_____|_____|_____|_____|_____|     |_____||_____|
        *                     i16(1,0,0)  i18(1,0,2)  i20(1,1,0)  i22(1,1,2)        i30(1,3,2)
        *                     v_____ _____v_____ _____v_____ _____v_____ _____      v_____  _____
        *                     |     |     |     |     |     |     |     |     |     |     ||     |
        *                     |     |     |     |     |     |     |     |     |     |     ||     |
        *                     | C0  |  C1 |  C2 |  C3 | C4  | C5  | C6  | C7  | ... | C14 || C15 |
        *                     |     |     |     |     |     |     |     |     |     |     ||     |
        *                     |_____|_____|_____|_____|_____|_____|_____|_____|     |_____||_____|
        *                     ^(128, 0)                                                           ^(BlockDim, BlockK)
        *   ...                                          ...
        *
        * Register file (for all VectorWidths = [1, MaxVectorWidth]):
        *
        * Elements 0..............63
        *           ______________
        *  Reg0    |  C0 [63:0]    |
        *  Reg1    |  C1 [63:0]    |
        *  Reg2    |  C2 [63:0]    |
        *  ...       ...
        *  Reg15   |  C15[63:0]    |
        *  Reg16   |  C0 [127:64]  |
        *  ...       ...
        *  Reg31   |  C15 [127:64] |
        }*/

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct ColOrthoVW : public MatrixLayoutBase<
                                ColOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
        {
            struct Traits
            {
                // Number of threads per wave
                static constexpr uint32_t WaveSize = Constants::AMDGCN_WAVE_SIZE;

                // Stride between tiles
                static constexpr uint32_t BlockDimStride_X = min(BlockDim, WaveSize);
                static constexpr uint32_t BlockDimStride_Y = 0u;

                static constexpr uint32_t BlockKStride_X = 0u;
                static constexpr uint32_t BlockKStride_Y
                    = WaveSize * MaxVectorWidth / BlockDimStride_X;

                static constexpr uint32_t VWStride_X = 0u;
                static constexpr uint32_t VWStride_Y = VectorWidth;

                // Stride space
                static constexpr uint32_t BlockDimSegs = BlockDim / BlockDimStride_X;
                static constexpr uint32_t BlockKSegs   = BlockK / BlockKStride_Y;
                static constexpr uint32_t VWSegs       = MaxVectorWidth / VWStride_Y;

                // Thread-tile perspective
                // TODO: rename to ThreadTile...
                static constexpr uint32_t DimPerThread = BlockKSegs;
                static constexpr uint32_t KPerThread   = MaxVectorWidth;
                static constexpr uint32_t ElementsPerThread
                    = DimPerThread * KPerThread * BlockDimSegs;

                static_assert(MaxVectorWidth <= BlockK, "MaxVectorWidth cannot exceed BlockK");
                static_assert(BlockDim >= BlockDimStride_X,
                              "BlockDim must be larger than BlockDimStride_X");
                static_assert(BlockDim % BlockDimStride_X == 0,
                              "BlockDim must be a multiple of BlockDimStride_X");
                static_assert(BlockK >= BlockKStride_Y,
                              "BlockK must be larger than BlockKStride_Y");
                static_assert(BlockK % BlockKStride_Y == 0,
                              "BlockK must be a multiple of BlockKStride_Y");
                static_assert(MaxVectorWidth >= VWStride_Y,
                              "MaxVectorWidth must larger than VWStride_Y");
                static_assert(MaxVectorWidth % VWStride_Y == 0,
                              "MaxVectorWidth must be a multiple of VWStride_Y");
            };

            ROCWMMA_DEVICE constexpr static inline auto strideCounts()
            {
                return make_vector(Traits::BlockDimSegs, // BlockDim Segments
                                   Traits::BlockKSegs, // BlockK Segments
                                   Traits::VWSegs); // VW Segments
            }

            ROCWMMA_DEVICE constexpr static inline auto strides()
            {
                return make_vector(make_coord2d(Traits::BlockDimStride_X, Traits::BlockDimStride_Y),
                                   make_coord2d(Traits::BlockKStride_X, Traits::BlockKStride_Y),
                                   make_coord2d(Traits::VWStride_X, Traits::VWStride_Y));
            }

            ROCWMMA_DEVICE static inline auto baseOffset()
            {
                if constexpr(Traits::BlockDimStride_X >= Traits::WaveSize)
                {
                    // Don't need initial offset calc in Y direction: all threads fit in neighbouring rows
                    return make_coord2d(threadIdx.x % Traits::BlockDimStride_X, 0u);
                }
                else
                {
                    // Threads need to spread over the Y direction as well
                    return make_coord2d(threadIdx.x % Traits::BlockDimStride_X,
                                        (threadIdx.x / Traits::BlockDimStride_X) * MaxVectorWidth
                                            % Traits::BlockKStride_Y);
                }
            }
        };

        /* Pattern that maps threads to matrix columns and assumes
        * that VW will be mapped inline with the column.
        * This pattern considers VW up to MaxVW, BlockDim <= 64 and BlockDim > 64.
        *
        * Iterative thread stride cycles (same for all threads):
        *   Fill MaxVW => Fill BlockK => Fill BlockDim
        *
        * Example:
        * BlockDim = 256   BlockK = 4
        * MaxVW = 2       VW = 1
        *
        * BlockDim Stride Count = 4, BlockDimStride = (64, 0)
        * BlockK   Stride Count = 2, BlockKStride   = (0,  2)
        * VW       Stride Count = 2, VWStride       = (1,  0)
        *
        * Stride mapping (BlockDim, BlockK, VW)
        *  C_n = Matrix column
        *  i_n = cumulative iteration
        *
        *  Cartesian iteration offsets (row, col):
        *  i0  = (0,   0) i1  = (1,   0) i2  = (0,   2) i3  = (1,   2)
        *  i4  = (64,  0) i5  = (65,  0) i6  = (64,  2) i7  = (65,  2)
        *  i8  = (128, 0) i9  = (129, 0) i10 = (128, 2) i11 = (129, 2)
        *  i12 = (192, 0) i13 = (193, 0) i14 = (192, 2) i15 = (192, 2)
        *
        *  Strides iteration offsets (BlockDim, BlockK, VW):
        *  i0  = (0,0,0) i1  = (0,0,1)
        *  i2  = (0,1,0) i3  = (0,1,1)
        *  i4  = (1,0,0) i5  = (1,0,1)
        *  i6  = (1,1,0) i7  = (1,1,1)
        *  i8  = (2,0,0) i9  = (2,0,1)
        *  i10 = (2,1,0) i11 = (2,1,1)
        *  i12 = (3,0,0) i13 = (3,0,1)
        *  i14 = (3,1,0) i15 = (3,1,1)
        *
        * Let's follow thread 0:
        *
        *   kDim --------->
        *
        *   BlockDim1
        *   |                           |-- BlockK Stride = 2 --|
        *   |                           i0(0,0,0)   i2(0,1,0)
        *   |            _         _    v_____ _____v_____ _____
        *   v            |         |    |     |     |     |     |
        *                |  VW     1    |     |     |     |     |
        *       BlockDim |  Stride |    | C0  |  C1 |  C2 |  C3 |
        *        Stride  |         _    v     |     v     |     |
        *               64              i1(0,0,1)   i3(0,1,1)   |
        *                |              |     |     |     |     |
        *                |              |     |     |     |     |
        *                |              | C0  |  C1 |  C2 |  C3 |
        *                _              |_____|_____|_____|_____|
        *                               i4(1,0,0)   i6(1,1,0)
        *                               v_____ _____v_____ _____
        *                               |     |     |     |     |
        *                               |     |     |     |     |
        *                               | C0  |  C1 |  C2 |  C3 |
        *                               v     |     v     |     |
        *                               i5(1,0,1)   i7(1,1,1)   |
        *                               |     |     |     |     |
        *                               |     |     |     |     |
        *                               | C0  |  C1 |  C2 |  C3 |
        *                               |_____|_____|_____|_____|
        *                               ...                     ...
        *                               ...                     ...
        *                               ...                     ...
        *                               v     |     v     |     |
        *                               i13(3,0,1)   i14(3,1,1)   |
        *                               |     |     |     |     |
        *                               |     |     |     |     |
        *                               | C0  |  C1 |  C2 |  C3 |
        *                               |_____|_____|_____|_____|
        *
        *                               ^(BlockDim, 0)          ^(BlockDim, BlockK)
        *
        * Register file (for all VectorWidths = [MaxVectorWidth, 1]):
        *
        * Elements 0...........1........................................... ............64
        *         ________________________________________________________________________
        * Reg0   |  C0E0   |  C0E2   | ... |  C0E62   |  C1E0   |  C1E2   | ... |  C1E62  |
        * Reg1   |  C0E1   |  C0E3   | ... |  C0E63   |  C1E1   |  C1E3   | ... |  C1E63  |
        * Reg2   |  C2E0   |  C2E2   | ... |  C2E62   |  C3E0   |  C3E2   | ... |  C3E62  |
        * Reg3   |  C2E1   |  C2E3   | ... |  C2E63   |  C3E1   |  C3E3   | ... |  C3E63  |
        * Reg4   |  C0E64  |  C0E66  | ... |  C0E126  |  C1E64  |  C1E66  | ... |  C1E126 |
        * Reg5   |  C0E65  |  C0E67  | ... |  C0E127  |  C1E65  |  C1E67  | ... |  C1E127 |
        * ...      ...
        * Reg10  |  C2E192 |  C2E194 | ... |  C2E254  |  C3E192 |  C3E194 | ... |  C3E254 |
        * Reg11  |  C2E193 |  C2E195 | ... |  C2E255  |  C3E193 |  C3E195 | ... |  C3E255 |
        *
        */

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct ColInlineVW : public MatrixLayoutBase<
                                 ColInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
        {

            struct Traits
            {
                // Number of threads per wave
                static constexpr uint32_t WaveSize = Constants::AMDGCN_WAVE_SIZE;

                // Strides
                static constexpr uint32_t BlockDimStride_X = min(BlockDim, WaveSize);
                static constexpr uint32_t BlockDimStride_Y = 0u;

                static constexpr uint32_t BlockKStride_X = 0u;
                static constexpr uint32_t BlockKStride_Y
                    = WaveSize * MaxVectorWidth / BlockDimStride_X;

                static constexpr uint32_t VWStride_X = VectorWidth;
                static constexpr uint32_t VWStride_Y = 0u;

                // Stride Space
                static constexpr uint32_t BlockDimSegs = BlockDim / BlockDimStride_X;
                static constexpr uint32_t BlockKSegs   = BlockK / BlockKStride_Y;
                static constexpr uint32_t VWSegs       = MaxVectorWidth / VWStride_X;

                // Thread-tile perspective
                // TODO: rename to ThreadTile...
                static constexpr uint32_t DimPerThread = MaxVectorWidth;
                static constexpr uint32_t KPerThread   = BlockKSegs;
                static constexpr uint32_t ElementsPerThread
                    = DimPerThread * KPerThread * BlockDimSegs;

                // Sanity checks for strides sizes
                static_assert(MaxVectorWidth <= BlockDim, "MaxVectorWidth cannot exceed BlockDim");
                static_assert(BlockDim >= BlockDimStride_X,
                              "BlockDim must be larger than BlockDimStride_X");
                static_assert(BlockDim % BlockDimStride_X == 0,
                              "BlockDim must be a multiple of BlockDimStride_X");
                static_assert(BlockK >= BlockKStride_Y,
                              "BlockK must be larger than BlockKStride_Y");
                static_assert(BlockK % BlockKStride_Y == 0,
                              "BlockK must be a multiple of BlockKStride_Y");
                static_assert(MaxVectorWidth >= VWStride_X,
                              "MaxVectorWidth must larger than VWStride_X");
                static_assert(MaxVectorWidth % VWStride_X == 0,
                              "MaxVectorWidth must be a multiple of VWStride_X");
            };

            ROCWMMA_DEVICE constexpr static inline auto strideCounts()
            {
                return make_vector(Traits::BlockDimSegs, // BlockDim Segments
                                   Traits::BlockKSegs, // BlockK Segments
                                   Traits::VWSegs); // VW Segments
            }

            ROCWMMA_DEVICE constexpr static inline auto strides()
            {
                return make_vector(make_coord2d(Traits::BlockDimStride_X, Traits::BlockDimStride_Y),
                                   make_coord2d(Traits::BlockKStride_X, Traits::BlockKStride_Y),
                                   make_coord2d(Traits::VWStride_X, Traits::VWStride_Y));
            }

            ROCWMMA_DEVICE static inline auto baseOffset()
            {
                if constexpr((Traits::BlockDimStride_X >= Traits::WaveSize)
                             && (MaxVectorWidth == 1))
                {
                    // Don't need initial offset calc in Y direction: all threads fit in neighbouring rows
                    return make_coord2d(threadIdx.x % Traits::BlockDimStride_X, 0u);
                }
                else
                {
                    // Threads need to spread over the Y direction as well
                    return make_coord2d(threadIdx.x * MaxVectorWidth % Traits::BlockDimStride_X,
                                        threadIdx.x * MaxVectorWidth / Traits::BlockDimStride_X
                                            % Traits::BlockKStride_Y);
                }
            }
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MmaDim, // Mma instruction size
                  uint32_t SplitK /* = 1*/> // # of splits
        struct ColInlineInt
            : public MatrixLayoutBase<ColInlineInt<BlockDim, BlockK, DataT, MmaDim, SplitK>>
        {
            struct Traits
            {
                // Number of threads per wave
                static constexpr uint32_t WaveSize = Constants::AMDGCN_WAVE_SIZE;

                // Number of elements each thread will fetch in BlockDim direction
                static constexpr uint32_t DimPerThread = BlockDim / MmaDim;

                // Number of elements each thread will fetch in BlockK direction
                static constexpr uint32_t KPerThread = BlockK * MmaDim / (WaveSize * SplitK);

                // How many elements each thread will gather
                static constexpr uint32_t ElementsPerThread = DimPerThread * KPerThread;

                // Strides
                static constexpr uint32_t SplitKStride_X = 0u;
                static constexpr uint32_t SplitKStride_Y = BlockK / SplitK;

                static constexpr uint32_t BlockKStride_X = 0u;
                static constexpr uint32_t BlockKStride_Y = 1u;

                static constexpr uint32_t VWStride_X = DimPerThread;
                static constexpr uint32_t VWStride_Y = 0u;

                // Stride Space
                static constexpr uint32_t SplitKSegs = BlockK / SplitKStride_Y;
                static constexpr uint32_t BlockKSegs = KPerThread / BlockKStride_Y;
                static constexpr uint32_t VWSegs     = DimPerThread / VWStride_X;

                // Check KPerThread validity
                static_assert(BlockK >= KPerThread, "Invalid KPerThread");
                static_assert(KPerThread > 0, "Invalid KPerThread - try a larger block size");
                static_assert(BlockK % KPerThread == 0, "BlockK is not a multiple of KPerThread");

                // Check SplitK validity
                static_assert(BlockK >= SplitK, "Invalid SplitK");
                static_assert(SplitK > 0, "Invalid SplitK");
                static_assert(BlockK % SplitK == 0, "BlockK is not a multiple of SplitK");

                // Check MmaDim validity
                static_assert(BlockDim >= MmaDim, "BlockDim must be larger than MmaDim");
                static_assert(BlockDim % MmaDim == 0, "BlockDim must be a multiple of MmaDim");
            };

            ROCWMMA_DEVICE constexpr static inline auto strideCounts()
            {
                return make_vector(Traits::SplitKSegs, Traits::BlockKSegs, Traits::VWSegs);
            }

            ROCWMMA_DEVICE constexpr static inline auto strides()
            {
                return make_vector(make_coord2d(Traits::SplitKStride_X, Traits::SplitKStride_Y),
                                   make_coord2d(Traits::BlockKStride_X, Traits::BlockKStride_Y),
                                   make_coord2d(Traits::VWStride_X, Traits::VWStride_Y));
            }

            ROCWMMA_DEVICE static inline auto baseOffset()
            {
                return make_coord2d((threadIdx.x * Traits::DimPerThread) % BlockDim,
                                    ((threadIdx.x / MmaDim) * Traits::KPerThread)
                                        % (BlockK / SplitK));
            }
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MmaDim, // Mma instruction size
                  uint32_t SplitK /*= 1*/> // # of splits
        struct ColOrthoInt
            : public MatrixLayoutBase<ColOrthoInt<BlockDim, BlockK, DataT, MmaDim, SplitK>>
        {
            struct Traits
            {
                // Number of threads per wave
                static constexpr uint32_t WaveSize = Constants::AMDGCN_WAVE_SIZE;

                // Number of elements each thread will fetch in BlockDim direction
                static constexpr uint32_t DimPerThread = BlockDim / MmaDim;

                // Number of elements each thread will fetch in BlockK direction
                static constexpr uint32_t KPerThread = BlockK * MmaDim / (WaveSize * SplitK);

                // Number of elements that each thread is responsible for
                static constexpr uint32_t ElementsPerThread = DimPerThread * KPerThread;

                // Strides
                static constexpr uint32_t SplitKStride_X = 0u;
                static constexpr uint32_t SplitKStride_Y = BlockK / SplitK;

                static constexpr uint32_t BlockKStride_X = 1u;
                static constexpr uint32_t BlockKStride_Y = 0u;

                static constexpr uint32_t VWStride_X = 0u;
                static constexpr uint32_t VWStride_Y = KPerThread;

                // Stride Space
                static constexpr uint32_t SplitKSegs = BlockK / SplitKStride_Y;
                static constexpr uint32_t BlockKSegs = DimPerThread / BlockKStride_X;
                static constexpr uint32_t VWSegs     = KPerThread / VWStride_Y;

                // Check KPerThread validity
                static_assert(BlockK >= KPerThread, "Invalid KPerThread");
                static_assert(KPerThread > 0, "Invalid KPerThread - try a larger block size");
                static_assert(BlockK % KPerThread == 0, "BlockK is not a multiple of KPerThread");

                // Check SplitK validity
                static_assert(BlockK >= SplitK, "Invalid SplitK");
                static_assert(SplitK > 0, "Invalid SplitK");
                static_assert(BlockK % SplitK == 0, "BlockK is not a multiple of SplitK");

                // Check MmaDim validity
                static_assert(BlockDim >= MmaDim, "BlockDim must be larger than MmaDim");
                static_assert(BlockDim % MmaDim == 0, "BlockDim must be a multiple of MmaDim");
            };

            ROCWMMA_DEVICE constexpr static inline auto strideCounts()
            {
                return make_vector(Traits::SplitKSegs, // WaveKSegs Segments
                                   Traits::BlockKSegs, // BlockK Segments
                                   Traits::VWSegs); // VW Segments
            }

            ROCWMMA_DEVICE constexpr static inline auto strides()
            {
                return make_vector(make_coord2d(Traits::SplitKStride_X, Traits::SplitKStride_Y),
                                   make_coord2d(Traits::BlockKStride_X, Traits::BlockKStride_Y),
                                   make_coord2d(Traits::VWStride_X, Traits::VWStride_Y));
            }

            ROCWMMA_DEVICE static inline auto baseOffset()
            {
                return make_coord2d((threadIdx.x * Traits::DimPerThread) % BlockDim,
                                    (threadIdx.x / MmaDim * Traits::KPerThread)
                                        % (BlockK / SplitK));
            }
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct RowOrthoVW
            : public OrthoImpl<ColOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct RowInlineVW
            : public OrthoImpl<ColInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MmaDim, // Mma instruction size
                  uint32_t SplitK /*= 1*/> // # of splits
        struct RowOrthoInt : public OrthoImpl<ColOrthoInt<BlockDim, BlockK, DataT, MmaDim, SplitK>>
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MmaDim, // Mma instruction size
                  uint32_t SplitK /*= 1*/> // # of splits
        struct RowInlineInt
            : public OrthoImpl<ColInlineInt<BlockDim, BlockK, DataT, MmaDim, SplitK>>
        {
        };

    } // namespace MatrixLayout

} // namespace rocwmma

#if !defined(__HIPCC_RTC__)
namespace std
{

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth>
    inline ostream& operator<<(
        ostream& stream,
        rocwmma::MatrixLayout::
            ColOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth> const& matrix_layout)
    {
        return stream << "ColOrthoVW<" << BlockDim << ", " << BlockK << ", "
                      << rocwmma::dataTypeToString<DataT>() << ", " << VectorWidth << ", "
                      << MaxVectorWidth << ">";
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth>
    inline ostream& operator<<(
        ostream& stream,
        rocwmma::MatrixLayout::
            ColInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth> const& matrix_layout)
    {
        return stream << "ColInlineVW<" << BlockDim << ", " << BlockK << ", "
                      << rocwmma::dataTypeToString<DataT>() << ", " << VectorWidth << ", "
                      << MaxVectorWidth << ">";
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth>
    inline ostream& operator<<(
        ostream& stream,
        rocwmma::MatrixLayout::
            RowOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth> const& matrix_layout)
    {
        return stream << "RowOrthoVW<" << BlockDim << ", " << BlockK << ", "
                      << rocwmma::dataTypeToString<DataT>() << ", " << VectorWidth << ", "
                      << MaxVectorWidth << ">";
    }

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth>
    inline ostream& operator<<(
        ostream& stream,
        rocwmma::MatrixLayout::
            RowInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth> const& matrix_layout)
    {
        return stream << "RowInlineVW<" << BlockDim << ", " << BlockK << ", "
                      << rocwmma::dataTypeToString<DataT>() << ", " << VectorWidth << ", "
                      << MaxVectorWidth << ">";
    }

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT, uint32_t MmaDim, uint32_t SplitK>
    inline ostream& operator<<(
        ostream& stream,
        rocwmma::MatrixLayout::ColOrthoInt<BlockDim, BlockK, DataT, MmaDim, SplitK> const&
            matrix_layout)
    {
        return stream << "ColOrthoInt<" << BlockDim << ", " << BlockK << ", "
                      << rocwmma::dataTypeToString<DataT>() << ", " << MmaDim << ", " << SplitK
                      << ">";
    }

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT, uint32_t MmaDim, uint32_t SplitK>
    inline ostream& operator<<(
        ostream& stream,
        rocwmma::MatrixLayout::ColInlineInt<BlockDim, BlockK, DataT, MmaDim, SplitK> const&
            matrix_layout)
    {
        return stream << "ColInlineInt<" << BlockDim << ", " << BlockK << ", "
                      << rocwmma::dataTypeToString<DataT>() << ", " << MmaDim << ", " << SplitK
                      << ">";
    }

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT, uint32_t MmaDim, uint32_t SplitK>
    inline ostream& operator<<(
        ostream& stream,
        rocwmma::MatrixLayout::RowOrthoInt<BlockDim, BlockK, DataT, MmaDim, SplitK> const&
            matrix_layout)
    {
        return stream << "RowOrthoInt<" << BlockDim << ", " << BlockK << ", "
                      << rocwmma::dataTypeToString<DataT>() << ", " << MmaDim << ", " << SplitK
                      << ">";
    }

    template <uint32_t BlockDim, uint32_t BlockK, typename DataT, uint32_t MmaDim, uint32_t SplitK>
    inline ostream& operator<<(
        ostream& stream,
        rocwmma::MatrixLayout::RowInlineInt<BlockDim, BlockK, DataT, MmaDim, SplitK> const&
            matrix_layout)
    {
        return stream << "RowInlineInt<" << BlockDim << ", " << BlockK << ", "
                      << rocwmma::dataTypeToString<DataT>() << ", " << MmaDim << ", " << SplitK
                      << ">";
    }

} // namespace std

#endif // !defined(__HIPCC_RTC__)

#endif // ROCWMMA_MATRIX_LAYOUT_IMPL_HPP
