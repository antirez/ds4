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
#ifndef ROCWMMA_PERMUTE_HPP
#define ROCWMMA_PERMUTE_HPP

#include "cross_lane_ops.hpp"
#include "permute_impl.hpp"
#include "vector.hpp"

namespace rocwmma
{
    namespace Permute
    {

        template <typename PermuteOp>
        struct Driver
        {
            // Sanity checks
            static_assert((PermuteOp::opImpl() == CrossLaneOps::Properties::OP_IMPL_PERMUTE)
                              || (PermuteOp::opImpl()
                                  == CrossLaneOps::Properties::OP_IMPL_BPERMUTE),
                          "PermuteOp must use permute or permute backend");
            static_assert((PermuteOp::opId() == CrossLaneOps::Properties::OP_ID_BLOCK_BCAST)
                              || (PermuteOp::opId() == CrossLaneOps::Properties::OP_ID_SHUFFLE)
                              || (PermuteOp::opId() == CrossLaneOps::Properties::OP_ID_GATHER)
                              || (PermuteOp::opId() == CrossLaneOps::Properties::OP_ID_SCATTER)
                              || (PermuteOp::opId() == CrossLaneOps::Properties::OP_ID_ROTATE),
                          "PermuteOp is unsupported");
            template <typename DataT>
            ROCWMMA_DEVICE static inline auto exec(DataT const& src)
            {
                // Ensure that we can vectorize to B32
                static_assert(sizeof(DataT) % sizeof(uint32_t) == 0,
                              "DataT size must be a multiple of B32");

                // Vectorize to B32.
                // This way we can support B64+ types
                using B32VecT = VecT<uint32_t, sizeof(DataT) / sizeof(uint32_t)>;

                static_assert(sizeof(B32VecT) == sizeof(DataT), "Unable to vectorize DataT");

                // Forward to vectorized function
                auto result = exec(reinterpret_cast<B32VecT const&>(src));

                // Restore result to input type
                return reinterpret_cast<DataT&>(result);
            }

            template <typename DataT, uint32_t VecSize>
            ROCWMMA_DEVICE static inline auto exec(VecT<DataT, VecSize> const& src)
            {
                // Reinterpret vector as B32 so we can support B64+ elements.
                constexpr uint32_t B32VecSize = sizeof(DataT) / sizeof(uint32_t) * VecSize;

                // Ensure that DataT is a multiple of B32
                static_assert(B32VecSize >= 1, "DataT must be a multiple of B32");

                using B32VecT   = VecT<uint32_t, B32VecSize>;
                using InputVecT = VecT<DataT, VecSize>;

                // Ensure that we can vectorize to B32
                static_assert(sizeof(InputVecT) % sizeof(uint32_t) == 0,
                              "VecT size must be a multiple of B32");
                static_assert(sizeof(B32VecT) == sizeof(InputVecT),
                              "Unable to vectorize src0 to B32");

                auto op = [](auto&& idx, auto&& v0, auto&& opCtrl) {
                    // Pair up the b32 vector elements with the appropriate b32 scalar elements.
                    constexpr auto i = decay_t<decltype(idx)>::value;
                    return PermuteOp::exec(get<i>(v0), opCtrl);
                };

                // Give the current threadId to the threadCtrl modifier.
                // Then static unroll with cached modifier.
                auto result = vector_generator<uint32_t, B32VecSize>()(
                    op,
                    reinterpret_cast<B32VecT const&>(src),
                    PermuteOp::threadCtrl(detail::WaveSpace<>::localLaneId()));

                // Restore result to input type
                return reinterpret_cast<InputVecT&>(result);
            }
        };

        /*! \class BlockBCast32
        *  \brief  Permute class that broadcasts one block of 32 threads to all other blocks
        *  @tparam BlockIdx block index [0 - WaveSize/32]
        */
        template <uint32_t BlockIdx>
        using BlockBCast32 = Driver<PermuteImpl::Ops::BlockBCast32<BlockIdx>>;

        /*! \class BlockBCast16
        *  \brief  Permute class that broadcasts one block of 16 threads to all other blocks
        *  @tparam BlockIdx block index [0 - WaveSize/16]
        */
        template <uint32_t BlockIdx>
        using BlockBCast16 = Driver<PermuteImpl::Ops::BlockBCast16<BlockIdx>>;

        /*! \class BlockBCast8
        *  \brief  Permute class that broadcasts one block of 8 threads to all other blocks
        *  @tparam BlockIdx block index [0 - WaveSize/8]
        */
        template <uint32_t BlockIdx>
        using BlockBCast8 = Driver<PermuteImpl::Ops::BlockBCast8<BlockIdx>>;

        /*! \class BlockBCast4
        *  \brief  Permute class that broadcasts one block of 4 threads to all other blocks
        *  @tparam BlockIdx block index [0 - WaveSize/4]
        */
        template <uint32_t BlockIdx>
        using BlockBCast4 = Driver<PermuteImpl::Ops::BlockBCast4<BlockIdx>>;

        /*! \class BlockBCast2
        *  \brief  Permute class that broadcasts one block of 2 threads to all other blocks
        *  @tparam BlockIdx block index [0 - WaveSize/2]
        */
        template <uint32_t BlockIdx>
        using BlockBCast2 = Driver<PermuteImpl::Ops::BlockBCast2<BlockIdx>>;

        /*! \class GatherWave
        *  \brief  Permute class that pulls interleaved values based on VW in each group size.
        * Interleaved offsets are where each thread will read from.
        *  @tparam VW vector width [1, 2, 4, 8, 16]
        *  @tparam ElementShift rotation offset [0 - VW]
        */
        template <uint32_t VW, uint32_t ElementShift>
        using GatherWave = Driver<PermuteImpl::Ops::GatherWave<VW, ElementShift>>;

        /*! \class Gather32
        *  \brief  Permute class that pulls interleaved values based on VW in each group size of 32.
        * Interleaved offsets are where each thread will read from.
        *  @tparam VW vector width [1, 2, 4, 8, 16]
        *  @tparam ElementShift rotation offset [0 - VW]
        */
        template <uint32_t VW, uint32_t ElementShift>
        using Gather32 = Driver<PermuteImpl::Ops::Gather32<VW, ElementShift>>;

        /*! \class Gather16
        *  \brief  Permute class that pulls interleaved values based on VW in each group size of 16.
        * Interleaved offsets are where each thread will read from.
        *  @tparam VW vector width [1, 2, 4, 8, 16]
        *  @tparam ElementShift rotation offset [0 - VW]
        */
        template <uint32_t VW, uint32_t ElementShift>
        using Gather16 = Driver<PermuteImpl::Ops::Gather16<VW, ElementShift>>;

        /*! \class ScatterWave
        *  \brief  Permute class that pushes interleaved values based on VW in each group size.
        * Interleaved offsets are where each thread will write to.
        *  @tparam VW vector width [1, 2, 4, 8, 16]
        *  @tparam ElementShift rotation offset [0 - VW]
        */
        template <uint32_t VW, uint32_t ElementShift>
        using ScatterWave = Driver<PermuteImpl::Ops::ScatterWave<VW, ElementShift>>;

        /*! \class Scatter32
        *  \brief  Permute class that pushes interleaved values based on VW in each group size of 32.
        * Interleaved offsets are where each thread will write to.
        *  @tparam VW vector width [1, 2, 4, 8, 16]
        *  @tparam ElementShift rotation offset [0 - VW]
        */
        template <uint32_t VW, uint32_t ElementShift>
        using Scatter32 = Driver<PermuteImpl::Ops::Scatter32<VW, ElementShift>>;

        /*! \class Scatter16
        *  \brief  Permute class that pushes interleaved values based on VW in each group size of 16.
        * Interleaved offsets are where each thread will write to.
        *  @tparam VW vector width [1, 2, 4, 8, 16]
        *  @tparam ElementShift rotation offset [0 - VW]
        */
        template <uint32_t VW, uint32_t ElementShift>
        using Scatter16 = Driver<PermuteImpl::Ops::Scatter16<VW, ElementShift>>;

        /*! \class RotateWaveL
        *  \brief  Swizzle class that rotates all threads to the left
        *  @tparam RotateDistance thread index [0 - WaveSize-1]
        */
        template <uint32_t RotateDistance>
        using RotateWaveL = Driver<PermuteImpl::Ops::RotateWaveL<RotateDistance>>;

        /*! \class RotateWaveR
        *  \brief  Swizzle class that rotates all threads to the right
        *  @tparam RotateDistance thread index [0 - WaveSize-1]
        */
        template <uint32_t RotateDistance>
        using RotateWaveR = Driver<PermuteImpl::Ops::RotateWaveR<RotateDistance>>;

    } // namespace Permute

} // namespace rocwmma

#endif // ROCWMMA_PERMUTE_HPP
