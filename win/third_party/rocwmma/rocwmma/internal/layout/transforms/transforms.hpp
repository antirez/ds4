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
#ifndef ROCWMMA_LAYOUT_TRANSFORMS_HPP
#define ROCWMMA_LAYOUT_TRANSFORMS_HPP

#include "../../config.hpp"

namespace rocwmma
{
    namespace Transforms
    {
        // Transforms for non-interleaved layouts
        template<uint32_t BlockDim, uint32_t MaxVectorWidth, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) soa_to_aos(VecT&& v);

        template<uint32_t BlockDim, uint32_t MaxVectorWidth, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) aos_to_soa(VecT&& v);

        // Transforms for interleaved layouts
        template <uint32_t KPerThread, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) soa_int_to_aos_int(VecT&& v);

        template <uint32_t DimPerThread, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) aos_int_to_soa_int(VecT&& v);

        template <uint32_t AccVecSize, uint32_t MmaBlocksA, uint32_t AccMaxVW, uint32_t MmaDim, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) soa_int_to_mma_acc_int_a_major(VecT&& v);

        template <uint32_t AccVecSize, uint32_t MmaBlocksB, uint32_t AccMaxVW, uint32_t MmaDim, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) mma_acc_int_a_major_to_soa_int(VecT&& v);

        template <uint32_t AccVecSize, uint32_t MmaBlocksA, uint32_t MmaBlocksB, uint32_t AccMaxVW, uint32_t MmaDim, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) aos_int_to_mma_acc_int_a_major(VecT&& v);

        template <uint32_t AccVecSize, uint32_t AccMaxVW, uint32_t MmaDim, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) mma_acc_int_a_major_to_aos_int(VecT&& v);

        // Transforms for wmma layouts (gfx11)
        template<typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) to_wmma_input_gfx11(VecT&& v);

        template<typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) from_wmma_input_gfx11(VecT&& v);

        template<typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) to_wmma_acc_gfx11(VecT&& v);

        template<typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) from_wmma_acc_gfx11(VecT&& v);

        template <typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) soa_to_wmma_input_gfx11(VecT&& v);

        template<uint32_t BlockDim, uint32_t MaxVectorWidth, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) aos_to_wmma_input_gfx11(VecT&& v);

        template <typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) wmma_input_gfx11_to_soa(VecT&& v);

        template<uint32_t BlockDim, uint32_t MaxVectorWidth, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) wmma_input_gfx11_to_aos(VecT&& v);

        template <typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) soa_to_wmma_acc_gfx11(VecT&& v);

        template<uint32_t BlockDim, uint32_t MaxVectorWidth, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) aos_to_wmma_acc_gfx11(VecT&& v);

        template <typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) wmma_acc_gfx11_to_soa(VecT&& v);

        template<uint32_t BlockDim, uint32_t MaxVectorWidth, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) wmma_acc_gfx11_to_aos(VecT&& v);

        // Transforms for interleaved wmma layouts (gfx11)
        template <uint32_t InputBlockSize, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) soa_int_to_wmma_input_gfx11(VecT&& v);

        template <uint32_t DimPerThread, uint32_t InputBlockSize, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) aos_int_to_wmma_input_gfx11(VecT&& v);

        template <uint32_t InputBlockSize, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) wmma_input_gfx11_to_soa_int(VecT&& v);

        template <uint32_t KPerThread, uint32_t InputBlockSize, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) wmma_input_gfx11_to_aos_int(VecT&& v);

        template <typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) mma_acc_int_a_major_to_wmma_acc_gfx11(VecT&& v);

        template <typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) wmma_acc_gfx11_to_mma_acc_int_a_major(VecT&& v);

        template <uint32_t AccVecSize, uint32_t MmaBlocksA, uint32_t AccMaxVW, uint32_t MmaDim, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) soa_int_to_wmma_acc_gfx11(VecT&& v);

        template <uint32_t AccVecSize, uint32_t MmaBlocksA, uint32_t MmaBlocksB, uint32_t AccMaxVW, uint32_t MmaDim, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) aos_int_to_wmma_acc_gfx11(VecT&& v);

        template <uint32_t AccVecSize, uint32_t MmaBlocksB, uint32_t AccMaxVW, uint32_t MmaDim, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) wmma_acc_gfx11_to_soa_int(VecT&& v);

        template <uint32_t AccVecSize, uint32_t AccMaxVW, uint32_t MmaDim, typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) wmma_acc_gfx11_to_aos_int(VecT&& v);

    } // namespace Transforms

} // namespace rocwmma

#include "transforms_impl.hpp"
#include "transforms_int_impl.hpp"
#include "transforms_int_wmma_impl.hpp"
#include "transforms_wmma_impl.hpp"

#endif // ROCWMMA_LAYOUT_TRANSFORMS_HPP
