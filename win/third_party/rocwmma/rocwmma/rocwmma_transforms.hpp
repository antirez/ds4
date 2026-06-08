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
#ifndef ROCWMMA_TRANSFORMS_API_HPP
#define ROCWMMA_TRANSFORMS_API_HPP

#include "rocwmma.hpp"
#include "rocwmma_transforms_impl.hpp"

namespace rocwmma
{
    //! Static fragment type transform that applies a transpose transform
    //! @note To be paired with usage of apply_transpose() below
    template <typename FragT>
    using apply_transpose_t = typename detail::template ApplyTranspose<FragT>::Type;

    //! Static fragment type transform that applies a data layout change
    //! @note To be paired with usage of apply_data_layout() below
    template <typename FragT, typename NewDataLayoutT>
    using apply_data_layout_t =
        typename detail::template ApplyDataLayout<FragT, NewDataLayoutT>::Type;

    //! Static fragment transform that reinterprets the fragment as a register file
    //! E.g. BlockDim = WaveSize, BlockK = Vector Size
    //! @note Unsafe for use with cooperative fragments
    template <typename FragT>
    using apply_register_file_t = typename detail::template ApplyRegisterFile<FragT>::Type;

    //! Applies the transpose transform the input fragment. Transpose is defined as orthogonal matrix and data layout.
    //! E.g. T(fragment<matrix_a, BlockM, BlockN, BlockK, DataT, row_major>) = fragment<matrix_b, BlockN, BlockM, BlockK, DataT, col_major>
    //! @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
    //! @tparam FragT The incoming fragment type
    //! @returns Transposed (orthogonal) fragment
    template <typename FragT>
    ROCWMMA_DEVICE static inline decltype(auto) apply_transpose(FragT&& frag);

    //! Transforms the input fragment to have the desired data layout.
    //! @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
    //! @tparam DataLayoutT The desired fragment data layout to apply
    //! @tparam FragT The incoming fragment type
    //! @returns Fragment with transformed data layout
    template <typename DataLayoutT, typename FragT>
    ROCWMMA_DEVICE static inline decltype(auto) apply_data_layout(FragT&& frag);

    //! Transforms the input fragment to the target fragment type. This could include changing matrix context
    //! and/or changing data layout, as long as there is a path from the source register layout to the
    //! destination register layout.
    //! @param frag Source fragment of type MatrixT with its associated block sizes, data type and layout
    //! @tparam DstFragT The target fragment type to transform to
    //! @tparam FragT The source incoming fragment type
    //! @returns Target fragment after transformation
    template <typename DstFragT, typename FragT>
    ROCWMMA_DEVICE static inline decltype(auto) apply_fragment(FragT&& frag);

    //! Transforms the input fragment to a "register file" fragment type. Register contents are directly mapped
    //! to a 2D matrix space represented by [RegCount x WaveSize]. This transform is a geometry reinterpretation.
    //! @param frag Source fragment of type MatrixT with its associated block sizes, data type and layout
    //! @tparam FragT The source incoming fragment type
    //! @returns Target fragment after transformation
    template <typename FragT>
    ROCWMMA_DEVICE static inline decltype(auto) to_register_file(FragT&& frag);

    //! Transforms the "register file" fragment type to a target fragment type. Register contents are directly mapped
    //! to a 2D matrix space represented by [RegCount x WaveSize]. This transform is a geometry reinterpretation.
    //! @param frag Source fragment of type MatrixT with its associated block sizes, data type and layout
    //! @tparam DstFragT The target frag to transform to
    //! @tparam FragT The source incoming fragment type as register file
    //! @returns Fragment after transformation
    template <typename DstFragT, typename FragT>
    ROCWMMA_DEVICE static inline decltype(auto) from_register_file(FragT&& frag);

} // namespace rocwmma

#endif // ROCWMMA_TRANSFORMS_API_HPP
