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
#ifndef ROCWMMA_MATRIX_COOP_LAYOUT_TRAITS_IMPL_HPP
#define ROCWMMA_MATRIX_COOP_LAYOUT_TRAITS_IMPL_HPP

#include "layout_traits.hpp"

namespace rocwmma
{
    namespace MatrixLayout
    {
        // Fwd declare MatrixCoopLayout
        template <typename MatrixLayout, uint32_t WaveCount>
        struct MatrixCoopLayout;
    }

    // Traits overloads for matrix and register layouts
    namespace LayoutTraits_impl
    {
        using MatrixLayout::MatrixCoopLayout;
        using RegisterLayout::Storage;

        template <typename MatrixLayout, uint32_t WaveCountIn>
        struct matrix_layout_traits<MatrixCoopLayout<MatrixLayout, WaveCountIn>>
            : public matrix_layout_traits<MatrixLayout>
        {
            static constexpr uint32_t WaveCount = WaveCountIn;
        };

        template <typename MatrixLayout, typename DataLayout, uint32_t WaveCountIn>
        struct register_layout_traits<
            Storage<MatrixCoopLayout<MatrixLayout, WaveCountIn>, DataLayout>>
            : public register_layout_traits<Storage<MatrixLayout, DataLayout>>
        {
            static constexpr uint32_t WaveCount = WaveCountIn;
        };

        template <typename MatrixLayout, uint32_t WaveCount>
        struct orthogonal_layout<MatrixCoopLayout<MatrixLayout, WaveCount>>
        {
            using type
                = MatrixCoopLayout<typename orthogonal_layout<MatrixLayout>::type, WaveCount>;
        };

        template <typename MatrixLayout, typename DataLayout, uint32_t WaveCount>
        struct orthogonal_layout<Storage<MatrixCoopLayout<MatrixLayout, WaveCount>, DataLayout>>
        {
            using type = Storage<
                MatrixCoopLayout<typename orthogonal_layout<MatrixLayout>::type, WaveCount>,
                DataLayout>;
        };

    } // namespace LayoutTraits_impl

} // namespace rocwmma

#endif // ROCWMMA_MATRIX_COOP_LAYOUT_TRAITS_IMPL_HPP
