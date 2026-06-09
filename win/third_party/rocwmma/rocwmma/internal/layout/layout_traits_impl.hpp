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
#ifndef ROCWMMA_LAYOUT_TRAITS_IMPL_HPP
#define ROCWMMA_LAYOUT_TRAITS_IMPL_HPP

#include "../utility/type_traits.hpp"

namespace rocwmma
{
    namespace LayoutTraits_impl
    {
        // Classifier to test layout sameness
        template <typename LayoutLhs, typename LayoutRhs, typename Enabler = void>
        struct is_layout_same : public false_type
        {
        };

        // Classifer to test layout orthogonality
        template <typename LayoutLhs, typename LayoutRhs, typename Enabler = void>
        struct is_layout_orthogonal : public false_type
        {
        };

        // Orthogonality guide
        template <typename Layout>
        struct orthogonal_layout;

        // Meta traits for layouts
        template <typename Layout, typename Enabler = void>
        struct layout_traits;

    } // namespace LayoutTraits_impl

} // namespace rocwmma

#endif // ROCWMMA_LAYOUT_TRAITS_IMPL_HPP
