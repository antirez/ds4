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
#ifndef ROCWMMA_LAYOUT_TRAITS_HPP
#define ROCWMMA_LAYOUT_TRAITS_HPP

// Need strict inclusion order here
// clang-format off
#include "layout_traits_impl.hpp"
#include "data_layout_traits_impl.hpp"
#include "matrix_layout_traits_impl.hpp"
#include "register_layout_traits_impl.hpp"
// Must stay last
#include "matrix_coop_layout_traits_impl.hpp"
// clang-format on

namespace rocwmma
{
    /*! \class is_layout_same
    *  \brief  Compares layout types are the same, or are equivalent.
    * Applicable to layout contexts: DataLayout, MatrixLayout and RegisterLayout.
    * DataLayouts are same if they have the same 1D layout in memory.
    * MatrixLayouts are the same if they have the same 2D matrix layout in memory.
    * RegisterLayouts are the same if they have the same thread mapping in register.
    *  @tparam LhsLayout Comparative left hand side
    *  @tparam RhsLayout Comparative right hand side
    */
    template <typename LhsLayout, typename RhsLayout>
    struct is_layout_same : public LayoutTraits_impl::is_layout_same<LhsLayout, RhsLayout>
    {
    };

    /*! \class is_layout_same_v
    *  \brief  Evaluates is_layout_same
    *  @tparam LhsLayout Comparative left hand side
    *  @tparam RhsLayout Comparative right hand side
    */
    template <typename LhsLayout, typename RhsLayout>
    constexpr static inline bool is_layout_same_v = is_layout_same<LhsLayout, RhsLayout>::value;

    /*! \class is_layout_orthogonal
    *  \brief  Describes a relationship between two layout endpoints. They are considered orthogonal if they
    * are not the same, and there exists a reversible transformation path from one to the other.
    * Applicable to layout contexts: DataLayout, MatrixLayout and RegisterLayout
    * DataLayouts are orthogonal if their 1D layouts in memory are transformable (e.g., row major vs col major).
    * MatrixLayouts are orthogonal if their 2D matrix layout geometry is transformable (e.g., layout transpose).
    * RegisterLayouts are orthogonal if their in-register layouts are transformable (e.g., AOS vs SOA)
    *  @tparam LhsLayout Comparative left hand side
    *  @tparam RhsLayout Comparative right hand side
    */
    template <typename LhsLayout, typename RhsLayout>
    struct is_layout_orthogonal
        : public LayoutTraits_impl::is_layout_orthogonal<LhsLayout, RhsLayout>
    {
    };

    /*! \class is_layout_orthogonal
    *  \brief  Evaluates is_layout_orthogonal
    *  @tparam LhsLayout Comparative left hand side
    *  @tparam RhsLayout Comparative right hand side
    */
    template <typename LhsLayout, typename RhsLayout>
    constexpr static inline bool is_layout_orthogonal_v
        = is_layout_orthogonal<LhsLayout, RhsLayout>::value;

    /*! \class orthogonal_layout
    *  \brief  Provides a guide to an orthogonal layout of the source layout.
    * Applicable to layout contexts: DataLayout, MatrixLayout and RegisterLayout
    *  @tparam Layout the source layout
    */
    template <typename Layout>
    struct orthogonal_layout : public LayoutTraits_impl::orthogonal_layout<Layout>
    {
    };

    /*! \class layout_transpose_t
    *  \brief  Transforms the layout type into its orthogonal layout.
    *  @tparam Layout the source layout
    */
    template <typename Layout>
    using orthogonal_layout_t = typename orthogonal_layout<Layout>::type;

    template <typename Layout>
    struct layout_traits : public LayoutTraits_impl::layout_traits<Layout>
    {
    };

} // namespace rocwmma

#endif // ROCWMMA_LAYOUT_TRAITS_HPP
