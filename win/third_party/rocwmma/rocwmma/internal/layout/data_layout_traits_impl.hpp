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
#ifndef ROCWMMA_DATA_LAYOUT_TRAITS_IMPL_HPP
#define ROCWMMA_DATA_LAYOUT_TRAITS_IMPL_HPP

#include "layout.hpp"
#include "layout_traits.hpp"

namespace rocwmma
{
    namespace LayoutTraits_impl
    {
        // Reference regular layouts
        using DataLayout::ColMajor;
        using DataLayout::RowMajor;

        // Build a basic set of meta-data classifiers.
        // We will be interested in knowing things about our data layouts:
        // - is_row_major
        // - is_col_major
        // - is_data_layout
        //
        // Note: We will qualify both:
        // row_major / col_major (as meta-tags)
        // RowMajor and ColMajor (as functional classes)
        template <typename DataLayout>
        struct is_row_major : public false_type
        {
        };

        template <>
        struct is_row_major<row_major> : public true_type
        {
        };

        template <>
        struct is_row_major<DataLayout::RowMajor> : public true_type
        {
        };

        template <typename DataLayout>
        struct is_col_major : public false_type
        {
        };

        template <>
        struct is_col_major<col_major> : public true_type
        {
        };

        template <>
        struct is_col_major<DataLayout::ColMajor> : public true_type
        {
        };

        // Convenience evaluators
        template <typename DataLayout>
        static constexpr bool is_row_major_v = is_row_major<DataLayout>::value;

        template <typename DataLayout>
        static constexpr bool is_col_major_v = is_col_major<DataLayout>::value;

        template <typename DataLayout>
        struct is_data_layout
            : public integral_constant<bool,
                                       is_row_major_v<DataLayout> || is_col_major_v<DataLayout>>
        {
        };

        // Convenience evaluator
        template <typename DataLayout>
        static constexpr bool is_data_layout_v = is_data_layout<DataLayout>::value;

        // Cumulative traits about our data layouts
        template <typename DataLayout>
        struct data_layout_traits
        {
            static constexpr bool is_row_major   = is_row_major_v<DataLayout>;
            static constexpr bool is_col_major   = is_col_major_v<DataLayout>;
            static constexpr bool is_data_layout = is_data_layout_v<DataLayout>;
        };

// Tidy some traits accesses
#define traits_lhs data_layout_traits<DataLayoutLhs>
#define traits_rhs data_layout_traits<DataLayoutRhs>

        template <typename DataLayoutLhs, typename DataLayoutRhs>
        ROCWMMA_HOST_DEVICE constexpr static bool testDataLayoutSame()
        {
            return (traits_lhs::is_row_major && traits_rhs::is_row_major)
                   || (traits_lhs::is_col_major && traits_rhs::is_col_major);
        }

        template <typename DataLayoutLhs, typename DataLayoutRhs>
        ROCWMMA_HOST_DEVICE constexpr static bool testDataLayoutOrthogonal()
        {
            return (traits_lhs::is_row_major && traits_rhs::is_col_major)
                   || (traits_lhs::is_col_major && traits_rhs::is_row_major);
        }

        // Implement sameness classifier for data layouts
        template <typename DataLayoutLhs, typename DataLayoutRhs>
        struct is_layout_same<DataLayoutLhs,
                              DataLayoutRhs,
                              enable_if_t<traits_lhs::is_data_layout && traits_rhs::is_data_layout>>
            : public integral_constant<bool, testDataLayoutSame<DataLayoutLhs, DataLayoutRhs>()>
        {
        };

        // Implement orthogonality classifier for data layouts
        template <typename DataLayoutLhs, typename DataLayoutRhs>
        struct is_layout_orthogonal<
            DataLayoutLhs,
            DataLayoutRhs,
            enable_if_t<traits_lhs::is_data_layout && traits_rhs::is_data_layout>>
            : public integral_constant<bool,
                                       testDataLayoutOrthogonal<DataLayoutLhs, DataLayoutRhs>()>
        {
        };

#undef traits_lhs
#undef traits_rhs

        // Orthogonal layout guides
        template <>
        struct orthogonal_layout<row_major>
        {
            using type = col_major;
        };

        template <>
        struct orthogonal_layout<col_major>
        {
            using type = row_major;
        };

        template <typename DataLayoutT>
        struct orthogonal_layout<DataLayout::template Array1d<DataLayoutT>>
        {
            using type
                = DataLayout::template Array1d<typename orthogonal_layout<DataLayoutT>::type>;
        };

        template <typename DataLayout>
        struct layout_traits<DataLayout,
                             enable_if_t<data_layout_traits<DataLayout>::is_data_layout>>
            : public data_layout_traits<DataLayout>
        {
        };

    } // namespace LayoutTraits_impl

} // namespace rocwmma

#if !defined(__HIPCC_RTC__)
namespace std
{

    template <typename DataLayout>
    inline ostream&
        operator<<(ostream&                                                          stream,
                   rocwmma::LayoutTraits_impl::data_layout_traits<DataLayout> const& traits)
    {
        using data_traits = decay_t<decltype(traits)>;
        stream << "DataLayout Traits: " << DataLayout{} << std::endl;
        stream << "is_row_major: " << data_traits::is_row_major << std::endl;
        stream << "is_col_major: " << data_traits::is_col_major << std::endl;
        stream << "is_data_layout: " << data_traits::is_data_layout << std::endl;
        return stream;
    }

} // namespace std

#endif // !defined(__HIPCC_RTC__)

#endif // ROCWMMA_DATA_LAYOUT_TRAITS_IMPL_HPP
