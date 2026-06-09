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
#ifndef ROCWMMA_MATRIX_LAYOUT_TRAITS_IMPL_HPP
#define ROCWMMA_MATRIX_LAYOUT_TRAITS_IMPL_HPP

#include "../config.hpp"
#include "layout.hpp"
#include "layout_traits.hpp"

namespace rocwmma
{
    // Common helpers for supported traits
    namespace LayoutTraits_impl
    {
        // Reference regular layouts
        using MatrixLayout::ColInlineVW;
        using MatrixLayout::ColOrthoVW;
        using MatrixLayout::RowInlineVW;
        using MatrixLayout::RowOrthoVW;

        // Reference interleaved layouts
        using MatrixLayout::ColInlineInt;
        using MatrixLayout::ColOrthoInt;
        using MatrixLayout::RowInlineInt;
        using MatrixLayout::RowOrthoInt;

        // Build a basic set of meta-data classifiers.
        // We will be interested in knowing things about our matrix layouts:
        // - is_col_ortho
        // - is_row_ortho
        // - is_col_inline
        // - is_row_inline
        // - is_interleaved
        // - is_matrix_layout
        template <typename MatrixLayout>
        struct is_col_ortho : public false_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct is_col_ortho<ColOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
            : public true_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MfmaDim,
                  uint32_t SplitK>
        struct is_col_ortho<ColOrthoInt<BlockDim, BlockK, DataT, MfmaDim, SplitK>>
            : public true_type
        {
        };

        template <typename MatrixLayout>
        struct is_row_ortho : public false_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct is_row_ortho<RowOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
            : public true_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MfmaDim,
                  uint32_t SplitK>
        struct is_row_ortho<RowOrthoInt<BlockDim, BlockK, DataT, MfmaDim, SplitK>>
            : public true_type
        {
        };

        template <typename MatrixLayout>
        struct is_col_inline : public false_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct is_col_inline<ColInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
            : public true_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MfmaDim,
                  uint32_t SplitK>
        struct is_col_inline<ColInlineInt<BlockDim, BlockK, DataT, MfmaDim, SplitK>>
            : public true_type
        {
        };

        template <typename MatrixLayout>
        struct is_row_inline : public false_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct is_row_inline<RowInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
            : public true_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MfmaDim,
                  uint32_t SplitK>
        struct is_row_inline<RowInlineInt<BlockDim, BlockK, DataT, MfmaDim, SplitK>>
            : public true_type
        {
        };

        template <typename MatrixLayout>
        struct is_interleaved : public false_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MfmaDim,
                  uint32_t SplitK>
        struct is_interleaved<ColOrthoInt<BlockDim, BlockK, DataT, MfmaDim, SplitK>>
            : public true_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MfmaDim,
                  uint32_t SplitK>
        struct is_interleaved<RowOrthoInt<BlockDim, BlockK, DataT, MfmaDim, SplitK>>
            : public true_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MfmaDim,
                  uint32_t SplitK>
        struct is_interleaved<ColInlineInt<BlockDim, BlockK, DataT, MfmaDim, SplitK>>
            : public true_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MfmaDim,
                  uint32_t SplitK>
        struct is_interleaved<RowInlineInt<BlockDim, BlockK, DataT, MfmaDim, SplitK>>
            : public true_type
        {
        };

        // Convenience evaluators
        template <typename MatrixLayout>
        constexpr static bool is_col_ortho_v = is_col_ortho<MatrixLayout>::value;

        template <typename MatrixLayout>
        constexpr static bool is_row_ortho_v = is_row_ortho<MatrixLayout>::value;

        template <typename MatrixLayout>
        constexpr static bool is_col_inline_v = is_col_inline<MatrixLayout>::value;

        template <typename MatrixLayout>
        constexpr static bool is_row_inline_v = is_row_inline<MatrixLayout>::value;

        template <typename MatrixLayout>
        constexpr static bool is_interleaved_v = is_interleaved<MatrixLayout>::value;

        template <typename MatrixLayout>
        struct is_matrix_layout
            : public integral_constant<
                  bool,
                  is_col_ortho_v<
                      MatrixLayout> || is_col_inline_v<MatrixLayout> || is_row_ortho_v<MatrixLayout> || is_row_inline_v<MatrixLayout>>
        {
        };

        template <typename MatrixLayout>
        constexpr static bool is_matrix_layout_v = is_matrix_layout<MatrixLayout>::value;

        template <typename MatrixLayout>
        struct matrix_layout_classifier_traits
        {
            // Add associative traits
            constexpr static bool is_col_ortho     = is_col_ortho_v<MatrixLayout>;
            constexpr static bool is_col_inline    = is_col_inline_v<MatrixLayout>;
            constexpr static bool is_row_ortho     = is_row_ortho_v<MatrixLayout>;
            constexpr static bool is_row_inline    = is_row_inline_v<MatrixLayout>;
            constexpr static bool is_interleaved   = is_interleaved_v<MatrixLayout>;
            constexpr static bool is_matrix_layout = is_matrix_layout_v<MatrixLayout>;
        };

        template <typename MatrixLayout, typename Enabler = void>
        struct matrix_layout_derived_traits
        {
            // Interface for params we want to derive from matrix layouts
            constexpr static uint32_t BlockDim       = 0u;
            constexpr static uint32_t KDim           = 0u;
            using DataT                              = void;
            constexpr static uint32_t VectorWidth    = 0u;
            constexpr static uint32_t MaxVectorWidth = 0u;
            constexpr static uint32_t MmaDim         = 0u;
            constexpr static uint32_t SplitK         = 0u;
        };

#define matrix_layout \
    MatrixLayout<LayoutBlockDim, LayoutBlockK, LayoutDataT, LayoutVectorWidth, LayoutMaxVectorWidth>

        // Combine internal layout traits with template params
        template <uint32_t LayoutBlockDim,
                  uint32_t LayoutBlockK,
                  typename LayoutDataT,
                  uint32_t LayoutVectorWidth,
                  uint32_t LayoutMaxVectorWidth,
                  template <uint32_t, uint32_t, typename, uint32_t, uint32_t>
                  class MatrixLayout>
        struct matrix_layout_derived_traits<
            matrix_layout,
            enable_if_t<is_matrix_layout_v<matrix_layout> && !is_interleaved_v<matrix_layout>>>
            : public matrix_layout::Traits // Base traits
        {
            // Common params derived from template params
            constexpr static uint32_t BlockDim       = LayoutBlockDim;
            constexpr static uint32_t KDim           = LayoutBlockK;
            using DataT                              = LayoutDataT;
            constexpr static uint32_t VectorWidth    = LayoutVectorWidth;
            constexpr static uint32_t MaxVectorWidth = LayoutMaxVectorWidth;
            constexpr static uint32_t MmaDim         = LayoutBlockDim; // Effective MmaDim
            constexpr static uint32_t SplitK         = 0; // Unused
        };

#undef matrix_layout

#define matrix_layout \
    MatrixLayout<LayoutBlockDim, LayoutBlockK, LayoutDataT, LayoutMmaDim, LayoutSplitK>

        // Represent interleaved MatrixLayout instances
        template <uint32_t LayoutBlockDim,
                  uint32_t LayoutBlockK,
                  typename LayoutDataT,
                  uint32_t LayoutMmaDim,
                  uint32_t LayoutSplitK,
                  template <uint32_t, uint32_t, typename, uint32_t, uint32_t>
                  class MatrixLayout>
        struct matrix_layout_derived_traits<
            matrix_layout,
            enable_if_t<is_matrix_layout_v<matrix_layout> && is_interleaved_v<matrix_layout>>>
            : public matrix_layout::Traits // Base traits
        {
        private:
            // Wrapper to get fixed MaxVectorWidth / VectorWidth from layout
            constexpr static inline uint32_t calcMaxVw()
            {
                if constexpr(is_col_inline_v<matrix_layout> || is_row_inline_v<matrix_layout>)
                {
                    return matrix_layout::Traits::DimPerThread;
                }
                else if constexpr(is_col_ortho_v<matrix_layout> || is_row_ortho_v<matrix_layout>)
                {
                    return matrix_layout::Traits::KPerThread;
                }
                else
                {
                    return 0;
                }
            }

        public:
            // Common params derived from template params
            constexpr static uint32_t BlockDim       = LayoutBlockDim;
            constexpr static uint32_t KDim           = LayoutBlockK;
            using DataT                              = LayoutDataT;
            constexpr static uint32_t VectorWidth    = calcMaxVw();
            constexpr static uint32_t MaxVectorWidth = calcMaxVw();
            constexpr static uint32_t MmaDim         = LayoutMmaDim;
            constexpr static uint32_t SplitK         = LayoutSplitK;
        };

#undef matrix_layout

        // Combine base instance traits with specific layout classifiers
        template <typename MatrixLayout>
        struct matrix_layout_traits : public matrix_layout_derived_traits<MatrixLayout>,
                                      public matrix_layout_classifier_traits<MatrixLayout>
        {
        };

        // NOTE: MatrixLayout assumptions
        // When determining MatrixLayout traits, there are several strong assumptions.
        // 1. Regarding same-ness: MatrixLayouts must match, as defined below:
        //  ____________________________________________________________________
        // | MatrixLayoutLhs | MatrixLayoutRhs | Compatibility test:            |
        // |                 |     (Same)      | Required Fixed Params          |
        // | ------------------------------------------------------------------ |
        // | ColOrthoVW      | ColOrthoVW      | BlockDim, KDim, MaxVectorWidth |
        // | ColInlineVW     | ColInlineVW     | BlockDim, KDim, MaxVectorWidth |
        // | RowOrthoVW      | RowOrthoVW      | BlockDim, KDim, MaxVectorWidth |
        // | RowInlineVW     | RowInlineVW     | BlockDim, KDim, MaxVectorWidth |
        // | ------------------------------------------------------------------ |
        // | ColOrthoInt     | ColOrthoInt     | BlockDim, KDim, MmaDim, SplitK |
        // | ColInlineInt    | ColInlineInt    | BlockDim, KDim, MmaDim, SplitK |
        // | RowOrthoInt     | RowOrthoInt     | BlockDim, KDim, MmaDim, SplitK |
        // | RowInlineInt    | RowInlineInt    | BlockDim, KDim, MmaDim, SplitK |
        //  --------------------------------------------------------------------
        //
        // 2. Regarding orthogonality: for all Col* layouts, their Row*
        // orthogonal counterparts are implemented by row / col coordinate swaps.
        // This is valid as long as we have some fixed parameters, as defined below:
        //  ____________________________________________________________________
        // | MatrixLayoutLhs | MatrixLayoutRhs | Compatibility test:            |
        // |                 |   (Orthogonal)  | Required Fixed Params          |
        // | ------------------------------------------------------------------ |
        // | ColOrthoVW      | RowOrthoVW      | BlockDim, KDim, MaxVectorWidth |
        // | ColInlineVW     | RowInlineVW     | BlockDim, KDim, MaxVectorWidth |
        // | RowOrthoVW      | ColOrthoVW      | BlockDim, KDim, MaxVectorWidth |
        // | RowInlineVW     | ColInlineVW     | BlockDim, KDim, MaxVectorWidth |
        // | ------------------------------------------------------------------ |
        // | ColOrthoInt     | RowOrthoInt     | BlockDim, KDim, MmaDim, SplitK |
        // | ColInlineInt    | RowInlineInt    | BlockDim, KDim, MmaDim, SplitK |
        // | RowOrthoInt     | ColOrthoInt     | BlockDim, KDim, MmaDim, SplitK |
        // | RowInlineInt    | ColInlineInt    | BlockDim, KDim, MmaDim, SplitK |
        //  --------------------------------------------------------------------
        // This defines the need for MatrixLayout classifiers based upon:
        // - ColOrtho / RowOrtho
        // - ColInline / RowInline
        // - Non-interleave / non-interleaved
        //
        // Following the above traits, we can build more complicated traits such as
        // is_same, is_orthogonal and orthogonal_layout.

// Tidy access to matrix layout traits.
#define traits_lhs matrix_layout_traits<MatrixLayoutLhs>
#define traits_rhs matrix_layout_traits<MatrixLayoutRhs>

        // For a fixed maxVW, we can change VW of a matrix layout to any common divisor
        ROCWMMA_HOST_DEVICE constexpr static inline bool
            testSupportedVW(uint32_t maxVW, uint32_t vw0, uint32_t vw1)
        {
            return (vw0 <= maxVW) && (vw1 <= maxVW) && (maxVW % vw0 == 0) && (maxVW % vw1 == 0);
        }

        // As a predicate to is_layout_same or is_layout_orthogonal, their matrix parameters must
        // be compatible (see above table).
        template <typename MatrixLayoutLhs, typename MatrixLayoutRhs>
        ROCWMMA_HOST_DEVICE constexpr static bool testCompatibleMatrixParams()
        {
            if constexpr(!traits_lhs::is_matrix_layout && !traits_rhs::is_matrix_layout)
            {
                return false;
            }
            else if constexpr(!traits_lhs::is_interleaved && !traits_rhs::is_interleaved)
            {
                // Non-interleaved matrix layout compatibility requires:
                // 1. Fixed: BlockDim, KDim, MaxVectorWidth
                // 2. VectorWidths must satisfy criterion in testSupportedVW().
                return (traits_lhs::BlockDim == traits_rhs::BlockDim)
                       && (traits_lhs::KDim == traits_rhs::KDim)
                       && (traits_lhs::MaxVectorWidth == traits_rhs::MaxVectorWidth)
                       && (testSupportedVW(traits_lhs::MaxVectorWidth,
                                           traits_lhs::VectorWidth,
                                           traits_rhs::VectorWidth));
            }
            else if constexpr(traits_lhs::is_interleaved && traits_rhs::is_interleaved)
            {
                // Interleaved matrix layout compatibility requires:
                // 1. Must have fixed BlockDim, BlockK, MmaDim, SplitK
                // 2. MmaDim values must satisfy criterion in testSupportedMmaDim().
                return (traits_lhs::BlockDim == traits_rhs::BlockDim)
                       && (traits_lhs::KDim == traits_rhs::KDim)
                       && (traits_lhs::MmaDim == traits_rhs::MmaDim)
                       && (traits_lhs::SplitK == traits_rhs::SplitK)
                       && (traits_lhs::DimPerThread == traits_rhs::DimPerThread)
                       && (traits_lhs::KPerThread == traits_rhs::KPerThread);
            }
            else
            {
                return false;
            }
        }

        // Test for same layout
        template <typename MatrixLayoutLhs, typename MatrixLayoutRhs>
        ROCWMMA_HOST_DEVICE constexpr static bool testMatrixLayoutSame()
        {
            return ((traits_lhs::is_col_ortho && traits_rhs::is_col_ortho)
                    || (traits_lhs::is_row_ortho && traits_rhs::is_row_ortho)
                    || (traits_lhs::is_col_inline && traits_rhs::is_col_inline)
                    || (traits_lhs::is_row_inline && traits_rhs::is_row_inline))
                   && testCompatibleMatrixParams<MatrixLayoutLhs, MatrixLayoutRhs>();
        }

        // Test for orthogonal layout
        template <typename MatrixLayoutLhs, typename MatrixLayoutRhs>
        ROCWMMA_HOST_DEVICE constexpr static bool testMatrixLayoutOrthogonal()
        {
            return ((traits_lhs::is_col_ortho && traits_rhs::is_row_ortho)
                    || (traits_lhs::is_row_ortho && traits_rhs::is_col_ortho)
                    || (traits_lhs::is_col_inline && traits_rhs::is_row_inline)
                    || (traits_lhs::is_row_inline && traits_rhs::is_col_inline))
                   && testCompatibleMatrixParams<MatrixLayoutLhs, MatrixLayoutRhs>();
        }

        // Now to implement the interfaces for is_layout_same and is_layout_orthogonal,
        // with MatrixLayout types.

        // Implement sameness classifier for matrix layouts
        template <typename MatrixLayoutLhs, typename MatrixLayoutRhs>
        struct is_layout_same<
            MatrixLayoutLhs,
            MatrixLayoutRhs,
            enable_if_t<traits_lhs::is_matrix_layout && traits_rhs::is_matrix_layout>>
            : public integral_constant<bool,
                                       testMatrixLayoutSame<MatrixLayoutLhs, MatrixLayoutRhs>()>
        {
        };

        // Implement orthogonality classifier for matrix layouts
        template <typename MatrixLayoutLhs, typename MatrixLayoutRhs>
        struct is_layout_orthogonal<
            MatrixLayoutLhs,
            MatrixLayoutRhs,
            enable_if_t<traits_lhs::is_matrix_layout && traits_rhs::is_matrix_layout>>
            : public integral_constant<
                  bool,
                  testMatrixLayoutOrthogonal<MatrixLayoutLhs, MatrixLayoutRhs>()>
        {
        };

#undef traits_lhs
#undef traits_rhs

        // Matrix space transpose guide: Swap rows / cols
        // VW stays consistent.
        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct orthogonal_layout<ColOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
        {
            using type = RowOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>;
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct orthogonal_layout<RowOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
        {
            using type = ColOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>;
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct orthogonal_layout<ColInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
        {
            using type = RowInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>;
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct orthogonal_layout<RowInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
        {
            using type = ColInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>;
        };

        // Orthogonal guide for interleaved layouts
        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MmaDim,
                  uint32_t SplitK>
        struct orthogonal_layout<ColOrthoInt<BlockDim, BlockK, DataT, MmaDim, SplitK>>
        {
            using type = RowOrthoInt<BlockDim, BlockK, DataT, MmaDim, SplitK>;
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MmaDim,
                  uint32_t SplitK>
        struct orthogonal_layout<RowOrthoInt<BlockDim, BlockK, DataT, MmaDim, SplitK>>
        {
            using type = ColOrthoInt<BlockDim, BlockK, DataT, MmaDim, SplitK>;
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MmaDim,
                  uint32_t SplitK>
        struct orthogonal_layout<ColInlineInt<BlockDim, BlockK, DataT, MmaDim, SplitK>>
        {
            using type = RowInlineInt<BlockDim, BlockK, DataT, MmaDim, SplitK>;
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MmaDim,
                  uint32_t SplitK>
        struct orthogonal_layout<RowInlineInt<BlockDim, BlockK, DataT, MmaDim, SplitK>>
        {
            using type = ColInlineInt<BlockDim, BlockK, DataT, MmaDim, SplitK>;
        };

        template <typename MatrixLayout>
        struct layout_traits<MatrixLayout,
                             enable_if_t<matrix_layout_traits<MatrixLayout>::is_matrix_layout>>
            : public matrix_layout_traits<MatrixLayout>
        {
        };

    } // namespace LayoutTraits_impl

} // namespace rocwmma

#if !defined(__HIPCC_RTC__)
namespace std
{

    template <typename MatrixLayout>
    inline ostream&
        operator<<(ostream&                                                              stream,
                   rocwmma::LayoutTraits_impl::matrix_layout_traits<MatrixLayout> const& traits)
    {
        using matrix_traits = decay_t<decltype(traits)>;

        stream << "MatrixLayout Traits: " << MatrixLayout{} << std::endl;
        stream << "is_col_ortho: " << matrix_traits::is_col_ortho << std::endl;
        stream << "is_row_ortho: " << matrix_traits::is_row_ortho << std::endl;
        stream << "is_col_inline: " << matrix_traits::is_col_inline << std::endl;
        stream << "is_row_inline: " << matrix_traits::is_row_inline << std::endl;
        stream << "is_interleaved: " << matrix_traits::is_interleaved << std::endl;
        stream << "is_matrix_layout: " << matrix_traits::is_matrix_layout << std::endl;
        stream << "BlockDim: " << matrix_traits::BlockDim << std::endl;
        stream << "KDim: " << matrix_traits::KDim << std::endl;
        stream << "MmaDim: " << matrix_traits::MmaDim << std::endl;
        stream << "SplitK: " << matrix_traits::SplitK << std::endl;
        stream << "VectorWidth: " << matrix_traits::VectorWidth << std::endl;
        stream << "MaxVectorWidth: " << matrix_traits::MaxVectorWidth << std::endl;
        return stream;
    }

} // namespace std

#endif // !defined(__HIPCC_RTC__)

#endif // ROCWMMA_MATRIX_LAYOUT_TRAITS_IMPL_HPP
