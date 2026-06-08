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
#ifndef ROCWMMA_TRANSFORMS_API_IMPL_HPP
#define ROCWMMA_TRANSFORMS_API_IMPL_HPP

#include "internal/layout/layout.hpp"
#include "internal/layout/register_layout_transforms.hpp"
#include "internal/transforms.hpp"
#include "rocwmma_transforms.hpp"

namespace rocwmma
{
    namespace detail
    {
        template <typename SrcFragT, typename DstFragT>
        struct ApplyFragmentTransform
        {
        private:
            using SrcFragLayout = typename GetIOConfig_t<SrcFragT>::IOLayout::FragmentLayout;
            using DstFragLayout = typename GetIOConfig_t<DstFragT>::IOLayout::FragmentLayout;

        public:
            // Result type
            using Type = DstFragT;

            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(SrcFragT const& frag)
            {
                // Same fragment type
                if constexpr(is_same_v<SrcFragT, DstFragT>)
                {
                    return frag;
                }
                // Different fragment type but same register layout
                else if constexpr(is_layout_same_v<SrcFragLayout, DstFragLayout>)
                {
                    return reinterpret_cast<DstFragT const&>(frag);
                }
                // Apply register transform
                else
                {
                    using SrcFragTraits = fragment_traits<SrcFragT>;
                    using DstFragTraits = fragment_traits<DstFragT>;

                    using SrcSchedulerTraits = scheduler_traits<typename SrcFragTraits::Scheduler>;
                    using DstSchedulerTraits = scheduler_traits<typename DstFragTraits::Scheduler>;

                    static_assert(SrcSchedulerTraits::WaveCount == DstSchedulerTraits::WaveCount,
                                  "WaveCounts for Src and Dst frags don't match");

                    auto result    = DstFragT{};
                    result.mAccess = register_layout_transform<SrcFragLayout, DstFragLayout>::exec(
                        frag.mAccess);

                    return result;
                }
            }
        };

        ///
        /// Apply logical transpose of fragment
        ///

        // Below are defined as fast implicit transposes:
        // - We re-interpret meaning between cols of A and rows of B,
        // in order to change the shape of our data for reading / writing.
        // Implicit transposes of fragment objects are designed to be
        // relatively cheap, and should only require a signature cast.
        // Assumptions:
        // - BlockDim and KDim are identical
        // - Matrix Layouts are orthogonal (exchange rows / cols)
        // - Data layouts are orthogonal (exchange row / col major)
        // - Register layouts match. (No change)
        // Example:
        // - A matrix_a fragment of (BlockM x BlockK) = 32x8 in col_major may be reinterpreted
        //   as a matrix_b fragment of (BlockK x BlockN) = 8x32 in row_major.
        //   Here, we have transposed (implicitly) 8 cols of matrix_a into 8 rows of matrix_b.
        template <typename FragT>
        struct ApplyTranspose;

        template <uint32_t BlockM,
                  uint32_t BlockN,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayoutT,
                  typename Scheduler>
        struct ApplyTranspose<
            fragment<matrix_a, BlockM, BlockN, BlockK, DataT, DataLayoutT, Scheduler>>
        {
        private:
            // Original frag A type
            using FragA = fragment<matrix_a, BlockM, BlockN, BlockK, DataT, DataLayoutT, Scheduler>;

            // Transpose to frag B type in opposite data layout:
            // - Exchange Block M for BlockN
            // - Exchange row_major for col_major and vice-versa
            using FragB = fragment<matrix_b,
                                   BlockN,
                                   BlockM,
                                   BlockK,
                                   DataT,
                                   orthogonal_layout_t<DataLayoutT>,
                                   Scheduler>;

            using IOConfigA = GetIOConfig_t<FragA>;
            using IOConfigB = GetIOConfig_t<FragB>;

            using ApplyXForm = ApplyFragmentTransform<FragA, FragB>;

            // Assumptions check
            static_assert(IOConfigA::IOShape::BlockDim == IOConfigB::IOShape::BlockDim,
                          "BlockDim of transposed fragment doesn't match");

            static_assert(IOConfigA::IOShape::KDim == IOConfigB::IOShape::KDim,
                          "KDim of transposed fragment doesn't match");

            static_assert(is_layout_orthogonal_v<typename IOConfigA::IOLayout::DataLayout,
                                                 typename IOConfigB::IOLayout::DataLayout>,
                          "Data Layouts are not orthogonal");

            static_assert(is_layout_orthogonal_v<typename IOConfigA::IOLayout::MatrixLayout,
                                                 typename IOConfigB::IOLayout::MatrixLayout>,
                          "Matrix Layouts are not orthogonal");

            static_assert(is_layout_same_v<typename IOConfigA::IOLayout::FragmentLayout,
                                           typename IOConfigB::IOLayout::FragmentLayout>,
                          "Register layouts do not match");

        public:
            // Result type
            using Type = typename ApplyXForm::Type;

            // Because of the expectation that matrix_a data is orthogonal to matrix_b
            // with the same register layout, the transpose comes as a simple re-cast.
            ROCWMMA_DEVICE static inline decltype(auto) exec(FragA const& frag)
            {
                return ApplyXForm::exec(frag);
            }
        };

        template <uint32_t BlockM,
                  uint32_t BlockN,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayoutT,
                  typename Scheduler>
        struct ApplyTranspose<
            fragment<matrix_b, BlockM, BlockN, BlockK, DataT, DataLayoutT, Scheduler>>
        {
        private:
            // Original frag B type
            using FragB = fragment<matrix_b, BlockM, BlockN, BlockK, DataT, DataLayoutT, Scheduler>;

            // Transpose to frag A type in opposite data layout:
            // - Exchange Block M for BlockN
            // - Exchange row_major for col_major and vice-versa
            using FragA = fragment<matrix_a,
                                   BlockN,
                                   BlockM,
                                   BlockK,
                                   DataT,
                                   orthogonal_layout_t<DataLayoutT>,
                                   Scheduler>;

            using IOConfigA = GetIOConfig_t<FragA>;
            using IOConfigB = GetIOConfig_t<FragB>;

            using ApplyXForm = ApplyFragmentTransform<FragB, FragA>;

            // Assumptions check
            static_assert(IOConfigA::IOShape::BlockDim == IOConfigB::IOShape::BlockDim,
                          "BlockDim of transposed frag doesn't match");

            static_assert(IOConfigA::IOShape::KDim == IOConfigB::IOShape::KDim,
                          "KDim of transposed frag doesn't match");

            static_assert(is_layout_orthogonal_v<typename IOConfigA::IOLayout::DataLayout,
                                                 typename IOConfigB::IOLayout::DataLayout>,
                          "Data Layouts are not orthogonal");

            static_assert(is_layout_orthogonal_v<typename IOConfigA::IOLayout::MatrixLayout,
                                                 typename IOConfigB::IOLayout::MatrixLayout>,
                          "Matrix Layouts are not orthogonal");

            static_assert(is_layout_same_v<typename IOConfigA::IOLayout::FragmentLayout,
                                           typename IOConfigB::IOLayout::FragmentLayout>,
                          "Fragment register layouts do not match");

        public:
            // Result type
            using Type = typename ApplyXForm::Type;

            // Because of the expectation that matrix_a data is orthogonal to matrix_b
            // with the same register layout, the transpose comes as a simple re-cast.
            ROCWMMA_DEVICE static inline FragA const& exec(FragB const& frag)
            {
                return ApplyXForm::exec(frag);
            }
        };

        // Below are defined data layout transforms:
        // - The same fragment data is to be re-arranged in the format
        //   of another another Data Layout.
        // - This can be achieved through AOS<->SOA transformations
        //   where required, as long as the matrix context, block
        //   dimensions and MaxVectorWidths have not changed.
        // Assumptions:
        // - Matrix contexts are identical, as well as Block dimensions
        // - Matrix layouts may change, but adhere to strict AOS or SOA formats.
        // - Register layout transforms are needed when they do not match
        // Example:
        // - A matrix_a fragment of (BlockM x BlockK) = 32x8 in col_major may be reinterpreted as
        //   a matrix_a fragment of (BlockM x BlockK) = 32x8 in row_major.
        //   Here, we have rearranged col_major for row_major ordering.
        template <typename FragT, typename NewDataLayoutT>
        struct ApplyDataLayout;

        // Other layout case
        template <typename MatrixT,
                  uint32_t BlockM,
                  uint32_t BlockN,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayoutT,
                  typename Scheduler,
                  typename NewDataLayoutT>
        struct ApplyDataLayout<
            fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT, Scheduler>,
            NewDataLayoutT>
        {
        private:
            using SrcFragT
                = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT, Scheduler>;
            using DstFragT
                = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, NewDataLayoutT, Scheduler>;

        public:
            // Result type
            using Type = DstFragT;

            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(SrcFragT const& frag)
            {
                using ApplyXForm = ApplyFragmentTransform<SrcFragT, DstFragT>;
                return ApplyXForm::exec(frag);
            }
        };

        // Other layout case
        template <typename FragT>
        struct ApplyRegisterFile;

        template <typename MatrixT,
                  uint32_t BlockM,
                  uint32_t BlockN,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayoutT,
                  typename Scheduler>
        struct ApplyRegisterFile<
            fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT, Scheduler>>
        {
        private:
            constexpr static const uint32_t registerFileWidth = Constants::AMDGCN_WAVE_SIZE;
            using SrcFragT
                = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT, Scheduler>;

            // Use geometry size because the incoming fragment may be cooperative
            // and only have partial data.
            using SrcIOShape = GetIOShape_t<SrcFragT>;
            constexpr static auto DstK
                = SrcIOShape::BlockWidth * SrcIOShape::BlockHeight / registerFileWidth;

            using DstFragT = fragment<matrix_b,
                                      registerFileWidth,
                                      registerFileWidth,
                                      DstK,
                                      DataT,
                                      DataLayoutT,
                                      Scheduler>;

            static_assert(SrcFragT::size() == DstFragT::size(),
                          "Registerfile must have same vector size as input");

        public:
            using Type = DstFragT;

            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec_to(SrcFragT const& frag)
            {
                return reinterpret_cast<DstFragT const&>(frag);
            }

            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec_from(DstFragT const& frag)
            {
                return reinterpret_cast<SrcFragT const&>(frag);
            }
        };

    } // namespace detail

    /// These wrappers must perfect-forward and perfect-return because the return types and
    // arguments above could be references or copy types.
    // @cond
    template <typename FragT>
    ROCWMMA_DEVICE static inline decltype(auto) apply_transpose(FragT&& frag)
    {
        return detail::template ApplyTranspose<decay_t<FragT>>::exec(forward<FragT>(frag));
    }

    template <typename DataLayoutT, typename FragT>
    ROCWMMA_DEVICE static inline decltype(auto) apply_data_layout(FragT&& frag)
    {
        return detail::template ApplyDataLayout<decay_t<FragT>, DataLayoutT>::exec(
            forward<FragT>(frag));
    }

    template <typename DstFragT, typename FragT>
    ROCWMMA_DEVICE static inline decltype(auto) apply_fragment(FragT&& frag)
    {
        return detail::template ApplyFragmentTransform<decay_t<FragT>, DstFragT>::exec(
            forward<FragT>(frag));
    }

    template <typename FragT>
    ROCWMMA_DEVICE static inline decltype(auto) to_register_file(FragT&& frag)
    {
        return detail::template ApplyRegisterFile<decay_t<FragT>>::exec_to(forward<FragT>(frag));
    }

    template <typename DstFragT, typename FragT>
    ROCWMMA_DEVICE static inline decltype(auto) from_register_file(FragT&& frag)
    {
        return detail::template ApplyRegisterFile<DstFragT>::exec_from(forward<FragT>(frag));
    }

    // @endcond

} // namespace rocwmma

#endif // ROCWMMA_TRANSFORMS_API_IMPL_HPP
