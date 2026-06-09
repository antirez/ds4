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
#ifndef ROCWMMA_REGISTER_LAYOUT_TRANSFORMS_IMPL_HPP
#define ROCWMMA_REGISTER_LAYOUT_TRANSFORMS_IMPL_HPP

#include "layout.hpp"
#include "layout_traits.hpp"
#include "transforms/transforms.hpp"

namespace rocwmma
{
    namespace LayoutTransforms_impl
    {
        using RegisterLayout::Format;

        template <typename LayoutTraits>
        struct InterleavedParamHelper : public LayoutTraits
        {
            // Additional input traits for layout
            static constexpr uint32_t InputBlockSize
                = LayoutTraits::MmaDim * LayoutTraits::KDim / Constants::AMDGCN_WAVE_SIZE;

            // Additional acc traits for layout
            static constexpr uint32_t AccVecSize
                = LayoutTraits::MmaDim * LayoutTraits::MmaDim / Constants::AMDGCN_WAVE_SIZE;
            static constexpr uint32_t MmaBlocksA = LayoutTraits::KDim / LayoutTraits::MmaDim;
            static constexpr uint32_t MmaBlocksB = LayoutTraits::BlockDim / LayoutTraits::MmaDim;
            static constexpr uint32_t AccMaxVW   = detail::MaxVWSelector<accumulator,
                                                                       LayoutTraits::BlockDim,
                                                                       LayoutTraits::KDim,
                                                                       typename LayoutTraits::DataT,
                                                                       void>::Result;
        };

        // Specific transform from one format to another
        template <RegisterLayout::Format Src, RegisterLayout::Format Dst, typename... Traits>
        struct register_layout_transform_impl;

        // Non-interleaved formats
        // SOA <-> AOS
        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::SOA, Format::AOS, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::soa_to_aos<LayoutTraits::BlockDim, LayoutTraits::MaxVectorWidth>(
                    forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::AOS, Format::SOA, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::aos_to_soa<LayoutTraits::BlockDim, LayoutTraits::MaxVectorWidth>(
                    forward<VecT>(v));
            }
        };

        // Non-interleaved gfx11 formats
        // SOA, AOS <-> WMMA input
        // SOA, AOS <-> WMMA acc
        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::SOA, Format::WMMA_INPUT_GFX11, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::soa_to_wmma_input_gfx11(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::AOS, Format::WMMA_INPUT_GFX11, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::aos_to_wmma_input_gfx11<LayoutTraits::BlockDim,
                                                           LayoutTraits::MaxVectorWidth>(
                    forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::WMMA_INPUT_GFX11, Format::SOA, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::wmma_input_gfx11_to_soa(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::WMMA_INPUT_GFX11, Format::AOS, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::wmma_input_gfx11_to_aos<LayoutTraits::BlockDim,
                                                           LayoutTraits::MaxVectorWidth>(
                    forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::SOA, Format::WMMA_ACC_GFX11, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::soa_to_wmma_acc_gfx11(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::AOS, Format::WMMA_ACC_GFX11, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::aos_to_wmma_acc_gfx11<LayoutTraits::BlockDim,
                                                         LayoutTraits::MaxVectorWidth>(
                    forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::WMMA_ACC_GFX11, Format::SOA, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::wmma_acc_gfx11_to_soa(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::WMMA_ACC_GFX11, Format::AOS, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::wmma_acc_gfx11_to_aos<LayoutTraits::BlockDim,
                                                         LayoutTraits::MaxVectorWidth>(
                    forward<VecT>(v));
            }
        };

        // Interleaved formats
        // SOA_INT <-> AOS_INT
        // SOA_INT, AOS_INT <-> A-major acc fmt
        // SOA_INT, AOS_INT <-> B-major acc fmt
        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::SOA_INT, Format::AOS_INT, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::soa_int_to_aos_int<LayoutTraits::KPerThread>(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::AOS_INT, Format::SOA_INT, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::aos_int_to_soa_int<LayoutTraits::DimPerThread>(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::SOA_INT,
                                              Format::ACC_INT_A_MAJOR,
                                              LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                using Helper = InterleavedParamHelper<LayoutTraits>;

                return Transforms::soa_int_to_mma_acc_int_a_major<Helper::AccVecSize,
                                                                  Helper::MmaBlocksA,
                                                                  Helper::AccMaxVW,
                                                                  Helper::MmaDim>(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::AOS_INT,
                                              Format::ACC_INT_A_MAJOR,
                                              LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                using Helper = InterleavedParamHelper<LayoutTraits>;

                return Transforms::aos_int_to_mma_acc_int_a_major<Helper::AccVecSize,
                                                                  Helper::MmaBlocksA,
                                                                  Helper::MmaBlocksB,
                                                                  Helper::AccMaxVW,
                                                                  Helper::MmaDim>(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::ACC_INT_A_MAJOR,
                                              Format::SOA_INT,
                                              LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                using Helper = InterleavedParamHelper<LayoutTraits>;

                return Transforms::mma_acc_int_a_major_to_soa_int<Helper::AccVecSize,
                                                                  Helper::MmaBlocksB,
                                                                  Helper::AccMaxVW,
                                                                  Helper::MmaDim>(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::ACC_INT_A_MAJOR,
                                              Format::AOS_INT,
                                              LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                using Helper = InterleavedParamHelper<LayoutTraits>;

                return Transforms::mma_acc_int_a_major_to_aos_int<Helper::AccVecSize,
                                                                  Helper::AccMaxVW,
                                                                  Helper::MmaDim>(forward<VecT>(v));
            }
        };

        // Interleaved gfx11 formats
        // SOA_INT, AOS_INT <-> WMMA input
        // SOA_INT, AOS_INT <-> WMMA acc
        // A-major acc fmt <-> WMMA acc
        // B-major acc fmt <-> WMMA acc
        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::SOA_INT,
                                              Format::WMMA_INPUT_GFX11,
                                              LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                using Helper = InterleavedParamHelper<LayoutTraits>;

                return Transforms::soa_int_to_wmma_input_gfx11<Helper::InputBlockSize>(
                    forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::AOS_INT,
                                              Format::WMMA_INPUT_GFX11,
                                              LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                using Helper = InterleavedParamHelper<LayoutTraits>;

                return Transforms::aos_int_to_wmma_input_gfx11<Helper::DimPerThread,
                                                               Helper::InputBlockSize>(
                    forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::WMMA_INPUT_GFX11,
                                              Format::SOA_INT,
                                              LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                using Helper = InterleavedParamHelper<LayoutTraits>;

                return Transforms::wmma_input_gfx11_to_soa_int<Helper::InputBlockSize>(
                    forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::WMMA_INPUT_GFX11,
                                              Format::AOS_INT,
                                              LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                using Helper = InterleavedParamHelper<LayoutTraits>;

                return Transforms::wmma_input_gfx11_to_aos_int<Helper::KPerThread,
                                                               Helper::InputBlockSize>(
                    forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::SOA_INT, Format::WMMA_ACC_GFX11, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                using Helper = InterleavedParamHelper<LayoutTraits>;

                return Transforms::soa_int_to_wmma_acc_gfx11<Helper::AccVecSize,
                                                             Helper::MmaBlocksA,
                                                             Helper::AccMaxVW,
                                                             Helper::MmaDim>(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::AOS_INT, Format::WMMA_ACC_GFX11, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                using Helper = InterleavedParamHelper<LayoutTraits>;

                return Transforms::aos_int_to_wmma_acc_gfx11<Helper::AccVecSize,
                                                             Helper::MmaBlocksA,
                                                             Helper::MmaBlocksB,
                                                             Helper::AccMaxVW,
                                                             Helper::MmaDim>(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::WMMA_ACC_GFX11, Format::SOA_INT, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                using Helper = InterleavedParamHelper<LayoutTraits>;

                return Transforms::wmma_acc_gfx11_to_aos_int<Helper::AccVecSize,
                                                             Helper::MmaBlocksB,
                                                             Helper::AccMaxVW,
                                                             Helper::MmaDim>(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::WMMA_ACC_GFX11, Format::AOS_INT, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                using Helper = InterleavedParamHelper<LayoutTraits>;

                return Transforms::wmma_acc_gfx11_to_aos_int<Helper::AccVecSize,
                                                             Helper::AccMaxVW,
                                                             Helper::MmaDim>(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::ACC_INT_A_MAJOR,
                                              Format::WMMA_ACC_GFX11,
                                              LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::mma_acc_int_a_major_to_wmma_acc_gfx11(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::WMMA_ACC_GFX11,
                                              Format::ACC_INT_A_MAJOR,
                                              LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::wmma_acc_gfx11_to_mma_acc_int_a_major(forward<VecT>(v));
            }
        };

        // Keeps things a bit more tidy. Quick access to register layout traits.
        using LayoutTraits_impl::register_layout_traits;
#define traits_lhs register_layout_traits<RegisterLayoutLhs>
#define traits_rhs register_layout_traits<RegisterLayoutRhs>

        // Note: If you arrive at an undefined register_transform error, it is likely
        // the layout transformation is not currently supported. Need to either implement
        // the transform or ensure your layout transform mapping is correct.

        // Interface to transform from one register layout to another.
        template <typename RegisterLayoutSrc, typename RegisterLayoutDst, typename Enabler = void>
        struct register_layout_transform;

        // Passthrough transform (NOP):
        // - Layouts are the same
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        struct register_layout_transform<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            enable_if_t<is_layout_same_v<RegisterLayoutLhs, RegisterLayoutRhs>>>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return v;
            }
        };

        // Unsupported transform:
        // - Invalid RegisterLayouts
        // - Non-orthogonal (no transform path)
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        struct register_layout_transform<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            enable_if_t<
                !is_layout_same_v<
                    RegisterLayoutLhs,
                    RegisterLayoutRhs> && (!traits_lhs::is_register_layout || !traits_rhs::is_register_layout || !is_layout_orthogonal_v<RegisterLayoutLhs, RegisterLayoutRhs>)>>
        {
            template <typename VecT>
            ROCWMMA_UNSUPPORTED_IMPL("Register layout transform is not supported")
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return forward<VecT>(v);
            }
        };

        // Valid transform:
        // - RegisterLayouts are valid
        // - Orthogonal (transform path exists)
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        struct register_layout_transform<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            enable_if_t<(traits_lhs::is_register_layout && traits_rhs::is_register_layout)
                        && is_layout_orthogonal_v<RegisterLayoutLhs, RegisterLayoutRhs>>>
        {

            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                // Extract traits for functional implementation
                using RegisterLayout::Format;
                using storage_traits
                    = conditional_t<traits_lhs::is_storage, traits_lhs, traits_rhs>;
                using transform_impl = register_layout_transform_impl<traits_lhs::Format,
                                                                      traits_rhs::Format,
                                                                      storage_traits>;

                return transform_impl::exec(forward<VecT>(v));
            }
        };

#undef traits_lhs
#undef traits_rhs

    } // namespace LayoutTransforms_impl

} // namespace rocWMMA

#endif // ROCWMMA_REGISTER_LAYOUT_TRANSFORMS_HPP
