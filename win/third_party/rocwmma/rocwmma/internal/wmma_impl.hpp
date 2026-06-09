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
#ifndef ROCWMMA_WMMA_IMPL_HPP
#define ROCWMMA_WMMA_IMPL_HPP

#include "constants.hpp"
#include "types.hpp"
#include "utility/type_traits.hpp"

namespace rocwmma
{
    namespace detail
    {
        struct Unsupported;

        // Enabler for targets.
        // Given a TargetId, enable if it exists in the TargetIds list
        template <uint32_t TargetId, uint32_t... TargetIds>
        using enable_target_id_t = enable_if_t<contains_number_v<uint32_t, TargetId, TargetIds...>>;

        // Enabler for all of gfx11
        template <uint32_t TargetId, bool Cond = true>
        using enable_gfx11_t
            = enable_if_t<contains_number_v<uint32_t,
                                            TargetId,
                                            Constants::AMDGCN_ARCH_ID_GFX1100,
                                            Constants::AMDGCN_ARCH_ID_GFX1101,
                                            Constants::AMDGCN_ARCH_ID_GFX1102,
                                            Constants::AMDGCN_ARCH_ID_GFX1151> && Cond>;

        // Enabler for all of gfx12
        template <uint32_t TargetId, bool Cond = true>
        using enable_gfx12_t
            = enable_if_t<contains_number_v<uint32_t,
                                            TargetId,
                                            Constants::AMDGCN_ARCH_ID_GFX1200,
                                            Constants::AMDGCN_ARCH_ID_GFX1201> && Cond>;

        // Enabler for all of gfx11 and gfx12
        template <uint32_t TargetId, bool Cond = true>
        using enable_gfx11_gfx12_t
            = enable_if_t<contains_number_v<uint32_t,
                                            TargetId,
                                            Constants::AMDGCN_ARCH_ID_GFX1100,
                                            Constants::AMDGCN_ARCH_ID_GFX1101,
                                            Constants::AMDGCN_ARCH_ID_GFX1102,
                                            Constants::AMDGCN_ARCH_ID_GFX1151,
                                            Constants::AMDGCN_ARCH_ID_GFX1200,
                                            Constants::AMDGCN_ARCH_ID_GFX1201> && Cond>;

        /*! \class amdgcn_wmma
        *  \brief  Builtin wrapper for wmma instructions
        *  @tparam InputTA Datatype of input A
        *  @tparam InputTB Datatype of input B
        *  @tparam ComputeT Datatype of accumulator
        *  @tparam BlockM M-dimension of wmma block
        *  @tparam BlockN N-dimension of wmma block
        *  @tparam GfxTarget The current gfx family target of interest being compiled
        *  @tparam TargetEnable Enabler for the current target if supported
        */
        template <typename InputTA,
                  typename InputTB,
                  typename ComputeT,
                  uint32_t BlockM,
                  uint32_t BlockN,
                  uint32_t BlockK,
                  uint32_t GfxTargetId = Constants::AMDGCN_CURRENT_ARCH_ID,
                  typename Enabler     = void>
        struct amdgcn_wmma
        {
            // This is a pass-through implementation that isn't supported, and doesn't
            // do anything practical. The following trait will allow us to identify
            // unsupported instances, as we won't include it in the overloads to follow.
            using Unsupported = Unsupported;

        private:
            using PackTraitsA   = PackTraits<InputTA>;
            using PackTraitsB   = PackTraits<InputTB>;
            using PackTraitsAcc = PackTraits<ComputeT>;

            constexpr static uint32_t InputASize
                = BlockM * BlockK / (Constants::AMDGCN_WAVE_SIZE * PackTraitsA::PackRatio);
            constexpr static uint32_t InputBSize
                = BlockN * BlockK / (Constants::AMDGCN_WAVE_SIZE * PackTraitsB::PackRatio);
            constexpr static uint32_t AccumSize
                = BlockM * BlockM / (Constants::AMDGCN_WAVE_SIZE * PackTraitsAcc::PackRatio);

        public:
            using ARegsT = VecT<typename PackTraitsA::PackedT, InputASize>;
            using BRegsT = VecT<typename PackTraitsB::PackedT, InputBSize>;
            using CRegsT = VecT<typename PackTraitsAcc::PackedT, AccumSize>;
            using DRegsT = VecT<typename PackTraitsAcc::PackedT, AccumSize>;
        };

        enum struct WmmaCtrlFlags : bool
        {
            // Output register selection of WMMA.
            // Low = bits [15:0]
            // High = bits[31:16]
            LOW  = false,
            HIGH = true,

            // Signage indicator of inputs / accum
            UNSIGNED = false,
            SIGNED   = true
        };

        // gfx11 implementations
        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float32_t,
                           16u,
                           16u,
                           4u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<float16_t, float16_t, float32_t, 16u, 16u, 8u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float32_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<float16_t, float16_t, float32_t, 16u, 16u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                    to_native_vector(regsA), to_native_vector(regsB), to_native_vector(regsC))};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float16_t,
                           16u,
                           16u,
                           4u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<float16_t, float16_t, float16_t, 16u, 16u, 8u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float16_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<float16_t, float16_t, float16_t, 16u, 16u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float16_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_wmma_f16_16x16x16_f16_w32(to_native_vector(regsA),
                                                                  to_native_vector(regsB),
                                                                  to_native_vector(regsC),
                                                                  (bool)AccumBits)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           16u,
                           16u,
                           4u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<bfloat16_t, bfloat16_t, float32_t, 16u, 16u, 8u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<bfloat16_t, bfloat16_t, float32_t, 16u, 16u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                    to_native_vector(regsA), to_native_vector(regsB), to_native_vector(regsC))};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           bfloat16_t,
                           16u,
                           16u,
                           4u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<bfloat16_t, bfloat16_t, bfloat16_t, 16u, 16u, 8u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           bfloat16_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<bfloat16_t, bfloat16_t, bfloat16_t, 16u, 16u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           bfloat16_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(to_native_vector(regsA),
                                                                    to_native_vector(regsB),
                                                                    to_native_vector(regsC),
                                                                    (bool)AccumBits)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<int8_t,
                           int8_t,
                           int32_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegI32x2;
            using BRegsT = VRegI32x2;
            using CRegsT = AccRegI32x8;
            using DRegsT = AccRegI32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<int8_t, int8_t, int32_t, 16u, 16u, 16u, GfxTargetId>::exec(
                    concat(regsA, ARegsT{0}),
                    concat(regsB, BRegsT{0}),
                    forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<int8_t,
                           int8_t,
                           int32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegI32x4;
            using BRegsT = VRegI32x4;
            using CRegsT = AccRegI32x8;
            using DRegsT = AccRegI32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32((bool)InputSign,
                                                                  to_native_vector(regsA),
                                                                  (bool)InputSign,
                                                                  to_native_vector(regsB),
                                                                  to_native_vector(regsC),
                                                                  (bool)AccumSign)};
                return result;
            }
        };

        // gfx12 implementations
        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float32_t,
                           16u,
                           16u,
                           4u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<float16_t, float16_t, float32_t, 16u, 16u, 8u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float32_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<float16_t, float16_t, float32_t, 16u, 16u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    to_native_vector(regsA), to_native_vector(regsB), to_native_vector(regsC))};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float16_t,
                           16u,
                           16u,
                           4u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<float16_t, float16_t, float16_t, 16u, 16u, 8u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float16_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<float16_t, float16_t, float16_t, 16u, 16u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float16_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(
                    to_native_vector(regsA), to_native_vector(regsB), to_native_vector(regsC))};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           16u,
                           16u,
                           4u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<bfloat16_t, bfloat16_t, float32_t, 16u, 16u, 8u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<bfloat16_t, bfloat16_t, float32_t, 16u, 16u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                    to_native_vector(regsA), to_native_vector(regsB), to_native_vector(regsC))};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           bfloat16_t,
                           16u,
                           16u,
                           4u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = VRegF32x4;
            using DRegsT = VRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<bfloat16_t, bfloat16_t, bfloat16_t, 16u, 16u, 8u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           bfloat16_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = VRegF32x4;
            using DRegsT = VRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<bfloat16_t, bfloat16_t, bfloat16_t, 16u, 16u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           bfloat16_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = VRegF32x4;
            using DRegsT = VRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12(
                    to_native_vector(regsA), to_native_vector(regsB), to_native_vector(regsC))};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<int8_t,
                           int8_t,
                           int32_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegI32x1;
            using BRegsT = VRegI32x1;
            using CRegsT = AccRegI32x8;
            using DRegsT = AccRegI32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<int8_t, int8_t, int32_t, 16u, 16u, 16u, GfxTargetId>::exec(
                    concat(regsA, ARegsT{0}),
                    concat(regsB, BRegsT{0}),
                    forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<int8_t,
                           int8_t,
                           int32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegI32x2;
            using BRegsT = VRegI32x2;
            using CRegsT = AccRegI32x8;
            using DRegsT = AccRegI32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12((bool)InputSign,
                                                                        to_native_vector(regsA),
                                                                        (bool)InputSign,
                                                                        to_native_vector(regsB),
                                                                        to_native_vector(regsC),
                                                                        (bool)AccumSign)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float8_t,
                           float8_t,
                           float32_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<float8_t, float8_t, float32_t, 16u, 16u, 16u, GfxTargetId>::exec(
                    concat(regsA, ARegsT{0}),
                    concat(regsB, BRegsT{0}),
                    forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float8_t,
                           float8_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 2>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    to_native_vector(regsC))};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat8_t,
                           bfloat8_t,
                           float32_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<bfloat8_t, bfloat8_t, float32_t, 16u, 16u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat8_t,
                           bfloat8_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 2>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    to_native_vector(regsC))};
                return result;
            }
        };

        // Derivative implementations
        template <uint32_t GfxTargetId, uint32_t BlockK>
        struct amdgcn_wmma<hfloat16_t,
                           hfloat16_t,
                           float32_t,
                           16u,
                           16u,
                           BlockK,
                           GfxTargetId,
                           enable_gfx11_gfx12_t<GfxTargetId, !(bool)ROCWMMA_NO_HALF>>
            : public amdgcn_wmma<float16_t, float16_t, float32_t, 16u, 16u, BlockK, GfxTargetId>
        {
        };

        template <uint32_t GfxTargetId, uint32_t BlockK>
        struct amdgcn_wmma<hfloat16_t,
                           hfloat16_t,
                           hfloat16_t,
                           16u,
                           16u,
                           BlockK,
                           GfxTargetId,
                           enable_gfx11_gfx12_t<GfxTargetId, !(bool)ROCWMMA_NO_HALF>>
            : public amdgcn_wmma<float16_t, float16_t, float16_t, 16u, 16u, BlockK, GfxTargetId>
        {
        };

    } // namespace detail

    namespace MmaTraits_impl
    {
        template <typename WmmaOp>
        struct is_wmma : public false_type
        {
        };

        template <typename InputTA_In,
                  typename InputTB_In,
                  typename ComputeT_in,
                  uint32_t BlockM_In,
                  uint32_t BlockN_In,
                  uint32_t BlockK_In>
        struct is_wmma<
            detail::
                amdgcn_wmma<InputTA_In, InputTB_In, ComputeT_in, BlockM_In, BlockN_In, BlockK_In>>
            : public true_type
        {
        };

        template <typename WmmaOp>
        constexpr static bool is_wmma_v = is_wmma<WmmaOp>::value;

        // All of the overrides won't have the Unsupported tag
        template <typename WmmaOp, typename Enabler = void>
        struct is_wmma_supported : public true_type
        {
        };

        // Default implementation will have the Unsupported tag
        template <typename WmmaOp>
        struct is_wmma_supported<
            WmmaOp,
            enable_if_t<is_same_v<typename WmmaOp::Unsupported, detail::Unsupported>>>
            : public false_type
        {
        };

        template <typename WmmaOp>
        constexpr static bool is_wmma_supported_v = is_wmma_supported<WmmaOp>::value;

        template <typename MfmaOp>
        struct wmma_traits;

        template <typename InputTA_In,
                  typename InputTB_In,
                  typename ComputeT_In,
                  uint32_t BlockM_In,
                  uint32_t BlockN_In,
                  uint32_t BlockK_In>
        struct wmma_traits<
            detail::
                amdgcn_wmma<InputTA_In, InputTB_In, ComputeT_In, BlockM_In, BlockN_In, BlockK_In>>
        {
            // Base implementation
            using Impl = detail::
                amdgcn_wmma<InputTA_In, InputTB_In, ComputeT_In, BlockM_In, BlockN_In, BlockK_In>;

            // Operand types
            using InputTA  = InputTA_In;
            using InputTB  = InputTB_In;
            using ComputeT = ComputeT_In;

            // Raw input / output types
            using ARegsT = typename Impl::ARegsT;
            using BRegsT = typename Impl::BRegsT;
            using CRegsT = typename Impl::CRegsT;
            using DRegsT = typename Impl::DRegsT;

            // Geometric block sizes
            constexpr static uint32_t BlockM = BlockM_In;
            constexpr static uint32_t BlockN = BlockN_In;
            constexpr static uint32_t BlockK = BlockK_In;

            // Vector sizes per block (packed)
            constexpr static uint32_t BlockSizeA = VecTraits<ARegsT>::size();
            constexpr static uint32_t BlockSizeB = VecTraits<BRegsT>::size();
            constexpr static uint32_t BlockSizeC = VecTraits<CRegsT>::size();

            // Backend flags
            constexpr static bool is_wmma      = is_wmma_v<Impl>;
            constexpr static bool is_mfma      = false;
            constexpr static bool is_supported = is_wmma_supported_v<Impl>;
        };

        // MmaTraits implemented for mfma backend
        template <typename MmaOp>
        struct MmaTraits<MmaOp, enable_if_t<is_wmma_v<MmaOp>>> : public wmma_traits<MmaOp>
        {
        };

    } // namespace MmaTraits_impl

} // namespace rocwmma

#endif // ROCWMMA_WMMA_IMPL_HPP
