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
#ifndef ROCWMMA_MFMA_IMPL_HPP
#define ROCWMMA_MFMA_IMPL_HPP

#include "constants.hpp"
#include "convert.hpp"
#include "io_traits.hpp"
#include "types.hpp"
#include "vector.hpp"

namespace rocwmma
{

    namespace detail
    {

        enum struct MfmaCtrlFlags : uint32_t
        {
            DEFAULT = 0u,
        };

        struct Unsupported;

        // Enabler for targets.
        // Given a TargetId, enable if it exists in the TargetIds list
        template <uint32_t TargetId, uint32_t... TargetIds>
        using enable_target_id_t = enable_if_t<contains_number_v<uint32_t, TargetId, TargetIds...>>;

        // Enabler for all of gfx9, with binary condition.
        template <uint32_t TargetId, bool Cond = true>
        using enable_gfx9_t
            = enable_if_t<contains_number_v<uint32_t,
                                            TargetId,
                                            Constants::AMDGCN_ARCH_ID_GFX908,
                                            Constants::AMDGCN_ARCH_ID_GFX90A,
                                            Constants::AMDGCN_ARCH_ID_GFX942,
                                            Constants::AMDGCN_ARCH_ID_GFX950> && Cond>;

        /*! \class amdgcn_mfma
        *  \brief  Builtin wrapper for mfma instructions
        *  @tparam InputTA Datatype of input A
        *  @tparam InputTB Datatype of input B
        *  @tparam ComputeT Datatype of accumulator
        *  @tparam BlockM M-dimension of mfma block
        *  @tparam BlockN N-dimension of mfma block
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
        struct amdgcn_mfma
        {
            // This is a pass-through implementation that isn't supported, and doesn't
            // do anything practical. The following trait will allow us to identify
            // unsupported instances, as we won't include it in the overloads to follow.
            using Unsupported = Unsupported;

            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

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

            template <typename RegsA, typename RegsB, typename RegsC>
            ROCWMMA_DEVICE static inline decltype(auto)
                exec(RegsA&& regsA, RegsB&& regsB, RegsC&& regsC)
            {
                return forward<RegsC>(regsC);
            }
        };

        // Non-B32 compute types
        // Note: MFMA unit accum type is always b32 size.
        // Since we cannot natively accumulate in the desired type,
        // we must convert to native accum type, perform mfma and convert
        // the accum result back to the desired type.
        // Warning: This can be very slow!
        template <typename InputTA,
                  typename InputTB,
                  typename ComputeT,
                  uint32_t BlockM,
                  uint32_t BlockN,
                  uint32_t BlockK,
                  uint32_t GfxTargetId>
        struct amdgcn_mfma<InputTA,
                           InputTB,
                           ComputeT,
                           BlockM,
                           BlockN,
                           BlockK,
                           GfxTargetId,
                           enable_gfx9_t<GfxTargetId, (sizeof(ComputeT) < 4u)>>
            : public amdgcn_mfma<InputTA,
                                 InputTB,
                                 typename PackTraits<ComputeT>::PackedT,
                                 BlockM,
                                 BlockN,
                                 BlockK,
                                 GfxTargetId>
        {
        private:
            using PackTraits = PackTraits<ComputeT>;
            using PackUtil   = PackUtil<ComputeT>;

            // B32 mfma traits
            using AccumDataT = typename PackTraits::PackedT;
            using MfmaB32
                = amdgcn_mfma<InputTA, InputTB, AccumDataT, BlockM, BlockN, BlockK, GfxTargetId>;

            using AccumTraitsB32 = VecTraits<typename MfmaB32::CRegsT>;

            // Prevent public use of the base function
            using MfmaB32::exec;

            // ComputeT mfma traits
            // Scale accum registers by pack ratio, due to ComputeT
            using AccumRegsT = typename AccumTraitsB32::
                template VecT<AccumDataT, AccumTraitsB32::size() / PackTraits::PackRatio>;

        public:
            constexpr static uint32_t      KPerMma = MfmaB32::KPerMma;
            constexpr static MfmaCtrlFlags Cbsz    = MfmaB32::Cbsz;
            constexpr static MfmaCtrlFlags Abid    = MfmaB32::Abid;
            constexpr static MfmaCtrlFlags Blgp    = MfmaB32::Blgp;

            // Packed register types
            using ARegsT = typename MfmaB32::ARegsT;
            using BRegsT = typename MfmaB32::BRegsT;
            using CRegsT = AccumRegsT;
            using DRegsT = AccumRegsT;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // 'Packing' dichotomy:
                // - Registers arguments going in or out are assumed to
                //   be in 'packed' form (e.g., the physical space they consume).
                // - B32 native compute type is the 'packed' datatype;
                //   Desired compute type is the 'unpacked' datatype.
                using ConvertUp
                    = Convert<typename PackTraits::UnpackedT, typename PackTraits::PackedT>;
                using ConvertDown
                    = Convert<typename PackTraits::PackedT, typename PackTraits::UnpackedT>;

                auto unpacked_result
                    = MfmaB32::exec(regsA, regsB, ConvertUp::exec(PackUtil::unpack(regsC)));
                return PackUtil::pack(ConvertDown::exec(unpacked_result));
            }
        };

        // fp16
        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float16_t,
                           float16_t,
                           float32_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx9_t<GfxTargetId>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<float16_t, float16_t, float32_t, 16u, 16u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float16_t,
                           float16_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_gfx9_t<GfxTargetId>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_mfma_f32_16x16x16f16(to_native_vector(regsA),
                                                             to_native_vector(regsB),
                                                             to_native_vector(regsC),
                                                             (int)Cbsz,
                                                             (int)Abid,
                                                             (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float16_t,
                           float16_t,
                           float32_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_mfma_f32_16x16x32_f16(to_native_vector(regsA),
                                                              to_native_vector(regsB),
                                                              to_native_vector(regsC),
                                                              (int)Cbsz,
                                                              (int)Abid,
                                                              (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float16_t,
                           float16_t,
                           float32_t,
                           32u,
                           32u,
                           4u,
                           GfxTargetId,
                           enable_gfx9_t<GfxTargetId>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<float16_t, float16_t, float32_t, 32u, 32u, 8u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float16_t,
                           float16_t,
                           float32_t,
                           32u,
                           32u,
                           8u,
                           GfxTargetId,
                           enable_gfx9_t<GfxTargetId>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_mfma_f32_32x32x8f16(to_native_vector(regsA),
                                                            to_native_vector(regsB),
                                                            to_native_vector(regsC),
                                                            (int)Cbsz,
                                                            (int)Abid,
                                                            (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float16_t,
                           float16_t,
                           float32_t,
                           32u,
                           32u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_mfma_f32_32x32x16_f16(to_native_vector(regsA),
                                                              to_native_vector(regsB),
                                                              to_native_vector(regsC),
                                                              (int)Cbsz,
                                                              (int)Abid,
                                                              (int)Blgp)};
                return result;
            }
        };

        // hfloat16 derivative
        template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, uint32_t GfxTargetId>
        struct amdgcn_mfma<hfloat16_t,
                           hfloat16_t,
                           float32_t,
                           BlockM,
                           BlockN,
                           BlockK,
                           GfxTargetId,
                           enable_gfx9_t<GfxTargetId, !(bool)ROCWMMA_NO_HALF>>
            : public amdgcn_mfma<float16_t,
                                 float16_t,
                                 float32_t,
                                 BlockM,
                                 BlockN,
                                 BlockK,
                                 GfxTargetId>
        {
        };

        template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, uint32_t GfxTargetId>
        struct amdgcn_mfma<hfloat16_t,
                           hfloat16_t,
                           hfloat16_t,
                           BlockM,
                           BlockN,
                           BlockK,
                           GfxTargetId,
                           enable_gfx9_t<GfxTargetId, !(bool)ROCWMMA_NO_HALF>>
            : public amdgcn_mfma<float16_t,
                                 float16_t,
                                 float16_t,
                                 BlockM,
                                 BlockN,
                                 BlockK,
                                 GfxTargetId>
        {
        };

        // bf16
        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId,
                                              Constants::AMDGCN_ARCH_ID_GFX908,
                                              Constants::AMDGCN_ARCH_ID_GFX90A>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects unpacked vector of short.
                // Strange, but OK we can do that here.
                using TypeIn = VecT<short, 2>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_mfma_f32_16x16x8bf16(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    to_native_vector(regsC),
                    (int)Cbsz,
                    (int)Abid,
                    (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId,
                                              Constants::AMDGCN_ARCH_ID_GFX942,
                                              Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<bfloat16_t, bfloat16_t, float32_t, 16u, 16u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId,
                                              Constants::AMDGCN_ARCH_ID_GFX90A,
                                              Constants::AMDGCN_ARCH_ID_GFX942,
                                              Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(to_native_vector(regsA),
                                                                 to_native_vector(regsB),
                                                                 to_native_vector(regsC),
                                                                 (int)Cbsz,
                                                                 (int)Abid,
                                                                 (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_mfma_f32_16x16x32_bf16(to_native_vector(regsA),
                                                               to_native_vector(regsB),
                                                               to_native_vector(regsC),
                                                               (int)Cbsz,
                                                               (int)Abid,
                                                               (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           32u,
                           32u,
                           4u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId,
                                              Constants::AMDGCN_ARCH_ID_GFX908,
                                              Constants::AMDGCN_ARCH_ID_GFX90A>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects unpacked vector of short.
                // Strange, but OK we can do that here.
                using TypeIn = VecT<short, 2>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_mfma_f32_32x32x4bf16(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    to_native_vector(regsC),
                    (int)Cbsz,
                    (int)Abid,
                    (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           32u,
                           32u,
                           4u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId,
                                              Constants::AMDGCN_ARCH_ID_GFX942,
                                              Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<bfloat16_t, bfloat16_t, float32_t, 32u, 32u, 8u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           32u,
                           32u,
                           8u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId,
                                              Constants::AMDGCN_ARCH_ID_GFX90A,
                                              Constants::AMDGCN_ARCH_ID_GFX942,
                                              Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_mfma_f32_32x32x8bf16_1k(to_native_vector(regsA),
                                                                to_native_vector(regsB),
                                                                to_native_vector(regsC),
                                                                (int)Cbsz,
                                                                (int)Abid,
                                                                (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           32u,
                           32u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_mfma_f32_32x32x16_bf16(to_native_vector(regsA),
                                                               to_native_vector(regsB),
                                                               to_native_vector(regsC),
                                                               (int)Cbsz,
                                                               (int)Abid,
                                                               (int)Blgp)};
                return result;
            }
        };

        // fp32
        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float32_t,
                           float32_t,
                           float32_t,
                           16u,
                           16u,
                           4u,
                           GfxTargetId,
                           enable_gfx9_t<GfxTargetId>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_mfma_f32_16x16x4f32(to_native_vector(regsA)[0],
                                                            to_native_vector(regsB)[0],
                                                            to_native_vector(regsC),
                                                            (int)Cbsz,
                                                            (int)Abid,
                                                            (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float32_t,
                           float32_t,
                           float32_t,
                           32u,
                           32u,
                           2u,
                           GfxTargetId,
                           enable_gfx9_t<GfxTargetId>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_mfma_f32_32x32x2f32(to_native_vector(regsA)[0],
                                                            to_native_vector(regsB)[0],
                                                            to_native_vector(regsC),
                                                            (int)Cbsz,
                                                            (int)Abid,
                                                            (int)Blgp)};
                return result;
            }
        };

        // fp64
        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float64_t,
                           float64_t,
                           float64_t,
                           16u,
                           16u,
                           4u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId,
                                              Constants::AMDGCN_ARCH_ID_GFX90A,
                                              Constants::AMDGCN_ARCH_ID_GFX942,
                                              Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF64x1;
            using BRegsT = VRegF64x1;
            using CRegsT = AccRegF64x4;
            using DRegsT = AccRegF64x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_mfma_f64_16x16x4f64(to_native_vector(regsA)[0],
                                                            to_native_vector(regsB)[0],
                                                            to_native_vector(regsC),
                                                            (int)Cbsz,
                                                            (int)Abid,
                                                            (int)Blgp)};
                return result;
            }
        };

        // int8
        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<int8_t,
                           int8_t,
                           int32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId,
                                              Constants::AMDGCN_ARCH_ID_GFX908,
                                              Constants::AMDGCN_ARCH_ID_GFX90A>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegI32x1;
            using BRegsT = VRegI32x1;
            using CRegsT = AccRegI32x4;
            using DRegsT = AccRegI32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_mfma_i32_16x16x16i8(to_native_vector(regsA)[0],
                                                            to_native_vector(regsB)[0],
                                                            to_native_vector(regsC),
                                                            (int)Cbsz,
                                                            (int)Abid,
                                                            (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<int8_t,
                           int8_t,
                           int32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId,
                                              Constants::AMDGCN_ARCH_ID_GFX942,
                                              Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegI32x1;
            using BRegsT = VRegI32x1;
            using CRegsT = AccRegI32x4;
            using DRegsT = AccRegI32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<int8_t, int8_t, int32_t, 16u, 16u, 32u, GfxTargetId>::exec(
                    concat(regsA, ARegsT{0}),
                    concat(regsB, BRegsT{0}),
                    forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<int8_t,
                           int8_t,
                           int32_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId,
                                              Constants::AMDGCN_ARCH_ID_GFX942,
                                              Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegI32x2;
            using BRegsT = VRegI32x2;
            using CRegsT = AccRegI32x4;
            using DRegsT = AccRegI32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_mfma_i32_16x16x32_i8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA))[0],
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB))[0],
                    to_native_vector(regsC),
                    (int)Cbsz,
                    (int)Abid,
                    (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<int8_t,
                           int8_t,
                           int32_t,
                           16u,
                           16u,
                           64u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegI32x4;
            using BRegsT = VRegI32x4;
            using CRegsT = AccRegI32x4;
            using DRegsT = AccRegI32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {

                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_mfma_i32_16x16x64_i8(to_native_vector(regsA),
                                                             to_native_vector(regsB),
                                                             to_native_vector(regsC),
                                                             (int)Cbsz,
                                                             (int)Abid,
                                                             (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<int8_t,
                           int8_t,
                           int32_t,
                           32u,
                           32u,
                           8u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId,
                                              Constants::AMDGCN_ARCH_ID_GFX908,
                                              Constants::AMDGCN_ARCH_ID_GFX90A>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegI32x1;
            using BRegsT = VRegI32x1;
            using CRegsT = AccRegI32x16;
            using DRegsT = AccRegI32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_mfma_i32_32x32x8i8(to_native_vector(regsA)[0],
                                                           to_native_vector(regsB)[0],
                                                           to_native_vector(regsC),
                                                           (int)Cbsz,
                                                           (int)Abid,
                                                           (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<int8_t,
                           int8_t,
                           int32_t,
                           32u,
                           32u,
                           8u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId,
                                              Constants::AMDGCN_ARCH_ID_GFX942,
                                              Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegI32x1;
            using BRegsT = VRegI32x1;
            using CRegsT = AccRegI32x16;
            using DRegsT = AccRegI32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<int8_t, int8_t, int32_t, 32u, 32u, 16u, GfxTargetId>::exec(
                    concat(regsA, ARegsT{0}),
                    concat(regsB, BRegsT{0}),
                    forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<int8_t,
                           int8_t,
                           int32_t,
                           32u,
                           32u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId,
                                              Constants::AMDGCN_ARCH_ID_GFX942,
                                              Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegI32x2;
            using BRegsT = VRegI32x2;
            using CRegsT = AccRegI32x16;
            using DRegsT = AccRegI32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_mfma_i32_32x32x16_i8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA))[0],
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB))[0],
                    to_native_vector(regsC),
                    (int)Cbsz,
                    (int)Abid,
                    (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<int8_t,
                           int8_t,
                           int32_t,
                           32u,
                           32u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegI32x4;
            using BRegsT = VRegI32x4;
            using CRegsT = AccRegI32x16;
            using DRegsT = AccRegI32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_mfma_i32_32x32x32_i8(to_native_vector(regsA),
                                                             to_native_vector(regsB),
                                                             to_native_vector(regsC),
                                                             (int)Cbsz,
                                                             (int)Abid,
                                                             (int)Blgp)};
                return result;
            }
        };

        // f8_fnuz
        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float8_fnuz_t,
                           float8_fnuz_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX942>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<float8_fnuz_t,
                                   float8_fnuz_t,
                                   float32_t,
                                   16u,
                                   16u,
                                   32u,
                                   GfxTargetId>::exec(concat(regsA, ARegsT{0}),
                                                      concat(regsB, BRegsT{0}),
                                                      forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float8_fnuz_t,
                           float8_fnuz_t,
                           float32_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX942>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA))[0],
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB))[0],
                    to_native_vector(regsC),
                    (int)Cbsz,
                    (int)Abid,
                    (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float8_fnuz_t,
                           float8_fnuz_t,
                           float32_t,
                           32u,
                           32u,
                           8u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX942>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<float8_fnuz_t,
                                   float8_fnuz_t,
                                   float32_t,
                                   32u,
                                   32u,
                                   16u,
                                   GfxTargetId>::exec(concat(regsA, ARegsT{0}),
                                                      concat(regsB, BRegsT{0}),
                                                      forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float8_fnuz_t,
                           float8_fnuz_t,
                           float32_t,
                           32u,
                           32u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX942>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA))[0],
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB))[0],
                    to_native_vector(regsC),
                    (int)Cbsz,
                    (int)Abid,
                    (int)Blgp)};
                return result;
            }
        };

        // bf8_fnuz
        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat8_fnuz_t,
                           bfloat8_fnuz_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX942>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<bfloat8_fnuz_t,
                                   bfloat8_fnuz_t,
                                   float32_t,
                                   16u,
                                   16u,
                                   32u,
                                   GfxTargetId>::exec(concat(regsA, ARegsT{0}),
                                                      concat(regsB, BRegsT{0}),
                                                      forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat8_fnuz_t,
                           bfloat8_fnuz_t,
                           float32_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX942>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA))[0],
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB))[0],
                    to_native_vector(regsC),
                    (int)Cbsz,
                    (int)Abid,
                    (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat8_fnuz_t,
                           bfloat8_fnuz_t,
                           float32_t,
                           32u,
                           32u,
                           8u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX942>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<bfloat8_fnuz_t,
                                   bfloat8_fnuz_t,
                                   float32_t,
                                   32u,
                                   32u,
                                   16u,
                                   GfxTargetId>::exec(concat(regsA, ARegsT{0}),
                                                      concat(regsB, BRegsT{0}),
                                                      forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat8_fnuz_t,
                           bfloat8_fnuz_t,
                           float32_t,
                           32u,
                           32u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX942>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA))[0],
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB))[0],
                    to_native_vector(regsC),
                    (int)Cbsz,
                    (int)Abid,
                    (int)Blgp)};
                return result;
            }
        };

        // Mixed f8_fnuz / bf8_fnuz
        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float8_fnuz_t,
                           bfloat8_fnuz_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX942>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<float8_fnuz_t,
                                   bfloat8_fnuz_t,
                                   float32_t,
                                   16u,
                                   16u,
                                   32u,
                                   GfxTargetId>::exec(concat(regsA, ARegsT{0}),
                                                      concat(regsB, BRegsT{0}),
                                                      forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float8_fnuz_t,
                           bfloat8_fnuz_t,
                           float32_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX942>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA))[0],
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB))[0],
                    to_native_vector(regsC),
                    (int)Cbsz,
                    (int)Abid,
                    (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float8_fnuz_t,
                           bfloat8_fnuz_t,
                           float32_t,
                           32u,
                           32u,
                           8u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX942>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<float8_fnuz_t,
                                   bfloat8_fnuz_t,
                                   float32_t,
                                   32u,
                                   32u,
                                   16u,
                                   GfxTargetId>::exec(concat(regsA, ARegsT{0}),
                                                      concat(regsB, BRegsT{0}),
                                                      forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float8_fnuz_t,
                           bfloat8_fnuz_t,
                           float32_t,
                           32u,
                           32u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX942>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA))[0],
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB))[0],
                    to_native_vector(regsC),
                    (int)Cbsz,
                    (int)Abid,
                    (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat8_fnuz_t,
                           float8_fnuz_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX942>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<bfloat8_fnuz_t,
                                   float8_fnuz_t,
                                   float32_t,
                                   16u,
                                   16u,
                                   32u,
                                   GfxTargetId>::exec(concat(regsA, ARegsT{0}),
                                                      concat(regsB, BRegsT{0}),
                                                      forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat8_fnuz_t,
                           float8_fnuz_t,
                           float32_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX942>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA))[0],
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB))[0],
                    to_native_vector(regsC),
                    (int)Cbsz,
                    (int)Abid,
                    (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat8_fnuz_t,
                           float8_fnuz_t,
                           float32_t,
                           32u,
                           32u,
                           8u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX942>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<bfloat8_fnuz_t,
                                   float8_fnuz_t,
                                   float32_t,
                                   32u,
                                   32u,
                                   16u,
                                   GfxTargetId>::exec(concat(regsA, ARegsT{0}),
                                                      concat(regsB, BRegsT{0}),
                                                      forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat8_fnuz_t,
                           float8_fnuz_t,
                           float32_t,
                           32u,
                           32u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX942>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA))[0],
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB))[0],
                    to_native_vector(regsC),
                    (int)Cbsz,
                    (int)Abid,
                    (int)Blgp)};
                return result;
            }
        };

        // fp8
        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float8_t,
                           float8_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<float8_t, float8_t, float32_t, 16u, 16u, 32u, GfxTargetId>::exec(
                    concat(regsA, ARegsT{0}),
                    concat(regsB, BRegsT{0}),
                    forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float8_t,
                           float8_t,
                           float32_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA))[0],
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB))[0],
                    to_native_vector(regsC),
                    (int)Cbsz,
                    (int)Abid,
                    (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float8_t,
                           float8_t,
                           float32_t,
                           32u,
                           32u,
                           8u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<float8_t, float8_t, float32_t, 32u, 32u, 16u, GfxTargetId>::exec(
                    concat(regsA, ARegsT{0}),
                    concat(regsB, BRegsT{0}),
                    forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float8_t,
                           float8_t,
                           float32_t,
                           32u,
                           32u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA))[0],
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB))[0],
                    to_native_vector(regsC),
                    (int)Cbsz,
                    (int)Abid,
                    (int)Blgp)};
                return result;
            }
        };

        // bf8
        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat8_t,
                           bfloat8_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<bfloat8_t, bfloat8_t, float32_t, 16u, 16u, 32u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat8_t,
                           bfloat8_t,
                           float32_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA))[0],
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB))[0],
                    to_native_vector(regsC),
                    (int)Cbsz,
                    (int)Abid,
                    (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat8_t,
                           bfloat8_t,
                           float32_t,
                           32u,
                           32u,
                           8u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<bfloat8_t, bfloat8_t, float32_t, 32u, 32u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat8_t,
                           bfloat8_t,
                           float32_t,
                           32u,
                           32u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA))[0],
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB))[0],
                    to_native_vector(regsC),
                    (int)Cbsz,
                    (int)Abid,
                    (int)Blgp)};
                return result;
            }
        };

        // Mixed f8 / bf8
        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float8_t,
                           bfloat8_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<float8_t, bfloat8_t, float32_t, 16u, 16u, 32u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float8_t,
                           bfloat8_t,
                           float32_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA))[0],
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB))[0],
                    to_native_vector(regsC),
                    (int)Cbsz,
                    (int)Abid,
                    (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float8_t,
                           bfloat8_t,
                           float32_t,
                           32u,
                           32u,
                           8u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<float8_t, bfloat8_t, float32_t, 32u, 32u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<float8_t,
                           bfloat8_t,
                           float32_t,
                           32u,
                           32u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA))[0],
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB))[0],
                    to_native_vector(regsC),
                    (int)Cbsz,
                    (int)Abid,
                    (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat8_t,
                           float8_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<bfloat8_t, float8_t, float32_t, 16u, 16u, 32u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat8_t,
                           float8_t,
                           float32_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA))[0],
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB))[0],
                    to_native_vector(regsC),
                    (int)Cbsz,
                    (int)Abid,
                    (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat8_t,
                           float8_t,
                           float32_t,
                           32u,
                           32u,
                           8u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<bfloat8_t, float8_t, float32_t, 32u, 32u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<bfloat8_t,
                           float8_t,
                           float32_t,
                           32u,
                           32u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX950>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA))[0],
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB))[0],
                    to_native_vector(regsC),
                    (int)Cbsz,
                    (int)Abid,
                    (int)Blgp)};
                return result;
            }
        };

        // xf32
        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<xfloat32_t,
                           xfloat32_t,
                           float32_t,
                           16u,
                           16u,
                           4u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX942>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<xfloat32_t, xfloat32_t, float32_t, 16u, 16u, 8u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<xfloat32_t,
                           xfloat32_t,
                           float32_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX942>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_mfma_f32_16x16x8_xf32(to_native_vector(regsA),
                                                              to_native_vector(regsB),
                                                              to_native_vector(regsC),
                                                              (int)Cbsz,
                                                              (int)Abid,
                                                              (int)Blgp)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<xfloat32_t,
                           xfloat32_t,
                           float32_t,
                           32u,
                           32u,
                           2u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX942>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Pad with 0s
                return amdgcn_mfma<xfloat32_t, xfloat32_t, float32_t, 32u, 32u, 4u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_mfma<xfloat32_t,
                           xfloat32_t,
                           float32_t,
                           32u,
                           32u,
                           4u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX942>>
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_mfma_f32_32x32x4_xf32(to_native_vector(regsA),
                                                              to_native_vector(regsB),
                                                              to_native_vector(regsC),
                                                              (int)Cbsz,
                                                              (int)Abid,
                                                              (int)Blgp)};
                return result;
            }
        };

    } // namespace detail

    namespace MmaTraits_impl
    {
        template <typename MfmaOp>
        struct is_mfma : public false_type
        {
        };

        template <typename InputTA_In,
                  typename InputTB_In,
                  typename ComputeT_in,
                  uint32_t BlockM_In,
                  uint32_t BlockN_In,
                  uint32_t BlockK_In>
        struct is_mfma<
            detail::
                amdgcn_mfma<InputTA_In, InputTB_In, ComputeT_in, BlockM_In, BlockN_In, BlockK_In>>
            : public true_type
        {
        };

        template <typename MfmaOp>
        constexpr static bool is_mfma_v = is_mfma<MfmaOp>::value;

        // All of the overrides won't have the Unsupported tag
        template <typename MfmaOp, typename Enabler = void>
        struct is_mfma_supported : public true_type
        {
        };

        // Default implementation will have the Unsupported tag
        template <typename MfmaOp>
        struct is_mfma_supported<
            MfmaOp,
            enable_if_t<is_same_v<typename MfmaOp::Unsupported, detail::Unsupported>>>
            : public false_type
        {
        };

        template <typename MfmaOp>
        constexpr static bool is_mfma_supported_v = is_mfma_supported<MfmaOp>::value;

        template <typename MfmaOp>
        struct mfma_traits;

        template <typename InputTA_In,
                  typename InputTB_In,
                  typename ComputeT_In,
                  uint32_t BlockM_In,
                  uint32_t BlockN_In,
                  uint32_t BlockK_In>
        struct mfma_traits<
            detail::
                amdgcn_mfma<InputTA_In, InputTB_In, ComputeT_In, BlockM_In, BlockN_In, BlockK_In>>
        {
            // Base implementation
            using Impl = detail::
                amdgcn_mfma<InputTA_In, InputTB_In, ComputeT_In, BlockM_In, BlockN_In, BlockK_In>;

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
            constexpr static bool is_wmma      = false;
            constexpr static bool is_mfma      = is_mfma_v<Impl>;
            constexpr static bool is_supported = is_mfma_supported_v<Impl>;
        };

        // MmaTraits implemented for mfma backend
        template <typename MmaOp>
        struct MmaTraits<MmaOp, enable_if_t<is_mfma_v<MmaOp>>> : public mfma_traits<MmaOp>
        {
        };

    } // namespace MmaTraits_impl

} // namespace rocwmma

#endif // ROCWMMA_MFMA_IMPL_HPP
