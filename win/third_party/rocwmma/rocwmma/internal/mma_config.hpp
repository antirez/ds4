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
#ifndef ROCWMMA_MMA_CONFIG_HPP
#define ROCWMMA_MMA_CONFIG_HPP

#include "accessors.hpp"
#include "io_config.hpp"
#include "layout/register_layout_transforms.hpp"
#include "mfma.hpp"
#include "pack_util.hpp"
#include "types.hpp"
#include "wmma.hpp"

namespace rocwmma
{
    template <typename FragT>
    struct GetIOConfig;

    // The purpose of this class is to interface from fragment level traits
    // into specific Mma operations.
    // Mma operations can be classified into transforms (e.g., pre / post mma)
    // and mma function hooks.
    template <typename FragA, typename FragB, typename FragC, typename FragD>
    struct MmaConfig
    {
        using FragATraits = fragment_traits<FragA>;
        using FragBTraits = fragment_traits<FragB>;
        using FragCTraits = fragment_traits<FragC>;
        using FragDTraits = fragment_traits<FragD>;

        // Sanity checks fragment dimensions
        static_assert((FragATraits::FragM == FragBTraits::FragM)
                          && (FragBTraits::FragM == FragCTraits::FragM)
                          && (FragCTraits::FragM == FragDTraits::FragM),
                      "Mma fragment FragM traits must match");
        static_assert((FragATraits::FragN == FragBTraits::FragN)
                          && (FragBTraits::FragN == FragCTraits::FragN)
                          && (FragCTraits::FragN == FragDTraits::FragN),
                      "Mma fragment FragN traits must match");
        static_assert((FragATraits::FragK == FragBTraits::FragK)
                          && (FragBTraits::FragK == FragCTraits::FragK)
                          && (FragCTraits::FragK == FragDTraits::FragK),
                      "Mma fragment FragK traits must match");

        using InputTA  = typename FragATraits::DataT;
        using InputTB  = typename FragBTraits::DataT;
        using ComputeT = typename FragCTraits::DataT;

        // Sanity check datatypes
        static_assert(sizeof(InputTA) == sizeof(InputTB), "Input datatypes must be same size");
        static_assert(is_same_v<typename FragCTraits::DataT, typename FragDTraits::DataT>,
                      "Accum fragments C and D must have the same type");

        // Fragment dimensions are constant
        static constexpr uint32_t FragM = FragATraits::FragM;
        static constexpr uint32_t FragN = FragATraits::FragN;
        static constexpr uint32_t FragK = FragATraits::FragK;

        using SchedulerA = typename FragATraits::Scheduler;
        using SchedulerB = typename FragBTraits::Scheduler;
        using SchedulerC = typename FragCTraits::Scheduler;
        using SchedulerD = typename FragDTraits::Scheduler;

        // Sanity check schedulers
        static_assert(
            is_same_v<
                SchedulerA,
                SchedulerB> && is_same_v<SchedulerB, SchedulerC> && is_same_v<SchedulerC, SchedulerD>,
            "Mma fragment scheduler traits must match");

        using SchedulerTraits = scheduler_traits<SchedulerA>;

        static_assert(!SchedulerTraits::is_cooperative && SchedulerTraits::WaveCount == 1u,
                      "Mma does not support cooperative fragments");

        using IOConfigA = typename GetIOConfig<FragA>::type;
        using IOConfigB = typename GetIOConfig<FragB>::type;
        using IOConfigC = typename GetIOConfig<FragC>::type;
        using IOConfigD = typename GetIOConfig<FragD>::type;

        using IOLayoutA = typename IOConfigA::IOLayout;
        using IOLayoutB = typename IOConfigB::IOLayout;
        using IOLayoutC = typename IOConfigC::IOLayout;
        using IOLayoutD = typename IOConfigD::IOLayout;

        // Sanity mma layouts
        static_assert(
            is_layout_same_v<typename IOLayoutA::MmaLayout, typename IOLayoutB::MmaLayout>,
            "Input fragment register layouts do not match");
        static_assert(
            is_layout_same_v<typename IOLayoutC::MmaLayout, typename IOLayoutD::MmaLayout>,
            "Accumulator fragment register layouts do not match");

        // Check valid mma layouts
        // TODO: eventually should enforce
        // static_assert(layout_traits<typename IOLayoutA::MmaLayout>::is_valid, "Invalid MmaLayout for matrix_a");
        // static_assert(layout_traits<typename IOLayoutB::MmaLayout>::is_valid, "Invalid MmaLayout for matrix_b");
        // static_assert(layout_traits<typename IOLayoutC::MmaLayout>::is_valid, "Invalid MmaLayout for accumulator C");
        // static_assert(layout_traits<typename IOLayoutD::MmaLayout>::is_valid, "Invalid MmaLayout for accumulator D");

        // Input transforms
        using PreMmaXFormA = register_layout_transform<typename IOLayoutA::FragmentLayout,
                                                       typename IOLayoutA::MmaLayout>;

        using PreMmaXFormB = register_layout_transform<typename IOLayoutB::FragmentLayout,
                                                       typename IOLayoutB::MmaLayout>;

        using PreMmaXFormC = register_layout_transform<typename IOLayoutC::FragmentLayout,
                                                       typename IOLayoutC::MmaLayout>;

        // Output accum transform
        using PostMmaXFormD = register_layout_transform<typename IOLayoutD::MmaLayout,
                                                        typename IOLayoutD::FragmentLayout>;

        // Pack util
        using PackB = typename IOConfigB::PackUtil;
        using PackA = typename IOConfigA::PackUtil;
        using PackC = typename IOConfigC::PackUtil;
        using PackD = typename IOConfigD::PackUtil;

        // Mma block size selections come from the layout config
        // BlockK selection will come from the Mma classes below
        constexpr static uint32_t MmaDimM = IOLayoutA::MmaDim;
        constexpr static uint32_t MmaDimN = IOLayoutB::MmaDim;

        // Sanity checks:
        // - For now, MmaDimM/N/Acc must match
        // - MmaLayout for input A/B must match
        // - MmaLayout for accumulators must match
        static_assert(MmaDimM == MmaDimN, "MmaDims must match");
        static_assert((MmaDimN == IOLayoutC::MmaDim) && (MmaDimN == IOLayoutD::MmaDim),
                      "Mismatched accumulator MmaDim");

        // Ensure to use padded tiles if necessary
        using IOTile = IOTile<FragM, FragN, FragK, InputTA>;

        // Gfx9 uses MFMA, gfx11/12 uses WMMA
        using Mma = conditional_t<(bool)ROCWMMA_ARCH_GFX9,
                                  Mfma<IOTile::BlockM,
                                       IOTile::BlockN,
                                       IOTile::BlockK,
                                       InputTA,
                                       InputTB,
                                       ComputeT,
                                       MmaDimM,
                                       MmaDimN>,
                                  Wmma<IOTile::BlockM,
                                       IOTile::BlockN,
                                       IOTile::BlockK,
                                       InputTA,
                                       InputTB,
                                       ComputeT,
                                       MmaDimM,
                                       MmaDimN>>;
    };

} // namespace rocwmma

#endif // ROCWMMA_MMA_CONFIG_HPP
