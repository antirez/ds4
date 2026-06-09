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
#ifndef ROCWMMA_CONSTANTS_HPP
#define ROCWMMA_CONSTANTS_HPP

#include "config.hpp"

namespace rocwmma
{
    struct Constants
    {
        ///
        /// Architecture IDs
        ///
        static constexpr uint32_t AMDGCN_ARCH_ID_GFX908  = 0x0908;
        static constexpr uint32_t AMDGCN_ARCH_ID_GFX90A  = 0x090A;
        static constexpr uint32_t AMDGCN_ARCH_ID_GFX942  = 0x0942;
        static constexpr uint32_t AMDGCN_ARCH_ID_GFX950  = 0x0950;
        static constexpr uint32_t AMDGCN_ARCH_ID_GFX1100 = 0x1100;
        static constexpr uint32_t AMDGCN_ARCH_ID_GFX1101 = 0x1101;
        static constexpr uint32_t AMDGCN_ARCH_ID_GFX1102 = 0x1102;
        static constexpr uint32_t AMDGCN_ARCH_ID_GFX1151 = 0x1151;
        static constexpr uint32_t AMDGCN_ARCH_ID_GFX1200 = 0x1200;
        static constexpr uint32_t AMDGCN_ARCH_ID_GFX1201 = 0x1201;
        static constexpr uint32_t AMDGCN_ARCH_ID_NONE    = 0x0000;

        ///
        /// Wave sizes
        ///
        static constexpr uint32_t AMDGCN_WAVE_SIZE_64   = 64u;
        static constexpr uint32_t AMDGCN_WAVE_SIZE_32   = 32u;
        static constexpr uint32_t AMDGCN_WAVE_SIZE_NONE = 0u;

        ///
        /// Architecture ID currently being compiled
        ///
#if ROCWMMA_ARCH_GFX908
        static constexpr uint32_t AMDGCN_CURRENT_ARCH_ID = AMDGCN_ARCH_ID_GFX908;
#elif ROCWMMA_ARCH_GFX90A
        static constexpr uint32_t AMDGCN_CURRENT_ARCH_ID = AMDGCN_ARCH_ID_GFX90A;
#elif ROCWMMA_ARCH_GFX942
        static constexpr uint32_t AMDGCN_CURRENT_ARCH_ID = AMDGCN_ARCH_ID_GFX942;
#elif ROCWMMA_ARCH_GFX950
        static constexpr uint32_t AMDGCN_CURRENT_ARCH_ID = AMDGCN_ARCH_ID_GFX950;
#elif ROCWMMA_ARCH_GFX1100
        static constexpr uint32_t AMDGCN_CURRENT_ARCH_ID = AMDGCN_ARCH_ID_GFX1100;
#elif ROCWMMA_ARCH_GFX1101
        static constexpr uint32_t AMDGCN_CURRENT_ARCH_ID = AMDGCN_ARCH_ID_GFX1101;
#elif ROCWMMA_ARCH_GFX1102
        static constexpr uint32_t AMDGCN_CURRENT_ARCH_ID = AMDGCN_ARCH_ID_GFX1102;
#elif ROCWMMA_ARCH_GFX1151
        static constexpr uint32_t AMDGCN_CURRENT_ARCH_ID = AMDGCN_ARCH_ID_GFX1151;
#elif ROCWMMA_ARCH_GFX1200
        static constexpr uint32_t AMDGCN_CURRENT_ARCH_ID = AMDGCN_ARCH_ID_GFX1200;
#elif ROCWMMA_ARCH_GFX1201
        static constexpr uint32_t AMDGCN_CURRENT_ARCH_ID = AMDGCN_ARCH_ID_GFX1201;
#else
        static constexpr uint32_t AMDGCN_CURRENT_ARCH_ID = AMDGCN_ARCH_ID_NONE;
#endif

        ///
        /// Constants for architecture currently being compiled
        ///
#if ROCWMMA_WAVE64_MODE
        static constexpr uint32_t AMDGCN_WAVE_SIZE = AMDGCN_WAVE_SIZE_64;
#elif ROCWMMA_WAVE32_MODE
        static constexpr uint32_t AMDGCN_WAVE_SIZE       = AMDGCN_WAVE_SIZE_32;
#else // Host default to 64 to avoid host compile time asserts.
        static constexpr uint32_t AMDGCN_WAVE_SIZE       = AMDGCN_WAVE_SIZE_64;
#endif

        static constexpr uint32_t AMDGCN_REGISTER_ELEMENT_SIZE_BYTES = 4u;
        static constexpr uint32_t AMDGCN_REGISTER_SIZE_BYTES
            = AMDGCN_REGISTER_ELEMENT_SIZE_BYTES * AMDGCN_WAVE_SIZE;

        static constexpr uint32_t AMDGCN_LDS_MAX_SIZE_BYTES    = 65536u;
        static constexpr uint32_t AMDGCN_CACHE_LINE_SIZE_BYTES = 64u;
        static constexpr uint32_t AMDGCN_DWORD_SIZE_BYTES      = 4u;
    };

} // namespace rocwmma

#endif // ROCWMMA_CONSTANTS_HPP
