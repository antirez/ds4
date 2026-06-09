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

#ifndef ROCWMMA_UTILITY_NUMERIC_LIMITS_IMPL_HPP
#define ROCWMMA_UTILITY_NUMERIC_LIMITS_IMPL_HPP

namespace rocwmma
{
    namespace detail
    {
        enum float_round_style
        {
            round_indeterminate       = -1,
            round_toward_zero         = 0,
            round_to_nearest          = 1,
            round_toward_infinity     = 2,
            round_toward_neg_infinity = 3
        };

        // Currently does not have implementation as there is no current
        // library needs for regular arithmetic types.
        // Specializations do exist for f8, bf8 and xf32 types where they
        // are currently defined.
        template <typename T>
        class numeric_limits
        {
        public:
            ROCWMMA_HOST_DEVICE static constexpr T min() noexcept;
            ROCWMMA_HOST_DEVICE static constexpr T lowest() noexcept;
            ROCWMMA_HOST_DEVICE static constexpr T max() noexcept;
            ROCWMMA_HOST_DEVICE static constexpr T epsilon() noexcept;
            ROCWMMA_HOST_DEVICE static constexpr T round_error() noexcept;
            ROCWMMA_HOST_DEVICE static constexpr T infinity() noexcept;
            ROCWMMA_HOST_DEVICE static constexpr T quiet_NaN() noexcept;
            ROCWMMA_HOST_DEVICE static constexpr T signaling_NaN() noexcept;
            ROCWMMA_HOST_DEVICE static constexpr T denorm_min() noexcept;
        };

    } // namespace detail

} // namespace rocwmma

#endif // ROCWMMA_UTILITY_NUMERIC_LIMITS_IMPL_HPP
