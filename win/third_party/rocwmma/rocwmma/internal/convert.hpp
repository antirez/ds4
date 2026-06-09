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
#ifndef ROCWMMA_CONVERT_HPP
#define ROCWMMA_CONVERT_HPP

#include "types.hpp"
#include "utility/forward.hpp"
#include "utility/get.hpp"
#include "utility/static_for.hpp"

namespace rocwmma
{

    namespace detail
    {

        template <typename InputT, typename OutputT>
        struct amdgcn_convert
        {
            template <template <typename, uint32_t> class VecT, uint32_t NumRegs>
            ROCWMMA_DEVICE static inline auto exec(VecT<InputT, NumRegs> const& regsIn)
                -> VecT<OutputT, NumRegs>
            {
                VecT<OutputT, NumRegs> result;

                rocwmma::static_for<0u, NumRegs, 1u>(
                    [&](auto i) { get<i>(result) = static_cast<OutputT>(get<i>(regsIn)); });

                return result;
            }
        };

        template <typename T>
        struct amdgcn_convert<T, T>
        {
            template <typename IncomingT>
            ROCWMMA_DEVICE static inline decltype(auto) exec(IncomingT&& regsIn)
            {
                return forward<IncomingT>(regsIn);
            }
        };

#if !ROCWMMA_NO_HALF
        template <>
        struct amdgcn_convert<hfloat16_t, float32_t>
        {
            template <template <typename, uint32_t> class VecT, uint32_t NumRegs>
            ROCWMMA_DEVICE static inline auto exec(VecT<hfloat16_t, NumRegs> const& regsIn)
            {
                VecT<float32_t, NumRegs> result;

                rocwmma::static_for<0u, NumRegs, 1u>(
                    [&](auto i) { get<i>(result) = __half2float(get<i>(regsIn)); });

                return result;
            }
        };

        template <>
        struct amdgcn_convert<float32_t, hfloat16_t>
        {
            template <template <typename, uint32_t> class VecT, uint32_t NumRegs>
            ROCWMMA_DEVICE static inline auto exec(VecT<float32_t, NumRegs> const& regsIn)
            {
                VecT<hfloat16_t, NumRegs> result;

                rocwmma::static_for<0u, NumRegs, 1u>(
                    [&](auto i) { get<i>(result) = __float2half(get<i>(regsIn)); });

                return result;
            }
        };

#endif // !ROCWMMA_NO_HALF

    } // namespace detail

    template <typename InputT, typename OutputT>
    using Convert = detail::amdgcn_convert<InputT, OutputT>;

} // namespace rocwmma

#endif // ROCWMMA_CONVERT_HPP
