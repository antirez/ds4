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

#ifndef ROCWMMA_UTILITY_STATIC_FOR_IMPL_HPP
#define ROCWMMA_UTILITY_STATIC_FOR_IMPL_HPP

#include "../types.hpp"
#include "sequence.hpp"

namespace rocwmma
{
    namespace detail
    {

        template <index_t NBegin, index_t NEnd, index_t Increment>
        struct static_for
        {
            ROCWMMA_HOST_DEVICE constexpr static_for()
            {
                static_assert(Increment != 0 && (NEnd - NBegin) % Increment == 0,
                              "Sequence length is not divisible by Increment");
                static_assert((Increment > 0 && NBegin <= NEnd)
                                  || (Increment < 0 && NBegin >= NEnd),
                              "Invalid inputs");
            }

        private:
            // F signature: F(I<Iter>)
            template <class F, size_t... Indices, typename... ArgsT>
            ROCWMMA_HOST_DEVICE constexpr void
                impl(F f, index_sequence<Indices...>, ArgsT&&... args) const
            {
                ((f(I<Indices>{}, forward<ArgsT>(args)...), 0), ...);
            }

        public:
            // F signature: F(I<Iter>)
            template <class F, typename... ArgsT>
            ROCWMMA_HOST_DEVICE constexpr void operator()(F f, ArgsT&&... args) const
            {
                impl(f,
                     make_offset_index_sequence<(NEnd - NBegin) / Increment, NBegin, Increment>(),
                     forward<ArgsT>(args)...);
            }
        };

    } // namespace detail

} // namespace rocwmma

#endif // ROCWMMA_UTILITY_STATIC_FOR_IMPL_HPP
