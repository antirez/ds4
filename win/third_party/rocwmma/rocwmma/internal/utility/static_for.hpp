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

#ifndef ROCWMMA_UTILITY_STATIC_FOR_HPP
#define ROCWMMA_UTILITY_STATIC_FOR_HPP

#include "static_for_impl.hpp"

namespace rocwmma
{
    //! Statically unrolls an iterative for-loop sequence [from i = NBegin; i < NEnd; i += Increment]
    //! Each iteration will invoke a Functor F with signature: F(I<i> it, args...), where I<i> is a
    //! compile-time value of the current iterator i. It may be used with constexpr checks, where
    //! you may access the value by decay_t<decltype(it)>::value
    //! @param f Functor Functor F with signature: F(I<i> it, args...), where I<i> the iterator
    //! @param args Additional arguments that are passed to functor
    //! @tparam NBegin The start iterator value
    //! @tparam NEnd The end iterator value
    //! @tparam Increment The space between each iteration
    //! @tparam F the type of input functor f
    //! @tparam ArgsT the variadic list of additional input args to functor f.
    template <index_t NBegin, index_t NEnd, index_t Increment, class F, typename... ArgsT>
    ROCWMMA_HOST_DEVICE constexpr inline void static_for(F f, ArgsT&&... args)
    {
        detail::static_for<NBegin, NEnd, Increment>()(f, forward<ArgsT>(args)...);
    }
}

#endif // ROCWMMA_UTILITY_STATIC_FOR_HPP
