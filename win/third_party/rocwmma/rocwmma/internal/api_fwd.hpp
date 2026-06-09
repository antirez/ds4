/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc.
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
#ifndef ROCWMMA_INTERNAL_API_FWD_HPP
#define ROCWMMA_INTERNAL_API_FWD_HPP

#if !defined(__align__)
#define __align__(x) __attribute__((aligned(x)))
#endif

namespace rocwmma
{
    // Fwd declare API tags
    struct col_major;
    struct row_major;
    struct matrix_a;
    struct matrix_b;
    struct accumulator;

    template <typename MatrixT,
              unsigned int FragM,
              unsigned int FragN,
              unsigned int FragK,
              typename DataT,
              typename DataLayoutT,
              typename Scheduler>
    class __align__(4) fragment;

} // namespace rocwmma

#endif // ROCWMMA_INTERNAL_API_FWD_HPP
