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
#ifndef ROCWMMA_ACCESSORS_HPP
#define ROCWMMA_ACCESSORS_HPP

#include "accessors_impl.hpp"

namespace rocwmma
{
    ///
    /// DataType access
    ///

    template <typename FragT>
    struct GetDataType;

    template <typename FragT>
    using GetDataType_t = typename GetDataType<FragT>::type;

    ///
    /// IOConfig access
    ///

    template <typename FragT>
    struct GetIOConfig;

    template <typename FragT>
    using GetIOConfig_t = typename GetIOConfig<FragT>::type;

    template <typename FragA, typename FragB, typename FragC, typename FragD>
    struct GetMmaConfig;

    template <typename FragA, typename FragB, typename FragC, typename FragD>
    using GetMmaConfig_t = typename GetMmaConfig<FragA, FragB, FragC, FragD>::type;

    ///
    /// IOShape access
    ///

    template <typename FragT>
    struct GetIOShape;

    template <typename FragT>
    using GetIOShape_t = typename GetIOShape<FragT>::type;

    ///
    /// DataLayout access
    ///

    template <typename FragT>
    struct GetDataLayout;

    template <typename FragT>
    using GetDataLayout_t = typename GetDataLayout<FragT>::type;

    ///
    /// MappingUtil access
    ///

    template <typename FragT>
    struct GetMappingUtil;

    template <typename FragT>
    using GetMappingUtil_t = typename GetMappingUtil<FragT>::type;

} // namespace rocwmma

#endif // ROCWMMA_ACCESSORS_HPP
