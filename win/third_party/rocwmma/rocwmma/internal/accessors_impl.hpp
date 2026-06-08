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
#ifndef ROCWMMA_ACCESSORS_IMPL_HPP
#define ROCWMMA_ACCESSORS_IMPL_HPP

#include "accessors.hpp"
#include "api_fwd.hpp"
#include "fragment_traits.hpp"
#include "io_config.hpp"
#include "io_scheduler.hpp"
#include "io_shape.hpp"
#include "mma_config.hpp"

namespace rocwmma
{
    template <typename FragT>
    struct GetDataType;

    template <typename MatrixT,
              uint32_t FragM,
              uint32_t FragN,
              uint32_t FragK,
              typename DataT,
              typename DataLayoutT,
              typename Scheduler>
    struct GetDataType<fragment<MatrixT, FragM, FragN, FragK, DataT, DataLayoutT, Scheduler>>
    {
        using type = DataT;
    };

    template <typename FragT>
    struct GetIOConfig;

    template <typename MatrixT,
              uint32_t FragM,
              uint32_t FragN,
              uint32_t FragK,
              typename DataT,
              typename DataLayoutT,
              typename Scheduler>
    struct GetIOConfig<fragment<MatrixT, FragM, FragN, FragK, DataT, DataLayoutT, Scheduler>>
    {
        using type = IOConfig<MatrixT, FragM, FragN, FragK, DataT, DataLayoutT, Scheduler>;
    };

    template <typename FragA, typename FragB, typename FragC, typename FragD>
    struct GetMmaConfig
    {
        using type = MmaConfig<FragA, FragB, FragC, FragD>;
    };

    template <typename FragT>
    struct GetIOShape
    {
    private:
        using IOConfig = typename GetIOConfig<FragT>::type;

    public:
        using type = typename IOConfig::IOShape;
    };

    template <typename FragT>
    struct GetDataLayout
    {
    private:
        using IOConfig = typename GetIOConfig<FragT>::type;

    public:
        using type = typename IOConfig::IOLayout::DataLayout;
    };

    template <typename FragT>
    struct GetMappingUtil
    {
    private:
        using IOConfig = typename GetIOConfig<FragT>::type;

    public:
        using type = typename IOConfig::MappingUtil;
    };

} // namespace rocwmma

#endif // ROCWMMA_ACCESSORS_IMPL_HPP
