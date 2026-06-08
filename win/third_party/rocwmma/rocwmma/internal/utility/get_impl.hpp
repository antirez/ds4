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

#ifndef ROCWMMA_UTILITY_GET_IMPL_HPP
#define ROCWMMA_UTILITY_GET_IMPL_HPP

#include "../vector.hpp"

namespace rocwmma::detail
{
    // HIP_vector_type overloads
    template <uint32_t Idx, typename DataT, uint32_t VecSize>
    ROCWMMA_HOST_DEVICE constexpr inline DataT get(HIP_vector_type<DataT, VecSize>&& v)
    {
#if defined(__HIP_PLATFORM_AMD__) && (HIP_VERSION_MAJOR < 7)
        return v.data[Idx];
#else
        return v[Idx];
#endif // defined(__HIP_PLATFORM_AMD__) && (HIP_VERSION_MAJOR < 7)
    }

    template <uint32_t Idx, typename DataT, uint32_t VecSize>
    ROCWMMA_HOST_DEVICE constexpr inline DataT& get(HIP_vector_type<DataT, VecSize>& v)
    {
#if defined(__HIP_PLATFORM_AMD__) && (HIP_VERSION_MAJOR < 7)
        return reinterpret_cast<DataT*>(&v.data)[Idx];
#else
        return v[Idx];
#endif // defined(__HIP_PLATFORM_AMD__) && (HIP_VERSION_MAJOR < 7)
    }

    template <uint32_t Idx, typename DataT, uint32_t VecSize>
    ROCWMMA_HOST_DEVICE constexpr inline DataT get(HIP_vector_type<DataT, VecSize> const& v)
    {
#if defined(__HIP_PLATFORM_AMD__) && (HIP_VERSION_MAJOR < 7)
        return v.data[Idx];
#else
        return v[Idx];
#endif // defined(__HIP_PLATFORM_AMD__) && (HIP_VERSION_MAJOR < 7)
    }

    // non_native_vector_base overloads
    template <uint32_t Idx, typename DataT, uint32_t VecSize>
    ROCWMMA_HOST_DEVICE constexpr static inline DataT
        get(non_native_vector_base<DataT, VecSize>&& v)
    {
        return v[Idx];
    }

    template <uint32_t Idx, typename DataT, uint32_t VecSize>
    ROCWMMA_HOST_DEVICE constexpr static inline DataT&
        get(non_native_vector_base<DataT, VecSize>& v)
    {
        return v[Idx];
    }

    template <uint32_t Idx, typename DataT, uint32_t VecSize>
    ROCWMMA_HOST_DEVICE constexpr static inline DataT
        get(non_native_vector_base<DataT, VecSize> const& v)
    {
        return v[Idx];
    }

    // Wrapper for Number<I>
    template <typename Idx, typename VecT>
    ROCWMMA_HOST_DEVICE constexpr inline decltype(auto) get(VecT&& v)
    {
        return get<Idx::value>(forward<VecT>(v));
    }

} // namespace rocwmma::detail

#endif // ROCWMMA_UTILITY_GET_IMPL_HPP
