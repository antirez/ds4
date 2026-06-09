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

#ifndef ROCWMMA_UTILITY_VECTOR_HPP
#define ROCWMMA_UTILITY_VECTOR_HPP

#include "../vector_types.hpp"
#include "vector_impl.hpp"

namespace rocwmma
{
    /*! \brief Returns the element count, or size of the input vector
    * @param v Input vector
    * @tparam VecT Input vector type
    */
    template <typename VecT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto vector_size(VecT&& v);

    /*! \brief Creates a vector with the given input values
    * @param ts Variadic list of inputs values
    * @tparam Ts Types of incoming input values
    */
    template <typename... Ts>
    ROCWMMA_HOST_DEVICE constexpr static inline auto make_vector(Ts&&... ts);

    /*! \class vector_generator
    *  \brief A flexible vector generator class that calls a functor to generate vector element.
    * Functor signature: F(Number<Idx>, args...), where Idx is the vector element number. The functor
    * may accept any number of arguments and generates a single value to be used as element Idx in the
    * result.
    *
    * @tparam DataT Data type of the vector
    * @tparam VecSize The number of vector elements
    */
    template <typename DataT, uint32_t VecSize>
    struct vector_generator : public detail::vector_generator<VecT, DataT, VecSize>
    {
    };

    /*! \brief Returns a concatenated vector of (lhs, rhs)
    * @param lhs Input vector, lower elements
    * @param rhs Input vector, upper elements
    * @tparam Lhs Input vector type of lhs
    * @tparam Rhs Input vector type of rhs
    */
    template <typename Lhs, typename Rhs>
    ROCWMMA_HOST_DEVICE constexpr static inline auto vector_cat(Lhs&& lhs, Rhs&& rhs);

    /*! \brief Returns the reduction result of bit-wise and between all elements in the input vector
    * @param v Input vector
    * @tparam VecT Input vector type
    */
    template <typename VecT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto vector_reduce_and(VecT&& v) noexcept;

    /*! \brief Returns the reduction result of bit-wise and between all elements in the input vector
    * @param v Input vector
    * @tparam VecT Input vector type
    */
    template <typename VecT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto vector_reduce_or(VecT&& v) noexcept;

    /*! \brief Swaps elements in vector size of 2
    * @param v Input vector
    * @tparam VecT Templated input vector class
    */
    template <template <typename, uint32_t> class VecT, typename DataT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto swap(VecT<DataT, 2> const& v);

    /*! \brief Splits input vector into sub-vectors. Func is applied to each sub-vector, which are then
    * concatenated and returned as a result. Does not change input vector v.
    * @param v Input vector
    * @param func Functor with signature Func(VecT<DataT, SubVecSize>, Number<I> idx, args...)
    * @param args Arguments that are forwarded to the functor
    * @tparam SubVecSize Sub-vector size (default 1)
    * @tparam VecT Type of input vector
    * @tparam Func Type of functor
    * @tparam ArgsT Type of forwarded arguments to functor
    */
    template <uint32_t SubVecSize = 1u, typename VecT, class Func, typename... ArgsT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto
        vector_for_each(VecT&& v, Func&& func, ArgsT&&... args);

    /*! \brief Splits input vector into sub-vectors. Func is applied to each sub-vector in-place. Returns
    * a reference to modified input vector.
    * @param v Input vector
    * @param func Functor with signature Func(VecT<DataT, SubVecSize>, Number<I> idx, args...)
    * @param args Arguments that are forwarded to the functor
    * @tparam SubVecSize Sub-vector size (default 1)
    * @tparam VecT Type of input vector
    * @tparam Func Type of functor
    * @tparam ArgsT Type of forwarded arguments to functor
    */
    template <uint32_t SubVecSize = 1u, typename VecT, class Func, typename... ArgsT>
    ROCWMMA_HOST_DEVICE constexpr static inline decltype(auto)
        vector_mutate_for_each(VecT&& v, Func&& func, ArgsT&&... args);

    /*! \brief Splits input vector into sub-vectors. Func is applied to each sub-vector and updates the accumulator.
    * Returns the final accumulator result.
    * @param v Input vector
    * @param init The initial accumulation value
    * @param func Functor with signature Func(VecT<DataT, SubVecSize>, AccumT, Number<I> idx, args...)
    * @param args Arguments that are forwarded to the functor
    * @tparam SubVecSize Sub-vector size (default 1)
    * @tparam VecT Type of input vector
    * @tparam AccumT Type of accumulator
    * @tparam Func Type of functor
    * @tparam ArgsT Type of forwarded arguments to functor
    */
    template <uint32_t SubVecSize = 1u,
              typename VecT,
              typename AccumT,
              class Func,
              typename... ArgsT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto
        vector_reduce(VecT&& v, AccumT&& init, Func&& func, ArgsT&&... args);

    /*! \brief Splits input vector into sub-vectors. Func is applied to each sub-vector and updates the accumulator.
    * Returns the final accumulator result. Initial accumulator of VecT<DataT, SubVecSize>{0} is assumed.
    * @param v Input vector
    * @param func Functor with signature Func(VecT<DataT, SubVecSize>, AccumT, Number<I> idx, args...)
    * @param args Arguments that are forwarded to the functor
    * @tparam SubVecSize Sub-vector size (default 1)
    * @tparam VecT Type of input vector
    * @tparam Func Type of functor
    * @tparam ArgsT Type of forwarded arguments to functor
    */
    template <uint32_t SubVecSize = 1u, typename VecT, class Func, typename... ArgsT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto
        vector_reduce(VecT&& v, Func&& func, ArgsT&&... args);

    /*! \brief Splits two input vector into sub-vectors. Func is applied to each sub-vector and updates the accumulator.
    * Input vectors may be different lengths, provided their size / subvecsize is the same.
    * Returns the final accumulator result.
    * @param v0 Input vector 0
    * @param v1 Input vector 1
    * @param init The initial accumulation value
    * @param func Functor with signature Func(VecT<DataT0, SubVecSize0>, VecT<DataT1, SubVecSize1>, AccumT, Number<I> idx, args...)
    * @param args Arguments that are forwarded to the functor
    * @tparam SubVecSize0 Sub-vector0 size (default 1)
    * @tparam SubVecSize1 Sub-vector1 size (default SubVecSize0)
    * @tparam VecT0 Type of input vector 0
    * @tparam VecT1 Type of input vector 1
    * @tparam AccumT Type of accumulator
    * @tparam Func Type of functor
    * @tparam ArgsT Type of forwarded arguments to functor
    */
    template <uint32_t SubVecSize0 = 1u,
              uint32_t SubVecSize1 = SubVecSize0,
              typename VecT0,
              typename VecT1,
              typename AccumT,
              class Func,
              typename... ArgsT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto
        vector_reduce2(VecT0&& v0, VecT1&& v1, AccumT&& init, Func&& func, ArgsT&&... args);

    /*! \brief Splits two input vector into sub-vectors. Func is applied to each sub-vector and updates the accumulator.
    * Input vectors may be different lengths, provided their size / subvecsize is the same.
    * Initial accumulator of VecT<DataT, SubVecSize0>{0} is assumed.
    * Returns the final accumulator result.
    * @param v0 Input vector 0
    * @param v1 Input vector 1
    * @param func Functor with signature Func(VecT<DataT0, SubVecSize0>, VecT<DataT1, SubVecSize1>, AccumT, Number<I> idx, args...)
    * @param args Arguments that are forwarded to the functor
    * @tparam SubVecSize0 Sub-vector0 size (default 1)
    * @tparam SubVecSize1 Sub-vector1 size (default SubVecSize0)
    * @tparam VecT0 Type of input vector 0
    * @tparam VecT1 Type of input vector 1
    * @tparam Func Type of functor
    * @tparam ArgsT Type of forwarded arguments to functor
    */
    template <uint32_t SubVecSize0 = 1u,
              uint32_t SubVecSize1 = SubVecSize0,
              typename VecT0,
              typename VecT1,
              class Func,
              typename... ArgsT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto
        vector_reduce2(VecT0&& v0, VecT1&& v1, Func&& func, ArgsT&&... args);

    /*! \brief Creates a vector that contains a sequence of numbers starting at Start, incrementing
    * by Stride for a VecSize number of steps.
    * @tparam DataT Data type
    * @tparam VecSize The size of the result vector
    * @tparam Start The starting offset for the number set (Default = 0)
    * @tparam Stride The step size between successive values (Default = 1)
    */
    template <typename DataT, uint32_t VecSize, uint32_t Start = 0u, uint32_t Stride = 1u>
    ROCWMMA_HOST_DEVICE constexpr static inline auto make_vector_sequence();

    /*! \brief Creates a vector that broadcasts the input value to all elements
    * @tparam DataT Data type
    * @tparam VecSize The size of the result vector
    */
    template <typename DataT, uint32_t VecSize, typename ArgT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto make_vector(ArgT&& value);

    /*! \brief Transforms the vector object to it's raw storage type (e.g., when using builtin wrappers)
    * @tparam VecT API vector type
    */
    template <typename VecT>
    ROCWMMA_HOST_DEVICE static inline decltype(auto) to_native_vector(VecT const& v);

    /*! \brief Transforms the vector object to it's raw storage type (e.g., when using builtin wrappers)
    * @tparam VecT API vector type
    */
    template <typename VecT>
    ROCWMMA_HOST_DEVICE static inline decltype(auto) to_native_vector(VecT& v);

} // namespace rocwmma

#endif // ROCWMMA_UTILITY_VECTOR_HPP
