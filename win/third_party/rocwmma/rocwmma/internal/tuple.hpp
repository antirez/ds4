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
#ifndef ROCWMMA_TUPLE_HPP
#define ROCWMMA_TUPLE_HPP

#if !defined(__HIPCC_RTC__)

#include <iostream>
#include <tuple>

#endif // !defined(__HIPCC_RTC__)

#include "vector.hpp"
#include "utility/forward.hpp"
#include "utility/sequence.hpp"
#include "utils.hpp"

namespace rocwmma
{
    namespace detail
    {
        template <typename VecT, size_t... Indices>
        constexpr static auto copy_impl(VecT&& t, index_sequence<Indices...>&&)
        {
            return make_vector(get<Indices>(forward<VecT>(t))...);
        }
    }

    template <uint32_t Count = 1u, typename VecT>
    constexpr static auto pop_back(VecT&& t)
    {
        return detail::copy_impl(
            forward<VecT>(t),
            make_offset_index_sequence<VecTraits<decay_t<VecT>>::size() - Count>());
    }

    template <uint32_t Count = 1u, typename VecT>
    constexpr static auto pop_front(VecT&& t)
    {
        return detail::copy_impl(
            forward<VecT>(t),
            make_offset_index_sequence<VecTraits<decay_t<VecT>>::size() - Count, Count>());
    }

    template <typename VecT>
    constexpr static decltype(auto) get_first(VecT&& t)
    {
        return get<0>(forward<VecT>(t));
    }

    template <typename VecT>
    constexpr static decltype(auto) get_last(VecT&& t)
    {
        return get<VecTraits<decay_t<VecT>>::size() - 1u>(forward<VecT>(t));
    }

    namespace detail
    {
        template <typename VecT, size_t... Indices>
        constexpr static decltype(auto) reverse_impl(VecT&& t, index_sequence<Indices...>)
        {
            return make_vector(get<sizeof...(Indices) - 1 - Indices>(forward<VecT>(t))...);
        }
    }

    template <typename VecT>
    constexpr static decltype(auto) reverse(VecT&& t)
    {
        return detail::reverse_impl(forward<VecT>(t),
                                    make_index_sequence<VecTraits<decay_t<VecT>>::size()>{});
    }

    namespace detail
    {
        template <typename Vec0, typename Vec1, size_t... Indices>
        constexpr static decltype(auto)
            flatten_coord_right_impl(Vec0&& coord, Vec1&& dims, index_sequence<Indices...>)
        {
            static_assert(VecTraits<decay_t<Vec0>>::size() == sizeof...(Indices)
                              && VecTraits<decay_t<Vec1>>::size() == sizeof...(Indices),
                          "coord and dims vectors must be the same size");

            auto flatten = [](auto&& c, auto&& d, auto& mul) {
                auto result = c * mul;
                mul *= d;
                return result;
            };

            auto mult = typename VecTraits<decay_t<Vec0>>::DataT{1};
            return (flatten(get<Indices>(forward<Vec0>(coord)),
                            get<Indices>(forward<Vec1>(dims)),
                            forward<decltype(mult)&>(mult))
                    + ...);
        }
    }

    template <typename Vec0, typename Vec1>
    constexpr static decltype(auto) flatten_coord_right(Vec0&& coord, Vec1&& dims)
    {
        return detail::flatten_coord_right_impl(
            forward<Vec0>(coord),
            forward<Vec1>(dims),
            make_index_sequence<VecTraits<decay_t<Vec0>>::size()>{});
    }

    namespace detail
    {
        template <typename Vec0, typename Vec1, size_t... Indices>
        constexpr static decltype(auto)
            flatten_coord_left_impl(Vec0&& coord, Vec1&& dims, index_sequence<Indices...>)
        {
            static_assert(VecTraits<decay_t<Vec0>>::size() == sizeof...(Indices)
                              && VecTraits<decay_t<Vec1>>::size() == sizeof...(Indices),
                          "coord and dims vectors must be the same size");

            auto flatten = [](auto&& c, auto&& d, auto& mul) {
                auto result = c * mul;
                mul *= d;
                return result;
            };

            auto mult = typename VecTraits<decay_t<Vec0>>::DataT{1};
            return (flatten(get<sizeof...(Indices) - 1 - Indices>(forward<Vec0>(coord)),
                            get<sizeof...(Indices) - 1 - Indices>(forward<Vec1>(dims)),
                            forward<decltype(mult)&>(mult))
                    + ...);
        }
    }

    template <typename Vec0, typename Vec1>
    constexpr static decltype(auto) flatten_coord_left(Vec0&& coord, Vec1&& dims)
    {
        return detail::flatten_coord_left_impl(
            forward<Vec0>(coord),
            forward<Vec1>(dims),
            make_index_sequence<VecTraits<decay_t<Vec0>>::size()>{});
    }

    namespace detail
    {
        template <typename Coord1d, typename VecT, size_t... Indices>
        constexpr static inline decltype(auto)
            inflate_coord_right_impl(Coord1d&& flatCoord, VecT&& dims, index_sequence<Indices...>)
        {
            auto inflate = [](auto&& c, auto&& d, auto& div, auto&& is_last) {
                if constexpr((bool)decay_t<decltype(is_last)>::value == true)
                {
                    auto result = c / div;
                    div *= d;
                    return result;
                }
                else
                {
                    auto result = c / div % d;
                    div *= d;
                    return result;
                }
            };

            auto div = decay_t<Coord1d>{1};
            return make_vector(inflate(forward<Coord1d>(flatCoord),
                                       get<Indices>(forward<VecT>(dims)),
                                       forward<decltype(div)&>(div),
                                       I<Indices == sizeof...(Indices) - 1>{})...);
        }
    }

    template <typename Coord1d, typename VecT>
    constexpr static inline decltype(auto) inflate_coord_right(Coord1d&& flatCoord, VecT&& dims)
    {
        return detail::inflate_coord_right_impl(
            forward<Coord1d>(flatCoord),
            forward<VecT>(dims),
            make_index_sequence<VecTraits<decay_t<VecT>>::size()>{});
    }

    namespace detail
    {
        template <typename Coord1d, typename VecT, size_t... Indices>
        constexpr static inline decltype(auto)
            inflate_coord_left_impl(Coord1d&& flatCoord, VecT&& dims, index_sequence<Indices...>)
        {
            auto inflate = [](auto&& c, auto&& d, auto& div, auto&& is_last) {
                if constexpr((bool)decay_t<decltype(is_last)>::value == true)
                {
                    auto result = c / div;
                    div *= d;
                    return result;
                }
                else
                {
                    auto result = c / div % d;
                    div *= d;
                    return result;
                }
            };

            auto div = decay_t<Coord1d>{1};
            return reverse(make_vector(
                inflate(forward<Coord1d>(flatCoord),
                        get<VecTraits<decay_t<VecT>>::size() - 1 - Indices>(forward<VecT>(dims)),
                        forward<decltype(div)&>(div),
                        I<Indices == sizeof...(Indices) - 1>{})...));
        }
    }

    template <typename Coord1d, typename VecT>
    constexpr static inline decltype(auto) inflate_coord_left(Coord1d&& flatCoord, VecT&& dims)
    {
        return detail::inflate_coord_left_impl(
            forward<Coord1d>(flatCoord),
            forward<VecT>(dims),
            make_index_sequence<VecTraits<decay_t<VecT>>::size()>{});
    }

    namespace detail
    {
        template <typename VecT0, typename VecT1, size_t... Indices>
        constexpr static inline decltype(auto)
            to_matrix_space_impl(VecT0&& strideCoord, VecT1&& strides, index_sequence<Indices...>)
        {
            static_assert(VecTraits<decay_t<VecT0>>::size() == sizeof...(Indices)
                              && VecTraits<decay_t<VecT1>>::size() == sizeof...(Indices),
                          "strideCoord and strides vectors must be the same size");

            auto matrix_space_offset
                = [](auto&& component, auto&& stride) { return component * stride; };

            // Calculate a final matrix-space offset by accumulating
            // each stride coordinate component multiplied by their respective stride:
            // result = Sum(strideCoord[0]* strides[0] + ... + strideCoord[N-1] * strides[N-1])
            return (matrix_space_offset(get<Indices>(forward<VecT0>(strideCoord)),
                                        get<Indices>(forward<VecT1>(strides)))
                    + ...);
        }
    }

    template <typename VecT0, typename VecT1>
    constexpr static inline decltype(auto) to_matrix_space(VecT0&& strideCoord, VecT1&& strides)
    {
        return detail::to_matrix_space_impl(
            forward<VecT0>(strideCoord),
            forward<VecT1>(strides),
            make_index_sequence<VecTraits<decay_t<VecT0>>::size()>{});
    }

#if !defined(__HIPCC_RTC__)

    template <class T, size_t... I>
    auto& print(std::ostream& os, T&& t, index_sequence<I...>&&)
    {
        os << "(";
        (..., (os << (I == 0 ? "" : ", ") << get<I>(forward<T>(t))));
        return os << ")\n";
    }

    template <class... ArgsT>
    auto& print(std::ostream& os, std::tuple<ArgsT...> const& t)
    {
        return print(os, t, make_index_sequence<sizeof...(ArgsT)>());
    }

#endif // !defined(__HIPCC_RTC__)

} // namespace rocwmma

#if !defined(__HIPCC_RTC__)
namespace std
{
    template <typename... Args>
    ostream& operator<<(ostream& os, tuple<Args...> const& t)
    {
        return rocwmma::print(os, t);
    }
}
#endif // !defined(__HIPCC_RTC__)

#endif // ROCWMMA_TUPLE_HPP
