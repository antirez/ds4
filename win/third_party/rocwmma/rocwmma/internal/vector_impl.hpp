/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCWMMA_VECTOR_IMPL_HPP
#define ROCWMMA_VECTOR_IMPL_HPP

#include "utility/sequence.hpp"

namespace rocwmma
{
    namespace detail
    {
        namespace ArithmeticOp
        {
            struct Add
            {
                template <typename TT>
                ROCWMMA_HOST_DEVICE constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs + rhs;
                }
            };
            struct Sub
            {
                template <typename TT>
                ROCWMMA_HOST_DEVICE constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs - rhs;
                }
            };
            struct Mult
            {
                template <typename TT>
                ROCWMMA_HOST_DEVICE constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs * rhs;
                }
            };
            struct Div
            {
                template <typename TT>
                ROCWMMA_HOST_DEVICE constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs / rhs;
                }
            };
            struct Mod
            {
                template <typename TT, typename enable_if<is_integral<TT>{}>::type* = nullptr>
                ROCWMMA_HOST_DEVICE constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs % rhs;
                }
            };
            struct Minus
            {
                template <typename TT, typename enable_if<is_signed<TT>{}>::type* = nullptr>
                ROCWMMA_HOST_DEVICE constexpr static inline auto exec(TT lhs)
                {
                    return -lhs;
                }
            };

        } // namespace ArithmeticOp

        namespace BitwiseOp
        {
            struct And
            {
                template <typename TT, typename enable_if<is_integral<TT>{}>::type* = nullptr>
                ROCWMMA_HOST_DEVICE constexpr static inline TT exec(TT lhs, TT rhs)
                {
                    return lhs & rhs;
                }
            };

            struct Or
            {
                template <typename TT, typename enable_if<is_integral<TT>{}>::type* = nullptr>
                ROCWMMA_HOST_DEVICE constexpr static inline TT exec(TT lhs, TT rhs)
                {
                    return lhs | rhs;
                }
            };

            struct Not
            {
                template <typename TT, typename enable_if<is_integral<TT>{}>::type* = nullptr>
                ROCWMMA_HOST_DEVICE constexpr static inline TT exec(TT lhs)
                {
                    return ~lhs;
                }
            };

            struct Xor
            {
                template <typename TT, typename enable_if<is_integral<TT>{}>::type* = nullptr>
                ROCWMMA_HOST_DEVICE constexpr static inline TT exec(TT lhs, TT rhs)
                {
                    return lhs ^ rhs;
                }
            };

            struct ShiftR
            {
                template <typename TT, typename enable_if<is_integral<TT>{}>::type* = nullptr>
                ROCWMMA_HOST_DEVICE constexpr static inline TT exec(TT lhs, TT rhs)
                {
                    return lhs >> rhs;
                }
            };

            struct ShiftL
            {
                template <typename TT, typename enable_if<is_integral<TT>{}>::type* = nullptr>
                ROCWMMA_HOST_DEVICE constexpr static inline TT exec(TT lhs, TT rhs)
                {
                    return lhs >> rhs;
                }
            };

        } // namespace BitwiseOp

        namespace LogicalOp
        {
            struct And
            {
                template <typename TT,
                          typename enable_if<is_convertible<TT, bool>{}>::type* = nullptr>
                ROCWMMA_HOST_DEVICE constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs && rhs;
                }
            };

            struct Or
            {
                template <typename TT,
                          typename enable_if<is_convertible<TT, bool>{}>::type* = nullptr>
                ROCWMMA_HOST_DEVICE constexpr static inline auto exec(TT lhs, TT rhs)
                {
                    return lhs || rhs;
                }
            };

            struct Not
            {
                template <typename TT,
                          typename enable_if<is_convertible<TT, bool>{}>::type* = nullptr>
                ROCWMMA_HOST_DEVICE constexpr static inline auto exec(TT lhs)
                {
                    return !lhs;
                }
            };

        } // namespace LogicalOp

        namespace RelationalOp
        {
            struct Eq
            {
                template <typename TT>
                ROCWMMA_HOST_DEVICE constexpr static inline uint32_t exec(TT lhs, TT rhs)
                {
                    return lhs == rhs;
                }
            };

            struct Neq
            {
                template <typename TT>
                ROCWMMA_HOST_DEVICE constexpr static inline uint32_t exec(TT lhs, TT rhs)
                {
                    return lhs != rhs;
                }
            };

            struct Gte
            {
                template <typename TT>
                ROCWMMA_HOST_DEVICE constexpr static inline uint32_t exec(TT lhs, TT rhs)
                {
                    return lhs >= rhs;
                }
            };

            struct Lte
            {
                template <typename TT>
                ROCWMMA_HOST_DEVICE constexpr static inline uint32_t exec(TT lhs, TT rhs)
                {
                    return lhs <= rhs;
                }
            };

            struct Gt
            {
                template <typename TT>
                ROCWMMA_HOST_DEVICE constexpr static inline uint32_t exec(TT lhs, TT rhs)
                {
                    return lhs > rhs;
                }
            };

            struct Lt
            {
                template <typename TT>
                ROCWMMA_HOST_DEVICE constexpr static inline uint32_t exec(TT lhs, TT rhs)
                {
                    return lhs < rhs;
                }
            };

        } // namespace RelationalOp

        // Helpers for expression expansion, specific to non_native_vector_base

        // Use with operations that have 2 operands
        template <class BinOp, typename VecT, uint32_t... indices>
        ROCWMMA_HOST_DEVICE constexpr static inline VecT
            binOp(VecT const& lhs, VecT const& rhs, SeqT<indices...>) noexcept
        {
            // Construct a new vector via fold expression over all indices
            return VecT{(BinOp::exec(lhs.d[indices], rhs.d[indices]))...};
        }

        // Use with operations that have 1 operands
        template <class UnOp, typename VecT, uint32_t... indices>
        ROCWMMA_HOST_DEVICE constexpr static inline VecT unOp(VecT const& lhs,
                                                              SeqT<indices...>) noexcept
        {
            return VecT{(UnOp::exec(lhs.d[indices]))...};
        }

        // Use with relational operations that have 2 operands
        template <class BoolOp, typename VecT, uint32_t... indices>
        ROCWMMA_HOST_DEVICE constexpr static inline typename VecT::BoolVecT
            boolOp(VecT const& lhs, VecT const& rhs, SeqT<indices...>) noexcept
        {
            return typename VecT::BoolVecT{(BoolOp::exec(lhs.d[indices], rhs.d[indices]))...};
        }

        // Used to broadcast a single value to the entire vector.
        template <typename VecT, typename T, uint32_t... indices>
        ROCWMMA_HOST_DEVICE constexpr static inline VecT bCast(T val, SeqT<indices...>) noexcept
        {
            // Indices value ignored, but broadcast for all
            return VecT{((void)indices, val)...};
        }

    } // namespace detail

    // NOTE: The single-valued 'broadcast' constructor can only be used with Rank > 1.
    // Why? The bCast function constructs the vector with <Rank> copies of the input val.
    // When Rank == 1, this would create an endless ctor->bcast->ctor->bcast... loop.
    // As a solution, Rank == 1 should fall into the ctor(Ts... args) for initializer
    // list construction, and NOT bCast initialization.
    template <typename T, unsigned int Rank>
    template <typename U, typename enable_if<(is_same<U, T>{}) && (Rank > 1)>::type*>
    ROCWMMA_HOST_DEVICE constexpr non_native_vector_base<T, Rank>::non_native_vector_base(
        T x_) noexcept
        : non_native_vector_base(detail::template bCast<VecT>(x_, detail::Seq<Rank>{}))
    {
    }

    // TODO: should add type compatibility check.
    // Default template depth is currently not deep enough to
    // support vector sizes of 512
    template <typename T, unsigned int Rank>
    template <typename... Ts, typename U, typename enable_if<(sizeof...(Ts) == Rank)>::type*>
    ROCWMMA_HOST_DEVICE constexpr non_native_vector_base<T, Rank>::non_native_vector_base(
        Ts... args) noexcept
        : d{static_cast<T>(args)...}
    {
    }

    template <typename T, unsigned int Rank>
    ROCWMMA_HOST_DEVICE constexpr inline T&
        non_native_vector_base<T, Rank>::operator[](unsigned int idx) noexcept
    {
        return d[idx];
    }

    template <typename T, unsigned int Rank>
    ROCWMMA_HOST_DEVICE constexpr inline T
        non_native_vector_base<T, Rank>::operator[](unsigned int idx) const noexcept
    {
        return d[idx];
    }
    template <typename T, unsigned int Rank>
    ROCWMMA_HOST_DEVICE constexpr inline auto
        non_native_vector_base<T, Rank>::operator+=(const VecT& x_) noexcept -> VecT&
    {
        return (*this = detail::binOp<detail::ArithmeticOp::Add>(*this, x_, detail::Seq<Rank>{}));
    }

    template <typename T, unsigned int Rank>
    ROCWMMA_HOST_DEVICE constexpr inline auto
        non_native_vector_base<T, Rank>::operator-=(const VecT& x_) noexcept -> VecT&
    {
        return (*this = detail::binOp<detail::ArithmeticOp::Sub>(*this, x_, detail::Seq<Rank>{}));
    }

    template <typename T, unsigned int Rank>
    ROCWMMA_HOST_DEVICE constexpr inline auto
        non_native_vector_base<T, Rank>::operator*=(const VecT& x_) noexcept -> VecT&
    {
        return (*this = detail::binOp<detail::ArithmeticOp::Mult>(*this, x_, detail::Seq<Rank>{}));
    }

    template <typename T, unsigned int Rank>
    ROCWMMA_HOST_DEVICE constexpr inline auto
        non_native_vector_base<T, Rank>::operator/=(const VecT& x_) noexcept -> VecT&
    {
        return (*this = detail::binOp<detail::ArithmeticOp::Div>(*this, x_, detail::Seq<Rank>{}));
    }

    template <typename T, unsigned int Rank>
    template <typename U, enable_if_integral_t<U>*>
    ROCWMMA_HOST_DEVICE inline auto
        non_native_vector_base<T, Rank>::operator%=(const VecT& x_) noexcept -> VecT&
    {
        return (*this = detail::binOp<detail::ArithmeticOp::Mod>(*this, x_, detail::Seq<Rank>{}));
    }

    template <typename T, unsigned int Rank>
    ROCWMMA_HOST_DEVICE constexpr inline auto non_native_vector_base<T, Rank>::operator++() noexcept
        -> VecT&
    {
        return *this += VecT{static_cast<T>(1.0f)};
    }

    template <typename T, unsigned int Rank>
    ROCWMMA_HOST_DEVICE constexpr inline auto
        non_native_vector_base<T, Rank>::operator++(int) noexcept -> VecT
    {
        auto before(*this);
        ++(*this);
        return before;
    }

    template <typename T, unsigned int Rank>
    ROCWMMA_HOST_DEVICE constexpr inline auto non_native_vector_base<T, Rank>::operator--() noexcept
        -> VecT&
    {
        return *this -= VecT{static_cast<T>(1.0f)};
    }

    template <typename T, unsigned int Rank>
    ROCWMMA_HOST_DEVICE constexpr inline auto
        non_native_vector_base<T, Rank>::operator--(int) noexcept -> VecT
    {
        auto before(*this);
        --(*this);
        return before;
    }

    template <typename T, unsigned int Rank>
    template <typename U, enable_if_signed_t<U>*>
    ROCWMMA_HOST_DEVICE inline auto non_native_vector_base<T, Rank>::operator-() const noexcept
        -> VecT
    {
        return detail::unOp<detail::ArithmeticOp::Minus>(*this, detail::Seq<Rank>{});
    }

    template <typename T, unsigned int Rank>
    template <typename U, enable_if_integral_t<U>*>
    ROCWMMA_HOST_DEVICE inline auto non_native_vector_base<T, Rank>::operator~() const noexcept
        -> VecT
    {
        return detail::unOp<detail::BitwiseOp::Not>(*this, detail::Seq<Rank>{});
    }

    // @cond
    template <typename T, unsigned int Rank>
    template <typename U, enable_if_integral_t<U>*>
    ROCWMMA_HOST_DEVICE inline auto
        non_native_vector_base<T, Rank>::operator&=(const VecT& x_) noexcept -> VecT&
    {
        return (*this = detail::binOp<detail::BitwiseOp::And>(*this, x_, detail::Seq<Rank>{}));
    }
    // @endcond

    template <typename T, unsigned int Rank>
    template <typename U, enable_if_integral_t<U>*>
    ROCWMMA_HOST_DEVICE inline auto
        non_native_vector_base<T, Rank>::operator|=(const VecT& x_) noexcept -> VecT&
    {
        return (*this = detail::binOp<detail::BitwiseOp::Or>(*this, x_, detail::Seq<Rank>{}));
    }

    template <typename T, unsigned int Rank>
    template <typename U, enable_if_integral_t<U>*>
    ROCWMMA_HOST_DEVICE inline auto
        non_native_vector_base<T, Rank>::operator^=(const VecT& x_) noexcept -> VecT&
    {
        return (*this = detail::binOp<detail::BitwiseOp::Xor>(*this, x_, detail::Seq<Rank>{}));
    }

    template <typename T, unsigned int Rank>
    template <typename U, enable_if_integral_t<U>*>
    ROCWMMA_HOST_DEVICE inline auto
        non_native_vector_base<T, Rank>::operator>>=(const VecT& x_) noexcept -> VecT&
    {
        return (*this = detail::binOp<detail::BitwiseOp::ShiftR>(*this, x_, detail::Seq<Rank>{}));
    }

    template <typename T, unsigned int Rank>
    template <typename U, enable_if_integral_t<U>*>
    ROCWMMA_HOST_DEVICE inline auto
        non_native_vector_base<T, Rank>::operator<<=(const VecT& x_) noexcept -> VecT&
    {
        return (*this = detail::binOp<detail::BitwiseOp::ShiftL>(*this, x_, detail::Seq<Rank>{}));
    }

    template <typename T, unsigned int Rank>
    ROCWMMA_HOST_DEVICE inline auto
        non_native_vector_base<T, Rank>::operator==(const VecT& x_) const noexcept -> BoolVecT
    {
        return detail::boolOp<detail::RelationalOp::Eq>(*this, x_, detail::Seq<Rank>{});
    }

    template <typename T, unsigned int Rank>
    ROCWMMA_HOST_DEVICE inline auto
        non_native_vector_base<T, Rank>::operator!=(const VecT& x_) const noexcept -> BoolVecT
    {
        return detail::boolOp<detail::RelationalOp::Neq>(*this, x_, detail::Seq<Rank>{});
    }

    template <typename T, unsigned int Rank>
    ROCWMMA_HOST_DEVICE inline auto
        non_native_vector_base<T, Rank>::operator>=(const VecT& x_) const noexcept -> BoolVecT
    {
        return detail::boolOp<detail::RelationalOp::Gte>(*this, x_, detail::Seq<Rank>{});
    }

    template <typename T, unsigned int Rank>
    ROCWMMA_HOST_DEVICE inline auto
        non_native_vector_base<T, Rank>::operator<=(const VecT& x_) const noexcept -> BoolVecT
    {
        return detail::boolOp<detail::RelationalOp::Lte>(*this, x_, detail::Seq<Rank>{});
    }

    template <typename T, unsigned int Rank>
    ROCWMMA_HOST_DEVICE inline auto
        non_native_vector_base<T, Rank>::operator>(const VecT& x_) const noexcept -> BoolVecT
    {
        return detail::boolOp<detail::RelationalOp::Gt>(*this, x_, detail::Seq<Rank>{});
    }

    template <typename T, unsigned int Rank>
    ROCWMMA_HOST_DEVICE inline auto
        non_native_vector_base<T, Rank>::operator<(const VecT& x_) const noexcept -> BoolVecT
    {
        return detail::boolOp<detail::RelationalOp::Lt>(*this, x_, detail::Seq<Rank>{});
    }

    // External ops
    template <typename T, unsigned int Rank>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator+(non_native_vector_base<T, Rank> const& x,
                  non_native_vector_base<T, Rank> const& y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} += y;
    }

    template <typename T, unsigned int Rank, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator+(non_native_vector_base<T, Rank> const& x, U y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} += non_native_vector_base<T, Rank>{y};
    }

    template <typename T, unsigned int Rank, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator+(U x, non_native_vector_base<T, Rank> const& y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} += y;
    }

    template <typename T, unsigned int Rank>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator-(non_native_vector_base<T, Rank> const& x,
                  non_native_vector_base<T, Rank> const& y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} -= y;
    }

    template <typename T, unsigned int Rank, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator-(non_native_vector_base<T, Rank> const& x, U y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} -= non_native_vector_base<T, Rank>{y};
    }

    template <typename T, unsigned int Rank, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator-(U x, non_native_vector_base<T, Rank> const& y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} -= y;
    }

    template <typename T, unsigned int Rank>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator*(non_native_vector_base<T, Rank> const& x,
                  non_native_vector_base<T, Rank> const& y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} *= y;
    }

    template <typename T, unsigned int Rank, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator*(non_native_vector_base<T, Rank> const& x, U y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} *= non_native_vector_base<T, Rank>{y};
    }

    template <typename T, unsigned int Rank, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator*(U x, non_native_vector_base<T, Rank> const& y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} *= y;
    }

    template <typename T, unsigned int Rank>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator/(non_native_vector_base<T, Rank> const& x,
                  non_native_vector_base<T, Rank> const& y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} /= y;
    }

    template <typename T, unsigned int Rank, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator/(non_native_vector_base<T, Rank> const& x, U y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} /= non_native_vector_base<T, Rank>{y};
    }

    template <typename T, unsigned int Rank, typename U>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator/(U x, non_native_vector_base<T, Rank> const& y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} /= y;
    }

    template <typename T, unsigned int Rank, enable_if_integral_t<T>* /* = nullptr */>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator%(non_native_vector_base<T, Rank> const& x,
                  non_native_vector_base<T, Rank> const& y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} %= y;
    }

    template <typename T, unsigned int Rank, typename U, enable_if_integral_t<T>* /* = nullptr */>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator%(non_native_vector_base<T, Rank> const& x, U y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} %= non_native_vector_base<T, Rank>{y};
    }

    template <typename T, unsigned int Rank, typename U, enable_if_integral_t<T>* /* = nullptr */>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator%(U x, non_native_vector_base<T, Rank> const& y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} %= y;
    }

    template <typename T, unsigned int Rank, enable_if_integral_t<T>* /* = nullptr */>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator&(non_native_vector_base<T, Rank> const& x,
                  non_native_vector_base<T, Rank> const& y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} &= y;
    }

    template <typename T, unsigned int Rank, typename U, enable_if_integral_t<T>* /* = nullptr */>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator&(non_native_vector_base<T, Rank> const& x, U y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} &= non_native_vector_base<T, Rank>{y};
    }

    template <typename T, unsigned int Rank, typename U, enable_if_integral_t<T>* /* = nullptr */>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator&(U x, non_native_vector_base<T, Rank> const& y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} &= y;
    }

    template <typename T, unsigned int Rank, enable_if_integral_t<T>* /* = nullptr */>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator|(non_native_vector_base<T, Rank> const& x,
                  non_native_vector_base<T, Rank> const& y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} |= y;
    }

    template <typename T, unsigned int Rank, typename U, enable_if_integral_t<T>* /* = nullptr */>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator|(non_native_vector_base<T, Rank> const& x, U y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} |= non_native_vector_base<T, Rank>{y};
    }

    template <typename T, unsigned int Rank, typename U, enable_if_integral_t<T>* /* = nullptr */>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator|(U x, non_native_vector_base<T, Rank> const& y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} |= y;
    }

    template <typename T, unsigned int Rank, enable_if_integral_t<T>* /* = nullptr */>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator^(non_native_vector_base<T, Rank> const& x,
                  non_native_vector_base<T, Rank> const& y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} ^= y;
    }

    template <typename T, unsigned int Rank, typename U, enable_if_integral_t<T>* /* = nullptr */>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator^(non_native_vector_base<T, Rank> const& x, U y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} ^= non_native_vector_base<T, Rank>{y};
    }

    template <typename T, unsigned int Rank, typename U, enable_if_integral_t<T>* /* = nullptr */>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator^(U x, non_native_vector_base<T, Rank> const& y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} ^= y;
    }

    template <typename T, unsigned int Rank, enable_if_integral_t<T>* /* = nullptr */>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator>>(non_native_vector_base<T, Rank> const& x,
                   non_native_vector_base<T, Rank> const& y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} >>= y;
    }

    template <typename T, unsigned int Rank, typename U, enable_if_integral_t<T>* /* = nullptr */>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator>>(non_native_vector_base<T, Rank> const& x, U y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} >>= non_native_vector_base<T, Rank>{y};
    }

    template <typename T, unsigned int Rank, typename U, enable_if_integral_t<T>* /* = nullptr */>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator>>(U x, non_native_vector_base<T, Rank> const& y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} >>= y;
    }

    template <typename T, unsigned int Rank, enable_if_integral_t<T>* /* = nullptr */>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator<<(non_native_vector_base<T, Rank> const& x,
                   non_native_vector_base<T, Rank> const& y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} <<= y;
    }

    template <typename T, unsigned int Rank, typename U, enable_if_integral_t<T>* /* = nullptr */>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator<<(non_native_vector_base<T, Rank> const& x, U y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} <<= non_native_vector_base<T, Rank>{y};
    }

    template <typename T, unsigned int Rank, typename U, enable_if_integral_t<T>* /* = nullptr */>
    ROCWMMA_HOST_DEVICE inline constexpr non_native_vector_base<T, Rank>
        operator<<(U x, non_native_vector_base<T, Rank> const& y) noexcept
    {
        return non_native_vector_base<T, Rank>{x} <<= y;
    }

} // namespace rocwmma

////
/// VECTORIZATION:
///
/// NATIVE data types (char, int, float, _Float16 etc...) may have support for native vector types
/// if extensions are available (see ROCWMMA_NATIVE_VECTOR_STORAGE_IMPL). Where available,
/// native vector types shall be used for native data types. Otherwise, vectors will
/// be implemented as built-in arrays (see ROCWMMA_NON_NATIVE_VECTOR_STORAGE_IMPL).
///
/// NON_NATIVE data types (hip_bfloat16, __half, etc.. any struct or class that is not system-native)
/// do not have native vector support, therefore they are always implemented as built-in arrays.
/// (see ROCWMMA_NON_NATIVE_VECTOR_STORAGE_IMPL)
///
/// ADDING VECTORIZATION SUPPORT in HIP:
/// - For NATIVE data types: ROCWMMA_REGISTER_NATIVE_VECTOR_TYPE(TYPE, RANK)
/// - For NON_NATIVE data types: ROCWMMA_REGISTER_NATIVE_VECTOR_TYPE(TYPE, RANK)
////

//////////////////////////////////////
/// Definition of accessor aliases ///
//////////////////////////////////////

#define ROCWMMA_HIP_ACCESSOR_ALIAS_IMPL_RANK1(TYPE) \
    struct                                          \
    {                                               \
        TYPE x;                                     \
    };

#define ROCWMMA_HIP_ACCESSOR_ALIAS_IMPL_RANK2(TYPE) \
    struct                                          \
    {                                               \
        TYPE x;                                     \
        TYPE y;                                     \
    };

#define ROCWMMA_HIP_ACCESSOR_ALIAS_IMPL_RANK3(TYPE) \
    struct                                          \
    {                                               \
        TYPE x;                                     \
        TYPE y;                                     \
        TYPE z;                                     \
    };

#define ROCWMMA_HIP_ACCESSOR_ALIAS_IMPL_RANK4(TYPE) \
    struct                                          \
    {                                               \
        TYPE x;                                     \
        TYPE y;                                     \
        TYPE z;                                     \
        TYPE w;                                     \
    };

// Untenable individual accessor maintenance for larger vectors: skip them
#define ROCWMMA_HIP_ACCESSOR_ALIAS_IMPL_RANK8(TYPE)
#define ROCWMMA_HIP_ACCESSOR_ALIAS_IMPL_RANK16(TYPE)
#define ROCWMMA_HIP_ACCESSOR_ALIAS_IMPL_RANK32(TYPE)
#define ROCWMMA_HIP_ACCESSOR_ALIAS_IMPL_RANK64(TYPE)
#define ROCWMMA_HIP_ACCESSOR_ALIAS_IMPL_RANK128(TYPE)
#define ROCWMMA_HIP_ACCESSOR_ALIAS_IMPL_RANK256(TYPE)
#define ROCWMMA_HIP_ACCESSOR_ALIAS_IMPL_RANK512(TYPE)

/////////////////////////////////////////////////////////////////////////////////
/// Definition of storage implementation (vector extension vs built-in array) ///
/////////////////////////////////////////////////////////////////////////////////

#define ROCWMMA_HIP_NON_NATIVE_VECTOR_STORAGE_IMPL(TYPE, RANK)       \
    using Native_vec_ = rocwmma::non_native_vector_base<TYPE, RANK>; \
                                                                     \
    union                                                            \
    {                                                                \
        Native_vec_ data;                                            \
        ROCWMMA_HIP_ACCESSOR_ALIAS_IMPL_RANK##RANK(TYPE);            \
    };

#define ROCWMMA_HIP_NATIVE_VECTOR_STORAGE_IMPL(TYPE, RANK)           \
    using Native_vec_ = TYPE __attribute__((ext_vector_type(RANK))); \
                                                                     \
    union alignas(RANK * sizeof(TYPE))                               \
    {                                                                \
        Native_vec_ data;                                            \
        ROCWMMA_HIP_ACCESSOR_ALIAS_IMPL_RANK##RANK(TYPE);            \
    };

////////////////////////////////////////////////////////////////////////////////////////////////
/// Definition of HIP_vector_base override for any type and rank, using above storage impl   ///
///                                                                                          ///
/// NOTE: Same Rank restrictions on broadcast construction, as previously seen with the      ///
/// implementation of non_native_vector<T, Rank>.                                            ///
/// Why is this needed again here? Because STORAGE_IMPL may be either non_native_vector_type ///
/// OR native vector extension. The latter doesn't have the required built-in broadcast.     ///
////////////////////////////////////////////////////////////////////////////////////////////////
// Fwd declare some items that are needed
namespace rocwmma::detail
{

    template <typename T, unsigned int Rank>
    struct storage_impl;

} // namespace rocwmma::detail

// clang-format off
#if defined(__HIP_PLATFORM_AMD__) && (HIP_VERSION_MAJOR < 7)
#define ROCWMMA_REGISTER_HIP_VECTOR_BASE(TYPE, RANK, STORAGE_IMPL)                              \
    template <>                                                                                 \
    struct HIP_vector_base<TYPE, RANK>                                                          \
    {                                                                                           \
        STORAGE_IMPL(TYPE, RANK);                                                               \
                                                                                                \
        using value_type = TYPE;                                                                \
                                                                                                \
        ROCWMMA_HOST_DEVICE                                                                     \
        HIP_vector_base() = default;                                                            \
        template <typename... ArgsT,                                                            \
                  typename U                                        = TYPE,                     \
                  rocwmma::enable_if_t<(sizeof...(ArgsT) == RANK)>* = nullptr>                  \
        ROCWMMA_HOST_DEVICE constexpr HIP_vector_base(ArgsT... args) noexcept                   \
            : data{args...}                                                                     \
        {                                                                                       \
        }                                                                                       \
                                                                                                \
        /* HIP < 7 treats this ctor as a broadcast ctor, writes val to all elements */          \
        template <typename U                                                         = TYPE,    \
                  rocwmma::enable_if_t<(rocwmma::is_same<U, TYPE>{}) && (RANK > 1)>* = nullptr> \
        ROCWMMA_HOST_DEVICE constexpr explicit HIP_vector_base(TYPE val) noexcept               \
            : HIP_vector_base(rocwmma::detail::template bCast<HIP_vector_base>(                 \
                val, rocwmma::detail::Seq<RANK>{}))                                             \
        {                                                                                       \
        }                                                                                       \
                                                                                                \
        ROCWMMA_HOST_DEVICE                                                                     \
        constexpr HIP_vector_base(const HIP_vector_base&) = default;                            \
                                                                                                \
        ROCWMMA_HOST_DEVICE                                                                     \
        constexpr HIP_vector_base(HIP_vector_base&&) = default;                                 \
                                                                                                \
        ROCWMMA_HOST_DEVICE                                                                     \
        ~HIP_vector_base() = default;                                                           \
                                                                                                \
        ROCWMMA_HOST_DEVICE                                                                     \
        HIP_vector_base& operator=(const HIP_vector_base& x_) noexcept = default;               \
    };

#else

#define ROCWMMA_REGISTER_HIP_VECTOR_BASE(TYPE, RANK, STORAGE_IMPL)                              \
    namespace rocwmma::detail                                                                   \
    {                                                                                           \
        template <>                                                                             \
        struct storage_impl<TYPE, RANK>                                                         \
        {                                                                                       \
            STORAGE_IMPL(TYPE, RANK);                                                           \
        };                                                                                      \
    }                                                                                           \
                                                                                                \
    template <>                                                                                 \
    struct HIP_vector_base<TYPE, RANK> : public rocwmma::detail::storage_impl<TYPE, RANK>       \
    {                                                                                           \
        /* Disallow public access to data member, following ROCm 7.0 changes in HIP */          \
        private:                                                                                \
        using StorageImpl = rocwmma::detail::storage_impl <TYPE, RANK>;                         \
        using StorageImpl::data;                                                                \
                                                                                                \
    public:                                                                                     \
        using value_type = TYPE;                                                                \
                                                                                                \
        ROCWMMA_HOST_DEVICE                                                                     \
        HIP_vector_base() = default;                                                            \
        template <typename... ArgsT,                                                            \
                  typename U                                        = TYPE,                     \
                  rocwmma::enable_if_t<(sizeof...(ArgsT) == RANK)>* = nullptr>                  \
        ROCWMMA_HOST_DEVICE constexpr HIP_vector_base(ArgsT... args) noexcept                   \
        {                                                                                       \
            data = {args...};                                                                   \
        }                                                                                       \
                                                                                                \
        template <typename U                                                         = TYPE,    \
                  rocwmma::enable_if_t<(rocwmma::is_same<U, TYPE>{}) && (RANK > 1)>* = nullptr> \
        ROCWMMA_HOST_DEVICE constexpr explicit HIP_vector_base(TYPE val) noexcept               \
        : HIP_vector_base(rocwmma::detail::template bCast<HIP_vector_base>(                     \
                static_cast<TYPE>(0.0f), rocwmma::detail::Seq<RANK>{}))                         \
        {                                                                                       \
            /* New behavior in HIP 7 only writes first element with val and rest is 0*/         \
            data[0] = val;                                                                      \
        }                                                                                       \
                                                                                                \
        ROCWMMA_HOST_DEVICE                                                                     \
        constexpr HIP_vector_base(const HIP_vector_base&) = default;                            \
                                                                                                \
        ROCWMMA_HOST_DEVICE                                                                     \
        constexpr HIP_vector_base(HIP_vector_base&&) = default;                                 \
                                                                                                \
        ROCWMMA_HOST_DEVICE                                                                     \
        ~HIP_vector_base() = default;                                                           \
                                                                                                \
        ROCWMMA_HOST_DEVICE                                                                     \
        HIP_vector_base& operator=(const HIP_vector_base& x_) noexcept = default;               \
    };

#endif /* defined(__HIP_PLATFORM_AMD__) && (HIP_VERSION_MAJOR < 7) */
// clang-format on

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Setup macros to implement HIP_vector_type for any T and Rank, specifying if platform native ///
///////////////////////////////////////////////////////////////////////////////////////////////////

#if __has_attribute(ext_vector_type)

#define ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(TYPE, RANK) \
    ROCWMMA_REGISTER_HIP_VECTOR_BASE(TYPE, RANK, ROCWMMA_HIP_NATIVE_VECTOR_STORAGE_IMPL)

#else

#define ROCWMMA_REGISTER_HIP_NATIVE_VECTOR_TYPE(TYPE, RANK) \
    ROCWMMA_REGISTER_HIP_VECTOR_BASE(TYPE, RANK, ROCWMMA_HIP_NON_NATIVE_VECTOR_STORAGE_IMPL)

#endif // __has_attribute(ext_vector_type)

#define ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(TYPE, RANK) \
    ROCWMMA_REGISTER_HIP_VECTOR_BASE(TYPE, RANK, ROCWMMA_HIP_NON_NATIVE_VECTOR_STORAGE_IMPL)

#if defined(__HIPCC_RTC__)
#define ROCWMMA_VEC_OPERATOR ROCWMMA_DEVICE
#else
#define ROCWMMA_VEC_OPERATOR ROCWMMA_HOST_DEVICE
#endif

// Quirk: explicit specialization for ++ / -- operators in HIP_vector_type<bfloat16_t, N>.
// Why? bfloat16_t doesn't have automatic conversion from integers so we must override the default implementation;
// Override such that in(de)crement operators use 1.f instead of 1(int)
#define ROCWMMA_IMPL_VECTOR_INC_DEC_OPS_AS_FLOAT(FLOAT_TYPE, RANK)                             \
    template <>                                                                                \
    ROCWMMA_VEC_OPERATOR inline HIP_vector_type<FLOAT_TYPE, RANK>&                             \
        HIP_vector_type<FLOAT_TYPE, RANK>::operator++() noexcept                               \
    {                                                                                          \
        return *this += rocwmma::make_vector<FLOAT_TYPE, RANK>(static_cast<FLOAT_TYPE>(1.0f)); \
    }                                                                                          \
                                                                                               \
    template <>                                                                                \
    ROCWMMA_VEC_OPERATOR inline HIP_vector_type<FLOAT_TYPE, RANK>&                             \
        HIP_vector_type<FLOAT_TYPE, RANK>::operator--() noexcept                               \
    {                                                                                          \
        return *this -= rocwmma::make_vector<FLOAT_TYPE, RANK>(static_cast<FLOAT_TYPE>(1.0f)); \
    }

// Roll the quirk into the registration macro
#define ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE_WITH_INC_DEC_OPS_AS_FLOAT(TYPE, RANK) \
    ROCWMMA_REGISTER_HIP_NON_NATIVE_VECTOR_TYPE(TYPE, RANK)                               \
    ROCWMMA_IMPL_VECTOR_INC_DEC_OPS_AS_FLOAT(TYPE, RANK)

#endif // ROCWMMA_VECTOR_IMPL_HPP
