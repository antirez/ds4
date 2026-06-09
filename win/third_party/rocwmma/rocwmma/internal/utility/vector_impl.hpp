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

#ifndef ROCWMMA_UTILITY_VECTOR_IMPL_HPP
#define ROCWMMA_UTILITY_VECTOR_IMPL_HPP

#include "../vector_iterator.hpp"
#include "get.hpp"
#include "sequence.hpp"
#include "type_traits.hpp"

namespace rocwmma
{
    template <typename VecT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto vector_size(VecT&& v)
    {
        return VecTraits<decay_t<VecT>>::size();
    }

    namespace detail
    {
        // Can be used to build any vector class of <DataT, VecSize>
        // Either VecT or non_native_vector_vase.
        // Class acts as a static for_each style generator:
        // Incoming functor F will be called with each index + args in sequence.
        // Results of functor calls are used to construct a new vector.
        template <template <typename, uint32_t> class VecT, typename DataT, uint32_t VecSize>
        struct vector_generator
        {
            static_assert(VecSize > 0, "VectorSize must be at least 1");

            ROCWMMA_HOST_DEVICE constexpr vector_generator() {}

            // F signature: F(Number<Iter>, args...)
            template <class F, typename... ArgsT>
            ROCWMMA_HOST_DEVICE constexpr inline auto operator()(F f, ArgsT&&... args) const
            {
                // Build the number sequence to be expanded below.
                return operator()(f, detail::Seq<VecSize>{}, forward<ArgsT>(args)...);
            }

        private:
            template <class F, uint32_t... Indices, typename... ArgsT>
            ROCWMMA_HOST_DEVICE constexpr inline auto
                operator()(F f, detail::SeqT<Indices...>, ArgsT&&... args) const
            {
                // Execute incoming functor f with each index, as well as forwarded args.
                // The resulting vector is constructed with the results of each functor call.
                return VecT<DataT, VecSize>{(f(I<Indices>{}, forward<ArgsT>(args)...))...};
            }
        };
    }

    template <template <typename, uint32_t> class VecT, typename DataT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto swap(VecT<DataT, 2> const& v)
    {
        return VecT<DataT, 2>{get<1>(v), get<0>(v)};
    }

    namespace detail
    {
        template <template <typename, uint32_t> class VecT, typename... Ts>
        ROCWMMA_HOST_DEVICE constexpr static inline auto make_vector_impl(Ts&&... ts)
        {
            // TODO: When HIP_vector_type becomes constexpr replace with non_native_vector type.

            // Ensure that all the arguments are the same type
            static_assert(detail::is_same_v<decay_t<Ts>...>,
                          "Vector arguments must all be the same type");

            using DataT = first_type_t<decay_t<Ts>...>;
            return VecT<DataT, sizeof...(Ts)>{forward<Ts>(ts)...};
        }

    } // namespace detail

    template <typename... Ts>
    ROCWMMA_HOST_DEVICE constexpr static inline auto make_vector(Ts&&... ts)
    {
        // TODO: When HIP_vector_type becomes constexpr replace
        // non_native_vector_base to VecT.
        return detail::make_vector_impl<non_native_vector_base>(forward<Ts>(ts)...);
    }

    namespace detail
    {
        template <typename Lhs, size_t... Is0, typename Rhs, size_t... Is1>
        constexpr static inline auto
            vector_cat_impl(Lhs&& lhs, index_sequence<Is0...>, Rhs&& rhs, index_sequence<Is1...>)
        {
            return make_vector(get<Is0>(forward<Lhs>(lhs))..., get<Is1>(forward<Rhs>(rhs))...);
        }

    } // namespace detail

    template <typename Lhs, typename Rhs>
    ROCWMMA_HOST_DEVICE constexpr static inline auto vector_cat(Lhs&& lhs, Rhs&& rhs)
    {
        constexpr size_t Size0 = VecTraits<decay_t<decltype(lhs)>>::size();
        constexpr size_t Size1 = VecTraits<decay_t<decltype(rhs)>>::size();

        return detail::vector_cat_impl(forward<Lhs>(lhs),
                                       detail::make_index_sequence<Size0>(),
                                       forward<Rhs>(rhs),
                                       detail::make_index_sequence<Size1>());
    }

    namespace detail
    {
        template <typename DataT0, typename DataT1, uint32_t Rank, size_t... Is>
        constexpr static inline auto
            mult_poly_vec_impl(non_native_vector_base<DataT0, Rank> const& lhs,
                               non_native_vector_base<DataT1, Rank> const& rhs,
                               index_sequence<Is...>)
        {
            return make_vector((get<Is>(lhs) * get<Is>(rhs))...);
        }

    } // namespace detail

    template <typename DataT0, typename DataT1, uint32_t Rank>
    constexpr static inline auto operator*(non_native_vector_base<DataT0, Rank> const& lhs,
                                           non_native_vector_base<DataT1, Rank> const& rhs)
    {
        return detail::mult_poly_vec_impl(lhs, rhs, detail::make_index_sequence<Rank>());
    }

    namespace detail
    {
        template <class BinOp, typename T, typename... Ts>
        ROCWMMA_HOST_DEVICE constexpr static inline auto reduceOp_impl(T&& t, Ts&&... ts) noexcept
        {
            using CastT = decay_t<T>;
            if constexpr(sizeof...(Ts) >= 1)
            {
                return BinOp::exec(static_cast<CastT>(t), reduceOp_impl<BinOp>(forward<Ts>(ts)...));
            }
            else
            {
                return static_cast<CastT>(t);
            }
        }

        template <class BinOp, typename VecT, size_t... Is>
        ROCWMMA_HOST_DEVICE constexpr static inline auto
            vector_reduce_impl(VecT&& v, index_sequence<Is...>) noexcept
        {
            return reduceOp_impl<BinOp>(get<Is>(forward<VecT>(v))...);
        }

        // Use with operations that have 1 operands
        template <class BinOp, typename VecT>
        ROCWMMA_HOST_DEVICE constexpr static inline auto vector_reduce(VecT&& lhs) noexcept
        {
            return vector_reduce_impl<BinOp>(
                forward<VecT>(lhs),
                detail::make_index_sequence<VecTraits<decay_t<VecT>>::size()>{});
        }

        // Fwd declare bitwise ops from vector class
        namespace BitwiseOp
        {
            struct And;
            struct Or;
        } // namespace BitwiseOp

    } // namespace detail

    template <typename VecT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto vector_reduce_and(VecT&& lhs) noexcept
    {
        return detail::vector_reduce<detail::BitwiseOp::And>(forward<VecT>(lhs));
    }

    template <typename VecT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto vector_reduce_or(VecT&& lhs) noexcept
    {
        return detail::vector_reduce<detail::BitwiseOp::Or>(forward<VecT>(lhs));
    }

    namespace detail
    {
        template <typename VecT0,
                  typename VecT1,
                  class Func,
                  uint32_t... Indices,
                  typename... ArgsT>
        ROCWMMA_HOST_DEVICE constexpr static inline auto vector_for_each_impl(
            VecT0&& vOut, VecT1&& vIn, Func&& func, detail::SeqT<Indices...>, ArgsT&&... args)
        {
            using VecTraits0 = VecTraits<decay_t<VecT0>>;
            using VecTraits1 = VecTraits<decay_t<VecT1>>;

            constexpr uint32_t VecSize    = VecTraits0::size();
            constexpr uint32_t SubVecSize = VecSize / sizeof...(Indices);

            // Sanity checks
            static_assert(VecTraits0::size() == VecTraits1::size(), "Vector sizes don't match");
            static_assert(VecSize % SubVecSize == 0u, "VecSize must be a multiple of SubVecSize");

            // Initialize iterators for read / write
            auto writeIt = makeVectorIterator<SubVecSize>(forward<VecT0>(vOut)).begin();
            auto readIt  = makeVectorIterator<SubVecSize>(forward<VecT1>(vIn)).begin();

            // Check the return type of the given func
            using RetT = decltype(func(*readIt, I<0>{}, forward<ArgsT>(args)...));

            // Fold over each subvector, in order
            if constexpr(!is_same_v<RetT, void>)
            {
                // Write result if the fwded function has a result
                ((*writeIt = func(*readIt, I<Indices>{}, forward<ArgsT>(args)...),
                  readIt++,
                  writeIt++),
                 ...);

                // Propagate a non-void return type
                return true;
            }
            else
            {
                // Skip write if the fwded function doesn't have a result
                ((func(*readIt, I<Indices>{}, forward<ArgsT>(args)...), readIt++), ...);

                // Propagate void return type;
                return;
            }
        }

    } // namespace detail

    // Func signature: Func(VecT<DataT, SubVecSize>, uint32_t idx, args...)
    template <uint32_t SubVecSize /*= 1u*/, typename VecT, class Func, typename... ArgsT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto
        vector_for_each(VecT&& v, Func&& func, ArgsT&&... args)
    {
        // Incoming vector Size
        using VecTraits            = VecTraits<decay_t<VecT>>;
        constexpr uint32_t VecSize = VecTraits::size();

        // Generate a result
        using ResultT = decay_t<VecT>;
        auto result   = ResultT{};

        static_assert(VecSize >= SubVecSize, "SubVecSize exceeds VecSize");

// Setup an impl call
#define INVOKE_IMPL                                                                \
    detail::vector_for_each_impl(forward<add_lvalue_reference_t<ResultT>>(result), \
                                 forward<VecT>(v),                                 \
                                 forward<Func>(func),                              \
                                 detail::Seq<VecSize / SubVecSize>{},              \
                                 forward<ArgsT>(args)...)

        // Let the impl function tell us if the function has a returnable value
        using RetT = decltype(INVOKE_IMPL);

        // Invoke the actual function
        INVOKE_IMPL;

// Cleanup symbol
#undef INVOKE_IMPL

        if constexpr(!is_same_v<RetT, void>)
        {
            return result;
        }
    }

    template <uint32_t SubVecSize /*= 1u*/, typename VecT, class Func, typename... ArgsT>
    ROCWMMA_HOST_DEVICE constexpr static inline decltype(auto)
        vector_mutate_for_each(VecT&& v, Func&& func, ArgsT&&... args)
    {
        using VecTraits            = VecTraits<decay_t<VecT>>;
        constexpr uint32_t VecSize = VecTraits::size();

        static_assert(VecSize >= SubVecSize, "SubVecSize exceeds VecSize");

// Setup an impl call
#define INVOKE_IMPL                                                   \
    detail::vector_for_each_impl(forward<VecT>(v),                    \
                                 forward<VecT>(v),                    \
                                 forward<Func>(func),                 \
                                 detail::Seq<VecSize / SubVecSize>{}, \
                                 forward<ArgsT>(args)...)

        // Let the impl function tell us if the function has a returnable value
        using RetT = decltype(INVOKE_IMPL);

        // Invoke the actual function
        INVOKE_IMPL;

// Cleanup symbol
#undef INVOKE_IMPL

        constexpr bool has_return_value = !is_same_v<RetT, void>;

        static_assert(!has_return_value || is_lvalue_reference_v<VecT>,
                      "Mutate requires lvalue reference input");

        if constexpr(has_return_value)
        {
            return forward<VecT>(v);
        }
    }

    namespace detail
    {

        template <uint32_t SubVecSize,
                  typename VecT,
                  typename AccumT,
                  class Func,
                  uint32_t... Indices,
                  typename... ArgsT>
        ROCWMMA_HOST_DEVICE constexpr static inline auto vector_reduce_impl(
            VecT&& v, AccumT&& init, Func&& func, detail::SeqT<Indices...>, ArgsT&&... args)
        {
            // Initialize accumulation and input iterators
            auto accum = init;
            auto it    = makeVectorIterator<SubVecSize>(forward<VecT>(v)).begin();

            // - Forward input sub-vector
            // - Forward accumulator
            // - Fold over each input subvector from ReadIt
            ((accum = func(*it, accum, I<Indices>{}, forward<ArgsT>(args)...), it++), ...);

            return accum;
        }

        template <uint32_t SubVecSize0,
                  uint32_t SubVecSize1,
                  typename VecT0,
                  typename VecT1,
                  typename AccumT,
                  class Func,
                  uint32_t... Indices,
                  typename... ArgsT>
        ROCWMMA_HOST_DEVICE constexpr static inline auto
            vector_reduce2_impl(VecT0&&  v0,
                                VecT1&&  v1,
                                AccumT&& init,
                                Func&&   func,
                                detail::SeqT<Indices...>,
                                ArgsT&&... args)
        {
            // Initialize accumulation and input iterators
            auto accum = init;
            auto it0   = makeVectorIterator<SubVecSize0>(forward<VecT0>(v0)).begin();
            auto it1   = makeVectorIterator<SubVecSize1>(forward<VecT1>(v1)).begin();

            // - Forward input sub-vector
            // - Forward accumulator
            // - Fold over each input subvector from ReadIt
            ((accum = func(*it0, *it1, accum, I<Indices>{}, forward<ArgsT>(args)...), it0++, it1++),
             ...);

            return accum;
        }

    } // namespace detail

    // Func signature: Func(VecT<DataT, SubVecSize>, uint32_t idx, args...)
    template <uint32_t SubVecSize /*= 1u*/,
              typename VecT,
              typename AccumT,
              class Func,
              typename... ArgsT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto
        vector_reduce(VecT&& v, AccumT&& init, Func&& func, ArgsT&&... args)
    {
        using VecTraits            = VecTraits<decay_t<VecT>>;
        constexpr uint32_t VecSize = VecTraits::size();

        // Sanity checks
        static_assert(VecSize >= SubVecSize, "SubVecSize exceeds VecSize");
        static_assert(VecSize % SubVecSize == 0u, "VecSize must be a multiple of SubVecSize");

        // Forward to impl with index sequence
        return detail::template vector_reduce_impl<SubVecSize>(forward<VecT>(v),
                                                               forward<AccumT>(init),
                                                               forward<Func>(func),
                                                               detail::Seq<VecSize / SubVecSize>{},
                                                               forward<ArgsT>(args)...);
    }

    // Func signature: Func(VecT<DataT, SubVecSize>, uint32_t idx, args...)
    template <uint32_t SubVecSize /*= 1u*/, typename VecT, class Func, typename... ArgsT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto
        vector_reduce(VecT&& v, Func&& func, ArgsT&&... args)
    {
        // Default accumulator is a vector same size as SubVecSize
        using VecTraits            = VecTraits<decay_t<VecT>>;
        using DataT                = typename VecTraits::DataT;
        constexpr uint32_t VecSize = VecTraits::size();
        using AccumT               = typename VecTraits::template VecT<DataT, SubVecSize>;

        // Sanity checks
        static_assert(VecSize >= SubVecSize, "SubVecSize exceeds VecSize");
        static_assert(VecSize % SubVecSize == 0u, "VecSize must be a multiple of SubVecSize");

        // Forward to impl with index sequence
        return vector_reduce<SubVecSize>(forward<VecT>(v),
                                         move(AccumT{static_cast<DataT>(0)}), // Default accum of 0
                                         forward<Func>(func),
                                         forward<ArgsT>(args)...);
    }

    // Func signature: Func(VecT<DataT, SubVecSize>, uint32_t idx, args...)
    template <uint32_t SubVecSize0 /*= 1u*/,
              uint32_t SubVecSize1 /*= SubVecSize0*/,
              typename VecT0,
              typename VecT1,
              typename AccumT,
              class Func,
              typename... ArgsT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto
        vector_reduce2(VecT0&& v0, VecT1&& v1, AccumT&& init, Func&& func, ArgsT&&... args)
    {
        //
        using VecTraits0 = VecTraits<decay_t<VecT0>>;
        using VecTraits1 = VecTraits<decay_t<VecT1>>;

        constexpr uint32_t VecSize0     = VecTraits0::size();
        constexpr uint32_t VecSize1     = VecTraits1::size();
        constexpr uint32_t StrideCount0 = VecSize0 / SubVecSize0;
        constexpr uint32_t StrideCount1 = VecSize1 / SubVecSize1;

        // Sanity checks
        static_assert(StrideCount0 == StrideCount1, "Stride counts must match");
        static_assert(VecSize0 >= StrideCount0, "Stride exceeds VecSize");
        static_assert(VecSize0 % StrideCount0 == 0u, "VecSize must be a multiple of Stride");
        static_assert(VecSize1 >= StrideCount1, "Stride exceeds VecSize");
        static_assert(VecSize1 % StrideCount1 == 0u, "VecSize must be a multiple of Stride");

        // Forward to impl with index sequence
        return detail::template vector_reduce2_impl<SubVecSize0, SubVecSize1>(
            forward<VecT0>(v0),
            forward<VecT1>(v1),
            forward<AccumT>(init),
            forward<Func>(func),
            detail::Seq<StrideCount0>{},
            forward<ArgsT>(args)...);
    }

    // Func signature: Func(VecT<DataT, SubVecSize>, uint32_t idx, args...)
    template <uint32_t SubVecSize0 /*= 1u*/,
              uint32_t SubVecSize1 /*= SubVecSize0*/,
              typename VecT0,
              typename VecT1,
              class Func,
              typename... ArgsT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto
        vector_reduce2(VecT0&& v0, VecT1&& v1, Func&& func, ArgsT&&... args)
    {
        // Default accumulator is a vector of SubVecSize0
        using VecTraits0 = VecTraits<decay_t<VecT0>>;
        using DataT      = typename VecTraits0::DataT;
        using AccumT     = typename VecTraits0::template VecT<DataT, SubVecSize0>;

        // Forward default accum
        return vector_reduce2<SubVecSize0, SubVecSize1>(
            forward<VecT0>(v0),
            forward<VecT1>(v1),
            move(AccumT{static_cast<DataT>(0)}), // Default of 0
            forward<Func>(func),
            forward<ArgsT>(args)...);
    }

    template <typename VecT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto reduce_add(VecT&& v0)
    {
        return apply([](auto&&... items) { return (items + ...); }, v0);
    }

    template <typename VecT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto reduce_mult(VecT&& v0)
    {
        return apply([](auto&&... items) { return (items * ...); }, v0);
    }

    // Fwd declare
    template <typename DataT, uint32_t VecSize>
    struct vector_generator;

    template <typename DataT, uint32_t VecSize, uint32_t Start /*= 0u*/, uint32_t Stride /*= 1u*/>
    ROCWMMA_HOST_DEVICE constexpr static inline auto make_vector_sequence()
    {
        constexpr auto buildSeq = [](auto&& idx) {
            constexpr auto Index = decay_t<decltype(idx)>::value;
            return static_cast<DataT>(Start + Index * Stride);
        };

        return vector_generator<DataT, VecSize>()(buildSeq);
    }

    template <typename DataT, uint32_t VecSize, typename ArgT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto make_vector(ArgT&& value)
    {
#if defined(__HIP_PLATFORM_AMD__) && (HIP_VERSION_MAJOR < 7)
        return VecT<DataT, VecSize>{static_cast<DataT>(value)};
#else
        // Uses HIP_vector_type
        return make_vector_type<DataT, VecSize>(static_cast<DataT>(value));
#endif // defined(__HIP_PLATFORM_AMD__) && (HIP_VERSION_MAJOR < 7)
    }

    template <typename VecT>
    ROCWMMA_HOST_DEVICE static inline decltype(auto) to_native_vector(VecT& v)
    {
#if defined(__HIP_PLATFORM_AMD__) && (HIP_VERSION_MAJOR < 7)
        using NativeT = typename decay_t<VecT>::Native_vec_;

        return reinterpret_cast<NativeT&>(v.data);
#else
        return get_native_vector(v);
#endif // defined(__HIP_PLATFORM_AMD__) && (HIP_VERSION_MAJOR < 7)
    }

    template <typename VecT>
    ROCWMMA_HOST_DEVICE static inline decltype(auto) to_native_vector(VecT const& v)
    {
#if defined(__HIP_PLATFORM_AMD__) && (HIP_VERSION_MAJOR < 7)
        using NativeT = typename decay_t<VecT>::Native_vec_;

        return reinterpret_cast<NativeT const&>(v.data);
#else
        return get_native_vector(v);
#endif // defined(__HIP_PLATFORM_AMD__) && (HIP_VERSION_MAJOR < 7)
    }

} // namespace rocwmma

#endif // ROCWMMA_UTILITY_VECTOR_IMPL_HPP
