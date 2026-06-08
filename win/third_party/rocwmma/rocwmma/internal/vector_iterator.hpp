#ifndef ROCWMMA_VECTOR_ITERATOR_HPP
#define ROCWMMA_VECTOR_ITERATOR_HPP

#include "vector.hpp"

inline constexpr unsigned int next_pot(unsigned int x)
{
    // Precondition: x > 1.
    return x > 1 ? (1u << (32u - __builtin_clz(x - 1u))) : x;
}
namespace rocwmma
{
    // Fwd declare
    template <typename VecT>
    struct VecTraits;

    // Vector iterator class: handles for const and non-const vectors
    template <class VecT, uint32_t SubVecSize = 1>
    struct VectorIterator
    {
        using VecTraits                = VecTraits<VecT>;
        using DataT                    = typename VecTraits::DataT;
        constexpr static uint32_t Rank = VecTraits::size();

        template <typename VDataT, uint32_t VRank>
        using VecImpl = typename VecTraits::template VecT<VDataT, VRank>;

        using RefVecT = VecImpl<DataT, Rank>;
        using ItVecT  = VecImpl<DataT, SubVecSize>;

        struct iterator
        {
            RefVecT const&   mRef;
            mutable uint32_t mIdx;

            struct Traits
            {
                enum : int32_t
                {
                    Range = Rank / SubVecSize
                };
            };

            static_assert(Rank % SubVecSize == 0, "VecSize not iterable by SubVecSize");
            static_assert(sizeof(RefVecT) == sizeof(VecT), "Cannot alias subvector");
            static_assert(sizeof(ItVecT) == SubVecSize * sizeof(DataT), "Cannot alias subvector");
            static_assert(sizeof(RefVecT) == sizeof(ItVecT) * Traits::Range,
                          "Cannot alias subvector");

            ROCWMMA_HOST_DEVICE constexpr iterator() noexcept = delete;

            ROCWMMA_HOST_DEVICE constexpr iterator(RefVecT const& ref, uint32_t idx = 0) noexcept
                : mRef(ref)
                , mIdx(idx)
            {
            }

            ROCWMMA_HOST_DEVICE ~iterator() = default;

            ROCWMMA_HOST_DEVICE inline ItVecT const& operator*() const
            {
                // Cast as array of sub-vectors
                return reinterpret_cast<ItVecT const*>(&mRef)[mIdx];
            }

            ROCWMMA_HOST_DEVICE inline ItVecT& operator*()
            {
                // Cast as array of sub-vectors
                return reinterpret_cast<ItVecT*>(&const_cast<RefVecT&>(mRef))[mIdx];
            }

            ROCWMMA_HOST_DEVICE inline iterator const& operator++() const
            {
                mIdx++;
                return *this;
            }
            ROCWMMA_HOST_DEVICE inline iterator operator++(int) const
            {
                auto retval = *this;
                ++mIdx;
                return retval;
            }

            ROCWMMA_HOST_DEVICE inline iterator const& operator--() const
            {
                mIdx--;
                return *this;
            }
            ROCWMMA_HOST_DEVICE inline iterator operator--(int) const
            {
                auto retval = *this;
                --mIdx;
                return retval;
            }

            ROCWMMA_HOST_DEVICE inline iterator const& operator+=(int i) const
            {
                mIdx += i;
                return *this;
            }

            ROCWMMA_HOST_DEVICE inline iterator const& operator-=(int i) const
            {
                mIdx -= i;
                return *this;
            }

            ROCWMMA_HOST_DEVICE inline iterator operator+(int i) const
            {
                auto retval = *this;
                return retval += i;
            }

            ROCWMMA_HOST_DEVICE inline iterator operator-(int i) const
            {
                auto retval = *this;
                return retval -= i;
            }

            ROCWMMA_HOST_DEVICE inline bool operator==(iterator const& other) const
            {
                return (&mRef == &other.mRef) && (mIdx == other.mIdx);
            }
            ROCWMMA_HOST_DEVICE inline bool operator!=(iterator const& other) const
            {
                return !(*this == other);
            }

            //__device__ inline Iterator<SubVecSize, IsConst> next() const;
            // __device__ inline Iterator<SubVecSize, IsConst> prev() const;
            ROCWMMA_HOST_DEVICE inline uint32_t index() const
            {
                return mIdx;
            }
            ROCWMMA_HOST_DEVICE inline bool valid() const
            {
                return (mIdx >= 0) && (mIdx < Traits::Range);
            }
            // __device__ bool                                 valid() const;

            ROCWMMA_HOST_DEVICE constexpr static inline int32_t range()
            {
                return Traits::Range;
            }
            // __device__ constexpr static inline bool    isConst();
        };

        ROCWMMA_HOST_DEVICE
        constexpr VectorIterator(RefVecT const& refVec) noexcept
            : mRef(refVec)
        {
        }

        ROCWMMA_HOST_DEVICE
        ~VectorIterator() = default;

        ROCWMMA_HOST_DEVICE
        inline iterator it(uint32_t startIdx = 0)
        {
            return iterator(mRef, startIdx);
        }

        ROCWMMA_HOST_DEVICE
        inline iterator begin()
        {
            return iterator(mRef, 0u);
        }

        ROCWMMA_HOST_DEVICE
        inline iterator end()
        {
            return iterator(mRef, Rank / SubVecSize);
        }

        RefVecT const& mRef;
    };

    template <uint32_t SubVecSize = 1, typename VecT>
    constexpr auto makeVectorIterator(VecT&& vec)
    {
        return VectorIterator<decay_t<VecT>, SubVecSize>{forward<VecT>(vec)};
    }

} // namespace rocwmma

#endif // ROCWMMA_VECTOR_ITERATOR_HPP
