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
#ifndef ROCWMMA_API_IMPL_HPP
#define ROCWMMA_API_IMPL_HPP

#include "rocwmma.hpp"
#include "rocwmma_transforms.hpp"

#include "internal/accessors.hpp"
#include "internal/blend.hpp"
#include "internal/broadcast.hpp"
#include "internal/constants.hpp"
#include "internal/convert.hpp"
#include "internal/dpp.hpp"
#include "internal/flow_control.hpp"
#include "internal/fragment_traits.hpp"
#include "internal/io_config.hpp"
#include "internal/io_layout.hpp"
#include "internal/io_shape.hpp"
#include "internal/io_traits.hpp"
#include "internal/layout/layout.hpp"
#include "internal/mapping_util.hpp"
#include "internal/mfma.hpp"
#include "internal/mma.hpp"
#include "internal/mma_config.hpp"
#include "internal/opaque_load.hpp"
#include "internal/opaque_store.hpp"
#include "internal/pack_util.hpp"
#include "internal/permute.hpp"
#include "internal/swizzle.hpp"
#include "internal/transforms.hpp"
#include "internal/types.hpp"
#include "internal/utils.hpp"
#include "internal/vector.hpp"
#include "internal/vector_iterator.hpp"
#include "internal/vector_util.hpp"
#include "internal/wmma.hpp"

namespace rocwmma
{

#define FragmentTypesDecl                                                             \
    typename MatrixT, uint32_t FragM, uint32_t FragN, uint32_t FragK, typename DataT, \
        typename DataLayoutT, typename Scheduler

#define FragmentTypesImpl MatrixT, FragM, FragN, FragK, DataT, DataLayoutT, Scheduler

    // fragment implementations
    template <FragmentTypesDecl>
    ROCWMMA_DEVICE fragment<FragmentTypesImpl>::fragment(const fragment& other)
        : mStorage(other.mStorage)
    {
    }

    template <FragmentTypesDecl>
    ROCWMMA_DEVICE fragment<FragmentTypesImpl>&
        fragment<FragmentTypesImpl>::operator=(const fragment<FragmentTypesImpl>& other)
    {
        mStorage = other.mStorage;
        return *this;
    }

    template <FragmentTypesDecl>
    ROCWMMA_DEVICE inline DataT& fragment<FragmentTypesImpl>::operator[](uint32_t index)
    {
#if defined(__HIP_PLATFORM_AMD__) && (HIP_VERSION_MAJOR < 7)
        return mAccess.data[index];
#else
        return mAccess[index];
#endif // defined(__HIP_PLATFORM_AMD__) && (HIP_VERSION_MAJOR < 7)
    }

    template <FragmentTypesDecl>
    ROCWMMA_DEVICE inline auto fragment<FragmentTypesImpl>::operator*() ->
        typename Traits::StorageT&
    {
        return mStorage;
    }

    template <FragmentTypesDecl>
    ROCWMMA_DEVICE inline DataT const& fragment<FragmentTypesImpl>::operator[](uint32_t index) const
    {
#if defined(__HIP_PLATFORM_AMD__) && (HIP_VERSION_MAJOR < 7)
        return mAccess.data[index];
#else
        return mAccess[index];
#endif // defined(__HIP_PLATFORM_AMD__) && (HIP_VERSION_MAJOR < 7)
    }

    template <FragmentTypesDecl>
    ROCWMMA_DEVICE inline auto fragment<FragmentTypesImpl>::operator*() const ->
        typename Traits::StorageT const&
    {
        return mStorage;
    }

    template <FragmentTypesDecl>
    ROCWMMA_DEVICE constexpr inline uint32_t fragment<FragmentTypesImpl>::height()
    {
        return GetIOShape_t<decltype(fragment())>::BlockHeight;
    }

    template <FragmentTypesDecl>
    ROCWMMA_DEVICE constexpr inline uint32_t fragment<FragmentTypesImpl>::width()
    {
        return GetIOShape_t<decltype(fragment())>::BlockWidth;
    }

    template <FragmentTypesDecl>
    ROCWMMA_DEVICE constexpr inline uint32_t fragment<FragmentTypesImpl>::blockDim()
    {
        return GetIOShape_t<decltype(fragment())>::BlockDim;
    }

    template <FragmentTypesDecl>
    ROCWMMA_DEVICE constexpr inline uint32_t fragment<FragmentTypesImpl>::kDim()
    {
        return GetIOShape_t<decltype(fragment())>::KDim;
    }

    template <FragmentTypesDecl>
    ROCWMMA_DEVICE constexpr inline uint32_t fragment<FragmentTypesImpl>::size()
    {
        return num_elements;
    }

#undef FragmentTypesDecl
#undef FragmentTypesImpl

    template <typename FragT, typename DataT>
    ROCWMMA_DEVICE void fill_fragment(FragT& frag, DataT value)
    {
        using FragTraits  = fragment_traits<decay_t<FragT>>;
        using IOConfig    = GetIOConfig_t<decay_t<FragT>>;
        using Broadcaster = typename IOConfig::Broadcaster;

        // Sanity checks
        static_assert(
            is_same_v<typename Broadcaster::Traits::BroadcastT, typename FragTraits::AccessT>,
            "Broadcast input and fragment access types do not match");

        Broadcaster::exec(frag.mAccess, typename FragTraits::DataT{value});
    }

    template <typename FragT, typename DataT>
    ROCWMMA_DEVICE void load_matrix_sync(FragT& frag, const DataT* data, uint32_t ldm)
    {
        using FragTraits    = fragment_traits<decay_t<FragT>>;
        using IOConfig      = GetIOConfig_t<decay_t<FragT>>;
        using Loader        = typename IOConfig::Loader;
        using PostLoadXForm = typename IOConfig::PostLoadXForm;

        // Sanity checks
        static_assert(is_same_v<typename FragTraits::DataT, DataT>,
                      "Fragment DataT doesn't match incoming pointer DataT");
        static_assert(!is_same_v<typename FragTraits::DataLayoutT, void>,
                      "Must provide layout information. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");
        static_assert(is_same_v<typename FragTraits::AccessT, typename Loader::BufferT>,
                      "Fragment access and load buffer types do not match");

        // Load then implicit pack
        Loader::exec(frag.mAccess, data, ldm);

        // Post-load transformation
        frag.mAccess = PostLoadXForm::exec(frag.mAccess);
    }

    template <typename FragT, typename DataT>
    ROCWMMA_DEVICE void
        load_matrix_sync(FragT& frag, const DataT* data, uint32_t ldm, layout_t layout)
    {
        using FragTraits = fragment_traits<decay_t<FragT>>;

        // Dispatch on layout type
        if(layout == layout_t::mem_row_major)
        {
            // Load as row major, then transform to fragment layout
            auto tmp = apply_data_layout_t<decay_t<FragT>, row_major>{};
            load_matrix_sync(tmp, data, ldm);
            frag = apply_data_layout<typename FragTraits::DataLayoutT>(tmp);
        }
        else
        {
            // Load as col major, then transform to fragment layout
            auto tmp = apply_data_layout_t<decay_t<FragT>, col_major>{};
            load_matrix_sync(tmp, data, ldm);
            frag = apply_data_layout<typename FragTraits::DataLayoutT>(tmp);
        }
    }

    template <typename FragT, typename DataT>
    ROCWMMA_DEVICE void store_matrix_sync(DataT* data, FragT const& frag, uint32_t ldm)
    {
        using FragTraits    = fragment_traits<decay_t<FragT>>;
        using IOConfig      = GetIOConfig_t<decay_t<FragT>>;
        using PreStoreXForm = typename IOConfig::PreStoreXForm;
        using Storer        = typename IOConfig::Storer;

        // Sanity checks
        static_assert(is_same_v<typename FragTraits::DataT, DataT>,
                      "Fragment DataT doesn't match outgoing pointer DataT");
        static_assert(!is_same_v<typename FragTraits::DataLayoutT, void>,
                      "Must provide data layout. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");
        static_assert(is_same_v<typename FragTraits::AccessT, typename Storer::BufferT>,
                      "Fragment access and store input types do not match");

        Storer::exec(data, PreStoreXForm::exec(frag.mAccess), ldm);
    }

    template <typename FragT, typename DataT>
    ROCWMMA_DEVICE void
        store_matrix_sync(DataT* data, FragT const& frag, uint32_t ldm, layout_t layout)
    {
        // Dispatch on layout type
        if(layout == layout_t::mem_row_major)
        {
            store_matrix_sync(data, apply_data_layout<row_major>(frag), ldm);
        }
        else
        {
            store_matrix_sync(data, apply_data_layout<col_major>(frag), ldm);
        }
    }

    template <typename FragA, typename FragB, typename FragAccumIn, typename FragAccumOut>
    ROCWMMA_DEVICE void mma_sync(FragAccumOut& d, FragA const& a, FragB const& b, FragAccumIn& c)
    {
        // Get the MmaConfig
        using MmaConfig = GetMmaConfig_t<decay_t<FragA>,
                                         decay_t<FragB>,
                                         decay_t<FragAccumIn>,
                                         decay_t<FragAccumOut>>;

        // Transforms
        using XA = typename MmaConfig::PreMmaXFormA;
        using XB = typename MmaConfig::PreMmaXFormB;
        using XC = typename MmaConfig::PreMmaXFormC;
        using XD = typename MmaConfig::PostMmaXFormD;

        // PackUtil
        using PackA = typename MmaConfig::PackA;
        using PackB = typename MmaConfig::PackB;
        using PackC = typename MmaConfig::PackC;
        using PackD = typename MmaConfig::PackD;

        using Mma = typename MmaConfig::Mma;

        // 1. Perform input pre-ops on A, B, Acc (unpacked mAccess)
        // 2. Mma (packed)
        // 3. Perform acc post-op on Acc
        // 4. Pack back to register
        d.mAccess = XD::exec(PackD::unpack(Mma::exec(PackA::pack(XA::exec(a.mAccess)),
                                                     PackB::pack(XB::exec(b.mAccess)),
                                                     PackC::pack(XC::exec(c.mAccess)))));
    }

    ROCWMMA_DEVICE ROCWMMA_INLINE void synchronize_workgroup()
    {
        __syncthreads();
    }

} // namespace rocwmma

#endif // ROCWMMA_API_IMPL_HPP
