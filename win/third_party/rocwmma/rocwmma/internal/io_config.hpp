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
#ifndef ROCWMMA_IO_CONFIG_HPP
#define ROCWMMA_IO_CONFIG_HPP

#include "broadcast.hpp"
#include "io_bounds_ctrl.hpp"
#include "io_layout.hpp"
#include "io_shape.hpp"
#include "io_tile.hpp"
#include "io_traits.hpp"
#include "layout/register_layout_transforms.hpp"
#include "opaque_load.hpp"
#include "opaque_store.hpp"
#include "pack_util.hpp"
#include "types.hpp"

namespace rocwmma
{

    /**
     * \defgroup Rocwmma_ioconf ROCWMMA IOConfig
     * @brief ROCWMMA cooperative fragment input and output configurations
     * @{
     */

    //! @struct IOConfig
    //! @brief Definition of fragment input / output configurations
    //!         in specific matrix context.
    //! @tparam MatrixT Matrix fragment context
    //! @tparam BlockM/N/K block dimensions
    //! @tparam DataT data type
    //! @tparam DataLayoutT in-memory layout as col_major or row_major
    //! @tparam Scheduler wave-wise scheduler
    //! @param IOBounds a rectangular bounds condition handler
    //! @param IOTile a handler to pad FragMNK dimensions as needed for block-wise decomp
    //! @param IOShape dimensional properties of the fragment
    //! @param IOLayout 1d and 2d layouts of the fragment
    //! @param IOTraits meta-properties for input and output of the fragment
    //! @param MappingUtil global mapping utility for current fragment
    //! @param PackUtil utility for packing / unpacking fragment data
    //! @param Broadcaster utility for broadcasting a scalar element to the entire vector
    //! @param Loader issues load instructions for raw fragment data
    //! @param PostLoadXForm handler to transform from memory to register layout
    //! @param PreStoreXForm handler to transform from register to memory layout
    //! @param Storer Issues store instructions for raw fragment data
    template <typename MatrixT,
              uint32_t FragM,
              uint32_t FragN,
              uint32_t FragK,
              typename DataT,
              typename DataLayoutT,
              typename Scheduler>
    struct IOConfig
    {
        // The specific size of the requested fragment
        using IOBounds = IOShape<MatrixT, FragM, FragN, FragK>;

        // Internally, block-wise decomposition dictates the need for quantized block dimensions.
        // IOTile will determine ideal block size.
        using IOTile  = IOTile<FragM, FragN, FragK, DataT>;
        using IOShape = IOShape<MatrixT, IOTile::BlockM, IOTile::BlockN, IOTile::BlockK>;

        // Bounds control is needed if the original frag size doesn't match block size quantizations.
        static constexpr bool IOBoundsCtrlRequired
            = (FragM != IOTile::BlockM) || (FragN != IOTile::BlockN) || (FragK != IOTile::BlockK);

        // Using quantized block dimensions, determine the layout characteristics we will use with
        // this fragment.
        using IOLayout
            = IOLayoutInt<MatrixT, IOShape::BlockDim, IOShape::KDim, DataT, DataLayoutT, Scheduler>;

        using IOTraits = IOTraits<IOShape::BlockDim, IOShape::KDim, DataT, IOLayout::VW>;

        using MappingUtil
            = MappingUtil<IOShape::BlockHeight, IOShape::BlockWidth, DataT, DataLayoutT>;

        using PackUtil    = PackUtil<DataT>;
        using Broadcaster = Broadcast<DataT, IOTraits::UnpackedSize>;

        // When loading for Mma, we need to replace clipped areas with 0's to ensure correctness.
        using IOBoundsCtrlLoad = conditional_t<
            IOBoundsCtrlRequired,
            IOBoundsCtrl::ClipAndReplace2d<IOBounds::BlockHeight, IOBounds::BlockWidth>,
            IOBoundsCtrl::Default>;

        using Loader = OpaqueLoad<typename IOLayout::DataLayout,
                                  typename IOLayout::MatrixLayout,
                                  IOBoundsCtrlLoad,
                                  Scheduler>;

        // TODO: remove the waveCount dependency by adjusting the buffer size statically.
        using PostLoadXForm = register_layout_transform<typename IOLayout::StorageLayout,
                                                        typename IOLayout::FragmentLayout>;

        using PreStoreXForm = register_layout_transform<typename IOLayout::FragmentLayout,
                                                        typename IOLayout::StorageLayout>;

        // Storage requires only clipping.
        using IOBoundsCtrlStore
            = conditional_t<IOBoundsCtrlRequired,
                            IOBoundsCtrl::Clip2d<IOBounds::BlockHeight, IOBounds::BlockWidth>,
                            IOBoundsCtrl::Default>;

        using Storer = OpaqueStore<typename IOLayout::DataLayout,
                                   typename IOLayout::MatrixLayout,
                                   IOBoundsCtrlStore,
                                   Scheduler>;
    };

    //! @note Matrix C/D (accumulator) with undetermined DataLayout
    //!
    //! Fewer specific indications for matrix data geometry I/O, however
    //! general IOShape, PackUtil, Broadcast still available.
    template <uint32_t FragM, uint32_t FragN, uint32_t FragK, typename DataT, typename Scheduler>
    struct IOConfig<accumulator, FragM, FragN, FragK, DataT, void, Scheduler>
    {
        using IOTile  = IOTile<FragM, FragN, FragK, DataT>;
        using IOShape = IOShape<accumulator, IOTile::BlockM, IOTile::BlockN, IOTile::BlockK>;
        using IOLayout
            = IOLayoutInt<accumulator, IOShape::BlockDim, IOShape::KDim, DataT, void, Scheduler>;
        using IOTraits    = IOTraits<IOShape::BlockDim, IOShape::KDim, DataT>;
        using PackUtil    = PackUtil<DataT>;
        using Broadcaster = Broadcast<DataT, IOTraits::UnpackedSize>;
    };
    /** @}*/

} // namespace rocwmma

#endif // ROCWMMA_IO_CONFIG_HPP
