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
#ifndef ROCWMMA_IO_BOUNDS_CTRL_HPP
#define ROCWMMA_IO_BOUNDS_CTRL_HPP

#include "utility/vector.hpp"

namespace rocwmma
{
    namespace IOBoundsCtrl
    {
        //! @struct Default
        //! @brief Default behaviour policy is to not check or apply IO Bounds control
        struct Default
        {
            static constexpr uint32_t BoundX = 0u;
            static constexpr uint32_t BoundY = 0u;

            template <typename... Ts>
            ROCWMMA_DEVICE static constexpr inline auto checkBounds(Ts&&... ts)
            {
                // Pass-through
                return false;
            }

            template <typename... Ts>
            ROCWMMA_DEVICE static constexpr inline auto exec(Ts&&... ts)
            {
                // No action
            }
        };

        //! @struct Clip2d
        //! @brief Clipping policy checks rectangular bounds and takes no further action
        //! @tparam ClipX Bounding box height
        //! @tparam ClipY Bounding box width
        template <uint32_t ClipX, uint32_t ClipY>
        struct Clip2d
        {
            static constexpr uint32_t BoundX = ClipX;
            static constexpr uint32_t BoundY = ClipY;

            template <typename Coord2d, typename Size2d, typename... Ts>
            ROCWMMA_DEVICE static constexpr inline auto
                checkBounds(Coord2d&& baseOffset2d, Size2d&& size2d, Ts&&... ts)
            {
                // Mask high if the data area exceeds bounds.
                return vector_reduce_or((baseOffset2d + size2d) > make_coord2d(BoundX, BoundY));
            }

            template <typename... Ts>
            ROCWMMA_DEVICE static constexpr inline auto exec(Ts&&... ts)
            {
                // No action
            }
        };

        //! @struct ClipAndReplace2d
        //! @brief Clipping policy checks rectangular bounds and replaces the transaction buffer with a value
        //! @tparam ClipX Bounding box height
        //! @tparam ClipY Bounding box width
        template <uint32_t ClipX, uint32_t ClipY>
        struct ClipAndReplace2d : public Clip2d<ClipX, ClipY>
        {
        private:
            // Static override of Clip2d
            using Clip2d<ClipX, ClipY>::exec;

        public:
            template <typename BuffT, typename ScalarT, typename... Ts>
            ROCWMMA_DEVICE static constexpr inline auto
                exec(BuffT& buff, ScalarT&& clipVal, Ts&&... ts)
            {
                // Fill the buffer with clipVal
                using BuffTraits = VecTraits<BuffT>;
                buff = make_vector<typename BuffTraits::DataT, BuffTraits::size()>(clipVal);
            }
        };

    } // namespace IOBoundsCtrl

} // namespace rocwmma

#endif // ROCWMMA_IO_BOUNDS_CTRL_HPP
