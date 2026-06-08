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

#ifndef ROCWMMA_VECTOR_TYPES_HPP
#define ROCWMMA_VECTOR_TYPES_HPP

#include "types.hpp"
#include "vector.hpp"

namespace rocwmma
{
    /*! \class VecT
    *  \brief  HIP vector class
    *  @tparam DataT vector data type
    *  @tparam Rank vector size
    */
    template <typename DataT, uint32_t Rank>
    using VecT = HIP_vector_type<DataT, Rank>;

    // MFMA vector registers
    using VRegI8x1  = VecT<int8_t, 1>; // Single i8 register
    using VRegI8x2  = VecT<int8_t, 2>; // Two i8 registers
    using VRegI8x4  = VecT<int8_t, 4>; // ...
    using VRegI8x8  = VecT<int8_t, 8>; //
    using VRegI8x16 = VecT<int8_t, 16>; //
    using VRegI8x32 = VecT<int8_t, 32>; // 32 i8 registers

    using VRegI32x1  = VecT<int32_t, 1>; // Single i32 register
    using VRegI32x2  = VecT<int32_t, 2>; // Two i32 registers
    using VRegI32x4  = VecT<int32_t, 4>; // ...
    using VRegI32x8  = VecT<int32_t, 8>; //
    using VRegI32x16 = VecT<int32_t, 16>; //
    using VRegI32x32 = VecT<int32_t, 32>; // 32 i32 registers

    using VRegI64x1  = VecT<int64_t, 1>; // Single i64 register
    using VRegI64x2  = VecT<int64_t, 2>; // Two i64 registers
    using VRegI64x4  = VecT<int64_t, 4>; // ...
    using VRegI64x8  = VecT<int64_t, 8>; //
    using VRegI64x16 = VecT<int64_t, 16>; //
    using VRegI64x32 = VecT<int64_t, 32>; // 32 i64 registers

    using VRegF16x1  = VecT<float16_t, 1>; // Single f16 register
    using VRegF16x2  = VecT<float16_t, 2>; // Two f16 registers
    using VRegF16x4  = VecT<float16_t, 4>; // ...
    using VRegF16x8  = VecT<float16_t, 8>; //
    using VRegF16x16 = VecT<float16_t, 16>; //
    using VRegF16x32 = VecT<float16_t, 32>; // 32 f16 registers

    using VRegF32x1  = VecT<float32_t, 1>; // Single f32 register
    using VRegF32x2  = VecT<float32_t, 2>; // Two f32 registers
    using VRegF32x4  = VecT<float32_t, 4>; // ...
    using VRegF32x8  = VecT<float32_t, 8>; //
    using VRegF32x16 = VecT<float32_t, 16>; //
    using VRegF32x32 = VecT<float32_t, 32>; // 32 f32 registers

    using VRegF64x1  = VecT<float64_t, 1>; // Single f64 register
    using VRegF64x2  = VecT<float64_t, 2>; // Two f64 registers
    using VRegF64x4  = VecT<float64_t, 4>; // ...
    using VRegF64x8  = VecT<float64_t, 8>; //
    using VRegF64x16 = VecT<float64_t, 16>; //
    using VRegF64x32 = VecT<float64_t, 32>; // 32 f64 registers

    // Acc registers
    using AccRegI32x1  = VecT<int32_t, 1>;
    using AccRegI32x2  = VecT<int32_t, 2>;
    using AccRegI32x4  = VecT<int32_t, 4>;
    using AccRegI32x8  = VecT<int32_t, 8>;
    using AccRegI32x16 = VecT<int32_t, 16>;
    using AccRegI32x32 = VecT<int32_t, 32>;

    using AccRegF32x1  = VecT<float32_t, 1>;
    using AccRegF32x2  = VecT<float32_t, 2>;
    using AccRegF32x4  = VecT<float32_t, 4>;
    using AccRegF32x8  = VecT<float32_t, 8>;
    using AccRegF32x16 = VecT<float32_t, 16>;
    using AccRegF32x32 = VecT<float32_t, 32>;

    using AccRegF64x1  = VecT<float64_t, 1>;
    using AccRegF64x2  = VecT<float64_t, 2>;
    using AccRegF64x4  = VecT<float64_t, 4>;
    using AccRegF64x8  = VecT<float64_t, 8>;
    using AccRegF64x16 = VecT<float64_t, 16>;
    using AccRegF64x32 = VecT<float64_t, 32>;

    using Coord2dDataT = uint32_t;
    using Coord2d      = non_native_vector_base<Coord2dDataT, 2>;

} // namespace rocwmma

#endif // ROCWMMA_VECTOR_TYPES_HPP
