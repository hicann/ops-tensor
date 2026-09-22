/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "tensor_api/tensor.h"
#include "kernel_operator.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"

namespace Blaze {
namespace Gemm {

// FP16/BF16 weight UB layouts in logical (K,N) coordinates, shape ((k0,k1),(n0,n1)).
// ZN: stride ((1,pitch),(k0,n0*k0)); NZ: stride ((n0,k0*n0),(1,pitch)).
// A full tile adds one k0/n0 padding segment between groups. Tail tiles retain
// the VF allocation pitch instead of recomputing it from the effective shape.
struct zn_row_padding_layout_ptn {};
struct nz_col_padding_layout_ptn {};

// Construct the physical (K,N) region copied from FP16/BF16 UB storage.
// groupPitch is in elements and describes the allocated VF group spacing,
// including padding. It must not shrink when the effective tile is a tail.
template <typename T>
struct ZnRowPaddingUBLayout {
    static_assert(AscendC::Std::is_same_v<T, half> || AscendC::Std::is_same_v<T, bfloat16_t>,
                  "ZnRowPaddingUBLayout expects FP16/BF16 elements");

    __aicore__ inline auto operator()(int64_t kSize, int64_t nSize, int64_t groupPitch) const
    {
        constexpr int64_t C0 = asc::te::c0_element<T>;
        using Block = AscendC::Std::Int<C0>;
        using One = AscendC::Std::Int<1>;
        // Preserve the valid N length; align K to a 32-byte group (C0 = 16 elements).
        auto shape = asc::te::make_shape(asc::te::make_shape(Block{}, CeilDiv(kSize, C0)),
                                         asc::te::make_shape(One{}, nSize));
        auto stride = asc::te::make_stride(asc::te::make_stride(One{}, groupPitch),
                                           asc::te::make_stride(Block{}, Block{}));
        return asc::te::make_pattern_layout<zn_row_padding_layout_ptn, asc::te::layout_trait_default<T>>(shape, stride);
    }
};

template <typename T>
struct NzColPaddingUBLayout {
    static_assert(AscendC::Std::is_same_v<T, half> || AscendC::Std::is_same_v<T, bfloat16_t>,
                  "NzColPaddingUBLayout expects FP16/BF16 elements");

    __aicore__ inline auto operator()(int64_t kSize, int64_t nSize, int64_t groupPitch) const
    {
        constexpr int64_t C0 = asc::te::c0_element<T>;
        using Block = AscendC::Std::Int<C0>;
        using One = AscendC::Std::Int<1>;
        // Preserve the valid K length; align N to a 32-byte group (C0 = 16 elements).
        auto shape = asc::te::make_shape(asc::te::make_shape(One{}, kSize),
                                         asc::te::make_shape(Block{}, CeilDiv(nSize, C0)));
        auto stride = asc::te::make_stride(asc::te::make_stride(Block{}, Block{}),
                                           asc::te::make_stride(One{}, groupPitch));
        return asc::te::make_pattern_layout<nz_col_padding_layout_ptn, asc::te::layout_trait_default<T>>(shape, stride);
    }
};

// Layout for 8-bit weight tiles in Unified Buffer (ZN conversion path).
//
// This is a specialized UB layout for 8-bit weight data.  Unlike the standard
// FRACTAL_FIXED=16, this uses N0=VEC_REG_ELEM/C0=8 because the weight data is
// produced by Vector unit writes in 256-element (VEC_REG_ELEM) chunks.
// The InnerStride template parameter exposes inter-chunk spacing control to
// the MTE3 copy path, which is not expressible with built-in layout formulas.
template <typename T>
struct Weight8BitZnToZnUBLayout;

template <typename T, uint64_t InnerStride>
struct Weight8BitUBLayout {
    __aicore__ inline decltype(auto) operator()(int64_t kSize, int64_t nSize)
    {
        return Weight8BitZnToZnUBLayout<T>{}(kSize, nSize, InnerStride);
    }
};

template <typename T>
struct Weight8BitZnToZnUBLayout {
    static_assert(sizeof(T) == 1, "Weight8BitDnToZnUBLayout expects an 8-bit element type");

    __aicore__ inline decltype(auto) operator()(int64_t kSize, int64_t nSize, uint64_t innerStride) const
    {
        static constexpr int64_t C0 = 32;
        static constexpr int64_t VEC_REG_ELEM = 256;
        static constexpr int64_t N0 = VEC_REG_ELEM / C0;
        static constexpr int64_t STRIDE_UNIT = 1;
        int64_t k1 = CeilDiv(kSize, C0);
        int64_t n1 = CeilDiv(nSize, N0);

        // Shape: ((C0, k1), (N0, n1))
        auto shape = asc::te::make_shape(asc::te::make_shape(AscendC::Std::Int<C0>{}, k1),
                                         asc::te::make_shape(AscendC::Std::Int<N0>{}, n1));
        // Stride: (MakeStride(1, n1 * InnerStride), MakeStride(C0, InnerStride))
        auto stride = asc::te::make_stride(asc::te::make_stride(AscendC::Std::Int<STRIDE_UNIT>{}, n1 * innerStride),
                                           asc::te::make_stride(AscendC::Std::Int<C0>{}, innerStride));
        using Trait = asc::te::layout_trait<AscendC::Std::ignore_t, AscendC::Std::Int<C0>>;
        return asc::te::make_pattern_layout<Weight8BitZnToZnUbLayoutPtn, Trait>(shape, stride);
    }
};

// Specialized UB layout emitted when the input weight format is ND. The VF
// output is ZN-like, but each K32 slab has one extra 32-byte block in its N
// stride to avoid UB bank conflicts. The UB-to-L1 copy removes that gap.
template <typename T>
struct Weight8BitDnToZnUBLayout {
    static_assert(sizeof(T) == 1, "Weight8BitDnToZnUBLayout expects an 8-bit element type");

    __aicore__ inline auto operator()(int64_t kSize, int64_t nSize) const
    {
        static constexpr int64_t C0 = asc::te::c0_element<T>;
        static constexpr int64_t EXTRA_N_BLOCK = 1;
        int64_t k1 = CeilDiv(kSize, C0);
        int64_t nStride = static_cast<int64_t>(Align16(static_cast<uint64_t>(nSize))) + EXTRA_N_BLOCK;
        auto shape = asc::te::make_shape(asc::te::make_shape(AscendC::Std::Int<C0>{}, k1),
                                         asc::te::make_shape(AscendC::Std::Int<1>{}, nSize));
        auto stride = asc::te::make_stride(asc::te::make_stride(AscendC::Std::Int<1>{}, nStride * C0),
                                           asc::te::make_stride(AscendC::Std::Int<C0>{}, AscendC::Std::Int<C0>{}));
        using Trait = asc::te::layout_trait<AscendC::Std::ignore_t, AscendC::Std::Int<C0>>;
        return asc::te::make_pattern_layout<Weight8BitDnToZnUbLayoutPtn, Trait>(shape, stride);
    }
};

} // namespace Gemm
} // namespace Blaze
