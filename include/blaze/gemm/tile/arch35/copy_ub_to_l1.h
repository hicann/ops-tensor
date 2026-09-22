/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file copy_ub_to_l1.h
 * \brief FP16/BF16 UB-to-L1 copy for padded ZN and NZ source layouts.
 */
#pragma once

#include "tensor_api/tensor.h"
#include "blaze/gemm/utils/layout_struct.h"

namespace Blaze::Gemm::Tile {

// Each outer row group (ZN) or column group (NZ) must contain a contiguous span.
// The source pattern selects the grouping; destination strides must describe
// the same contiguous groups in L1. Only gaps may differ. Singleton
// dimensions may have zero stride, and source/destination shapes need not match.
// Callers provide nonempty, compatible layouts, 32-byte-aligned spans/pitches,
// nonnegative gaps fitting the instruction fields, and sufficient buffer capacity.
struct CopyPaddedUBToL1 {
    template <typename Traits, const Traits& traits, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void Copy(const DstTensor& dst, const SrcTensor& src)
    {
        // Deduce unqualified scalars through address-space-qualified pointers;
        // Tensor element types themselves can retain CCE address-space qualifiers.
        using DstType = decltype(GetL1Scalar(dst.data().get()));
        using SrcType = decltype(GetUbScalar(src.data().get()));
        static_assert(AscendC::Std::is_same_v<DstType, SrcType>,
                      "UB-to-L1 copy requires matching source and destination types");
        static_assert(AscendC::Std::is_same_v<DstType, half> || AscendC::Std::is_same_v<DstType, bfloat16_t>,
                      "UB-to-L1 copy supports only FP16/BF16 elements");

        using SrcPattern = asc::te::get_layout_pattern<typename SrcTensor::layout_type>;
        constexpr bool IS_ZN_ROW_PADDING = AscendC::Std::is_same_v<SrcPattern, Blaze::Gemm::zn_row_padding_layout_ptn>;
        constexpr bool IS_NZ_COL_PADDING = AscendC::Std::is_same_v<SrcPattern, Blaze::Gemm::nz_col_padding_layout_ptn>;
        static_assert(IS_ZN_ROW_PADDING || IS_NZ_COL_PADDING,
                      "UB-to-L1 copy requires a supported padding source layout");

        if constexpr (IS_ZN_ROW_PADDING) {
            CopyZnRowPaddingToL1<DstType>(dst, src);
        } else if constexpr (IS_NZ_COL_PADDING) {
            CopyNzColPaddingToL1<DstType>(dst, src);
        }
    }

private:
    static constexpr int64_t BLOCK_BYTES = 32;

    template <typename Scalar>
    __aicore__ static Scalar GetL1Scalar(__cbuf__ Scalar*);

    template <typename Scalar>
    __aicore__ static Scalar GetUbScalar(__ubuf__ Scalar*);

    template <typename Scalar, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void CopyZnRowPaddingToL1(const DstTensor& dst, const SrcTensor& src)
    {
        constexpr int64_t ELEMS_PER_BLOCK = BLOCK_BYTES / sizeof(Scalar);
        const auto srcShape = asc::te::get_shape(src.layout());
        const auto srcStride = asc::te::get_stride(src.layout());
        const auto dstStride = asc::te::get_stride(dst.layout());

        // Each K group contains k0*n0*n1 contiguous weight elements; lengths and gaps use 32-byte units.
        const auto rowShape = AscendC::Std::get<0>(srcShape);
        const auto columnShape = AscendC::Std::get<1>(srcShape);
        const int64_t rowInnerSize = static_cast<int64_t>(AscendC::Std::get<0>(rowShape));
        const int64_t columnInnerSize = static_cast<int64_t>(AscendC::Std::get<0>(columnShape));
        const int64_t columnOuterSize = static_cast<int64_t>(AscendC::Std::get<1>(columnShape));
        const int64_t blockCount = static_cast<int64_t>(AscendC::Std::get<1>(rowShape));
        const int64_t blockLen = rowInnerSize * columnInnerSize * columnOuterSize / ELEMS_PER_BLOCK;
        const int64_t srcBlockSpan = static_cast<int64_t>(AscendC::Std::get<1>(AscendC::Std::get<0>(srcStride))) /
                                     ELEMS_PER_BLOCK;
        const int64_t dstBlockSpan = static_cast<int64_t>(AscendC::Std::get<1>(AscendC::Std::get<0>(dstStride))) /
                                     ELEMS_PER_BLOCK;
        const int64_t srcGap = srcBlockSpan - blockLen;
        const int64_t dstGap = dstBlockSpan - blockLen;

        asc_copy_ub2l1((__cbuf__ void*)dst.data().get(), (__ubuf__ void*)src.data().get(),
                       static_cast<uint16_t>(blockCount), static_cast<uint16_t>(blockLen),
                       static_cast<uint16_t>(srcGap), static_cast<uint16_t>(dstGap));
    }

    template <typename Scalar, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void CopyNzColPaddingToL1(const DstTensor& dst, const SrcTensor& src)
    {
        constexpr int64_t ELEMS_PER_BLOCK = BLOCK_BYTES / sizeof(Scalar);
        const auto srcShape = asc::te::get_shape(src.layout());
        const auto srcStride = asc::te::get_stride(src.layout());
        const auto dstStride = asc::te::get_stride(dst.layout());

        // Each N group contains k0*k1*n0 contiguous weight elements; lengths and gaps use 32-byte units.
        const auto rowShape = AscendC::Std::get<0>(srcShape);
        const auto columnShape = AscendC::Std::get<1>(srcShape);
        const int64_t rowInnerSize = static_cast<int64_t>(AscendC::Std::get<0>(rowShape));
        const int64_t rowOuterSize = static_cast<int64_t>(AscendC::Std::get<1>(rowShape));
        const int64_t columnInnerSize = static_cast<int64_t>(AscendC::Std::get<0>(columnShape));
        const int64_t blockCount = static_cast<int64_t>(AscendC::Std::get<1>(columnShape));
        const int64_t blockLen = rowInnerSize * rowOuterSize * columnInnerSize / ELEMS_PER_BLOCK;
        const int64_t srcBlockSpan = static_cast<int64_t>(AscendC::Std::get<1>(AscendC::Std::get<1>(srcStride))) /
                                     ELEMS_PER_BLOCK;
        const int64_t dstBlockSpan = static_cast<int64_t>(AscendC::Std::get<1>(AscendC::Std::get<1>(dstStride))) /
                                     ELEMS_PER_BLOCK;
        const int64_t srcGap = srcBlockSpan - blockLen;
        const int64_t dstGap = dstBlockSpan - blockLen;

        asc_copy_ub2l1((__cbuf__ void*)dst.data().get(), (__ubuf__ void*)src.data().get(),
                       static_cast<uint16_t>(blockCount), static_cast<uint16_t>(blockLen),
                       static_cast<uint16_t>(srcGap), static_cast<uint16_t>(dstGap));
    }
};

} // namespace Blaze::Gemm::Tile

namespace asc::te {

template <typename Traits>
struct copy_traits<Blaze::Gemm::Tile::CopyPaddedUBToL1, Traits>
    : public copy_traits<Blaze::Gemm::Tile::CopyPaddedUBToL1, Traits, Blaze::Gemm::Tile::CopyPaddedUBToL1, Traits> {};

template <>
struct copy_traits<Blaze::Gemm::Tile::CopyPaddedUBToL1>
    : public copy_traits<Blaze::Gemm::Tile::CopyPaddedUBToL1, ub_to_l1_trait_default> {};

} // namespace asc::te
