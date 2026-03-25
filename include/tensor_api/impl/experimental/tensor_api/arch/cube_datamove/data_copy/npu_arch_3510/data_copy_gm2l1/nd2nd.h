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
 * \file nd2nd.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2L1_ND2ND_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2L1_ND2ND_H

#include "impl/experimental/tensor_api/arch/cube_datamove/data_copy/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {

class CopyGmToCbufAlignV2ND {
public:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline static void Run(const T& dst, const U& src)
    {
        DataCopyImpl<trait, T, U>(dst, src);
    }

private:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline static constexpr void CheckTemplate()
    {
        CheckFormat::CheckNDTemplate<T>();
        CheckFormat::CheckNDTemplate<U>();
        CheckDataTypeFor3510::CheckGm2L1AlignV2NDDataType<T, U>();
    }

    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline static void DataCopyImpl(const T& dst, const U& src)
    {
        CheckTemplate<trait, T, U>();

        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();

        auto srcShapeRows = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        auto srcShapeColumns = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        auto dstShapeColumns = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);
        auto dstStrideRows = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);
        auto srcStrideRows = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);

        using type = typename U::elementType;
        uint8_t cacheMode = GetCacheModeFromTensor(src);

        using ShapeRow1 = typename GetFourDimType<U, AttrInfo::SHAPE, AttrInfo::ROW, 1>::type;
        // compact mode, dst_stride equals burst_len, padding cnt is zero
        // src and dst contiguous case, can directly copy without padding, only one row copy is needed
        // the src is 1D tensor with only column shape, we can directly copy with burst len as column shape
        if constexpr (Std::is_constant<1, ShapeRow1>::value) {
            CopyGmToCbufAlignV2Base::DataCopy(dst, src, 1, srcShapeColumns * sizeof(type), 0, 0, cacheMode, 0,
                                              srcShapeColumns * sizeof(type));
            return;
        }

        // lprp mode, dst_stride % C0_SIZE should be 0
        // multi rows copy, dst non-contiguous case
        constexpr uint8_t blockSizeElem = C0_SIZE<> / sizeof(type);
        uint32_t validCol = srcShapeColumns;
        uint32_t gapElem = dstShapeColumns - validCol;
        uint8_t padCount = gapElem % blockSizeElem;

        uint32_t blockCount = srcShapeRows;
        uint32_t blockLen = srcShapeColumns * sizeof(type);
        uint64_t srcStride = srcStrideRows * sizeof(type);
        uint32_t dstStride = dstStrideRows * sizeof(type);

        uint8_t leftPaddingCnt = 0;
        uint8_t rightPaddingCnt = padCount;

        if ((srcShapeRows == 1)
            || (srcStrideRows == srcShapeColumns && dstStrideRows == dstShapeColumns
                && srcStrideRows == dstStrideRows)) {
            // compact mode
            blockCount = 1;
            // must use srcShape, there is a scenario of small to large, using dstShape will cause src out of bound
            blockLen = srcShapeRows * srcShapeColumns * sizeof(type);
            leftPaddingCnt = 0;
            rightPaddingCnt = 0;
            srcStride = 0;
            dstStride = blockLen;
        }
        CopyGmToCbufAlignV2Base::DataCopy(dst, src, blockCount, blockLen, leftPaddingCnt, rightPaddingCnt, cacheMode,
                                          srcStride, dstStride);
    }
};
} // namespace Te
} // namespace AscendC

#endif
