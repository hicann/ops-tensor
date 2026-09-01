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

#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "blaze/gemm/utils/common_utils.h"

namespace Blaze::Gemm::Tile {

// Lookup addresses for gathering one 16-N by 16-K-group ScaleB fragment from
// the padded ND staging buffer.  Value(i, j) = i + j * 80 in uint16 units.
static constexpr volatile __gm__ uint16_t MX_SCALE_TRANS_ID[128] = {
    0,    80,   160,  240,  320,  400,  480,  560,  640,  720,  800,  880,  960,  1040, 1120, 1200, 1,    81,   161,
    241,  321,  401,  481,  561,  641,  721,  801,  881,  961,  1041, 1121, 1201, 2,    82,   162,  242,  322,  402,
    482,  562,  642,  722,  802,  882,  962,  1042, 1122, 1202, 3,    83,   163,  243,  323,  403,  483,  563,  643,
    723,  803,  883,  963,  1043, 1123, 1203, 4,    84,   164,  244,  324,  404,  484,  564,  644,  724,  804,  884,
    964,  1044, 1124, 1204, 5,    85,   165,  245,  325,  405,  485,  565,  645,  725,  805,  885,  965,  1045, 1125,
    1205, 6,    86,   166,  246,  326,  406,  486,  566,  646,  726,  806,  886,  966,  1046, 1126, 1206, 7,    87,
    167,  247,  327,  407,  487,  567,  647,  727,  807,  887,  967,  1047, 1127, 1207};

struct MxScaleTransposeParams {
    __ubuf__ uint16_t* input;
    __ubuf__ uint16_t* output;
    __ubuf__ uint16_t* transId;
    uint16_t nBlockCount;
    uint16_t groupBlockCount;
    uint16_t outputNBlockStride;
    uint16_t tailStoreCount;
};

class MxScaleTranspose {
public:
    __aicore__ inline static void Transpose(__ubuf__ uint16_t* src, __ubuf__ uint16_t* dst, __ubuf__ uint16_t* transId,
                                            uint64_t nSize, uint64_t scaleKSize);

private:
    static __simd_vf__ inline void TransposeVf(MxScaleTransposeParams params)
    {
        namespace MicroAPI = AscendC::MicroAPI;
        constexpr uint16_t INPUT_ROW_STRIDE_U16 = 80;
        constexpr uint16_t OUTPUT_GROUP_STRIDE_U16 = 128;
        MicroAPI::RegTensor<uint16_t> scale;
        MicroAPI::RegTensor<uint16_t> transId;
        MicroAPI::LoadAlign(transId, params.transId);
        MicroAPI::MaskReg mask = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        for (uint16_t nBlock = 0; nBlock < params.nBlockCount; ++nBlock) {
            for (uint16_t groupBlock = 0; groupBlock < params.groupBlockCount; ++groupBlock) {
                MicroAPI::Gather(scale, params.input + nBlock * BLOCK_CUBE * INPUT_ROW_STRIDE_U16 + groupBlock * 8,
                                 transId, mask);
                MicroAPI::AddrReg outputAddr = MicroAPI::CreateAddrReg<uint16_t>(nBlock, params.outputNBlockStride,
                                                                                 groupBlock, OUTPUT_GROUP_STRIDE_U16);
                if (groupBlock + 1 == params.groupBlockCount && params.tailStoreCount != 128) {
                    uint32_t tailCount = params.tailStoreCount;
                    MicroAPI::MaskReg tailMask = MicroAPI::UpdateMask<uint16_t>(tailCount);
                    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM_B16>(params.output, scale, outputAddr,
                                                                                       tailMask);
                } else {
                    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM_B16>(params.output, scale, outputAddr,
                                                                                       mask);
                }
            }
        }
    }
};

// ScaleBDN is physically N-major: each N row contains all K-group bytes.
// The UB staging row is widened to 160 bytes (128 data + 32 bank-conflict
// padding), matching the gather lookup table above.
template <typename GmScaleTensor>
__aicore__ inline void CopyMxScaleGmToUb(const GmScaleTensor& src, __ubuf__ uint8_t* dst, uint64_t nSize,
                                         uint64_t scaleKSize)
{
    using SrcLayoutPattern = AscendC::Te::GetLayoutPattern<typename GmScaleTensor::layoutType>;
    static_assert(AscendC::Std::is_same_v<SrcLayoutPattern, AscendC::Te::ScaleBDNLayoutPtn>,
                  "MX ScaleB staging requires a ScaleBDN GM tensor");
    static_assert(sizeof(typename GmScaleTensor::elementType) == 1, "MX ScaleB staging requires an 8-bit element type");

    const auto& srcStride = src.Layout().Stride();
    uint64_t srcNStride = AscendC::Std::get<1>(AscendC::Std::get<1>(srcStride));
    uint8_t cacheMode = src.Engine().GetCacheMode();
    asc_copy_gm2ub_align(dst, (__gm__ uint8_t*)src.Data().Get(), static_cast<uint16_t>(nSize),
                         static_cast<uint32_t>(scaleKSize), 0, 0, false, cacheMode, srcNStride, 160);
}

__aicore__ inline void MxScaleTranspose::Transpose(__ubuf__ uint16_t* src, __ubuf__ uint16_t* dst,
                                                   __ubuf__ uint16_t* transId, uint64_t nSize, uint64_t scaleKSize)
{
    uint64_t scaleKTail = scaleKSize % 16;
    MxScaleTransposeParams params{src,
                                  dst,
                                  transId,
                                  static_cast<uint16_t>(CeilDiv(nSize, static_cast<uint64_t>(16))),
                                  static_cast<uint16_t>(CeilDiv(scaleKSize, static_cast<uint64_t>(16))),
                                  static_cast<uint16_t>((scaleKSize * 16) >> 1),
                                  static_cast<uint16_t>(scaleKTail == 0 ? 128 : (scaleKTail * 16) / sizeof(uint16_t))};
    // A direct call keeps this VF in the caller's MTE2/V/MTE3 pipeline.
    TransposeVf(params);
}

__aicore__ inline void CopyMxScaleUbToL1(__cbuf__ void* dst, __ubuf__ void* src, uint32_t sizeBytes)
{
    asc_copy_ub2l1(dst, src, sizeBytes);
}

} // namespace Blaze::Gemm::Tile
