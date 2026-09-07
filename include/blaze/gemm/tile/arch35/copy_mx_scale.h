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
 * \file copy_mx_scale.h
 * \brief Tensor copy and transpose primitives for MX ScaleB staging.
 */
#pragma once

#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "blaze/gemm/utils/common_utils.h"

namespace Blaze::Gemm::Tile {

constexpr uint64_t MX_SCALE_INPUT_ROW_STRIDE = 160;
constexpr uint64_t MX_SCALE_TRANS_ID_SIZE = 128;

// Lookup addresses for gathering one 16-N by 16-K-group ScaleB fragment from
// the padded ND staging buffer.  Value(i, j) = i + j * 80 in uint16 units.
static constexpr volatile __gm__ uint16_t MX_SCALE_TRANS_ID[MX_SCALE_TRANS_ID_SIZE] = {
    0,    80,   160,  240,  320,  400,  480,  560,  640,  720,  800,  880,  960,  1040, 1120, 1200, 1,    81,   161,
    241,  321,  401,  481,  561,  641,  721,  801,  881,  961,  1041, 1121, 1201, 2,    82,   162,  242,  322,  402,
    482,  562,  642,  722,  802,  882,  962,  1042, 1122, 1202, 3,    83,   163,  243,  323,  403,  483,  563,  643,
    723,  803,  883,  963,  1043, 1123, 1203, 4,    84,   164,  244,  324,  404,  484,  564,  644,  724,  804,  884,
    964,  1044, 1124, 1204, 5,    85,   165,  245,  325,  405,  485,  565,  645,  725,  805,  885,  965,  1045, 1125,
    1205, 6,    86,   166,  246,  326,  406,  486,  566,  646,  726,  806,  886,  966,  1046, 1126, 1206, 7,    87,
    167,  247,  327,  407,  487,  567,  647,  727,  807,  887,  967,  1047, 1127, 1207};

// ScaleBDN is physically N-major: each N row contains all K-group bytes.
// The destination layout carries the padded UB row stride used by the VF.
struct CopyGM2UBMxScale {
    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void Copy(const T& dst, const U& src)
    {
        using SrcLayoutPattern = AscendC::Te::GetLayoutPattern<typename U::layoutType>;
        using DstLayoutPattern = AscendC::Te::GetLayoutPattern<typename T::layoutType>;
        static_assert(AscendC::Std::is_same_v<SrcLayoutPattern, AscendC::Te::ScaleBDNLayoutPtn> &&
                          AscendC::Std::is_same_v<DstLayoutPattern, AscendC::Te::ScaleBDNLayoutPtn>,
                      "MX ScaleB staging requires ScaleBDN source and destination layouts");
        static_assert(AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<U>, AscendC::Te::Location::GM> &&
                          AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<T>, AscendC::Te::Location::UB>,
                      "MX ScaleB staging requires GM source and UB destination tensors");
        static_assert(sizeof(typename T::elementType) == 1 && sizeof(typename U::elementType) == 1,
                      "MX ScaleB staging requires 8-bit elements");

        const auto& srcLayout = src.Layout();
        const auto& dstLayout = dst.Layout();
        uint16_t nSize = static_cast<uint16_t>(AscendC::Te::GetTotalColumnShape(srcLayout));
        uint32_t scaleKSize = static_cast<uint32_t>(AscendC::Te::GetTotalRowShape(srcLayout));
        uint64_t srcNStride = AscendC::Std::get<1>(AscendC::Std::get<1>(srcLayout.Stride()));
        uint64_t dstNStride = AscendC::Std::get<1>(AscendC::Std::get<1>(dstLayout.Stride()));
        asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint8_t*>(dst.Data().Get()),
                             reinterpret_cast<__gm__ uint8_t*>(src.Data().Get()), nSize, scaleKSize, 0, 0, false,
                             src.Engine().GetCacheMode(), srcNStride, dstNStride);
    }
};

// Transpose padded ScaleBDN UB data into contiguous NN scale blocks (C0 = 2).
// The source must retain a compile-time 160-byte N-row stride after slicing;
// transId must contain MX_SCALE_TRANS_ID in a contiguous uint16 UB tensor.
struct MxScaleTranspose {
    template <typename T, typename U, typename V>
    __aicore__ inline static void Transpose(const T& dst, const U& src, const V& transId)
    {
        using SrcLayoutPattern = AscendC::Te::GetLayoutPattern<typename U::layoutType>;
        using DstLayoutPattern = AscendC::Te::GetLayoutPattern<typename T::layoutType>;
        using IndexLayoutPattern = AscendC::Te::GetLayoutPattern<typename V::layoutType>;
        using IndexElementType = AscendC::Te::GetAttributeElementType<typename V::elementType*>;
        static_assert(AscendC::Std::is_same_v<SrcLayoutPattern, AscendC::Te::ScaleBDNLayoutPtn> &&
                          AscendC::Std::is_same_v<DstLayoutPattern, AscendC::Te::NNLayoutPtn> &&
                          AscendC::Std::is_same_v<IndexLayoutPattern, AscendC::Te::NDExtLayoutPtn>,
                      "MX scale transpose requires ScaleBDN input, NN output and NDExt indices");
        static_assert(AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<T>, AscendC::Te::Location::UB> &&
                          AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<U>, AscendC::Te::Location::UB> &&
                          AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<V>, AscendC::Te::Location::UB>,
                      "MX scale transpose only supports UB tensors");
        static_assert(sizeof(typename T::elementType) == 1 && sizeof(typename U::elementType) == 1 &&
                          AscendC::Std::is_same_v<IndexElementType, uint16_t>,
                      "MX scale transpose requires 8-bit scales and uint16 indices");
        using SrcNStride = AscendC::Std::remove_cvref_t<decltype(AscendC::Std::get<1>(
            AscendC::Std::get<1>(src.Layout().Stride())))>;
        static_assert(SrcNStride::value == MX_SCALE_INPUT_ROW_STRIDE,
                      "MX scale transpose requires a compile-time 160-byte input row stride");

        uint64_t nSize = AscendC::Te::GetTotalColumnShape(src.Layout());
        uint64_t scaleKSize = AscendC::Te::GetTotalRowShape(src.Layout());
        uint64_t scaleKTail = scaleKSize % BLOCK_CUBE;
        uint64_t dstNBlockStride = AscendC::Std::get<1>(AscendC::Std::get<1>(dst.Layout().Stride()));
        MxScaleTransposeParams params{
            reinterpret_cast<__ubuf__ uint16_t*>(src.Data().Get()),
            reinterpret_cast<__ubuf__ uint16_t*>(dst.Data().Get()),
            reinterpret_cast<__ubuf__ uint16_t*>(transId.Data().Get()),
            static_cast<uint16_t>(CeilDiv(nSize, BLOCK_CUBE)),
            static_cast<uint16_t>(CeilDiv(scaleKSize, BLOCK_CUBE)),
            static_cast<uint16_t>(dstNBlockStride / sizeof(uint16_t)),
            static_cast<uint16_t>(scaleKTail == 0 ? MX_SCALE_TRANS_ID_SIZE :
                                                    (scaleKTail * BLOCK_CUBE) / sizeof(uint16_t))};
        // Hoist the tail decision out of the VF loops. The aligned hot loop
        // contains no branch or mask update. Keep direct calls in this pipeline.
        if (scaleKTail == 0) {
            TransposeVf<false>(params);
        } else {
            TransposeVf<true>(params);
        }
    }

private:
    struct MxScaleTransposeParams {
        __ubuf__ uint16_t* input;
        __ubuf__ uint16_t* output;
        __ubuf__ uint16_t* transId;
        uint16_t nBlockCount;
        uint16_t groupBlockCount;
        uint16_t outputNBlockStride;
        uint16_t tailStoreCount;
    };

    template <bool HasTail>
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
                if constexpr (HasTail) {
                    if (groupBlock + 1 == params.groupBlockCount) {
                        uint32_t tailCount = params.tailStoreCount;
                        MicroAPI::MaskReg tailMask = MicroAPI::UpdateMask<uint16_t>(tailCount);
                        MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM_B16>(params.output, scale,
                                                                                           outputAddr, tailMask);
                    } else {
                        MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM_B16>(params.output, scale,
                                                                                           outputAddr, mask);
                    }
                } else {
                    MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM_B16>(params.output, scale, outputAddr,
                                                                                       mask);
                }
            }
        }
    }
};

// Copy complete, contiguous N16 blocks from the transposed UB tile into L1.
struct CopyUB2L1MxScale {
    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void Copy(const T& dst, const U& src)
    {
        using SrcLayoutPattern = AscendC::Te::GetLayoutPattern<typename U::layoutType>;
        using DstLayoutPattern = AscendC::Te::GetLayoutPattern<typename T::layoutType>;
        static_assert(AscendC::Std::is_same_v<SrcLayoutPattern, AscendC::Te::NNLayoutPtn> &&
                          AscendC::Std::is_same_v<DstLayoutPattern, AscendC::Te::NNLayoutPtn>,
                      "MX ScaleB copy to L1 requires NN source and destination layouts");
        static_assert(AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<U>, AscendC::Te::Location::UB> &&
                          AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<T>, AscendC::Te::Location::L1>,
                      "MX ScaleB copy to L1 requires UB source and L1 destination tensors");
        static_assert(sizeof(typename T::elementType) == 1 && sizeof(typename U::elementType) == 1,
                      "MX ScaleB copy to L1 requires 8-bit elements");

        const auto& srcLayout = src.Layout();
        uint64_t nBlockCount = AscendC::Std::get<1>(AscendC::Std::get<1>(srcLayout.Shape()));
        uint64_t nBlockStride = AscendC::Std::get<1>(AscendC::Std::get<1>(srcLayout.Stride()));
        // Use the physical N-block span, including its padding, rather than
        // multiplying the logical NN shape (whose K dimension is paired).
        uint32_t sizeBytes = static_cast<uint32_t>(nBlockCount * nBlockStride);
        asc_copy_ub2l1(reinterpret_cast<__cbuf__ void*>(dst.Data().Get()),
                       reinterpret_cast<__ubuf__ void*>(src.Data().Get()), sizeBytes);
    }
};

} // namespace Blaze::Gemm::Tile

namespace AscendC {
namespace Te {

template <typename Traits>
struct CopyTraits<Blaze::Gemm::Tile::CopyGM2UBMxScale, Traits>
    : public CopyTraits<Blaze::Gemm::Tile::CopyGM2UBMxScale, Traits, Blaze::Gemm::Tile::CopyGM2UBMxScale, Traits> {};

template <>
struct CopyTraits<Blaze::Gemm::Tile::CopyGM2UBMxScale>
    : public CopyTraits<Blaze::Gemm::Tile::CopyGM2UBMxScale, CopyGM2UBTraitDefault> {};

template <>
struct CopyTraits<Blaze::Gemm::Tile::CopyUB2L1MxScale>
    : public CopyTraits<Blaze::Gemm::Tile::CopyUB2L1MxScale, CopyUB2L1TraitDefault, Blaze::Gemm::Tile::CopyUB2L1MxScale,
                        CopyUB2L1TraitDefault> {};

} // namespace Te
} // namespace AscendC
