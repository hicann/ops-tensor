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
 * \file rms_softmax.h
 * \brief Apply RMS normalization and stable softmax to FP32 UB tensors.
 */
#pragma once

#include "tensor_api/tensor.h"
#include "kernel_operator.h"

namespace Blaze::Epilogue::Tile {

class RmsSoftmax {
public:
    template <typename SumSquareTensor, typename DotTensor, typename MaxTensor, typename SumTensor>
    __aicore__ inline static void Run(const SumSquareTensor& sumSquareTensor, const DotTensor& dotTensor,
                                      const MaxTensor& maxTensor, const SumTensor& sumTensor, float reciprocalD,
                                      float epsilon)
    {
        using SumSquareElementType = AscendC::Te::GetAttributeElementType<typename SumSquareTensor::elementType*>;
        using DotElementType = AscendC::Te::GetAttributeElementType<typename DotTensor::elementType*>;
        using MaxElementType = AscendC::Te::GetAttributeElementType<typename MaxTensor::elementType*>;
        using SumElementType = AscendC::Te::GetAttributeElementType<typename SumTensor::elementType*>;
        using SumSquareLayoutPattern = AscendC::Te::GetLayoutPattern<typename SumSquareTensor::layoutType>;
        using DotLayoutPattern = AscendC::Te::GetLayoutPattern<typename DotTensor::layoutType>;
        using MaxLayoutPattern = AscendC::Te::GetLayoutPattern<typename MaxTensor::layoutType>;
        using SumLayoutPattern = AscendC::Te::GetLayoutPattern<typename SumTensor::layoutType>;
        static_assert(
            AscendC::Std::is_same_v<SumSquareElementType, float> && AscendC::Std::is_same_v<DotElementType, float> &&
                AscendC::Std::is_same_v<MaxElementType, float> && AscendC::Std::is_same_v<SumElementType, float>,
            "RmsSoftmax only supports FP32 tensors.");
        static_assert(
            AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<SumSquareTensor>, AscendC::Te::Location::UB> &&
                AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<DotTensor>, AscendC::Te::Location::UB> &&
                AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<MaxTensor>, AscendC::Te::Location::UB> &&
                AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<SumTensor>, AscendC::Te::Location::UB>,
            "RmsSoftmax only supports UB tensors.");
        static_assert(AscendC::Std::is_same_v<SumSquareLayoutPattern, AscendC::Te::NDExtLayoutPtn> &&
                          AscendC::Std::is_same_v<DotLayoutPattern, AscendC::Te::NDExtLayoutPtn> &&
                          AscendC::Std::is_same_v<MaxLayoutPattern, AscendC::Te::NDExtLayoutPtn> &&
                          AscendC::Std::is_same_v<SumLayoutPattern, AscendC::Te::NDExtLayoutPtn>,
                      "RmsSoftmax requires NDExt tensor layouts.");

        const uint32_t validN = static_cast<uint32_t>(AscendC::Te::GetTotalColumnShape(dotTensor.Layout()));
        const uint32_t nAlign = static_cast<uint32_t>(
            AscendC::Te::Get<1>(AscendC::Te::Get<0>(dotTensor.Layout().Stride())));
        auto sumSquareAddr = reinterpret_cast<__ubuf__ float*>(sumSquareTensor.Data().Get());
        auto dotAddr = reinterpret_cast<__ubuf__ float*>(dotTensor.Data().Get());
        auto maxAddr = reinterpret_cast<__ubuf__ float*>(maxTensor.Data().Get());
        auto sumAddr = reinterpret_cast<__ubuf__ float*>(sumTensor.Data().Get());
        asc_vf_call<RmsSoftmaxVf>(sumSquareAddr, dotAddr, maxAddr, sumAddr, validN, nAlign, reciprocalD, epsilon);
    }

private:
    static __simd_vf__ inline void RmsSoftmaxVf(__ubuf__ float* sumSquareAddr, __ubuf__ float* dotAddr,
                                                __ubuf__ float* maxAddr, __ubuf__ float* sumAddr, uint32_t validN,
                                                uint32_t nAlign, float reciprocalD, float epsilon)
    {
        AscendC::Reg::RegTensor<float> sumSquareReg;
        AscendC::Reg::RegTensor<float> dotReg;
        AscendC::Reg::RegTensor<float> rmsReg;
        AscendC::Reg::RegTensor<float> normalizedReg;
        AscendC::Reg::RegTensor<float> maxReg;
        AscendC::Reg::RegTensor<float> maxBroadcastReg;
        AscendC::Reg::RegTensor<float> expReg;
        AscendC::Reg::RegTensor<float> expSumReg;
        AscendC::Reg::RegTensor<float> zeroReg;
        AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg oneMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        uint32_t validRemaining = validN;
        uint32_t alignRemaining = nAlign;
        AscendC::Reg::MaskReg validMask = AscendC::Reg::UpdateMask<float>(validRemaining);
        AscendC::Reg::MaskReg alignMask = AscendC::Reg::UpdateMask<float>(alignRemaining);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(sumSquareReg, sumSquareAddr);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(dotReg, dotAddr);
        AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(sumSquareReg, sumSquareReg, reciprocalD,
                                                                               validMask);
        AscendC::Reg::Adds<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(sumSquareReg, sumSquareReg, epsilon,
                                                                               validMask);
        AscendC::Reg::Sqrt<float, AscendC::Reg::MaskMergeMode::ZEROING>(rmsReg, sumSquareReg, validMask);
        AscendC::Reg::Div<float, AscendC::Reg::MaskMergeMode::ZEROING>(normalizedReg, dotReg, rmsReg, validMask);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX, float, float, AscendC::Reg::MaskMergeMode::ZEROING>(
            maxReg, normalizedReg, validMask);
        AscendC::Reg::Duplicate<float, AscendC::Reg::HighLowPart::LOWEST, AscendC::Reg::MaskMergeMode::ZEROING>(
            maxBroadcastReg, maxReg, allMask);
        AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(expReg, normalizedReg, maxBroadcastReg,
                                                                       validMask);
        AscendC::Reg::Exp<float, AscendC::Reg::MaskMergeMode::ZEROING>(expReg, expReg, validMask);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM, float, float, AscendC::Reg::MaskMergeMode::ZEROING>(
            expSumReg, expReg, validMask);
        AscendC::Reg::Duplicate(zeroReg, 0.0F, allMask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(dotAddr, zeroReg, alignMask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(dotAddr, expReg, validMask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(maxAddr, maxReg, oneMask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(sumAddr, expSumReg, oneMask);
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }
};

} // namespace Blaze::Epilogue::Tile
