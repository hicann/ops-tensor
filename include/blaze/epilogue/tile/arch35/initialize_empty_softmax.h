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
 * \file initialize_empty_softmax.h
 * \brief Initialize empty softmax statistics in UB.
 */
#pragma once

#include "tensor_api/tensor.h"
#include "kernel_operator.h"

namespace Blaze::Epilogue::Tile {

class InitializeEmptySoftmax {
public:
    template <typename MaxTensor, typename SumTensor>
    __aicore__ inline static void Run(const MaxTensor& maxTensor, const SumTensor& sumTensor)
    {
        using MaxElementType = AscendC::Te::GetAttributeElementType<typename MaxTensor::elementType*>;
        using SumElementType = AscendC::Te::GetAttributeElementType<typename SumTensor::elementType*>;
        using MaxLayoutPattern = AscendC::Te::GetLayoutPattern<typename MaxTensor::layoutType>;
        using SumLayoutPattern = AscendC::Te::GetLayoutPattern<typename SumTensor::layoutType>;
        static_assert(AscendC::Std::is_same_v<MaxElementType, float> && AscendC::Std::is_same_v<SumElementType, float>,
                      "InitializeEmptySoftmax only supports FP32 tensors.");
        static_assert(AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<MaxTensor>, AscendC::Te::Location::UB> &&
                          AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<SumTensor>, AscendC::Te::Location::UB>,
                      "InitializeEmptySoftmax only supports UB tensors.");
        static_assert(AscendC::Std::is_same_v<MaxLayoutPattern, AscendC::Te::NDExtLayoutPtn> &&
                          AscendC::Std::is_same_v<SumLayoutPattern, AscendC::Te::NDExtLayoutPtn>,
                      "InitializeEmptySoftmax requires NDExt tensor layouts.");

        auto maxAddr = reinterpret_cast<__ubuf__ float*>(maxTensor.Data().Get());
        auto sumAddr = reinterpret_cast<__ubuf__ float*>(sumTensor.Data().Get());
        asc_vf_call<InitializeEmptySoftmaxVf>(maxAddr, sumAddr);
    }

private:
    static constexpr float FP32_LOWEST_FINITE = -__FLT_MAX__;

    static __simd_vf__ inline void InitializeEmptySoftmaxVf(__ubuf__ float* maxAddr, __ubuf__ float* sumAddr)
    {
        AscendC::Reg::RegTensor<float> maxReg;
        AscendC::Reg::RegTensor<float> sumReg;
        AscendC::Reg::MaskReg oneMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        AscendC::Reg::Duplicate(maxReg, FP32_LOWEST_FINITE, oneMask);
        AscendC::Reg::Duplicate(sumReg, 0.0F, oneMask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(maxAddr, maxReg, oneMask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(sumAddr, sumReg, oneMask);
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }
};

} // namespace Blaze::Epilogue::Tile
