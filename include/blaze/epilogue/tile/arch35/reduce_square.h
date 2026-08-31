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
 * \file reduce_square.h
 * \brief Reduce FP32 UB rows to their square sums.
 */
#pragma once

#include "tensor_api/tensor.h"
#include "kernel_operator.h"

namespace Blaze::Epilogue::Tile {

template <bool FirstTile_>
class ReduceSquare {
public:
    template <typename InputTensor, typename SumSquareTensor>
    __aicore__ inline static void Run(const InputTensor& inputTensor, const SumSquareTensor& sumSquareTensor)
    {
        using InputElementType = AscendC::Te::GetAttributeElementType<typename InputTensor::elementType*>;
        using SumSquareElementType = AscendC::Te::GetAttributeElementType<typename SumSquareTensor::elementType*>;
        using InputLayoutPattern = AscendC::Te::GetLayoutPattern<typename InputTensor::layoutType>;
        using SumSquareLayoutPattern = AscendC::Te::GetLayoutPattern<typename SumSquareTensor::layoutType>;
        static_assert(
            AscendC::Std::is_same_v<InputElementType, float> && AscendC::Std::is_same_v<SumSquareElementType, float>,
            "ReduceSquare only supports FP32 tensors.");
        static_assert(
            AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<InputTensor>, AscendC::Te::Location::UB> &&
                AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<SumSquareTensor>, AscendC::Te::Location::UB>,
            "ReduceSquare only supports UB tensors.");
        static_assert(AscendC::Std::is_same_v<InputLayoutPattern, AscendC::Te::NDExtLayoutPtn> &&
                          AscendC::Std::is_same_v<SumSquareLayoutPattern, AscendC::Te::NDExtLayoutPtn>,
                      "ReduceSquare requires NDExt tensor layouts.");

        const uint32_t rowCount = static_cast<uint32_t>(AscendC::Te::GetTotalRowShape(inputTensor.Layout()));
        const uint32_t validElements = static_cast<uint32_t>(AscendC::Te::GetTotalColumnShape(inputTensor.Layout()));
        const uint32_t rowPitch = static_cast<uint32_t>(
            AscendC::Te::Get<1>(AscendC::Te::Get<0>(inputTensor.Layout().Stride())));
        const uint16_t loopCount = static_cast<uint16_t>((validElements + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
        auto inputAddr = reinterpret_cast<__ubuf__ float*>(inputTensor.Data().Get());
        auto sumSquareAddr = reinterpret_cast<__ubuf__ float*>(sumSquareTensor.Data().Get());
        RunRows(inputAddr, sumSquareAddr, rowCount, validElements, loopCount, rowPitch);
    }

private:
    static constexpr uint32_t FP32_REG_ELEMS = AscendC::VECTOR_REG_WIDTH / sizeof(float);

    static __simd_vf__ inline void AccumulateSquareVf(__ubuf__ float* inputAddr, __ubuf__ float* sumSquareAddr,
                                                      uint32_t validElements, uint16_t loopCount, uint32_t rowIndex)
    {
        AscendC::Reg::RegTensor<float> inputReg;
        AscendC::Reg::RegTensor<float> squareReg;
        AscendC::Reg::RegTensor<float> squareAccReg;
        AscendC::Reg::RegTensor<float> squareReduceReg;
        AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg oneMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        AscendC::Reg::Duplicate(squareAccReg, 0.0F, allMask);
        uint32_t remaining = validElements;
        for (uint16_t loop = 0U; loop < loopCount; ++loop) {
            AscendC::Reg::MaskReg validMask = AscendC::Reg::UpdateMask<float>(remaining);
            const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(inputReg, inputAddr + offset);
            AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(squareReg, inputReg, inputReg, validMask);
            AscendC::Reg::Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(squareAccReg, squareAccReg, squareReg,
                                                                           allMask);
        }
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM, float, float, AscendC::Reg::MaskMergeMode::ZEROING>(
            squareReduceReg, squareAccReg, allMask);
        if constexpr (!FirstTile_) {
            AscendC::Reg::RegTensor<float> previousValueReg;
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(previousValueReg,
                                                                                 sumSquareAddr + rowIndex);
            AscendC::Reg::Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(squareReduceReg, squareReduceReg,
                                                                           previousValueReg, oneMask);
        }
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(sumSquareAddr + rowIndex,
                                                                                         squareReduceReg, oneMask);
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }

    __aicore__ inline static void RunRows(__ubuf__ float* inputAddr, __ubuf__ float* sumSquareAddr, uint32_t rowCount,
                                          uint32_t validElements, uint16_t loopCount, uint32_t rowPitch)
    {
        for (uint32_t rowIndex = 0U; rowIndex < rowCount; ++rowIndex) {
            asc_vf_call<AccumulateSquareVf>(inputAddr + static_cast<uint64_t>(rowIndex) * rowPitch, sumSquareAddr,
                                            validElements, loopCount, rowIndex);
        }
    }
};

} // namespace Blaze::Epilogue::Tile
