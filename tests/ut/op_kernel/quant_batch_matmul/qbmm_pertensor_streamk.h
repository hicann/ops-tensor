/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file qbmm_pertensor_streamk.h
 * \brief QBMM per-tensor StreamK Kernel UT wrapper.
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "qbmm_cpu_debug_stub.h"

#include "blaze/epilogue/block/block_epilogue_qbmm_pertensor_streamk.h"
#include "blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h"
#include "blaze/gemm/block/block_scheduler_matmul_streamk.h"
#include "blaze/gemm/kernel/kernel_qbmm_pertensor_streamk.h"
#include "qbmm_pertensor_streamk_tiling_data.h"

namespace QBMMUT {

template <typename AType, typename BType, typename X2ScaleType, typename CType, typename BiasType,
          uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE>
__aicore__ inline void QBMMPertensorStreamKWrapper(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR perTokenScaleGM, GM_ADDR scaleGM,
                                                   GM_ADDR biasGM, GM_ADDR yGM, GM_ADDR workspaceGM,
                                                   const QBMMPertensorStreamKTilingData& tilingData)
{
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::NDExtLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<FullLoadMode, false,
                                                                    Blaze::Gemm::KernelQbmmPertensorMultiBlockStreamK>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA,
                                                    AscendC::Std::tuple<BType, X2ScaleType>, LayoutB, CType, LayoutC,
                                                    BiasType, LayoutC>;
    using WorkspaceType = typename BlockMmad::WorkspaceType;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<WorkspaceType, CType,
                                                                                    DispatchPolicy, X2ScaleType, float>;
    using QBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QBMMKernel::Params;

    bool hasBias = tilingData.isBias != 0U;
    GM_ADDR biasMmadGM = hasBias && BlockMmad::BIAS_IN_MMAD ? biasGM : nullptr;
    GM_ADDR biasEpilogueGM = hasBias && !BlockMmad::BIAS_IN_MMAD ? biasGM : nullptr;
    bool isBiasEpilogue = biasEpilogueGM != nullptr;

    Params params{{tilingData.m, tilingData.n, tilingData.k, tilingData.b},
                  {x1GM, x2GM, yGM, biasMmadGM, perTokenScaleGM, scaleGM},
                  {yGM, workspaceGM, scaleGM, perTokenScaleGM, biasEpilogueGM, isBiasEpilogue, tilingData.biasDtype},
                  {tilingData.usedCoreNum, tilingData.baseM, tilingData.baseN, tilingData.baseK, tilingData.singleCoreK,
                   tilingData.kL1}};

    QBMMKernel kernel;
    kernel(params);
}

} // namespace QBMMUT

template <class DTYPE_X1, class DTYPE_X2, class SCALE_TYPE, class DTYPE_Y, class DTYPE_BIAS>
__global__ __aicore__ void qbmm_pertensor_streamk_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR perTokenScaleGM,
                                                               GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
                                                               GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMUT::QBMMPertensorStreamKTilingData*>(tilingGM);
    QBMMUT::QBMMPertensorStreamKWrapper<DTYPE_X1, DTYPE_X2, SCALE_TYPE, DTYPE_Y, DTYPE_BIAS>(
        x1GM, x2GM, perTokenScaleGM, scaleGM, biasGM, yGM, workspaceGM, *tilingData);
}
