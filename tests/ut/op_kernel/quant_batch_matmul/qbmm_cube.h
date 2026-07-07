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
 * \file qbmm_cube.h
 * \brief QBMM Cube Kernel UT Wrapper
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "qbmm_cpu_debug_stub.h"

#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_qbmm_cube.h"
#include "blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "qbmm_tiling_data.h"
#include "qbmm_ut_fill_helpers.h"

namespace QBMMUT {

template <typename AType, typename BType, typename CType, typename BiasType>
__aicore__ inline void QBMMCubeWrapper(
    GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM, const QBMMV3TilingData& tilingData)
{
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::NDExtLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;

    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    // BTypeTuple = (BType, uint64_t) for B + X2Scale
    using BTypeTuple = AscendC::Std::tuple<BType, uint64_t>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<0, false>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, 0, LayoutA, LayoutB, AType>;

    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AType, LayoutA, BTypeTuple, LayoutB, CType, LayoutC, BiasType, LayoutBias>;

    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;

    using QBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<
        ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    using Params = typename QBMMKernel::Params;

    Params params;
    params.problemShape = {tilingData.m, tilingData.n, tilingData.k, tilingData.b};
    params.mmadParams.aGmAddr = reinterpret_cast<GM_ADDR>(x1GM);
    params.mmadParams.bGmAddr = reinterpret_cast<GM_ADDR>(x2GM);
    params.mmadParams.cGmAddr = reinterpret_cast<GM_ADDR>(yGM);
    params.mmadParams.biasGmAddr = reinterpret_cast<GM_ADDR>(biasGM);
    params.mmadParams.scaleAGmAddr = reinterpret_cast<GM_ADDR>(pertokenScaleGM);
    params.mmadParams.scaleBGmAddr = reinterpret_cast<GM_ADDR>(scaleGM);

    params.schParams.baseM = tilingData.baseM;
    params.schParams.baseN = tilingData.baseN;
    params.schParams.mTailTile = tilingData.mTailTile;
    params.schParams.nTailTile = tilingData.nTailTile;
    params.schParams.mBaseTailSplitCnt = tilingData.mBaseTailSplitCnt;
    params.schParams.nBaseTailSplitCnt = tilingData.nBaseTailSplitCnt;
    params.schParams.mTailMain = tilingData.mTailMain;
    params.schParams.nTailMain = tilingData.nTailMain;

    FillQbmmBatchParams(params.qbmmParams, tilingData);
    FillQbmmTileParams(params.qbmmParams, tilingData);

    QBMMKernel kernel;
    kernel(params);
}

} // namespace QBMMUT
