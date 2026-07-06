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
 * \file qbmm_mx_l0c_pingpong.h
 * \brief QBMM MX L0C ping-pong Kernel UT Wrapper
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx_l0c_pingpong.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"

namespace QBMMUT {

#pragma pack(push, 8)
struct QBMML0CPingpongTilingData {
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t b;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t kL1;
    uint32_t scaleKL1;
    uint32_t nBufferNum;
    uint32_t dbL0C;
};
#pragma pack(pop)

template <typename AType, typename BType, typename CType, typename BiasType, uint64_t FullLoadMode = 0>
__aicore__ inline void QBMML0CPingpongWrapper(
    GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
    const QBMML0CPingpongTilingData& tilingData)
{
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::NDExtLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMxL0CPingpong<FullLoadMode, false>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, FullLoadMode, LayoutA, LayoutB, AType>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using QBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QBMMKernel::Params;

    Params params;
    params.problemShape = {tilingData.m, tilingData.n, tilingData.k, tilingData.b};
    params.mmadParams.aGmAddr = x1GM;
    params.mmadParams.bGmAddr = x2GM;
    params.mmadParams.cGmAddr = yGM;
    params.mmadParams.biasGmAddr = biasGM;
    params.mmadParams.scaleAGmAddr = pertokenScaleGM;
    params.mmadParams.scaleBGmAddr = scaleGM;
    params.l1Params.kL1 = tilingData.kL1;
    params.l1Params.scaleKL1 = tilingData.scaleKL1;
    params.l1Params.l1BufNum = tilingData.nBufferNum;
    params.schParams.baseM = tilingData.baseM;
    params.schParams.baseN = tilingData.baseN;
    params.schParams.mTailTile = 1;
    params.schParams.nTailTile = 1;
    params.schParams.mBaseTailSplitCnt = 1;
    params.schParams.nBaseTailSplitCnt = 1;
    params.schParams.mTailMain = 0;
    params.schParams.nTailMain = 0;
    params.qbmmParams = {
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        0,
        tilingData.baseM,
        tilingData.baseN,
        tilingData.baseK,
        0,
        tilingData.dbL0C};

    QBMMKernel kernel;
    kernel(params);
}

} // namespace QBMMUT
