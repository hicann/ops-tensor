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
 * \file qbmm_mx.h
 * \brief QBMM MX Kernel UT Wrapper
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "qbmm_cpu_debug_stub.h"

#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx_without_batch.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "qbmm_tiling_data.h"
#include "qbmm_ut_fill_helpers.h"

namespace QBMMUT {

template <typename AType, typename BType, typename CType, typename BiasType,
    uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE>
__aicore__ inline void QBMMMxWrapper(
    GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
    const QBMMV3TilingData& tilingData)
{
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::NDExtLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<FullLoadMode, false>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, FullLoadMode, LayoutA, LayoutB, AType>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using QBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QBMMKernel::Params;

    Params params;
    params.problemShape = {tilingData.m, tilingData.n, tilingData.k, tilingData.b};
    params.mmadParams.aGmAddr = reinterpret_cast<GM_ADDR>(x1GM);
    params.mmadParams.bGmAddr = reinterpret_cast<GM_ADDR>(x2GM);
    params.mmadParams.cGmAddr = reinterpret_cast<GM_ADDR>(yGM);
    params.mmadParams.biasGmAddr = reinterpret_cast<GM_ADDR>(biasGM);
    params.mmadParams.scaleAGmAddr = reinterpret_cast<GM_ADDR>(pertokenScaleGM);
    params.mmadParams.scaleBGmAddr = reinterpret_cast<GM_ADDR>(scaleGM);
    params.l1Params.kL1 = tilingData.kAL1;
    params.l1Params.scaleKL1 = tilingData.kBL1;
    params.l1Params.l1BufNum = tilingData.nBufferNum;

    FillQbmmSchParams(params.schParams, tilingData);

    params.qbmmParams.batchA1 = tilingData.batchA1;
    params.qbmmParams.batchA2 = tilingData.batchA2;
    params.qbmmParams.batchA3 = tilingData.batchA3;
    params.qbmmParams.batchA4 = tilingData.batchA4;
    params.qbmmParams.batchB1 = tilingData.batchB1;
    params.qbmmParams.batchB2 = tilingData.batchB2;
    params.qbmmParams.batchB3 = tilingData.batchB3;
    params.qbmmParams.batchB4 = tilingData.batchB4;
    params.qbmmParams.batchC1 = tilingData.batchC1;
    params.qbmmParams.batchC2 = tilingData.batchC2;
    params.qbmmParams.batchC3 = tilingData.batchC3;
    params.qbmmParams.batchC4 = tilingData.batchC4;
    params.qbmmParams.biasThreeDim = tilingData.biasThreeDim;
    params.qbmmParams.baseM = tilingData.baseM_qbmm;
    params.qbmmParams.baseN = tilingData.baseN_qbmm;
    params.qbmmParams.baseK = tilingData.baseK_qbmm;
    params.qbmmParams.isBias = tilingData.isBias;
    params.qbmmParams.dbL0C = tilingData.dbL0C;

    QBMMKernel kernel;
    kernel(params);
}

template <typename AType, typename BType, typename CType, typename BiasType,
    uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE>
__aicore__ inline void QBMMMxWithoutBatchWrapper(
    GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
    const QBMMV3TilingData& tilingData)
{
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::NDExtLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<
        FullLoadMode, false, Blaze::Gemm::KernelMmadWithScaleMxWithoutBatch>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, FullLoadMode, LayoutA, LayoutB, AType>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using QBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QBMMKernel::Params;

    Params params;
    params.problemShape = {tilingData.m, tilingData.n, tilingData.k, tilingData.b};
    params.mmadParams.aGmAddr = reinterpret_cast<GM_ADDR>(x1GM);
    params.mmadParams.bGmAddr = reinterpret_cast<GM_ADDR>(x2GM);
    params.mmadParams.cGmAddr = reinterpret_cast<GM_ADDR>(yGM);
    params.mmadParams.biasGmAddr = reinterpret_cast<GM_ADDR>(biasGM);
    params.mmadParams.scaleAGmAddr = reinterpret_cast<GM_ADDR>(pertokenScaleGM);
    params.mmadParams.scaleBGmAddr = reinterpret_cast<GM_ADDR>(scaleGM);
    params.l1Params.kL1 = tilingData.kAL1;
    params.l1Params.scaleKL1 = tilingData.kBL1;
    params.l1Params.l1BufNum = tilingData.nBufferNum;

    FillQbmmSchParams(params.schParams, tilingData);

    params.qbmmParams.baseM = tilingData.baseM_qbmm;
    params.qbmmParams.baseN = tilingData.baseN_qbmm;
    params.qbmmParams.baseK = tilingData.baseK_qbmm;
    params.qbmmParams.isBias = tilingData.isBias;
    params.qbmmParams.dbL0C = tilingData.dbL0C;

    QBMMKernel kernel;
    kernel(params);
}

} // namespace QBMMUT

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS>
__global__ __aicore__ void qbmm_mx_kernel_entry(
    GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
    GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMV3TilingData*>(tilingGM);
    QBMMUT::QBMMMxWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS>(
        x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM, *tilingData);
}

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS>
__global__ __aicore__ void qbmm_mx_without_batch_kernel_entry(
    GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
    GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMV3TilingData*>(tilingGM);
    QBMMUT::QBMMMxWithoutBatchWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS>(
        x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM, *tilingData);
}

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS>
__global__ __aicore__ void qbmm_mx_a_full_load_kernel_entry(
    GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
    GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMV3TilingData*>(tilingGM);
    QBMMUT::QBMMMxWrapper<
        DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, Blaze::Gemm::A_FULL_LOAD_MODE>(
        x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM, *tilingData);
}

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS>
__global__ __aicore__ void qbmm_mx_without_batch_a_full_load_kernel_entry(
    GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
    GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMV3TilingData*>(tilingGM);
    QBMMUT::QBMMMxWithoutBatchWrapper<
        DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, Blaze::Gemm::A_FULL_LOAD_MODE>(
        x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM, *tilingData);
}
