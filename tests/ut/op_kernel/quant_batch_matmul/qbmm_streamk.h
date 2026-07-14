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
 * \file qbmm_streamk.h
 * \brief QBMM MX StreamK Kernel UT Wrapper
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_qbmm_streamk.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_matmul_streamk.h"
#include "qbmm_tiling_data.h"

namespace QBMMUT {

template <class WorkspaceType_, class OutType_, class DispatchPolicy_>
struct BlockEpilogueStreamKForUt {
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;
    using WorkspaceType = WorkspaceType_;
    using OutType = OutType_;
    using DispatchPolicy = DispatchPolicy_;

    struct Params {
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR workspaceGmAddr{nullptr};
    };

    __aicore__ inline void Init(
        Params const&, BlockShape, BlockShape, BlockCoord, uint64_t, bool)
    {}

    __aicore__ inline void operator()()
    {}
};

template <typename AType, typename BType, typename CType, typename BiasType>
__aicore__ inline void QBMMStreamKWrapper(
    GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
    GM_ADDR workspaceGM, const QBMMStreamKTilingData& tilingData)
{
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::NDExtLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<
        Blaze::Gemm::NONE_FULL_LOAD_MODE, false, Blaze::Gemm::KernelQbmmMultiBlockStreamK>;
    using EpilogueDispatchPolicy = Blaze::Gemm::MatmulMultiBlockWithStreamK<>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutC>;
    using BlockEpilogue = BlockEpilogueStreamKForUt<float, CType, EpilogueDispatchPolicy>;
    using QBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QBMMKernel::Params;

    typename QBMMKernel::QBMMStreamKParams qbmmParams{tilingData.scaleKL1, tilingData.dbL0C};
    Params params{
        {tilingData.m, tilingData.n, tilingData.k, tilingData.b},
        {x1GM, x2GM, yGM, biasGM, pertokenScaleGM, scaleGM},
        {yGM, workspaceGM},
        {tilingData.usedCoreNum, tilingData.baseM, tilingData.baseN, tilingData.baseK,
         tilingData.singleCoreK, tilingData.kL1},
        qbmmParams};

    QBMMKernel kernel;
    kernel(params);
}

} // namespace QBMMUT

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS>
__global__ __aicore__ void qbmm_streamk_kernel_entry(
    GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMUT::QBMMStreamKTilingData*>(tilingGM);
    QBMMUT::QBMMStreamKWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS>(
        x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM, workspaceGM, *tilingData);
}
