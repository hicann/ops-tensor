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
 * \file qbmm_mix.h
 * \brief QBMM MIX A8W8 Kernel UT Wrapper（AIC int8 GEMM + AIV 反量化）。
 *        与 qbmm_cube.h 的 fixpipe wrapper 平级：装配 MatmulWithScaleMix / BlockEpilogueDequant /
 *        GemmUniversal（多 batch / 单 batch without_batch 特化），并填好各子 Params。
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "qbmm_cpu_debug_stub.h"

#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_qbmm_mix.h"
#include "blaze/gemm/kernel/kernel_qbmm_mix_without_batch.h"
#include "blaze/gemm/block/block_mmad_a8w8_mix.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_dequant.h"
#include "qbmm_tiling_data.h"
#include "qbmm_ut_fill_helpers.h"

namespace QBMMUT {

// MIX 路径的公共类型装配：AIC 侧 int8 GEMM（int32 L0C→UB），AIV 侧反量化。
template <typename AType, typename BType, typename OutType, typename X2ScaleType, typename X1ScaleType,
    typename BiasType, uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE>
struct QBMMMixTypes {
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::NDExtLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;

    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    // BTypeTuple 第 1 个元素为权重类型；第 2 个元素在 MIX mmad 中未使用（scale 在 epilogue 施加）。
    using BTypeTuple = AscendC::Std::tuple<BType, uint64_t>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMix<FullLoadMode, false>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, FullLoadMode, LayoutA, LayoutB, AType>;

    // CType 占位为 int32_t（UB 累加器类型），MIX mmad 类模板体内未直接引用该形参。
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AType, LayoutA, BTypeTuple, LayoutB, int32_t, LayoutC, BiasType, LayoutBias>;

    // BiasType 对 int8 输入编译期固定为 int32_t，实际 bias dtype 由 epilogueParams.biasDtype 运行时解释。
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueDequant<
        OutType, int32_t, X2ScaleType, X1ScaleType, int32_t>;
};

// 用 tilingData 填充 MIX BlockMmad::Params（aGm/bGm + 全套 tile/L1/L0C 配置）。
template <typename MmadParams>
__aicore__ inline void FillMixMmadParams(MmadParams& mmadParams, GM_ADDR x1GM, GM_ADDR x2GM,
    const QBMMV3TilingData& tilingData)
{
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    mmadParams.aGmAddr = reinterpret_cast<GM_ADDR>(x1GM);
    mmadParams.bGmAddr = reinterpret_cast<GM_ADDR>(x2GM);
    mmadParams.problemShape = {tilingData.m, tilingData.n, tilingData.k, tilingData.b};
    mmadParams.l0TileShape = BlockShape{
        static_cast<int64_t>(tilingData.baseM_qbmm), static_cast<int64_t>(tilingData.baseN_qbmm),
        static_cast<int64_t>(tilingData.baseK_qbmm), 0};
    mmadParams.kAL1 = tilingData.kAL1;
    mmadParams.kBL1 = tilingData.kBL1;
    mmadParams.l1BufferNum = tilingData.nBufferNum;
    mmadParams.enableL0CPingPong = (tilingData.dbL0C > 1);
}

// Fill BlockEpilogueDequant::Params from tiling and GM addresses.
template <typename EpilogueParams>
__aicore__ inline void FillEpilogueParams(EpilogueParams& epilogueParams, GM_ADDR pertokenScaleGM,
    GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM, const QBMMV3TilingData& tilingData)
{
    epilogueParams.x2ScaleGmAddr = reinterpret_cast<GM_ADDR>(scaleGM);        // 权重 scale
    epilogueParams.x1ScaleGmAddr = reinterpret_cast<GM_ADDR>(pertokenScaleGM); // 激活 per-token scale
    epilogueParams.biasGmAddr = reinterpret_cast<GM_ADDR>(biasGM);
    epilogueParams.outGmAddr = reinterpret_cast<GM_ADDR>(yGM);
    epilogueParams.m = tilingData.m;
    epilogueParams.n = tilingData.n;
    epilogueParams.baseM = static_cast<int64_t>(tilingData.baseM_qbmm);
    epilogueParams.baseN = static_cast<int64_t>(tilingData.baseN_qbmm);
    epilogueParams.x1QuantMode = tilingData.x1QuantMode;
    epilogueParams.x2QuantMode = tilingData.x2QuantMode;
    epilogueParams.isBias = (tilingData.isBias != 0);
    epilogueParams.biasDtype = tilingData.biasDtype;
}

// 多 batch MIX：GemmUniversal（KernelMmadWithScaleMix 特化）。
template <typename AType, typename BType, typename OutType, typename X2ScaleType = float,
    typename X1ScaleType = float, typename BiasType = int32_t,
    uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE>
__aicore__ inline void QBMMMixWrapper(
    GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
    const QBMMV3TilingData& tilingData)
{
    using Types = QBMMMixTypes<AType, BType, OutType, X2ScaleType, X1ScaleType, BiasType, FullLoadMode>;
    using QBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<
        typename Types::ProblemShape, typename Types::BlockMmad, typename Types::BlockEpilogue,
        typename Types::BlockScheduler>;
    using Params = typename QBMMKernel::Params;

    Params params;
    params.problemShape = {tilingData.m, tilingData.n, tilingData.k, tilingData.b};
    FillMixMmadParams(params.mmadParams, x1GM, x2GM, tilingData);
    FillQbmmSchParams(params.schParams, tilingData);

    FillQbmmBatchParams(params.qbmmParams, tilingData);
    FillQbmmTileParams(params.qbmmParams, tilingData);

    FillEpilogueParams(params.epilogueParams, pertokenScaleGM, scaleGM, biasGM, yGM, tilingData);

    QBMMKernel kernel;
    kernel(params);
}

// 单 batch MIX：GemmUniversal（KernelMmadWithScaleMixWithoutBatch 特化，裁剪 batch 广播路径）。
template <typename AType, typename BType, typename OutType, typename X2ScaleType = float,
    typename X1ScaleType = float, typename BiasType = int32_t,
    uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE>
__aicore__ inline void QBMMMixWithoutBatchWrapper(
    GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
    const QBMMV3TilingData& tilingData)
{
    using Types = QBMMMixTypes<AType, BType, OutType, X2ScaleType, X1ScaleType, BiasType, FullLoadMode>;
    using DispatchPolicy =
        Blaze::Gemm::MatmulWithScaleMix<FullLoadMode, false, Blaze::Gemm::KernelMmadWithScaleMixWithoutBatch>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AType, typename Types::LayoutA, typename Types::BTypeTuple, typename Types::LayoutB, int32_t,
        typename Types::LayoutC, BiasType, typename Types::LayoutBias>;
    using QBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<
        typename Types::ProblemShape, BlockMmad, typename Types::BlockEpilogue,
        typename Types::BlockScheduler>;
    using Params = typename QBMMKernel::Params;

    Params params;
    params.problemShape = {tilingData.m, tilingData.n, tilingData.k, tilingData.b};
    FillMixMmadParams(params.mmParams, x1GM, x2GM, tilingData);
    FillQbmmSchParams(params.schParams, tilingData);

    FillEpilogueParams(params.epilogueParams, pertokenScaleGM, scaleGM, biasGM, yGM, tilingData);

    QBMMKernel kernel;
    kernel(params);
}

} // namespace QBMMUT

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS>
__global__ __aicore__ void qbmm_mix_kernel_entry(
    GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
    GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMV3TilingData*>(tilingGM);
    QBMMUT::QBMMMixWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, float, float, DTYPE_BIAS>(
        x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM, *tilingData);
}

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS>
__global__ __aicore__ void qbmm_mix_without_batch_kernel_entry(
    GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
    GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMV3TilingData*>(tilingGM);
    QBMMUT::QBMMMixWithoutBatchWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, float, float, DTYPE_BIAS>(
        x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM, *tilingData);
}

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS>
__global__ __aicore__ void qbmm_mix_a_full_load_kernel_entry(
    GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
    GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMV3TilingData*>(tilingGM);
    QBMMUT::QBMMMixWrapper<
        DTYPE_X1, DTYPE_X2, DTYPE_Y, float, float, DTYPE_BIAS, Blaze::Gemm::A_FULL_LOAD_MODE>(
        x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM, *tilingData);
}

template <class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS>
__global__ __aicore__ void qbmm_mix_without_batch_a_full_load_kernel_entry(
    GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM,
    GM_ADDR tilingGM)
{
    const auto* tilingData = reinterpret_cast<const QBMMV3TilingData*>(tilingGM);
    QBMMUT::QBMMMixWithoutBatchWrapper<
        DTYPE_X1, DTYPE_X2, DTYPE_Y, float, float, DTYPE_BIAS, Blaze::Gemm::A_FULL_LOAD_MODE>(
        x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM, *tilingData);
}
