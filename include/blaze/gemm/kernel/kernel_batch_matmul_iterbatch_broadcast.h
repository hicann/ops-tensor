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
 * \file kernel_batch_matmul_iterbatch_broadcast.h
 * \brief GemmUniversal partial specialization for IterBatch-Broadcast path
 *        Combines BmmBroadcast % operator with iterbatch L1/L0 pipelining
 *        Uses BlockSchedulerIterBatchBroadcast for batch grouping
 *        Uses BlockMmad<MatmulIterBatchBroadcast<ABc,BBc>> for iterbatch MMAD
 *        Broadcast side determined at compile time via DispatchPolicy template params
 *        Kernel does gmA/gmB Slice for broadcast mapping before calling MMAD block
 */

#pragma once

#define ASCENDC_CUBE_ONLY

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_iterbatch_broadcast.h"
#include "blaze/gemm/block/block_scheduler_iterbatch_broadcast.h"
#include "blaze/gemm/utils/common_utils.h"
#include "kernel_universal.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<
    ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_same_v<KernelIterBatchBroadcast, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    __aicore__ inline GemmUniversal()
    {}
    __aicore__ inline ~GemmUniversal()
    {}

    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;
    using DispatchPolicy = typename BlockMmad::DispatchPolicy;
    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;
    static constexpr bool aBroadcast = DispatchPolicy::aBroadcast;
    static constexpr bool bBroadcast = DispatchPolicy::bBroadcast;
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutBias = typename BlockMmad::LayoutBias;
    using TupleShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<AscendC::AuxGetC0Size<AType>()>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<AscendC::AuxGetC0Size<BType>()>>;
    using MakeLayoutC = AscendC::Te::FrameLayoutFormat<LayoutC, AscendC::Std::Int<AscendC::AuxGetC0Size<CType>()>>;
    using MakeLayoutBias =
        AscendC::Te::FrameLayoutFormat<LayoutBias, AscendC::Std::Int<AscendC::AuxGetC0Size<BiasType>()>>;
    using BlockSchedulerParams = typename Block::BlockSchedulerIterBatchBroadcast<ProblemShape>::Params;
    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schedulerParams;
        Params() = default;
    };

    __aicore__ inline void Init(Params const& params)
    {
        m_ = static_cast<uint64_t>(AscendC::Te::Get<MNK_M>(params.problemShape));
        n_ = static_cast<uint64_t>(AscendC::Te::Get<MNK_N>(params.problemShape));
        k_ = static_cast<uint64_t>(AscendC::Te::Get<MNK_K>(params.problemShape));
        b_ = static_cast<uint64_t>(params.schedulerParams.cBatchDim0) *
             static_cast<uint64_t>(params.schedulerParams.cBatchDim1) *
             static_cast<uint64_t>(params.schedulerParams.cBatchDim2) *
             static_cast<uint64_t>(params.schedulerParams.cBatchDim3);
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
        bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
        if (params.mmadParams.biasGmAddr != nullptr) {
            isBias_ = true;
            biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
        }
    }

    __aicore__ inline void operator()(Params const& params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        BlockMmad blockMmad;
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        Init(params);
        const BlockSchedulerParams& schP = params.schedulerParams;
        Block::BlockSchedulerIterBatchBroadcast<ProblemShape> bs(params.problemShape, curBlockIdx, blockNum, schP);
        int64_t tileNum = bs.GetTileNum();
        int64_t realBlockNum = bs.GetBlockNum(params.problemShape, blockNum);
        if (curBlockIdx >= realBlockNum) {
            return;
        }
        uint64_t mainIterBatchL1 = static_cast<uint64_t>(schP.iterBatchL1);
        uint64_t mainIterBatchL0 = static_cast<uint64_t>(schP.iterBatchL0);
        uint64_t baseM = static_cast<uint64_t>(schP.baseM);
        uint64_t baseN = static_cast<uint64_t>(schP.baseN);
        uint64_t baseK = static_cast<uint64_t>(schP.baseK);
        uint64_t bcAxisA = static_cast<uint64_t>(schP.broadcastAxisA);
        uint64_t bcAxisB = static_cast<uint64_t>(schP.broadcastAxisB);
        if (params.schedulerParams.isHf32) {
            AscendC::SetHF32Mode(1);
            AscendC::SetHF32TransMode(1);
        }
        blockMmad.Init(params.problemShape, mainIterBatchL1, mainIterBatchL0, isBias_,
            baseM, baseN, baseK, bcAxisA, bcAxisB);
        uint64_t totalABatches = static_cast<uint64_t>(
            params.schedulerParams.aBatchDim0 * params.schedulerParams.aBatchDim1 *
            params.schedulerParams.aBatchDim2 * params.schedulerParams.aBatchDim3);
        uint64_t totalBBatches = static_cast<uint64_t>(
            params.schedulerParams.bBatchDim0 * params.schedulerParams.bBatchDim1 *
            params.schedulerParams.bBatchDim2 * params.schedulerParams.bBatchDim3);
        uint64_t totalCBatches = static_cast<uint64_t>(
            params.schedulerParams.cBatchDim0 * params.schedulerParams.cBatchDim1 *
            params.schedulerParams.cBatchDim2 * params.schedulerParams.cBatchDim3);

        auto layoutA3D = AscendC::Te::MakeFrameLayout<LayoutA>(totalABatches, m_, k_);
        auto layoutB3D = AscendC::Te::MakeFrameLayout<LayoutB>(totalBBatches, k_, n_);
        auto layoutC3D = AscendC::Te::MakeFrameLayout<
            AscendC::Te::NDExtLayoutPtn>(totalCBatches, m_, n_);
        auto layoutBias = MakeLayoutBias{}(1L, n_);
        auto gmA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aGmAddr_), layoutA3D);
        auto gmB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bGmAddr_), layoutB3D);
        auto gmC = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGmAddr_), layoutC3D);
         auto gmBias = AscendC::Te::MakeTensor(
             AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(biasGmAddr_), layoutBias);
         for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
            auto blockShape = bs.GetBlockShape(tileIdx, tileNum);
            auto blockCoord = bs.GetBlockCoord(tileIdx);
            uint64_t startBatchIdx = static_cast<uint64_t>(AscendC::Te::Get<MNK_B>(blockCoord));
            uint64_t curIterBatchL1 = static_cast<uint64_t>(AscendC::Te::Get<MNK_B>(blockShape));
            uint64_t aGmStartBatch = static_cast<uint64_t>(bs.ComputeABroadcastIndex(startBatchIdx));
            uint64_t bGmStartBatch = static_cast<uint64_t>(bs.ComputeBBroadcastIndex(startBatchIdx));
            uint64_t al1Count = (aBroadcast && bcAxisA == LAST_BATCH_DIM) ? 1UL : curIterBatchL1;
            uint64_t bl1Count = (bBroadcast && bcAxisB == LAST_BATCH_DIM) ? 1UL : curIterBatchL1;
            uint64_t agmStart = aBroadcast ? aGmStartBatch : startBatchIdx;
            uint64_t bgmStart = bBroadcast ? bGmStartBatch : startBatchIdx;
            auto gmASlice = gmA.Slice(
                AscendC::Te::MakeCoord(agmStart, AscendC::Te::MakeCoord(0, 0)),
                AscendC::Te::MakeShape(al1Count, AscendC::Te::MakeShape(m_, k_)));
            auto gmBSlice = gmB.Slice(
                AscendC::Te::MakeCoord(bgmStart, AscendC::Te::MakeCoord(0, 0)),
                AscendC::Te::MakeShape(bl1Count, AscendC::Te::MakeShape(k_, n_)));
            auto gmCSlice = gmC.Slice(
                AscendC::Te::MakeCoord(startBatchIdx, AscendC::Te::MakeCoord(0, 0)),
                AscendC::Te::MakeShape(curIterBatchL1, AscendC::Te::MakeShape(m_, n_)));
            blockMmad(gmCSlice, gmASlice, gmBSlice, gmBias, curIterBatchL1);
        }
        AscendC::SetMMLayoutTransform(false);
        UnsetHf32();
    }

private:
    bool isBias_ = false;
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t b_{1};
    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr;

    __aicore__ inline void UnsetHf32()
    {
        AscendC::SetHF32Mode(0);
    }

};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
