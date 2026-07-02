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
 * \file kernel_qgmm_mx.h
 * \brief
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_scheduler_gmm_aswt_with_tail_split.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "kernel_universal.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {

namespace Kernel {
namespace {
constexpr uint64_t GROUP_LIST_TYPE_OFFSET = 0UL;
constexpr uint64_t GROUP_LIST_TYPE_LENGTH = 1UL;
constexpr uint64_t GROUP_LIST_TYPE_SPARSE = 2UL;
constexpr uint64_t SPARSE_GROUP_LIST_ITEM_STRIDE = 2UL;
constexpr uint64_t SPARSE_GROUP_LIST_SPLIT_VALUE_OFFSET = 1UL;
constexpr int64_t BLOCK_CUBE_MASK = BLOCK_CUBE - 1;
constexpr int64_t SCALE_CACHE_MASK = 0xff;
} // namespace

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
class GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_same_v<KernelGroupedMmadWithScaleMx, typename BlockMmad::DispatchPolicy::ScheduleType>>> {
public:
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    static constexpr bool transA = IsTrans<LayoutA>::value;
    static constexpr bool transB = IsTrans<LayoutB>::value;
    static constexpr bool weightNz = IsWeightNz<LayoutB>::value;

    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using L1Params = typename BlockMmad::L1Params;

    using BlockShape = typename BlockMmad::BlockShape;
    using SchedulerShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;

    struct GMMTiling {
        uint32_t groupNum;
        int64_t m;
        int64_t n;
        int64_t k;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t kAL1;
        uint32_t kBL1;
        uint32_t scaleKAL1; // ScaleA L1 K-axis split size; must equal scaleKBL1 for MX.
        uint32_t scaleKBL1; // ScaleB L1 K-axis split size; kept for tiling compatibility.
        uint8_t isBias;
        uint8_t dbL0C;
        // Reserved for GMM tiling compatibility. Current kernel does not read this field;
        // the split axis is selected by LayoutA: !transA means split-M, transA means split-K.
        int8_t groupType;
        uint8_t groupListType;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        GM_ADDR groupListGmAddr;
        GMMTiling gmmParams;
    };

    __aicore__ inline void operator()(const Params& params)
    {
        Run(params);
    }

private:
    static constexpr uint32_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutC = AscendC::Te::FrameLayoutFormat<LayoutC, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<CType>>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        transA, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleADNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        transB, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBDNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBNDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;

    __aicore__ inline void SetSchedulerTailAlign(BlockScheduler& scheduler)
    {
        if constexpr (!transA) {
            constexpr uint32_t mTailAlign = 1;
            constexpr uint32_t nTailAlign = transB ? static_cast<uint32_t>(BLOCK_CUBE) : static_cast<uint32_t>(C0_SIZE);
            scheduler.SetTailAlign(mTailAlign, nTailAlign);
        } else {
            constexpr uint32_t mTailAlign = static_cast<uint32_t>(Block::INNER_AXIS_MIN_SPLIT_VAL);
            constexpr uint32_t nTailAlign = transB ? static_cast<uint32_t>(BLOCK_CUBE) :
                                                    static_cast<uint32_t>(Block::INNER_AXIS_MIN_SPLIT_VAL);
            scheduler.SetTailAlign(mTailAlign, nTailAlign);
        }
    }

    template <typename TensorB, typename TensorScaleB>
    __aicore__ inline void SetL2CacheHint(TensorB& gmB, TensorScaleB& gmScaleB, int64_t mSize, int64_t curBaseM,
                                          int64_t baseN)
    {
        const int64_t problemN = AscendC::Te::Get<MNK_N>(problemShape_);
        const int64_t problemK = AscendC::Te::Get<MNK_K>(problemShape_);
        if constexpr (weightNz) {
            if (curBaseM >= mSize) {
                gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
            } else {
                gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
            }
        } else {
            if constexpr (transB) {
                if (curBaseM >= mSize && (problemK & SCALE_CACHE_MASK) == 0) {
                    gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                    gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                } else {
                    gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                    gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                }
            } else {
                if (curBaseM >= mSize && (problemN & SCALE_CACHE_MASK) == 0 && (baseN & SCALE_CACHE_MASK) == 0) {
                    gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                    gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                } else {
                    gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                    gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                }
            }
        }
    }

    __aicore__ inline void Run(const Params& params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        Init(params);
        if (groupNum_ == 0) {
            return;
        }
        const auto& gmmParams = params.gmmParams;
        BlockScheduler scheduler(gmmParams.baseM, gmmParams.baseN, gmmParams.baseK);
        SetSchedulerTailAlign(scheduler);
        const uint32_t lastGroupIdx = groupNum_ - 1;
        for (uint32_t loopIdx = 0; loopIdx < lastGroupIdx; ++loopIdx) {
            uint32_t groupIdx = loopIdx;
            if (groupListType_ == GROUP_LIST_TYPE_SPARSE) {
                groupIdx = static_cast<uint32_t>(groupListGlobal_.GetValue(loopIdx * SPARSE_GROUP_LIST_ITEM_STRIDE));
            }
            SetMNK(loopIdx);
            const int64_t problemM = AscendC::Te::Get<MNK_M>(problemShape_);
            const int64_t problemN = AscendC::Te::Get<MNK_N>(problemShape_);
            const int64_t problemK = AscendC::Te::Get<MNK_K>(problemShape_);
            if (problemM <= 0 || problemN <= 0 || problemK <= 0) {
                if (groupListType_ == GROUP_LIST_TYPE_SPARSE && problemM <= 0) {
                    break;
                }
                continue;
            }
            BaseMBalance(scheduler, problemM, gmmParams.baseM);
            scheduler.UpdateNextProblem(SchedulerShape{problemM, problemN, problemK, 0});
            ProcessSingleGroup<false>(scheduler, groupIdx);
        }

        uint32_t groupIdx = lastGroupIdx;
        if (groupListType_ == GROUP_LIST_TYPE_SPARSE) {
            groupIdx = static_cast<uint32_t>(groupListGlobal_.GetValue(lastGroupIdx * SPARSE_GROUP_LIST_ITEM_STRIDE));
        }
        SetMNK(lastGroupIdx);
        const int64_t problemM = AscendC::Te::Get<MNK_M>(problemShape_);
        const int64_t problemN = AscendC::Te::Get<MNK_N>(problemShape_);
        const int64_t problemK = AscendC::Te::Get<MNK_K>(problemShape_);
        if (problemM > 0 && problemN > 0 && problemK > 0) {
            BaseMBalance(scheduler, problemM, gmmParams.baseM);
            scheduler.UpdateNextProblem(SchedulerShape{problemM, problemN, problemK, 0});
            if (IsLastGroupAndNeedSplit(scheduler)) {
                scheduler.UpdateTailTile();
                ProcessSingleGroup<true>(scheduler, groupIdx);
            } else {
                ProcessSingleGroup<false>(scheduler, groupIdx);
            }
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        const auto& gmmParams = params.gmmParams;
        aBasePtr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
        bBasePtr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
        cBasePtr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
        scaleABasePtr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.mmadParams.scaleAGmAddr);
        scaleBBasePtr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.mmadParams.scaleBGmAddr);
        if (gmmParams.isBias == 1) {
            biasBasePtr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
            isBias_ = true;
        }
        if (params.groupListGmAddr != nullptr) {
            groupListGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(params.groupListGmAddr));
        }
        const ProblemShape initProblemShape{gmmParams.m, gmmParams.n, gmmParams.k, 0};
        problemShape_ = initProblemShape;
        groupNum_ = gmmParams.groupNum;
        groupListType_ = gmmParams.groupListType;
        curBaseM_ = gmmParams.baseM;
        baseN_ = gmmParams.baseN;
        if constexpr (!transA) {
            if constexpr (!weightNz) {
                perGroupBOffset_ = gmmParams.n * gmmParams.k;
            } else {
                if constexpr (transB) {
                    perGroupBOffset_ = static_cast<int64_t>(
                        Align16(gmmParams.n) * (IsFp4<AType>() ? Align64(gmmParams.k) : Align32(gmmParams.k)));
                } else {
                    perGroupBOffset_ = static_cast<int64_t>(
                        (IsFp4<AType>() ? Align64(gmmParams.n) : Align32(gmmParams.n)) * Align16(gmmParams.k));
                }
            }
        }

        const BlockShape l0Shape{static_cast<int64_t>(gmmParams.baseM), static_cast<int64_t>(gmmParams.baseN),
                                 static_cast<int64_t>(gmmParams.baseK), 0};
        // MX uses one shared scale K window for ScaleA/ScaleB. Tiling must provide scaleKAL1 == scaleKBL1.
        const L1Params l1Params{static_cast<uint64_t>(gmmParams.kAL1), static_cast<uint64_t>(gmmParams.kBL1),
                                static_cast<uint64_t>(gmmParams.scaleKAL1)};
        const typename BlockMmad::MmadParams mmadParams{l0Shape, l1Params, isBias_, gmmParams.dbL0C == 2};
        mmadOp_.Init(initProblemShape, mmadParams);
    }

    __aicore__ inline void BaseMBalance(BlockScheduler& scheduler, int64_t m, int64_t baseM)
    {
        if constexpr (!transA) {
            if (m <= 0) {
                return;
            }
            const int64_t safeBaseM = baseM > 0 ? baseM : static_cast<int64_t>(BLOCK_CUBE);
            const int64_t mCnt = (m + safeBaseM - 1) / safeBaseM;
            const int64_t balancedBaseM = (m + mCnt - 1) / mCnt;
            curBaseM_ = static_cast<uint32_t>((balancedBaseM + BLOCK_CUBE_MASK) & ~BLOCK_CUBE_MASK);
            scheduler.UpdateBaseM(curBaseM_);
        }
    }

    __aicore__ inline bool IsLastGroupAndNeedSplit(const BlockScheduler& scheduler)
    {
        return (scheduler.GetEndBlockIdx() + 1) <= (AscendC::GetBlockNum() >> 1);
    }

    __aicore__ inline void SetMNK(uint32_t groupIdx)
    {
        const int64_t splitValue = GetSplitValueFromGroupList(groupIdx);
        const int64_t n = AscendC::Te::Get<MNK_N>(problemShape_);
        // Current MX scalar path selects the split axis from LayoutA, not from GMMTiling::groupType.
        if constexpr (!transA) {
            problemShape_ = ProblemShape{splitValue, n, AscendC::Te::Get<MNK_K>(problemShape_), 0};
        } else {
            problemShape_ = ProblemShape{AscendC::Te::Get<MNK_M>(problemShape_), n, splitValue, 0};
        }
    }

    __aicore__ inline int64_t GetSplitValueFromGroupList(uint32_t groupIdx)
    {
        int64_t splitValue = 0;
        if (groupListType_ == GROUP_LIST_TYPE_OFFSET) {
            const int64_t offset = groupListGlobal_.GetValue(groupIdx);
            splitValue = offset - preOffset_;
            preOffset_ = offset;
        } else if (groupListType_ == GROUP_LIST_TYPE_LENGTH) {
            splitValue = groupListGlobal_.GetValue(groupIdx);
            preOffset_ += splitValue;
        } else {
            const uint32_t splitValueIdx =
                groupIdx * SPARSE_GROUP_LIST_ITEM_STRIDE + SPARSE_GROUP_LIST_SPLIT_VALUE_OFFSET;
            splitValue = groupListGlobal_.GetValue(splitValueIdx);
            preOffset_ += splitValue;
        }
        return splitValue;
    }

    __aicore__ inline void UpdateBaseOffsets(uint32_t groupIdx)
    {
        if (groupIdx == 0) {
            return;
        }

        const int64_t m = AscendC::Te::Get<MNK_M>(problemShape_);
        const int64_t n = AscendC::Te::Get<MNK_N>(problemShape_);
        const int64_t k = AscendC::Te::Get<MNK_K>(problemShape_);
        // The previous split offset follows the same current-scenario binding as SetMNK.
        const int64_t prevSplitOffset = preOffset_ - (transA ? k : m);
        if constexpr (!transA) {
            aOffset_ = IsFp4<AType>() ? ((prevSplitOffset * k) >> 1) : (prevSplitOffset * k);
            const int64_t scaleK = GetScaleK(k);
            bOffset_ = perGroupBOffset_ * static_cast<int64_t>(groupIdx);
            if constexpr (IsFp4<BType>()) {
                bOffset_ >>= 1;
            }
            scaleAOffset_ = prevSplitOffset * scaleK;
            scaleBOffset_ = static_cast<int64_t>(groupIdx) * n * scaleK;
            cOffset_ = prevSplitOffset * n;
        } else {
            aOffset_ = IsFp4<AType>() ? ((m * prevSplitOffset) >> 1) : (m * prevSplitOffset);
            bOffset_ = n * prevSplitOffset;
            const int64_t transScaleGroupOffset = prevSplitOffset >> MXFP_DIVISOR_SHIFT;
            const int64_t transScaleStart =
                (transScaleGroupOffset + static_cast<int64_t>(groupIdx)) << MXFP_MULTI_BASE_SHIFT;
            scaleAOffset_ = m * transScaleStart;
            scaleBOffset_ = n * transScaleStart;
            cOffset_ = static_cast<int64_t>(groupIdx) * m * n;
        }
        biasOffset_ = static_cast<int64_t>(groupIdx) * n;
    }

    __aicore__ inline int64_t GetScaleK(int64_t k) const
    {
        return ((k + MXFP_DIVISOR_SIZE - 1) >> MXFP_DIVISOR_SHIFT) << MXFP_MULTI_BASE_SHIFT;
    }

    template <bool isLastGroupAndNeedSplit>
    __aicore__ inline void ProcessSingleGroup(BlockScheduler& scheduler, uint32_t groupIdx)
    {
        BlockCoord tileIdx;
        if (!scheduler.GetTileIdx(tileIdx)) {
            return;
        }
        UpdateBaseOffsets(groupIdx);

        // Different groups may carry different split-M / split-K sizes. Keep the double-buffer phase continuous
        // across groups and only refresh the problem shape.
        mmadOp_.UpdateParamsForNextProblem(problemShape_);

        const int64_t problemM = AscendC::Te::Get<MNK_M>(problemShape_);
        const int64_t problemN = AscendC::Te::Get<MNK_N>(problemShape_);
        const int64_t problemK = AscendC::Te::Get<MNK_K>(problemShape_);
        const int64_t scaleK = GetScaleK(problemK);
        const int64_t baseN = static_cast<int64_t>(baseN_);
        auto layoutA = MakeLayoutA{}(problemM, problemK);
        auto layoutScaleA = MakeLayoutScaleA{}(problemM, scaleK);
        auto layoutB = MakeLayoutB{}(problemK, problemN);
        auto layoutScaleB = MakeLayoutScaleB{}(scaleK, problemN);
        auto layoutC = MakeLayoutC{}(problemM, problemN);
        auto gmA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aBasePtr_ + aOffset_), layoutA);
        auto gmScaleA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleABasePtr_ + scaleAOffset_), layoutScaleA);
        auto gmB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bBasePtr_ + bOffset_), layoutB);
        auto gmScaleB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleBBasePtr_ + scaleBOffset_), layoutScaleB);
        auto gmC = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cBasePtr_ + cOffset_), layoutC);
        auto biasPtr = isBias_ ? (biasBasePtr_ + biasOffset_) : nullptr;
        auto layoutBias = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(static_cast<int64_t>(1), problemN);
        auto gmBias = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(biasPtr), layoutBias);

        if constexpr (!isLastGroupAndNeedSplit) {
            SetL2CacheHint(gmB, gmScaleB, problemM, static_cast<int64_t>(curBaseM_), static_cast<int64_t>(baseN_));
        }

        do {
            const SchedulerShape schedulerBlockShape = scheduler.GetBlockShape(tileIdx);
            const int64_t tileM = AscendC::Te::Get<MNK_M>(schedulerBlockShape);
            const int64_t tileN = AscendC::Te::Get<MNK_N>(schedulerBlockShape);
            if (tileM <= 0 || tileN <= 0) {
                continue;
            }
            const int64_t tileK = AscendC::Te::Get<MNK_K>(schedulerBlockShape);
            const int64_t tileB = AscendC::Te::Get<MNK_B>(schedulerBlockShape);
            const int64_t tileIdxM = AscendC::Te::Get<MNK_M>(tileIdx);
            const int64_t tileIdxN = AscendC::Te::Get<MNK_N>(tileIdx);
            const BlockShape singleShape{tileM, tileN, tileK, tileB};
            const int64_t mPos = tileIdxM * curBaseM_ + tileK;
            const int64_t nPos = tileIdxN * baseN + tileB;

            auto gmBlockA = gmA.Slice(
                AscendC::Te::MakeCoord(mPos, static_cast<int64_t>(0)),
                AscendC::Te::MakeShape(tileM, problemK));
            auto gmBlockScaleA = gmScaleA.Slice(
                AscendC::Te::MakeCoord(mPos, static_cast<int64_t>(0)),
                AscendC::Te::MakeShape(tileM, scaleK));
            auto gmBlockB = gmB.Slice(
                AscendC::Te::MakeCoord(static_cast<int64_t>(0), nPos),
                AscendC::Te::MakeShape(problemK, tileN));
            auto gmBlockScaleB = gmScaleB.Slice(
                AscendC::Te::MakeCoord(static_cast<int64_t>(0), nPos),
                AscendC::Te::MakeShape(scaleK, tileN));
            auto gmBlockBias = gmBias.Slice(
                AscendC::Te::MakeCoord(static_cast<int64_t>(0), nPos),
                AscendC::Te::MakeShape(static_cast<int64_t>(1), tileN));
            auto gmBlockC = gmC.Slice(
                AscendC::Te::MakeCoord(mPos, nPos),
                AscendC::Te::MakeShape(tileM, tileN));
            mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleShape);
        } while (scheduler.GetTileIdx(tileIdx));
    }

private:
    BlockMmad mmadOp_;
    ProblemShape problemShape_{};
    AscendC::GlobalTensor<int64_t> groupListGlobal_;

    __gm__ AType* aBasePtr_{nullptr};
    __gm__ BType* bBasePtr_{nullptr};
    __gm__ CType* cBasePtr_{nullptr};
    __gm__ BiasType* biasBasePtr_{nullptr};
    __gm__ fp8_e8m0_t* scaleABasePtr_{nullptr};
    __gm__ fp8_e8m0_t* scaleBBasePtr_{nullptr};

    int64_t preOffset_{0};
    int64_t aOffset_{0};
    int64_t bOffset_{0};
    int64_t perGroupBOffset_{0};
    int64_t scaleAOffset_{0};
    int64_t scaleBOffset_{0};
    int64_t biasOffset_{0};
    int64_t cOffset_{0};
    uint32_t groupNum_{0};
    uint32_t curBaseM_{0};
    uint32_t baseN_{0};
    uint8_t groupListType_{GROUP_LIST_TYPE_OFFSET};
    bool isBias_{false};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
