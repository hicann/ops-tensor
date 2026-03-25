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
 * \file kernel_qbmm_mx.h
 * \brief
 */

#ifndef MATMUL_KERNEL_KERNEL_QBMM_MX_H
#define MATMUL_KERNEL_KERNEL_QBMM_MX_H
#include "kernel_basic_intf.h"
#include "../utils/common_utils.h"
#include "../utils/fill_utils.h"
#include "../utils/quant_batch_matmul_constant.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../utils/coord_utils.h"
#include "../utils/tensor_utils.h"
#include "../block/block_scheduler_qbmm.h"
#include "include/experimental/tensor_api/tensor.h"

namespace Cmct {
namespace Gemm {
namespace Kernel {
#define QBMM_MX_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler, bool isAtomicAdd>
#define QBMM_MX_KERNEL_FUN_TEM_PARAMS ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler, isAtomicAdd

using namespace Cmct;
using namespace Cmct::Gemm;
using namespace AscendC;
using namespace AscendC::Te;

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
class QuantMmBatchMX {
public:
    __aicore__ inline QuantMmBatchMX()
    {}
    __aicore__ inline ~QuantMmBatchMX()
    {}

    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;

    using BlockSchedulerOp = typename Block::BlockSchedulerSelector<
        ProblemShape, typename BlockMmad::L1TileShape, typename BlockMmad::L0TileShape, BlockScheduler, transA,
        transB>::SchedulerOp;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutB = typename BlockMmad::LayoutB;
    static constexpr CubeFormat FormatB = TagToFormat<LayoutB>::format;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    // x1,x2,x1Scale,x2Scale,bias,y
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;

    using MakeLayoutA =
        AscendC::Std::conditional_t<transA, AscendC::Te::DNLayoutFormat<AType>, AscendC::Te::NDLayoutFormat<AType>>;
    using MakeLayoutB = AscendC::Std::conditional_t<
        transB,
        AscendC::Std::conditional_t<
            FormatB == CubeFormat::NZ, AscendC::Te::ZnLayoutFormat<BType>, AscendC::Te::DNLayoutFormat<BType>>,
        AscendC::Std::conditional_t<
            FormatB == CubeFormat::NZ, AscendC::Te::NzLayoutFormat<BType>, AscendC::Te::NDLayoutFormat<BType>>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        transA, AscendC::Te::ScaleADNLayoutFormat<fp8_e8m0_t>, AscendC::Te::ScaleANDLayoutFormat<fp8_e8m0_t>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        transB, AscendC::Te::ScaleBDNLayoutFormat<fp8_e8m0_t>, AscendC::Te::ScaleBNDLayoutFormat<fp8_e8m0_t>>;

    struct QBMMTiling {
        uint32_t batchA1;
        uint32_t batchA2;
        uint32_t batchA3;
        uint32_t batchA4;
        uint32_t batchB1;
        uint32_t batchB2;
        uint32_t batchB3;
        uint32_t batchB4;
        uint32_t batchC1;
        uint32_t batchC2;
        uint32_t batchC3;
        uint32_t batchC4;
        uint32_t biasThreeDim;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t isBias;
        uint32_t dbL0C;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        L1Params l1Params;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
    };

public:
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void Run(const Params& params);
    __aicore__ inline void operator()(const Params& params)
    {
        Run(params);
    }

private:
    __aicore__ inline void ResetGmAddr(const Params& params);
    __aicore__ inline void ProcessSingleBatch(
        const Params& params, BlockSchedulerOp& bs, uint64_t batchCnt, bool isTailRound);

    __aicore__ inline void ProcessWithBatch(const Params& params, BlockSchedulerOp& bs);
    __aicore__ inline TupleShape ToShapeTuple(const ProblemShape& problemShape)
    {
        return {problemShape.m, problemShape.n, problemShape.k};
    }
    __aicore__ inline void AddBatchOffset(const Params& params);

private:
    BlockMmad mmadOp_;
    TupleShape problemShape_{};
    BlockOffset blockOffset_{0, 0, 0, 0, 0, 0};

    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr; // 可选输入，直接初始化
    __gm__ fp8_e8m0_t* pertokenScaleGmAddr_;
    __gm__ fp8_e8m0_t* scaleGmAddr_;

    uint64_t blockIdx_;
    uint64_t batchCOffset_{0};
    uint64_t batchAOffset_{0};
    uint64_t batchBOffset_{0};
    bool isBiasThreeDim_{false};
    bool isBias_{false};
    bool needUpdateTail_{false};
};

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchMX<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::Run(const Params& params)
{
    Init(params);
    BlockSchedulerOp bs(params.problemShape, params.schParams);
    problemShape_ = ToShapeTuple(params.problemShape);

    BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
    bool enableL0CPingPong = (params.qbmmParams.dbL0C > 1);
    mmadOp_.Init(problemShape_, l0TileShape, params.l1Params, isBias_, enableL0CPingPong);

    if (params.problemShape.b == 1) {
        ProcessSingleBatch(params, bs, 0, true);
        return;
    }
    ProcessWithBatch(params, bs);
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchMX<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::Init(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }
    if (params.qbmmParams.isBias == 1) {
        if (params.qbmmParams.biasThreeDim == 1) {
            isBiasThreeDim_ = true;
        }
        isBias_ = true;
    }

    ResetGmAddr(params);
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchMX<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::ResetGmAddr(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }

    aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    pertokenScaleGmAddr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.mmadParams.pertokenScaleGmAddr);
    scaleGmAddr_ = reinterpret_cast<__gm__ fp8_e8m0_t*>(params.mmadParams.scaleGmAddr);
    if (isBias_) {
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchMX<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::ProcessWithBatch(
    const Params& params, BlockSchedulerOp& bs)
{
    uint64_t batchC3C4 = static_cast<uint64_t>(params.qbmmParams.batchC3) * params.qbmmParams.batchC4;
    uint64_t batchC2C3C4 = params.qbmmParams.batchC2 * batchC3C4;
    uint64_t batchB3B4 = static_cast<uint64_t>(params.qbmmParams.batchB3) * params.qbmmParams.batchB4;
    uint64_t batchB2B3B4 = params.qbmmParams.batchB2 * batchB3B4;
    uint64_t batchA3A4 = static_cast<uint64_t>(params.qbmmParams.batchA3) * params.qbmmParams.batchA4;
    uint64_t batchA2A3A4 = params.qbmmParams.batchA2 * batchA3A4;
    uint32_t multiA1C1 = params.qbmmParams.batchA1 / params.qbmmParams.batchC1;
    uint32_t multiA2C2 = params.qbmmParams.batchA2 / params.qbmmParams.batchC2;
    uint32_t multiA3C3 = params.qbmmParams.batchA3 / params.qbmmParams.batchC3;
    uint32_t multiA4C4 = params.qbmmParams.batchA4 / params.qbmmParams.batchC4;
    uint32_t multiB1C1 = params.qbmmParams.batchB1 / params.qbmmParams.batchC1;
    uint32_t multiB2C2 = params.qbmmParams.batchB2 / params.qbmmParams.batchC2;
    uint32_t multiB3C3 = params.qbmmParams.batchB3 / params.qbmmParams.batchC3;
    uint32_t multiB4C4 = params.qbmmParams.batchB4 / params.qbmmParams.batchC4;

    uint64_t batchC1Offset = 0;
    uint64_t batchA1Offset = 0;
    uint64_t batchB1Offset = 0;
    uint64_t curBatchC = 1UL;
    uint64_t totalCnt = bs.GetTotalCnt() * params.problemShape.b;
    for (uint64_t b1Index = 0; b1Index < params.qbmmParams.batchC1; ++b1Index) {
        uint64_t batchC2Offset = batchC1Offset;
        uint64_t batchA2Offset = batchA1Offset;
        uint64_t batchB2Offset = batchB1Offset;
        for (uint64_t b2Index = 0; b2Index < params.qbmmParams.batchC2; ++b2Index) {
            uint64_t batchC3Offset = batchC2Offset;
            uint64_t batchA3Offset = batchA2Offset;
            uint64_t batchB3Offset = batchB2Offset;
            for (uint64_t b3Index = 0; b3Index < params.qbmmParams.batchC3; ++b3Index) {
                batchCOffset_ = batchC3Offset;
                batchAOffset_ = batchA3Offset;
                batchBOffset_ = batchB3Offset;
                for (uint64_t b4Index = 0; b4Index < params.qbmmParams.batchC4; ++b4Index) {
                    bool isTailRound =
                        curBatchC * bs.GetTotalCnt() > (totalCnt / AscendC::GetBlockNum()) * AscendC::GetBlockNum();
                    AddBatchOffset(params);
                    ProcessSingleBatch(params, bs, (params.problemShape.b - curBatchC), isTailRound);
                    curBatchC++;
                    batchCOffset_ += 1;
                    batchAOffset_ += multiA4C4;
                    batchBOffset_ += multiB4C4;
                }
                batchC3Offset += params.qbmmParams.batchC4;
                batchA3Offset += params.qbmmParams.batchA4 * static_cast<uint64_t>(multiA3C3);
                batchB3Offset += params.qbmmParams.batchB4 * static_cast<uint64_t>(multiB3C3);
            }
            batchC2Offset += batchC3C4;
            batchA2Offset += batchA3A4 * multiA2C2;
            batchB2Offset += batchB3B4 * multiB2C2;
        }
        batchC1Offset += batchC2C3C4;
        batchA1Offset += batchA2A3A4 * multiA1C1;
        batchB1Offset += batchB2B3B4 * multiB1C1;
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchMX<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::AddBatchOffset(const Params& params)
{
    ResetGmAddr(params);
    constexpr bool isFp4Type = AscendC::IsSameType<AType, fp4x2_e2m1_t>::value;
    constexpr uint64_t sizeShift = isFp4Type ? 1 : 0;
    aGmAddr_ += (batchAOffset_ * params.problemShape.m * params.problemShape.k) >> sizeShift;
    if constexpr (FormatB == CubeFormat::NZ) {
        if constexpr (transB) {
            bGmAddr_ += batchBOffset_ * Cmct::Gemm::CeilDiv(params.problemShape.k, C0_SIZE_B8) *
                        Cmct::Gemm::CeilDiv(params.problemShape.n, AscendC::BLOCK_CUBE) * AscendC::BLOCK_CUBE *
                        C0_SIZE_B8;
        } else {
            bGmAddr_ += batchBOffset_ * Cmct::Gemm::CeilDiv(params.problemShape.n, C0_SIZE_B8) *
                        Cmct::Gemm::CeilDiv(params.problemShape.k, AscendC::BLOCK_CUBE) * AscendC::BLOCK_CUBE *
                        C0_SIZE_B8;
        }
    } else {
        bGmAddr_ += (batchBOffset_ * params.problemShape.n * params.problemShape.k) >> sizeShift;
    }
    cGmAddr_ += batchCOffset_ * params.problemShape.m * params.problemShape.n;
    if (isBiasThreeDim_) {
        biasGmAddr_ += batchCOffset_ * params.problemShape.n;
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchMX<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::ProcessSingleBatch(
    const Params& params, BlockSchedulerOp& bs, uint64_t restBatch, bool isTailRound)
{
    auto layoutA = MakeLayoutA{}(params.problemShape.m, params.problemShape.k);
    auto layoutScaleA = MakeLayoutScaleA{}(params.problemShape.m, Cmct::Gemm::CeilDiv(params.problemShape.k, 64) * 2);
    auto layoutB = MakeLayoutB{}(params.problemShape.k, params.problemShape.n);
    auto layoutScaleB = MakeLayoutScaleB{}(Cmct::Gemm::CeilDiv(params.problemShape.k, 64) * 2, params.problemShape.n);
    auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(aGmAddr_), layoutA);
    auto gmScaleA = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(pertokenScaleGmAddr_), layoutScaleA);
    auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(bGmAddr_), layoutB);
    auto gmScaleB = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(scaleGmAddr_), layoutScaleB);
    auto layoutBias = AscendC::Te::MakeNDLayout<BiasType>(1L, params.problemShape.n);
    auto gmBias = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(biasGmAddr_), layoutBias);
    auto layoutC = AscendC::Te::MakeNDLayout<CType>(params.problemShape.m, params.problemShape.n);
    auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(cGmAddr_), layoutC);

    BlockCoord blockIdx;
    auto& mTailTile = params.schParams.mTailTile;
    auto& nTailTile = params.schParams.nTailTile;
    // both tail of current batch and rest batch are tail round
    if (needUpdateTail_ ||
        (isTailRound && ((bs.GetEndBlockIdx() + 1) + (restBatch * bs.GetTotalCnt())) * mTailTile * nTailTile <=
                            AscendC::GetBlockNum())) {
        needUpdateTail_ = true;
        bs.UpdateTailTile(mTailTile, nTailTile);
    }
    int64_t mPos = 0L;
    int64_t nPos = 0L;
    constexpr int64_t kPos = 0L; // 不切K，所以坐标是0
    while (bs.GetTileIdx(blockIdx)) {
        BlockShape singleShape = bs.template GetBlockShape<
            QuantBatchMatmul::QuantMode::MX_PERGROUP_MODE, QuantBatchMatmul::QuantMode::MX_PERGROUP_MODE, FormatB>(
            blockIdx);
        if (Get<MNK_M>(singleShape) <= 0 || Get<MNK_N>(singleShape) <= 0) {
            return;
        }

        bs.GetTileCoord(blockIdx, mPos, nPos);
        auto gmBlockA = gmA(
            AscendC::Te::MakeCoord(mPos, kPos), AscendC::Te::MakeShape(Get<MNK_M>(singleShape), params.problemShape.k));
        auto gmBlockScaleA = gmScaleA(
            AscendC::Te::MakeCoord(mPos, kPos),
            AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Cmct::Gemm::CeilDiv(params.problemShape.k, 64) * 2));
        auto gmBlockB = gmB(
            AscendC::Te::MakeCoord(kPos, nPos), AscendC::Te::MakeShape(params.problemShape.k, Get<MNK_N>(singleShape)));
        auto gmBlockScaleB = gmScaleB(
            AscendC::Te::MakeCoord(kPos, nPos),
            AscendC::Te::MakeShape(Cmct::Gemm::CeilDiv(params.problemShape.k, 64) * 2, Get<MNK_N>(singleShape)));
        auto gmBlockBias =
            gmBias(AscendC::Te::MakeCoord(0L, nPos), AscendC::Te::MakeShape(1L, Get<MNK_N>(singleShape)));
        auto gmBlockC =
            gmC(AscendC::Te::MakeCoord(mPos, nPos),
                AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape)));
        mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleShape);
    }
    bs.UpdateNextBatchBlockRoundParams();
}
} // namespace Kernel
} // namespace Gemm
} // namespace Cmct
#endif