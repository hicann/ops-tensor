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
 * \file block_mmad_iterbatch_broadcast.h
 * \brief MMAD block for IterBatch-Broadcast: batched L1/L0 pipelining + broadcast data sharing
 *        Two modes based on iterBatchL0:
 *        - Batched mode (iterBatchL0 > 1): no MNK tiling (tiling guarantees baseM>=M),
 *          3D L0 layouts, batched copy, batched fixpipe, single L0A/L0B buffer
 *        - Single-batch mode (iterBatchL0 == 1): MNK tiling with 2D L0 layouts,
 *          avoids 3D layout Slice issues; L1→L0 uses computed byte offsets for per-batch
 *          2D L1 tensors; MMAD uses 2D L0 tensors; fixpipe uses 2D NZ L0C + 2D NDExt GM
 *        GM→L1: kernel layer does gmA/gmB Slice for broadcast mapping before calling MMAD
 *        MMAD CopyGM2L1A/B receives pre-sliced GM tensors, no broadcast index logic needed
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "block_mmad.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <
    bool ABc, bool BBc,
    class AType_, class LayoutA_, class BType_, class LayoutB_,
    class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<
    MatmulIterBatchBroadcast<ABc, BBc>, AType_, LayoutA_, BType_, LayoutB_,
    CType_, LayoutC_, BiasType_, LayoutBias_> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using LayoutBias = LayoutBias_;
    using DispatchPolicy = MatmulIterBatchBroadcast<ABc, BBc>;
    using TupleShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    static constexpr bool transA = IsTrans<LayoutA>::value;
    static constexpr bool transB = IsTrans<LayoutB>::value;
    static constexpr bool aBroadcast = ABc;
    static constexpr bool bBroadcast = BBc;
    static constexpr uint64_t BUFFER_NUM = 2;
    constexpr static uint16_t MTE1_MTE2_EVENT_ID_NUM = 4;
    constexpr static uint64_t HALF_L1_SIZE = AscendC::TOTAL_L1_SIZE / BUFFER_NUM;
    constexpr static uint64_t HALF_L0_SIZE = AscendC::TOTAL_L0A_SIZE / BUFFER_NUM;
    constexpr static uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / BUFFER_NUM;
    using A_T = AType;
    using B_T = BType;
    using C_T = CType;
    using Bias_T = BiasType;
    using LayoutAl1Ptn = AscendC::Std::conditional_t<transA,
        AscendC::Te::ZNLayoutPtn, AscendC::Te::NZLayoutPtn>;
    using LayoutBl1Ptn = AscendC::Std::conditional_t<transB,
        AscendC::Te::ZNLayoutPtn, AscendC::Te::NZLayoutPtn>;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR groupListGmAddr{nullptr};
        GM_ADDR workspaceGmAddr{nullptr};
    };

public:
    __aicore__ inline BlockMmad()
    {
        if ASCEND_IS_NOT_AIV {
            for (uint16_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; i++) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(SIXTH_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(SEVENTH_FLAG);
        }
    }

    __aicore__ inline ~BlockMmad()
    {
        if ASCEND_IS_NOT_AIV {
            for (uint16_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
            }
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(SIXTH_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(SEVENTH_FLAG);
        }
    }

    __aicore__ inline void Init(
        const TupleShape& shape,
        uint64_t mainIterBatchL1, uint64_t mainIterBatchL0,
        bool isBias,
        uint64_t baseM, uint64_t baseN, uint64_t baseK,
        uint64_t broadcastAxisA, uint64_t broadcastAxisB)
    {
        m_ = AscendC::Te::Get<MNK_M>(shape);
        n_ = AscendC::Te::Get<MNK_N>(shape);
        k_ = AscendC::Te::Get<MNK_K>(shape);
        baseM_ = baseM;
        baseN_ = baseN;
        baseK_ = baseK;
        isBias_ = isBias;
        mainIterBatchL1_ = mainIterBatchL1;
        mainIterBatchL0_ = mainIterBatchL0;
        abL1BufId_ = 0;
        broadcastAxisA_ = broadcastAxisA;
        broadcastAxisB_ = broadcastAxisB;
        uint64_t al1BatchCount;
        if constexpr (aBroadcast) {
            al1BatchCount = (broadcastAxisA_ == LAST_BATCH_DIM) ? 1 : mainIterBatchL1;
        } else {
            al1BatchCount = mainIterBatchL1;
        }
        uint64_t bl1BatchCount;
        if constexpr (bBroadcast) {
            bl1BatchCount = (broadcastAxisB_ == LAST_BATCH_DIM) ? 1 : mainIterBatchL1;
        } else {
            bl1BatchCount = mainIterBatchL1;
        }
        const uint64_t c0Size = BLOCK_BYTE_SIZE / sizeof(A_T);
        if constexpr (!transA) {
            aL1BatchStrideElems_ = CeilAlign(m_, static_cast<uint64_t>(BLOCK_CUBE)) * CeilAlign(k_, c0Size);
        } else {
            aL1BatchStrideElems_ = CeilAlign(m_, c0Size) * CeilAlign(k_, static_cast<uint64_t>(BLOCK_CUBE));
        }
        if constexpr (!transB) {
            bL1BatchStrideElems_ = CeilAlign(k_, static_cast<uint64_t>(BLOCK_CUBE)) * CeilAlign(n_, c0Size);
        } else {
            bL1BatchStrideElems_ = CeilAlign(k_, c0Size) * CeilAlign(n_, static_cast<uint64_t>(BLOCK_CUBE));
        }
        aL1OneBuffer_ = aL1BatchStrideElems_ * sizeof(A_T) * al1BatchCount;
        bL1OneBuffer_ = bL1BatchStrideElems_ * sizeof(B_T) * bl1BatchCount;
    }

    template <typename TensorC, typename TensorA, typename TensorB, typename TensorBias>
    __aicore__ inline void operator()(
        TensorC gmC, TensorA gmA, TensorB gmB, TensorBias gmBias,
        uint64_t curIterBatchL1)
    {
        uint64_t curAl1Count = GetAl1Count(curIterBatchL1);
        uint64_t curBl1Count = GetBl1Count(curIterBatchL1);
        uint64_t l1BufId = abL1BufId_ & 0x1;
        uint64_t offsetAl1 = HALF_L1_SIZE * l1BufId;
        uint64_t offsetBl1 = offsetAl1 + aL1OneBuffer_;

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufId));
        auto al1Tensor = CopyGM2L1A(gmA, offsetAl1, curAl1Count);
        auto bl1Tensor = CopyGM2L1B(gmB, offsetBl1, curBl1Count);
        auto biasL1Tensor = CopyGM2L1Bias(gmBias, offsetAl1);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(static_cast<uint16_t>(l1BufId));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(static_cast<uint16_t>(l1BufId));

        auto copyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{}, AscendC::Te::CopyL12L0ATraitDefault{});
        auto copyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{}, AscendC::Te::CopyL12L0BTraitDefault{});
        auto copyL12BT = AscendC::Te::MakeCopy(AscendC::Te::CopyL12BT{}, AscendC::Te::CopyL12BTTraitDefault{});
        auto copyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{}, AscendC::Te::CopyL0C2GMTraitDefault{});
        constexpr auto mmadAtom = AscendC::Te::MakeMmad(AscendC::Te::MmadOperation{}, AscendC::Te::MmadTraitDefault{});

        if (mainIterBatchL0_ > 1) {
            BatchedMmadLoop(al1Tensor, bl1Tensor, gmC, biasL1Tensor, curIterBatchL1,
                copyL12L0A, copyL12L0B, copyL12BT, copyL0C2GM, mmadAtom, offsetAl1);
        } else {
            SingleBatchMmadLoop(al1Tensor, bl1Tensor, gmC, biasL1Tensor, curIterBatchL1,
                copyL12L0A, copyL12L0B, copyL12BT, copyL0C2GM, mmadAtom);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufId));
        abL1BufId_++;
    }

private:
    struct L0TileInfo {
        uint64_t curAl0Cnt;
        uint64_t curBl0Cnt;
        uint64_t curM;
        uint64_t curN;
        uint64_t curK;
        uint64_t al0BatchStart;
        uint64_t bl0BatchStart;
        uint64_t iterK;
        uint64_t iterN;
        uint64_t iterM;
    };
    struct L0SingleBatchTileInfo {
        uint64_t curM;
        uint64_t curN;
        uint64_t curK;
        uint64_t iterM;
        uint64_t iterN;
        uint64_t iterK;
        uint64_t al1BatchIdx;
        uint64_t bl1BatchIdx;
        uint64_t l0BufId;
    };
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t baseM_{1};
    uint64_t baseN_{1};
    uint64_t baseK_{1};
    uint64_t mainIterBatchL1_{1};
    uint64_t mainIterBatchL0_{1};
    bool isBias_{false};
    uint64_t abL1BufId_{0};
    uint64_t aL1OneBuffer_{0};
    uint64_t bL1OneBuffer_{0};
    uint64_t aL1BatchStrideElems_{1};
    uint64_t bL1BatchStrideElems_{1};
    uint64_t broadcastAxisA_{MAX_BATCH_DIM};
    uint64_t broadcastAxisB_{MAX_BATCH_DIM};

    template <typename TensorA>
    __aicore__ inline auto CopyGM2L1A(const TensorA& gmA, uint64_t offsetAl1, uint64_t al1Count)
    {
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        auto al1Layout = AscendC::Te::MakeFrameLayout<
            LayoutAl1Ptn, AscendC::Te::LayoutTraitDefault<AType>>(al1Count, m_, k_);
        auto al1Tensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(offsetAl1), al1Layout);
        AscendC::Te::Copy(copyGM2L1, al1Tensor, gmA);
        return al1Tensor;
    }

    template <typename TensorB>
    __aicore__ inline auto CopyGM2L1B(const TensorB& gmB, uint64_t offsetBl1, uint64_t bl1Count)
    {
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        auto bl1Layout = AscendC::Te::MakeFrameLayout<
            LayoutBl1Ptn, AscendC::Te::LayoutTraitDefault<BType>>(bl1Count, k_, n_);
        auto bl1Tensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(offsetBl1), bl1Layout);
        AscendC::Te::Copy(copyGM2L1, bl1Tensor, gmB);
        return bl1Tensor;
    }

    template <typename TensorBias>
    __aicore__ inline auto CopyGM2L1Bias(const TensorBias& gmBias, uint64_t offsetAl1)
    {
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        uint64_t nAlign = CeilAlign(n_, static_cast<uint64_t>(BLOCK_CUBE));
        auto biasL1Layout = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, nAlign);
        auto biasL1Tensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BiasType>(offsetAl1 + aL1OneBuffer_ + bL1OneBuffer_),
            biasL1Layout);
        if (isBias_) {
            AscendC::Te::Copy(copyGM2L1, biasL1Tensor, gmBias);
        }
        return biasL1Tensor;
    }

    // --- Batched mode (iterBatchL0 > 1): 3D L0 layouts, batched copy/fixpipe ---
    template <typename TensorAl1, typename TensorBl1, typename TensorC,
              typename TensorBiasL1, typename CopyL12L0AT, typename CopyL12L0BT, typename CopyL12BTT,
              typename CopyL0C2GMT, typename MmadAtomT>
    __aicore__ inline void BatchedMmadLoop(
        const TensorAl1& al1Tensor, const TensorBl1& bl1Tensor,
        TensorC gmC, const TensorBiasL1& biasL1Tensor, uint64_t curIterBatchL1,
        CopyL12L0AT copyL12L0A, CopyL12L0BT copyL12L0B, CopyL12BTT copyL12BT,
        CopyL0C2GMT copyL0C2GM, MmadAtomT mmadAtom, uint64_t offsetAl1)
    {
        uint64_t batchStepCnt = CeilDiv(curIterBatchL1, mainIterBatchL0_);
        const uint64_t c0Size = BLOCK_BYTE_SIZE / sizeof(A_T);
        uint64_t l0aPerBatchBytes = CeilAlign(m_, static_cast<uint64_t>(BLOCK_CUBE)) *
            CeilAlign(k_, c0Size) * sizeof(A_T);
        uint64_t l0bPerBatchBytes = CeilAlign(k_, c0Size) *
            CeilAlign(n_, static_cast<uint64_t>(BLOCK_CUBE)) * sizeof(B_T);
        uint64_t l0cPerBatchBytes = CeilAlign(m_, static_cast<uint64_t>(BLOCK_CUBE)) *
            CeilAlign(n_, static_cast<uint64_t>(BLOCK_CUBE)) * sizeof(float);
        uint64_t l0BufId = 0;
        for (uint64_t iter1 = 0; iter1 < batchStepCnt; ++iter1) {
            uint64_t curIterBatchL0 = (iter1 + 1 == batchStepCnt) ?
                (curIterBatchL1 - mainIterBatchL0_ * iter1) : mainIterBatchL0_;
            uint64_t curAl0Cnt = GetAl0Count(curIterBatchL0);
            uint64_t curBl0Cnt = GetBl0Count(curIterBatchL0);
            uint64_t al0BatchStart = GetAl0BatchStart(iter1);
            uint64_t bl0BatchStart = GetBl0BatchStart(iter1);
            l0BufId = iter1 % BUFFER_NUM;
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(static_cast<uint16_t>(l0BufId));
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(l0BufId) + SIXTH_FLAG);
            L0TileInfo tileInfo{curAl0Cnt, curBl0Cnt, m_, n_, k_, al0BatchStart, bl0BatchStart, 0, 0, 0};
            CopyL1ToL0(al1Tensor, bl1Tensor, biasL1Tensor, l0BufId, tileInfo, offsetAl1,
                copyL12L0A, copyL12L0B, copyL12BT);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(l0BufId));
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(l0BufId));
            for (uint64_t batchL0Idx = 0; batchL0Idx < curIterBatchL0; ++batchL0Idx) {
                uint64_t al0BatchIdx = GetABatchIdx(batchL0Idx);
                uint64_t bl0BatchIdx = GetBBatchIdx(batchL0Idx);
                ComputeMmad(m_, n_, k_, 0, mmadAtom,
                    HALF_L0_SIZE * l0BufId + al0BatchIdx * l0aPerBatchBytes,
                    HALF_L0_SIZE * l0BufId + bl0BatchIdx * l0bPerBatchBytes,
                    HALF_L0C_SIZE * l0BufId + batchL0Idx * l0cPerBatchBytes);
            }
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(l0BufId) + SIXTH_FLAG);
            FixL0CToGM(gmC, l0BufId, curIterBatchL0, iter1, copyL0C2GM);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(static_cast<uint16_t>(l0BufId));
        }
    }

    template <typename TensorAl1, typename TensorBl1, typename TensorBiasL1,
              typename CopyL12L0AT, typename CopyL12L0BT, typename CopyL12BTT>
    __aicore__ inline void CopyL1ToL0(
        const TensorAl1& al1Tensor, const TensorBl1& bl1Tensor,
        const TensorBiasL1& biasL1Tensor, uint64_t l0BufId,
        const L0TileInfo& tileInfo, uint64_t offsetAl1,
        CopyL12L0AT copyL12L0A, CopyL12L0BT copyL12L0B, CopyL12BTT copyL12BT)
    {
        // A-side: transA always 2D per-batch loop; !transA uses 3D or 2D for FP32+broadcastAxis==3
        if constexpr (transA) {
            CopyAl1ToL0PerBatch<AscendC::Te::ZNLayoutPtn>(l0BufId, tileInfo, offsetAl1, copyL12L0A);
        } else {
            if (broadcastAxisA_ == LAST_BATCH_DIM && sizeof(AType) == sizeof(float)) {
                CopyAl1ToL0PerBatch<AscendC::Te::NZLayoutPtn>(l0BufId, tileInfo, offsetAl1, copyL12L0A);
            } else {
                auto al0KLayout = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn,
                    AscendC::Te::LayoutTraitDefault<AType>>(tileInfo.curAl0Cnt, tileInfo.curM, tileInfo.curK);
                auto al0KTensor = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(HALF_L0_SIZE * l0BufId), al0KLayout);
                auto al1Slice = al1Tensor.Slice(AscendC::Te::MakeCoord(tileInfo.al0BatchStart,
                    AscendC::Te::MakeCoord(tileInfo.iterM * baseM_, tileInfo.iterK * baseK_)),
                    AscendC::Te::MakeShape(tileInfo.curAl0Cnt, AscendC::Te::MakeShape(tileInfo.curM, tileInfo.curK)));
                AscendC::Te::Copy(copyL12L0A, al0KTensor, al1Slice);
            }
        }
        // B-side: !transB+FP32 always 2D per-batch; transB or !FP32 uses 3D default
        if constexpr (!transB && sizeof(BType) == sizeof(float)) {
            CopyBl1ToL0PerBatch<AscendC::Te::NZLayoutPtn>(l0BufId, tileInfo, offsetAl1, copyL12L0B);
        } else {
            auto bl0KLayout = AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn,
                AscendC::Te::LayoutTraitDefault<BType>>(tileInfo.curBl0Cnt, tileInfo.curK, tileInfo.curN);
            auto bl0KTensor = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(HALF_L0_SIZE * l0BufId), bl0KLayout);
            auto bl1Slice = bl1Tensor.Slice(AscendC::Te::MakeCoord(tileInfo.bl0BatchStart,
                AscendC::Te::MakeCoord(tileInfo.iterK * baseK_, tileInfo.iterN * baseN_)),
                AscendC::Te::MakeShape(tileInfo.curBl0Cnt, AscendC::Te::MakeShape(tileInfo.curK, tileInfo.curN)));
            AscendC::Te::Copy(copyL12L0B, bl0KTensor, bl1Slice);
        }
        if (isBias_ && tileInfo.iterK == 0) {
            uint64_t nl0Align = CeilAlign(tileInfo.curN, static_cast<uint64_t>(BLOCK_CUBE));
            auto biasL0Layout = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, nl0Align);
            auto biasL0Tensor = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<
                AscendC::Te::Location::BIAS, float>(0), biasL0Layout);
            auto biasL1Slice = biasL1Tensor.Slice(AscendC::Te::MakeCoord(0, 0),
                AscendC::Te::MakeShape(1UL, tileInfo.curN));
            AscendC::Te::Copy(copyL12BT, biasL0Tensor, biasL1Slice);
        }
    }

    template <typename MmadAtomT>
    __aicore__ inline void ComputeMmad(
        uint64_t curM, uint64_t curN, uint64_t curK,
        uint64_t iterK, MmadAtomT mmadAtom,
        uint64_t al0ByteOff, uint64_t bl0ByteOff, uint64_t l0cByteOff)
    {
        auto al0Layout = AscendC::Te::MakeFrameLayout<
            AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>(curM, curK);
        auto al0Tensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(al0ByteOff), al0Layout);
        auto bl0Layout = AscendC::Te::MakeFrameLayout<
            AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>(curK, curN);
        auto bl0Tensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(bl0ByteOff), bl0Layout);
        auto l0cLayout = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<16>>(curM, curN);
        auto l0cTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(l0cByteOff), l0cLayout);
        bool cmatrixInitVal = (iterK == 0 && !isBias_);
        AscendC::Te::MmadParams mmadParams(curM, curN, curK, 0, cmatrixInitVal);
        if (isBias_ && iterK == 0) {
            uint64_t nl0Align = CeilAlign(curN, static_cast<uint64_t>(BLOCK_CUBE));
            auto biasL0Layout = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, nl0Align);
            auto biasL0Tensor = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<
                AscendC::Te::Location::BIAS, float>(0), biasL0Layout);
            AscendC::Te::Mmad(mmadAtom.with(mmadParams), l0cTensor, al0Tensor, bl0Tensor, biasL0Tensor);
        } else {
            AscendC::Te::Mmad(mmadAtom.with(mmadParams), l0cTensor, al0Tensor, bl0Tensor);
        }
    }

    template <typename TensorC, typename CopyL0C2GMT>
    __aicore__ inline void FixL0CToGM(
        TensorC& gmC, uint64_t l0cBufId, uint64_t curIterBatchL0,
        uint64_t iter1, CopyL0C2GMT copyL0C2GM)
    {
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(static_cast<uint16_t>(l0cBufId));
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(static_cast<uint16_t>(l0cBufId));

        auto l0cOutLayout = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn,
            AscendC::Std::Int<16>>(curIterBatchL0, m_, n_);
        auto l0cOutTensor = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<
            AscendC::Te::Location::L0C, float>(HALF_L0C_SIZE * l0cBufId), l0cOutLayout);
        auto gmCSlice = gmC.Slice(AscendC::Te::MakeCoord(iter1 * mainIterBatchL0_,
            AscendC::Te::MakeCoord(0, 0)), AscendC::Te::MakeShape(curIterBatchL0, AscendC::Te::MakeShape(m_, n_)));

        AscendC::Te::FixpipeParams fixpParams(0);
        AscendC::Te::Copy(copyL0C2GM.with(fixpParams), gmCSlice, l0cOutTensor);
    }

    template <typename L1Ptn, typename CopyL12L0AT>
    __aicore__ inline void CopyAl1ToL0PerBatch(
        uint64_t l0BufId, const L0TileInfo& tileInfo, uint64_t offsetAl1,
        CopyL12L0AT copyL12L0A)
    {
        const uint64_t c0SizeA = BLOCK_BYTE_SIZE / sizeof(A_T);
        uint64_t l0aBatchStrideBytes = CeilAlign(tileInfo.curM,
            static_cast<uint64_t>(BLOCK_CUBE)) * CeilAlign(tileInfo.curK, c0SizeA) * sizeof(A_T);
        for (uint64_t bIdx = 0; bIdx < tileInfo.curAl0Cnt; ++bIdx) {
            uint64_t al1AbsBatchIdx = tileInfo.al0BatchStart + GetABatchIdx(bIdx);
            uint64_t al1ByteOff = offsetAl1 + al1AbsBatchIdx * aL1BatchStrideElems_ * sizeof(A_T);
            auto al1BatchLayout = AscendC::Te::MakeFrameLayout<L1Ptn, AscendC::Te::LayoutTraitDefault<AType>>(m_, k_);
            auto al1BatchTensor = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(al1ByteOff), al1BatchLayout);
            auto al1Slice = al1BatchTensor.Slice(
                AscendC::Te::MakeCoord(tileInfo.iterM * baseM_, tileInfo.iterK * baseK_),
                AscendC::Te::MakeShape(tileInfo.curM, tileInfo.curK));
            uint64_t al0ByteOff = HALF_L0_SIZE * l0BufId + bIdx * l0aBatchStrideBytes;
            auto al0BatchLayout = AscendC::Te::MakeFrameLayout<
                AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>(tileInfo.curM, tileInfo.curK);
            auto al0BatchTensor = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(al0ByteOff), al0BatchLayout);
            AscendC::Te::Copy(copyL12L0A, al0BatchTensor, al1Slice);
        }
    }

    template <typename L1Ptn, typename CopyL12L0BT>
    __aicore__ inline void CopyBl1ToL0PerBatch(
        uint64_t l0BufId, const L0TileInfo& tileInfo, uint64_t offsetAl1,
        CopyL12L0BT copyL12L0B)
    {
        const uint64_t c0SizeB = BLOCK_BYTE_SIZE / sizeof(B_T);
        uint64_t l0bBatchStrideBytes = CeilAlign(tileInfo.curK, c0SizeB) *
            CeilAlign(tileInfo.curN, static_cast<uint64_t>(BLOCK_CUBE)) * sizeof(B_T);
        for (uint64_t bIdx = 0; bIdx < tileInfo.curBl0Cnt; ++bIdx) {
            uint64_t bl1AbsBatchIdx = tileInfo.bl0BatchStart + GetBBatchIdx(bIdx);
            uint64_t bl1ByteOff = offsetAl1 + aL1OneBuffer_ + bl1AbsBatchIdx * bL1BatchStrideElems_ * sizeof(B_T);
            auto bl1BatchLayout = AscendC::Te::MakeFrameLayout<L1Ptn,
                AscendC::Te::LayoutTraitDefault<BType>>(k_, n_);
            auto bl1BatchTensor = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(bl1ByteOff), bl1BatchLayout);
            auto bl1Slice = bl1BatchTensor.Slice(
                AscendC::Te::MakeCoord(tileInfo.iterK * baseK_, tileInfo.iterN * baseN_),
                AscendC::Te::MakeShape(tileInfo.curK, tileInfo.curN));
            uint64_t bl0ByteOff = HALF_L0_SIZE * l0BufId + bIdx * l0bBatchStrideBytes;
            auto bl0BatchLayout = AscendC::Te::MakeFrameLayout<
                AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>(tileInfo.curK, tileInfo.curN);
            auto bl0BatchTensor = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(bl0ByteOff), bl0BatchLayout);
            AscendC::Te::Copy(copyL12L0B, bl0BatchTensor, bl1Slice);
        }
    }

    // --- Single-batch mode (iterBatchL0 == 1): 2D L0 layouts, per-batch processing ---
    template <typename TensorAl1, typename TensorBl1, typename TensorC,
              typename TensorBiasL1, typename CopyL12L0AT, typename CopyL12L0BT, typename CopyL12BTT,
              typename CopyL0C2GMT, typename MmadAtomT>
    __aicore__ inline void SingleBatchMmadLoop(
        const TensorAl1& al1Tensor, const TensorBl1& bl1Tensor,
        TensorC gmC, const TensorBiasL1& biasL1Tensor, uint64_t curIterBatchL1,
        CopyL12L0AT copyL12L0A, CopyL12L0BT copyL12L0B, CopyL12BTT copyL12BT,
        CopyL0C2GMT copyL0C2GM, MmadAtomT mmadAtom)
    {
        uint64_t ml0Cnt = CeilDiv(m_, baseM_);
        uint64_t nl0Cnt = CeilDiv(n_, baseN_);
        uint64_t kl0Cnt = CeilDiv(k_, baseK_);
        uint64_t l0cBufIdx = 0;
        for (uint64_t batchIdx = 0; batchIdx < curIterBatchL1; ++batchIdx) {
            uint64_t al1BatchIdx = GetABatchIdx(batchIdx);
            uint64_t bl1BatchIdx = GetBBatchIdx(batchIdx);
            for (uint64_t iterNl0 = 0; iterNl0 < nl0Cnt; ++iterNl0) {
                uint64_t curN = (iterNl0 == nl0Cnt - 1) ? (n_ - iterNl0 * baseN_) : baseN_;
                for (uint64_t iterMl0 = 0; iterMl0 < ml0Cnt; ++iterMl0) {
                    uint64_t curM = (iterMl0 == ml0Cnt - 1) ? (m_ - iterMl0 * baseM_) : baseM_;
                    uint64_t l0cBufId = l0cBufIdx % BUFFER_NUM;
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(static_cast<uint16_t>(l0cBufId));
                    for (uint64_t iterKl0 = 0; iterKl0 < kl0Cnt; ++iterKl0) {
                        uint64_t curK = (iterKl0 == kl0Cnt - 1) ? (k_ - iterKl0 * baseK_) : baseK_;
                        uint64_t l0BufId = iterKl0 % BUFFER_NUM;
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(l0BufId) + SIXTH_FLAG);
                        L0SingleBatchTileInfo tileInfo{curM, curN, curK,
                            iterMl0, iterNl0, iterKl0, al1BatchIdx, bl1BatchIdx, l0BufId};
                        CopyL1ToL0SingleBatch(al1Tensor, bl1Tensor, biasL1Tensor, tileInfo,
                            copyL12L0A, copyL12L0B, copyL12BT);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(l0BufId));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(l0BufId));
                        ComputeMmad(curM, curN, curK, iterKl0, mmadAtom,
                            HALF_L0_SIZE * l0BufId, HALF_L0_SIZE * l0BufId, HALF_L0C_SIZE * l0cBufId);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(l0BufId) + SIXTH_FLAG);
                    }
                    FixL0CToGMSingleBatch(gmC, l0cBufId, curM, curN, batchIdx, iterMl0, iterNl0, copyL0C2GM);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(static_cast<uint16_t>(l0cBufId));
                    l0cBufIdx++;
                }
            }
        }
    }

    template <typename TensorAl1, typename TensorBl1, typename TensorBiasL1,
              typename CopyL12L0AT, typename CopyL12L0BT, typename CopyL12BTT>
    __aicore__ inline void CopyL1ToL0SingleBatch(
        const TensorAl1& al1Tensor, const TensorBl1& bl1Tensor,
        const TensorBiasL1& biasL1Tensor, const L0SingleBatchTileInfo& tileInfo,
        CopyL12L0AT copyL12L0A, CopyL12L0BT copyL12L0B, CopyL12BTT copyL12BT)
    {
        auto al1PerBatch = al1Tensor.Slice(
            AscendC::Te::MakeCoord(tileInfo.al1BatchIdx, AscendC::Te::MakeCoord(0, 0)),
            AscendC::Te::MakeShape(1UL, AscendC::Te::MakeShape(m_, k_)));
        auto al1PerBatch2D = AscendC::Te::MakeTensor(al1PerBatch.Data(),
            AscendC::Te::RemoveBatchDim(al1PerBatch.Layout()));
        auto al1Slice = al1PerBatch2D.Slice(
            AscendC::Te::MakeCoord(tileInfo.iterM * baseM_, tileInfo.iterK * baseK_),
            AscendC::Te::MakeShape(tileInfo.curM, tileInfo.curK));
        auto al0Layout = AscendC::Te::MakeFrameLayout<
            AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>(tileInfo.curM, tileInfo.curK);
        auto al0Tensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(HALF_L0_SIZE * tileInfo.l0BufId), al0Layout);
        AscendC::Te::Copy(copyL12L0A, al0Tensor, al1Slice);

        auto bl1PerBatch = bl1Tensor.Slice(
            AscendC::Te::MakeCoord(tileInfo.bl1BatchIdx, AscendC::Te::MakeCoord(0, 0)),
            AscendC::Te::MakeShape(1UL, AscendC::Te::MakeShape(k_, n_)));
        auto bl1PerBatch2D = AscendC::Te::MakeTensor(bl1PerBatch.Data(),
            AscendC::Te::RemoveBatchDim(bl1PerBatch.Layout()));
        auto bl1Slice = bl1PerBatch2D.Slice(
            AscendC::Te::MakeCoord(tileInfo.iterK * baseK_, tileInfo.iterN * baseN_),
            AscendC::Te::MakeShape(tileInfo.curK, tileInfo.curN));
        auto bl0Layout = AscendC::Te::MakeFrameLayout<
            AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>(tileInfo.curK, tileInfo.curN);
        auto bl0Tensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(HALF_L0_SIZE * tileInfo.l0BufId), bl0Layout);
        AscendC::Te::Copy(copyL12L0B, bl0Tensor, bl1Slice);

        if (isBias_ && tileInfo.iterK == 0) {
            uint64_t nl0Align = CeilAlign(tileInfo.curN, static_cast<uint64_t>(BLOCK_CUBE));
            auto biasL0Layout = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, nl0Align);
            auto biasL0Tensor = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<
                AscendC::Te::Location::BIAS, float>(0), biasL0Layout);
            auto biasL1Slice = biasL1Tensor.Slice(
                AscendC::Te::MakeCoord(0, tileInfo.iterN * baseN_), AscendC::Te::MakeShape(1UL, tileInfo.curN));
            AscendC::Te::Copy(copyL12BT, biasL0Tensor, biasL1Slice);
        }
    }

    template <typename TensorC, typename CopyL0C2GMT>
    __aicore__ inline void FixL0CToGMSingleBatch(
        TensorC gmC, uint64_t l0cBufId, uint64_t curM, uint64_t curN,
        uint64_t batchIdx, uint64_t iterMl0, uint64_t iterNl0,
        CopyL0C2GMT copyL0C2GM)
    {
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(static_cast<uint16_t>(l0cBufId));
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(static_cast<uint16_t>(l0cBufId));
        auto l0cOutLayout = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<16>>(curM, curN);
        auto l0cOutTensor = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<
            AscendC::Te::Location::L0C, float>(HALF_L0C_SIZE * l0cBufId), l0cOutLayout);
        auto gmCPerBatch = gmC.Slice(AscendC::Te::MakeCoord(batchIdx, AscendC::Te::MakeCoord(0, 0)),
            AscendC::Te::MakeShape(1, AscendC::Te::MakeShape(m_, n_)));
        auto gmCPerBatch2D = AscendC::Te::MakeTensor(gmCPerBatch.Data(),
            AscendC::Te::RemoveBatchDim(gmCPerBatch.Layout()));
        auto gmDstTensor = gmCPerBatch2D.Slice(AscendC::Te::MakeCoord(iterMl0 * baseM_, iterNl0 * baseN_),
            AscendC::Te::MakeShape(curM, curN));
        AscendC::Te::FixpipeParams fixpParams(0);
        AscendC::Te::Copy(copyL0C2GM.with(fixpParams), gmDstTensor, l0cOutTensor);
    }

    // --- Shared helpers for both modes ---
    __aicore__ inline uint64_t GetAl1Count(uint64_t iterBatchL1) const
    {
        if constexpr (aBroadcast) {
            if (broadcastAxisA_ == LAST_BATCH_DIM) { return 1; }
        }
        return iterBatchL1;
    }

    __aicore__ inline uint64_t GetBl1Count(uint64_t iterBatchL1) const
    {
        if constexpr (bBroadcast) {
            if (broadcastAxisB_ == LAST_BATCH_DIM) { return 1; }
        }
        return iterBatchL1;
    }

    __aicore__ inline uint64_t GetAl0Count(uint64_t iterBatchL0) const
    {
        if constexpr (aBroadcast) {
            if (broadcastAxisA_ == LAST_BATCH_DIM) { return 1; }
        }
        return iterBatchL0;
    }

    __aicore__ inline uint64_t GetBl0Count(uint64_t iterBatchL0) const
    {
        if constexpr (bBroadcast) {
            if (broadcastAxisB_ == LAST_BATCH_DIM) { return 1; }
        }
        return iterBatchL0;
    }

    __aicore__ inline uint64_t GetAl0BatchStart(uint64_t iterIdx) const
    {
        if constexpr (aBroadcast) {
            if (broadcastAxisA_ == LAST_BATCH_DIM) { return 0; }
        }
        return iterIdx * mainIterBatchL0_;
    }

    __aicore__ inline uint64_t GetBl0BatchStart(uint64_t iterIdx) const
    {
        if constexpr (bBroadcast) {
            if (broadcastAxisB_ == LAST_BATCH_DIM) { return 0; }
        }
        return iterIdx * mainIterBatchL0_;
    }

    __aicore__ inline uint64_t GetABatchIdx(uint64_t batchL1Idx) const
    {
        if constexpr (aBroadcast) {
            if (broadcastAxisA_ == LAST_BATCH_DIM) { return 0; }
        }
        return batchL1Idx;
    }

    __aicore__ inline uint64_t GetBBatchIdx(uint64_t batchL1Idx) const
    {
        if constexpr (bBroadcast) {
            if (broadcastAxisB_ == LAST_BATCH_DIM) { return 0; }
        }
        return batchL1Idx;
    }

};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
