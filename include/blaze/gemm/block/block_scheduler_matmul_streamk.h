/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file block_scheduler_matmul_streamk.h
 * \brief
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <class ProblemShape_>
class BlockSchedulerMatmulStreamK {
public:
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    struct Params {
        int64_t usedCoreNum{0};
        int64_t baseM{0};
        int64_t baseN{0};
        int64_t baseK{0};
        int64_t singleCoreK{0};
        int64_t kL1{0};
        uint8_t isHf32{0};
        uint32_t l2CacheMode = L2_CACHE_DEFAULT;
    };

public:
    __aicore__ inline BlockSchedulerMatmulStreamK(const ProblemShape& shape, const Params& params)
    {
        usedCoreNum_ = params.usedCoreNum;
        if (usedCoreNum_ <= 0) {
            return;
        }
        m_ = AscendC::Te::Get<MNK_M>(shape);
        n_ = AscendC::Te::Get<MNK_N>(shape);
        k_ = AscendC::Te::Get<MNK_K>(shape);
        batch_ = AscendC::Std::max(AscendC::Te::Get<MNK_B>(shape), 1L);

        mL1_ = params.baseM; // size of m in L1 & L0 & singlecore, per core use L1 once in stream k
        nL1_ = params.baseN; // size of n in L1 & L0 & singlecore, per core use L1 once in stream k
        skSingleCoreK_ = params.singleCoreK; // size of k in singlecore

        mTileNum_ = CeilDiv(m_, mL1_);
        nTileNum_ = CeilDiv(n_, nL1_);
        skKTileNum_ = CeilDiv(k_, skSingleCoreK_);
        int64_t tailMNTileNum = (mTileNum_ * nTileNum_) % usedCoreNum_; // tail mCnt * nCnt num of SK
        // core num of DP (m*n) + tail core num of SK (m*n*k)
        tileNum_ = (mTileNum_ * nTileNum_ - tailMNTileNum) + tailMNTileNum * skKTileNum_;
        totalMNTileNumInDP_ = mTileNum_ * nTileNum_ - tailMNTileNum;
    }

    __aicore__ inline int64_t GetTileNum()
    {
        return tileNum_ * batch_;
    }

    __aicore__ inline int64_t GetBlockNum(ProblemShape shape)
    {
        int64_t tilingBlockNum = 0;
        if (tileNum_ * batch_ < AscendC::GetBlockNum()) {
            tilingBlockNum = tileNum_ * batch_;
        } else {
            tilingBlockNum = AscendC::GetBlockNum();
        }
        return tilingBlockNum;
    }

    __aicore__ inline AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t> GetMNKTileNum()
    {
        return {mTileNum_, nTileNum_, skKTileNum_, 1};
    }

    __aicore__ inline BlockShape GetBlockShape(int64_t tileIdx)
    {
        UpdateMNTileIdx(tileIdx);
        int64_t tailL1M = m_ - (mTileNum_ - 1) * mL1_;
        int64_t tailL1N = n_ - (nTileNum_ - 1) * nL1_;
        int64_t tailSingleCoreK = k_ - (curKTileNum_ - 1) * skSingleCoreK_;
        int64_t blkM = (mTileIdx_ == (mTileNum_ - 1)) ? tailL1M : mL1_;
        int64_t blkN = (nTileIdx_ == (nTileNum_ - 1)) ? tailL1N : nL1_;
        int64_t blkK = (kTileIdx_ == (curKTileNum_ - 1)) ? tailSingleCoreK : skSingleCoreK_;
        return {blkM, blkN, blkK, 0};
    }

    __aicore__ inline BlockCoord GetBlockCoord(int64_t tileIdx)
    {
        UpdateMNTileIdx(tileIdx);
        return {mTileIdx_, nTileIdx_, kTileIdx_, 0};
    }

    __aicore__ inline int64_t GetCurKSingleCore(int64_t tileIdx)
    {
        return (CheckIsSkScene(tileIdx) ? skSingleCoreK_ : k_);
    }

    __aicore__ inline bool CheckIsSkScene(int64_t tileIdx)
    {
        return CeilDiv((tileIdx + 1), usedCoreNum_) == CeilDiv(tileNum_, usedCoreNum_); // true is sk, false is dp
    }

private:
    __aicore__ inline void UpdateMNTileIdx(int64_t tileIdx)
    {
        // judge now in dp loop (kTileNum = 1) or in sk loop
        curKTileNum_ = CheckIsSkScene(tileIdx) ? skKTileNum_ : 1;
        int64_t mnIdxInCurLoop = 0;
        if (CheckIsSkScene(tileIdx)) { // SK scene
            kTileIdx_ = (tileIdx % usedCoreNum_) % curKTileNum_;
            mnIdxInCurLoop = (tileIdx % usedCoreNum_) / curKTileNum_ + totalMNTileNumInDP_;
        } else { // DP scene
            kTileIdx_ = 0;
            mnIdxInCurLoop = tileIdx / curKTileNum_;
        }
        int64_t mainWindow = AscendC::Std::min(WINDOW_LEN, mTileNum_);
        int64_t mainRow = mTileNum_ / mainWindow - 1UL;
        int64_t tailWindow = mTileNum_ - mainRow * mainWindow;
        int64_t rowIdx = mnIdxInCurLoop / nTileNum_ / mainWindow;
        if (rowIdx < mainRow) {
            mTileIdx_ = rowIdx * mainWindow + mnIdxInCurLoop % mainWindow;
            nTileIdx_ = (mnIdxInCurLoop / mainWindow) % nTileNum_;
        } else {
            rowIdx = mainRow;
            int64_t tailIndex = mnIdxInCurLoop - mainRow * mainWindow * nTileNum_;
            mTileIdx_ = mainRow * mainWindow + tailIndex % tailWindow;
            nTileIdx_ = (tailIndex / tailWindow) % nTileNum_;
        }
        // mod 2 means even row, need reverse scan
        if (rowIdx % 2 != 0UL) {
            nTileIdx_ = nTileNum_ - 1UL - nTileIdx_;
        }
    }

private:
    int64_t usedCoreNum_{0};
    int64_t mTileNum_{0};
    int64_t nTileNum_{0};
    int64_t skKTileNum_{0};
    int64_t tileNum_{1};
    int64_t totalMNTileNumInDP_{0};

    int64_t batch_{0};
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};

    int64_t mTileIdx_{1};
    int64_t nTileIdx_{1};
    int64_t kTileIdx_{1};
    int64_t curKTileNum_{1};

    int64_t mL1_{0};
    int64_t nL1_{0};
    int64_t skSingleCoreK_{0};
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze