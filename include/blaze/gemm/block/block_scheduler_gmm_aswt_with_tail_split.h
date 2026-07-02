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
 * \file block_scheduler_gmm_aswt_with_tail_split.h
 * \brief
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {
constexpr int64_t GMM_WINDOW_LEN = 4; // Sliding window length along the M tile axis.
constexpr int64_t INNER_AXIS_MIN_SPLIT_VAL = 128;

class BlockSchedulerGmmSwatWithTailSplit {
public:
    using TupleShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;
    struct Params {
        int32_t baseM{0};
        int32_t baseN{0};
    };

    __aicore__ inline BlockSchedulerGmmSwatWithTailSplit(const Params& params)
        : baseM_(params.baseM), baseN_(params.baseN), blockNum_(AscendC::GetBlockNum()),
          blockIdx_(AscendC::GetBlockIdx() / AscendC::GetTaskRation())
    {
    }

    __aicore__ inline BlockSchedulerGmmSwatWithTailSplit(int32_t baseM, int32_t baseN, int32_t baseK)
        : baseM_(baseM), baseN_(baseN), blockNum_(AscendC::GetBlockNum()),
          blockIdx_(AscendC::GetBlockIdx() / AscendC::GetTaskRation())
    {
        (void)baseK;
    }

    __aicore__ inline void UpdateNextProblem(const TupleShape& problemShape)
    {
        // Scheduler maps only M/N tiles; K is kept for shape consistency and does not drive split-K here.
        k_ = AscendC::Te::Get<MNK_K>(problemShape);
        if (m_ != AscendC::Te::Get<MNK_M>(problemShape) || n_ != AscendC::Te::Get<MNK_N>(problemShape)) {
            m_ = AscendC::Te::Get<MNK_M>(problemShape);
            n_ = AscendC::Te::Get<MNK_N>(problemShape);
            mCnt_ = CeilDiv(m_, static_cast<int64_t>(baseM_));
            nCnt_ = CeilDiv(n_, static_cast<int64_t>(baseN_));
            mBaseTail_ = m_ - (mCnt_ - 1) * baseM_;
            nBaseTail_ = n_ - (nCnt_ - 1) * baseN_;
            totalCnt_ = mCnt_ * nCnt_;
            mainMWindow_ = GMM_WINDOW_LEN < mCnt_ ? GMM_WINDOW_LEN : mCnt_;
            mainRow_ = mCnt_ / mainMWindow_ - 1;
            tailWindow_ = mCnt_ - mainMWindow_ * mainRow_;
        }
        roundIdx_ = 0;
        round_ = CeilDiv(totalCnt_, static_cast<int64_t>(blockNum_));
        // Continue from the next physical core after the previous group to balance grouped problems.
        startBlockIdx_ = endBlockIdx_ == blockNum_ - 1 ? 0 : (endBlockIdx_ + 1);
        // The last physical core that owns a tile before optional tail splitting.
        endBlockIdx_ = (totalCnt_ + startBlockIdx_ - 1) % blockNum_;
        // Adjust the per-core round count when this group wraps around the physical core range.
        if (startBlockIdx_ > endBlockIdx_ && (blockIdx_ > endBlockIdx_ && blockIdx_ < startBlockIdx_)) {
            round_ -= 1;
        } else if (startBlockIdx_ <= endBlockIdx_ && (blockIdx_ > endBlockIdx_ || blockIdx_ < startBlockIdx_)) {
            round_ -= 1;
        }
    }

    __aicore__ inline void UpdateBaseM(uint32_t baseM)
    {
        baseM_ = baseM;
    }

    __aicore__ inline void SetTailAlign(uint32_t mTailAlign, uint32_t nTailAlign)
    {
        mTailAlign_ = mTailAlign;
        nTailAlign_ = nTailAlign;
    }

    __aicore__ inline int64_t GetTailTileCnt()
    {
        return Min(static_cast<int64_t>(endBlockIdx_ + 1), totalCnt_);
    }

    __aicore__ inline void UpdateTailTile(uint32_t mTailCnt, uint32_t nTailCnt)
    {
        mTailCnt_ = mTailCnt;
        nTailCnt_ = nTailCnt;
        tailCnt_ = mTailCnt_ * nTailCnt_;
        int64_t tailOriCnt = GetTailTileCnt();
        int64_t newEndBlockIdx = endBlockIdx_ + tailOriCnt * (tailCnt_ - 1);
        if (blockIdx_ > endBlockIdx_ && blockIdx_ <= newEndBlockIdx) {
            round_ += 1;
        }
        if (blockIdx_ > newEndBlockIdx) {
            mTailCnt_ = 1;
            nTailCnt_ = 1;
            tailCnt_ = 1;
            tailBlockBase_ = 0;
        } else if (tailCnt_ > 1) {
            // Base physical block of the original tail tile window before M/N tail splitting.
            tailBlockBase_ = endBlockIdx_ + 1 - tailOriCnt;
        } else {
            tailBlockBase_ = 0;
        }
        endBlockIdx_ = newEndBlockIdx;
    }

    __aicore__ inline void UpdateTailTile()
    {
        // Use the idle cores after normal tiles to split the final M/N tail tile.
        int64_t tailTileCnt = GetTailTileCnt();
        if (tailTileCnt == 0) {
            return;
        }
        int64_t remainTile = (AscendC::GetBlockNum() - endBlockIdx_ - 1) / tailTileCnt + 1;
        if (remainTile <= 1) {
            return;
        }

        // Tail split granularity is bounded by cube alignment and the caller's tail alignment.
        int64_t mMin = Min(static_cast<int64_t>(BLOCK_CUBE), mTailAlign_);
        int64_t nMin = Min(static_cast<int64_t>(BLOCK_CUBE), nTailAlign_);

        int64_t mTile = Min(CeilDiv(static_cast<int64_t>(mBaseTail_), mMin), remainTile);
        int64_t nTile = Min(CeilDiv(static_cast<int64_t>(nBaseTail_), nMin), remainTile);
        while (mTile * nTile > remainTile) {
            if (mTile >= nTile) {
                mTile -= 1;
            } else {
                nTile -= 1;
            }
        }
        UpdateTailTile(mTile, nTile);
    }

    __aicore__ inline bool GetTileIdx(BlockCoord& blockCoord)
    {
        if (round_ == 0 || roundIdx_ > round_ - 1) {
            return false;
        }
        int64_t newBlockIdx = static_cast<int64_t>(blockIdx_);
        if (roundIdx_ == round_ - 1 && tailCnt_ > 1) {
            newBlockIdx = (tailBlockBase_ + ((newBlockIdx - tailBlockBase_) / tailCnt_) * tailCnt_) / tailCnt_;
        }
        int64_t index = newBlockIdx + roundIdx_ * blockNum_;
        if (blockIdx_ < startBlockIdx_) {
            index += blockNum_ - startBlockIdx_;
        } else if (tailCnt_ > 1 && endBlockIdx_ + 1 >= tailCnt_ * totalCnt_) {
            index -= (tailBlockBase_ + ((startBlockIdx_ - tailBlockBase_) / tailCnt_) * tailCnt_) / tailCnt_;
        } else {
            index -= startBlockIdx_;
        }

        // SWAT order walks a short M window first, then N; odd rows reverse N for locality.
        int64_t rowIdx = index / nCnt_ / mainMWindow_;
        if (rowIdx < mainRow_) {
            AscendC::Std::get<MNK_M>(blockCoord) = rowIdx * mainMWindow_ + index % mainMWindow_;
            AscendC::Std::get<MNK_N>(blockCoord) = (index / mainMWindow_) % nCnt_;
        } else {
            rowIdx = mainRow_;
            int64_t tailIndex = index - mainRow_ * mainMWindow_ * nCnt_;
            AscendC::Std::get<MNK_M>(blockCoord) = mainRow_ * mainMWindow_ + tailIndex % tailWindow_;
            AscendC::Std::get<MNK_N>(blockCoord) = (tailIndex / tailWindow_) % nCnt_;
        }
        if (rowIdx & 1) {
            AscendC::Std::get<MNK_N>(blockCoord) = nCnt_ - 1 - AscendC::Te::Get<MNK_N>(blockCoord);
        }
        roundIdx_++;
        return true;
    }

    __aicore__ inline TupleShape GetBlockShape(const BlockCoord& blockCoord)
    {
        int64_t singleCoreM = AscendC::Te::Get<MNK_M>(blockCoord) != (mCnt_ - 1) ? baseM_ : mBaseTail_;
        int64_t singleCoreN = AscendC::Te::Get<MNK_N>(blockCoord) != (nCnt_ - 1) ? baseN_ : nBaseTail_;
        // Return offsets only for the final split tail tile; bias and split-K are handled by the kernel/block.
        if (tailCnt_ == 1 || roundIdx_ < round_) {
            return {singleCoreM, singleCoreN, 0, 0};
        }

        int64_t singleCoreMSplit = (singleCoreM + mTailCnt_ - 1) / mTailCnt_;
        int64_t singleCoreNSplit = (singleCoreN + nTailCnt_ - 1) / nTailCnt_;
        singleCoreMSplit = CeilAlign(singleCoreMSplit, mTailAlign_);
        singleCoreNSplit = CeilAlign(singleCoreNSplit, nTailAlign_);

        int64_t mSplitIdx = (blockIdx_ % tailCnt_) % mTailCnt_;
        int64_t nSplitIdx = (blockIdx_ % tailCnt_) / mTailCnt_;
        int64_t mSplitAddrOffset = mSplitIdx * singleCoreMSplit;
        int64_t nSplitAddrOffset = nSplitIdx * singleCoreNSplit;
        if (mSplitAddrOffset >= singleCoreM || nSplitAddrOffset >= singleCoreN) {
            return {0, 0, 0, 0};
        }
        singleCoreM = Min(singleCoreM - mSplitAddrOffset, singleCoreMSplit);
        singleCoreN = Min(singleCoreN - nSplitAddrOffset, singleCoreNSplit);
        return {singleCoreM, singleCoreN, mSplitAddrOffset, nSplitAddrOffset};
    }

    __aicore__ inline int64_t GetEndBlockIdx() const
    {
        return endBlockIdx_;
    }

private:
    int64_t mCnt_{0};
    int64_t nCnt_{0};
    int64_t totalCnt_{0};
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t mTailCnt_{1};
    int64_t nTailCnt_{1};
    int64_t mTailAlign_{1};
    int64_t nTailAlign_{1};
    int64_t tailCnt_{1};
    int64_t tailBlockBase_{0};
    int64_t mainMWindow_{0};
    int64_t tailWindow_{0};
    int64_t mainRow_{0};
    int64_t round_{0};
    int64_t roundIdx_{0};
    int32_t baseM_{0};
    int32_t baseN_{0};
    int32_t mBaseTail_{0};
    int32_t nBaseTail_{0};
    uint32_t blockNum_{0};
    uint32_t blockIdx_{0};
    uint32_t startBlockIdx_{0};
    uint32_t endBlockIdx_{blockNum_ - 1};
};

using BlockSchedulerGmmAswtWithTailSplit = BlockSchedulerGmmSwatWithTailSplit;
} // namespace Block
} // namespace Gemm
} // namespace Blaze
