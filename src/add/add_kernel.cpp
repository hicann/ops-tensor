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
 * @file add_kernel.cpp
 * @brief Add算子Kernel实现
 */

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "arch35/add_struct.h"

using namespace AscendC;
using namespace AddOp;

constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t UB_BYTENUM_PER_BLOCK = 32;
constexpr uint32_t ELEMENTS_PER_BLOCK = UB_BYTENUM_PER_BLOCK / BYTENUM_PER_FLOAT32;  // 8

template <typename T>
class AddKernel {
public:
    __aicore__ inline AddKernel() {}
    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        GM_ADDR tilingGm, TPipe* pipeIn);
    __aicore__ inline void Process();
    __aicore__ inline void ParseTilingData(GM_ADDR tilingGm);

private:
    TPipe* pipe;
    static constexpr uint16_t BUFFER_NUM = 2;

    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> x2Gm_;
    GlobalTensor<T> yGm_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX2_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;

    // Tiling 数据成员
    int64_t elemNum_;
    int64_t usedCoreNum_;
    int64_t ubFormer_;
    int64_t elementsPerCore_;
    int64_t blockFormer_;
    int64_t blockLoopCnt_;
    int64_t blockTail_;
    int64_t tailCoreElements_;
    int64_t tailCoreBlockLoopCnt_;
    int64_t tailCoreBlockTail_;

    uint64_t blockOffset_;
    uint32_t ubLength_;
    uint32_t ubLoopCnt_;
    uint32_t tailBlockLength_;
};

template <typename T>
__aicore__ inline void AddKernel<T>::ParseTilingData(GM_ADDR tilingGm)
{
    auto tilingAddr = reinterpret_cast<__gm__ uint8_t *>(tilingGm);

    // 按照 AddTilingData 结构体的布局，通过偏移量获取各字段
    // 结构体布局（所有字段都是 int64_t，每个占 8 字节）：
    // elemNum, usedCoreNum, ubFormer,
    // elementsPerCore, blockFormer, blockLoopCnt, blockTail,
    // tailCoreElements, tailCoreBlockLoopCnt, tailCoreBlockTail

    size_t offset = 0;
    elemNum_ = *(__gm__ int64_t *)(tilingAddr + offset); offset += sizeof(int64_t);
    usedCoreNum_ = *(__gm__ int64_t *)(tilingAddr + offset); offset += sizeof(int64_t);
    ubFormer_ = *(__gm__ int64_t *)(tilingAddr + offset); offset += sizeof(int64_t);
    elementsPerCore_ = *(__gm__ int64_t *)(tilingAddr + offset); offset += sizeof(int64_t);
    blockFormer_ = *(__gm__ int64_t *)(tilingAddr + offset); offset += sizeof(int64_t);
    blockLoopCnt_ = *(__gm__ int64_t *)(tilingAddr + offset); offset += sizeof(int64_t);
    blockTail_ = *(__gm__ int64_t *)(tilingAddr + offset); offset += sizeof(int64_t);
    tailCoreElements_ = *(__gm__ int64_t *)(tilingAddr + offset); offset += sizeof(int64_t);
    tailCoreBlockLoopCnt_ = *(__gm__ int64_t *)(tilingAddr + offset); offset += sizeof(int64_t);
    tailCoreBlockTail_ = *(__gm__ int64_t *)(tilingAddr + offset);
}

template <typename T>
__aicore__ inline void AddKernel<T>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
    GM_ADDR tilingGm, TPipe* pipeIn)
{
    pipe = pipeIn;

    // 解析 tiling 数据
    ParseTilingData(tilingGm);

    auto blockIdx = GetBlockIdx();
    bool isTailCore = (blockIdx == usedCoreNum_ - 1);

    uint64_t coreElements;
    if (!isTailCore) {
        coreElements = elementsPerCore_;
        blockOffset_ = blockIdx * coreElements;
    } else {
        coreElements = tailCoreElements_;
        blockOffset_ = (usedCoreNum_ - 1) * elementsPerCore_;
    }

    x1Gm_.SetGlobalBuffer((__gm__ T*)x1 + blockOffset_);
    x2Gm_.SetGlobalBuffer((__gm__ T*)x2 + blockOffset_);
    yGm_.SetGlobalBuffer((__gm__ T*)y + blockOffset_);

    ubLength_ = blockFormer_;

    if (!isTailCore) {
        ubLoopCnt_ = blockLoopCnt_;
        tailBlockLength_ = blockTail_;
    } else {
        ubLoopCnt_ = tailCoreBlockLoopCnt_;
        tailBlockLength_ = tailCoreBlockTail_;
    }

    pipe->InitBuffer(inQueueX1_, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe->InitBuffer(inQueueX2_, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe->InitBuffer(outQueueY_, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void AddKernel<T>::Process()
{
    for (uint32_t i = 0; i < ubLoopCnt_; ++i) {
        uint32_t currentOffset = i * ubLength_;
        uint32_t currentLength = (i == ubLoopCnt_ - 1) ? tailBlockLength_ : ubLength_;

        // 统一使用 DataCopyPad，对齐时 paddingNum = 0
        uint32_t paddingNum = (currentLength % ELEMENTS_PER_BLOCK == 0) ?
                              0 : ELEMENTS_PER_BLOCK - (currentLength % ELEMENTS_PER_BLOCK);

        DataCopyExtParams copyParams{1, currentLength * BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        LocalTensor<T> x1Local = inQueueX1_.AllocTensor<T>();
        LocalTensor<T> x2Local = inQueueX2_.AllocTensor<T>();
        DataCopyPad(x1Local, x1Gm_[currentOffset], copyParams, padParams);
        DataCopyPad(x2Local, x2Gm_[currentOffset], copyParams, padParams);
        inQueueX1_.EnQue<T>(x1Local);
        inQueueX2_.EnQue<T>(x2Local);

        // 取出 x1/x2 进行计算
        LocalTensor<T> x1Que = inQueueX1_.DeQue<T>();
        LocalTensor<T> x2Que = inQueueX2_.DeQue<T>();

        // 分配 yLocal 并计算（使用对齐后的长度）
        LocalTensor<T> yLocal = outQueueY_.AllocTensor<T>();
        uint32_t alignedLength = currentLength + paddingNum;
        Add(yLocal, x1Que, x2Que, alignedLength);
        outQueueY_.EnQue<T>(yLocal);

        // 释放输入队列
        inQueueX1_.FreeTensor(x1Que);
        inQueueX2_.FreeTensor(x2Que);

        // 拷贝输出回 GM（使用 DataCopyPad）
        LocalTensor<T> yQue = outQueueY_.DeQue<T>();
        DataCopyPad(yGm_[currentOffset], yQue, copyParams);
        outQueueY_.FreeTensor(yQue);
    }
}

extern "C" __global__ __aicore__ void add(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                          GM_ADDR workspace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    TPipe pipe;
    AddKernel<float> op;
    op.Init(x1, x2, y, tilingGm, &pipe);
    op.Process();
}

void add_kernel_do(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                   GM_ADDR workspace, GM_ADDR tilingGm,
                   uint32_t numBlocks, void *stream)
{
    add<<<numBlocks, nullptr, stream>>>(x1, x2, y, workspace, tilingGm);
}
