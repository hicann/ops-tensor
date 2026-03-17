/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_solution.cpp
 * \brief Add 算子解决方案注册
 */

#include "cann_ops_tensor.h"
#include "lib/elementwise/elementwise.hpp"
#include "arch35/add_struct.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstring>

#define GM_ADDR uint8_t*

// 外部内核函数声明
extern void add_kernel_do(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                          GM_ADDR workspace, GM_ADDR tilingGm,
                          uint32_t numBlocks, void *stream);

namespace acltensor {

/**
 * @brief 计算 Add 算子的 tiling 数据
 */
static int CalculateAddTilingData(int64_t n, AddOp::AddTilingData& tilingData, uint32_t& numBlocks)
{
    memset(&tilingData, 0, sizeof(AddOp::AddTilingData));
    tilingData.elemNum = n;

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t maxCoreNum = ascendcPlatform->GetCoreNumAiv();

    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    constexpr uint32_t NUM_QUEUES = 3;
    constexpr uint32_t BUFFER_NUM = 2;
    uint32_t ubFormSize = ubSize / (NUM_QUEUES * BUFFER_NUM * sizeof(float));

    constexpr uint32_t MIN_ELEMENTS_PER_CORE = 8;
    constexpr uint32_t ALIGN_ELEMENTS = 8;

    uint64_t maxElementsPerCore = (n + maxCoreNum - 1) / maxCoreNum;

    uint32_t usedCoreNum;
    uint64_t elementsPerCore;

    if (maxElementsPerCore < MIN_ELEMENTS_PER_CORE) {
        elementsPerCore = MIN_ELEMENTS_PER_CORE;
        usedCoreNum = (n + MIN_ELEMENTS_PER_CORE - 1) / MIN_ELEMENTS_PER_CORE;
    } else {
        elementsPerCore = maxElementsPerCore;
        usedCoreNum = (n + maxElementsPerCore - 1) / maxElementsPerCore;
    }

    if (usedCoreNum > maxCoreNum) {
        usedCoreNum = maxCoreNum;
    }

    uint32_t maxBlockByElements = elementsPerCore;
    uint32_t maxBlockByUB = ubFormSize;
    uint32_t blockFormer = std::min(maxBlockByElements, maxBlockByUB);
    blockFormer = (blockFormer + ALIGN_ELEMENTS - 1) / ALIGN_ELEMENTS * ALIGN_ELEMENTS;

    tilingData.elementsPerCore = elementsPerCore;
    tilingData.blockFormer = blockFormer;
    tilingData.blockLoopCnt = (elementsPerCore + blockFormer - 1) / blockFormer;
    tilingData.blockTail = elementsPerCore % blockFormer;
    if (tilingData.blockTail == 0) {
        tilingData.blockTail = blockFormer;
    }

    int64_t totalElementsForNormalCores = (int64_t)(usedCoreNum - 1) * elementsPerCore;
    int64_t remainingElements = n - totalElementsForNormalCores;

    tilingData.tailCoreElements = remainingElements;
    tilingData.tailCoreBlockLoopCnt = (remainingElements + blockFormer - 1) / blockFormer;
    tilingData.tailCoreBlockTail = remainingElements % blockFormer;
    if (tilingData.tailCoreBlockTail == 0) {
        tilingData.tailCoreBlockTail = blockFormer;
    }

    tilingData.usedCoreNum = usedCoreNum;
    tilingData.ubFormer = ubFormSize;
    numBlocks = usedCoreNum;

    return 0;
}

/**
 * @brief Add FP32 执行函数
 *        注意：参数顺序与 ElementwiseBinaryExecuteFunc 签名一致
 *        (A, C, D, elemNum, alpha, gamma, stream)
 */
static acltensorStatus_t AddF32Execute(
    const void* A, const void* C, void* D,
    int64_t elemNum,
    const void* alpha, const void* gamma,
    aclrtStream stream)
{
    // 阶段一：忽略 alpha 和 gamma，直接执行 A + C = D
    (void)alpha;
    (void)gamma;

    if (A == nullptr || C == nullptr || D == nullptr || elemNum <= 0) {
        return ACLTENSOR_STATUS_INVALID_VALUE;
    }

    // 计算 tiling 数据
    AddOp::AddTilingData tilingData;
    uint32_t numBlocks = 0;
    int ret = CalculateAddTilingData(elemNum, tilingData, numBlocks);
    if (ret != 0) {
        return ACLTENSOR_STATUS_INTERNAL_ERROR;
    }

    // 分配 tiling 设备内存
    void* tilingDevice = nullptr;
    aclError aclRet = aclrtMalloc(&tilingDevice, sizeof(AddOp::AddTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        return ACLTENSOR_STATUS_ALLOC_FAILED;
    }

    // 拷贝 tiling 数据到设备
    aclrtMemcpy(tilingDevice, &tilingData, sizeof(AddOp::AddTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    // 调用内核（假设输入已经是设备指针）
    add_kernel_do(
        (GM_ADDR)A, (GM_ADDR)C, (GM_ADDR)D,
        nullptr, (GM_ADDR)tilingDevice,
        numBlocks, stream);

    // 释放 tiling 内存（注意：异步操作完成后才能释放，这里简化处理）
    // 实际实现需要使用事件或其他同步机制
    aclrtFree(tilingDevice);

    return ACLTENSOR_STATUS_SUCCESS;
}

/**
 * @brief 注册 Add FP32 解决方案到注册表
 * @param registry 解决方案注册表引用
 */
void RegisterAddF32Solution(ElementwiseSolutionRegistry& registry)
{
    // numModes=0 表示通用解决方案，支持任意维度
    SolutionUid uid{ACLTENSOR_OP_ADD, ACLTENSOR_R_32F, 0};

    auto solution = std::make_unique<ElementwiseBinarySolution>(uid, AddF32Execute);
    registry.registerSolution(std::move(solution));
}

} // namespace acltensor
