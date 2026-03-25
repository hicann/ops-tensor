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

#include <cstring>
#include <iostream>
#include "securec.h"

#include "arch35/add_struct.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"

#include "cann_ops_tensor.h"
#include "elementwise/elementwise.hpp"

#define GM_ADDR uint8_t*

// 内核函数声明（由 add_kernel.cpp 实现）
extern void add_kernel_do(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                          GM_ADDR workspace, GM_ADDR tilingGm,
                          uint32_t numBlocks, void *stream);

namespace acltensor {

/**
 * @brief 计算 Add 算子的 tiling 数据
 */
/**
 * @brief 获取平台信息
 * @param maxCoreNum 输出：AI 核心数量
 * @param ubFormSize 输出：UB 可用元素数量
 * @return 0 成功，-1 失败
 */
static int GetPlatformInfo(uint32_t& maxCoreNum, uint32_t& ubFormSize)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    maxCoreNum = ascendcPlatform->GetCoreNumAiv();
    if (maxCoreNum == 0) {
        std::cerr << "[ERROR] GetPlatformInfo: maxCoreNum is 0" << std::endl;
        return -1;
    }

    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) {
        std::cerr << "[ERROR] GetPlatformInfo: ubSize is 0" << std::endl;
        return -1;
    }

    constexpr uint32_t NUM_QUEUES = 3;     // 流水线队列数量（加载/计算/存储）
    constexpr uint32_t BUFFER_NUM = 2;      // 双缓冲，允许计算与数据搬运并行
    ubFormSize = ubSize / (NUM_QUEUES * BUFFER_NUM * sizeof(float));

    return 0;
}

static int CalculateAddTilingData(int64_t n, AddOp::AddTilingData& tilingData, uint32_t& numBlocks)
{
    int ret = memset_s(&tilingData, sizeof(AddOp::AddTilingData), 0, sizeof(AddOp::AddTilingData));
    if (ret != EOK) {
        std::cerr << "[ERROR] CalculateAddTilingData: memset_s failed with code " << ret << std::endl;
        return -1;
    }
    tilingData.elemNum = n;

    uint32_t maxCoreNum = 0;
    uint32_t ubFormSize = 0;
    if (GetPlatformInfo(maxCoreNum, ubFormSize) != 0) {
        return -1;
    }

    constexpr uint32_t MIN_ELEMENTS_PER_CORE = 8;   // 每核最小处理元素数，避免核心浪费
    constexpr uint32_t ALIGN_ELEMENTS = 8;          // 32字节对齐（8个float），满足Vector指令要求

    uint64_t maxElementsPerCore = (n + maxCoreNum - 1) / maxCoreNum;

    uint32_t usedCoreNum;
    uint64_t elementsPerCore = (maxElementsPerCore < MIN_ELEMENTS_PER_CORE) ? MIN_ELEMENTS_PER_CORE : maxElementsPerCore;
    usedCoreNum = (n + elementsPerCore - 1) / elementsPerCore;
    usedCoreNum = (usedCoreNum > maxCoreNum) ? maxCoreNum : usedCoreNum;

    uint32_t maxBlockByElements = elementsPerCore;
    uint32_t maxBlockByUB = ubFormSize;
    uint32_t blockFormer = std::min(maxBlockByElements, maxBlockByUB);
    blockFormer = (blockFormer + ALIGN_ELEMENTS - 1) / ALIGN_ELEMENTS * ALIGN_ELEMENTS;
    if (blockFormer == 0) {
        std::cerr << "[ERROR] CalculateAddTilingData: blockFormer is 0 after alignment" << std::endl;
        return -1;
    }

    tilingData.elementsPerCore = elementsPerCore;
    tilingData.blockFormer = blockFormer;
    tilingData.blockLoopCnt = (elementsPerCore + blockFormer - 1) / blockFormer;
    tilingData.blockTail = elementsPerCore % blockFormer;
    if (tilingData.blockTail == 0) {
        tilingData.blockTail = blockFormer;
    }

    int64_t totalElementsForNormalCores = static_cast<int64_t>(usedCoreNum - 1) * elementsPerCore;
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
 * @brief Add FP32 执行函数（内部实现，保持原有逻辑）
 *        注意：参数顺序与原签名一致
 */
static acltensorStatus_t AddF32Execute_Impl(
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
    aclrtMemcpy(tilingDevice, sizeof(AddOp::AddTilingData), &tilingData, sizeof(AddOp::AddTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    // 调用内核（异步执行）
    add_kernel_do(
        (GM_ADDR)A, (GM_ADDR)C, (GM_ADDR)D,
        nullptr, (GM_ADDR)tilingDevice,
        numBlocks, stream);

    // 同步等待内核完成，确保设备端不再访问 tiling 内存
    aclError syncRet = aclrtSynchronizeStream(stream);
    if (syncRet != ACL_SUCCESS) {
        aclrtFree(tilingDevice);
        return ACLTENSOR_STATUS_INTERNAL_ERROR;
    }

    // 安全释放 tiling 内存
    aclrtFree(tilingDevice);

    return ACLTENSOR_STATUS_SUCCESS;
}

/**
 * @brief Add FP32 执行函数（新签名，支持可扩展参数）
 */
static acltensorStatus_t AddF32Execute(const ElementwiseArgs& args)
{
    // 从 ElementwiseArgs 中提取参数
    const void* A = args.bufferA;
    const void* C = args.bufferC;
    void* D = args.bufferD;

    // 计算总元素数
    int64_t elemNum = 1;
    if (args.numModesD > 0 && args.lengthsD != nullptr)
    {
        for (uint32_t i = 0; i < args.numModesD; ++i)
        {
            elemNum *= args.lengthsD[i];
        }
    }

    const void* alpha = args.alpha;
    const void* gamma = args.gamma;
    aclrtStream stream = args.stream;

    // 调用原有实现
    return AddF32Execute_Impl(A, C, D, elemNum, alpha, gamma, stream);
}

/**
 * @brief 注册 Add FP32 解决方案（使用自动注册宏）
 *        支持 Binary 通用解决方案（任意维度）
 */
REGISTER_ELEMENTWISE_SOLUTION(ADD, R_32F, 0, BINARY, AddF32Execute)

} // namespace acltensor
