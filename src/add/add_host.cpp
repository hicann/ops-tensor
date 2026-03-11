/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file add_host.cpp
 * @brief Add算子Host端实现 - 简化版本，只支持float类型
 */

#include "acl/acl.h"
#include "arch35/add_struct.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include <iostream>
#include <cstring>

#define GM_ADDR uint8_t*

// 外部内核函数声明
extern void add_kernel_do(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                          GM_ADDR workspace, GM_ADDR tilingGm,
                          uint32_t numBlocks, void *stream);

// Tiling 调试打印函数
static void PrintTilingDebugInfo(uint32_t maxCoreNum, uint64_t ubSize, uint32_t ubFormSize,
                                  const AddOp::AddTilingData& tilingData, uint32_t numBlocks)
{
    std::cout << "\n========== Tiling 调试信息 ==========\n"
              << "系统参数: maxCoreNum=" << maxCoreNum << ", ubSize=" << ubSize
              << " bytes, ubFormSize=" << ubFormSize << "\n"
              << "输入信息: 总元素数 elemNum=" << tilingData.elemNum << "\n"
              << "普通核: elementsPerCore=" << tilingData.elementsPerCore
              << ", blockFormer=" << tilingData.blockFormer
              << ", blockLoopCnt=" << tilingData.blockLoopCnt
              << ", blockTail=" << tilingData.blockTail << "\n"
              << "尾核: tailCoreElements=" << tilingData.tailCoreElements
              << ", tailCoreBlockLoopCnt=" << tilingData.tailCoreBlockLoopCnt
              << ", tailCoreBlockTail=" << tilingData.tailCoreBlockTail << "\n"
              << "最终: elemNum=" << tilingData.elemNum
              << ", usedCoreNum=" << tilingData.usedCoreNum
              << ", ubFormer=" << tilingData.ubFormer
              << ", numBlocks=" << numBlocks << "\n"
              << "======================================\n" << std::endl;
}

/**
 * @brief 计算 tiling 数据
 */
static int CalculateTilingData(int64_t n, AddOp::AddTilingData& tilingData, uint32_t& numBlocks)
{
    // 初始化为0
    memset(&tilingData, 0, sizeof(AddOp::AddTilingData));

    tilingData.elemNum = n;

    // 从系统获取核数和 UB 大小
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t maxCoreNum = ascendcPlatform->GetCoreNumAiv();

    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // UB 大小用于计算
    // Kernel 使用：3个队列 × 每个队列2块 = 6块 buffer
    // ubFormer = UB总大小 / 6 / sizeof(float)
    constexpr uint32_t NUM_QUEUES = 3;      // inQueueX1, inQueueX2, outQueueY
    constexpr uint32_t BUFFER_NUM = 2;       // 每个队列的 buffer 数量
    uint32_t ubFormSize = ubSize / (NUM_QUEUES * BUFFER_NUM * sizeof(float));

    // 单核最小处理元素数
    constexpr uint32_t MIN_ELEMENTS_PER_CORE = 8;
    // 32字节对齐（32 / sizeof(float) = 8个元素）
    constexpr uint32_t ALIGN_ELEMENTS = 8;

    // 计算单核处理的最大元素个数：总量 ceil 总核数
    uint64_t maxElementsPerCore = (n + maxCoreNum - 1) / maxCoreNum;

    uint32_t usedCoreNum;
    uint64_t elementsPerCore;

    if (maxElementsPerCore < MIN_ELEMENTS_PER_CORE) {
        // 如果小于8，按单核处理8个数重新折算实际使用核数
        elementsPerCore = MIN_ELEMENTS_PER_CORE;
        usedCoreNum = (n + MIN_ELEMENTS_PER_CORE - 1) / MIN_ELEMENTS_PER_CORE;
    } else {
        // 如果大于等于8，按最大元素个数计算实际需要的核数
        elementsPerCore = maxElementsPerCore;
        usedCoreNum = (n + maxElementsPerCore - 1) / maxElementsPerCore;
    }

    // 计算 blockFormer：按 32 字节对齐，且不超过 elementsPerCore 和 ubFormSize
    uint32_t maxBlockByElements = elementsPerCore;
    uint32_t maxBlockByUB = ubFormSize;

    // 按 8 个元素对齐
    uint32_t blockFormer = std::min(maxBlockByElements, maxBlockByUB);
    blockFormer = (blockFormer + ALIGN_ELEMENTS - 1) / ALIGN_ELEMENTS * ALIGN_ELEMENTS;

    // 普通核的 block 切分信息
    tilingData.elementsPerCore = elementsPerCore;
    tilingData.blockFormer = blockFormer;

    // 计算普通核的 block 循环次数
    tilingData.blockLoopCnt = (elementsPerCore + blockFormer - 1) / blockFormer;

    // 计算普通核的尾 block 元素数
    tilingData.blockTail = elementsPerCore % blockFormer;
    if (tilingData.blockTail == 0) {
        tilingData.blockTail = blockFormer;
    }

    // 计算尾核信息（最后一个核）
    int64_t totalElementsForNormalCores = (int64_t)(usedCoreNum - 1) * elementsPerCore;
    int64_t remainingElements = n - totalElementsForNormalCores;

    // 尾核处理的元素个数
    tilingData.tailCoreElements = remainingElements;
    tilingData.tailCoreBlockLoopCnt = (remainingElements + blockFormer - 1) / blockFormer;
    tilingData.tailCoreBlockTail = remainingElements % blockFormer;
    if (tilingData.tailCoreBlockTail == 0) {
        tilingData.tailCoreBlockTail = blockFormer;
    }

    // 设置其他信息
    tilingData.usedCoreNum = usedCoreNum;
    tilingData.ubFormer = ubFormSize;

    // 返回实际使用的核数
    numBlocks = usedCoreNum;

    // 打印调试信息
    // PrintTilingDebugInfo(maxCoreNum, ubSize, ubFormSize, tilingData, numBlocks);

    return 0;
}

/**
 * @brief 张量加法操作 - 对外接口（只支持float类型）
 */
extern "C" aclError acltensorAdd(float* x1, float* x2, float* y, int64_t n, void* stream)
{
    if (x1 == nullptr || x2 == nullptr || y == nullptr) {
        std::cerr << "Error: null pointer input" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    if (n <= 0) {
        std::cerr << "Error: invalid element count " << n << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    // 计算 tiling 数据
    AddOp::AddTilingData tilingData;
    uint32_t numBlocks = 0;
    int ret = CalculateTilingData(n, tilingData, numBlocks);
    if (ret != 0) {
        std::cerr << "Error: failed to calculate tiling data" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    size_t totalByteSize = n * sizeof(float);

    // 将 float* 指针转换为 uint8_t*（参照 scopy 的做法）
    uint8_t *x1Host = reinterpret_cast<uint8_t *>(x1);
    uint8_t *x2Host = reinterpret_cast<uint8_t *>(x2);
    uint8_t *yHost = reinterpret_cast<uint8_t *>(y);

    // 分配设备内存
    uint8_t *x1Device = nullptr;
    uint8_t *x2Device = nullptr;
    uint8_t *yDevice = nullptr;
    uint8_t *tilingDevice = nullptr;

    aclrtMalloc((void**)&x1Device, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&x2Device, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&yDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&tilingDevice, sizeof(AddOp::AddTilingData), ACL_MEM_MALLOC_HUGE_FIRST);

    // 拷贝数据到设备（使用 x1Host, x2Host）
    aclrtMemcpy(x1Device, totalByteSize, x1Host, totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(x2Device, totalByteSize, x2Host, totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(tilingDevice, sizeof(AddOp::AddTilingData), &tilingData, sizeof(AddOp::AddTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    // 调用内核
    add_kernel_do(x1Device, x2Device, yDevice, nullptr, tilingDevice, numBlocks, stream);

    // 同步流
    aclrtSynchronizeStream(stream);

    // 拷贝结果回主机（拷贝到 yHost，不是 y！）
    aclrtMemcpy(yHost, totalByteSize, yDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // 释放设备内存
    aclrtFree(x1Device);
    aclrtFree(x2Device);
    aclrtFree(yDevice);
    aclrtFree(tilingDevice);

    return ACL_SUCCESS;
}
