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
 * \file test_qbmm_streamk.cpp
 * \brief QBMM MX StreamK Kernel UT case
 */

#include <algorithm>
#include <cstdint>
#include "gtest/gtest.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "qbmm_streamk.h"

namespace {
constexpr size_t WORKSPACE_TILE_SIZE = 256UL * 256UL * sizeof(float);
constexpr size_t WORKSPACE_OVERHEAD = 20UL * 1024UL * 1024UL;

struct StreamKCaseConfig {
    int64_t m;
    int64_t n;
    int64_t k;
    uint32_t blockNum;
};

size_t GetScaleKLen(int64_t k)
{
    return static_cast<size_t>((k + 63) / 64) * 2UL;
}

struct StreamKBuffers {
    GM_ADDR x1{nullptr};
    GM_ADDR x2{nullptr};
    GM_ADDR pertokenScale{nullptr};
    GM_ADDR scale{nullptr};
    GM_ADDR bias{nullptr};
    GM_ADDR y{nullptr};
    GM_ADDR workspace{nullptr};
    GM_ADDR tiling{nullptr};
};

GM_ADDR Alloc(size_t size)
{
    return static_cast<GM_ADDR>(AscendC::GmAlloc(size));
}

void Release(GM_ADDR& addr)
{
    if (addr != nullptr) {
        AscendC::GmFree(reinterpret_cast<void*>(addr));
        addr = nullptr;
    }
}

void ReleaseBuffers(StreamKBuffers& buffers)
{
    Release(buffers.x1);
    Release(buffers.x2);
    Release(buffers.pertokenScale);
    Release(buffers.scale);
    Release(buffers.bias);
    Release(buffers.y);
    Release(buffers.workspace);
    Release(buffers.tiling);
}

void FillBuffer(GM_ADDR addr, size_t size, uint8_t value)
{
    auto* buffer = reinterpret_cast<uint8_t*>(addr);
    std::fill_n(buffer, size, value);
}

void AllocBuffers(StreamKBuffers& buffers, const StreamKCaseConfig& config)
{
    size_t scaleKLen = GetScaleKLen(config.k);
    buffers.x1 = Alloc(config.m * config.k * sizeof(fp8_e4m3fn_t));
    buffers.x2 = Alloc(config.k * config.n * sizeof(fp8_e5m2_t));
    buffers.pertokenScale = Alloc(config.m * scaleKLen * sizeof(AscendC::fp8_e8m0_t));
    buffers.scale = Alloc(scaleKLen * config.n * sizeof(AscendC::fp8_e8m0_t));
    buffers.bias = Alloc(config.n * sizeof(float));
    buffers.y = Alloc(config.m * config.n * sizeof(half));
    buffers.workspace = Alloc(config.blockNum * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD);
    buffers.tiling = Alloc(sizeof(QBMMUT::QBMMStreamKTilingData));
}

bool CheckBuffers(const StreamKBuffers& buffers)
{
    return buffers.x1 != nullptr && buffers.x2 != nullptr && buffers.pertokenScale != nullptr &&
        buffers.scale != nullptr && buffers.bias != nullptr && buffers.y != nullptr &&
        buffers.workspace != nullptr && buffers.tiling != nullptr;
}

void InitBuffers(StreamKBuffers& buffers, const StreamKCaseConfig& config)
{
    size_t scaleKLen = GetScaleKLen(config.k);
    FillBuffer(buffers.x1, config.m * config.k * sizeof(fp8_e4m3fn_t), 0U);
    FillBuffer(buffers.x2, config.k * config.n * sizeof(fp8_e5m2_t), 0U);
    FillBuffer(buffers.pertokenScale, config.m * scaleKLen * sizeof(AscendC::fp8_e8m0_t), 0x7fU);
    FillBuffer(buffers.scale, scaleKLen * config.n * sizeof(AscendC::fp8_e8m0_t), 0x7fU);
    FillBuffer(buffers.bias, config.n * sizeof(float), 0U);
    FillBuffer(buffers.y, config.m * config.n * sizeof(half), 0U);
    FillBuffer(buffers.workspace, config.blockNum * WORKSPACE_TILE_SIZE + WORKSPACE_OVERHEAD, 0U);
}

void SetTilingData(StreamKBuffers& buffers, const StreamKCaseConfig& config)
{
    auto* tilingData = reinterpret_cast<QBMMUT::QBMMStreamKTilingData*>(buffers.tiling);
    *tilingData = QBMMUT::QBMMStreamKTilingData{
        config.m, config.n, config.k, 1, config.blockNum, 16, 16, 64, 64, 64, 64, 1};
}

bool RunKernel(const StreamKBuffers& buffers, uint32_t blockNum)
{
    auto kernelFunc =
        QBMMUT::qbmm_streamk_kernel_entry<fp8_e4m3fn_t, fp8_e5m2_t, half, float>;
    return KERNEL_RUN_KF(kernelFunc, blockNum, buffers.x1, buffers.x2, buffers.pertokenScale, buffers.scale,
                         buffers.bias, buffers.y, buffers.workspace, buffers.tiling);
}
} // namespace

TEST(QBMMStreamK, TestMxfp8StreamK)
{
    const StreamKCaseConfig config{16, 16, 128, 2};
    StreamKBuffers buffers;

    AllocBuffers(buffers, config);
    ASSERT_TRUE(CheckBuffers(buffers));
    InitBuffers(buffers, config);
    SetTilingData(buffers, config);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    EXPECT_TRUE(RunKernel(buffers, config.blockNum)) << "QBMM MX StreamK kernel execution failed";
    ReleaseBuffers(buffers);
}
