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
 * \file test_qbmm_mx_l0c_pingpong.cpp
 * \brief QBMM MX L0C ping-pong Kernel UT case
 */

#include <algorithm>
#include <cstdint>
#include "gtest/gtest.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "quant_batch_matmul.cpp"

namespace {
struct L0CPingpongCaseConfig {
    int64_t m;
    int64_t n;
    int64_t k;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t kL1;
    uint32_t blockNum;
};

size_t GetScaleKLen(int64_t k)
{
    return static_cast<size_t>((k + 63) / 64) * 2UL;
}

template <typename T>
size_t GetMxInputSize(int64_t elementCount)
{
    if constexpr (Blaze::Gemm::IsFp4<T>()) {
        return ((static_cast<size_t>(elementCount) + 1UL) / 2UL) * sizeof(T);
    } else {
        return static_cast<size_t>(elementCount) * sizeof(T);
    }
}

struct L0CPingpongBuffers {
    GM_ADDR x1{nullptr};
    GM_ADDR x2{nullptr};
    GM_ADDR pertokenScale{nullptr};
    GM_ADDR scale{nullptr};
    GM_ADDR bias{nullptr};
    GM_ADDR y{nullptr};
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

void ReleaseBuffers(L0CPingpongBuffers& buffers)
{
    Release(buffers.x1);
    Release(buffers.x2);
    Release(buffers.pertokenScale);
    Release(buffers.scale);
    Release(buffers.bias);
    Release(buffers.y);
    Release(buffers.tiling);
}

void FillBuffer(GM_ADDR addr, size_t size, uint8_t value)
{
    auto* buffer = reinterpret_cast<uint8_t*>(addr);
    std::fill_n(buffer, size, value);
}

template <typename AType, typename BType>
void AllocBuffers(L0CPingpongBuffers& buffers, const L0CPingpongCaseConfig& config)
{
    const size_t scaleKLen = GetScaleKLen(config.k);
    buffers.x1 = Alloc(GetMxInputSize<AType>(config.m * config.k));
    buffers.x2 = Alloc(GetMxInputSize<BType>(config.k * config.n));
    buffers.pertokenScale = Alloc(static_cast<size_t>(config.m) * scaleKLen * sizeof(AscendC::fp8_e8m0_t));
    buffers.scale = Alloc(scaleKLen * static_cast<size_t>(config.n) * sizeof(AscendC::fp8_e8m0_t));
    buffers.bias = Alloc(static_cast<size_t>(config.n) * sizeof(float));
    buffers.y = Alloc(static_cast<size_t>(config.m) * config.n * sizeof(half));
    buffers.tiling = Alloc(sizeof(QBMMUT::QBMML0CPingpongTilingData));
}

bool CheckBuffers(const L0CPingpongBuffers& buffers)
{
    return buffers.x1 != nullptr && buffers.x2 != nullptr && buffers.pertokenScale != nullptr &&
        buffers.scale != nullptr && buffers.bias != nullptr && buffers.y != nullptr && buffers.tiling != nullptr;
}

template <typename AType, typename BType>
void InitBuffers(L0CPingpongBuffers& buffers, const L0CPingpongCaseConfig& config)
{
    const size_t scaleKLen = GetScaleKLen(config.k);
    FillBuffer(buffers.x1, GetMxInputSize<AType>(config.m * config.k), 0U);
    FillBuffer(buffers.x2, GetMxInputSize<BType>(config.k * config.n), 0U);
    FillBuffer(buffers.pertokenScale, static_cast<size_t>(config.m) * scaleKLen * sizeof(AscendC::fp8_e8m0_t), 0x7fU);
    FillBuffer(buffers.scale, scaleKLen * static_cast<size_t>(config.n) * sizeof(AscendC::fp8_e8m0_t), 0x7fU);
    FillBuffer(buffers.bias, static_cast<size_t>(config.n) * sizeof(float), 0U);
    FillBuffer(buffers.y, static_cast<size_t>(config.m) * config.n * sizeof(half), 0U);
}

void SetTilingData(L0CPingpongBuffers& buffers, const L0CPingpongCaseConfig& config)
{
    auto* tilingData = reinterpret_cast<QBMMUT::QBMML0CPingpongTilingData*>(buffers.tiling);
    *tilingData = QBMMUT::QBMML0CPingpongTilingData{
        config.m, config.n, config.k, 1, config.baseM, config.baseN, config.baseK,
        config.kL1, config.kL1, 2, 2};
}

template <typename AType, typename BType>
bool RunKernel(const L0CPingpongBuffers& buffers, uint32_t blockNum)
{
    auto kernelFunc = qbmm_kernel_entry<OP_TYPE_QBMM_MX_L0C_PINGPONG, AType, BType, half, float>;
    return KERNEL_RUN_KF(
        kernelFunc, blockNum, buffers.x1, buffers.x2, buffers.pertokenScale, buffers.scale, buffers.bias,
        buffers.y, buffers.tiling);
}
} // namespace

TEST(QBMML0CPingpong, TestMxfp8L0CPingpong)
{
    using MxType = fp8_e4m3fn_t;
    const L0CPingpongCaseConfig config{64, 128, 128, 64, 128, 64, 64, 1};
    L0CPingpongBuffers buffers;

    AllocBuffers<MxType, MxType>(buffers, config);
    ASSERT_TRUE(CheckBuffers(buffers));
    InitBuffers<MxType, MxType>(buffers, config);
    SetTilingData(buffers, config);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    EXPECT_TRUE((RunKernel<MxType, MxType>(buffers, config.blockNum)))
        << "QBMM MX L0C ping-pong kernel execution failed";
    ReleaseBuffers(buffers);
}

TEST(QBMML0CPingpong, TestMxfp4L0CPingpongSplitN)
{
    using MxType = fp4x2_e2m1_t;
    const L0CPingpongCaseConfig config{128, 256, 128, 128, 256, 64, 64, 1};
    L0CPingpongBuffers buffers;

    AllocBuffers<MxType, MxType>(buffers, config);
    ASSERT_TRUE(CheckBuffers(buffers));
    InitBuffers<MxType, MxType>(buffers, config);
    SetTilingData(buffers, config);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    EXPECT_TRUE((RunKernel<MxType, MxType>(buffers, config.blockNum)))
        << "QBMM MX L0C ping-pong kernel execution failed";
    ReleaseBuffers(buffers);
}
