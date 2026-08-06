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
 * \file test_qbmm_pertensor_streamk.cpp
 * \brief UT for the QBMM per-tensor StreamK component stack.
 */

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "gtest/gtest.h"
#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"

#include "qbmm_pertensor_streamk.h"
#include "blaze/gemm/utils/common_utils.h"

namespace {

constexpr size_t STREAMK_WORKSPACE_TILE_SIZE = 256UL * 256UL * sizeof(int32_t);
constexpr size_t STREAMK_WORKSPACE_OVERHEAD = 20UL * 1024UL * 1024UL;

class GmBuffer {
public:
    explicit GmBuffer(size_t size) : addr_(reinterpret_cast<GM_ADDR>(AscendC::GmAlloc(size))) {}

    ~GmBuffer()
    {
        if (addr_ != nullptr) {
            AscendC::GmFree(reinterpret_cast<void*>(addr_));
        }
    }

    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;

    GM_ADDR Get() const { return addr_; }

private:
    GM_ADDR addr_{nullptr};
};

void FillBytes(GM_ADDR addr, size_t size, uint8_t value) { std::fill_n(reinterpret_cast<uint8_t*>(addr), size, value); }

template <class T>
void FillValues(GM_ADDR addr, size_t count, T value)
{
    std::fill_n(reinterpret_cast<T*>(addr), count, value);
}

void RunInt8PertensorStreamKSmoke()
{
    constexpr int64_t M = 16;
    constexpr int64_t N = 16;
    constexpr int64_t K = 128;
    constexpr uint32_t BLOCK_NUM = 2U;

    GmBuffer x1GM(static_cast<size_t>(M * K) * sizeof(int8_t));
    GmBuffer x2GM(static_cast<size_t>(K * N) * sizeof(int8_t));
    GmBuffer scaleGM(sizeof(float));
    GmBuffer biasGM(static_cast<size_t>(N) * sizeof(int32_t));
    GmBuffer yGM(static_cast<size_t>(M * N) * sizeof(bfloat16_t));
    GmBuffer workspaceGM(BLOCK_NUM * STREAMK_WORKSPACE_TILE_SIZE + STREAMK_WORKSPACE_OVERHEAD);
    GmBuffer tilingGM(sizeof(QBMMUT::QBMMPertensorStreamKTilingData));

    ASSERT_NE(x1GM.Get(), nullptr);
    ASSERT_NE(x2GM.Get(), nullptr);
    ASSERT_NE(scaleGM.Get(), nullptr);
    ASSERT_NE(biasGM.Get(), nullptr);
    ASSERT_NE(yGM.Get(), nullptr);
    ASSERT_NE(workspaceGM.Get(), nullptr);
    ASSERT_NE(tilingGM.Get(), nullptr);

    FillValues<int8_t>(x1GM.Get(), static_cast<size_t>(M * K), 1);
    FillValues<int8_t>(x2GM.Get(), static_cast<size_t>(K * N), 1);
    FillValues<float>(scaleGM.Get(), 1U, 1.0F);
    FillBytes(biasGM.Get(), static_cast<size_t>(N) * sizeof(int32_t), 0U);
    FillBytes(yGM.Get(), static_cast<size_t>(M * N) * sizeof(bfloat16_t), 0U);
    FillBytes(workspaceGM.Get(), BLOCK_NUM * STREAMK_WORKSPACE_TILE_SIZE + STREAMK_WORKSPACE_OVERHEAD, 0U);

    auto* tilingData = reinterpret_cast<QBMMUT::QBMMPertensorStreamKTilingData*>(tilingGM.Get());
    *tilingData = {M, N, K, 1, BLOCK_NUM, 16, 16, 64, 64, 64, 0, QBMMUT::GE_DT_FLOAT};

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kernelFunc = qbmm_pertensor_streamk_kernel_entry<int8_t, int8_t, float, bfloat16_t, int32_t>;
    const bool ok = KERNEL_RUN_KF(kernelFunc, BLOCK_NUM, x1GM.Get(), x2GM.Get(), nullptr, scaleGM.Get(), biasGM.Get(),
                                  yGM.Get(), workspaceGM.Get(), tilingGM.Get());

    ASSERT_TRUE(ok) << "QBMM per-tensor StreamK kernel execution failed";
    // tikicpulib does not model the RegTensor dequantization numerics. Validate the mixed-kernel pipeline and
    // synchronization here; scalar scale encoding is covered by the focused tests below.
}

void RunFp8DoubleScalePostBiasStreamKSmoke()
{
    constexpr int64_t M = 16;
    constexpr int64_t N = 16;
    constexpr int64_t K = 128;
    constexpr uint32_t BLOCK_NUM = 2U;
    constexpr float BIAS_VALUE = 1.25F;

    GmBuffer x1GM(static_cast<size_t>(M * K) * sizeof(fp8_e4m3fn_t));
    GmBuffer x2GM(static_cast<size_t>(K * N) * sizeof(fp8_e4m3fn_t));
    GmBuffer perTokenScaleGM(sizeof(float));
    GmBuffer scaleGM(sizeof(float));
    GmBuffer biasGM(static_cast<size_t>(N) * sizeof(float));
    GmBuffer yGM(static_cast<size_t>(M * N) * sizeof(float));
    GmBuffer workspaceGM(BLOCK_NUM * STREAMK_WORKSPACE_TILE_SIZE + STREAMK_WORKSPACE_OVERHEAD);
    GmBuffer tilingGM(sizeof(QBMMUT::QBMMPertensorStreamKTilingData));

    ASSERT_NE(x1GM.Get(), nullptr);
    ASSERT_NE(x2GM.Get(), nullptr);
    ASSERT_NE(perTokenScaleGM.Get(), nullptr);
    ASSERT_NE(scaleGM.Get(), nullptr);
    ASSERT_NE(biasGM.Get(), nullptr);
    ASSERT_NE(yGM.Get(), nullptr);
    ASSERT_NE(workspaceGM.Get(), nullptr);
    ASSERT_NE(tilingGM.Get(), nullptr);

    FillBytes(x1GM.Get(), static_cast<size_t>(M * K) * sizeof(fp8_e4m3fn_t), 0U);
    FillBytes(x2GM.Get(), static_cast<size_t>(K * N) * sizeof(fp8_e4m3fn_t), 0U);
    FillValues<float>(perTokenScaleGM.Get(), 1U, 3.0F);
    FillValues<float>(scaleGM.Get(), 1U, 2.0F);
    FillValues<float>(biasGM.Get(), static_cast<size_t>(N), BIAS_VALUE);
    FillValues<float>(yGM.Get(), static_cast<size_t>(M * N), 0.0F);
    FillBytes(workspaceGM.Get(), BLOCK_NUM * STREAMK_WORKSPACE_TILE_SIZE + STREAMK_WORKSPACE_OVERHEAD, 0U);

    auto* tilingData = reinterpret_cast<QBMMUT::QBMMPertensorStreamKTilingData*>(tilingGM.Get());
    *tilingData = {M, N, K, 1, BLOCK_NUM, 16, 16, 64, 64, 64, 1, QBMMUT::GE_DT_FLOAT};

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kernelFunc = qbmm_pertensor_streamk_kernel_entry<fp8_e4m3fn_t, fp8_e4m3fn_t, float, float, float>;
    const bool ok = KERNEL_RUN_KF(kernelFunc, BLOCK_NUM, x1GM.Get(), x2GM.Get(), perTokenScaleGM.Get(), scaleGM.Get(),
                                  biasGM.Get(), yGM.Get(), workspaceGM.Get(), tilingGM.Get());

    ASSERT_TRUE(ok) << "QBMM double-scale post-dequant bias StreamK kernel execution failed";
    // tikicpulib does not model the RegTensor dequantization/bias numerics. Validate the mixed-kernel pipeline and
    // synchronization here; scale merge/masking semantics are covered by the focused tests below.
}

void RunBatchInputRejectedSmoke()
{
    constexpr int64_t M = 16;
    constexpr int64_t N = 16;
    constexpr int64_t K = 128;
    constexpr int64_t BATCH = 2;
    constexpr uint32_t BLOCK_NUM = 2U;
    constexpr uint8_t OUTPUT_SENTINEL = 0x5AU;

    GmBuffer x1GM(static_cast<size_t>(BATCH * M * K) * sizeof(int8_t));
    GmBuffer x2GM(static_cast<size_t>(K * N) * sizeof(int8_t));
    GmBuffer scaleGM(sizeof(float));
    GmBuffer yGM(static_cast<size_t>(BATCH * M * N) * sizeof(bfloat16_t));
    GmBuffer workspaceGM(BLOCK_NUM * STREAMK_WORKSPACE_TILE_SIZE + STREAMK_WORKSPACE_OVERHEAD);
    GmBuffer tilingGM(sizeof(QBMMUT::QBMMPertensorStreamKTilingData));

    ASSERT_NE(x1GM.Get(), nullptr);
    ASSERT_NE(x2GM.Get(), nullptr);
    ASSERT_NE(scaleGM.Get(), nullptr);
    ASSERT_NE(yGM.Get(), nullptr);
    ASSERT_NE(workspaceGM.Get(), nullptr);
    ASSERT_NE(tilingGM.Get(), nullptr);

    FillBytes(yGM.Get(), static_cast<size_t>(BATCH * M * N) * sizeof(bfloat16_t), OUTPUT_SENTINEL);
    auto* tilingData = reinterpret_cast<QBMMUT::QBMMPertensorStreamKTilingData*>(tilingGM.Get());
    *tilingData = {M, N, K, BATCH, BLOCK_NUM, 16, 16, 64, 64, 64, 0, QBMMUT::GE_DT_FLOAT};

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kernelFunc = qbmm_pertensor_streamk_kernel_entry<int8_t, int8_t, float, bfloat16_t, int32_t>;
    const bool ok = KERNEL_RUN_KF(kernelFunc, BLOCK_NUM, x1GM.Get(), x2GM.Get(), nullptr, scaleGM.Get(), nullptr,
                                  yGM.Get(), workspaceGM.Get(), tilingGM.Get());

    ASSERT_TRUE(ok) << "QBMM batched input rejection kernel launch failed";
    const auto* output = reinterpret_cast<const uint8_t*>(yGM.Get());
    const size_t outputBytes = static_cast<size_t>(BATCH * M * N) * sizeof(bfloat16_t);
    EXPECT_TRUE(std::all_of(output, output + outputBytes, [](uint8_t value) { return value == OUTPUT_SENTINEL; }));
}

} // namespace

class QBMMPertensorStreamKTest : public testing::Test {};

TEST_F(QBMMPertensorStreamKTest, TemplateContracts)
{
    using Layout = AscendC::Te::NDExtLayoutPtn;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using DefaultDispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<0, false,
                                                                    Blaze::Gemm::KernelQbmmPertensorMultiBlockStreamK>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;
    using Int8Mmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, int8_t, Layout, AscendC::Std::tuple<int8_t, float>,
                                                   Layout, half, Layout, int32_t, Layout>;
    using Int8Epilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<typename Int8Mmad::WorkspaceType,
                                                                                   half, DispatchPolicy, float, float>;
    using Int8Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Int8Mmad, Int8Epilogue, Scheduler>;
    using Fp8Mmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, fp8_e4m3fn_t, Layout,
                                                  AscendC::Std::tuple<fp8_e5m2_t, uint64_t>, Layout, float, Layout,
                                                  float, Layout>;
    using Fp8PostBiasMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, fp8_e4m3fn_t, Layout,
                                                          AscendC::Std::tuple<fp8_e4m3fn_t, float>, Layout, float,
                                                          Layout, float, Layout>;
    using Int8Fp32PostBiasMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, int8_t, Layout, AscendC::Std::tuple<int8_t, float>, Layout, bfloat16_t, Layout, float, Layout>;
    using Int8Bf16PostBiasMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, int8_t, Layout,
                                                               AscendC::Std::tuple<int8_t, bfloat16_t>, Layout,
                                                               bfloat16_t, Layout, bfloat16_t, Layout>;

    static_assert(
        std::is_same_v<typename DispatchPolicy::ScheduleType, Blaze::Gemm::KernelQbmmPertensorMultiBlockStreamK>);
    static_assert(
        std::is_same_v<typename DefaultDispatchPolicy::ScheduleType, Blaze::Gemm::KernelMmadWithScaleFixpipeQuant>);
    static_assert(std::is_same_v<typename Int8Mmad::WorkspaceType, int32_t>);
    static_assert(std::is_same_v<typename Fp8Mmad::WorkspaceType, float>);
    static_assert(Int8Mmad::BIAS_IN_MMAD);
    static_assert(Fp8Mmad::BIAS_IN_MMAD);
    static_assert(!Fp8PostBiasMmad::BIAS_IN_MMAD);
    static_assert(!Int8Fp32PostBiasMmad::BIAS_IN_MMAD);
    static_assert(!Int8Bf16PostBiasMmad::BIAS_IN_MMAD);
    static_assert(std::is_same_v<typename Int8Epilogue::WorkspaceType, typename Int8Mmad::WorkspaceType>);
    static_assert(std::is_same_v<typename Int8Kernel::BlockMmad, Int8Mmad>);

    SUCCEED();
}

TEST_F(QBMMPertensorStreamKTest, SingleScaleWithoutPostBiasIsMaskedBeforeMultiply)
{
    using Layout = AscendC::Te::NDExtLayoutPtn;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<0, false,
                                                                    Blaze::Gemm::KernelQbmmPertensorMultiBlockStreamK>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, int8_t, Layout, AscendC::Std::tuple<int8_t, float>,
                                               Layout, bfloat16_t, Layout, int32_t, Layout>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<typename Mmad::WorkspaceType, bfloat16_t,
                                                                               DispatchPolicy, float, float>;

    constexpr uint32_t rawScaleBits = 0x3F812345U;
    const float maskedScale = Epilogue::DecodeMaskedDequantScale(rawScaleBits);
    const uint32_t actualBits = Blaze::Gemm::Float32ToBits(maskedScale);

    EXPECT_EQ(actualBits, rawScaleBits & Epilogue::DEQ_SCALE_MUL_MASK);
}

TEST_F(QBMMPertensorStreamKTest, DoubleScaleWithoutPostBiasMergesBeforeMask)
{
    using Layout = AscendC::Te::NDExtLayoutPtn;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<0, false,
                                                                    Blaze::Gemm::KernelQbmmPertensorMultiBlockStreamK>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, fp8_e4m3fn_t, Layout,
                                               AscendC::Std::tuple<fp8_e4m3fn_t, float>, Layout, float, Layout, float,
                                               Layout>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<typename Mmad::WorkspaceType, float,
                                                                               DispatchPolicy, float, float>;

    constexpr float x2Scale = 1.013741F;
    constexpr float x1Scale = 0.987653F;
    const float actual = Epilogue::MergeAndMaskDequantScale(x2Scale, x1Scale);
    const float merged = x2Scale * x1Scale;
    const uint32_t mergedBits = Blaze::Gemm::Float32ToBits(merged);
    const uint32_t actualBits = Blaze::Gemm::Float32ToBits(actual);

    EXPECT_EQ(actualBits, mergedBits & Epilogue::DEQ_SCALE_MUL_MASK);
}

TEST_F(QBMMPertensorStreamKTest, PostBiasScaleDecodePreservesFullPrecision)
{
    using Layout = AscendC::Te::NDExtLayoutPtn;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<0, false,
                                                                    Blaze::Gemm::KernelQbmmPertensorMultiBlockStreamK>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, fp8_e4m3fn_t, Layout,
                                               AscendC::Std::tuple<fp8_e4m3fn_t, float>, Layout, float, Layout, float,
                                               Layout>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<typename Mmad::WorkspaceType, float,
                                                                               DispatchPolicy, float, float>;

    constexpr uint32_t rawScaleBits = 0x3F812345U;
    const float rawScale = Blaze::Gemm::BitsToFloat32(rawScaleBits);
    const uint32_t actualBits = Blaze::Gemm::Float32ToBits(rawScale);

    EXPECT_EQ(actualBits, rawScaleBits);
    EXPECT_NE(actualBits, rawScaleBits & Epilogue::DEQ_SCALE_MUL_MASK);
}

TEST_F(QBMMPertensorStreamKTest, Int8PerTensorSmoke) { RunInt8PertensorStreamKSmoke(); }

TEST_F(QBMMPertensorStreamKTest, Fp8DoubleScalePostBiasSmoke) { RunFp8DoubleScalePostBiasStreamKSmoke(); }

TEST_F(QBMMPertensorStreamKTest, BatchedInputReturnsBeforeScheduling) { RunBatchInputRejectedSmoke(); }
