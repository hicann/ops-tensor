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
 * \file test_quant_batch_matmul.cpp
 * \brief QBMMV3 Kernel UT测试用例
 */

#include <fstream>
#include <string>
#include "gtest/gtest.h"
#include "blaze_kernel_stub.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "kernel_operator.h"

#include "quant_batch_matmul.cpp"

class QBMMV3Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
    }

    static void TearDownTestCase()
    {
        std::string cleanCmd = std::string("cd ") + UT_KERNEL_SRC_DIR + "/quant_batch_matmul/qbmm_data && rm -rf *.bin";
        system(cleanCmd.c_str());
    }

    void SetUp() override
    {
        x1GM = nullptr;
        x2GM = nullptr;
        pertokenScaleGM = nullptr;
        scaleGM = nullptr;
        biasGM = nullptr;
        yGM = nullptr;
        tilingGM = nullptr;
    }

    void TearDown() override
    {
        if (x1GM)
            AscendC::GmFree((void*)x1GM);
        if (x2GM)
            AscendC::GmFree((void*)x2GM);
        if (pertokenScaleGM)
            AscendC::GmFree((void*)pertokenScaleGM);
        if (scaleGM)
            AscendC::GmFree((void*)scaleGM);
        if (biasGM)
            AscendC::GmFree((void*)biasGM);
        if (yGM)
            AscendC::GmFree((void*)yGM);
        if (tilingGM)
            AscendC::GmFree((void*)tilingGM);
    }

    GM_ADDR x1GM;
    GM_ADDR x2GM;
    GM_ADDR pertokenScaleGM;
    GM_ADDR scaleGM;
    GM_ADDR biasGM;
    GM_ADDR yGM;
    GM_ADDR tilingGM;
};

namespace {

// QuantMode 编码（与 BlockEpilogueDequant::QuantMode 一致）
constexpr uint32_t QM_DEFAULT = 0;
constexpr uint32_t QM_PERTENSOR = 1;
constexpr uint32_t QM_PERCHANNEL = 2;
constexpr uint32_t QM_PERTOKEN = 4;

// ge::DataType 编码（bias 运行时 dtype）
constexpr uint32_t GE_DT_FLOAT = 0;
constexpr uint32_t GE_DT_FLOAT16 = 1;
constexpr uint32_t GE_DT_BF16 = 27;

struct MixCaseCfg {
    int64_t M;
    int64_t N;
    int64_t K;
    uint32_t blockNum;
    uint32_t x1QuantMode;   // 激活量化模式
    uint32_t x2QuantMode;   // 权重量化模式
    bool isBias;
    uint32_t biasDtype;     // ge::DataType 编码，需与 bias.bin 元素类型一致
    size_t biasElemSize;    // bias 单元素字节数（与 biasDtype 匹配）
    size_t outElemSize;     // 输出单元素字节数（half/bf16=2, float=4）
    const char* genArgs;    // 传给 gen_data.py 的量化模式参数
};

// 填充单 tile MxNxK 场景的 tiling（其余字段沿用 fixpipe 用例默认）。
void FillMixTiling(QBMMV3TilingData* t, const MixCaseCfg& cfg)
{
    t->m = cfg.M;
    t->n = cfg.N;
    t->k = cfg.K;
    t->b = 1;

    t->aGmAddr = 0;
    t->bGmAddr = 0;
    t->cGmAddr = 0;
    t->biasGmAddr = 0;
    t->scaleAGmAddr = 0;
    t->scaleBGmAddr = 0;

    t->baseM = cfg.M;
    t->baseN = cfg.N;
    t->mTailTile = 1;
    t->nTailTile = 1;
    t->mBaseTailSplitCnt = 1;
    t->nBaseTailSplitCnt = 1;
    t->mTailMain = 0;
    t->nTailMain = 0;

    t->batchA1 = 1;
    t->batchA2 = 1;
    t->batchA3 = 1;
    t->batchA4 = 1;
    t->batchB1 = 1;
    t->batchB2 = 1;
    t->batchB3 = 1;
    t->batchB4 = 1;
    t->batchC1 = 1;
    t->batchC2 = 1;
    t->batchC3 = 1;
    t->batchC4 = 1;
    t->biasThreeDim = 0;
    t->x1QuantMode = cfg.x1QuantMode;
    t->x2QuantMode = cfg.x2QuantMode;
    t->kAL1 = static_cast<uint32_t>(cfg.K);
    t->kBL1 = static_cast<uint32_t>(cfg.K);
    t->nBufferNum = 1;
    t->baseM_qbmm = static_cast<uint32_t>(cfg.M);
    t->baseN_qbmm = static_cast<uint32_t>(cfg.N);
    t->baseK_qbmm = static_cast<uint32_t>(cfg.K);
    t->isBias = cfg.isBias ? 1 : 0;
    t->dbL0C = 1;
    t->biasDtype = cfg.biasDtype;
}

// 清理旧 .bin 并调用 gen_data.py 生成输入数据；genArgs 为量化模式等附加参数（fixpipe 用例传 ""）。
// 供 RunMixSmoke 与 fixpipe 用例复用，避免数据生成命令块重复。
void RunGenData(const std::string& dataDir, int64_t M, int64_t N, int64_t K, const std::string& genArgs)
{
    std::string cleanCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
    std::string genDataCmd = std::string("cd ") + dataDir + " && python3 gen_data.py --m " +
        std::to_string(M) + " --n " + std::to_string(N) + " --k " + std::to_string(K) +
        (genArgs.empty() ? std::string("") : (std::string(" ") + genArgs));
    int genRet = system(cleanCmd.c_str());
    ASSERT_EQ(genRet, 0) << "Failed to clean old .bin files in qbmm_data";
    genRet = system(genDataCmd.c_str());
    ASSERT_EQ(genRet, 0) << "gen_data.py failed with exit code " << genRet;
}

// 读取 .bin 到 GM 缓冲区，带打开/读取断言。供 RunMixSmoke 与 fixpipe 用例复用。
void ReadBinToGm(const std::string& path, GM_ADDR gm, size_t size, const char* what)
{
    std::ifstream f(path, std::ios::binary);
    ASSERT_TRUE(f.is_open()) << "Failed to open " << what;
    f.read(reinterpret_cast<char*>(gm), size);
    ASSERT_TRUE(f.good()) << "Failed to read " << what << " (expected " << size << " bytes)";
}

// 通用 MIX smoke 执行体：生成数据 → 读入 GM → 填 tiling → KERNEL_RUN_KF（仅崩溃检测，与 PR #61 一致）。
template <typename Func>
void RunMixSmoke(Func kernelFunc, const MixCaseCfg& cfg)
{
    const int64_t M = cfg.M;
    const int64_t N = cfg.N;
    const int64_t K = cfg.K;
    const size_t x1ScaleCount = (cfg.x1QuantMode == QM_PERTOKEN) ? static_cast<size_t>(M) : 1;
    const size_t x2ScaleCount = (cfg.x2QuantMode == QM_PERCHANNEL) ? static_cast<size_t>(N) : 1;

    size_t x1Size = static_cast<size_t>(M) * K * sizeof(int8_t);
    size_t x2Size = static_cast<size_t>(K) * N * sizeof(int8_t);
    size_t pertokenScaleSize = x1ScaleCount * sizeof(float);
    size_t scaleSize = x2ScaleCount * sizeof(float);
    size_t biasSize = static_cast<size_t>(N) * cfg.biasElemSize;
    size_t ySize = static_cast<size_t>(M) * N * cfg.outElemSize;

    GM_ADDR x1GM = (GM_ADDR)AscendC::GmAlloc(x1Size);
    GM_ADDR x2GM = (GM_ADDR)AscendC::GmAlloc(x2Size);
    GM_ADDR pertokenScaleGM = (GM_ADDR)AscendC::GmAlloc(pertokenScaleSize);
    GM_ADDR scaleGM = (GM_ADDR)AscendC::GmAlloc(scaleSize);
    GM_ADDR biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    GM_ADDR yGM = (GM_ADDR)AscendC::GmAlloc(ySize);
    GM_ADDR tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(QBMMV3TilingData));

    ASSERT_NE(x1GM, nullptr);
    ASSERT_NE(x2GM, nullptr);
    ASSERT_NE(pertokenScaleGM, nullptr);
    ASSERT_NE(scaleGM, nullptr);
    ASSERT_NE(biasGM, nullptr);
    ASSERT_NE(yGM, nullptr);
    ASSERT_NE(tilingGM, nullptr);

    std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/quant_batch_matmul/qbmm_data";
    ASSERT_NO_FATAL_FAILURE(RunGenData(dataDir, M, N, K, cfg.genArgs));

    ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/input_a.bin", x1GM, x1Size, "input_a.bin"));
    ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/input_b.bin", x2GM, x2Size, "input_b.bin"));
    ASSERT_NO_FATAL_FAILURE(
        ReadBinToGm(dataDir + "/pertoken_scale.bin", pertokenScaleGM, pertokenScaleSize, "pertoken_scale.bin"));
    ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/scale.bin", scaleGM, scaleSize, "scale.bin"));

    if (cfg.isBias) {
        ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/bias.bin", biasGM, biasSize, "bias.bin"));
    }

    QBMMV3TilingData* tilingData = reinterpret_cast<QBMMV3TilingData*>(tilingGM);
    FillMixTiling(tilingData, cfg);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, cfg.blockNum, x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM, tilingGM))
        << "Kernel execution failed: one or more cores exited with non-zero status";

    AscendC::GmFree((void*)x1GM);
    AscendC::GmFree((void*)x2GM);
    AscendC::GmFree((void*)pertokenScaleGM);
    AscendC::GmFree((void*)scaleGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)yGM);
    AscendC::GmFree((void*)tilingGM);
}

} // namespace


TEST_F(QBMMV3Test, Test_INT8_A8W8_PERTENSOR)
{
    const int64_t M = 16;
    const int64_t N = 16;
    const int64_t K = 16;
    const uint32_t blockNum = 1;

    size_t x1Size = M * K * sizeof(int8_t);
    size_t x2Size = K * N * sizeof(int8_t);
    size_t pertokenScaleSize = sizeof(float);
    size_t scaleSize = sizeof(uint64_t);
    size_t biasSize = N * sizeof(int32_t);
    size_t ySize = M * N * sizeof(half);

    x1GM = (GM_ADDR)AscendC::GmAlloc(x1Size);
    x2GM = (GM_ADDR)AscendC::GmAlloc(x2Size);
    pertokenScaleGM = (GM_ADDR)AscendC::GmAlloc(pertokenScaleSize);
    scaleGM = (GM_ADDR)AscendC::GmAlloc(scaleSize);
    biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    yGM = (GM_ADDR)AscendC::GmAlloc(ySize);
    tilingGM = (GM_ADDR)AscendC::GmAlloc(sizeof(QBMMV3TilingData));

    ASSERT_NE(x1GM, nullptr);
    ASSERT_NE(x2GM, nullptr);
    ASSERT_NE(pertokenScaleGM, nullptr);
    ASSERT_NE(scaleGM, nullptr);
    ASSERT_NE(biasGM, nullptr);
    ASSERT_NE(yGM, nullptr);
    ASSERT_NE(tilingGM, nullptr);

    std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/quant_batch_matmul/qbmm_data";
    ASSERT_NO_FATAL_FAILURE(RunGenData(dataDir, M, N, K, ""));

    ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/input_a.bin", x1GM, x1Size, "input_a.bin"));
    ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/input_b.bin", x2GM, x2Size, "input_b.bin"));
    ASSERT_NO_FATAL_FAILURE(
        ReadBinToGm(dataDir + "/pertoken_scale.bin", pertokenScaleGM, pertokenScaleSize, "pertoken_scale.bin"));
    ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/scale.bin", scaleGM, scaleSize, "scale.bin"));

    QBMMV3TilingData* tilingData = reinterpret_cast<QBMMV3TilingData*>(tilingGM);

    // ProblemShape
    tilingData->m = M;
    tilingData->n = N;
    tilingData->k = K;
    tilingData->b = 1;

    // BlockMmadParams - GM addresses (set by kernel entry via REGISTER_TILING)
    tilingData->aGmAddr = 0;
    tilingData->bGmAddr = 0;
    tilingData->cGmAddr = 0;
    tilingData->biasGmAddr = 0;
    tilingData->scaleAGmAddr = 0;
    tilingData->scaleBGmAddr = 0;

    // BlockSchedulerParams
    tilingData->baseM = 16;
    tilingData->baseN = 16;
    tilingData->mTailTile = 1;
    tilingData->nTailTile = 1;
    tilingData->mBaseTailSplitCnt = 1;
    tilingData->nBaseTailSplitCnt = 1;
    tilingData->mTailMain = 0;
    tilingData->nTailMain = 0;

    // QBMMTiling
    tilingData->batchA1 = 1;
    tilingData->batchA2 = 1;
    tilingData->batchA3 = 1;
    tilingData->batchA4 = 1;
    tilingData->batchB1 = 1;
    tilingData->batchB2 = 1;
    tilingData->batchB3 = 1;
    tilingData->batchB4 = 1;
    tilingData->batchC1 = 1;
    tilingData->batchC2 = 1;
    tilingData->batchC3 = 1;
    tilingData->batchC4 = 1;
    tilingData->biasThreeDim = 0;
    tilingData->x1QuantMode = 0;  // DEFAULT_MODE
    tilingData->x2QuantMode = 1;  // PERTENSOR_MODE
    tilingData->kAL1 = 16;
    tilingData->kBL1 = 16;
    tilingData->nBufferNum = 1;
    tilingData->baseM_qbmm = 16;
    tilingData->baseN_qbmm = 16;
    tilingData->baseK_qbmm = 16;
    tilingData->isBias = 0;
    tilingData->dbL0C = 1;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = qbmm_kernel_entry<
        OP_TYPE_QBMM_CUBE, int8_t, int8_t, half, int32_t>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM, tilingGM))
        << "Kernel execution failed: one or more cores exited with non-zero status";
}

// ===================== MIX A8W8 dequant 路径用例矩阵（4.3）=====================
// 均为 smoke 测试：KERNEL_RUN_KF 仅检测 kernel 是否崩溃（与 PR #61 一致，不读回 golden 比对）。

// 最典型：激活 per-token + 权重 per-channel，双向量 scale，half 输出。
TEST_F(QBMMV3Test, Test_MIX_A8W8_PerChannel_PerToken)
{
    MixCaseCfg cfg{16, 16, 16, 1, QM_PERTOKEN, QM_PERCHANNEL, false, GE_DT_FLOAT,
        sizeof(float), sizeof(half),
        "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32"};
    auto kernelFunc = qbmm_kernel_entry<OP_TYPE_QBMM_MIX, int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

// 仅权重 scale：激活 DEFAULT（epilogue 忽略 x1 scale）+ 权重 per-channel，half 输出。
TEST_F(QBMMV3Test, Test_MIX_A8W8_PerChannel_NoPtScale)
{
    MixCaseCfg cfg{16, 16, 16, 1, QM_DEFAULT, QM_PERCHANNEL, false, GE_DT_FLOAT,
        sizeof(float), sizeof(half),
        "--x1_mode default --x2_mode perchannel --scale_dtype float32"};
    auto kernelFunc = qbmm_kernel_entry<OP_TYPE_QBMM_MIX, int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

// 权重标量 scale：激活 per-token + 权重 per-tensor，half 输出。
TEST_F(QBMMV3Test, Test_MIX_A8W8_PerTensor_PerToken)
{
    MixCaseCfg cfg{16, 16, 16, 1, QM_PERTOKEN, QM_PERTENSOR, false, GE_DT_FLOAT,
        sizeof(float), sizeof(half),
        "--x1_mode pertoken --x2_mode pertensor --scale_dtype float32"};
    auto kernelFunc = qbmm_kernel_entry<OP_TYPE_QBMM_MIX, int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

// 覆盖 bias 路径 + biasDtype=fp16：激活 per-token + 权重 per-channel + fp16 bias，half 输出。
TEST_F(QBMMV3Test, Test_MIX_A8W8_WithBias_FP16)
{
    MixCaseCfg cfg{16, 16, 16, 1, QM_PERTOKEN, QM_PERCHANNEL, true, GE_DT_FLOAT16,
        sizeof(half), sizeof(half),
        "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32 --bias --bias_dtype float16"};
    auto kernelFunc = qbmm_kernel_entry<OP_TYPE_QBMM_MIX, int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

// 覆盖 OutType=bf16：激活 per-token + 权重 per-channel，bfloat16 输出。
TEST_F(QBMMV3Test, Test_MIX_A8W8_Output_BF16)
{
    MixCaseCfg cfg{16, 16, 16, 1, QM_PERTOKEN, QM_PERCHANNEL, false, GE_DT_FLOAT,
        sizeof(float), sizeof(bfloat16_t),
        "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32"};
    auto kernelFunc = qbmm_kernel_entry<OP_TYPE_QBMM_MIX, int8_t, int8_t, bfloat16_t, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

// 单 batch 特化：走 QbmmMixWithoutBatch，激活 per-token + 权重 per-channel，half 输出。
TEST_F(QBMMV3Test, Test_MIX_A8W8_WithoutBatch)
{
    MixCaseCfg cfg{16, 16, 16, 1, QM_PERTOKEN, QM_PERCHANNEL, false, GE_DT_FLOAT,
        sizeof(float), sizeof(half),
        "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32"};
    auto kernelFunc = qbmm_kernel_entry<OP_TYPE_QBMM_MIX_NO_BATCH, int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}
