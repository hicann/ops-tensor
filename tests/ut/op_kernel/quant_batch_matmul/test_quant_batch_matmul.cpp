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
#include "../blaze_kernel_stub.h"
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
    std::string genCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
    std::string genDataCmd = std::string("cd ") + dataDir + " && python3 gen_data.py --m 16 --n 16 --k 16";
    int genRet = system(genCmd.c_str());
    ASSERT_EQ(genRet, 0) << "Failed to clean old .bin files in qbmm_data";
    genRet = system(genDataCmd.c_str());
    ASSERT_EQ(genRet, 0) << "gen_data.py failed with exit code " << genRet;

    std::ifstream x1File(dataDir + "/input_a.bin", std::ios::binary);
    ASSERT_TRUE(x1File.is_open()) << "Failed to open input_a.bin";
    std::ifstream x2File(dataDir + "/input_b.bin", std::ios::binary);
    ASSERT_TRUE(x2File.is_open()) << "Failed to open input_b.bin";
    std::ifstream pertokenScaleFile(dataDir + "/pertoken_scale.bin", std::ios::binary);
    ASSERT_TRUE(pertokenScaleFile.is_open()) << "Failed to open pertoken_scale.bin";
    std::ifstream scaleFile(dataDir + "/scale.bin", std::ios::binary);
    ASSERT_TRUE(scaleFile.is_open()) << "Failed to open scale.bin";
    x1File.read(reinterpret_cast<char*>(x1GM), x1Size);
    ASSERT_TRUE(x1File.good()) << "Failed to read input_a.bin (expected " << x1Size << " bytes)";
    x2File.read(reinterpret_cast<char*>(x2GM), x2Size);
    ASSERT_TRUE(x2File.good()) << "Failed to read input_b.bin (expected " << x2Size << " bytes)";
    pertokenScaleFile.read(reinterpret_cast<char*>(pertokenScaleGM), pertokenScaleSize);
    ASSERT_TRUE(pertokenScaleFile.good()) << "Failed to read pertoken_scale.bin (expected " << pertokenScaleSize << " bytes)";
    scaleFile.read(reinterpret_cast<char*>(scaleGM), scaleSize);
    ASSERT_TRUE(scaleFile.good()) << "Failed to read scale.bin (expected " << scaleSize << " bytes)";

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
    ICPU_RUN_KF(kernelFunc, blockNum, x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM, tilingGM);

    SUCCEED() << "QBMM kernel executed successfully (PV_MEM stubbed, output not verified)";
}
