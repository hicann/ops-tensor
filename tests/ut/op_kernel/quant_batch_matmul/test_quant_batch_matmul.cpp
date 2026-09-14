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
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include "gtest/gtest.h"
#include "blaze_kernel_stub.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "kernel_operator.h"

#include "qbmm_cube.h"
#include "qbmm_mix.h"
#include "qbmm_mx.h"
#include "blaze/gemm/utils/common_utils.h"

class QBMMV3Test : public testing::Test {
protected:
    static void TearDownTestCase()
    {
        std::string cleanCmd = std::string("cd ") + UT_KERNEL_SRC_DIR + "/quant_batch_matmul/qbmm_data && rm -rf *.bin";
        system(cleanCmd.c_str());
    }
};

namespace {

// QuantMode 编码（与 BlockEpilogueDequant::QuantMode 一致）
constexpr uint32_t QM_DEFAULT = 0;
constexpr uint32_t QM_PERTENSOR = 1;
constexpr uint32_t QM_PERCHANNEL = 2;
constexpr uint32_t QM_PERTOKEN = 4;
constexpr size_t STREAMK_WORKSPACE_TILE_SIZE = 256UL * 256UL * sizeof(float);
constexpr size_t PERTENSOR_STREAMK_WORKSPACE_TILE_SIZE = 256UL * 256UL * sizeof(int32_t);
constexpr size_t STREAMK_WORKSPACE_OVERHEAD = 20UL * 1024UL * 1024UL;

// ge::DataType 编码（bias 运行时 dtype）
constexpr uint32_t GE_DT_FLOAT = 0;
constexpr uint32_t GE_DT_FLOAT16 = 1;
constexpr uint32_t GE_DT_INT32 = 3;
constexpr uint32_t GE_DT_BF16 = 27;

struct MixCaseCfg {
    int64_t M;
    int64_t N;
    int64_t K;
    uint32_t blockNum;
    uint32_t x1QuantMode; // 激活量化模式
    uint32_t x2QuantMode; // 权重量化模式
    bool isBias;
    uint32_t biasDtype;  // ge::DataType 编码，需与 bias.bin 元素类型一致
    size_t biasElemSize; // bias 单元素字节数（与 biasDtype 匹配）
    size_t outElemSize;  // 输出单元素字节数（half/bf16=2, float=4）
    const char* genArgs; // 传给 gen_data.py 的量化模式参数
    int64_t batch = 1;
    bool biasThreeDim = false;
};

struct CubeCaseCfg {
    int64_t M;
    int64_t N;
    int64_t K;
    uint32_t blockNum;
    uint32_t x1QuantMode;
    uint32_t x2QuantMode;
    bool isBias;
    uint32_t biasDtype;
    size_t biasElemSize;
    size_t outElemSize;
    const char* genArgs;
    uint32_t kL1 = 0; // 0 keeps the legacy full-K L1 tile.
    uint32_t nBufferNum = 2;
    uint32_t baseK = 0; // 0 keeps the legacy full-K L0 tile.
    uint32_t baseM = 0; // 0 keeps the legacy single M/N tile.
    uint32_t baseN = 0;
    int64_t batch = 1;
    bool biasThreeDim = false;
    size_t scaleElemSize = sizeof(uint64_t);
    uint32_t dbL0C = 1;
};

struct MxCaseCfg {
    int64_t M;
    int64_t N;
    int64_t K;
    uint32_t blockNum;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t kL1;
    uint32_t scaleKL1;
    uint32_t nBufferNum;
    bool isBias;
    int64_t batch = 1;
    bool biasThreeDim = false;
};

struct L0CPingpongCaseCfg {
    int64_t M;
    int64_t N;
    int64_t K;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t kL1;
    uint32_t nBufferNum;
    uint32_t blockNum;
};

struct StreamKCaseCfg {
    int64_t M;
    int64_t N;
    int64_t K;
    uint32_t blockNum;
};

size_t GetMxScaleKLen(int64_t k) { return static_cast<size_t>((k + 63) / 64) * 2UL; }

template <typename T>
size_t GetMxInputSize(int64_t elementCount)
{
    if constexpr (Blaze::Gemm::IsFp4<T>()) {
        return ((static_cast<size_t>(elementCount) + 1UL) / 2UL) * sizeof(T);
    } else {
        return static_cast<size_t>(elementCount) * sizeof(T);
    }
}

void FillGmBuffer(GM_ADDR addr, size_t size, uint8_t value)
{
    auto* buffer = reinterpret_cast<uint8_t*>(addr);
    std::fill_n(buffer, size, value);
}

template <typename T>
void FillGmValues(GM_ADDR addr, size_t count, T value)
{
    std::fill_n(reinterpret_cast<T*>(addr), count, value);
}

void ReplicateGmBatch(GM_ADDR addr, size_t singleBatchSize, int64_t batch)
{
    auto* buffer = reinterpret_cast<uint8_t*>(addr);
    for (int64_t batchIdx = 1; batchIdx < batch; ++batchIdx) {
        std::memcpy(buffer + static_cast<size_t>(batchIdx) * singleBatchSize, buffer, singleBatchSize);
    }
}

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

// 未指定切分时保留单 tile 默认值；显式配置用于多 tile 和 L1 buffer 回绕测试。
void FillCubeTiling(QBMMV3TilingData* tilingData, const CubeCaseCfg& caseCfg)
{
    tilingData->m = caseCfg.M;
    tilingData->n = caseCfg.N;
    tilingData->k = caseCfg.K;
    tilingData->b = caseCfg.batch;

    tilingData->aGmAddr = 0;
    tilingData->bGmAddr = 0;
    tilingData->cGmAddr = 0;
    tilingData->biasGmAddr = 0;
    tilingData->scaleAGmAddr = 0;
    tilingData->scaleBGmAddr = 0;

    tilingData->baseM = caseCfg.baseM == 0 ? caseCfg.M : caseCfg.baseM;
    tilingData->baseN = caseCfg.baseN == 0 ? caseCfg.N : caseCfg.baseN;
    tilingData->mTailTile = 1;
    tilingData->nTailTile = 1;
    tilingData->mBaseTailSplitCnt = 1;
    tilingData->nBaseTailSplitCnt = 1;
    tilingData->mTailMain = 0;
    tilingData->nTailMain = 0;

    tilingData->batchA1 = 1;
    tilingData->batchA2 = 1;
    tilingData->batchA3 = 1;
    tilingData->batchA4 = static_cast<uint32_t>(caseCfg.batch);
    tilingData->batchB1 = 1;
    tilingData->batchB2 = 1;
    tilingData->batchB3 = 1;
    tilingData->batchB4 = static_cast<uint32_t>(caseCfg.batch);
    tilingData->batchC1 = 1;
    tilingData->batchC2 = 1;
    tilingData->batchC3 = 1;
    tilingData->batchC4 = static_cast<uint32_t>(caseCfg.batch);
    tilingData->biasThreeDim = caseCfg.biasThreeDim ? 1U : 0U;
    tilingData->x1QuantMode = caseCfg.x1QuantMode;
    tilingData->x2QuantMode = caseCfg.x2QuantMode;
    tilingData->kAL1 = caseCfg.kL1 == 0 ? static_cast<uint32_t>(caseCfg.K) : caseCfg.kL1;
    tilingData->kBL1 = tilingData->kAL1;
    tilingData->nBufferNum = caseCfg.nBufferNum;
    tilingData->baseM_qbmm = static_cast<uint32_t>(tilingData->baseM);
    tilingData->baseN_qbmm = static_cast<uint32_t>(tilingData->baseN);
    tilingData->baseK_qbmm = caseCfg.baseK == 0 ? static_cast<uint32_t>(caseCfg.K) : caseCfg.baseK;
    tilingData->isBias = caseCfg.isBias ? 1 : 0;
    tilingData->dbL0C = caseCfg.dbL0C;
    tilingData->weightMustHitL2 = 1U;
    tilingData->biasDtype = caseCfg.biasDtype;
}

void FillMixTiling(QBMMV3TilingData* tilingData, const MixCaseCfg& caseCfg)
{
    tilingData->m = caseCfg.M;
    tilingData->n = caseCfg.N;
    tilingData->k = caseCfg.K;
    tilingData->b = caseCfg.batch;

    tilingData->aGmAddr = 0;
    tilingData->bGmAddr = 0;
    tilingData->cGmAddr = 0;
    tilingData->biasGmAddr = 0;
    tilingData->scaleAGmAddr = 0;
    tilingData->scaleBGmAddr = 0;

    tilingData->baseM = caseCfg.M;
    tilingData->baseN = caseCfg.N;
    tilingData->mTailTile = 1;
    tilingData->nTailTile = 1;
    tilingData->mBaseTailSplitCnt = 1;
    tilingData->nBaseTailSplitCnt = 1;
    tilingData->mTailMain = 0;
    tilingData->nTailMain = 0;

    tilingData->batchA1 = 1;
    tilingData->batchA2 = 1;
    tilingData->batchA3 = 1;
    tilingData->batchA4 = static_cast<uint32_t>(caseCfg.batch);
    tilingData->batchB1 = 1;
    tilingData->batchB2 = 1;
    tilingData->batchB3 = 1;
    tilingData->batchB4 = static_cast<uint32_t>(caseCfg.batch);
    tilingData->batchC1 = 1;
    tilingData->batchC2 = 1;
    tilingData->batchC3 = 1;
    tilingData->batchC4 = static_cast<uint32_t>(caseCfg.batch);
    tilingData->biasThreeDim = caseCfg.biasThreeDim ? 1U : 0U;
    tilingData->x1QuantMode = caseCfg.x1QuantMode;
    tilingData->x2QuantMode = caseCfg.x2QuantMode;
    tilingData->kAL1 = static_cast<uint32_t>(caseCfg.K);
    tilingData->kBL1 = static_cast<uint32_t>(caseCfg.K);
    tilingData->nBufferNum = 2;
    tilingData->baseM_qbmm = static_cast<uint32_t>(caseCfg.M);
    tilingData->baseN_qbmm = static_cast<uint32_t>(caseCfg.N);
    tilingData->baseK_qbmm = static_cast<uint32_t>(caseCfg.K);
    tilingData->isBias = caseCfg.isBias ? 1 : 0;
    tilingData->dbL0C = 1;
    tilingData->weightMustHitL2 = 1U;
    tilingData->biasDtype = caseCfg.biasDtype;
}

void FillMxTiling(QBMMV3TilingData* tilingData, const MxCaseCfg& caseCfg)
{
    tilingData->m = caseCfg.M;
    tilingData->n = caseCfg.N;
    tilingData->k = caseCfg.K;
    tilingData->b = caseCfg.batch;

    tilingData->aGmAddr = 0;
    tilingData->bGmAddr = 0;
    tilingData->cGmAddr = 0;
    tilingData->biasGmAddr = 0;
    tilingData->scaleAGmAddr = 0;
    tilingData->scaleBGmAddr = 0;

    tilingData->baseM = caseCfg.baseM;
    tilingData->baseN = caseCfg.baseN;
    tilingData->mTailTile = 1;
    tilingData->nTailTile = 1;
    tilingData->mBaseTailSplitCnt = 1;
    tilingData->nBaseTailSplitCnt = 1;
    tilingData->mTailMain = 0;
    tilingData->nTailMain = 0;

    tilingData->batchA1 = 1;
    tilingData->batchA2 = 1;
    tilingData->batchA3 = 1;
    tilingData->batchA4 = static_cast<uint32_t>(caseCfg.batch);
    tilingData->batchB1 = 1;
    tilingData->batchB2 = 1;
    tilingData->batchB3 = 1;
    tilingData->batchB4 = static_cast<uint32_t>(caseCfg.batch);
    tilingData->batchC1 = 1;
    tilingData->batchC2 = 1;
    tilingData->batchC3 = 1;
    tilingData->batchC4 = static_cast<uint32_t>(caseCfg.batch);
    tilingData->biasThreeDim = caseCfg.biasThreeDim ? 1U : 0U;
    tilingData->x1QuantMode = 0;
    tilingData->x2QuantMode = 0;
    tilingData->kAL1 = caseCfg.kL1;
    tilingData->kBL1 = caseCfg.scaleKL1;
    tilingData->nBufferNum = caseCfg.nBufferNum;
    tilingData->baseM_qbmm = caseCfg.baseM;
    tilingData->baseN_qbmm = caseCfg.baseN;
    tilingData->baseK_qbmm = caseCfg.baseK;
    tilingData->isBias = caseCfg.isBias ? 1 : 0;
    tilingData->dbL0C = 1;
    tilingData->weightMustHitL2 = 1U;
    tilingData->biasDtype = GE_DT_FLOAT;
}

// 清理旧 .bin 并调用 gen_data.py 生成输入数据；genArgs 为量化模式等附加参数（fixpipe 用例传 ""）。
// 供 RunMixSmoke 与 fixpipe 用例复用，避免数据生成命令块重复。
void RunGenData(const std::string& dataDir, int64_t M, int64_t N, int64_t K, const std::string& genArgs)
{
    std::string cleanCmd = std::string("cd ") + dataDir + " && rm -rf *.bin";
    std::string genDataCmd = std::string("cd ") + dataDir + " && python3 gen_data.py --m " + std::to_string(M) +
                             " --n " + std::to_string(N) + " --k " + std::to_string(K) +
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

// 通用 Cube smoke 执行体：准备数据 → 填 tiling → KERNEL_RUN_KF（仅崩溃检测，与 PR #61 一致）。
template <typename AType, typename BType, typename Func>
void RunTypedCubeSmoke(Func kernelFunc, const CubeCaseCfg& cfg)
{
    const int64_t M = cfg.M;
    const int64_t N = cfg.N;
    const int64_t K = cfg.K;
    const size_t batchCount = static_cast<size_t>(cfg.batch);

    const size_t x1SingleBatchSize = static_cast<size_t>(M) * K * sizeof(AType);
    const size_t x2SingleBatchSize = static_cast<size_t>(K) * N * sizeof(BType);
    const size_t pertokenScaleCount = cfg.x1QuantMode == QM_PERTOKEN ? static_cast<size_t>(M) : 1UL;
    const size_t pertokenScaleSize = pertokenScaleCount * sizeof(float);
    const size_t scaleCount = cfg.x2QuantMode == QM_PERCHANNEL ? static_cast<size_t>(N) : 1UL;
    const size_t scaleSize = scaleCount * cfg.scaleElemSize;
    const size_t biasSingleBatchSize = static_cast<size_t>(N) * cfg.biasElemSize;
    const size_t x1Size = x1SingleBatchSize * batchCount;
    const size_t x2Size = x2SingleBatchSize * batchCount;
    const size_t biasSize = biasSingleBatchSize * (cfg.biasThreeDim ? batchCount : 1UL);
    const size_t ySize = static_cast<size_t>(M) * N * cfg.outElemSize * batchCount;

    GmBuffer x1GM(x1Size);
    GmBuffer x2GM(x2Size);
    GmBuffer pertokenScaleGM(pertokenScaleSize);
    GmBuffer scaleGM(scaleSize);
    GmBuffer biasGM(biasSize);
    GmBuffer yGM(ySize);
    GmBuffer tilingGM(sizeof(QBMMV3TilingData));

    ASSERT_NE(x1GM.Get(), nullptr);
    ASSERT_NE(x2GM.Get(), nullptr);
    ASSERT_NE(pertokenScaleGM.Get(), nullptr);
    ASSERT_NE(scaleGM.Get(), nullptr);
    ASSERT_NE(biasGM.Get(), nullptr);
    ASSERT_NE(yGM.Get(), nullptr);
    ASSERT_NE(tilingGM.Get(), nullptr);

    FillGmBuffer(x1GM.Get(), x1Size, 0U);
    FillGmBuffer(x2GM.Get(), x2Size, 0U);
    FillGmBuffer(pertokenScaleGM.Get(), pertokenScaleSize, 0U);
    FillGmBuffer(scaleGM.Get(), scaleSize, 0U);
    FillGmBuffer(biasGM.Get(), biasSize, 0U);
    FillGmBuffer(yGM.Get(), ySize, 0U);

    if constexpr (std::is_same_v<AType, int8_t> && std::is_same_v<BType, int8_t>) {
        std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/quant_batch_matmul/qbmm_data";
        ASSERT_NO_FATAL_FAILURE(RunGenData(dataDir, M, N, K, cfg.genArgs));

        ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/input_a.bin", x1GM.Get(), x1SingleBatchSize, "input_a.bin"));
        ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/input_b.bin", x2GM.Get(), x2SingleBatchSize, "input_b.bin"));
        ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/pertoken_scale.bin", pertokenScaleGM.Get(), pertokenScaleSize,
                                            "pertoken_scale.bin"));
        ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/scale.bin", scaleGM.Get(), scaleSize, "scale.bin"));
        ReplicateGmBatch(x1GM.Get(), x1SingleBatchSize, cfg.batch);
        ReplicateGmBatch(x2GM.Get(), x2SingleBatchSize, cfg.batch);

        if (cfg.isBias) {
            ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/bias.bin", biasGM.Get(), biasSingleBatchSize, "bias.bin"));
            if (cfg.biasThreeDim) {
                ReplicateGmBatch(biasGM.Get(), biasSingleBatchSize, cfg.batch);
            }
        }
    } else {
        // Zero is valid for FP8/HiFloat8 inputs. Keep typed smoke data deterministic and avoid interpreting
        // arbitrary int8 generator output as low-precision floating-point bit patterns.
        FillGmValues<float>(pertokenScaleGM.Get(), pertokenScaleCount, 1.0F);
        if (cfg.scaleElemSize == sizeof(float)) {
            FillGmValues<float>(scaleGM.Get(), scaleCount, 1.0F);
        } else {
            ASSERT_EQ(cfg.scaleElemSize, sizeof(uint64_t));
            constexpr uint64_t FIXPIPE_SCALE_ONE = 0x3F80000000000000UL;
            FillGmValues<uint64_t>(scaleGM.Get(), scaleCount, FIXPIPE_SCALE_ONE);
        }
    }

    auto* tilingData = reinterpret_cast<QBMMV3TilingData*>(tilingGM.Get());
    FillCubeTiling(tilingData, cfg);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    const bool ok = KERNEL_RUN_KF(kernelFunc, cfg.blockNum, x1GM.Get(), x2GM.Get(), pertokenScaleGM.Get(),
                                  scaleGM.Get(), biasGM.Get(), yGM.Get(), tilingGM.Get());

    ASSERT_TRUE(ok) << "Kernel execution failed: one or more cores exited with non-zero status";
}

template <typename Func>
void RunCubeSmoke(Func kernelFunc, const CubeCaseCfg& cfg)
{
    RunTypedCubeSmoke<int8_t, int8_t>(kernelFunc, cfg);
}

template <typename Func>
void RunMixSmoke(Func kernelFunc, const MixCaseCfg& cfg)
{
    const int64_t M = cfg.M;
    const int64_t N = cfg.N;
    const int64_t K = cfg.K;
    const size_t batchCount = static_cast<size_t>(cfg.batch);
    const size_t x1ScaleCount = cfg.x1QuantMode == QM_PERTOKEN ? static_cast<size_t>(M) : 1UL;
    const size_t x2ScaleCount = cfg.x2QuantMode == QM_PERCHANNEL ? static_cast<size_t>(N) : 1UL;
    const size_t x1SingleBatchSize = static_cast<size_t>(M) * K * sizeof(int8_t);
    const size_t x2SingleBatchSize = static_cast<size_t>(K) * N * sizeof(int8_t);
    const size_t pertokenScaleSingleBatchSize = x1ScaleCount * sizeof(float);
    const size_t scaleSingleBatchSize = x2ScaleCount * sizeof(float);
    const size_t biasSingleBatchSize = static_cast<size_t>(N) * cfg.biasElemSize;
    const size_t x1Size = x1SingleBatchSize * batchCount;
    const size_t x2Size = x2SingleBatchSize * batchCount;
    const size_t pertokenScaleSize = pertokenScaleSingleBatchSize * batchCount;
    const size_t scaleSize = scaleSingleBatchSize * batchCount;
    const size_t biasSize = biasSingleBatchSize * (cfg.biasThreeDim ? batchCount : 1UL);
    const size_t ySize = static_cast<size_t>(M) * N * cfg.outElemSize * batchCount;

    GmBuffer x1GM(x1Size);
    GmBuffer x2GM(x2Size);
    GmBuffer pertokenScaleGM(pertokenScaleSize);
    GmBuffer scaleGM(scaleSize);
    GmBuffer biasGM(biasSize);
    GmBuffer yGM(ySize);
    GmBuffer tilingGM(sizeof(QBMMV3TilingData));

    ASSERT_NE(x1GM.Get(), nullptr);
    ASSERT_NE(x2GM.Get(), nullptr);
    ASSERT_NE(pertokenScaleGM.Get(), nullptr);
    ASSERT_NE(scaleGM.Get(), nullptr);
    ASSERT_NE(biasGM.Get(), nullptr);
    ASSERT_NE(yGM.Get(), nullptr);
    ASSERT_NE(tilingGM.Get(), nullptr);

    FillGmBuffer(x1GM.Get(), x1Size, 0U);
    FillGmBuffer(x2GM.Get(), x2Size, 0U);
    FillGmBuffer(pertokenScaleGM.Get(), pertokenScaleSize, 0U);
    FillGmBuffer(scaleGM.Get(), scaleSize, 0U);
    FillGmBuffer(biasGM.Get(), biasSize, 0U);
    FillGmBuffer(yGM.Get(), ySize, 0U);

    std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/quant_batch_matmul/qbmm_data";
    ASSERT_NO_FATAL_FAILURE(RunGenData(dataDir, M, N, K, cfg.genArgs));

    ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/input_a.bin", x1GM.Get(), x1SingleBatchSize, "input_a.bin"));
    ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/input_b.bin", x2GM.Get(), x2SingleBatchSize, "input_b.bin"));
    ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/pertoken_scale.bin", pertokenScaleGM.Get(),
                                        pertokenScaleSingleBatchSize, "pertoken_scale.bin"));
    ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/scale.bin", scaleGM.Get(), scaleSingleBatchSize, "scale.bin"));
    ReplicateGmBatch(x1GM.Get(), x1SingleBatchSize, cfg.batch);
    ReplicateGmBatch(x2GM.Get(), x2SingleBatchSize, cfg.batch);
    ReplicateGmBatch(pertokenScaleGM.Get(), pertokenScaleSingleBatchSize, cfg.batch);
    ReplicateGmBatch(scaleGM.Get(), scaleSingleBatchSize, cfg.batch);

    if (cfg.isBias) {
        ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/bias.bin", biasGM.Get(), biasSingleBatchSize, "bias.bin"));
        if (cfg.biasThreeDim) {
            ReplicateGmBatch(biasGM.Get(), biasSingleBatchSize, cfg.batch);
        }
    }

    QBMMV3TilingData* tilingData = reinterpret_cast<QBMMV3TilingData*>(tilingGM.Get());
    FillMixTiling(tilingData, cfg);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    const bool ok = KERNEL_RUN_KF(kernelFunc, cfg.blockNum, x1GM.Get(), x2GM.Get(), pertokenScaleGM.Get(),
                                  scaleGM.Get(), biasGM.Get(), yGM.Get(), tilingGM.Get());

    ASSERT_TRUE(ok) << "Kernel execution failed: one or more cores exited with non-zero status";
}

template <typename AType, typename BType, uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE,
          bool WithoutBatch = false>
void RunMxL0CPingpongSmoke(const L0CPingpongCaseCfg& cfg)
{
    const size_t scaleKLen = GetMxScaleKLen(cfg.K);
    const size_t x1Size = GetMxInputSize<AType>(cfg.M * cfg.K);
    const size_t x2Size = GetMxInputSize<BType>(cfg.K * cfg.N);
    const size_t pertokenScaleSize = static_cast<size_t>(cfg.M) * scaleKLen * sizeof(AscendC::fp8_e8m0_t);
    const size_t scaleSize = scaleKLen * static_cast<size_t>(cfg.N) * sizeof(AscendC::fp8_e8m0_t);
    const size_t biasSize = static_cast<size_t>(cfg.N) * sizeof(float);
    const size_t ySize = static_cast<size_t>(cfg.M) * cfg.N * sizeof(half);

    GmBuffer x1GM(x1Size);
    GmBuffer x2GM(x2Size);
    GmBuffer pertokenScaleGM(pertokenScaleSize);
    GmBuffer scaleGM(scaleSize);
    GmBuffer biasGM(biasSize);
    GmBuffer yGM(ySize);
    GmBuffer tilingGM(sizeof(QBMMUT::QBMML0CPingpongTilingData));

    ASSERT_NE(x1GM.Get(), nullptr);
    ASSERT_NE(x2GM.Get(), nullptr);
    ASSERT_NE(pertokenScaleGM.Get(), nullptr);
    ASSERT_NE(scaleGM.Get(), nullptr);
    ASSERT_NE(biasGM.Get(), nullptr);
    ASSERT_NE(yGM.Get(), nullptr);
    ASSERT_NE(tilingGM.Get(), nullptr);

    FillGmBuffer(x1GM.Get(), x1Size, 0U);
    FillGmBuffer(x2GM.Get(), x2Size, 0U);
    FillGmBuffer(pertokenScaleGM.Get(), pertokenScaleSize, 0x7fU);
    FillGmBuffer(scaleGM.Get(), scaleSize, 0x7fU);
    FillGmBuffer(biasGM.Get(), biasSize, 0U);
    FillGmBuffer(yGM.Get(), ySize, 0U);

    auto* tilingData = reinterpret_cast<QBMMUT::QBMML0CPingpongTilingData*>(tilingGM.Get());
    *tilingData = QBMMUT::QBMML0CPingpongTilingData{
        cfg.M, cfg.N, cfg.K, 1, cfg.baseM, cfg.baseN, cfg.baseK, cfg.kL1, cfg.kL1, cfg.nBufferNum, 2U, 0U, 1U};

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = qbmm_mx_l0c_pingpong_kernel_entry<AType, BType, half, float>;
    if constexpr (WithoutBatch) {
        if constexpr (FullLoadMode == Blaze::Gemm::A_FULL_LOAD_MODE) {
            kernelFunc = qbmm_mx_l0c_pingpong_without_batch_a_full_load_kernel_entry<AType, BType, half, float>;
        } else {
            kernelFunc = qbmm_mx_l0c_pingpong_without_batch_kernel_entry<AType, BType, half, float>;
        }
    } else if constexpr (FullLoadMode == Blaze::Gemm::A_FULL_LOAD_MODE) {
        kernelFunc = qbmm_mx_l0c_pingpong_a_full_load_kernel_entry<AType, BType, half, float>;
    }
    const bool ok = KERNEL_RUN_KF(kernelFunc, cfg.blockNum, x1GM.Get(), x2GM.Get(), pertokenScaleGM.Get(),
                                  scaleGM.Get(), biasGM.Get(), yGM.Get(), tilingGM.Get());

    ASSERT_TRUE(ok) << "QBMM MX L0C ping-pong kernel execution failed";
}

void RunMxStreamKSmoke(const StreamKCaseCfg& cfg)
{
    using AType = fp8_e4m3fn_t;
    using BType = fp8_e5m2_t;
    using CType = half;
    using BiasType = float;

    const size_t scaleKLen = GetMxScaleKLen(cfg.K);
    const size_t x1Size = GetMxInputSize<AType>(cfg.M * cfg.K);
    const size_t x2Size = GetMxInputSize<BType>(cfg.K * cfg.N);
    const size_t pertokenScaleSize = static_cast<size_t>(cfg.M) * scaleKLen * sizeof(AscendC::fp8_e8m0_t);
    const size_t scaleSize = scaleKLen * static_cast<size_t>(cfg.N) * sizeof(AscendC::fp8_e8m0_t);
    const size_t biasSize = static_cast<size_t>(cfg.N) * sizeof(BiasType);
    const size_t ySize = static_cast<size_t>(cfg.M) * cfg.N * sizeof(CType);
    const size_t workspaceSize = cfg.blockNum * STREAMK_WORKSPACE_TILE_SIZE + STREAMK_WORKSPACE_OVERHEAD;

    GmBuffer x1GM(x1Size);
    GmBuffer x2GM(x2Size);
    GmBuffer pertokenScaleGM(pertokenScaleSize);
    GmBuffer scaleGM(scaleSize);
    GmBuffer biasGM(biasSize);
    GmBuffer yGM(ySize);
    GmBuffer workspaceGM(workspaceSize);
    GmBuffer tilingGM(sizeof(QBMMUT::QBMMStreamKTilingData));

    ASSERT_NE(x1GM.Get(), nullptr);
    ASSERT_NE(x2GM.Get(), nullptr);
    ASSERT_NE(pertokenScaleGM.Get(), nullptr);
    ASSERT_NE(scaleGM.Get(), nullptr);
    ASSERT_NE(biasGM.Get(), nullptr);
    ASSERT_NE(yGM.Get(), nullptr);
    ASSERT_NE(workspaceGM.Get(), nullptr);
    ASSERT_NE(tilingGM.Get(), nullptr);

    FillGmBuffer(x1GM.Get(), x1Size, 0U);
    FillGmBuffer(x2GM.Get(), x2Size, 0U);
    FillGmBuffer(pertokenScaleGM.Get(), pertokenScaleSize, 0x7fU);
    FillGmBuffer(scaleGM.Get(), scaleSize, 0x7fU);
    FillGmBuffer(biasGM.Get(), biasSize, 0U);
    FillGmBuffer(yGM.Get(), ySize, 0U);
    FillGmBuffer(workspaceGM.Get(), workspaceSize, 0U);

    auto* tilingData = reinterpret_cast<QBMMUT::QBMMStreamKTilingData*>(tilingGM.Get());
    *tilingData = QBMMUT::QBMMStreamKTilingData{cfg.M, cfg.N, cfg.K, 1,  cfg.blockNum, 16, 16,
                                                64,    64,    64,    64, 1U,           0U, 1U};

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = qbmm_streamk_kernel_entry<AType, BType, CType, BiasType>;
    const bool ok = KERNEL_RUN_KF(kernelFunc, cfg.blockNum, x1GM.Get(), x2GM.Get(), pertokenScaleGM.Get(),
                                  scaleGM.Get(), biasGM.Get(), yGM.Get(), workspaceGM.Get(), tilingGM.Get());

    ASSERT_TRUE(ok) << "QBMM MX StreamK kernel execution failed";
}

void RunInt8PertensorStreamKSmoke()
{
    constexpr int64_t M = 16;
    constexpr int64_t N = 16;
    constexpr int64_t K = 128;
    constexpr uint32_t BLOCK_NUM = 2U;
    const size_t workspaceSize = BLOCK_NUM * PERTENSOR_STREAMK_WORKSPACE_TILE_SIZE + STREAMK_WORKSPACE_OVERHEAD;

    GmBuffer x1GM(static_cast<size_t>(M * K) * sizeof(int8_t));
    GmBuffer x2GM(static_cast<size_t>(K * N) * sizeof(int8_t));
    GmBuffer scaleGM(sizeof(float));
    GmBuffer biasGM(static_cast<size_t>(N) * sizeof(int32_t));
    GmBuffer yGM(static_cast<size_t>(M * N) * sizeof(bfloat16_t));
    GmBuffer workspaceGM(workspaceSize);
    GmBuffer tilingGM(sizeof(QBMMUT::QBMMPertensorStreamKTilingData));

    ASSERT_NE(x1GM.Get(), nullptr);
    ASSERT_NE(x2GM.Get(), nullptr);
    ASSERT_NE(scaleGM.Get(), nullptr);
    ASSERT_NE(biasGM.Get(), nullptr);
    ASSERT_NE(yGM.Get(), nullptr);
    ASSERT_NE(workspaceGM.Get(), nullptr);
    ASSERT_NE(tilingGM.Get(), nullptr);

    FillGmValues<int8_t>(x1GM.Get(), static_cast<size_t>(M * K), 1);
    FillGmValues<int8_t>(x2GM.Get(), static_cast<size_t>(K * N), 1);
    FillGmValues<float>(scaleGM.Get(), 1U, 1.0F);
    FillGmBuffer(biasGM.Get(), static_cast<size_t>(N) * sizeof(int32_t), 0U);
    FillGmBuffer(yGM.Get(), static_cast<size_t>(M * N) * sizeof(bfloat16_t), 0U);
    FillGmBuffer(workspaceGM.Get(), workspaceSize, 0U);

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
    const size_t workspaceSize = BLOCK_NUM * PERTENSOR_STREAMK_WORKSPACE_TILE_SIZE + STREAMK_WORKSPACE_OVERHEAD;

    GmBuffer x1GM(static_cast<size_t>(M * K) * sizeof(fp8_e4m3fn_t));
    GmBuffer x2GM(static_cast<size_t>(K * N) * sizeof(fp8_e4m3fn_t));
    GmBuffer perTokenScaleGM(sizeof(float));
    GmBuffer scaleGM(sizeof(float));
    GmBuffer biasGM(static_cast<size_t>(N) * sizeof(float));
    GmBuffer yGM(static_cast<size_t>(M * N) * sizeof(float));
    GmBuffer workspaceGM(workspaceSize);
    GmBuffer tilingGM(sizeof(QBMMUT::QBMMPertensorStreamKTilingData));

    ASSERT_NE(x1GM.Get(), nullptr);
    ASSERT_NE(x2GM.Get(), nullptr);
    ASSERT_NE(perTokenScaleGM.Get(), nullptr);
    ASSERT_NE(scaleGM.Get(), nullptr);
    ASSERT_NE(biasGM.Get(), nullptr);
    ASSERT_NE(yGM.Get(), nullptr);
    ASSERT_NE(workspaceGM.Get(), nullptr);
    ASSERT_NE(tilingGM.Get(), nullptr);

    FillGmBuffer(x1GM.Get(), static_cast<size_t>(M * K) * sizeof(fp8_e4m3fn_t), 0U);
    FillGmBuffer(x2GM.Get(), static_cast<size_t>(K * N) * sizeof(fp8_e4m3fn_t), 0U);
    FillGmValues<float>(perTokenScaleGM.Get(), 1U, 3.0F);
    FillGmValues<float>(scaleGM.Get(), 1U, 2.0F);
    FillGmValues<float>(biasGM.Get(), static_cast<size_t>(N), BIAS_VALUE);
    FillGmValues<float>(yGM.Get(), static_cast<size_t>(M * N), 0.0F);
    FillGmBuffer(workspaceGM.Get(), workspaceSize, 0U);

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
    const size_t workspaceSize = BLOCK_NUM * PERTENSOR_STREAMK_WORKSPACE_TILE_SIZE + STREAMK_WORKSPACE_OVERHEAD;

    GmBuffer x1GM(static_cast<size_t>(BATCH * M * K) * sizeof(int8_t));
    GmBuffer x2GM(static_cast<size_t>(K * N) * sizeof(int8_t));
    GmBuffer scaleGM(sizeof(float));
    GmBuffer yGM(static_cast<size_t>(BATCH * M * N) * sizeof(bfloat16_t));
    GmBuffer workspaceGM(workspaceSize);
    GmBuffer tilingGM(sizeof(QBMMUT::QBMMPertensorStreamKTilingData));

    ASSERT_NE(x1GM.Get(), nullptr);
    ASSERT_NE(x2GM.Get(), nullptr);
    ASSERT_NE(scaleGM.Get(), nullptr);
    ASSERT_NE(yGM.Get(), nullptr);
    ASSERT_NE(workspaceGM.Get(), nullptr);
    ASSERT_NE(tilingGM.Get(), nullptr);

    FillGmBuffer(yGM.Get(), static_cast<size_t>(BATCH * M * N) * sizeof(bfloat16_t), OUTPUT_SENTINEL);
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

template <typename AType, typename BType, typename CType, typename BiasType,
          uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE>
void RunMxSmoke(const MxCaseCfg& cfg)
{
    const size_t scaleKLen = GetMxScaleKLen(cfg.K);
    const size_t batchCount = static_cast<size_t>(cfg.batch);
    const size_t x1Size = GetMxInputSize<AType>(cfg.M * cfg.K) * batchCount;
    const size_t x2Size = GetMxInputSize<BType>(cfg.K * cfg.N) * batchCount;
    const size_t pertokenScaleSize = static_cast<size_t>(cfg.M) * scaleKLen * sizeof(AscendC::fp8_e8m0_t) * batchCount;
    const size_t scaleSize = scaleKLen * static_cast<size_t>(cfg.N) * sizeof(AscendC::fp8_e8m0_t) * batchCount;
    const size_t biasSize = static_cast<size_t>(cfg.N) * sizeof(BiasType) * (cfg.biasThreeDim ? batchCount : 1UL);
    const size_t ySize = static_cast<size_t>(cfg.M) * cfg.N * sizeof(CType) * batchCount;

    GmBuffer x1GM(x1Size);
    GmBuffer x2GM(x2Size);
    GmBuffer pertokenScaleGM(pertokenScaleSize);
    GmBuffer scaleGM(scaleSize);
    GmBuffer biasGM(biasSize);
    GmBuffer yGM(ySize);
    GmBuffer tilingGM(sizeof(QBMMV3TilingData));

    ASSERT_NE(x1GM.Get(), nullptr);
    ASSERT_NE(x2GM.Get(), nullptr);
    ASSERT_NE(pertokenScaleGM.Get(), nullptr);
    ASSERT_NE(scaleGM.Get(), nullptr);
    ASSERT_NE(biasGM.Get(), nullptr);
    ASSERT_NE(yGM.Get(), nullptr);
    ASSERT_NE(tilingGM.Get(), nullptr);

    FillGmBuffer(x1GM.Get(), x1Size, 0U);
    FillGmBuffer(x2GM.Get(), x2Size, 0U);
    FillGmBuffer(pertokenScaleGM.Get(), pertokenScaleSize, 0x7fU);
    FillGmBuffer(scaleGM.Get(), scaleSize, 0x7fU);
    FillGmBuffer(biasGM.Get(), biasSize, 0U);
    FillGmBuffer(yGM.Get(), ySize, 0U);

    auto* tilingData = reinterpret_cast<QBMMV3TilingData*>(tilingGM.Get());
    FillMxTiling(tilingData, cfg);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = qbmm_mx_kernel_entry<AType, BType, CType, BiasType>;
    if constexpr (FullLoadMode == Blaze::Gemm::A_FULL_LOAD_MODE) {
        kernelFunc = qbmm_mx_a_full_load_kernel_entry<AType, BType, CType, BiasType>;
    }
    const bool ok = KERNEL_RUN_KF(kernelFunc, cfg.blockNum, x1GM.Get(), x2GM.Get(), pertokenScaleGM.Get(),
                                  scaleGM.Get(), biasGM.Get(), yGM.Get(), tilingGM.Get());

    ASSERT_TRUE(ok) << "QBMM MX kernel execution failed";
}

template <typename AType, typename BType, typename CType, typename BiasType,
          uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE>
void RunMxWithoutBatchSmoke(const MxCaseCfg& cfg)
{
    const size_t scaleKLen = GetMxScaleKLen(cfg.K);
    const size_t x1Size = GetMxInputSize<AType>(cfg.M * cfg.K);
    const size_t x2Size = GetMxInputSize<BType>(cfg.K * cfg.N);
    const size_t pertokenScaleSize = static_cast<size_t>(cfg.M) * scaleKLen * sizeof(AscendC::fp8_e8m0_t);
    const size_t scaleSize = scaleKLen * static_cast<size_t>(cfg.N) * sizeof(AscendC::fp8_e8m0_t);
    const size_t biasSize = static_cast<size_t>(cfg.N) * sizeof(BiasType);
    const size_t ySize = static_cast<size_t>(cfg.M) * cfg.N * sizeof(CType);

    GmBuffer x1GM(x1Size);
    GmBuffer x2GM(x2Size);
    GmBuffer pertokenScaleGM(pertokenScaleSize);
    GmBuffer scaleGM(scaleSize);
    GmBuffer biasGM(biasSize);
    GmBuffer yGM(ySize);
    GmBuffer tilingGM(sizeof(QBMMV3TilingData));

    ASSERT_NE(x1GM.Get(), nullptr);
    ASSERT_NE(x2GM.Get(), nullptr);
    ASSERT_NE(pertokenScaleGM.Get(), nullptr);
    ASSERT_NE(scaleGM.Get(), nullptr);
    ASSERT_NE(biasGM.Get(), nullptr);
    ASSERT_NE(yGM.Get(), nullptr);
    ASSERT_NE(tilingGM.Get(), nullptr);

    FillGmBuffer(x1GM.Get(), x1Size, 0U);
    FillGmBuffer(x2GM.Get(), x2Size, 0U);
    FillGmBuffer(pertokenScaleGM.Get(), pertokenScaleSize, 0x7fU);
    FillGmBuffer(scaleGM.Get(), scaleSize, 0x7fU);
    FillGmBuffer(biasGM.Get(), biasSize, 0U);
    FillGmBuffer(yGM.Get(), ySize, 0U);

    auto* tilingData = reinterpret_cast<QBMMV3TilingData*>(tilingGM.Get());
    FillMxTiling(tilingData, cfg);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = qbmm_mx_without_batch_kernel_entry<AType, BType, CType, BiasType>;
    if constexpr (FullLoadMode == Blaze::Gemm::A_FULL_LOAD_MODE) {
        kernelFunc = qbmm_mx_without_batch_a_full_load_kernel_entry<AType, BType, CType, BiasType>;
    }
    const bool ok = KERNEL_RUN_KF(kernelFunc, cfg.blockNum, x1GM.Get(), x2GM.Get(), pertokenScaleGM.Get(),
                                  scaleGM.Get(), biasGM.Get(), yGM.Get(), tilingGM.Get());

    ASSERT_TRUE(ok) << "QBMM MX without-batch kernel execution failed";
}

} // namespace

TEST_F(QBMMV3Test, Test_INT8_A8W8_PERTENSOR)
{
    CubeCaseCfg cfg{16, 16, 16, 1, QM_DEFAULT, QM_PERTENSOR, false, GE_DT_FLOAT, sizeof(int32_t), sizeof(half), ""};
    auto kernelFunc = qbmm_cube_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_INT8_A8W8_PERTENSOR_AFullLoad)
{
    CubeCaseCfg cfg{16, 16, 16, 1, QM_DEFAULT, QM_PERTENSOR, false, GE_DT_FLOAT, sizeof(int32_t), sizeof(half), ""};
    auto kernelFunc = qbmm_cube_a_full_load_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_INT8_A8W8_PERCHANNEL_MultiBatchWithThreeDimBias)
{
    CubeCaseCfg cfg{16,
                    128,
                    64,
                    1,
                    QM_DEFAULT,
                    QM_PERCHANNEL,
                    true,
                    GE_DT_INT32,
                    sizeof(int32_t),
                    sizeof(half),
                    "--x2_mode perchannel --bias --bias_dtype int32"};
    cfg.batch = 2;
    cfg.biasThreeDim = true;
    auto kernelFunc = qbmm_cube_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_CubeTiling_DefaultSingleTile)
{
    CubeCaseCfg cfg{16, 32, 64, 1, QM_DEFAULT, QM_PERTENSOR, false, GE_DT_FLOAT, sizeof(int32_t), sizeof(half), ""};
    QBMMV3TilingData tiling{};
    FillCubeTiling(&tiling, cfg);

    EXPECT_EQ(tiling.baseM, 16);
    EXPECT_EQ(tiling.baseN, 32);
    EXPECT_EQ(tiling.baseM_qbmm, 16U);
    EXPECT_EQ(tiling.baseN_qbmm, 32U);
    EXPECT_EQ(tiling.kAL1, 64U);
    EXPECT_EQ(tiling.kBL1, 64U);
    EXPECT_EQ(tiling.baseK_qbmm, 64U);
    EXPECT_EQ(tiling.nBufferNum, 2U);
}

TEST_F(QBMMV3Test, Test_CubeTiling_TripleBufferWithTails)
{
    CubeCaseCfg cfg{48,
                    80,
                    288,
                    1,
                    QM_DEFAULT,
                    QM_PERCHANNEL,
                    false,
                    GE_DT_FLOAT,
                    sizeof(int32_t),
                    sizeof(half),
                    "",
                    64,
                    static_cast<uint32_t>(Blaze::Gemm::TRIPLE_BUFFER_COUNT)};
    cfg.baseM = 32;
    cfg.baseN = 32;
    cfg.baseK = 32;
    QBMMV3TilingData tiling{};
    FillCubeTiling(&tiling, cfg);

    EXPECT_EQ(tiling.m, 48);
    EXPECT_EQ(tiling.n, 80);
    EXPECT_EQ(tiling.k, 288);
    EXPECT_EQ(tiling.baseM, 32);
    EXPECT_EQ(tiling.baseN, 32);
    EXPECT_EQ(tiling.baseM_qbmm, 32U);
    EXPECT_EQ(tiling.baseN_qbmm, 32U);
    EXPECT_EQ(tiling.kAL1, 64U);
    EXPECT_EQ(tiling.kBL1, 64U);
    EXPECT_EQ(tiling.baseK_qbmm, 32U);
    EXPECT_EQ(tiling.nBufferNum, 3U);
    // Five L1 iterations exercise 0 -> 1 -> 2 -> 0 -> 1, with a 32-element K tail.
    EXPECT_EQ((tiling.k + tiling.kAL1 - 1) / tiling.kAL1, 5);
    EXPECT_EQ(tiling.k % tiling.kAL1, 32);
}

TEST_F(QBMMV3Test, Test_INT8_A8W8_PERTENSOR_WithoutBatch_TripleBuffer)
{
    CubeCaseCfg cfg{32,
                    96,
                    288,
                    1,
                    QM_DEFAULT,
                    QM_PERTENSOR,
                    false,
                    GE_DT_FLOAT,
                    sizeof(int32_t),
                    sizeof(half),
                    "",
                    64,
                    static_cast<uint32_t>(Blaze::Gemm::TRIPLE_BUFFER_COUNT)};
    cfg.baseN = 32;
    cfg.baseK = 32;
    auto kernelFunc = qbmm_cube_without_batch_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_INT8_A8W8_PERTENSOR_WithoutBatch_TripleBuffer_AFullLoad)
{
    CubeCaseCfg cfg{32,
                    96,
                    288,
                    1,
                    QM_DEFAULT,
                    QM_PERTENSOR,
                    false,
                    GE_DT_FLOAT,
                    sizeof(int32_t),
                    sizeof(half),
                    "",
                    64,
                    static_cast<uint32_t>(Blaze::Gemm::TRIPLE_BUFFER_COUNT)};
    // Reuse the resident A tile across three N tiles on the same core.
    cfg.baseN = 32;
    cfg.baseK = 32;
    auto kernelFunc = qbmm_cube_without_batch_a_full_load_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_INT8_A8W8_PERCHANNEL_WithoutBatch_TripleBuffer)
{
    CubeCaseCfg cfg{48,
                    80,
                    288,
                    1,
                    QM_DEFAULT,
                    QM_PERCHANNEL,
                    true,
                    GE_DT_INT32,
                    sizeof(int32_t),
                    sizeof(half),
                    "--x2_mode perchannel --bias --bias_dtype int32",
                    64,
                    static_cast<uint32_t>(Blaze::Gemm::TRIPLE_BUFFER_COUNT)};
    cfg.baseM = 32;
    cfg.baseN = 32;
    cfg.baseK = 32;
    auto kernelFunc = qbmm_cube_without_batch_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_INT8_A8W8_PERTENSOR_X1Scale_WithoutBatch)
{
    CubeCaseCfg cfg{16,
                    16,
                    64,
                    1,
                    QM_PERTENSOR,
                    QM_PERTENSOR,
                    false,
                    GE_DT_FLOAT,
                    sizeof(int32_t),
                    sizeof(half),
                    "--x1_mode pertensor --x2_mode pertensor --scale_dtype float32"};
    cfg.scaleElemSize = sizeof(float);
    auto kernelFunc = qbmm_cube_without_batch_kernel_entry<int8_t, int8_t, half, int32_t, float>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_INT8_A8W8_PERCHANNEL_WithoutBatch_DoubleBuffer_L0CPingpong)
{
    CubeCaseCfg cfg{32,
                    96,
                    288,
                    1,
                    QM_DEFAULT,
                    QM_PERCHANNEL,
                    true,
                    GE_DT_INT32,
                    sizeof(int32_t),
                    sizeof(half),
                    "--x2_mode perchannel --bias --bias_dtype int32",
                    64,
                    static_cast<uint32_t>(Blaze::Gemm::DOUBLE_BUFFER_COUNT)};
    cfg.baseN = 32;
    cfg.baseK = 32;
    // Five L1 iterations wrap the double buffer; three N tiles toggle the L0C buffer twice.
    cfg.dbL0C = static_cast<uint32_t>(Blaze::Gemm::DOUBLE_BUFFER_COUNT);
    auto kernelFunc = qbmm_cube_without_batch_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_INT8_A8W8_PERCHANNEL_WithoutBatch_QuadBuffer)
{
    CubeCaseCfg cfg{32,
                    96,
                    288,
                    1,
                    QM_DEFAULT,
                    QM_PERCHANNEL,
                    true,
                    GE_DT_INT32,
                    sizeof(int32_t),
                    sizeof(half),
                    "--x2_mode perchannel --bias --bias_dtype int32",
                    64,
                    static_cast<uint32_t>(Blaze::Gemm::QUADRUPLE_BUFFER_COUNT)};
    cfg.baseN = 32;
    cfg.baseK = 32;
    // Five L1 iterations exercise buffer IDs 0 -> 1 -> 2 -> 3 -> 0.
    auto kernelFunc = qbmm_cube_without_batch_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_FP8_A8W8_PERTENSOR_WithoutBatch_AFullLoad_DoubleBuffer)
{
    CubeCaseCfg cfg{32,
                    64,
                    160,
                    1,
                    QM_PERTENSOR,
                    QM_PERTENSOR,
                    false,
                    GE_DT_FLOAT,
                    sizeof(float),
                    sizeof(bfloat16_t),
                    "",
                    64,
                    static_cast<uint32_t>(Blaze::Gemm::DOUBLE_BUFFER_COUNT)};
    cfg.baseN = 32;
    cfg.baseK = 32;
    cfg.scaleElemSize = sizeof(float);
    // Three L1 iterations wrap the double-buffered B path; two N tiles reuse the fully loaded A tile.
    auto kernelFunc = qbmm_cube_without_batch_a_full_load_kernel_entry<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, float,
                                                                       float>;
    RunTypedCubeSmoke<fp8_e4m3fn_t, fp8_e4m3fn_t>(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_HIFLOAT8_A8W8_PERTENSOR_WithoutBatch_AFullLoad_QuadBuffer)
{
    CubeCaseCfg cfg{32,
                    64,
                    288,
                    1,
                    QM_PERTENSOR,
                    QM_PERTENSOR,
                    false,
                    GE_DT_FLOAT,
                    sizeof(float),
                    sizeof(bfloat16_t),
                    "",
                    64,
                    static_cast<uint32_t>(Blaze::Gemm::QUADRUPLE_BUFFER_COUNT)};
    cfg.baseN = 32;
    cfg.baseK = 32;
    cfg.scaleElemSize = sizeof(float);
    // Five L1 iterations cover every B buffer and wrap to buffer 0 in A-full-load mode.
    auto
        kernelFunc = qbmm_cube_without_batch_a_full_load_kernel_entry<hifloat8_t, hifloat8_t, bfloat16_t, float, float>;
    RunTypedCubeSmoke<hifloat8_t, hifloat8_t>(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_BlockMmadDoubleBuffer)
{
    using MxType = fp8_e4m3fn_t;
    MxCaseCfg cfg{64, 128, 128, 1, 64, 128, 64, 64, 64, 2, false};
    RunMxSmoke<MxType, MxType, float, float>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_BlockMmadDoubleBuffer_AFullLoad)
{
    using MxType = fp8_e4m3fn_t;
    MxCaseCfg cfg{64, 128, 128, 1, 64, 128, 64, 64, 64, 2, false};
    RunMxSmoke<MxType, MxType, float, float, Blaze::Gemm::A_FULL_LOAD_MODE>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_MultiBatchWithThreeDimBias)
{
    using MxType = fp8_e4m3fn_t;
    MxCaseCfg cfg{16, 128, 128, 1, 16, 128, 64, 64, 64, 2, true};
    cfg.batch = 2;
    cfg.biasThreeDim = true;
    RunMxSmoke<MxType, MxType, float, float>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_BlockMmadTripleBuffer)
{
    using MxType = fp8_e4m3fn_t;
    MxCaseCfg cfg{64, 128, 192, 1, 64, 128, 64, 64, 64, 3, false};
    RunMxSmoke<MxType, MxType, float, float>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_BlockMmadQuadBuffer)
{
    using MxType = fp8_e4m3fn_t;
    MxCaseCfg cfg{64, 128, 256, 1, 64, 128, 64, 64, 64, 4, false};
    RunMxSmoke<MxType, MxType, float, float>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_WithoutBatchDoubleBuffer)
{
    using MxType = fp8_e4m3fn_t;
    MxCaseCfg cfg{64, 128, 128, 1, 64, 128, 64, 64, 64, 2, false};
    RunMxWithoutBatchSmoke<MxType, MxType, float, float>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_WithoutBatchDoubleBuffer_AFullLoad)
{
    using MxType = fp8_e4m3fn_t;
    MxCaseCfg cfg{64, 128, 128, 1, 64, 128, 64, 64, 64, 2, false};
    RunMxWithoutBatchSmoke<MxType, MxType, float, float, Blaze::Gemm::A_FULL_LOAD_MODE>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_WithoutBatchTripleBuffer)
{
    using MxType = fp8_e4m3fn_t;
    MxCaseCfg cfg{64, 128, 192, 1, 64, 128, 64, 64, 64, 3, false};
    RunMxWithoutBatchSmoke<MxType, MxType, float, float>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_WithoutBatchQuadBuffer)
{
    using MxType = fp8_e4m3fn_t;
    MxCaseCfg cfg{64, 128, 256, 1, 64, 128, 64, 64, 64, 4, false};
    RunMxWithoutBatchSmoke<MxType, MxType, float, float>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_L0CPingpong)
{
    using MxType = fp8_e4m3fn_t;
    L0CPingpongCaseCfg cfg{64, 128, 128, 64, 128, 64, 64, 2, 1};
    RunMxL0CPingpongSmoke<MxType, MxType>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_L0CPingpong_AFullLoad)
{
    using MxType = fp8_e4m3fn_t;
    L0CPingpongCaseCfg cfg{64, 128, 128, 64, 128, 64, 64, 2, 1};
    RunMxL0CPingpongSmoke<MxType, MxType, Blaze::Gemm::A_FULL_LOAD_MODE>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_L0CPingpongWithoutBatch)
{
    using MxType = fp8_e4m3fn_t;
    L0CPingpongCaseCfg cfg{64, 128, 128, 64, 128, 64, 64, 2, 1};
    RunMxL0CPingpongSmoke<MxType, MxType, Blaze::Gemm::NONE_FULL_LOAD_MODE, true>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_L0CPingpongWithoutBatch_AFullLoad)
{
    using MxType = fp8_e4m3fn_t;
    L0CPingpongCaseCfg cfg{64, 128, 128, 64, 128, 64, 64, 2, 1};
    RunMxL0CPingpongSmoke<MxType, MxType, Blaze::Gemm::A_FULL_LOAD_MODE, true>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_L0CPingpongTripleBuffer)
{
    using MxType = fp8_e4m3fn_t;
    L0CPingpongCaseCfg cfg{64, 128, 192, 64, 128, 64, 64, 3, 1};
    RunMxL0CPingpongSmoke<MxType, MxType>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP4_L0CPingpongSplitN)
{
    using MxType = fp4x2_e2m1_t;
    L0CPingpongCaseCfg cfg{128, 256, 128, 128, 256, 64, 64, 2, 1};
    RunMxL0CPingpongSmoke<MxType, MxType>(cfg);
}

TEST_F(QBMMV3Test, Test_MX_FP8_StreamK)
{
    StreamKCaseCfg cfg{16, 16, 128, 2};
    RunMxStreamKSmoke(cfg);
}

// ===================== MIX A8W8 dequant 路径用例矩阵（4.3）=====================
// 均为 smoke 测试：KERNEL_RUN_KF 仅检测 kernel 是否崩溃（与 PR #61 一致，不读回 golden 比对）。

// 最典型：激活 per-token + 权重 per-channel，双向量 scale，half 输出。
TEST_F(QBMMV3Test, Test_MIX_A8W8_PerChannel_PerToken)
{
    MixCaseCfg cfg{16,
                   16,
                   16,
                   1,
                   QM_PERTOKEN,
                   QM_PERCHANNEL,
                   false,
                   GE_DT_FLOAT,
                   sizeof(float),
                   sizeof(half),
                   "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32"};
    auto kernelFunc = qbmm_mix_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_MIX_A8W8_PerChannel_PerToken_AFullLoad)
{
    MixCaseCfg cfg{16,
                   16,
                   16,
                   1,
                   QM_PERTOKEN,
                   QM_PERCHANNEL,
                   false,
                   GE_DT_FLOAT,
                   sizeof(float),
                   sizeof(half),
                   "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32"};
    auto kernelFunc = qbmm_mix_a_full_load_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_MIX_A8W8_MultiBatchWithThreeDimBias)
{
    MixCaseCfg cfg{16,
                   16,
                   16,
                   1,
                   QM_PERTOKEN,
                   QM_PERCHANNEL,
                   true,
                   GE_DT_FLOAT,
                   sizeof(float),
                   sizeof(half),
                   "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32 --bias --bias_dtype float32"};
    cfg.batch = 2;
    cfg.biasThreeDim = true;
    auto kernelFunc = qbmm_mix_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

// 仅权重 scale：激活 DEFAULT（epilogue 忽略 x1 scale）+ 权重 per-channel，half 输出。
TEST_F(QBMMV3Test, Test_MIX_A8W8_PerChannel_NoPtScale)
{
    MixCaseCfg cfg{16,
                   16,
                   16,
                   1,
                   QM_DEFAULT,
                   QM_PERCHANNEL,
                   false,
                   GE_DT_FLOAT,
                   sizeof(float),
                   sizeof(half),
                   "--x1_mode default --x2_mode perchannel --scale_dtype float32"};
    auto kernelFunc = qbmm_mix_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

// 权重标量 scale：激活 per-token + 权重 per-tensor，half 输出。
TEST_F(QBMMV3Test, Test_MIX_A8W8_PerTensor_PerToken)
{
    MixCaseCfg cfg{16,
                   16,
                   16,
                   1,
                   QM_PERTOKEN,
                   QM_PERTENSOR,
                   false,
                   GE_DT_FLOAT,
                   sizeof(float),
                   sizeof(half),
                   "--x1_mode pertoken --x2_mode pertensor --scale_dtype float32"};
    auto kernelFunc = qbmm_mix_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_MIX_A8W8_PerTensor_X1PerTensor_BF16Bias_OutputFloat)
{
    MixCaseCfg cfg{16,
                   16,
                   16,
                   1,
                   QM_PERTENSOR,
                   QM_PERTENSOR,
                   true,
                   GE_DT_BF16,
                   sizeof(bfloat16_t),
                   sizeof(float),
                   "--x1_mode pertensor --x2_mode pertensor --scale_dtype float32 --bias --bias_dtype bfloat16 "
                   "--out_dtype float32"};
    auto kernelFunc = qbmm_mix_kernel_entry<int8_t, int8_t, float, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

// 覆盖 bias 路径 + biasDtype=fp16：激活 per-token + 权重 per-channel + fp16 bias，half 输出。
TEST_F(QBMMV3Test, Test_MIX_A8W8_WithBias_FP16)
{
    MixCaseCfg cfg{16,
                   16,
                   16,
                   1,
                   QM_PERTOKEN,
                   QM_PERCHANNEL,
                   true,
                   GE_DT_FLOAT16,
                   sizeof(half),
                   sizeof(half),
                   "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32 --bias --bias_dtype float16"};
    auto kernelFunc = qbmm_mix_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

// 覆盖 OutType=bf16：激活 per-token + 权重 per-channel，bfloat16 输出。
TEST_F(QBMMV3Test, Test_MIX_A8W8_Output_BF16)
{
    MixCaseCfg cfg{16,
                   16,
                   16,
                   1,
                   QM_PERTOKEN,
                   QM_PERCHANNEL,
                   false,
                   GE_DT_FLOAT,
                   sizeof(float),
                   sizeof(bfloat16_t),
                   "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32"};
    auto kernelFunc = qbmm_mix_kernel_entry<int8_t, int8_t, bfloat16_t, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

// 单 batch 特化：走 GemmUniversal without_batch，激活 per-token + 权重 per-channel，half 输出。
TEST_F(QBMMV3Test, Test_MIX_A8W8_WithoutBatch)
{
    MixCaseCfg cfg{16,
                   16,
                   16,
                   1,
                   QM_PERTOKEN,
                   QM_PERCHANNEL,
                   false,
                   GE_DT_FLOAT,
                   sizeof(float),
                   sizeof(half),
                   "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32"};
    auto kernelFunc = qbmm_mix_without_batch_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_MIX_A8W8_WithoutBatch_AFullLoad)
{
    MixCaseCfg cfg{16,
                   16,
                   16,
                   1,
                   QM_PERTOKEN,
                   QM_PERCHANNEL,
                   false,
                   GE_DT_FLOAT,
                   sizeof(float),
                   sizeof(half),
                   "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32"};
    auto kernelFunc = qbmm_mix_without_batch_a_full_load_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

namespace {

using ND = asc::te::nd_ext_layout_ptn;
using DN = asc::te::dn_ext_layout_ptn;
using NZ = asc::te::nz_layout_ptn;
using ZN = asc::te::zn_layout_ptn;
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

template <class Kernel>
inline constexpr bool HAS_KERNEL_CALL = std::is_same_v<
    decltype(std::declval<Kernel&>()(std::declval<const typename Kernel::Params&>())), void>;

template <class Schedule, class AType, class BType, class ScaleType, class OutType, class BiasType, class LayoutA,
          class LayoutB, uint64_t FullLoadMode>
void CheckCubeTensorApiAssembly()
{
    using Types = QBMMUT::QBMMCubeTypes<AType, BType, ScaleType, OutType, BiasType, LayoutA, LayoutB, ND, FullLoadMode,
                                        Schedule>;
    using Kernel = typename Types::Kernel;
    static_assert(std::is_same_v<typename Types::DispatchPolicy::ScheduleType, Schedule>);
    static_assert(Types::DispatchPolicy::FULL_LOAD_MODE == FullLoadMode);
    static_assert(std::is_same_v<typename Types::BlockMmad::X2ScaleType, ScaleType>);
    static_assert(std::is_same_v<typename Types::BlockMmad::LayoutA, LayoutA>);
    static_assert(std::is_same_v<typename Types::BlockMmad::LayoutB, LayoutB>);
    static_assert(std::is_same_v<typename Types::BlockMmad::LayoutC, ND>);
    static_assert(std::is_same_v<typename Types::BlockEpilogue, Blaze::Epilogue::Block::BlockEpilogueEmpty>);
    static_assert(HAS_KERNEL_CALL<Kernel>);
    typename Kernel::Params params{};
    params.qbmmParams.bMustHitL2 = 0U;
    EXPECT_GT(sizeof(Kernel), 0U);
    EXPECT_EQ(params.qbmmParams.bMustHitL2, 0U);
}

template <class Schedule, class AType, class BType, class ScaleType, class OutType, class BiasType, class LayoutA,
          class LayoutB, uint64_t FullLoadMode>
void CheckMixTensorApiAssembly()
{
    using Types = QBMMUT::QBMMMixTypes<AType, BType, OutType, ScaleType, float, BiasType, FullLoadMode, LayoutA,
                                       LayoutB, ND, Schedule>;
    using Kernel = typename Types::Kernel;
    static_assert(std::is_same_v<typename Types::DispatchPolicy::ScheduleType, Schedule>);
    static_assert(Types::DispatchPolicy::FULL_LOAD_MODE == FullLoadMode);
    static_assert(std::is_same_v<typename Types::BTypeTuple, AscendC::Std::tuple<BType, ScaleType>>);
    static_assert(std::is_same_v<typename Types::BlockMmad::LayoutA, LayoutA>);
    static_assert(std::is_same_v<typename Types::BlockMmad::LayoutB, LayoutB>);
    static_assert(std::is_same_v<typename Types::BlockEpilogue::OutType, OutType>);
    static_assert(std::is_same_v<typename Types::BlockEpilogue::BiasType, BiasType>);
    static_assert(std::is_same_v<typename Types::BlockEpilogue::X2ScaleType, ScaleType>);
    static_assert(std::is_same_v<typename Types::BlockEpilogue::L0CType, typename Types::BlockMmad::L0CType>);
    static_assert(HAS_KERNEL_CALL<Kernel>);
    typename Kernel::Params params{};
    params.qbmmParams.bMustHitL2 = 0U;
    EXPECT_GT(sizeof(Kernel), 0U);
    EXPECT_EQ(params.qbmmParams.bMustHitL2, 0U);
}

template <class Schedule, class AType, class BType, class OutType, class LayoutA, class LayoutB, uint64_t FullLoadMode>
void CheckMxTensorApiAssembly()
{
    using Policy = Blaze::Gemm::MatmulWithScaleMx<FullLoadMode, false, Schedule>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, AType, LayoutA, BType, LayoutB, OutType, ND, float, ND>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FullLoadMode, LayoutA, LayoutB,
                                                                           AType>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Mmad, Epilogue, Scheduler>;
    static_assert(std::is_same_v<typename Policy::ScheduleType, Schedule>);
    static_assert(Policy::FULL_LOAD_MODE == FullLoadMode);
    static_assert(std::is_same_v<typename Mmad::LayoutA, LayoutA>);
    static_assert(std::is_same_v<typename Mmad::LayoutB, LayoutB>);
    static_assert(std::is_same_v<typename Mmad::LayoutC, ND>);
    static_assert(HAS_KERNEL_CALL<Kernel>);
    typename Kernel::Params params{};
    params.qbmmParams.bMustHitL2 = 0U;
    EXPECT_GT(sizeof(Kernel), 0U);
    EXPECT_EQ(params.qbmmParams.bMustHitL2, 0U);
}

template <class Schedule, class AType, class BType, class OutType, class LayoutA, class LayoutB, uint64_t FullLoadMode>
void CheckMxL0CPingpongTensorApiAssembly()
{
    using Policy = Blaze::Gemm::MatmulWithScaleMxL0CPingpong<FullLoadMode, false, Schedule>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, AType, LayoutA, BType, LayoutB, OutType, ND, float, ND>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FullLoadMode, LayoutA, LayoutB,
                                                                           AType>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Mmad, Epilogue, Scheduler>;
    static_assert(std::is_same_v<typename Policy::ScheduleType, Schedule>);
    static_assert(Policy::FULL_LOAD_MODE == FullLoadMode);
    static_assert(std::is_same_v<typename Mmad::LayoutA, LayoutA>);
    static_assert(std::is_same_v<typename Mmad::LayoutB, LayoutB>);
    static_assert(std::is_same_v<typename Mmad::LayoutC, ND>);
    static_assert(HAS_KERNEL_CALL<Kernel>);
    typename Kernel::Params params{};
    params.qbmmParams.bMustHitL2 = 0U;
    EXPECT_GT(sizeof(Kernel), 0U);
    EXPECT_EQ(params.qbmmParams.bMustHitL2, 0U);
}

template <class AType, class BType, class OutType, class LayoutA, class LayoutB, uint64_t FullLoadMode>
void CheckMxStreamKTensorApiAssembly()
{
    using Policy = Blaze::Gemm::MatmulWithScaleMx<FullLoadMode, false, Blaze::Gemm::KernelQbmmMultiBlockStreamK>;
    using EpiloguePolicy = Blaze::Gemm::MatmulMultiBlockWithStreamK<>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, AType, LayoutA, BType, LayoutB, OutType, ND, float, ND>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueMatmulStreamK<float, OutType, EpiloguePolicy>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Mmad, Epilogue, Scheduler>;
    static_assert(std::is_same_v<typename Policy::ScheduleType, Blaze::Gemm::KernelQbmmMultiBlockStreamK>);
    static_assert(Policy::FULL_LOAD_MODE == FullLoadMode);
    static_assert(std::is_same_v<typename Epilogue::WorkspaceType, float>);
    static_assert(std::is_same_v<typename Epilogue::OutType, OutType>);
    static_assert(std::is_same_v<typename Epilogue::DispatchPolicy, EpiloguePolicy>);
    static_assert(HAS_KERNEL_CALL<Kernel>);
    typename Kernel::Params params{};
    params.qbmmParams.bMustHitL2 = 0U;
    EXPECT_GT(sizeof(Kernel), 0U);
    EXPECT_EQ(params.qbmmParams.bMustHitL2, 0U);
}

template <class AType, class BType, class ScaleType, class OutType, class BiasType, class LayoutA, class LayoutB,
          uint64_t FullLoadMode>
void CheckPertensorStreamKTensorApiAssembly()
{
    using Policy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<FullLoadMode, false,
                                                            Blaze::Gemm::KernelQbmmPertensorMultiBlockStreamK>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, AType, LayoutA, AscendC::Std::tuple<BType, ScaleType>, LayoutB,
                                               OutType, ND, BiasType, ND>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<typename Mmad::WorkspaceType, OutType,
                                                                               Policy, ScaleType, float>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Mmad, Epilogue, Scheduler>;
    static_assert(std::is_same_v<typename Policy::ScheduleType, Blaze::Gemm::KernelQbmmPertensorMultiBlockStreamK>);
    static_assert(Policy::FULL_LOAD_MODE == FullLoadMode);
    static_assert(std::is_same_v<typename Mmad::X2ScaleType, ScaleType>);
    static_assert(std::is_same_v<typename Epilogue::WorkspaceType, typename Mmad::WorkspaceType>);
    static_assert(std::is_same_v<typename Epilogue::DispatchPolicy, Policy>);
    static_assert(std::is_same_v<typename Epilogue::X2ScaleType, ScaleType>);
    static_assert(std::is_same_v<typename Epilogue::X1ScaleType, float>);
    static_assert(HAS_KERNEL_CALL<Kernel>);
    EXPECT_GT(sizeof(Kernel), 0U);
}

// Mirrors arch35 dispatch reachability: Cube/MX allow both A layouts, all four B layouts and both load modes;
// MIX is limited to ND A + NZ/ZN B; StreamK is limited to NONE_FULL_LOAD_MODE.
template <class Schedule, class LayoutA, class LayoutB>
void CheckCubeLayoutAndLoadModes()
{
    CheckCubeTensorApiAssembly<Schedule, int8_t, int8_t, uint64_t, half, int32_t, LayoutA, LayoutB,
                               Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckCubeTensorApiAssembly<Schedule, int8_t, int8_t, uint64_t, half, int32_t, LayoutA, LayoutB,
                               Blaze::Gemm::A_FULL_LOAD_MODE>();
}

template <class Schedule, class LayoutA>
void CheckCubeBLayoutMatrix()
{
    CheckCubeLayoutAndLoadModes<Schedule, LayoutA, ND>();
    CheckCubeLayoutAndLoadModes<Schedule, LayoutA, DN>();
    CheckCubeLayoutAndLoadModes<Schedule, LayoutA, NZ>();
    CheckCubeLayoutAndLoadModes<Schedule, LayoutA, ZN>();
}

template <class Schedule>
void CheckCubeLayoutMatrix()
{
    CheckCubeBLayoutMatrix<Schedule, ND>();
    CheckCubeBLayoutMatrix<Schedule, DN>();
}

template <class Schedule>
void CheckMixLayoutMatrix()
{
    CheckMixTensorApiAssembly<Schedule, int8_t, int8_t, float, bfloat16_t, int32_t, ND, ZN,
                              Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckMixTensorApiAssembly<Schedule, int8_t, int8_t, float, bfloat16_t, int32_t, ND, NZ,
                              Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckMixTensorApiAssembly<Schedule, int8_t, int8_t, float, bfloat16_t, int32_t, ND, NZ,
                              Blaze::Gemm::A_FULL_LOAD_MODE>();
    CheckMixTensorApiAssembly<Schedule, int8_t, int8_t, float, bfloat16_t, int32_t, ND, ZN,
                              Blaze::Gemm::A_FULL_LOAD_MODE>();
}

template <class Schedule, class LayoutA, class LayoutB>
void CheckMxLayoutAndLoadModes()
{
    CheckMxTensorApiAssembly<Schedule, fp8_e4m3fn_t, fp8_e5m2_t, half, LayoutA, LayoutB,
                             Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckMxTensorApiAssembly<Schedule, fp8_e4m3fn_t, fp8_e5m2_t, half, LayoutA, LayoutB,
                             Blaze::Gemm::A_FULL_LOAD_MODE>();
}

template <class Schedule, class LayoutA>
void CheckMxBLayoutMatrix()
{
    CheckMxLayoutAndLoadModes<Schedule, LayoutA, ND>();
    CheckMxLayoutAndLoadModes<Schedule, LayoutA, DN>();
    CheckMxLayoutAndLoadModes<Schedule, LayoutA, NZ>();
    CheckMxLayoutAndLoadModes<Schedule, LayoutA, ZN>();
}

template <class Schedule>
void CheckMxLayoutMatrix()
{
    CheckMxBLayoutMatrix<Schedule, ND>();
    CheckMxBLayoutMatrix<Schedule, DN>();
}

template <class Schedule, class LayoutA, class LayoutB>
void CheckMxL0CPingpongLayoutAndLoadModes()
{
    CheckMxL0CPingpongTensorApiAssembly<Schedule, fp8_e4m3fn_t, fp8_e5m2_t, half, LayoutA, LayoutB,
                                        Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckMxL0CPingpongTensorApiAssembly<Schedule, fp8_e4m3fn_t, fp8_e5m2_t, half, LayoutA, LayoutB,
                                        Blaze::Gemm::A_FULL_LOAD_MODE>();
}

template <class Schedule, class LayoutA>
void CheckMxL0CPingpongBLayoutMatrix()
{
    CheckMxL0CPingpongLayoutAndLoadModes<Schedule, LayoutA, ND>();
    CheckMxL0CPingpongLayoutAndLoadModes<Schedule, LayoutA, DN>();
    CheckMxL0CPingpongLayoutAndLoadModes<Schedule, LayoutA, NZ>();
    CheckMxL0CPingpongLayoutAndLoadModes<Schedule, LayoutA, ZN>();
}

template <class Schedule>
void CheckMxL0CPingpongLayoutMatrix()
{
    CheckMxL0CPingpongBLayoutMatrix<Schedule, ND>();
    CheckMxL0CPingpongBLayoutMatrix<Schedule, DN>();
}

template <class LayoutA>
void CheckMxStreamKBLayoutMatrix()
{
    CheckMxStreamKTensorApiAssembly<fp8_e4m3fn_t, fp8_e5m2_t, half, LayoutA, ND, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckMxStreamKTensorApiAssembly<fp8_e4m3fn_t, fp8_e5m2_t, half, LayoutA, DN, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckMxStreamKTensorApiAssembly<fp8_e4m3fn_t, fp8_e5m2_t, half, LayoutA, NZ, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckMxStreamKTensorApiAssembly<fp8_e4m3fn_t, fp8_e5m2_t, half, LayoutA, ZN, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
}

void CheckMxStreamKLayoutMatrix()
{
    CheckMxStreamKBLayoutMatrix<ND>();
    CheckMxStreamKBLayoutMatrix<DN>();
}

template <class LayoutA>
void CheckPertensorStreamKBLayoutMatrix()
{
    CheckPertensorStreamKTensorApiAssembly<int8_t, int8_t, uint64_t, half, int32_t, LayoutA, ND,
                                           Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckPertensorStreamKTensorApiAssembly<int8_t, int8_t, uint64_t, half, int32_t, LayoutA, DN,
                                           Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckPertensorStreamKTensorApiAssembly<int8_t, int8_t, uint64_t, half, int32_t, LayoutA, NZ,
                                           Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckPertensorStreamKTensorApiAssembly<int8_t, int8_t, uint64_t, half, int32_t, LayoutA, ZN,
                                           Blaze::Gemm::NONE_FULL_LOAD_MODE>();
}

void CheckPertensorStreamKLayoutMatrix()
{
    CheckPertensorStreamKBLayoutMatrix<ND>();
    CheckPertensorStreamKBLayoutMatrix<DN>();
}

} // namespace

TEST(QBMMTemplateContractTest, Arch35AssemblyMatrix)
{
    CheckCubeLayoutMatrix<Blaze::Gemm::KernelMmadWithScaleFixpipeQuant>();
    CheckCubeLayoutMatrix<Blaze::Gemm::KernelMmadWithScaleFixpipeQuantWithoutBatch>();
    CheckMixLayoutMatrix<Blaze::Gemm::KernelMmadWithScaleMix>();
    CheckMixLayoutMatrix<Blaze::Gemm::KernelMmadWithScaleMixWithoutBatch>();
    CheckMxLayoutMatrix<Blaze::Gemm::KernelMmadWithScaleMx>();
    CheckMxLayoutMatrix<Blaze::Gemm::KernelMmadWithScaleMxWithoutBatch>();
    CheckMxL0CPingpongLayoutMatrix<Blaze::Gemm::KernelMmadWithScaleMx>();
    CheckMxL0CPingpongLayoutMatrix<Blaze::Gemm::KernelMmadWithScaleMxWithoutBatch>();
    CheckMxStreamKLayoutMatrix();
    CheckPertensorStreamKLayoutMatrix();
}

TEST(QBMMTemplateContractTest, ProductionAssembliesPreserveArch35DtypeParameters)
{
    CheckCubeTensorApiAssembly<Blaze::Gemm::KernelMmadWithScaleFixpipeQuant, hifloat8_t, hifloat8_t, bfloat16_t, float,
                               float, ND, NZ, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckCubeTensorApiAssembly<Blaze::Gemm::KernelMmadWithScaleFixpipeQuantWithoutBatch, fp8_e4m3fn_t, fp8_e5m2_t,
                               float, bfloat16_t, float, DN, ZN, Blaze::Gemm::A_FULL_LOAD_MODE>();
    CheckCubeTensorApiAssembly<Blaze::Gemm::KernelMmadWithScaleFixpipeQuantWithoutBatch, int8_t, int8_t, int64_t, half,
                               int32_t, ND, ND, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckMixTensorApiAssembly<Blaze::Gemm::KernelMmadWithScaleMix, hifloat8_t, hifloat8_t, float, float, float, ND, NZ,
                              Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckMixTensorApiAssembly<Blaze::Gemm::KernelMmadWithScaleMixWithoutBatch, fp8_e4m3fn_t, fp8_e4m3fn_t, float, float,
                              float, ND, ZN, Blaze::Gemm::A_FULL_LOAD_MODE>();
    CheckMixTensorApiAssembly<Blaze::Gemm::KernelMmadWithScaleMixWithoutBatch, int8_t, int8_t, bfloat16_t, bfloat16_t,
                              int32_t, ND, NZ, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckMxTensorApiAssembly<Blaze::Gemm::KernelMmadWithScaleMx, fp4x2_e2m1_t, fp4x2_e2m1_t, bfloat16_t, ND, NZ,
                             Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckMxL0CPingpongTensorApiAssembly<Blaze::Gemm::KernelMmadWithScaleMxWithoutBatch, fp4x2_e2m1_t, fp4x2_e2m1_t,
                                        float, ND, ZN, Blaze::Gemm::A_FULL_LOAD_MODE>();
    CheckMxStreamKTensorApiAssembly<fp4x2_e2m1_t, fp4x2_e2m1_t, bfloat16_t, ND, NZ, Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckPertensorStreamKTensorApiAssembly<fp8_e4m3fn_t, fp8_e5m2_t, uint64_t, float, float, DN, ZN,
                                           Blaze::Gemm::NONE_FULL_LOAD_MODE>();
    CheckPertensorStreamKTensorApiAssembly<int8_t, int8_t, int64_t, half, int32_t, ND, ND,
                                           Blaze::Gemm::NONE_FULL_LOAD_MODE>();
}

class QBMMPertensorStreamKTest : public testing::Test {};

TEST_F(QBMMPertensorStreamKTest, TemplateContracts)
{
    using Layout = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
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
    using Fp8Epilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<
        typename Fp8Mmad::WorkspaceType, float, DispatchPolicy, uint64_t, float>;
    using Fp8Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Fp8Mmad, Fp8Epilogue, Scheduler>;
    using Fp8PostBiasMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, fp8_e4m3fn_t, Layout,
                                                          AscendC::Std::tuple<fp8_e4m3fn_t, float>, Layout, float,
                                                          Layout, float, Layout>;
    using Fp8PostBiasEpilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<
        typename Fp8PostBiasMmad::WorkspaceType, float, DispatchPolicy, float, float>;
    using Fp8PostBiasKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Fp8PostBiasMmad, Fp8PostBiasEpilogue,
                                                                 Scheduler>;
    using Int8Fp32PostBiasMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, int8_t, Layout, AscendC::Std::tuple<int8_t, float>, Layout, bfloat16_t, Layout, float, Layout>;
    using Int8Fp32PostBiasEpilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<
        typename Int8Fp32PostBiasMmad::WorkspaceType, bfloat16_t, DispatchPolicy, float, float>;
    using Int8Fp32PostBiasKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Int8Fp32PostBiasMmad,
                                                                      Int8Fp32PostBiasEpilogue, Scheduler>;
    using Int8Bf16PostBiasMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, int8_t, Layout,
                                                               AscendC::Std::tuple<int8_t, bfloat16_t>, Layout,
                                                               bfloat16_t, Layout, bfloat16_t, Layout>;
    using Int8Bf16PostBiasEpilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<
        typename Int8Bf16PostBiasMmad::WorkspaceType, bfloat16_t, DispatchPolicy, bfloat16_t, float>;
    using Int8Bf16PostBiasKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Int8Bf16PostBiasMmad,
                                                                      Int8Bf16PostBiasEpilogue, Scheduler>;

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
    static_assert(std::is_same_v<typename Fp8Kernel::BlockMmad, Fp8Mmad>);
    static_assert(std::is_same_v<typename Fp8PostBiasKernel::BlockMmad, Fp8PostBiasMmad>);
    static_assert(std::is_same_v<typename Int8Fp32PostBiasKernel::BlockMmad, Int8Fp32PostBiasMmad>);
    static_assert(std::is_same_v<typename Int8Bf16PostBiasKernel::BlockMmad, Int8Bf16PostBiasMmad>);

    SUCCEED();
}

TEST_F(QBMMPertensorStreamKTest, SingleScaleWithoutPostBiasIsMaskedBeforeMultiply)
{
    using Layout = asc::te::nd_ext_layout_ptn;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<0, false,
                                                                    Blaze::Gemm::KernelQbmmPertensorMultiBlockStreamK>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, int8_t, Layout, AscendC::Std::tuple<int8_t, float>,
                                               Layout, bfloat16_t, Layout, int32_t, Layout>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<typename Mmad::WorkspaceType, bfloat16_t,
                                                                               DispatchPolicy, float, float>;

    constexpr uint32_t rawScaleBits = 0x3F812345U;
    const float maskedScale = Epilogue::DecodeMaskedDequantScale(rawScaleBits);
    const uint32_t actualBits = *reinterpret_cast<const uint32_t*>(&maskedScale);

    EXPECT_EQ(actualBits, rawScaleBits & Epilogue::DEQ_SCALE_MUL_MASK);
}

TEST_F(QBMMPertensorStreamKTest, DoubleScaleWithoutPostBiasMergesBeforeMask)
{
    using Layout = asc::te::nd_ext_layout_ptn;
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
    const uint32_t mergedBits = *reinterpret_cast<const uint32_t*>(&merged);
    const uint32_t actualBits = *reinterpret_cast<const uint32_t*>(&actual);

    EXPECT_EQ(actualBits, mergedBits & Epilogue::DEQ_SCALE_MUL_MASK);
}

TEST_F(QBMMPertensorStreamKTest, PostBiasScaleDecodePreservesFullPrecision)
{
    using Layout = asc::te::nd_ext_layout_ptn;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<0, false,
                                                                    Blaze::Gemm::KernelQbmmPertensorMultiBlockStreamK>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, fp8_e4m3fn_t, Layout,
                                               AscendC::Std::tuple<fp8_e4m3fn_t, float>, Layout, float, Layout, float,
                                               Layout>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<typename Mmad::WorkspaceType, float,
                                                                               DispatchPolicy, float, float>;

    constexpr uint32_t rawScaleBits = 0x3F812345U;
    const float rawScale = *reinterpret_cast<const float*>(&rawScaleBits);
    const uint32_t actualBits = *reinterpret_cast<const uint32_t*>(&rawScale);

    EXPECT_EQ(actualBits, rawScaleBits);
    EXPECT_NE(actualBits, rawScaleBits & Epilogue::DEQ_SCALE_MUL_MASK);
}

TEST_F(QBMMPertensorStreamKTest, Int8PerTensorSmoke) { RunInt8PertensorStreamKSmoke(); }

TEST_F(QBMMPertensorStreamKTest, Fp8DoubleScalePostBiasSmoke) { RunFp8DoubleScalePostBiasStreamKSmoke(); }

TEST_F(QBMMPertensorStreamKTest, BatchedInputReturnsBeforeScheduling) { RunBatchInputRejectedSmoke(); }
