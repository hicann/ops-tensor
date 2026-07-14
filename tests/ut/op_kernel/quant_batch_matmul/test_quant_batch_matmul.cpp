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
#include "gtest/gtest.h"
#include "blaze_kernel_stub.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "kernel_operator.h"

#include "qbmm_cube.h"
#include "qbmm_mix.h"
#include "qbmm_mx.h"
#include "qbmm_mx_l0c_pingpong.h"
#include "qbmm_streamk.h"

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
constexpr size_t STREAMK_WORKSPACE_OVERHEAD = 20UL * 1024UL * 1024UL;

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

size_t GetMxScaleKLen(int64_t k)
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

void FillGmBuffer(GM_ADDR addr, size_t size, uint8_t value)
{
    auto* buffer = reinterpret_cast<uint8_t*>(addr);
    std::fill_n(buffer, size, value);
}

class GmBuffer {
public:
    explicit GmBuffer(size_t size) : addr_(reinterpret_cast<GM_ADDR>(AscendC::GmAlloc(size)))
    {
    }

    ~GmBuffer()
    {
        if (addr_ != nullptr) {
            AscendC::GmFree(reinterpret_cast<void*>(addr_));
        }
    }

    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;

    GM_ADDR Get() const
    {
        return addr_;
    }

private:
    GM_ADDR addr_{nullptr};
};

// 填充单 tile MxNxK 场景的 tiling（其余字段沿用 fixpipe 用例默认）。
void FillCubeTiling(QBMMV3TilingData* tilingData, const CubeCaseCfg& caseCfg)
{
    tilingData->m = caseCfg.M;
    tilingData->n = caseCfg.N;
    tilingData->k = caseCfg.K;
    tilingData->b = 1;

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
    tilingData->biasDtype = caseCfg.biasDtype;
}

void FillMixTiling(QBMMV3TilingData* tilingData, const MixCaseCfg& caseCfg)
{
    tilingData->m = caseCfg.M;
    tilingData->n = caseCfg.N;
    tilingData->k = caseCfg.K;
    tilingData->b = 1;

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
    tilingData->biasDtype = caseCfg.biasDtype;
}

void FillMxTiling(QBMMV3TilingData* tilingData, const MxCaseCfg& caseCfg)
{
    tilingData->m = caseCfg.M;
    tilingData->n = caseCfg.N;
    tilingData->k = caseCfg.K;
    tilingData->b = 1;

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
    tilingData->biasDtype = GE_DT_FLOAT;
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
void RunCubeSmoke(Func kernelFunc, const CubeCaseCfg& cfg)
{
    const int64_t M = cfg.M;
    const int64_t N = cfg.N;
    const int64_t K = cfg.K;

    size_t x1Size = static_cast<size_t>(M) * K * sizeof(int8_t);
    size_t x2Size = static_cast<size_t>(K) * N * sizeof(int8_t);
    size_t pertokenScaleSize = sizeof(float);
    size_t scaleSize = sizeof(uint64_t);
    size_t biasSize = static_cast<size_t>(N) * cfg.biasElemSize;
    size_t ySize = static_cast<size_t>(M) * N * cfg.outElemSize;

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

    std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/quant_batch_matmul/qbmm_data";
    ASSERT_NO_FATAL_FAILURE(RunGenData(dataDir, M, N, K, cfg.genArgs));

    ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/input_a.bin", x1GM.Get(), x1Size, "input_a.bin"));
    ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/input_b.bin", x2GM.Get(), x2Size, "input_b.bin"));
    ASSERT_NO_FATAL_FAILURE(
        ReadBinToGm(dataDir + "/pertoken_scale.bin", pertokenScaleGM.Get(), pertokenScaleSize, "pertoken_scale.bin"));
    ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/scale.bin", scaleGM.Get(), scaleSize, "scale.bin"));

    if (cfg.isBias) {
        ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/bias.bin", biasGM.Get(), biasSize, "bias.bin"));
    }

    auto* tilingData = reinterpret_cast<QBMMV3TilingData*>(tilingGM.Get());
    FillCubeTiling(tilingData, cfg);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    const bool ok = KERNEL_RUN_KF(
        kernelFunc, cfg.blockNum, x1GM.Get(), x2GM.Get(), pertokenScaleGM.Get(), scaleGM.Get(), biasGM.Get(),
        yGM.Get(), tilingGM.Get());

    ASSERT_TRUE(ok) << "Kernel execution failed: one or more cores exited with non-zero status";
}

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

    std::string dataDir = std::string(UT_KERNEL_SRC_DIR) + "/quant_batch_matmul/qbmm_data";
    ASSERT_NO_FATAL_FAILURE(RunGenData(dataDir, M, N, K, cfg.genArgs));

    ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/input_a.bin", x1GM.Get(), x1Size, "input_a.bin"));
    ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/input_b.bin", x2GM.Get(), x2Size, "input_b.bin"));
    ASSERT_NO_FATAL_FAILURE(
        ReadBinToGm(dataDir + "/pertoken_scale.bin", pertokenScaleGM.Get(), pertokenScaleSize, "pertoken_scale.bin"));
    ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/scale.bin", scaleGM.Get(), scaleSize, "scale.bin"));

    if (cfg.isBias) {
        ASSERT_NO_FATAL_FAILURE(ReadBinToGm(dataDir + "/bias.bin", biasGM.Get(), biasSize, "bias.bin"));
    }

    QBMMV3TilingData* tilingData = reinterpret_cast<QBMMV3TilingData*>(tilingGM.Get());
    FillMixTiling(tilingData, cfg);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    const bool ok = KERNEL_RUN_KF(
        kernelFunc, cfg.blockNum, x1GM.Get(), x2GM.Get(), pertokenScaleGM.Get(), scaleGM.Get(), biasGM.Get(),
        yGM.Get(), tilingGM.Get());

    ASSERT_TRUE(ok) << "Kernel execution failed: one or more cores exited with non-zero status";
}

template <typename AType, typename BType, uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE>
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
        cfg.M, cfg.N, cfg.K, 1, cfg.baseM, cfg.baseN, cfg.baseK, cfg.kL1, cfg.kL1, cfg.nBufferNum, 2};

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = qbmm_mx_l0c_pingpong_kernel_entry<AType, BType, half, float>;
    if constexpr (FullLoadMode == Blaze::Gemm::A_FULL_LOAD_MODE) {
        kernelFunc = qbmm_mx_l0c_pingpong_a_full_load_kernel_entry<AType, BType, half, float>;
    }
    const bool ok = KERNEL_RUN_KF(
        kernelFunc, cfg.blockNum, x1GM.Get(), x2GM.Get(), pertokenScaleGM.Get(), scaleGM.Get(), biasGM.Get(),
        yGM.Get(), tilingGM.Get());

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
    *tilingData = QBMMUT::QBMMStreamKTilingData{cfg.M, cfg.N, cfg.K, 1, cfg.blockNum, 16, 16, 64, 64, 64, 64, 1};

    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    auto kernelFunc = qbmm_streamk_kernel_entry<AType, BType, CType, BiasType>;
    const bool ok = KERNEL_RUN_KF(
        kernelFunc, cfg.blockNum, x1GM.Get(), x2GM.Get(), pertokenScaleGM.Get(), scaleGM.Get(), biasGM.Get(),
        yGM.Get(), workspaceGM.Get(), tilingGM.Get());

    ASSERT_TRUE(ok) << "QBMM MX StreamK kernel execution failed";
}

template <typename AType, typename BType, typename CType, typename BiasType,
    uint64_t FullLoadMode = Blaze::Gemm::NONE_FULL_LOAD_MODE>
void RunMxSmoke(const MxCaseCfg& cfg)
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

    auto kernelFunc = qbmm_mx_kernel_entry<AType, BType, CType, BiasType>;
    if constexpr (FullLoadMode == Blaze::Gemm::A_FULL_LOAD_MODE) {
        kernelFunc = qbmm_mx_a_full_load_kernel_entry<AType, BType, CType, BiasType>;
    }
    const bool ok = KERNEL_RUN_KF(
        kernelFunc, cfg.blockNum, x1GM.Get(), x2GM.Get(), pertokenScaleGM.Get(), scaleGM.Get(), biasGM.Get(),
        yGM.Get(), tilingGM.Get());

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
    const bool ok = KERNEL_RUN_KF(
        kernelFunc, cfg.blockNum, x1GM.Get(), x2GM.Get(), pertokenScaleGM.Get(), scaleGM.Get(), biasGM.Get(),
        yGM.Get(), tilingGM.Get());

    ASSERT_TRUE(ok) << "QBMM MX without-batch kernel execution failed";
}

} // namespace


TEST_F(QBMMV3Test, Test_INT8_A8W8_PERTENSOR)
{
    CubeCaseCfg cfg{16, 16, 16, 1, QM_DEFAULT, QM_PERTENSOR, false, GE_DT_FLOAT,
        sizeof(int32_t), sizeof(half), ""};
    auto kernelFunc = qbmm_cube_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_INT8_A8W8_PERTENSOR_AFullLoad)
{
    CubeCaseCfg cfg{16, 16, 16, 1, QM_DEFAULT, QM_PERTENSOR, false, GE_DT_FLOAT,
        sizeof(int32_t), sizeof(half), ""};
    auto kernelFunc = qbmm_cube_a_full_load_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunCubeSmoke(kernelFunc, cfg);
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
    RunMxWithoutBatchSmoke<
        MxType, MxType, float, float, Blaze::Gemm::A_FULL_LOAD_MODE>(cfg);
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
    MixCaseCfg cfg{16, 16, 16, 1, QM_PERTOKEN, QM_PERCHANNEL, false, GE_DT_FLOAT,
        sizeof(float), sizeof(half),
        "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32"};
    auto kernelFunc = qbmm_mix_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_MIX_A8W8_PerChannel_PerToken_AFullLoad)
{
    MixCaseCfg cfg{16, 16, 16, 1, QM_PERTOKEN, QM_PERCHANNEL, false, GE_DT_FLOAT,
        sizeof(float), sizeof(half),
        "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32"};
    auto kernelFunc = qbmm_mix_a_full_load_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

// 仅权重 scale：激活 DEFAULT（epilogue 忽略 x1 scale）+ 权重 per-channel，half 输出。
TEST_F(QBMMV3Test, Test_MIX_A8W8_PerChannel_NoPtScale)
{
    MixCaseCfg cfg{16, 16, 16, 1, QM_DEFAULT, QM_PERCHANNEL, false, GE_DT_FLOAT,
        sizeof(float), sizeof(half),
        "--x1_mode default --x2_mode perchannel --scale_dtype float32"};
    auto kernelFunc = qbmm_mix_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

// 权重标量 scale：激活 per-token + 权重 per-tensor，half 输出。
TEST_F(QBMMV3Test, Test_MIX_A8W8_PerTensor_PerToken)
{
    MixCaseCfg cfg{16, 16, 16, 1, QM_PERTOKEN, QM_PERTENSOR, false, GE_DT_FLOAT,
        sizeof(float), sizeof(half),
        "--x1_mode pertoken --x2_mode pertensor --scale_dtype float32"};
    auto kernelFunc = qbmm_mix_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

// 覆盖 bias 路径 + biasDtype=fp16：激活 per-token + 权重 per-channel + fp16 bias，half 输出。
TEST_F(QBMMV3Test, Test_MIX_A8W8_WithBias_FP16)
{
    MixCaseCfg cfg{16, 16, 16, 1, QM_PERTOKEN, QM_PERCHANNEL, true, GE_DT_FLOAT16,
        sizeof(half), sizeof(half),
        "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32 --bias --bias_dtype float16"};
    auto kernelFunc = qbmm_mix_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

// 覆盖 OutType=bf16：激活 per-token + 权重 per-channel，bfloat16 输出。
TEST_F(QBMMV3Test, Test_MIX_A8W8_Output_BF16)
{
    MixCaseCfg cfg{16, 16, 16, 1, QM_PERTOKEN, QM_PERCHANNEL, false, GE_DT_FLOAT,
        sizeof(float), sizeof(bfloat16_t),
        "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32"};
    auto kernelFunc = qbmm_mix_kernel_entry<int8_t, int8_t, bfloat16_t, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

// 单 batch 特化：走 QbmmMixWithoutBatch，激活 per-token + 权重 per-channel，half 输出。
TEST_F(QBMMV3Test, Test_MIX_A8W8_WithoutBatch)
{
    MixCaseCfg cfg{16, 16, 16, 1, QM_PERTOKEN, QM_PERCHANNEL, false, GE_DT_FLOAT,
        sizeof(float), sizeof(half),
        "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32"};
    auto kernelFunc = qbmm_mix_without_batch_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}

TEST_F(QBMMV3Test, Test_MIX_A8W8_WithoutBatch_AFullLoad)
{
    MixCaseCfg cfg{16, 16, 16, 1, QM_PERTOKEN, QM_PERCHANNEL, false, GE_DT_FLOAT,
        sizeof(float), sizeof(half),
        "--x1_mode pertoken --x2_mode perchannel --scale_dtype float32"};
    auto kernelFunc = qbmm_mix_without_batch_a_full_load_kernel_entry<int8_t, int8_t, half, int32_t>;
    RunMixSmoke(kernelFunc, cfg);
}
