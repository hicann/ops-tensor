/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/**
 * @file qgmm_mx.cpp
 * @brief QGMM MX grouped-matmul example covering MXFP4/MXFP8 and ND/DN/NZ/ZN weights.
 */
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "acl/acl.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_qgmm_mx.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "data_utils.h"
#include "kernel_basic_intf.h"
#include "platform/platform_ascendc.h"

using ScaleType = fp8_e8m0_t;
using NdLayout = AscendC::Te::NDExtLayoutPtn;

template <typename T>
inline constexpr bool IS_FP4_TYPE = std::is_same_v<T, fp4x2_e2m1_t> || std::is_same_v<T, fp4x2_e1m2_t>;

template <typename AType, typename BType, typename LayoutA, typename LayoutB>
__global__ __aicore__ void qgmm_mx_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR scaleA, GM_ADDR scaleB, GM_ADDR c, GM_ADDR bias,
                                          GM_ADDR groupList, uint32_t groupNum, int64_t m, int64_t n, int64_t k,
                                          uint8_t singleW, uint8_t groupListType, uint8_t l1BufferStage,
                                          uint8_t withBias)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using Policy = Blaze::Gemm::GroupedMatmulWithScaleMx<0>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, AType, LayoutA, BType, LayoutB, float, NdLayout, float,
                                               NdLayout>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Mmad, Blaze::Gemm::Block::BlockEpilogueEmpty,
                                                      Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit>;
    typename Kernel::Params p{};
    p.problemShape = {m, n, k, 0};
    p.mmadParams.aGmAddr = a;
    p.mmadParams.bGmAddr = b;
    p.mmadParams.cGmAddr = c;
    p.mmadParams.scaleAGmAddr = scaleA;
    p.mmadParams.scaleBGmAddr = scaleB;
    p.mmadParams.biasGmAddr = bias;
    p.groupListGmAddr = groupList;
    p.gmmParams = {groupNum, m, n, k, 16, 64, 64, 64, 64, 2, 2, withBias, 1, l1BufferStage, 0, groupListType, singleW};
    Kernel kernel;
    kernel(p);
}

template <typename T>
constexpr size_t MxBytes(size_t count)
{
    return IS_FP4_TYPE<T> ? (count + 1U) / 2U : count * sizeof(T);
}

struct CaseBytes {
    size_t a;
    size_t bGroup;
    size_t scaleA;
    size_t scaleBGroup;
    size_t c;
};

struct DeviceBuffers {
    uint8_t *a = nullptr, *b = nullptr, *scaleA = nullptr, *scaleB = nullptr, *c = nullptr, *bias = nullptr;
    uint8_t* groupList = nullptr;
    std::vector<uint8_t*> weights;
    std::vector<uint8_t*> scales;

    explicit DeviceBuffers(uint32_t groupNum) : weights(groupNum, nullptr), scales(groupNum, nullptr) {}

    ~DeviceBuffers()
    {
        if (groupList != nullptr)
            aclrtFree(groupList);
        if (c != nullptr)
            aclrtFree(c);
        if (bias != nullptr)
            aclrtFree(bias);
        if (scaleB != nullptr)
            aclrtFree(scaleB);
        if (b != nullptr)
            aclrtFree(b);
        if (scaleA != nullptr)
            aclrtFree(scaleA);
        if (a != nullptr)
            aclrtFree(a);
        for (size_t i = 0; i < weights.size(); ++i) {
            if (scales[i] != nullptr)
                aclrtFree(scales[i]);
            if (weights[i] != nullptr)
                aclrtFree(weights[i]);
        }
    }
};

void AllocateCommonBuffers(DeviceBuffers& device, const CaseBytes& bytes, size_t biasBytes, size_t groupListBytes)
{
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device.a), bytes.a, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device.scaleA), bytes.scaleA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device.c), bytes.c, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device.bias), biasBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device.groupList), groupListBytes, ACL_MEM_MALLOC_HUGE_FIRST));
}

std::vector<uint8_t> ReadBinary(const std::string& path, size_t size)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream.is_open() || static_cast<size_t>(stream.tellg()) != size) {
        throw std::runtime_error("invalid input file size: " + path);
    }
    std::vector<uint8_t> data(size);
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
    if (!stream) {
        throw std::runtime_error("failed to read input file: " + path);
    }
    return data;
}

template <bool MultiTensor>
void PrepareWeightBuffers(DeviceBuffers& device, const CaseBytes& bytes, uint32_t groupNum, const std::string& dataDir)
{
    if constexpr (!MultiTensor) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device.b), groupNum * bytes.bGroup, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device.scaleB), groupNum * bytes.scaleBGroup,
                              ACL_MEM_MALLOC_HUGE_FIRST));
        const auto allWeights = ReadBinary(dataDir + "/input_b.bin", groupNum * bytes.bGroup);
        const auto allScales = ReadBinary(dataDir + "/scale_b.bin", groupNum * bytes.scaleBGroup);
        ACL_CHECK(
            aclrtMemcpy(device.b, allWeights.size(), allWeights.data(), allWeights.size(), ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(device.scaleB, allScales.size(), allScales.data(), allScales.size(),
                              ACL_MEMCPY_HOST_TO_DEVICE));
        return;
    }
    std::vector<uint64_t> weightList(groupNum + 1U, sizeof(uint64_t));
    std::vector<uint64_t> scaleList(groupNum + 1U, sizeof(uint64_t));
    for (uint32_t i = 0; i < groupNum; ++i) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device.weights[i]), bytes.bGroup, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(
            aclrtMalloc(reinterpret_cast<void**>(&device.scales[i]), bytes.scaleBGroup, ACL_MEM_MALLOC_HUGE_FIRST));
        const auto values = ReadBinary(dataDir + "/input_b_" + std::to_string(i) + ".bin", bytes.bGroup);
        const auto scales = ReadBinary(dataDir + "/scale_b_" + std::to_string(i) + ".bin", bytes.scaleBGroup);
        ACL_CHECK(aclrtMemcpy(device.weights[i], bytes.bGroup, values.data(), bytes.bGroup, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(device.scales[i], bytes.scaleBGroup, scales.data(), bytes.scaleBGroup,
                              ACL_MEMCPY_HOST_TO_DEVICE));
        weightList[i + 1U] = reinterpret_cast<uint64_t>(device.weights[i]);
        scaleList[i + 1U] = reinterpret_cast<uint64_t>(device.scales[i]);
    }
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device.b), weightList.size() * sizeof(uint64_t),
                          ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device.scaleB), scaleList.size() * sizeof(uint64_t),
                          ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(device.b, weightList.size() * sizeof(uint64_t), weightList.data(),
                          weightList.size() * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(device.scaleB, scaleList.size() * sizeof(uint64_t), scaleList.data(),
                          scaleList.size() * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
}

void CopyCommonInputs(DeviceBuffers& device, const CaseBytes& bytes, size_t biasBytes, size_t groupListBytes,
                      const std::string& dataDir)
{
    const auto values = ReadBinary(dataDir + "/input_a.bin", bytes.a);
    const auto scales = ReadBinary(dataDir + "/scale_a.bin", bytes.scaleA);
    const auto bias = ReadBinary(dataDir + "/bias.bin", biasBytes);
    const auto groupList = ReadBinary(dataDir + "/group_list.bin", groupListBytes);
    ACL_CHECK(aclrtMemcpy(device.a, bytes.a, values.data(), bytes.a, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(device.scaleA, bytes.scaleA, scales.data(), bytes.scaleA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(device.bias, biasBytes, bias.data(), biasBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(
        aclrtMemcpy(device.groupList, groupListBytes, groupList.data(), groupListBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemset(device.c, bytes.c, 0, bytes.c));
}

void WriteOutput(const std::string& outputPath, const std::vector<float>& output)
{
    std::ofstream stream(outputPath, std::ios::binary);
    if (!stream.is_open()) {
        throw std::runtime_error("failed to open output file: " + outputPath);
    }
    stream.write(reinterpret_cast<const char*>(output.data()),
                 static_cast<std::streamsize>(output.size() * sizeof(float)));
    if (!stream) {
        throw std::runtime_error("failed to write output file: " + outputPath);
    }
}

template <typename AType, typename BType, typename LayoutA, typename LayoutB, bool MultiTensor>
void LaunchKernel(DeviceBuffers& device, const CaseBytes& bytes, uint32_t groupNum, int64_t m, int64_t n, int64_t k,
                  uint8_t groupListType, uint8_t l1BufferStage, bool withBias, const std::string& outputPath)
{
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    qgmm_mx_kernel<AType, BType, LayoutA, LayoutB><<<static_cast<uint32_t>(GetAicCoreNum()), 0, stream>>>(
        device.a, device.b, device.scaleA, device.scaleB, device.c, device.bias, device.groupList, groupNum, m, n, k,
        MultiTensor ? 0 : 1, groupListType, l1BufferStage, withBias ? 1 : 0);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::vector<float> output(bytes.c / sizeof(float), -1.0f);
    ACL_CHECK(aclrtMemcpy(output.data(), bytes.c, device.c, bytes.c, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtDestroyStream(stream));
    WriteOutput(outputPath, output);
}

std::vector<int64_t> MakeGroupList(uint32_t groupNum, int64_t splitValue, uint8_t groupListType)
{
    std::vector<int64_t> groupList;
    groupList.reserve(groupListType == 2 ? groupNum * 2U : groupNum);
    for (uint32_t i = 0; i < groupNum; ++i) {
        if (groupListType == 2) {
            groupList.push_back(static_cast<int64_t>(i));
        }
        groupList.push_back(groupListType == 0 ? static_cast<int64_t>(i + 1U) * splitValue : splitValue);
    }
    return groupList;
}

template <typename AType, typename BType, typename LayoutA, typename LayoutB, bool MultiTensor>
int RunCase(uint32_t groupNum, int64_t m, int64_t n, int64_t k, uint8_t groupListType, uint8_t l1BufferStage,
            bool withBias, const std::string& dataDir, const std::string& outputPath)
{
    constexpr bool transA = std::is_same_v<LayoutA, AscendC::Te::DNExtLayoutPtn>;
    constexpr bool weightNz = std::is_same_v<LayoutB, AscendC::Te::NZLayoutPtn> ||
                              std::is_same_v<LayoutB, AscendC::Te::ZNLayoutPtn>;
    constexpr bool transB = std::is_same_v<LayoutB, AscendC::Te::DNExtLayoutPtn> ||
                            std::is_same_v<LayoutB, AscendC::Te::ZNLayoutPtn>;
    const size_t c0 = IS_FP4_TYPE<BType> ? 64U : 32U;
    const size_t storedK = weightNz ? (transB ? ((k + c0 - 1U) / c0 * c0) : ((k + 15U) / 16U * 16U)) : k;
    const size_t storedN = weightNz ? (transB ? ((n + 15U) / 16U * 16U) : ((n + c0 - 1U) / c0 * c0)) : n;
    const size_t scaleK = static_cast<size_t>((k + 63) / 64) * 2U;
    const size_t scaleFactor = transA ? 2U : 1U;
    const CaseBytes bytes = {MxBytes<AType>(static_cast<size_t>(groupNum) * m * k), MxBytes<BType>(storedK * storedN),
                             static_cast<size_t>(groupNum) * m * scaleK * scaleFactor * sizeof(ScaleType),
                             static_cast<size_t>(n) * scaleK * scaleFactor * sizeof(ScaleType),
                             static_cast<size_t>(transA ? m : groupNum * m) * n * sizeof(float)};
    const std::vector<int64_t> groupList = MakeGroupList(groupNum, transA ? k : m, groupListType);
    DeviceBuffers device(groupNum);
    AllocateCommonBuffers(device, bytes, static_cast<size_t>(groupNum) * n * sizeof(float),
                          groupList.size() * sizeof(int64_t));
    PrepareWeightBuffers<MultiTensor>(device, bytes, groupNum, dataDir);
    CopyCommonInputs(device, bytes, static_cast<size_t>(groupNum) * n * sizeof(float),
                     groupList.size() * sizeof(int64_t), dataDir);
    LaunchKernel<AType, BType, LayoutA, LayoutB, MultiTensor>(device, bytes, groupNum, m, n, k, groupListType,
                                                              l1BufferStage, withBias, outputPath);
    return 0;
}

template <typename T, typename LayoutA, typename LayoutB>
int DispatchMulti(bool multi, uint32_t e, int64_t m, int64_t n, int64_t k, uint8_t groupListType, uint8_t l1BufferStage,
                  bool withBias, const std::string& dataDir, const std::string& outputPath)
{
    return multi ? RunCase<T, T, LayoutA, LayoutB, true>(e, m, n, k, groupListType, l1BufferStage, withBias, dataDir,
                                                         outputPath) :
                   RunCase<T, T, LayoutA, LayoutB, false>(e, m, n, k, groupListType, l1BufferStage, withBias, dataDir,
                                                          outputPath);
}

template <typename T, typename LayoutA>
int DispatchFormat(const std::string& format, bool multi, uint32_t e, int64_t m, int64_t n, int64_t k,
                   uint8_t groupListType, uint8_t l1BufferStage, bool withBias, const std::string& dataDir,
                   const std::string& outputPath)
{
    if (format == "nd")
        return DispatchMulti<T, LayoutA, NdLayout>(multi, e, m, n, k, groupListType, l1BufferStage, withBias, dataDir,
                                                   outputPath);
    if (format == "dn")
        return DispatchMulti<T, LayoutA, AscendC::Te::DNExtLayoutPtn>(multi, e, m, n, k, groupListType, l1BufferStage,
                                                                      withBias, dataDir, outputPath);
    if (format == "nz")
        return DispatchMulti<T, LayoutA, AscendC::Te::NZLayoutPtn>(multi, e, m, n, k, groupListType, l1BufferStage,
                                                                   withBias, dataDir, outputPath);
    if (format == "zn")
        return DispatchMulti<T, LayoutA, AscendC::Te::ZNLayoutPtn>(multi, e, m, n, k, groupListType, l1BufferStage,
                                                                   withBias, dataDir, outputPath);
    return 2;
}

template <typename T>
int DispatchTransA(const std::string& format, bool multi, bool transA, uint32_t e, int64_t m, int64_t n, int64_t k,
                   uint8_t groupListType, uint8_t l1BufferStage, bool withBias, const std::string& dataDir,
                   const std::string& outputPath)
{
    return transA ? DispatchFormat<T, AscendC::Te::DNExtLayoutPtn>(format, multi, e, m, n, k, groupListType,
                                                                   l1BufferStage, withBias, dataDir, outputPath) :
                    DispatchFormat<T, NdLayout>(format, multi, e, m, n, k, groupListType, l1BufferStage, withBias,
                                                dataDir, outputPath);
}

struct QgmmCaseConfig {
    std::string dtype;
    std::string format;
    std::string weightMode;
    uint32_t groupNum;
    int64_t m;
    int64_t n;
    int64_t k;
    bool transA;
    bool withBias;
    std::string groupList;
    uint8_t groupListType;
    uint8_t l1BufferStage;
    std::string dataDir;
    std::string outputPath;
};

void PrintUsage()
{
    std::cerr << "Usage: qgmm_mx <mxfp4_e2m1|mxfp4_e1m2|mxfp8_e4m3|mxfp8_e5m2> "
                 "<nd|dn|nz|zn> <single|multi> <e> <m> <n> <k> <transA> <bias> "
                 "<length|offset|sparse> <l1BufferStage> <dataDir> <outputPath>"
              << std::endl;
}

QgmmCaseConfig ParseConfig(char** argv)
{
    QgmmCaseConfig config{};
    config.dtype = argv[1];
    config.format = argv[2];
    config.weightMode = argv[3];
    config.groupNum = static_cast<uint32_t>(std::stoul(argv[4]));
    config.m = std::stoll(argv[5]);
    config.n = std::stoll(argv[6]);
    config.k = std::stoll(argv[7]);
    config.transA = std::string(argv[8]) == "true";
    config.withBias = std::string(argv[9]) == "true";
    config.groupList = argv[10];
    config.groupListType = config.groupList == "offset" ? 0 : (config.groupList == "length" ? 1 : 2);
    config.l1BufferStage = static_cast<uint8_t>(std::stoul(argv[11]));
    config.dataDir = argv[12];
    config.outputPath = argv[13];
    return config;
}

bool IsValidConfig(const QgmmCaseConfig& config)
{
    const bool validShape = config.groupNum > 0 && config.m > 0 && config.n > 0 && config.k > 0;
    const bool validGroupList = config.groupList == "offset" || config.groupList == "length" ||
                                config.groupList == "sparse";
    const bool validL1Buffer = config.l1BufferStage == 2 || config.l1BufferStage == 3;
    return validShape && validGroupList && validL1Buffer;
}

template <typename T>
int DispatchConfig(const QgmmCaseConfig& config)
{
    const bool multi = config.weightMode == "multi";
    return DispatchTransA<T>(config.format, multi, config.transA, config.groupNum, config.m, config.n, config.k,
                             config.groupListType, config.l1BufferStage, config.withBias, config.dataDir,
                             config.outputPath);
}

int RunConfiguredCase(const QgmmCaseConfig& config)
{
    if (config.weightMode != "single" && config.weightMode != "multi")
        return 2;
    if (config.dtype == "mxfp8_e4m3")
        return DispatchConfig<fp8_e4m3fn_t>(config);
    if (config.dtype == "mxfp8_e5m2")
        return DispatchConfig<fp8_e5m2_t>(config);
    if (config.dtype == "mxfp4_e2m1")
        return DispatchConfig<fp4x2_e2m1_t>(config);
    if (config.dtype == "mxfp4_e1m2")
        return DispatchConfig<fp4x2_e1m2_t>(config);
    return 2;
}

void PrintResult(int ret, const QgmmCaseConfig& config)
{
    if (ret == 0) {
        std::cout << "QGMM MX kernel execution completed, output=" << config.outputPath << std::endl;
    } else if (ret == 1) {
        std::cerr << "QGMM MX example FAILED: kernel execution failed" << std::endl;
    } else {
        std::cerr << "QGMM MX example FAILED: unsupported argument combination, dtype=" << config.dtype
                  << ", format=" << config.format << ", weight_mode=" << config.weightMode << std::endl;
    }
}

int main(int argc, char** argv)
{
    if (argc != 14) {
        PrintUsage();
        return 2;
    }
    const QgmmCaseConfig config = ParseConfig(argv);
    if (!IsValidConfig(config)) {
        std::cerr << "e, m, n and k must be positive" << std::endl;
        return 2;
    }
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    const int ret = RunConfiguredCase(config);
    aclrtResetDevice(0);
    aclFinalize();
    PrintResult(ret, config);
    return ret;
}
