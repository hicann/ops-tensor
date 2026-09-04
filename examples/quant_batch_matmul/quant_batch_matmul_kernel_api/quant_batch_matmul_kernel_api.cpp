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
 * @file quant_batch_matmul_kernel_api.cpp
 * @brief Executable examples for the public QBMM MIX, without-batch, MX, and StreamK kernel APIs.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "kernel_basic_intf.h"

// The QBMM epilogues receive ge::DataType-compatible values but use the AscendC
// enumerators in device code. Make those public enumerators visible before the
// template headers are parsed by the standalone example translation unit.
using AscendC::DT_BF16;
using AscendC::DT_FLOAT;
using AscendC::DT_FLOAT16;

#include "blaze/epilogue/block/block_epilogue_dequant.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/epilogue/block/block_epilogue_matmul_streamk.h"
#include "blaze/epilogue/block/block_epilogue_qbmm_pertensor_streamk.h"
#include "blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h"
#include "blaze/gemm/block/block_mmad_a8w8_mix.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_matmul_streamk.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/kernel/kernel_qbmm_mix.h"
#include "blaze/gemm/kernel/kernel_qbmm_mix_without_batch.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx_without_batch.h"
#include "blaze/gemm/kernel/kernel_qbmm_pertensor_streamk.h"
#include "blaze/gemm/kernel/kernel_qbmm_streamk.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "data_utils.h"

namespace {

using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
using Layout = AscendC::Te::NDExtLayoutPtn;

constexpr uint32_t QUANT_MODE_PERCHANNEL = 2U;
constexpr uint32_t QUANT_MODE_PERTOKEN = 4U;
constexpr uint32_t GE_DTYPE_FLOAT = 0U;
constexpr uint64_t MX_GROUP_SIZE = 32UL;
constexpr uint64_t MX_K_ALIGN = 64UL;
constexpr uint64_t STREAMK_WORKSPACE_TILE_BYTES = 256UL * 256UL * sizeof(float);
constexpr uint64_t STREAMK_WORKSPACE_OVERHEAD_BYTES = 20UL * 1024UL * 1024UL;
constexpr uint32_t L2_CACHE_DEFAULT_VALUE = 0U;

struct CliArgs {
    std::string variant;
    int64_t m{0};
    int64_t k{0};
    int64_t n{0};
};

bool IsMxVariant(const std::string& variant) { return variant == "mx_without_batch" || variant == "mx_streamk"; }

bool IsStreamKVariant(const std::string& variant) { return variant == "mx_streamk" || variant == "pertensor_streamk"; }

bool ParseArgs(int argc, const char** argv, CliArgs& args)
{
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <variant> <m> <k> <n>" << std::endl;
        return false;
    }
    args.variant = argv[1];
    args.m = std::atoll(argv[2]);
    args.k = std::atoll(argv[3]);
    args.n = std::atoll(argv[4]);
    const bool knownVariant = args.variant == "mix" || args.variant == "mix_without_batch" ||
                              args.variant == "mx_without_batch" || args.variant == "mx_streamk" ||
                              args.variant == "pertensor_streamk";
    if (!knownVariant || args.m <= 0 || args.k <= 0 || args.n <= 0) {
        std::cerr << "Invalid variant or shape." << std::endl;
        return false;
    }
    if (IsMxVariant(args.variant) && args.k % static_cast<int64_t>(MX_K_ALIGN) != 0) {
        std::cerr << "MX variants require K to be a multiple of 64." << std::endl;
        return false;
    }
    return true;
}

template <typename Params>
__aicore__ inline void FillMixMmadParams(Params& params, GM_ADDR x1, GM_ADDR x2, int64_t m, int64_t n, int64_t k)
{
    params.aGmAddr = x1;
    params.bGmAddr = x2;
    params.problemShape = {m, n, k, 1};
    params.l0TileShape = {m, n, k, 0};
    params.kAL1 = static_cast<uint32_t>(k);
    params.kBL1 = static_cast<uint32_t>(k);
    params.l1BufferNum = 2U;
    params.enableL0CPingPong = false;
}

template <typename Params>
__aicore__ inline void FillSchedulerParams(Params& params, int64_t m, int64_t n)
{
    params = {m, n, 1, 1, 1, 1, 0, 0};
}

template <typename Params>
__aicore__ inline void FillDequantParams(Params& params, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR y, int64_t m,
                                         int64_t n)
{
    params.x2ScaleGmAddr = x2Scale;
    params.x1ScaleGmAddr = x1Scale;
    params.biasGmAddr = nullptr;
    params.outGmAddr = y;
    params.m = m;
    params.n = n;
    params.baseM = m;
    params.baseN = n;
    params.x1QuantMode = QUANT_MODE_PERTOKEN;
    params.x2QuantMode = QUANT_MODE_PERCHANNEL;
    params.isBias = false;
    params.biasDtype = GE_DTYPE_FLOAT;
}

__global__ __aicore__ void QbmmMixKernel(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR y, int64_t m,
                                         int64_t k, int64_t n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using BTypeTuple = AscendC::Std::tuple<int8_t, uint64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMix<Blaze::Gemm::NONE_FULL_LOAD_MODE, false>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, int8_t, Layout, BTypeTuple, Layout, int32_t, Layout,
                                                    int32_t, Layout>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueDequant<half, int32_t, float, float, int32_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, Blaze::Gemm::NONE_FULL_LOAD_MODE, Layout, Layout, int8_t>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    Kernel::Params params{};
    params.problemShape = {m, n, k, 1};
    FillMixMmadParams(params.mmadParams, x1, x2, m, n, k);
    FillSchedulerParams(params.schParams, m, n);
    params.qbmmParams.batchA1 = 1U;
    params.qbmmParams.batchA2 = 1U;
    params.qbmmParams.batchA3 = 1U;
    params.qbmmParams.batchA4 = 1U;
    params.qbmmParams.batchB1 = 1U;
    params.qbmmParams.batchB2 = 1U;
    params.qbmmParams.batchB3 = 1U;
    params.qbmmParams.batchB4 = 1U;
    params.qbmmParams.batchC1 = 1U;
    params.qbmmParams.batchC2 = 1U;
    params.qbmmParams.batchC3 = 1U;
    params.qbmmParams.batchC4 = 1U;
    params.qbmmParams.x1QuantMode = QUANT_MODE_PERTOKEN;
    params.qbmmParams.x2QuantMode = QUANT_MODE_PERCHANNEL;
    params.qbmmParams.kAL1 = static_cast<uint32_t>(k);
    params.qbmmParams.kBL1 = static_cast<uint32_t>(k);
    params.qbmmParams.nBufferNum = 2U;
    params.qbmmParams.baseM = static_cast<uint32_t>(m);
    params.qbmmParams.baseN = static_cast<uint32_t>(n);
    params.qbmmParams.baseK = static_cast<uint32_t>(k);
    params.qbmmParams.dbL0C = 1U;
    FillDequantParams(params.epilogueParams, x1Scale, x2Scale, y, m, n);
    Kernel kernel;
    kernel(params);
}

__global__ __aicore__ void QbmmMixWithoutBatchKernel(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale,
                                                     GM_ADDR y, int64_t m, int64_t k, int64_t n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using BTypeTuple = AscendC::Std::tuple<int8_t, uint64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMix<Blaze::Gemm::NONE_FULL_LOAD_MODE, false,
                                                           Blaze::Gemm::KernelMmadWithScaleMixWithoutBatch>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, int8_t, Layout, BTypeTuple, Layout, int32_t, Layout,
                                                    int32_t, Layout>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueDequant<half, int32_t, float, float, int32_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, Blaze::Gemm::NONE_FULL_LOAD_MODE, Layout, Layout, int8_t>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    Kernel::Params params{};
    params.problemShape = {m, n, k, 1};
    FillMixMmadParams(params.mmParams, x1, x2, m, n, k);
    FillSchedulerParams(params.schParams, m, n);
    FillDequantParams(params.epilogueParams, x1Scale, x2Scale, y, m, n);
    Kernel kernel;
    kernel(params);
}

__global__ __aicore__ void QbmmMxWithoutBatchKernel(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR y,
                                                    int64_t m, int64_t k, int64_t n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using AType = fp8_e4m3fn_t;
    using BType = fp8_e4m3fn_t;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<Blaze::Gemm::NONE_FULL_LOAD_MODE, false,
                                                          Blaze::Gemm::KernelMmadWithScaleMxWithoutBatch>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, Layout, BType, Layout, half, Layout, float,
                                                    Layout>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, Blaze::Gemm::NONE_FULL_LOAD_MODE, Layout, Layout, AType>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, Blaze::Gemm::Block::BlockEpilogueEmpty,
                                                      BlockScheduler>;

    Kernel::Params params{};
    params.problemShape = {m, n, k, 1};
    params.mmadParams = {x1, x2, y, nullptr, x1Scale, x2Scale};
    params.l1Params = {64U, 64U, 2U};
    params.schParams = {16, 32, 1, 1, 1, 1, 0, 0};
    params.qbmmParams = {16U, 32U, 64U, 0U, 1U, 1U};
    Kernel kernel;
    kernel(params);
}

__global__ __aicore__ void QbmmMxStreamKKernel(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR y,
                                               GM_ADDR workspace, int64_t m, int64_t k, int64_t n, uint32_t usedCoreNum)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using AType = fp8_e4m3fn_t;
    using BType = fp8_e5m2_t;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<Blaze::Gemm::NONE_FULL_LOAD_MODE, false,
                                                          Blaze::Gemm::KernelQbmmMultiBlockStreamK>;
    using EpiloguePolicy = Blaze::Gemm::MatmulMultiBlockWithStreamK<>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, Layout, BType, Layout, half, Layout, float,
                                                    Layout>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueMatmulStreamK<float, half, EpiloguePolicy>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    Kernel::Params params{};
    params.problemShape = {m, n, k, 1};
    params.mmadParams = {x1, x2, y, nullptr, x1Scale, x2Scale};
    params.epilogueParams = {y, workspace};
    params.schParams = {usedCoreNum, m, 32, 64, 64, 64, 0U, L2_CACHE_DEFAULT_VALUE};
    params.qbmmParams = {64U, 1U, 1U};
    Kernel kernel;
    kernel(params);
}

__global__ __aicore__ void QbmmPertensorStreamKKernel(GM_ADDR x1, GM_ADDR x2, GM_ADDR x2Scale, GM_ADDR y,
                                                      GM_ADDR workspace, int64_t m, int64_t k, int64_t n,
                                                      uint32_t usedCoreNum)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<Blaze::Gemm::NONE_FULL_LOAD_MODE, false,
                                                                    Blaze::Gemm::KernelQbmmPertensorMultiBlockStreamK>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, int8_t, Layout, AscendC::Std::tuple<int8_t, float>,
                                                    Layout, half, Layout, int32_t, Layout>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<typename BlockMmad::WorkspaceType,
                                                                                    half, DispatchPolicy, float, float>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    Kernel::Params params{};
    params.problemShape = {m, n, k, 1};
    params.blockMmadParams = {x1, x2, y, nullptr, nullptr, x2Scale};
    params.epilogueParams = {y, workspace, x2Scale, nullptr, nullptr, false, GE_DTYPE_FLOAT};
    params.schParams = {usedCoreNum, m, n, 64, 64, 64, 0U, L2_CACHE_DEFAULT_VALUE};
    Kernel kernel;
    kernel(params);
}

struct DeviceBuffers {
    uint8_t* x1{nullptr};
    uint8_t* x2{nullptr};
    uint8_t* x1Scale{nullptr};
    uint8_t* x2Scale{nullptr};
    uint8_t* y{nullptr};
    uint8_t* workspace{nullptr};
};

void FreeBuffers(DeviceBuffers& buffers)
{
    for (uint8_t* ptr : {buffers.x1, buffers.x2, buffers.x1Scale, buffers.x2Scale, buffers.y, buffers.workspace}) {
        if (ptr != nullptr) {
            ACL_CHECK(aclrtFree(ptr));
        }
    }
}

void AllocateAndCopy(uint8_t** device, const std::vector<uint8_t>& host)
{
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(device), host.size(), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(*device, host.size(), host.data(), host.size(), ACL_MEMCPY_HOST_TO_DEVICE));
}

bool LoadBytes(const std::string& path, std::vector<uint8_t>& data) { return ReadFile(path, data.data(), data.size()); }

int Run(const CliArgs& args)
{
    aclrtStream stream{nullptr};
    ACLDeviceGuard guard(stream);
    const uint64_t scaleK = static_cast<uint64_t>(CeilAlign(args.k, static_cast<int64_t>(MX_K_ALIGN))) / MX_GROUP_SIZE;
    const size_t x1Bytes = static_cast<size_t>(args.m * args.k);
    const size_t x2Bytes = static_cast<size_t>(args.k * args.n);
    const size_t x1ScaleBytes = IsMxVariant(args.variant) ?
                                    static_cast<size_t>(args.m) * scaleK :
                                    (args.variant == "pertensor_streamk" ? 0U : args.m * sizeof(float));
    const size_t x2ScaleBytes = IsMxVariant(args.variant) ?
                                    static_cast<size_t>(args.n) * scaleK :
                                    (args.variant == "pertensor_streamk" ? sizeof(float) : args.n * sizeof(float));
    const size_t yBytes = static_cast<size_t>(args.m * args.n) * sizeof(half);
    const uint32_t launchBlocks = static_cast<uint32_t>(GetAicCoreNum());
    const size_t workspaceBytes = IsStreamKVariant(args.variant) ?
                                      launchBlocks * STREAMK_WORKSPACE_TILE_BYTES + STREAMK_WORKSPACE_OVERHEAD_BYTES :
                                      0U;

    std::vector<uint8_t> hostX1(x1Bytes);
    std::vector<uint8_t> hostX2(x2Bytes);
    std::vector<uint8_t> hostX1Scale(x1ScaleBytes);
    std::vector<uint8_t> hostX2Scale(x2ScaleBytes);
    std::vector<uint8_t> hostY(yBytes, 0U);
    if (!LoadBytes("./input/input_a.bin", hostX1) || !LoadBytes("./input/input_b.bin", hostX2) ||
        (!hostX1Scale.empty() && !LoadBytes("./input/scale_a.bin", hostX1Scale)) ||
        !LoadBytes("./input/scale_b.bin", hostX2Scale)) {
        return 1;
    }

    DeviceBuffers device;
    AllocateAndCopy(&device.x1, hostX1);
    AllocateAndCopy(&device.x2, hostX2);
    if (!hostX1Scale.empty()) {
        AllocateAndCopy(&device.x1Scale, hostX1Scale);
    }
    AllocateAndCopy(&device.x2Scale, hostX2Scale);
    AllocateAndCopy(&device.y, hostY);
    if (workspaceBytes > 0U) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&device.workspace), workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemset(device.workspace, workspaceBytes, 0, workspaceBytes));
    }

    if (args.variant == "mix") {
        QbmmMixKernel<<<launchBlocks, 0, stream>>>(device.x1, device.x2, device.x1Scale, device.x2Scale, device.y,
                                                   args.m, args.k, args.n);
    } else if (args.variant == "mix_without_batch") {
        QbmmMixWithoutBatchKernel<<<launchBlocks, 0, stream>>>(device.x1, device.x2, device.x1Scale, device.x2Scale,
                                                               device.y, args.m, args.k, args.n);
    } else if (args.variant == "mx_without_batch") {
        QbmmMxWithoutBatchKernel<<<launchBlocks, 0, stream>>>(device.x1, device.x2, device.x1Scale, device.x2Scale,
                                                              device.y, args.m, args.k, args.n);
    } else if (args.variant == "mx_streamk") {
        QbmmMxStreamKKernel<<<launchBlocks, 0, stream>>>(device.x1, device.x2, device.x1Scale, device.x2Scale, device.y,
                                                         device.workspace, args.m, args.k, args.n, launchBlocks);
    } else {
        QbmmPertensorStreamKKernel<<<launchBlocks, 0, stream>>>(device.x1, device.x2, device.x2Scale, device.y,
                                                                device.workspace, args.m, args.k, args.n, launchBlocks);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(hostY.data(), yBytes, device.y, yBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    const bool written = WriteFile("./output/npu_out.bin", hostY.data(), hostY.size());
    FreeBuffers(device);
    if (!written) {
        return 1;
    }
    std::cout << "PASS: launched " << args.variant << " with shape [" << args.m << ", " << args.k << ", " << args.n
              << "]" << std::endl;
    return 0;
}

} // namespace

int main(int argc, const char** argv)
{
    CliArgs args;
    if (!ParseArgs(argc, argv, args)) {
        return 1;
    }
    return Run(args);
}
