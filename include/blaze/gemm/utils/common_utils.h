/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file common_utils.h
 * \brief
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#include "std/algorithm.h"
#else
#include "kernel_operator.h"
#endif

namespace Blaze {
namespace Gemm {
constexpr int32_t MATMUL_MNK_ALIGN = 16;
constexpr uint64_t DOUBLE_BUFFER_COUNT = 2UL;
constexpr int64_t QUADRUPLE_BUFFER_COUNT = 4LL;
constexpr int32_t MNK_M = 0;
constexpr int32_t MNK_N = 1;
constexpr int32_t MNK_K = 2;
constexpr int32_t MNK_B = 3;
constexpr int32_t MNK_M0 = 4;
constexpr int32_t MNK_N0 = 5;
constexpr int32_t EVEN_NUMBER = 2;

static constexpr uint64_t A_FULL_LOAD_MODE = 1UL;
static constexpr uint64_t B_FULL_LOAD_MODE = 2UL;
static constexpr uint64_t NONE_FULL_LOAD_MODE = 0UL;
constexpr bool ENABLE_SPLIT_SINGLE_CORE_K = true;
constexpr bool NONE_SPLIT_SINGLE_CORE_K = false;

static constexpr int64_t WINDOW_LEN = 4UL;

constexpr static uint16_t DIMENSION_M = 0;
constexpr static uint16_t DIMENSION_N = 1;
constexpr static uint16_t DIMENSION_K = 2;

constexpr uint16_t ZERO_FLAG = 0;
constexpr uint16_t FIRST_FLAG = 1;
constexpr uint16_t SECOND_FLAG = 2;
constexpr uint16_t THIRD_FLAG = 3;
constexpr uint16_t FOURTH_FLAG = 4;
constexpr uint16_t FIFTH_FLAG = 5;
constexpr uint16_t SIXTH_FLAG = 6;
constexpr uint16_t SEVENTH_FLAG = 7;
static constexpr int64_t PER_BLOCK_SIZE = 128LL;

constexpr uint32_t L2_CACHE_DEFAULT = 0U;
constexpr uint32_t A_L2_CACHE_DISABLE = 1U;
constexpr uint32_t B_L2_CACHE_DISABLE = 2U;
constexpr uint32_t ALL_L2_CACHE_DISABLE = 3U;
constexpr uint64_t MXFP_DIVISOR_SIZE = 64UL;
constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2UL;
constexpr uint64_t BLOCK_CUBE = 16UL;
constexpr uint32_t FINAL_ACCUMULATION = 3;
constexpr uint32_t NON_FINAL_ACCUMULATION = 2;

constexpr uint64_t C0_SIZE_fp16 = 16UL;
constexpr uint64_t C0_SIZE_fp32 = 8UL;
constexpr uint64_t C0_SIZE_B8 = 32UL;
constexpr uint64_t C0_SIZE_B4 = 64UL;
constexpr uint64_t C0_SIZE_L0C = 16UL;

constexpr uint64_t ALIGN_MASK_64 = 63UL;
constexpr uint64_t ALIGN_MASK_32 = 31UL;
constexpr uint64_t ALIGN_MASK_16 = 15UL;
constexpr uint16_t ALIGN_64_BYTES_SHIFT = 6;
constexpr uint16_t ALIGN_32_BYTES_SHIFT = 5;
constexpr uint16_t MXFP_DIVISOR_SHIFT = 6;
constexpr uint16_t MXFP_MULTI_BASE_SHIFT = 1;
constexpr uint64_t SCALE_C0 = 2UL;
constexpr uint64_t SCALE_BUFFER_NUM = 2UL;

constexpr uint64_t NON_CONTIGUOUS_TYPE_SLICE = 1UL;
// FusedMatMul OpType
constexpr uint64_t OP_TYPE_EMPTY = 0UL;
constexpr uint64_t OP_TYPE_ADD = 1UL;
constexpr uint64_t OP_TYPE_MUL = 2UL;
constexpr uint64_t OP_TYPE_RELU = 5UL;
constexpr uint64_t BLOCK_BYTE_SIZE = 32UL;

constexpr uint64_t IDX_M_TILEIDX = 0UL;
constexpr uint64_t IDX_N_TILEIDX = 1UL;
constexpr uint64_t IDX_M_TAIL_SPLIT_TILEIDX = 2UL;
constexpr uint64_t IDX_N_TAIL_SPLIT_TILEIDX = 3UL;

constexpr uint64_t IDX_M_IDX = 0UL;
constexpr uint64_t IDX_N_IDX = 1UL;
constexpr uint64_t IDX_K_IDX = 2UL;

constexpr uint16_t INPUT_BUFFER_FLAG_0 = 0;
constexpr uint16_t INPUT_BUFFER_FLAG_1 = 1;
constexpr uint16_t INPUT_BUFFER_FLAG_2 = 2;
constexpr uint16_t INPUT_BUFFER_FLAG_3 = 3;
constexpr uint16_t SCALE_BUFFER_FLAG_0 = FOURTH_FLAG;
constexpr uint16_t SCALE_BUFFER_FLAG_1 = FIFTH_FLAG;
constexpr uint16_t M_MTE1_FLAG_0 = FOURTH_FLAG;
constexpr uint16_t M_MTE1_FLAG_1 = FIFTH_FLAG;

constexpr uint64_t LAST_BATCH_DIM = 3;
constexpr uint64_t MAX_BATCH_DIM = 4;

enum class QuantMode : uint32_t
{
    DEFAULT = 0x0U,
    PERTENSOR_MODE = 0x1U,
    PERCHANNEL_MODE = 0x1U << 1,
    PERTOKEN_MODE = 0x1U << 2,
    MX_PERGROUP_MODE = 0x1U << 3,
    PERBLOCK_MODE = 0x1U << 4,
    PERGROUP_MODE = 0x1U << 5,
};

template <typename...>
struct always_false : public AscendC::Std::false_type {};

template <typename... Tp>
constexpr bool always_false_v = always_false<Tp...>::value;

template <typename T>
__aicore__ inline constexpr bool IsFp4()
{
    return AscendC::IsSameType<T, fp4x2_e2m1_t>::value || AscendC::IsSameType<T, fp4x2_e1m2_t>::value;
}

template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T CeilAlign(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

__aicore__ inline uint64_t Align64(uint64_t x)
{
    return (x + ALIGN_MASK_64) & ~ALIGN_MASK_64;
}

__aicore__ inline uint64_t Align32(uint64_t x)
{
    return (x + ALIGN_MASK_32) & ~ALIGN_MASK_32;
}

__aicore__ inline uint64_t Align16(uint64_t x)
{
    return (x + ALIGN_MASK_16) & ~ALIGN_MASK_16;
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

__aicore__ inline int64_t GetPerBlockNum(int64_t coreNum, int64_t mTileNum, int64_t nTileNum, int64_t b = 1)
{
    int64_t perCoreBlockNum = Blaze::Gemm::CeilDiv(mTileNum * nTileNum * b, coreNum);
    return perCoreBlockNum;
}

__aicore__ inline uint64_t CalWeightNZGmAddrOffset(bool transB, int64_t batchIdx, int64_t n, int64_t k, int64_t c0_size)
{
    if (transB) {
        return batchIdx * Blaze::Gemm::CeilDiv(k, c0_size) * Blaze::Gemm::CeilDiv(n, static_cast<int64_t>(BLOCK_CUBE)) *
               BLOCK_CUBE * c0_size;
    } else {
        return batchIdx * Blaze::Gemm::CeilDiv(n, c0_size) * Blaze::Gemm::CeilDiv(k, static_cast<int64_t>(BLOCK_CUBE)) *
               BLOCK_CUBE * c0_size;
    }
}

} // namespace Gemm
} // namespace Blaze
