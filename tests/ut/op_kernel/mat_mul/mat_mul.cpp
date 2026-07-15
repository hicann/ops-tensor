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
 * \file mat_mul.cpp
 * \brief MatMulV3 Kernel UT统一入口
 */

#pragma once
#include <cstring>
#include "mat_mul_stream_k.h"
#include "mat_mul_basic.h"
#include "mat_mul_tiling_data.h"

enum OpType : int8_t
{
    OP_TYPE_MATMUL_BASIC = 0,
    OP_TYPE_MATMUL_STREAMK = 1,
};

template <
    int8_t OP_TYPE, typename DTYPE_X1, typename DTYPE_X2, typename DTYPE_Y, typename DTYPE_BIAS, 
    Blaze::Gemm::MatMulL0C2Out L0C2OUT_MODE = Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY, uint64_t FUSED_OP_TYPE = 0,
    uint64_t NON_CONTIGUOUS_TYPE = 0>
__global__ __aicore__ void mat_mul_v3_kernel_entry(
    GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR biasGM, GM_ADDR yGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    MatMulV3BasicTilingData tilingData;
    memcpy(&tilingData, tilingGM, sizeof(MatMulV3BasicTilingData));

    if constexpr (OP_TYPE == OP_TYPE_MATMUL_STREAMK) {
        MatMulV3UT::MatMulStreamKWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, L0C2OUT_MODE, FUSED_OP_TYPE>(
            x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData);
    } else if constexpr (OP_TYPE == OP_TYPE_MATMUL_BASIC) {
        MatMulV3UT::MatMulBasicWrapper<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, NON_CONTIGUOUS_TYPE>(
            x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData);
    } else {
        static_assert(sizeof(OP_TYPE) == 0, "Unsupported OP_TYPE value for mat_mul_v3_kernel_entry");
    }
}