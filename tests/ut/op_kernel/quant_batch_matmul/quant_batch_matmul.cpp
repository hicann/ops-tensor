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
 * \file quant_batch_matmul.cpp
 * \brief QBMMV3 Kernel UT统一入口
 */

#pragma once
#include <cstring>
#include "qbmm_cube.h"
#include "qbmm_tiling_data.h"

enum QBMMApiType : int
{
    OP_TYPE_QBMM_CUBE = 10,
};

template <
    int OP_TYPE, class DTYPE_X1, class DTYPE_X2, class DTYPE_Y, class DTYPE_BIAS>
__global__ __aicore__ void qbmm_kernel_entry(
    GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR pertokenScaleGM, GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR yGM, GM_ADDR tilingGM)
{
    QBMMV3TilingData tilingData;
    memcpy(&tilingData, tilingGM, sizeof(QBMMV3TilingData));

    if constexpr (OP_TYPE == OP_TYPE_QBMM_CUBE) {
        QBMMUT::QBMMCubeWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS>(
            x1GM, x2GM, pertokenScaleGM, scaleGM, biasGM, yGM, tilingData);
    } else {
        static_assert(sizeof(OP_TYPE) == 0, "Unsupported OP_TYPE value for qbmm_kernel_entry");
    }
}
