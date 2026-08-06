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
 * \file qbmm_pertensor_streamk_tiling_data.h
 * \brief QBMM per-tensor StreamK Kernel UT tiling data.
 */

#pragma once

#include "blaze_kernel_stub.h"

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

namespace QBMMUT {

constexpr uint32_t GE_DT_FLOAT = 0U;

#pragma pack(push, 8)
struct QBMMPertensorStreamKTilingData {
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t b;
    uint32_t usedCoreNum;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t singleCoreK;
    uint32_t kL1;
    uint32_t isBias;
    uint32_t biasDtype;
};
#pragma pack(pop)

} // namespace QBMMUT
