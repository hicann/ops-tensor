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
 * \file qbmm_ut_fill_helpers.h
 * \brief QBMM Kernel UT 公共 qbmmParams 填充助手：供 fixpipe(qbmm_cube.h) 与 MIX(qbmm_mix.h) wrapper 复用，
 *        消除 batch 维度 / tile 配置字段赋值块的重复代码。
 */

#pragma once

#include "qbmm_tiling_data.h"

namespace QBMMUT {

// 填充 batch 广播维度字段（A/B/C 各 4 维）+ bias/量化模式标志。
// QbmmParams 通过鸭子类型匹配 GemmUniversal 各特化的 QBMMTiling（fixpipe / MIX 多 batch）。
template <typename QbmmParams>
__aicore__ inline void FillQbmmBatchParams(QbmmParams& qp, const QBMMV3TilingData& td)
{
    qp.batchA1 = td.batchA1;
    qp.batchA2 = td.batchA2;
    qp.batchA3 = td.batchA3;
    qp.batchA4 = td.batchA4;
    qp.batchB1 = td.batchB1;
    qp.batchB2 = td.batchB2;
    qp.batchB3 = td.batchB3;
    qp.batchB4 = td.batchB4;
    qp.batchC1 = td.batchC1;
    qp.batchC2 = td.batchC2;
    qp.batchC3 = td.batchC3;
    qp.batchC4 = td.batchC4;
    qp.biasThreeDim = td.biasThreeDim;
    qp.x1QuantMode = td.x1QuantMode;
    qp.x2QuantMode = td.x2QuantMode;
}

// 填充 L1 载入 / L0C tile 配置字段（三种 QBMMTiling 均含这些字段）。
template <typename QbmmParams>
__aicore__ inline void FillQbmmTileParams(QbmmParams& qp, const QBMMV3TilingData& td)
{
    qp.kAL1 = td.kAL1;
    qp.kBL1 = td.kBL1;
    qp.nBufferNum = td.nBufferNum;
    qp.baseM = td.baseM_qbmm;
    qp.baseN = td.baseN_qbmm;
    qp.baseK = td.baseK_qbmm;
    qp.isBias = td.isBias;
    qp.dbL0C = td.dbL0C;
}

} // namespace QBMMUT
