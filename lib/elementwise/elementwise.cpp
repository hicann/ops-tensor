/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file elementwise.cpp
 * \brief Elementwise 框架实现
 *
 * 架构说明：
 * - elementwise_binary.cpp: 实现 acltensorElementwiseBinaryExecute (Phase 1)
 * - elementwise.cpp: 实现 ElementwiseSolutionRegistry::registerAllSolutions()
 *                   Phase 2 将在此文件实现 acltensorElementwiseTrinaryExecute
 */

#include "cann_ops_tensor.h"
#include "elementwise.hpp"
#include "lib/core/plan.hpp"

// ============================================================================
// Phase 2: Elementwise Trinary 执行函数（待实现）
// ============================================================================
//
// TODO: Phase 2 - 在此实现 acltensorElementwiseTrinaryExecute
//       参照 hiptensor_elementwise_trinary.cpp 的实现
//
// extern "C" {
//
// acltensorStatus_t acltensorElementwiseTrinaryExecute(
//     const acltensorHandle_t handle,
//     const acltensorPlan_t   plan,
//     const void*             alpha,
//     const void*             A,
//     const void*             beta,
//     const void*             B,
//     const void*             gamma,
//     const void*             C,
//     void*                   D,
//     aclrtStream             stream)
// {
//     // 参数验证
//     // 查询解决方案
//     // 初始化参数
//     // 执行内核
// }
//
// } // extern "C"
//
// ============================================================================

namespace acltensor {

// 声明算子解决方案注册函数（在各算子的 solution.cpp 中实现）
extern void RegisterAddF32Solutions(ElementwiseSolutionRegistry& registry);

/**
 * @brief 注册所有解决方案（延迟初始化时调用）
 *        参考 hiptensor 的延迟初始化模式
 */
void ElementwiseSolutionRegistry::registerAllSolutions()
{
    // 阶段一：注册 ADD + FP32 通用解决方案
    RegisterAddF32Solutions(*this);

    // TODO: Phase 2+ 注册更多算子和数据类型
    // RegisterMulF32Solutions(*this);
    // RegisterAddF16Solutions(*this);
    // RegisterTrinarySolutions(*this);
    // ...
}

} // namespace acltensor
