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
 * \file elementwise_binary.cpp
 * \brief Elementwise Binary 框架实现
 */

#include "cann_ops_tensor.h"
#include "elementwise.hpp"
#include "core/plan.hpp"

using namespace acltensor;

/**
 * @brief 执行 Elementwise Binary 操作
 */
acltensorStatus_t acltensorElementwiseBinaryExecute(
    const acltensorHandle_t handle,
    const acltensorPlan_t   plan,
    const void*             alpha,
    const void*             A,
    const void*             gamma,
    const void*             C,
    void*                   D,
    aclrtStream             stream)
{
    (void)handle;  // 暂时不使用

    if (plan == nullptr)
    {
        return ACLTENSOR_STATUS_INVALID_VALUE;
    }

    if (A == nullptr || C == nullptr || D == nullptr)
    {
        return ACLTENSOR_STATUS_INVALID_VALUE;
    }

    if (plan->opDesc == nullptr || plan->pref == nullptr)
    {
        return ACLTENSOR_STATUS_INVALID_VALUE;
    }

    // 阶段一：只支持 Elementwise Binary
    if (plan->opDesc->operationType != acltensor::OperationType::ELEMENTWISE_BINARY)
    {
        return ACLTENSOR_STATUS_NOT_SUPPORTED;
    }

    // ========== Execute 时选择并缓存 solution ==========

    // 1. 检查是否已有缓存的 solution
    std::shared_ptr<ElementwiseSolution> solution;
    if (plan->pref->solution != nullptr) {
        solution = std::static_pointer_cast<ElementwiseSolution>(plan->pref->solution);
    } else {
        // 2. 首次执行，从注册表查询解决方案
        auto& registry = ElementwiseSolutionRegistry::instance();

        uint32_t numModes = plan->opDesc->descD ? plan->opDesc->descD->numModes : 0;
        auto solutions = registry.getSolutions(
            plan->opDesc->opAC,
            plan->opDesc->descD->dataType,
            numModes);

        if (solutions.empty()) {
            // 尝试查找通用解决方案（numModes=0）
            solutions = registry.getSolutions(
                plan->opDesc->opAC,
                plan->opDesc->descD->dataType,
                0);
        }

        if (solutions.empty()) {
            return ACLTENSOR_STATUS_NOT_SUPPORTED;
        }

        // 3. 选择第一个并缓存到 pref->solution
        solution = solutions[0];
        plan->pref->solution = solution;
    }

    // ========== 构造 ElementwiseArgs ==========
    ElementwiseArgs args = CreateElementwiseBinaryArgs(
        plan->opDesc, alpha, A, gamma, C, D, stream);

    // 执行内核（调用新的统一接口）
    return solution->execute(args);
}

/* Phase 2 - Elementwise Trinary Execute 待实现 */
/*
acltensorStatus_t acltensorElementwiseTrinaryExecute(
    const acltensorHandle_t handle,
    const acltensorPlan_t   plan,
    const void*             alpha,
    const void*             A,
    const void*             beta,
    const void*             B,
    const void*             gamma,
    const void*             C,
    void*                   D,
    aclrtStream             stream)
{
    (void)handle;
    (void)plan;
    (void)alpha;
    (void)A;
    (void)beta;
    (void)B;
    (void)gamma;
    (void)C;
    (void)D;
    (void)stream;
    return ACLTENSOR_STATUS_NOT_SUPPORTED;
}
*/
