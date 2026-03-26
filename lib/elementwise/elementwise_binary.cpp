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
 * \file elementwise_binary.cpp
 * \brief Elementwise Binary 框架实现
 */

#include "elementwise.hpp"

#include "cann_ops_tensor.h"
#include "core/plan.hpp"

using namespace acltensor;

/**
 * @brief 执行 Elementwise Binary 操作
 */
acltensorStatus_t acltensorElementwiseBinaryExecute(const acltensorHandle_t handle, const acltensorPlan_t plan,
    const void* alpha, const void* A, const void* gamma, const void* C, void* D, aclrtStream stream)
{
    (void)handle;

    if (plan == nullptr || A == nullptr || C == nullptr || D == nullptr) {
        return ACLTENSOR_STATUS_INVALID_VALUE;
    }

    auto* opDesc = plan->opDesc;
    auto* pref = plan->pref;
    if (opDesc == nullptr || pref == nullptr) {
        return ACLTENSOR_STATUS_INVALID_VALUE;
    }

    if (opDesc->operationType != acltensor::OperationType::ELEMENTWISE_BINARY) {
        return ACLTENSOR_STATUS_NOT_SUPPORTED;
    }

    // Execute 时选择并缓存 solution
    std::shared_ptr<ElementwiseSolution> solution;
    if (pref->solution != nullptr) {
        solution = std::static_pointer_cast<ElementwiseSolution>(pref->solution);
    } else {
        auto& registry = ElementwiseSolutionRegistry::instance();
        auto* descD = opDesc->descD;
        uint32_t numModes = descD ? descD->numModes : 0;

        auto solutions = registry.getSolutions(opDesc->opAC, descD->dataType, numModes, opDesc->operationType);
        if (solutions.empty()) {
            solutions = registry.getSolutions(opDesc->opAC, descD->dataType, 0, opDesc->operationType);
        }

        if (solutions.empty())
            return ACLTENSOR_STATUS_NOT_SUPPORTED;

        solution = solutions[0];
        pref->solution = solution;
    }

    ElementwiseArgs args = CreateElementwiseBinaryArgs(opDesc, alpha, A, gamma, C, D, stream);
    return solution->execute(args);
}

// 三元元素级操作执行：acltensorElementwiseTrinaryExecute - 待后续版本实现
