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
 * \file plan.hpp
 * \brief Plan 头文件
 */

#ifndef ACLTENSOR_LIB_CORE_PLAN_HPP
#define ACLTENSOR_LIB_CORE_PLAN_HPP

#include "cann_ops_tensor_types.h"
#include "operation_descriptor.hpp"
#include "plan_preference.hpp"
#include <memory>

/**
 * @brief Plan 结构体
 *
 * Plan 是执行框架，包含操作描述和偏好引用
 * - opDesc：操作描述（输入/输出张量、操作符）
 * - pref：执行偏好（算法选择、缓存状态）
 * - requiredWorkspace：所需工作空间大小
 *
 * 注意：solution 不存储在 Plan 中，而是存储在 pref->cachedSolution 中
 * 这样设计保持 Plan 为不可变对象，Pref 承担执行状态
 */
struct acltensorPlan {
    acltensorOperationDescriptor_t opDesc = nullptr;  // 操作描述（不可变）
    acltensorPlanPreference_t      pref = nullptr;    // 执行偏好（包含缓存状态）
    uint64_t                        requiredWorkspace = 0;  // 所需工作空间

    // 辅助方法
    size_t getTotalElements() const {
        if (opDesc != nullptr) {
            return opDesc->getTotalElements();
        }
        return 0;
    }
};

#endif // ACLTENSOR_LIB_CORE_PLAN_HPP
