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
 * \file plan_preference.hpp
 * \brief Plan 偏好头文件
 */

#ifndef ACLTENSOR_LIB_CORE_PLAN_PREFERENCE_HPP
#define ACLTENSOR_LIB_CORE_PLAN_PREFERENCE_HPP

#include <memory>   // for std::shared_ptr (solution 字段)
#include <vector>   // for std::vector (candidates 字段)

#include "cann_ops_tensor_types.h"

/**
 * @brief Plan 偏好结构体
 *
 * PlanPreference 的定位：Solution 选择器
 * - 用户偏好：algo（算法选择策略）
 * - Solution 选择状态：solution（当前选定的解决方案）、candidates（候选列表）
 * - 用户通过 Preference 表达算法偏好
 * - 库根据偏好选择 Solution，并存储在 Preference 中
 */
struct acltensorPlanPreference {
    // ========== 用户偏好 ==========
    acltensorAlgo_t algo = ACLTENSOR_ALGO_DEFAULT;
    // cacheMode、autotuneMode、incrementalCount、kernelRank 等待后续版本实现

    // ========== Solution 选择状态 ==========
    // 候选解决方案列表（Autotune 功能待后续版本实现）
    std::vector<void*> candidates;

    // 当前选定的解决方案（Execute 时首次选择，后续复用）
    // 使用 shared_ptr 确保 solution 对象生命周期
    std::shared_ptr<void> solution = nullptr;
};

#endif // ACLTENSOR_LIB_CORE_PLAN_PREFERENCE_HPP
