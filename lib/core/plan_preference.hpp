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

#include "cann_ops_tensor_types.h"
#include <memory>   // for std::shared_ptr (solution 字段)
#include <vector>   // for std::vector (candidates 字段)

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

    /* TODO: Phase 2
    acltensorCacheMode_t cacheMode = ACLTENSOR_CACHE_MODE_NONE;
    acltensorAutotuneMode_t autotuneMode = ACLTENSOR_AUTOTUNE_MODE_NONE;
    int32_t incrementalCount = 0;
    int32_t kernelRank = 0;
    */

    // ========== Solution 选择状态 ==========
    // 候选解决方案列表（Phase 3 Autotune 时使用）
    // CreatePlan 时由库填充，Execute 时根据 autotuneMode 遍历选择
    std::vector<void*> candidates;

    // 当前选定的解决方案（Execute 时首次选择，后续复用）
    // 使用 shared_ptr 确保 solution 对象生命周期
    std::shared_ptr<void> solution = nullptr;
};

#endif // ACLTENSOR_LIB_CORE_PLAN_PREFERENCE_HPP
