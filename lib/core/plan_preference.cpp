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
 * \file plan_preference.cpp
 * \brief Plan 偏好实现
 */

#include "plan_preference.hpp"

#include <new>

#include "cann_ops_tensor.h"

acltensorStatus_t acltensorCreatePlanPreference(
    const acltensorHandle_t    handle,
    acltensorPlanPreference_t* pref,
    acltensorAlgo_t            algo)
{
    (void)handle;  // 暂时不使用 handle

    if (pref == nullptr) {
        return ACLTENSOR_STATUS_INVALID_VALUE;
    }

    // 阶段一只支持 DEFAULT 算法
    if (algo != ACLTENSOR_ALGO_DEFAULT) {
        return ACLTENSOR_STATUS_NOT_SUPPORTED;
    }

    acltensorPlanPreference* p = new (std::nothrow) acltensorPlanPreference();
    if (p == nullptr) {
        return ACLTENSOR_STATUS_ALLOC_FAILED;
    }

    p->algo = algo;
    *pref = p;
    return ACLTENSOR_STATUS_SUCCESS;
}

acltensorStatus_t acltensorDestroyPlanPreference(acltensorPlanPreference_t pref)
{
    if (pref == nullptr) {
        return ACLTENSOR_STATUS_SUCCESS;
    }

    delete pref;
    return ACLTENSOR_STATUS_SUCCESS;
}

// 计划偏好设置属性：acltensorPlanPreferenceSetAttribute - 待后续版本实现
