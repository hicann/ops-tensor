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
 * \file plan.cpp
 * \brief Plan 实现
 */

#include "cann_ops_tensor.h"
#include "plan.hpp"
#include "plan_preference.hpp"
#include "operation_descriptor.hpp"
#include "lib/elementwise/elementwise.hpp"
#include <new>
#include <cstring>  // for memcpy_s (C11 安全内存拷贝)

extern "C" {

acltensorStatus_t acltensorCreatePlan(
    const acltensorHandle_t              handle,
    acltensorPlan_t*                     plan,
    const acltensorOperationDescriptor_t desc,
    const acltensorPlanPreference_t      pref,
    uint64_t                             workspaceSizeLimit)
{
    (void)handle;       // 暂时不使用
    (void)workspaceSizeLimit;  // Elementwise 不需要 workspace

    if (plan == nullptr || desc == nullptr || pref == nullptr)
    {
        return ACLTENSOR_STATUS_INVALID_VALUE;
    }

    // 阶段一只支持 Elementwise Binary
    if (desc->operationType != acltensor::OperationType::ELEMENTWISE_BINARY)
    {
        return ACLTENSOR_STATUS_NOT_SUPPORTED;
    }

    // 创建 Plan
    acltensorPlan* p = new (std::nothrow) acltensorPlan();
    if (p == nullptr)
    {
        return ACLTENSOR_STATUS_ALLOC_FAILED;
    }

    p->opDesc = desc;
    p->pref = pref;
    p->requiredWorkspace = 0;

    // ==========对标 hiptensor：CreatePlan 时不选择 solution ==========
    // solution 将在 Execute 时从注册表查询并缓存到 pref->cachedSolution
    // 这样设计保持 Plan 为不可变对象，Pref 承担执行状态

    *plan = p;
    return ACLTENSOR_STATUS_SUCCESS;
}

acltensorStatus_t acltensorDestroyPlan(acltensorPlan_t plan)
{
    if (plan == nullptr)
    {
        return ACLTENSOR_STATUS_SUCCESS;
    }

    // solution 由注册表管理，无需释放
    delete plan;
    return ACLTENSOR_STATUS_SUCCESS;
}

/* TODO: Phase 2 */
acltensorStatus_t acltensorEstimateWorkspaceSize(
    const acltensorHandle_t              handle,
    const acltensorOperationDescriptor_t desc,
    const acltensorPlanPreference_t      planPref,
    const acltensorWorksizePreference_t  workspacePref,
    uint64_t*                            workspaceSizeEstimate)
{
    (void)handle;
    (void)desc;
    (void)planPref;
    (void)workspacePref;

    if (workspaceSizeEstimate == nullptr)
    {
        return ACLTENSOR_STATUS_INVALID_VALUE;
    }

    // Elementwise 操作不需要 workspace
    *workspaceSizeEstimate = 0;
    return ACLTENSOR_STATUS_SUCCESS;
}

acltensorStatus_t acltensorPlanGetAttribute(
    const acltensorHandle_t  handle,
    const acltensorPlan_t    plan,
    acltensorPlanAttribute_t attr,
    void*                    buf,
    size_t                   sizeInBytes)
{
    (void)handle;

    if (plan == nullptr || buf == nullptr)
    {
        return ACLTENSOR_STATUS_INVALID_VALUE;
    }

    switch (attr)
    {
        case ACLTENSOR_PLAN_REQUIRED_WORKSPACE:
        {
            // 使用 memcpy_s（对标 opbase，更安全的内存拷贝）
            // memcpy_s 会检查目标缓冲区大小，防止溢出
            errno_t ret = memcpy_s(buf, sizeInBytes, &plan->requiredWorkspace, sizeof(uint64_t));
            if (ret != EOK) {
                return ACLTENSOR_STATUS_INVALID_VALUE;
            }
            return ACLTENSOR_STATUS_SUCCESS;
        }

        case ACLTENSOR_PLAN_KERNEL_ID:
        {
            // 从 pref->solution 获取 Kernel ID
            uint64_t kernelId = 0;
            if (plan->pref && plan->pref->solution) {
                acltensor::ElementwiseBinarySolution* solution =
                    static_cast<acltensor::ElementwiseBinarySolution*>(plan->pref->solution);
                kernelId = solution->getSolutionId();
            }
            // 使用 memcpy_s（对标 opbase，更安全的内存拷贝）
            errno_t ret = memcpy_s(buf, sizeInBytes, &kernelId, sizeof(uint64_t));
            if (ret != EOK) {
                return ACLTENSOR_STATUS_INVALID_VALUE;
            }
            return ACLTENSOR_STATUS_SUCCESS;
        }

        default:
            return ACLTENSOR_STATUS_NOT_SUPPORTED;
    }
}

} // extern "C"
