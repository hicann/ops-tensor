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
 * \file handle.cpp
 * \brief Handle 管理实现
 */

#include "handle.hpp"
#include "cann_ops_tensor.h"
#include "acl/acl.h"
#include "lib/elementwise/elementwise_binary.hpp"
#include "tiling/platform/platform_ascendc.h"
#include <cstring>

namespace acltensor {

AscendDevice::AscendDevice()
{
    // 获取当前设备 ID
    aclError aclRet = aclrtGetDevice(&deviceId_);
    if (aclRet != ACL_SUCCESS)
    {
        // 设备未初始化，使用默认值
        deviceId_ = 0;
    }

    // 使用 AscendC Platform 接口获取设备信息
    auto platform = platform_ascendc::PlatformAscendC();

    // 获取 AIV 核数
    coreNum_ = platform.GetCoreNumAiv();

    // 获取 UB 大小
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);

    // 获取 SoC 版本
    auto socVersion = platform.GetSocVersion();
    switch (socVersion)
    {
        case platform_ascendc::SocVersion::ASCEND910B:
            strncpy(socName_, "Ascend910B", sizeof(socName_) - 1);
            socType_ = SocType::ASCEND910B;
            break;
        case platform_ascendc::SocVersion::ASCEND910_93:
            strncpy(socName_, "Ascend910-93", sizeof(socName_) - 1);
            socType_ = SocType::ASCEND910_93;
            break;
        case platform_ascendc::SocVersion::ASCEND950:
            strncpy(socName_, "Ascend950", sizeof(socName_) - 1);
            socType_ = SocType::ASCEND950;
            break;
        default:
            strncpy(socName_, "Unknown", sizeof(socName_) - 1);
            socType_ = SocType::UNKNOWN;
            break;
    }

    socName_[sizeof(socName_) - 1] = '\0';
}

} // namespace acltensor

extern "C" {

acltensorStatus_t acltensorCreate(acltensorHandle_t* handle)
{
    if (handle == nullptr)
    {
        return ACLTENSOR_STATUS_INVALID_VALUE;
    }

    acltensorHandle* h = new (std::nothrow) acltensorHandle();
    if (h == nullptr)
    {
        return ACLTENSOR_STATUS_ALLOC_FAILED;
    }

    // 设备信息在 AscendDevice 构造函数中初始化
    // 解决方案注册表采用延迟初始化模式，首次使用时自动注册
    *handle = h;
    return ACLTENSOR_STATUS_SUCCESS;
}

acltensorStatus_t acltensorDestroy(acltensorHandle_t handle)
{
    if (handle == nullptr)
    {
        return ACLTENSOR_STATUS_SUCCESS;
    }

    delete handle;
    return ACLTENSOR_STATUS_SUCCESS;
}

/* TODO: Phase 2 - PlanCache 相关接口 */
acltensorStatus_t acltensorHandleResizePlanCache(
    acltensorHandle_t handle, const uint32_t numEntries)
{
    (void)handle;
    (void)numEntries;
    return ACLTENSOR_STATUS_NOT_SUPPORTED;
}

acltensorStatus_t acltensorHandleWritePlanCacheToFile(
    const acltensorHandle_t handle, const char fileName[])
{
    (void)handle;
    (void)fileName;
    return ACLTENSOR_STATUS_NOT_SUPPORTED;
}

acltensorStatus_t acltensorHandleReadPlanCacheFromFile(
    acltensorHandle_t handle, const char fileName[], uint32_t* numCachelinesRead)
{
    (void)handle;
    (void)fileName;
    (void)numCachelinesRead;
    return ACLTENSOR_STATUS_NOT_SUPPORTED;
}

} // extern "C"
