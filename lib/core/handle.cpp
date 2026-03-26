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

#include <cstring>
#include "securec.h"

#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/soc_spec.h"
#include "cann_ops_tensor.h"

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
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();

    // 获取 AIV 核数
    coreNum_ = ascendcPlatform->GetCoreNumAiv();

    // 获取 UB 大小
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);

    // 获取 SoC 版本
    // 注意：当前版本仅支持 Ascend950
    auto npuArch = ascendcPlatform->GetCurNpuArch();
    const char* socNameStr = nullptr;
    switch (npuArch)
    {
        // 其他 SoC 型号暂不支持，待后续版本扩展
        case NpuArch::DAV_3510:
            socNameStr = "Ascend950";
            socType_ = SocType::ASCEND950;
            break;
        default:
            socNameStr = "Unknown";
            socType_ = SocType::UNKNOWN;
            break;
    }

    // 统一拷贝 SoC 名称
    if (socNameStr != nullptr) {
        int ret = memcpy_s(socName_, sizeof(socName_), socNameStr, strlen(socNameStr) + 1);
        if (ret != EOK) {
            socName_[0] = '\0';  // 复制失败时清空字符串
        }
    }
}

} // namespace acltensor

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

// 以下接口待后续版本实现：
// 计划缓存调整：acltensorHandleResizePlanCache
// 计划缓存写入文件：acltensorHandleWritePlanCacheToFile
// 计划缓存读取文件：acltensorHandleReadPlanCacheFromFile
