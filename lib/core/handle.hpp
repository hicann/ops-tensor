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
 * \file handle.hpp
 * \brief Handle 管理头文件
 */

#ifndef ACLTENSOR_LIB_CORE_HANDLE_HPP
#define ACLTENSOR_LIB_CORE_HANDLE_HPP

#include "cann_ops_tensor_types.h"
#include <cstdint>

namespace acltensor {

/**
 * @brief SoC 类型枚举
 */
enum class SocType : uint32_t
{
    UNKNOWN = 0,
    ASCEND910B,
    ASCEND910_93,
    ASCEND950,      // 重点支持
};

/**
 * @brief Ascend 设备信息
 */
class AscendDevice {
public:
    AscendDevice();
    ~AscendDevice() = default;

    int32_t  getDeviceId()   const { return deviceId_; }
    uint32_t getCoreNum()    const { return coreNum_; }    // AIV 核数
    uint64_t getUbSize()     const { return ubSize_; }     // UB 大小 (字节)
    SocType  getSocType()    const { return socType_; }
    const char* getSocName() const { return socName_; }

    // 能力查询 - 阶段一只关注 FP32
    bool supportsFp32() const { return true; }

    /* TODO: Phase 2
    bool supportsFp16() const { return supportsFp16_; }
    bool supportsBf16() const { return supportsBf16_; }
    */

private:
    int32_t  deviceId_     = -1;
    uint32_t coreNum_      = 0;
    uint64_t ubSize_       = 0;
    SocType  socType_      = SocType::UNKNOWN;
    char     socName_[64]  = {0};
    // bool     supportsFp16_ = false;
};

} // namespace acltensor

/**
 * @brief 对外句柄结构体（阶段一简化版）
 */
struct acltensorHandle {
    acltensor::AscendDevice device;

    /* TODO: Phase 2 */
    // acltensor::PlanCache* planCache = nullptr;
    // acltensorLogLevel_t logLevel = ACLTENSOR_LOG_LEVEL_OFF;
};

#endif // ACLTENSOR_LIB_CORE_HANDLE_HPP
