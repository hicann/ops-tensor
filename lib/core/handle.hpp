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

#include <cstdint>

#include "cann_ops_tensor_types.h"

namespace acltensor {

/**
 * @brief SoC 类型枚举
 *
 * 注意：当前版本仅支持 Ascend950，其他枚举项暂不支持。
 */
enum class SocType : uint32_t
{
    UNKNOWN = 0,
    // ASCEND910B - 暂不支持，保留用于未来扩展
    // ASCEND910_93 - 暂不支持，保留用于未来扩展
    ASCEND950       // 当前版本支持
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

    // 能力查询 - 当前仅支持 FP32
    bool supportsFp32() const { return true; }
    // FP16、BF16 等其他数据类型支持待后续版本实现

private:
    int32_t  deviceId_     = -1;
    uint32_t coreNum_      = 0;
    uint64_t ubSize_       = 0;
    SocType  socType_      = SocType::UNKNOWN;
    char     socName_[64]  = {0};  // SOC名称缓冲区，64字节足以存储 "Ascend950" 等型号名称
    // bool     supportsFp16_ = false;
};

} // namespace acltensor

/**
 * @brief 对外句柄结构体
 */
struct acltensorHandle {
    acltensor::AscendDevice device;
    // PlanCache、logLevel 等功能待后续版本实现
};

#endif // ACLTENSOR_LIB_CORE_HANDLE_HPP
