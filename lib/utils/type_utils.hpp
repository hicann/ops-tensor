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
 * \file type_utils.hpp
 * \brief 类型转换工具
 */

#ifndef ACLTENSOR_LIB_UTILS_TYPE_UTILS_HPP
#define ACLTENSOR_LIB_UTILS_TYPE_UTILS_HPP

#include "cann_ops_tensor_types.h"
#include <cstddef>
#include <cstdint>

namespace acltensor {

/**
 * @brief 获取数据类型对应的元素大小（字节）
 *        使用 sizeof 计算实际类型大小，避免硬编码
 */
inline size_t GetDataTypeSize(acltensorDataType_t dataType)
{
    // 复数类型包含实部和虚部两个分量
    constexpr size_t COMPLEX_COMPONENTS = 2;

    switch (dataType) {
        case ACLTENSOR_R_16F:       // FP16 半精度浮点
            return sizeof(int16_t);
        case ACLTENSOR_R_16BF:      // BF16 BFloat16
            return sizeof(int16_t);
        case ACLTENSOR_R_32F:       // FP32 单精度浮点
            return sizeof(float);
        case ACLTENSOR_R_64F:       // FP64 双精度浮点
            return sizeof(double);
        case ACLTENSOR_R_8I:        // INT8 有符号8位整数
            return sizeof(int8_t);
        case ACLTENSOR_R_8U:        // UINT8 无符号8位整数
            return sizeof(uint8_t);
        case ACLTENSOR_R_32I:       // INT32 有符号32位整数
            return sizeof(int32_t);
        case ACLTENSOR_R_32U:       // UINT32 无符号32位整数
            return sizeof(uint32_t);
        case ACLTENSOR_C_32F:       // Complex FP32 (实部 + 虚部)
            return COMPLEX_COMPONENTS * sizeof(float);
        case ACLTENSOR_C_64F:       // Complex FP64 (实部 + 虚部)
            return COMPLEX_COMPONENTS * sizeof(double);
        default:
            return 0;
    }
}

/**
 * @brief 检查数据类型是否为浮点类型
 */
inline bool IsFloatingPointType(acltensorDataType_t dataType)
{
    return dataType == ACLTENSOR_R_16F ||
           dataType == ACLTENSOR_R_32F ||
           dataType == ACLTENSOR_R_64F ||
           dataType == ACLTENSOR_R_16BF;
}

/**
 * @brief 检查数据类型是否为复数类型
 */
inline bool IsComplexType(acltensorDataType_t dataType)
{
    return dataType == ACLTENSOR_C_32F ||
           dataType == ACLTENSOR_C_64F;
}

/**
 * @brief 检查数据类型是否为整数类型
 */
inline bool IsIntegerType(acltensorDataType_t dataType)
{
    return dataType == ACLTENSOR_R_8I ||
           dataType == ACLTENSOR_R_8U ||
           dataType == ACLTENSOR_R_32I ||
           dataType == ACLTENSOR_R_32U;
}

} // namespace acltensor

#endif // ACLTENSOR_LIB_UTILS_TYPE_UTILS_HPP
