/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file validation.cpp
 * \brief 参数验证工具函数实现
 */

#include "validation.hpp"

namespace acltensor {

/**
 * @brief 检查操作符是否被当前阶段支持
 * @param op 操作符
 * @return ACLTENSOR_STATUS_SUCCESS - 支持
 *         ACLTENSOR_STATUS_NOT_SUPPORTED - 不支持
 */
acltensorStatus_t CheckOperator(acltensorOperator_t op)
{
    // 当前仅支持 IDENTITY 和 ADD，其他操作符待后续版本支持
    switch (op)
    {
        case ACLTENSOR_OP_IDENTITY:
        case ACLTENSOR_OP_ADD:
            return ACLTENSOR_STATUS_SUCCESS;

        default:
            return ACLTENSOR_STATUS_NOT_SUPPORTED;
    }
}

/**
 * @brief 检查计算描述符是否合法
 * @param descCompute 计算描述符
 * @return ACLTENSOR_STATUS_SUCCESS - 合法
 *         ACLTENSOR_STATUS_INVALID_VALUE - 参数为空
 *         ACLTENSOR_STATUS_NOT_SUPPORTED - 计算类型不支持
 */
acltensorStatus_t CheckComputeDescriptor(acltensorComputeDescriptor_t descCompute)
{
    // 检查是否为 NONE（无效值）
    if (descCompute == ACLTENSOR_COMPUTE_DESC_NONE || descCompute == 0)
    {
        return ACLTENSOR_STATUS_INVALID_VALUE;
    }

    // 当前仅支持 FP32 计算精度，其他数据类型待后续版本支持
    switch (descCompute)
    {
        case ACLTENSOR_COMPUTE_DESC_32F:
            return ACLTENSOR_STATUS_SUCCESS;

        default:
            return ACLTENSOR_STATUS_NOT_SUPPORTED;
    }
}

} // namespace acltensor
