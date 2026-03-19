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
 * \file validation.hpp
 * \brief 参数验证工具函数
 */

#ifndef ACLTENSOR_LIB_UTILS_VALIDATION_HPP
#define ACLTENSOR_LIB_UTILS_VALIDATION_HPP

#include "cann_ops_tensor_types.h"

namespace acltensor {

/**
 * @brief 检查操作符是否被当前阶段支持
 * @param op 操作符
 * @return ACLTENSOR_STATUS_SUCCESS - 支持
 *         ACLTENSOR_STATUS_NOT_SUPPORTED - 不支持
 */
acltensorStatus_t CheckOperator(acltensorOperator_t op);

/**
 * @brief 检查计算描述符是否合法
 * @param descCompute 计算描述符
 * @return ACLTENSOR_STATUS_SUCCESS - 合法
 *         ACLTENSOR_STATUS_INVALID_VALUE - 参数为空
 *         ACLTENSOR_STATUS_NOT_SUPPORTED - 计算类型不支持
 */
acltensorStatus_t CheckComputeDescriptor(acltensorComputeDescriptor_t descCompute);

} // namespace acltensor

#endif // ACLTENSOR_LIB_UTILS_VALIDATION_HPP
