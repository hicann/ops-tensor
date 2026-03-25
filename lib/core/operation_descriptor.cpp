/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file operation_descriptor.cpp
 * @brief 操作描述符实现
 */

#include "operation_descriptor.hpp"

#include <cstring>
#include <new>

#include "cann_ops_tensor.h"
#include "utils/validation.hpp"

acltensorStatus_t acltensorCreateElementwiseBinary(
    const acltensorHandle_t            handle,
    acltensorOperationDescriptor_t*    desc,
    const acltensorTensorDescriptor_t  descA,
    const int32_t                      modeA[],
    acltensorOperator_t                opA,
    const acltensorTensorDescriptor_t  descC,
    const int32_t                      modeC[],
    acltensorOperator_t                opC,
    const acltensorTensorDescriptor_t  descD,
    const int32_t                      modeD[],
    acltensorOperator_t                opAC,
    const acltensorComputeDescriptor_t descCompute)
{
    (void)handle;

    // ========== 第1步：参数验证 ==========
    if (desc == nullptr) {
        return ACLTENSOR_STATUS_INVALID_VALUE;
    }

    if (descA == nullptr || descC == nullptr || descD == nullptr) {
        return ACLTENSOR_STATUS_INVALID_VALUE;
    }

    if (modeA == nullptr || modeC == nullptr || modeD == nullptr) {
        return ACLTENSOR_STATUS_INVALID_VALUE;
    }

    // ========== 第2步：操作符验证 ==========
    acltensorStatus_t status = acltensor::CheckOperator(opA);
    if (status != ACLTENSOR_STATUS_SUCCESS) {
        return status;
    }

    status = acltensor::CheckOperator(opC);
    if (status != ACLTENSOR_STATUS_SUCCESS) {
        return status;
    }

    status = acltensor::CheckOperator(opAC);
    if (status != ACLTENSOR_STATUS_SUCCESS) {
        return status;
    }

    // ========== 第3步：计算精度验证 ==========
    status = acltensor::CheckComputeDescriptor(descCompute);
    if (status != ACLTENSOR_STATUS_SUCCESS) {
        return status;
    }

    // ========== 第4步：分配操作描述符 ==========
    acltensorOperationDescriptor* opDesc = new (std::nothrow) acltensorOperationDescriptor();
    if (opDesc == nullptr) {
        return ACLTENSOR_STATUS_ALLOC_FAILED;
    }

    // ========== 第5步：显式设置所有字段 ==========

    // 设置操作类型
    opDesc->operationType = acltensor::OperationType::ELEMENTWISE_BINARY;

    // ========== 设置张量描述符 ==========
    opDesc->descA = descA;
    opDesc->descC = descC;
    opDesc->descD = descD;
    opDesc->descB = nullptr;  // Binary 操作不使用，显式设置为 nullptr

    // ========== 设置模式数组 ==========
    opDesc->modeA.assign(modeA, modeA + descA->numModes);
    opDesc->modeC.assign(modeC, modeC + descC->numModes);
    opDesc->modeD.assign(modeD, modeD + descD->numModes);
    opDesc->modeB.clear();  // Binary 操作不使用，显式清空

    // ========== 设置操作符 ==========
    opDesc->opA  = opA;
    opDesc->opC  = opC;
    opDesc->opAC = opAC;
    opDesc->opB   = ACLTENSOR_OP_IDENTITY;  // Binary 操作不使用，显式设置默认值
    opDesc->opAB  = ACLTENSOR_OP_ADD;        // Binary 操作不使用，显式设置默认值
    opDesc->opABC = ACLTENSOR_OP_IDENTITY;  // Binary 操作不使用，显式设置默认值

    // ========== 设置计算精度 ==========
    opDesc->descCompute = descCompute;

    *desc = opDesc;
    return ACLTENSOR_STATUS_SUCCESS;
}

acltensorStatus_t acltensorDestroyOperationDescriptor(acltensorOperationDescriptor_t desc)
{
    if (desc == nullptr) {
        return ACLTENSOR_STATUS_SUCCESS;
    }

    // 不销毁关联的张量描述符，它们由用户管理
    delete desc;
    return ACLTENSOR_STATUS_SUCCESS;
}

// 以下操作类型创建接口待后续版本实现：
// 三元元素级操作：acltensorCreateElementwiseTrinary
// 张量收缩操作：acltensorCreateContraction
// 归约操作：acltensorCreateReduction
// 置换操作：acltensorCreatePermutation
// 属性设置：acltensorOperationDescriptorSetAttribute
// 属性获取：acltensorOperationDescriptorGetAttribute
