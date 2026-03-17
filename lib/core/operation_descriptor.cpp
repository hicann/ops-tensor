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

#include "cann_ops_tensor.h"
#include "operation_descriptor.hpp"
#include "lib/utils/validation.hpp"
#include <cstring>
#include <new>

extern "C" {

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

    // ========== 第6步：显式设置所有字段（参考 hiptensor） ==========

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

/* TODO: Phase 2 - Elementwise Trinary */
acltensorStatus_t acltensorCreateElementwiseTrinary(
    const acltensorHandle_t            handle,
    acltensorOperationDescriptor_t*    desc,
    const acltensorTensorDescriptor_t  descA,
    const int32_t                      modeA[],
    acltensorOperator_t                opA,
    const acltensorTensorDescriptor_t  descB,
    const int32_t                      modeB[],
    acltensorOperator_t                opB,
    const acltensorTensorDescriptor_t  descC,
    const int32_t                      modeC[],
    acltensorOperator_t                opC,
    const acltensorTensorDescriptor_t  descD,
    const int32_t                      modeD[],
    acltensorOperator_t                opAB,
    acltensorOperator_t                opABC,
    const acltensorComputeDescriptor_t descCompute)
{
    (void)handle;
    (void)desc;
    (void)descA;
    (void)modeA;
    (void)opA;
    (void)descB;
    (void)modeB;
    (void)opB;
    (void)descC;
    (void)modeC;
    (void)opC;
    (void)descD;
    (void)modeD;
    (void)opAB;
    (void)opABC;
    (void)descCompute;
    return ACLTENSOR_STATUS_NOT_SUPPORTED;
}

/* TODO: Phase 3 - Contraction */
acltensorStatus_t acltensorCreateContraction(
    const acltensorHandle_t            handle,
    acltensorOperationDescriptor_t*    desc,
    const acltensorTensorDescriptor_t  descA,
    const int32_t                      modeA[],
    acltensorOperator_t                opA,
    const acltensorTensorDescriptor_t  descB,
    const int32_t                      modeB[],
    acltensorOperator_t                opB,
    const acltensorTensorDescriptor_t  descC,
    const int32_t                      modeC[],
    acltensorOperator_t                opC,
    const acltensorTensorDescriptor_t  descD,
    const int32_t                      modeD[],
    const acltensorComputeDescriptor_t descCompute)
{
    (void)handle;
    (void)desc;
    (void)descA;
    (void)modeA;
    (void)opA;
    (void)descB;
    (void)modeB;
    (void)opB;
    (void)descC;
    (void)modeC;
    (void)opC;
    (void)descD;
    (void)modeD;
    (void)descCompute;
    return ACLTENSOR_STATUS_NOT_SUPPORTED;
}

/* TODO: Phase 3 - Reduction */
acltensorStatus_t acltensorCreateReduction(
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
    acltensorOperator_t                opReduce,
    const acltensorComputeDescriptor_t descCompute)
{
    (void)handle;
    (void)desc;
    (void)descA;
    (void)modeA;
    (void)opA;
    (void)descC;
    (void)modeC;
    (void)opC;
    (void)descD;
    (void)modeD;
    (void)opReduce;
    (void)descCompute;
    return ACLTENSOR_STATUS_NOT_SUPPORTED;
}

/* TODO: Phase 4 - Permutation */
acltensorStatus_t acltensorCreatePermutation(
    const acltensorHandle_t            handle,
    acltensorOperationDescriptor_t*    desc,
    const acltensorTensorDescriptor_t  descA,
    const int32_t                      modeA[],
    acltensorOperator_t                opA,
    const acltensorTensorDescriptor_t  descB,
    const int32_t                      modeB[],
    const acltensorComputeDescriptor_t descCompute)
{
    (void)handle;
    (void)desc;
    (void)descA;
    (void)modeA;
    (void)opA;
    (void)descB;
    (void)modeB;
    (void)descCompute;
    return ACLTENSOR_STATUS_NOT_SUPPORTED;
}

/* TODO: Phase 2 - 属性接口 */
acltensorStatus_t acltensorOperationDescriptorSetAttribute(
    const acltensorHandle_t                 handle,
    acltensorOperationDescriptor_t          desc,
    acltensorOperationDescriptorAttribute_t attr,
    const void*                             buf,
    size_t                                  sizeInBytes)
{
    (void)handle;
    (void)desc;
    (void)attr;
    (void)buf;
    (void)sizeInBytes;
    return ACLTENSOR_STATUS_NOT_SUPPORTED;
}

acltensorStatus_t acltensorOperationDescriptorGetAttribute(
    const acltensorHandle_t                 handle,
    acltensorOperationDescriptor_t          desc,
    acltensorOperationDescriptorAttribute_t attr,
    void*                                   buf,
    size_t                                  sizeInBytes)
{
    (void)handle;
    (void)desc;
    (void)attr;
    (void)buf;
    (void)sizeInBytes;
    return ACLTENSOR_STATUS_NOT_SUPPORTED;
}

} // extern "C"
