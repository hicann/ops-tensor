/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tensor_descriptor.cpp
 * \brief 张量描述符实现
 */

#include "tensor_descriptor.hpp"

#include <cstring>
#include <new>

#include "cann_ops_tensor_types.h"

acltensorStatus_t acltensorCreateTensorDescriptor(
    const acltensorHandle_t      handle,
    acltensorTensorDescriptor_t* desc,
    const uint32_t               rank,
    const int64_t                dimSizes[],
    const int64_t                stridesIn[],
    acltensorDataType_t          dType,
    uint32_t                     alignReq)
{
    (void)handle;

    if (desc == nullptr || dimSizes == nullptr) {
        return ACLTENSOR_STATUS_INVALID_VALUE;
    }

    if (rank == 0) {
        return ACLTENSOR_STATUS_INVALID_VALUE;
    }

    // 当前版本仅支持 FP32
    if (dType != ACLTENSOR_R_32F) {
        return ACLTENSOR_STATUS_NOT_SUPPORTED;
    }

    // 分配张量描述符对象
    acltensorTensorDescriptor* tensorDesc = new (std::nothrow) acltensorTensorDescriptor();
    if (tensorDesc == nullptr) {
        return ACLTENSOR_STATUS_ALLOC_FAILED;
    }

    // 初始化张量属性
    tensorDesc->dataType = dType;
    tensorDesc->numModes = rank;
    tensorDesc->alignmentRequirement = alignReq;

    // 拷贝维度信息
    tensorDesc->lens.assign(dimSizes, dimSizes + rank);

    // 用户提供了步长信息则直接使用
    if (stridesIn != nullptr) {
        tensorDesc->strides.assign(stridesIn, stridesIn + rank);
    }

    // 计算派生属性（元素总数、内存连续性、总字节数等）
    tensorDesc->computeDerivedAttributes();

    *desc = tensorDesc;
    return ACLTENSOR_STATUS_SUCCESS;
}

acltensorStatus_t acltensorDestroyTensorDescriptor(acltensorTensorDescriptor_t desc)
{
    if (desc == nullptr) {
        return ACLTENSOR_STATUS_SUCCESS;
    }

    delete desc;
    return ACLTENSOR_STATUS_SUCCESS;
}
