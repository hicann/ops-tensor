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
 * \file tensor_descriptor.cpp
 * \brief 张量描述符实现
 */

#include "cann_ops_tensor_types.h"
#include "tensor_descriptor.hpp"
#include <cstring>
#include <new>

extern "C" {

acltensorStatus_t acltensorCreateTensorDescriptor(
    const acltensorHandle_t      handle,
    acltensorTensorDescriptor_t* desc,
    const uint32_t               numModes,
    const int64_t                lens[],
    const int64_t                strides[],
    acltensorDataType_t          dataType,
    uint32_t                     alignmentRequirement)
{
    (void)handle;  // 暂时不使用 handle

    if (desc == nullptr || lens == nullptr) {
        return ACLTENSOR_STATUS_INVALID_VALUE;
    }

    if (numModes == 0) {
        return ACLTENSOR_STATUS_INVALID_VALUE;
    }

    // 阶段一只支持 FP32
    if (dataType != ACLTENSOR_R_32F) {
        return ACLTENSOR_STATUS_NOT_SUPPORTED;
    }

    // 创建描述符
    acltensorTensorDescriptor* d = new (std::nothrow) acltensorTensorDescriptor();
    if (d == nullptr) {
        return ACLTENSOR_STATUS_ALLOC_FAILED;
    }

    // 设置基本属性
    d->dataType = dataType;
    d->numModes = numModes;
    d->alignmentRequirement = alignmentRequirement;

    // 复制维度长度
    d->lens.resize(numModes);
    std::memcpy(d->lens.data(), lens, numModes * sizeof(int64_t));

    // ========== 步长处理：用户指定 vs 自动计算 ==========
    if (strides != nullptr) {
        // 用户显式提供步长：直接赋值
        d->strides.resize(numModes);
        std::memcpy(d->strides.data(), strides, numModes * sizeof(int64_t));
    } else {
        // 用户未提供步长：自动计算连续内存布局
        d->strides.resize(numModes);
        if (numModes > 0) {
            d->strides[numModes - 1] = 1;
            for (int32_t i = numModes - 2; i >= 0; --i) {
                d->strides[i] = d->strides[i + 1] * lens[i + 1];
            }
        }
    }

    // 计算派生属性
    d->computeDerivedAttributes();

    *desc = d;
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

} // extern "C"
