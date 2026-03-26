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
 * \file tensor_descriptor.hpp
 * \brief 张量描述符头文件
 */

#ifndef ACLTENSOR_LIB_CORE_TENSOR_DESCRIPTOR_HPP
#define ACLTENSOR_LIB_CORE_TENSOR_DESCRIPTOR_HPP

#include <vector>
#include <cstdint>

#include "cann_ops_tensor_types.h"
#include "utils/type_utils.hpp"

/**
 * @brief 张量描述符结构体
 */
struct acltensorTensorDescriptor {
    // ========== 基本属性 ==========
    acltensorDataType_t  dataType;              // 数据类型
    uint32_t             numModes;            // 维度数量（用 numModes 而非 rank，强调"模式"的多用途性）
    std::vector<int64_t> lens;               // 各维度长度
    std::vector<int64_t> strides;            // 各维度步长
    uint32_t             alignmentRequirement;  // 内存对齐要求

    // 派生属性（创建时计算）
    size_t               elementSize;    // 单元素字节数
    size_t               totalElements;  // 总元素数
    size_t               totalBytes;     // 总字节数
    bool                 isContiguous;   // 是否连续内存

    /**
     * @brief 计算派生属性
     */
    void computeDerivedAttributes()
    {
        elementSize = acltensor::GetDataTypeSize(dataType);

        // 计算总元素数
        totalElements = 1;
        for (uint32_t i = 0; i < numModes; ++i) {
            totalElements *= lens[i];
        }

        // 如果没有提供步长，计算连续内存的步长
        if (strides.empty() || strides.size() != numModes) {
            strides.resize(numModes);
            if (numModes > 0) {
                strides[numModes - 1] = 1;
                for (int32_t i = numModes - 2; i >= 0; --i) {
                    strides[i] = strides[i + 1] * lens[i + 1];
                }
            }
            isContiguous = true;
        } else {
            // 检查是否连续
            isContiguous = checkContiguous();
        }

        // 计算总字节数
        totalBytes = totalElements * elementSize;
    }

    /**
     * @brief 检查是否为连续内存布局
     */
    bool checkContiguous() const
    {
        if (numModes == 0) {
            return true;
        }

        int64_t expectedStride = 1;
        for (int32_t i = numModes - 1; i >= 0; --i) {
            if (strides[i] != expectedStride) {
                return false;
            }
            expectedStride *= lens[i];
        }
        return true;
    }
};

#endif // ACLTENSOR_LIB_CORE_TENSOR_DESCRIPTOR_HPP
