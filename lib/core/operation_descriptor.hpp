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
 * \file operation_descriptor.hpp
 * \brief 操作描述符头文件
 */

#ifndef ACLTENSOR_LIB_CORE_OPERATION_DESCRIPTOR_HPP
#define ACLTENSOR_LIB_CORE_OPERATION_DESCRIPTOR_HPP

#include <vector>

#include "cann_ops_tensor_types.h"
#include "tensor_descriptor.hpp"

namespace acltensor {

/**
 * @brief 操作类型枚举
 */
enum class OperationType : uint32_t {
    ELEMENTWISE_BINARY = 0,   // 当前版本支持
    ELEMENTWISE_TRINARY = 1,  // 待后续版本实现
    // CONTRACTION、REDUCTION、PERMUTATION 等操作类型待后续版本实现
};

} // namespace acltensor

/**
 * @brief 操作描述符结构体
 */
struct acltensorOperationDescriptor {
    // 操作类型
    acltensor::OperationType operationType;

    // ========== 张量描述符 ==========
    acltensorTensorDescriptor_t descA = nullptr;
    acltensorTensorDescriptor_t descB = nullptr;  // 仅 Trinary 使用
    acltensorTensorDescriptor_t descC = nullptr;
    acltensorTensorDescriptor_t descD = nullptr;

    // ========== 模式数组（维度标签，长度等于 numModes） ==========
    std::vector<int32_t> modeA;  // 标记张量 A 的每个维度对应的模式编号
    std::vector<int32_t> modeB;  // 仅 Trinary 使用
    std::vector<int32_t> modeC;
    std::vector<int32_t> modeD;

    // ========== 操作符 ==========
    // Elementwise Binary: D = opAC(α*opA(A), γ*opC(C))
    acltensorOperator_t opA  = ACLTENSOR_OP_IDENTITY;
    acltensorOperator_t opC  = ACLTENSOR_OP_IDENTITY;
    acltensorOperator_t opAC = ACLTENSOR_OP_ADD;       // 阶段一只支持 ADD

    // Elementwise Trinary 专用
    acltensorOperator_t opB   = ACLTENSOR_OP_IDENTITY;
    acltensorOperator_t opAB  = ACLTENSOR_OP_ADD;
    acltensorOperator_t opABC = ACLTENSOR_OP_IDENTITY;

    // ========== 计算精度 ==========
    acltensorComputeDescriptor_t descCompute;

    // ========== 辅助方法 ==========
    size_t getTotalElements() const {
        if (descD != nullptr) {
            return descD->totalElements;
        }
        return 0;
    }

    acltensorDataType_t getDataType() const {
        if (descD != nullptr) {
            return descD->dataType;
        }
        return ACLTENSOR_R_32F;
    }
};

#endif // ACLTENSOR_LIB_CORE_OPERATION_DESCRIPTOR_HPP
