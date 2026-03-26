/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You should refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_elementwise.cpp
 * @brief Elementwise 操作测试辅助函数实现
 *        支持任意维度、任意数据类型的 Binary 和 Trinary 操作
 */

#include "test_common.h"

namespace OpsTensorTest {

//========================================================================
// 子模块1：创建张量描述符
//========================================================================
acltensorStatus_t CreateTensorDescriptors(
    acltensorHandle_t handle, const ElementwiseBinaryTestConfig& config, acltensorTensorDescriptor_t& descA,
    acltensorTensorDescriptor_t& descC, acltensorTensorDescriptor_t& descD)
{
    // 创建张量 A 的描述符
    acltensorStatus_t status = acltensorCreateTensorDescriptor(
        handle, &descA, config.numModes(), config.dimensions.data(),
        nullptr, // strides（nullptr 表示连续内存）
        config.dataType,
        0); // alignmentRequirement（0 表示默认）
    if (status != ACLTENSOR_STATUS_SUCCESS) {
        std::cerr << "Failed to create descriptor A" << std::endl;
        return status;
    }

    // 创建张量 C 的描述符
    status = acltensorCreateTensorDescriptor(
        handle, &descC, config.numModes(), config.dimensions.data(), nullptr, config.dataType, 0);
    if (status != ACLTENSOR_STATUS_SUCCESS) {
        std::cerr << "Failed to create descriptor C" << std::endl;
        acltensorDestroyTensorDescriptor(descA);
        return status;
    }

    // 创建张量 D 的描述符
    status = acltensorCreateTensorDescriptor(
        handle, &descD, config.numModes(), config.dimensions.data(), nullptr, config.dataType, 0);
    if (status != ACLTENSOR_STATUS_SUCCESS) {
        std::cerr << "Failed to create descriptor D" << std::endl;
        acltensorDestroyTensorDescriptor(descA);
        acltensorDestroyTensorDescriptor(descC);
        return status;
    }

    return ACLTENSOR_STATUS_SUCCESS;
}

//========================================================================
// 子模块2：准备执行计划
//========================================================================
acltensorStatus_t PrepareExecutionPlan(
    acltensorHandle_t handle, const acltensorTensorDescriptor_t& descA, const acltensorTensorDescriptor_t& descC,
    const acltensorTensorDescriptor_t& descD, const ElementwiseBinaryTestConfig& config, acltensorOperator_t opType,
    acltensorOperationDescriptor_t& opDesc, acltensorPlanPreference_t& planPref, acltensorPlan_t& plan)
{
    // 使用配置中的 mode（如果为空则使用默认值 {0}）
    std::vector<int32_t> defaultMode(config.numModes(), 0);
    const int32_t* modeA = config.modeA.empty() ? defaultMode.data() : config.modeA.data();
    const int32_t* modeC = config.modeC.empty() ? defaultMode.data() : config.modeC.data();
    const int32_t* modeD = config.modeD.empty() ? defaultMode.data() : config.modeD.data();

    // 创建操作描述符
    acltensorStatus_t status = acltensorCreateElementwiseBinary(
        handle, &opDesc, descA, modeA, ACLTENSOR_OP_IDENTITY, descC, modeC, ACLTENSOR_OP_IDENTITY, descD, modeD, opType,
        ACLTENSOR_COMPUTE_DESC_32F);
    if (status != ACLTENSOR_STATUS_SUCCESS) {
        std::cerr << "Failed to create operation descriptor" << std::endl;
        return status;
    }

    // 创建 Plan Preference
    status = acltensorCreatePlanPreference(handle, &planPref, ACLTENSOR_ALGO_DEFAULT);
    if (status != ACLTENSOR_STATUS_SUCCESS) {
        std::cerr << "Failed to create plan preference" << std::endl;
        acltensorDestroyOperationDescriptor(opDesc);
        return status;
    }

    // 创建 Plan
    status = acltensorCreatePlan(handle, &plan, opDesc, planPref, 0);
    if (status != ACLTENSOR_STATUS_SUCCESS) {
        std::cerr << "Failed to create plan" << std::endl;
        acltensorDestroyOperationDescriptor(opDesc);
        acltensorDestroyPlanPreference(planPref);
        return status;
    }

    return ACLTENSOR_STATUS_SUCCESS;
}

//========================================================================
// 子模块3：在设备上执行计算
//========================================================================
acltensorStatus_t ExecuteOnDevice(
    acltensorHandle_t handle, const acltensorPlan_t& plan, const void* h_A, const void* h_C, void* h_D,
    int64_t numElements, acltensorDataType_t dataType, aclrtStream stream)
{
    // 使用现有的工具函数获取类型大小
    size_t elementSize = acltensor::GetDataTypeSize(dataType);
    size_t byteSize = numElements * elementSize;

    // 分配设备内存
    void* d_A = nullptr;
    void* d_C = nullptr;
    void* d_D = nullptr;

    if (aclrtMalloc(&d_A, byteSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_ERROR_NONE) {
        std::cerr << "Failed to allocate device memory for d_A" << std::endl;
        return ACLTENSOR_STATUS_INTERNAL_ERROR;
    }

    if (aclrtMalloc(&d_C, byteSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_ERROR_NONE) {
        std::cerr << "Failed to allocate device memory for d_C" << std::endl;
        aclrtFree(d_A);
        return ACLTENSOR_STATUS_INTERNAL_ERROR;
    }

    if (aclrtMalloc(&d_D, byteSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_ERROR_NONE) {
        std::cerr << "Failed to allocate device memory for d_D" << std::endl;
        aclrtFree(d_A);
        aclrtFree(d_C);
        return ACLTENSOR_STATUS_INTERNAL_ERROR;
    }

    // 拷贝数据到设备
    aclrtMemcpy(d_A, byteSize, h_A, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(d_C, byteSize, h_C, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // 执行内核（内部已同步等待完成）
    acltensorStatus_t status = acltensorElementwiseBinaryExecute(handle, plan, nullptr, d_A, nullptr, d_C, d_D, stream);

    // 拷贝结果
    aclrtMemcpy(h_D, byteSize, d_D, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // 清理设备内存
    aclrtFree(d_A);
    aclrtFree(d_C);
    aclrtFree(d_D);

    return status;
}

//==============================================================================
// 主函数：执行 Elementwise Binary 操作
//==============================================================================
acltensorStatus_t ExecuteElementwiseBinaryTest(const void* h_A, const void* h_C, void* h_D,
    const ElementwiseBinaryTestConfig& config, acltensorOperator_t opType, aclrtStream stream)
{
    // 1. 创建 Handle
    acltensorHandle_t handle;
    acltensorStatus_t status = acltensorCreate(&handle);
    if (status != ACLTENSOR_STATUS_SUCCESS) {
        std::cerr << "Failed to create handle" << std::endl;
        return status;
    }

    // 2. 创建张量描述符（子模块1）
    acltensorTensorDescriptor_t descA = nullptr;
    acltensorTensorDescriptor_t descC = nullptr;
    acltensorTensorDescriptor_t descD = nullptr;

    status = CreateTensorDescriptors(handle, config, descA, descC, descD);
    if (status != ACLTENSOR_STATUS_SUCCESS) {
        acltensorDestroy(handle);
        return status;
    }

    // 3. 准备执行计划（子模块2）
    acltensorOperationDescriptor_t opDesc = nullptr;
    acltensorPlanPreference_t planPref = nullptr;
    acltensorPlan_t plan = nullptr;

    status = PrepareExecutionPlan(handle, descA, descC, descD, config, opType, opDesc, planPref, plan);
    if (status != ACLTENSOR_STATUS_SUCCESS) {
        acltensorDestroyTensorDescriptor(descA);
        acltensorDestroyTensorDescriptor(descC);
        acltensorDestroyTensorDescriptor(descD);
        acltensorDestroy(handle);
        return status;
    }

    // 4. 在设备上执行计算（子模块3）
    status = ExecuteOnDevice(handle, plan, h_A, h_C, h_D, config.numElements(), config.dataType, stream);

    // 5. 清理所有资源（无论成功或失败）
    acltensorDestroyPlan(plan);
    acltensorDestroyPlanPreference(planPref);
    acltensorDestroyOperationDescriptor(opDesc);
    acltensorDestroyTensorDescriptor(descA);
    acltensorDestroyTensorDescriptor(descC);
    acltensorDestroyTensorDescriptor(descD);
    acltensorDestroy(handle);

    return status;
}

} // namespace OpsTensorTest
