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
 * \file utils.cpp
 * \brief 辅助工具函数实现
 */

#include "cann_ops_tensor.h"

/**
 * @brief 获取错误码对应的错误字符串
 */
const char* acltensorGetErrorString(const acltensorStatus_t error)
{
    switch (error) {
        case ACLTENSOR_STATUS_SUCCESS:
            return "The operation completed successfully.";
        case ACLTENSOR_STATUS_NOT_INITIALIZED:
            return "The library was not initialized.";
        case ACLTENSOR_STATUS_ALLOC_FAILED:
            return "Memory allocation failed.";
        case ACLTENSOR_STATUS_INVALID_VALUE:
            return "Invalid parameter value.";
        case ACLTENSOR_STATUS_ARCH_MISMATCH:
            return "Architecture mismatch.";
        case ACLTENSOR_STATUS_EXECUTION_FAILED:
            return "Kernel execution failed.";
        case ACLTENSOR_STATUS_INTERNAL_ERROR:
            return "Internal error.";
        case ACLTENSOR_STATUS_NOT_SUPPORTED:
            return "Operation not supported.";
        case ACLTENSOR_STATUS_INSUFFICIENT_WORKSPACE:
            return "Insufficient workspace size.";
        case ACLTENSOR_STATUS_INSUFFICIENT_DRIVER:
            return "Insufficient driver version.";
        case ACLTENSOR_STATUS_IO_ERROR:
            return "I/O error.";
        default:
            return "Unknown error.";
    }
}

/**
 * @brief 获取库版本号
 * @return 版本号 (格式: 0xMMmmPP, MM=Major, mm=Minor, PP=Patch)
 */
size_t acltensorGetVersion(void)
{
    // 版本 1.0.0
    return (1 << 16) | (0 << 8) | 0;
}
