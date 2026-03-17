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
 * \file cann_ops_tensor.h
 * \brief ops-tensor 公共 API 头文件 - 提供 Add 算子的对外接口
 */

#ifndef CANN_OPS_TENSOR_H
#define CANN_OPS_TENSOR_H

#include <stdint.h>
#include "acl/acl.h"

/* 导出宏定义 */
#if defined(_WIN32) || defined(__CYGWIN__)
    #define ACLTENSOR_API __declspec(dllexport)
#else
    #define ACLTENSOR_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Tensor Add operation: C = A + B (element-wise)
 *
 * @param x1    Input tensor A (device pointer, float type)
 * @param x2    Input tensor B (device pointer, float type)
 * @param y     Output tensor (device pointer, float type)
 * @param n     Number of elements in each tensor
 * @param stream ACL stream for execution
 * @return      0 (ACL_SUCCESS) on success, error code otherwise
 */
ACLTENSOR_API aclError acltensorAdd(float* x1, float* x2, float* y, int64_t n, void* stream);

#ifdef __cplusplus
}
#endif

#endif // CANN_OPS_TENSOR_H
