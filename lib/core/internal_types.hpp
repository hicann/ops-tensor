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
 * \file internal_types.hpp
 * \brief 内部类型定义 - 操作类型枚举
 */

#ifndef ACLTENSOR_LIB_CORE_INTERNAL_TYPES_HPP
#define ACLTENSOR_LIB_CORE_INTERNAL_TYPES_HPP

#include <cstdint>

namespace acltensor {

/**
 * @brief 操作类型枚举
 */
enum class OperationType : uint32_t
{
    ELEMENTWISE_BINARY = 0,
    ELEMENTWISE_TRINARY = 1,
    // TODO: Phase 2+ 添加更多操作类型
};

} // namespace acltensor

#endif // ACLTENSOR_LIB_CORE_INTERNAL_TYPES_HPP
