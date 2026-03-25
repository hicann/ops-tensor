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
 * \file elementwise.cpp
 * \brief Elementwise 框架实现
 *
 * 架构说明：
 * - elementwise_binary.cpp: 实现 acltensorElementwiseBinaryExecute
 * - elementwise.cpp: 实现 ElementwiseSolutionRegistry::registerAllSolutions()
 */

#include "elementwise.hpp"

#include "cann_ops_tensor.h"
#include "core/plan.hpp"

namespace acltensor {

// 解决方案通过各算子的自动注册宏（REGISTER_ELEMENTWISE_SOLUTION）进行注册
// 无需手动调用注册函数

} // namespace acltensor
