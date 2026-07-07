/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file qbmm_cpu_debug_stub.h
 * \brief CPU-debug 下 blaze cube 侧的公共桩（AuxGetC0Size / GetMmDstType）。
 *        由 fixpipe 与 mix 两套 wrapper 共用，`#pragma once` 保证同一编译单元只定义一次。
 */

#pragma once

#if defined(ASCENDC_CPU_DEBUG)
namespace AscendC {
template <typename T>
constexpr int32_t AuxGetC0Size() { return 32; }
template <>
constexpr int32_t AuxGetC0Size<half>() { return 16; }
template <>
constexpr int32_t AuxGetC0Size<float>() { return 16; }
template <>
constexpr int32_t AuxGetC0Size<bfloat16_t>() { return 16; }
template <>
constexpr int32_t AuxGetC0Size<int32_t>() { return 16; }
template <>
constexpr int32_t AuxGetC0Size<int8_t>() { return 32; }
template <typename T>
struct GetMmDstType { using Type = T; };
template <>
struct GetMmDstType<int8_t> { using Type = int32_t; };
template <>
struct GetMmDstType<half> { using Type = float; };
}
#endif
