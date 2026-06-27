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
 * \file blaze_kernel_stub.h
 * \brief CANN 9.1.0 兼容性补丁 - 为 Kernel UT 提供缺失的定义
 *
 * ascend950 (arch3510, __NPU_ARCH__=3510) 的 CANN 9.1.0 CPU debug 头文件缺少以下定义:
 * 1. __biasbuf__ 地址空间限定符 (tensor_api 使用但 tikicpulib 未定义)
 * 2. POS_LOWEST / POS_HIGHEST 常量 (C API 使用, 定义在 bisheng compiler 中,
 *    但 x86_64 CPU debug 模式不引入 bisheng 头文件)
 *    NOTE: MODE_ZEROING / MODE_MERGING / MODE_UNKNOWN 已存在于 stub_fun.h 的 Literal enum,
 *    不需重复定义, 否则会与 Literal enum 成员冲突 ("redeclared as different kind of entity")
 * 3. 融合操作类型常量
 * 4. BLOCK_SIZE 常量 (blaze epilogue 使用但未定义, 值为32字节对齐粒度)
 *
 * POS_LOWEST/POS_HIGHEST 定义为 constexpr int32_t (值为 bisheng Pos::LOWEST=0, Pos::HIGHEST=1),
 *    可匹配 stub_fun.h 中 vdup 等 5 参数 overload 的 int32_t type 参数;
 * MODE_* 由 stub_fun.h Literal enum 提供 (MODE_ZEROING=0, MODE_MERGING=63, MODE_UNKNOWN=65),
 * 可匹配 Literal 或 int32_t mode 参数的 overload.
 *
 */

#pragma once

#include <cstdint>

#ifndef __biasbuf__
#define __biasbuf__
#endif

#ifndef POS_LOWEST
constexpr int32_t POS_LOWEST = 0;
#endif

#ifndef POS_HIGHEST
constexpr int32_t POS_HIGHEST = 1;
#endif

#ifndef BLOCK_SIZE
constexpr uint64_t BLOCK_SIZE = 32UL;
#endif