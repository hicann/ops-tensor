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
 * \file cce_pipe_stub.h
 * \brief bisheng --asc-aicore-lang 模式 host 边 PIPE_FIX 兜底桩
 *
 * bisheng --asc-aicore-lang 对同一编译单元做 host/device 双边编译：
 * device 边定义 __CCE_AICORE__，编译器内置 pipe_t 枚举含 PIPE_FIX 成员；
 * host 边不定义 __CCE_AICORE__，PIPE_FIX 成员被条件编译裁剪。
 * blaze 的模板默认实参（pipe_t Pipe = PIPE_FIX）在 host 边同样要解析，
 * 导致 examples 编译报 "use of undeclared identifier 'PIPE_FIX'"。
 *
 * 本桩仅在 host 边生效：device 边 __CCE_AICORE__ 已定义，不定义宏，
 * 使用真实枚举成员。取值与 device 边 cce_aicore_intrinsics.h 的定义一致（PIPE_FIX = 10）。
 * 由 examples/CMakeLists.txt 通过 force-include 注入，业务 cpp 无需修改。
 */
#ifndef CCE_PIPE_STUB_H
#define CCE_PIPE_STUB_H

#ifndef __CCE_AICORE__
#define PIPE_FIX (pipe_t)(10)
#endif

#endif // CCE_PIPE_STUB_H
