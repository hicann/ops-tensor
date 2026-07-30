# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

ops_example_add_executable(weight_quant_batch_matmul_mx weight_quant_batch_matmul_mx.cpp)
# Prevent host glibc from enabling _Float128 in the Bisheng device compilation pass.
target_compile_options(
    weight_quant_batch_matmul_mx
    PRIVATE $<$<COMPILE_LANGUAGE:ASC>:-U__clang_major__> $<$<COMPILE_LANGUAGE:ASC>:-U__clang_minor__>)
