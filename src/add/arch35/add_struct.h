/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of conditions of the CANN Open Software License Agreement Version 2.0.
 */

/*!
 * \file add_struct.h
 * \brief add struct - 非模板版本（参照stateless_random_uniform_v2）
 */
#ifndef ADD_STRUCT_H_
#define ADD_STRUCT_H_

#include <cstdint>

namespace AddOp {

/**
 * @brief Add 算子的 Tiling 数据结构
 * 包含 kernel 需要的所有切分信息
 */
struct AddTilingData {
    int64_t elemNum;            // 总元素个数
    int64_t usedCoreNum;        // 实际使用的核数（包含尾核）
    int64_t ubFormer;           // UB 切块大小（元素数）

    // 普通核的切分信息
    int64_t elementsPerCore;    // 每个普通核处理的元素个数
    int64_t blockFormer;        // 每个 block(UB切块)包含的元素个数
    int64_t blockLoopCnt;       // 普通核的 block 循环次数
    int64_t blockTail;          // 普通核尾 block 包含的元素个数

    // 尾核的切分信息（最后一个核）
    int64_t tailCoreElements;   // 尾核处理的元素个数
    int64_t tailCoreBlockLoopCnt; // 尾核的 block 循环次数
    int64_t tailCoreBlockTail;  // 尾核尾 block 包含的元素个数
};

} // namespace AddOp

#endif // ADD_STRUCT_H_
