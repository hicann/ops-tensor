/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of conditions of the CANN Open Software License Agreement Version 2.0.
 */

/**
 * @file add_test.h
 * @brief Add算子测试函数声明
 */

#pragma once

#include "test_common.h"

namespace AddTest {
    void run_all_tests(aclrtStream stream, OpsTensorTest::TestStats& stats);
}
