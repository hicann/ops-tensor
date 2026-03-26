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
 * @file add_test.cpp
 * @brief Add算子单元测试
 */

#include "add_test.h"

#include <iostream>
#include <vector>
#include <cmath>

#include "cann_ops_tensor.h"

using namespace OpsTensorTest;

// 测试基本加法
static void test_basic_add(aclrtStream stream, TestStats& stats) {
    TEST_CASE_BEGIN("test_basic_add");

    const int64_t size = 5;
    std::vector<float> A{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> C{2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> expected{3.0f, 5.0f, 7.0f, 9.0f, 11.0f};
    std::vector<float> D(size);

    // 配置初始化（使用构造函数，兼容 C++11 及以上）
    ElementwiseBinaryTestConfig config(size);
    acltensorStatus_t result = ExecuteElementwiseBinaryTest(A.data(), C.data(), D.data(), config, ACLTENSOR_OP_ADD, stream);

    TEST_ASSERT(stats, result == ACLTENSOR_STATUS_SUCCESS, "return code failed");
    TEST_ASSERT_ARRAY_NEAR(stats, D, expected, size, 1e-6f, "array mismatch");

    TEST_CASE_PASS(stats, "test_basic_add");
}

// 测试大数组
static void test_large_array(aclrtStream stream, TestStats& stats) {
    TEST_CASE_BEGIN("test_large_array");

    const int64_t size = 1024;
    std::vector<float> A(size, 1.0f);
    std::vector<float> C(size, 2.0f);
    std::vector<float> expected(size, 3.0f);
    std::vector<float> D(size);

    // 配置初始化（使用构造函数，兼容 C++11 及以上）
    ElementwiseBinaryTestConfig config(size);
    acltensorStatus_t result = ExecuteElementwiseBinaryTest(A.data(), C.data(), D.data(), config, ACLTENSOR_OP_ADD, stream);

    TEST_ASSERT(stats, result == ACLTENSOR_STATUS_SUCCESS, "return code failed");
    TEST_ASSERT_ARRAY_NEAR(stats, D, expected, size, 1e-6f, "array mismatch");

    TEST_CASE_PASS(stats, "test_large_array");
}

// 测试负数
static void test_negative_numbers(aclrtStream stream, TestStats& stats) {
    TEST_CASE_BEGIN("test_negative_numbers");

    const int64_t size = 4;
    std::vector<float> A{-1.0f, -2.0f, 3.0f, 4.0f};
    std::vector<float> C{2.0f, -3.0f, -4.0f, 5.0f};
    std::vector<float> expected{1.0f, -5.0f, -1.0f, 9.0f};
    std::vector<float> D(size);

    // 配置初始化（使用构造函数，兼容 C++11 及以上）
    ElementwiseBinaryTestConfig config(size);
    acltensorStatus_t result = ExecuteElementwiseBinaryTest(A.data(), C.data(), D.data(), config, ACLTENSOR_OP_ADD, stream);

    TEST_ASSERT(stats, result == ACLTENSOR_STATUS_SUCCESS, "return code failed");
    TEST_ASSERT_ARRAY_NEAR(stats, D, expected, size, 1e-6f, "array mismatch");

    TEST_CASE_PASS(stats, "test_negative_numbers");
}

// 测试自定义维度标签
static void test_custom_modes(aclrtStream stream, TestStats& stats) {
    TEST_CASE_BEGIN("test_custom_modes");

    const int64_t size = 3;
    std::vector<float> A{1.0f, 2.0f, 3.0f};
    std::vector<float> C{4.0f, 5.0f, 6.0f};
    std::vector<float> expected{5.0f, 7.0f, 9.0f};
    std::vector<float> D(size);

    // 自定义维度标签
    ElementwiseBinaryTestConfig config(size);
    acltensorStatus_t result = ExecuteElementwiseBinaryTest(A.data(), C.data(), D.data(), config, ACLTENSOR_OP_ADD, stream);

    TEST_ASSERT(stats, result == ACLTENSOR_STATUS_SUCCESS, "return code failed");
    TEST_ASSERT_ARRAY_NEAR(stats, D, expected, size, 1e-6f, "array mismatch");

    TEST_CASE_PASS(stats, "test_custom_modes");
}

// 导出函数
namespace AddTest {
    void run_all_tests(aclrtStream stream, TestStats& stats) {
        test_basic_add(stream, stats);
        test_large_array(stream, stats);
        test_negative_numbers(stream, stats);
        test_custom_modes(stream, stats);
    }
}

// 自动注册到测试框架
REGISTER_OP_TEST(Add)
