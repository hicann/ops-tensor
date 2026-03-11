/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of conditions of the CANN Open Software License Agreement Version 2.0.
 */

/**
 * @file add_test.cpp
 * @brief Add算子单元测试
 */

#include "cann_ops_tensor.h"
#include "add_test.h"
#include <iostream>
#include <vector>
#include <cmath>

// 测试基本加法
void test_basic_add(aclrtStream stream, OpsTensorTest::TestStats& stats) {
    TEST_CASE_BEGIN("test_basic_add");

    const int64_t size = 5;
    float x1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float x2[] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float expected[] = {3.0f, 5.0f, 7.0f, 9.0f, 11.0f};
    float y[5];

    aclError result = acltensorAdd(x1, x2, y, size, stream);

    TEST_ASSERT(stats, result == ACL_SUCCESS, "return code failed");
    TEST_ASSERT_ARRAY_NEAR(stats, y, expected, size, 1e-6f, "array mismatch");

    TEST_CASE_PASS(stats, "test_basic_add");
}

// 测试空指针检查
void test_null_pointer(aclrtStream stream, OpsTensorTest::TestStats& stats) {
    TEST_CASE_BEGIN("test_null_pointer");

    const int64_t size = 5;
    float input[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y[5];

    aclError result1 = acltensorAdd(nullptr, input, y, size, stream);
    aclError result2 = acltensorAdd(input, nullptr, y, size, stream);
    aclError result3 = acltensorAdd(input, input, nullptr, size, stream);

    TEST_ASSERT(stats, result1 != ACL_SUCCESS, "x1 null should fail");
    TEST_ASSERT(stats, result2 != ACL_SUCCESS, "x2 null should fail");
    TEST_ASSERT(stats, result3 != ACL_SUCCESS, "y null should fail");

    TEST_CASE_PASS(stats, "test_null_pointer");
}

// 测试无效大小
void test_invalid_size(aclrtStream stream, OpsTensorTest::TestStats& stats) {
    TEST_CASE_BEGIN("test_invalid_size");

    float x1[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float x2[5] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float y[5];

    aclError result = acltensorAdd(x1, x2, y, 0, stream);

    TEST_ASSERT(stats, result != ACL_SUCCESS, "zero size should fail");

    TEST_CASE_PASS(stats, "test_invalid_size");
}

// 测试大数组
void test_large_array(aclrtStream stream, OpsTensorTest::TestStats& stats) {
    TEST_CASE_BEGIN("test_large_array");

    const int64_t size = 1024;
    std::vector<float> x1(size, 1.0f);
    std::vector<float> x2(size, 2.0f);
    std::vector<float> expected(size, 3.0f);
    std::vector<float> y(size);

    aclError result = acltensorAdd(x1.data(), x2.data(), y.data(), size, stream);

    TEST_ASSERT(stats, result == ACL_SUCCESS, "return code failed");
    TEST_ASSERT_ARRAY_NEAR(stats, y.data(), expected.data(), size, 1e-6f, "array mismatch");

    TEST_CASE_PASS(stats, "test_large_array");
}

// 测试负数
void test_negative_numbers(aclrtStream stream, OpsTensorTest::TestStats& stats) {
    TEST_CASE_BEGIN("test_negative_numbers");

    const int64_t size = 4;
    float x1[] = {-1.0f, -2.0f, 3.0f, 4.0f};
    float x2[] = {2.0f, -3.0f, -4.0f, 5.0f};
    float expected[] = {1.0f, -5.0f, -1.0f, 9.0f};
    float y[4];

    aclError result = acltensorAdd(x1, x2, y, size, stream);

    TEST_ASSERT(stats, result == ACL_SUCCESS, "return code failed");
    TEST_ASSERT_ARRAY_NEAR(stats, y, expected, size, 1e-6f, "array mismatch");

    TEST_CASE_PASS(stats, "test_negative_numbers");
}

// 导出函数
namespace AddTest {
    void run_all_tests(aclrtStream stream, OpsTensorTest::TestStats& stats) {
        test_basic_add(stream, stats);
        test_null_pointer(stream, stats);
        test_invalid_size(stream, stats);
        test_large_array(stream, stats);
        test_negative_numbers(stream, stats);
    }
}

// 自动注册到测试框架
REGISTER_OP_TEST(Add)
