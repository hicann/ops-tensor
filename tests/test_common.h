/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You should refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_common.h
 * @brief 公共测试框架 - 提供测试统计、断言宏和 ACL 初始化功能
 */

#ifndef CANN_OPS_TENSOR_TEST_COMMON_H
#define CANN_OPS_TENSOR_TEST_COMMON_H

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include "cann_ops_tensor_types.h"
#include "cann_ops_tensor.h"
#include "utils/type_utils.hpp"
#include "acl/acl.h"

namespace OpsTensorTest {

// 测试结果统计
struct TestStats {
    int total = 0;
    int passed = 0;
    int failed = 0;

    void print(const std::string& name = "") const {
        if (!name.empty()) {
            std::cout << name << ": ";
        }
        std::cout << "总测试数=" << total << ", 通过=" << passed << ", 失败=" << failed << std::endl;
    }
};

// 全局测试统计（所有算子累计）
extern TestStats g_global_stats;

// 测试用例开始标记
#define TEST_CASE_BEGIN(test_name) \
    do { \
        std::cout << "[RUN] " << test_name << "..." << std::endl; \
    } while(0)

// 测试用例通过标记（自动更新统计）
#define TEST_CASE_PASS(local_stats, test_name) \
    do { \
        (local_stats).total++; \
        (local_stats).passed++; \
        OpsTensorTest::g_global_stats.total++; \
        OpsTensorTest::g_global_stats.passed++; \
        std::cout << "[PASS] " << test_name << std::endl; \
    } while(0)

// 断言宏：失败时打印错误信息并退出，成功时不打印
#define TEST_ASSERT(local_stats, condition, error_msg) \
    do { \
        if (!(condition)) { \
            (local_stats).total++; \
            (local_stats).failed++; \
            OpsTensorTest::g_global_stats.total++; \
            OpsTensorTest::g_global_stats.failed++; \
            std::cerr << "  [ERROR] " << error_msg << std::endl; \
            std::exit(1); \
        } \
    } while(0)

// 精确数组比较宏（无容差，高效 - 使用 std::equal）
#define TEST_ASSERT_ARRAY_EQ(local_stats, actual, expected, length, error_msg) \
    do { \
        bool all_match = std::equal((expected), (expected) + (length), (actual)); \
        if (!all_match) { \
            auto iter = std::mismatch((expected), (expected) + (length), (actual)); \
            size_t mismatch_idx = std::distance((expected), iter.first); \
            std::cerr << "  [ERROR] " << error_msg << " at index " << mismatch_idx << std::endl; \
        } \
        TEST_ASSERT((local_stats), all_match, error_msg); \
    } while(0)

// 容差数组比较宏（浮点数，带容差）
#define TEST_ASSERT_ARRAY_NEAR(local_stats, actual, expected, length, tol, error_msg) \
    do { \
        bool all_match = true; \
        size_t first_mismatch = 0; \
        if ((actual).size() != (expected).size()) { \
            all_match = false; \
            std::cerr << "  [ERROR] " << error_msg << ": actual.size() (" << (actual).size() \
                      << ") != expected.size() (" << (expected).size() << ")" << std::endl; \
        } else { \
            for (size_t _i = 0; _i < static_cast<size_t>(length); ++_i) { \
                if (std::abs((actual)[_i] - (expected)[_i]) > (tol)) { \
                    all_match = false; \
                    first_mismatch = _i; \
                    break; \
                } \
            } \
            if (!all_match) { \
                std::cerr << "  [ERROR] " << error_msg << " at index " << first_mismatch \
                          << ": actual=" << (actual)[first_mismatch] \
                          << ", expected=" << (expected)[first_mismatch] << std::endl; \
            } \
        } \
        TEST_ASSERT((local_stats), all_match, error_msg); \
    } while(0)

// 测试抬头宏
#define TEST_PRINT_HEADER(op_name) \
    do { \
        std::cout << "========================================" << std::endl; \
        std::cout << "    " << op_name << "算子单元测试" << std::endl; \
        std::cout << "========================================" << std::endl; \
        std::cout << std::endl; \
    } while(0)

// 测试结果宏（包含抬头和结尾）
#define TEST_PRINT_RESULT(op_name, local_stats) \
    do { \
        std::cout << std::endl; \
        std::cout << "========================================" << std::endl; \
        std::cout << "       " << op_name << "算子测试结果" << std::endl; \
        std::cout << "========================================" << std::endl; \
        (local_stats).print(#op_name); \
        std::cout << "========================================" << std::endl; \
    } while(0)

// 打印单个测试结果宏（用于循环中，name 是字符串变量）
#define TEST_PRINT_RESULT_NAME(name, local_stats) \
    do { \
        std::cout << std::endl; \
        (local_stats).print(name); \
    } while(0)

// 打印全局测试结果宏
#define TEST_PRINT_GLOBAL_RESULT() \
    do { \
        std::cout << std::endl; \
        std::cout << "========================================" << std::endl; \
        std::cout << "       全局测试结果" << std::endl; \
        std::cout << "========================================" << std::endl; \
        OpsTensorTest::g_global_stats.print("全部算子"); \
        std::cout << "========================================" << std::endl; \
    } while(0)

// ACL 管理
class ACLManager {
public:
    static int init(aclrtStream& stream);
    static void finalize(aclrtStream& stream);
};

/*============================================================================
 *                        Elementwise 测试接口声明
 *============================================================================*/

/**
 * @brief Elementwise 操作配置基类
 *        包含所有 Elementwise 操作的公共配置
 */
struct ElementwiseTestConfigBase {
    std::vector<int64_t> dimensions;      // 各维度长度，如 {2, 3} 表示 2x3 矩阵
    acltensorDataType_t dataType = ACLTENSOR_R_32F;  // 数据类型（默认 float32）

    // 构造函数
    ElementwiseTestConfigBase(const std::vector<int64_t>& dims, acltensorDataType_t dt)
        : dimensions(dims), dataType(dt) {}

    // 获取总元素数量
    int64_t numElements() const {
        if (dimensions.empty()) return 0;
        int64_t total = 1;
        for (auto dim : dimensions) {
            total *= dim;
        }
        return total;
    }

    // 获取维度数量
    uint32_t numModes() const {
        return static_cast<uint32_t>(dimensions.size());
    }
};

/**
 * @brief Elementwise Binary 操作测试配置（2个输入：A、C）
 */
struct ElementwiseBinaryTestConfig : public ElementwiseTestConfigBase {
    std::vector<int32_t> modeA;           // 张量 A 的维度标签
    std::vector<int32_t> modeC;           // 张量 C 的维度标签
    std::vector<int32_t> modeD;           // 输出 D 的维度标签

    // 便捷构造函数1：一维默认场景（mode={0}）
    ElementwiseBinaryTestConfig(int64_t size, acltensorDataType_t dt = ACLTENSOR_R_32F)
        : ElementwiseTestConfigBase(std::vector<int64_t>{size}, dt)
        , modeA(1, 0)
        , modeC(1, 0)
        , modeD(1, 0) {}

    // 通用构造函数2：支持自定义 dimensions 和 mode
    ElementwiseBinaryTestConfig(
        const std::vector<int64_t>& dims,
        const std::vector<int32_t>& ma,
        const std::vector<int32_t>& mc,
        const std::vector<int32_t>& md,
        acltensorDataType_t dt = ACLTENSOR_R_32F)
        : ElementwiseTestConfigBase(dims, dt)
        , modeA(ma)
        , modeC(mc)
        , modeD(md) {}
};

/**
 * @brief Elementwise Trinary 操作测试配置（3个输入：A、B、C）
 *        未来实现：D = op2(op1(A, B), C)
 */
struct ElementwiseTrinaryTestConfig : public ElementwiseTestConfigBase {
    std::vector<int32_t> modeA;           // 张量 A 的维度标签
    std::vector<int32_t> modeB;           // 张量 B 的维度标签
    std::vector<int32_t> modeC;           // 张量 C 的维度标签
    std::vector<int32_t> modeD;           // 输出 D 的维度标签
};

/**
 * @brief 执行 Elementwise Binary 操作（通用测试函数）
 *
 * 支持的操作类型：ADD, SUB, MUL, DIV
 * 支持任意维度、任意数据类型
 *
 * @param h_A         输入张量 A（主机内存）
 * @param h_C         输入张量 C（主机内存）
 * @param h_D         输出张量 D（主机内存）
 * @param config      操作配置（包含维度、mode、数据类型等）
 * @param opType      操作类型 (ACLTENSOR_OP_ADD/SUB/MUL/DIV)
 * @param stream      ACL 流
 * @return ACLTENSOR_STATUS_SUCCESS 成功，否则返回错误码
 *
 * 使用示例：
 * @code
 * // 简单一维场景
 * float A[] = {1.0f, 2.0f, 3.0f};
 * float C[] = {4.0f, 5.0f, 6.0f};
 * float D[3];
 *
 * ElementwiseBinaryTestConfig config(3);  // 一维，大小为3
 * ExecuteElementwiseBinaryTest(A, C, D, config, ACLTENSOR_OP_ADD, stream);
 * @endcode
 */
acltensorStatus_t ExecuteElementwiseBinaryTest(
    const void* h_A,
    const void* h_C,
    void* h_D,
    const ElementwiseBinaryTestConfig& config,
    acltensorOperator_t opType,
    aclrtStream stream);

// 测试函数类型
using TestFunc = void(*)(aclrtStream, TestStats&);

// 测试注册表
struct TestRegistry {
    struct TestEntry {
        const char* name;
        TestFunc func;
    };

    static std::vector<TestEntry>& get_tests() {
        static std::vector<TestEntry> tests;
        return tests;
    }

    static void register_test(const char* name, TestFunc func) {
        get_tests().push_back({name, func});
    }
};

// 自动注册宏
#define REGISTER_OP_TEST(op_name) \
    namespace op_name##Test { \
        void run_all_tests(aclrtStream, OpsTensorTest::TestStats&); \
        namespace { \
            struct Registrar { \
                Registrar() { \
                    OpsTensorTest::TestRegistry::register_test(#op_name, run_all_tests); \
                } \
            }; \
            static Registrar registrar; \
        } \
    }

} // namespace OpsTensorTest

#endif // CANN_OPS_TENSOR_TEST_COMMON_H
