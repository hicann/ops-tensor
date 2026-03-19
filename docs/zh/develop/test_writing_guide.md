# 算子测试编写指南

## 文件结构

写一个算子的测试需要 **2 个文件**：

```
src/<op_name>/tests/
├── <op_name>_test.h      # 头文件（必需）
└── <op_name>_test.cpp    # 实现文件（必需）
```

**说明**：测试框架提供了 `ElementwiseBinaryTestConfig` 辅助类和 `ExecuteElementwiseBinaryTest` 通用函数，简化测试编写。

## 头文件（必需）

**文件**：`<op_name>_test.h`

**内容**：
```cpp
#pragma once
#include "test_common.h"

namespace <OpName>Test {
    void run_all_tests(aclrtStream stream, OpsTensorTest::TestStats& stats);
}
```

**说明**：
- 命名空间：`<OpName>Test`（首字母大写，用于隔离不同算子的测试）
- 函数名：`run_all_tests`（固定名称，测试框架会调用此函数）
- 参数说明：
  - `aclrtStream stream` - ACL 运行流，用于执行算子
  - `TestStats& stats` - 测试统计对象，用于记录测试结果

---

## 实现文件（必需）

**文件**：`<op_name>_test.cpp`

**必须包含的 3 个部分**：

### 1. 测试用例函数（必需，至少1个）

**函数签名**：
```cpp
void test_xxx(aclrtStream stream, OpsTensorTest::TestStats& stats);
```

**参数说明**：
- `stream` - 传递给算子调用，用于在 NPU 上执行
- `stats` - 传递给所有测试宏（TEST_ASSERT、TEST_CASE_PASS 等），用于自动统计

**示例**：

```cpp
void test_<test_case_name>(aclrtStream stream, OpsTensorTest::TestStats& stats) {
    TEST_CASE_BEGIN("test_<test_case_name>");

    // 准备数据
    std::vector<float> A{1.0f, 2.0f};
    std::vector<float> C{2.0f, 3.0f};
    std::vector<float> expected{3.0f, 5.0f};
    std::vector<float> D(2);

    // 使用测试框架提供的配置类
    ElementwiseBinaryTestConfig config(2);  // 一维，大小为2

    // 调用通用测试函数（自动处理内存管理、数据拷贝等）
    acltensorStatus_t result = ExecuteElementwiseBinaryTest(
        A.data(), C.data(), D.data(), config, ACLTENSOR_OP_<OP_NAME>, stream);

    // 验证结果
    TEST_ASSERT(stats, result == ACLTENSOR_STATUS_SUCCESS, "failed");
    TEST_ASSERT_ARRAY_NEAR(stats, D, expected, 2, 1e-6f, "mismatch");

    TEST_CASE_PASS(stats, "test_<test_case_name>");
}
```

### 2. run_all_tests 实现（必需）

```cpp
namespace <OpName>Test {
    void run_all_tests(aclrtStream stream, OpsTensorTest::TestStats& stats) {
        test_xxx(stream, stats);
        // 可以添加更多测试函数
    }
}
```

### 3. 自动注册（必需）

```cpp
REGISTER_OP_TEST(<OpName>)
```

---

## 完整示例

**<op_name>_test.h**
```cpp
#pragma once
#include "test_common.h"

namespace <OpName>Test {
    void run_all_tests(aclrtStream stream, OpsTensorTest::TestStats& stats);
}
```

**<op_name>_test.cpp**
```cpp
#include "cann_ops_tensor.h"
#include "<op_name>_test.h"
#include <vector>

void test_<case_name>(aclrtStream stream, OpsTensorTest::TestStats& stats) {
    TEST_CASE_BEGIN("test_<case_name>");

    std::vector<float> A{1.0f, 2.0f};
    std::vector<float> C{2.0f, 3.0f};
    std::vector<float> expected{3.0f, 5.0f};
    std::vector<float> D(2);

    ElementwiseBinaryTestConfig config(2);
    acltensorStatus_t result = ExecuteElementwiseBinaryTest(
        A.data(), C.data(), D.data(), config, ACLTENSOR_OP_<OP_NAME>, stream);

    TEST_ASSERT(stats, result == ACLTENSOR_STATUS_SUCCESS, "failed");
    TEST_ASSERT_ARRAY_NEAR(stats, D, expected, 2, 1e-6f, "mismatch");

    TEST_CASE_PASS(stats, "test_<case_name>");
}

namespace <OpName>Test {
    void run_all_tests(aclrtStream stream, OpsTensorTest::TestStats& stats) {
        test_<case_name>(stream, stats);
    }
}

REGISTER_OP_TEST(<op_name>)
```

---

## 可用的宏

### 用例控制
```cpp
TEST_CASE_BEGIN("test_name")    // 标记开始
TEST_CASE_PASS(stats, "test_name")  // 标记通过
```

### 断言
```cpp
TEST_ASSERT(stats, condition, "error")              // 条件断言
TEST_ASSERT_ARRAY_EQ(stats, actual, expected, size, "error")   // 精确比较
TEST_ASSERT_ARRAY_NEAR(stats, actual, expected, size, tol, "error")  // 容差比较
```

---

## 编译运行

```bash
./build.sh --ops=<op_name> --run
```

---

## 参考示例

完整示例：`src/add/tests/add_test.cpp`
