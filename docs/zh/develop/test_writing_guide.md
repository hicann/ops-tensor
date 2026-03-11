# 算子测试编写指南

## 文件结构

写一个算子的测试需要 **2 个文件**：

```
src/<op_name>/tests/
├── <op_name>_test.h      # 头文件（必需）
└── <op_name>_test.cpp    # 实现文件（必需）
```

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
void test_xxx(aclrtStream stream, OpsTensorTest::TestStats& stats) {
    TEST_CASE_BEGIN("test_xxx");

    // 准备数据
    float input1[] = {1.0f, 2.0f};
    float input2[] = {2.0f, 3.0f};
    float expected[] = {3.0f, 5.0f};
    float output[2];

    // 调用算子
    aclError result = acl<OpName>(input1, input2, output, 2, stream);

    // 验证结果
    TEST_ASSERT(stats, result == ACL_SUCCESS, "failed");
    TEST_ASSERT_ARRAY_NEAR(stats, output, expected, 2, 1e-6f, "mismatch");

    TEST_CASE_PASS(stats, "test_xxx");
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

**my_op_test.h**
```cpp
#pragma once
#include "test_common.h"

namespace MyOpTest {
    void run_all_tests(aclrtStream stream, OpsTensorTest::TestStats& stats);
}
```

**my_op_test.cpp**
```cpp
#include "cann_ops_tensor.h"
#include "my_op_test.h"

void test_basic(aclrtStream stream, OpsTensorTest::TestStats& stats) {
    TEST_CASE_BEGIN("test_basic");

    float x1[] = {1.0f, 2.0f};
    float x2[] = {2.0f, 3.0f};
    float expected[] = {3.0f, 5.0f};
    float y[2];

    aclError result = aclMyOp(x1, x2, y, 2, stream);

    TEST_ASSERT(stats, result == ACL_SUCCESS, "failed");
    TEST_ASSERT_ARRAY_NEAR(stats, y, expected, 2, 1e-6f, "mismatch");

    TEST_CASE_PASS(stats, "test_basic");
}

namespace MyOpTest {
    void run_all_tests(aclrtStream stream, OpsTensorTest::TestStats& stats) {
        test_basic(stream, stats);
    }
}

REGISTER_OP_TEST(MyOp)
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
