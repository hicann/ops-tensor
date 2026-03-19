# 算子开发指南

## 目录结构

开发一个算子需要以下文件：

```
src/
├── add/                        # 示例：Add 算子
│   ├── add.cpp                # Host + Kernel 实现
│   ├── add_struct.h           # Tiling 数据结构（可选）
│   ├── CMakeLists.txt         # 编译配置
│   └── tests/                 # 测试目录（强烈推荐）
│       ├── add_test.h
│       └── add_test.cpp
├── ...                        # 其他算子
└── CMakeLists.txt
```

**说明**：
- Host 和 Kernel 可以合并为一个 `.cpp` 文件
- TilingData 可以定义在 `.cpp` 文件中，也可以独立为 `_struct.h`
- `arch35/` 目录仅在需要区分不同 SOC 架构时使用
- 测试文件强烈推荐，但不是必需的

---

## 文件说明

### 1. 解决方案实现（`<op>_solution.cpp`）

**作用**：
- 实现**解决方案执行函数**（框架会自动调用）
- 计算 Tiling 参数
- 管理设备内存
- 调用核函数
- **注册解决方案到全局注册表**

**基本结构**：

```cpp
#include "acl/acl.h"
#include "kernel_operator.h"

#define GM_ADDR uint8_t*

// ========== Tiling 数据结构 ==========
namespace <OpName>Op {
    struct <OpName>TilingData {
        int64_t totalLength;
        int64_t usedCoreNum;
        // ... 其他 Tiling 参数
    };
}

// ========== 解决方案执行函数（框架自动调用） ==========
/*
  1. 参数获取和检查
  2. 计算 Tiling 数据
  3. 分配设备TilingData内存并拷贝 Tiling 数据
  4. 调用核函数
  5. 同步（异步执行安全）
  6. 释放TilingData设备内存
*/
static acltensorStatus_t Execute<OpName>Solution(const ElementwiseArgs& args)
{
    // 1. 参数获取和检查
    const void* A = args.bufferA;
    const void* C = args.bufferC;
    void* D = args.bufferD;
    if (A == nullptr || ...) {
        ... // 异常返回
    }

    // 2. 计算 Tiling 数据（可用函数封装）
    <OpName>Op::<OpName>TilingData tilingData;
    tilingData.totalLength = size;
    tilingData.usedCoreNum = CalculateCoreNum(size);  // 自定义函数
    // ... 其他 Tiling 参数

    // 3. 分配设备TilingData内存并拷贝 Tiling 数据
    uint8_t *tilingDevice;
    aclrtMalloc((void**)&tilingDevice, sizeof(tilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(tilingDevice, sizeof(tilingData), &tilingData, sizeof(tilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    // 4. 调用核函数 <<<Block, workspace, stream>>>
    <op_name>_kernel_do(inputDevice, outputDevice, tilingDevice,
                        nullptr, tilingData.usedCoreNum, stream);

    // 5. 同步
    aclrtSynchronizeStream(stream);

    // 6. 释放设备内存
    aclrtFree(tilingDevice);

    return ACL_SUCCESS;
}

// ========== Kernel 部分：核函数实现 ==========

using namespace AscendC;

extern "C" __global__ __aicore__ void <op_name>(GM_ADDR input, GM_ADDR output,
                                                GM_ADDR tiling)
{
    // Kernel 类型声明
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // 初始化
    TPipe pipe;
    // ... 初始化 LocalTensor、GlobalTensor、TQue 等

    // 解析 Tiling 数据
    auto tilingData = (<OpName>Op::<OpName>TilingData*)tiling;

    // 核心计算逻辑
    for (int i = 0; i < tilingData->blockLoopCnt; ++i) {
        // 1. DataCopy: GM -> LocalTensor
        // 2. 计算
        // 3. DataCopy: LocalTensor -> GM
    }
}

// ========== 核函数封装 ==========
void <op_name>_kernel_do(GM_ADDR input, GM_ADDR output, GM_ADDR tiling,
                         GM_ADDR workspace, uint32_t numBlocks, void *stream)
{
    <op_name><<<numBlocks, workspace, stream>>>(input, output, tiling);
}

// ========== 注册解决方案到全局注册表 ==========
namespace {
    std::shared_ptr<ElementwiseSolution> Create<OpName>Solution()
    {
        SolutionUid uid{ACLTENSOR_OP_<OP_NAME>, ACLTENSOR_R_32F, 0};  // 0 表示通用维度
        return std::make_shared<ElementwiseSolution>(uid, Execute<OpName>Solution);
    }

    struct <OpName>SolutionRegistrar {
        <OpName>SolutionRegistrar() {
            auto solution = Create<OpName>Solution();
            ElementwiseSolutionRegistry::instance().registerSolution(solution);
        }
    };

    static <OpName>SolutionRegistrar g_<op_name>SolutionRegistrar;
}
```

**解决方案关键点**：
1. 定义 TilingData 结构体（或 include 独立的 `_struct.h`）
2. 计算 Tiling 参数
3. 实现解决方案执行函数（签名：`static acltensorStatus_t Execute<OpName>Solution(const ElementwiseArgs& args)`）
4. 注册解决方案到全局注册表（使用静态注册器自动注册）

**Kernel 部分关键点**：(Kernel部分可提取成独立的 `_kernel.cpp`)
1. 使用 `__global__ __aicore__` 标记核函数
2. 实现 GM ↔ LocalTensor 的数据搬运
3. 实现核心计算逻辑

**注册机制说明**：
- **SolutionUid** - 解决方案唯一标识符，由 `{操作符, 数据类型, 维度数}` 三元组组成
  - `ACLTENSOR_OP_<OP_NAME>` - 操作符类型（ADD、SUB、MUL、DIV 等）
  - `ACLTENSOR_R_32F` - 数据类型（当前仅支持 float32）
  - `0` - 维度数（0 表示通用解决方案，适配任意维度）

- **Create<OpName>Solution** - 创建解决方案对象
  - 返回 `std::shared_ptr<ElementwiseSolution>` 智能指针
  - 传入 `Execute<OpName>Solution` 函数指针作为执行接口

- **静态注册器** - 利用全局变量初始化自动注册
  - 程序启动时自动执行，将解决方案注册到全局单例 `ElementwiseSolutionRegistry`
  - 框架在执行时根据操作符、数据类型、维度数从注册表查询解决方案

---

### 2. Tiling 数据结构（可选）

**文件**：`<op_name>_struct.h`

**作用**：定义 Host 传递给 Kernel 的 Tiling 参数

**基本结构**：

```cpp
#ifndef <OP_NAME>_STRUCT_H
#define <OP_NAME>_STRUCT_H

#include <cstdint>

namespace <OpName>Op {

struct <OpName>TilingData {
    int64_t totalLength;
    int64_t usedCoreNum;
    int64_t blockFormer;
    int64_t blockLoopCnt;
    int64_t blockTail;
    // ... 其他 Tiling 参数
};

} // namespace <OpName>Op

#endif
```

**说明**：
- 这是一个简单的 C 结构体
- 只包含基本数据类型（int64_t 等）
- Host 计算参数，Kernel 读取参数
- 也可以直接定义在 `.cpp` 文件中

---

### 3. 编译配置

**文件**：`CMakeLists.txt`

```cmake
register_operator(
    NAME <op_name>
    ARCH_DIR arch35    # 如果需要区分架构才加这个参数
)
```

---

### 4. 测试文件（强烈推荐）

参见 [测试编写指南](test_writing_guide.md)。

**说明**：虽然测试文件不是必需的，但强烈建议为每个算子编写单元测试，以确保算子实现的正确性。

---

## 开发流程

### 步骤 1：创建目录和文件

```bash
mkdir -p src/<op_name>/tests
touch src/<op_name>/<op_name>.cpp
touch src/<op_name>/CMakeLists.txt
```

（可选）独立的 struct 文件：
```bash
touch src/<op_name>/<op_name>_struct.h
```

（可选）测试文件：
```bash
touch src/<op_name>/tests/<op_name>_test.h
touch src/<op_name>/tests/<op_name>_test.cpp
```

### 步骤 2：编写解决方案实现

在 `<op_name>_solution.cpp` 中：
1. 定义 TilingData 结构体（或 include 独立的 `_struct.h`）
2. 实现解决方案执行函数（`Execute<OpName>Solution`）
3. 计算并注册解决方案到全局注册表

在 `<op_name>_kernel.cpp` 中：
1. 实现核函数（使用 `__global__ __aicore__`）

### 步骤 3：配置编译

在 `CMakeLists.txt` 中注册算子。

### 步骤 4：编写测试（推荐）

参考 [测试编写指南](test_writing_guide.md)。

### 步骤 5：编译验证

```bash
./build.sh --ops=<op_name> --run
```

---

## 关键概念

### 解决方案 vs Kernel

| 层面 | 运行位置 | 职责 |
|------|---------|------|
| **解决方案** | CPU | Tiling 计算、内存管理、调用 Kernel、注册到全局表 |
| **Kernel** | NPU AI Core | 实际计算逻辑 |

### Tiling

**目的**：将大任务切分成适合 NPU 执行的小块

**关键参数**：
- `usedCoreNum` - 使用多少个 AI Core
- `blockFormer` - 每次迭代处理多少数据
- `blockLoopCnt` - 每个核迭代多少次

**计算原则**：
- 充分利用 AI Core 并行能力
- 数据不超过 Unified Buffer 容量
- 对齐到 32 字节边界

### <<<>>> 核函数调用

```cpp
<kernel_func><<<numBlocks, workspace, stream>>>(args...);
```

**参数说明**：
- `numBlocks` - 使用多少个 AI Core（Block）
- `workspace` - 共享内存指针，通常设置为 `nullptr`
- `stream` - ACL 执行流

---

## 完整示例

参见 `src/add/` 目录：
- `add_solution.cpp` - 解决方案实现（Tiling 计算、解决方案执行函数、注册）
- `add_kernel.cpp` - Kernel 端实现（核函数逻辑）
- `arch35/add_struct.h` - Tiling 数据结构
- `tests/add_test.cpp` - 单元测试
- `CMakeLists.txt` - 编译配置

---

## 相关文档

- [测试编写指南](test_writing_guide.md)
- [build 参数说明](../context/build.md)
