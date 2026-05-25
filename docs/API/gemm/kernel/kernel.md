# Kernel基础框架
> 公共接口说明

## 概述
矩阵乘 Kernel 基础框架，提供统一的模板参数、数据结构和核心流程。不同实现（Basic、QBMM MX、StreamK）在此基础上扩展特定功能，其中 QBMM MX 支持 MxFP4/MxFP8 量化、Scale 因子处理、多 Batch 维度。

详见：[README.md](./README.md) 查看 API 清单和实现对比。

## 类模板概述

### 模板参数
| 参数 | 说明 |
|------|------|
| ProblemShape_ | 问题形状类型，通常为 `AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>` (m, n, k, batch) |
| BlockMmad_ | BlockMmad 类，矩阵乘计算组件 |
| BlockEpilogue_ | BlockEpilogue 类，后处理组件 |
| BlockScheduler_ | BlockScheduler 类，任务调度组件 |

### 特殊模板参数（量化 Kernel）
| 参数 | 说明 |
|------|------|
| isAtomicAdd | 是否启用 Atomic Add 模式（QBMM MX） |

说明：QBMM MX Kernel 支持 `isAtomicAdd` 参数，用于多核并行累加场景。

## 类模板概述

### 模板参数
| 参数 | 说明 |
|------|------|
| ProblemShape_ | 问题形状类型，通常为 `AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>` (m, n, k, batch) |
| BlockMmad_ | BlockMmad 类，矩阵乘计算组件 |
| BlockEpilogue_ | BlockEpilogue 类，后处理组件 |
| BlockScheduler_ | BlockScheduler 类，任务调度组件 |

### 类型别名
| 类型 | 说明 |
|------|------|
| ProblemShape | 问题形状类型（继承自模板参数） |
| BlockMmad | BlockMmad 类型（继承自模板参数） |
| BlockEpilogue | BlockEpilogue 类型（继承自模板参数） |
| BlockScheduler | BlockScheduler 类型（继承自模板参数） |
| AType | A 矩阵数据类型（继承自 BlockMmad） |
| BType | B 矩阵数据类型（继承自 BlockMmad） |
| CType | C 矩阵数据类型（继承自 BlockMmad） |
| BiasType | Bias 数据类型（继承自 BlockMmad） |
| LayoutA | A 矩阵布局类型（继承自 BlockMmad） |
| LayoutB | B 矩阵布局类型（继承自 BlockMmad） |
| LayoutC | C 矩阵布局类型（继承自 BlockMmad） |
| LayoutBias | Bias 布局类型（继承自 BlockMmad） |
| TupleShape | Tile 形状类型 `AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>` |

### Layout 构建类型
| 类型 | 说明 |
|------|------|
| MakeLayoutA | A 矩阵 Layout 构建器 `FrameLayoutFormat<LayoutA, ...>` |
| MakeLayoutB | B 矩阵 Layout 构建器 `FrameLayoutFormat<LayoutB, ...>` |
| MakeLayoutC | C 矩阵 Layout 构建器 `FrameLayoutFormat<LayoutC, ...>` |
| MakeLayoutBias | Bias Layout 构建器 `FrameLayoutFormat<LayoutBias, ...>` |

### 核心数据结构

#### Params
```
struct Params {
    ProblemShape problemShape;            // 问题 shape (m, n, k, batch)
    BlockMmadParams mmadParams;           // mmad 参数（包含 GM 地址）
    BlockEpilogueParams epilogueParams;   // epilogue 参数
    BlockSchedulerParams schedulerParams; // scheduler 参数
};
```

#### Arguments
```
struct Arguments {
    ProblemShape problemShape;             // 问题 shape (m, n, k, batch)
    BlockMmadArguments mmadArgs;           // mmad 参数
    BlockEpilogueArguments epilogueArgs;   // epilogue 参数
};
```

### 核心成员变量
| 变量 | 类型 | 说明 |
|------|------|------|
| problemShape_ | TupleShape | 问题规模 (m, n, k, batch) |
| isBias_ | bool | 是否启用 bias 计算 |
| aGmAddr_ | `__gm__ AType*` | A 矩阵 GM 地址 |
| bGmAddr_ | `__gm__ BType*` | B 矩阵 GM 地址 |
| cGmAddr_ | `__gm__ CType*` | C 矩阵 GM 地址 |
| biasGmAddr_ | `__gm__ BiasType*` | Bias GM 地址（可选） |

## 核心成员方法

### 构造函数
```
__aicore__ inline KernelMatmul()
```
功能：构造 Kernel 对象。

### 析构函数
```
__aicore__ inline ~KernelMatmul()
```
功能：析构 Kernel 对象。

### Init函数
```
__aicore__ inline void Init(Params const& params)
```
功能：初始化 Kernel，提取问题规模和 GM 地址。
执行流程：
1. 设置问题规模 `problemShape_`
2. 提取 BlockMmad 参数 `mmadParams`
3. 设置 A、B、C 的 GM 地址
4. 判断 bias 地址是否为 nullptr，设置 `isBias_` 和 `biasGmAddr_`

### operator函数
```
__aicore__ inline void operator()(Params const& params)
```
功能：执行矩阵乘 Kernel 计算。

**公共执行流程**：
1. **初始化**：调用 `Init(params)` 设置参数
2. **BlockScheduler 初始化**：创建调度器，获取 tile 信息
3. **Layout 构建**：构建 A、B、C、Bias 的 ND layout
4. **GM Tensor 创建**：创建 A、B、C、Bias 的 GM Tensor
5. **Tile 循环处理**：遍历 tile 执行矩阵乘计算
6. **清理**：关闭 HF32/MM Layout Transform

## 公共调用示例

### 组件组装模板
```
// 定义数据类型和布局
using AType = half;
using BType = half;
using CType = float;
using BiasType = float;
using LayoutA = AscendC::Te::Layout::RowMajor;
using LayoutB = AscendC::Te::Layout::ColMajor;
using LayoutC = AscendC::Te::Layout::RowMajor;
using LayoutBias = LayoutC;

// 定义问题 shape
using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

// 定义 BlockScheduler
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, FULL_LOAD_MODE>;

// 定义 Kernel（根据需求选择 Basic 或 StreamK）
using MatmulKernel = Blaze::Gemm::Kernel::KernelMatmulBasic<...>;
// 或
using MatmulKernel = Blaze::Gemm::Kernel::KernelMatmulStreamK<...>;
```

### 参数准备模板
```
using Params = typename MatmulKernel::Params;
Params params = {
    {m, n, k, batch},        // problem shape
    {aGM, bGM, cGM, biasGM}, // mmad args
    {...},                   // epilogue args
    {mL1, nL1, kL1, baseM, baseN, baseK, ...} // scheduler params
};
```

### Kernel 执行模板
```
MatmulKernel mm;
mm(params);
```

## 公共约束

1. **模板参数要求**：
   - ProblemShape 必须为 `AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>` 类型
   - BlockMmad 必须继承自相应的 BlockMmad 基类
   - BlockEpilogue 必须与 Kernel 类型匹配
   - BlockScheduler 必须提供 tile 切分和调度功能

2. **数据格式**：
   - A、C、Bias：支持 ND 格式
   - B 矩阵：支持 ND 和 NZ 格式

3. **Bias 支持**：可选 bias 输入，通过 `biasGmAddr` 是否为 nullptr 判断

4. **Layout 构建**：使用 `FrameLayoutFormat` 根据数据类型自动适配 layout

## 性能优化建议（公共）

1. **Tile 大小选择**：
   - L1 tile：充分利用 L1 容量（通常 1MB）
   - L0 tile：匹配 L0A/L0B 容量（各 128KB）

2. **Block 数量配置**：根据问题规模合理设置 block 数量

3. **HF32 模式**：FP16 输入 + FP32 输出场景建议启用

4. **数据布局**：权重矩阵（B）优先使用 NZ 格式