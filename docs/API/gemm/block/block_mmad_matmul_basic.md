# Block Mmad Basic
> [代码位置](../../../../include/blaze/gemm/block/block_mmad_matmul_basic.h)

## 功能说明
基础矩阵乘 Block，基于 Tensor API 实现，仅支持 AIC 计算。支持 L1/L0C 可配置双缓冲、Bias 加法，适用于 Basic Kernel 场景。

**继承自**：[Block Mmad 基础框架](./block_mmad.md)

## 特殊约束

### 调度策略限制
仅支持以下调度策略：
- `MatmulMultiBlockBasic<>`（非全载模式）
- `MatmulMultiBlockBasic<B_FULL_LOAD_MODE>`（B 矩阵全载）
- `MatmulMultiBlockBasic<A_FULL_LOAD_MODE>`（A 矩阵全载）

不支持 `MatmulMultiBlockWithStreamK` 等其他调度策略。

### 计算模式
仅支持 AIC 模式，不支持 AIV 计算。

### 输出目标
结果直接输出到 GM，不支持 workspace。

### weightNZFormat
支持 B 矩阵 NZ 格式，通过 `weightNZFormat` 静态常量标识。

### HF32 模式
HF32 模式由 Kernel 层控制，BlockMmad 层不直接处理。

## 特殊静态常量

| 常量 | 说明 |
|------|------|
| weightNZFormat | B 矩阵是否为 NZ 格式（继承自 BlockMmad，用于 Kernel 层判断） |
| HALF_L0_SIZE | L0 缓冲区半大小（按 A 类型计算） |
| HALF_L0C_SIZE | L0C 缓冲区半大小（按 float 计算） |
| HALF_L1_SIZE | L1 缓冲区半大小 |
| MTE1_MTE2_EVENT_ID_NUM | L1 双缓冲事件标志数量（固定 4 个） |

## 特殊成员方法

### 构造函数
```
__aicore__ inline BlockMmad()
```
功能：构造 BlockMmad 对象，初始化硬件事件标志。
执行流程：设置 4 个 MTE1_MTE2 标志、2 个 FIX_M 标志、2 个 M_MTE1 标志。

### 析构函数
```
__aicore__ inline ~BlockMmad()
```
功能：析构 BlockMmad 对象，等待硬件事件完成。
执行流程：等待 4 个 MTE1_MTE2 标志、2 个 FIX_M 标志、2 个 M_MTE1 标志。

### Init函数
```
template <uint64_t FULL_LOAD_MODE_ = B_FULL_LOAD_MODE>
__aicore__ inline void Init(
    const TupleShape& shape,     // 问题规模
    const TupleShape& tileL1,    // L1 切分形状
    const TupleShape& tileL0,    // L0 切分形状
    bool isBias,                 // 是否启用 bias
    uint64_t l1BufNum,           // L1 缓冲数量（1 或 2）
    bool l0cDB)                  // 是否启用 L0C 双缓冲
```
功能：初始化 BlockMmad 组件。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| shape | TupleShape | 问题规模 |
| tileL1 | TupleShape | L1 tile 形状 |
| tileL0 | TupleShape | L0 tile 形状 |
| isBias | bool | 是否包含 bias 计算 |
| l1BufNum | uint64_t | L1 双缓冲数量，1 或 2 |
| l0cDB | bool | 是否启用 L0C 双缓冲优化 |

说明：
- 模板参数 `FULL_LOAD_MODE_` 用于指定全载模式（默认 B_FULL_LOAD_MODE）
- L1 缓冲数量可配置（1 或 2），影响 GM→L1 流水线并行度
- L0C 双缓冲可配置，实现 L0C 搬出与写入的并行

### operator函数
```
template <typename TensorC, typename TensorA, typename TensorB, typename TensorBias>
__aicore__ inline void operator()(
    TensorC gmC,                // C 矩阵 GM Tensor
    TensorA gmA,                // A 矩阵 GM Tensor
    TensorB gmB,                // B 矩阵 GM Tensor
    TensorBias gmBias,          // Bias GM Tensor
    TupleL1L0Shape tileShape)   // Tile 形状
```
功能：执行单个 block 的矩阵乘计算。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| gmC | TensorC | C 矩阵输出 Tensor（已 Slice 到当前 block） |
| gmA | TensorA | A 矩阵输入 Tensor（已 Slice 到当前 block） |
| gmB | TensorB | B 矩阵输入 Tensor（已 Slice 到当前 block） |
| gmBias | TensorBias | Bias 输入 Tensor（已 Slice） |
| tileShape | TupleL1L0Shape | Tile 形状 `(m, n, k, m0, n0, k0)` |

## 特殊数据结构

### Arguments / Params
```
struct Arguments {
    GM_ADDR aGmAddr;         // A 矩阵 GM 起始地址
    GM_ADDR bGmAddr;         // B 矩阵 GM 起始地址
    GM_ADDR cGmAddr;         // C 矩阵 GM 起始地址
    GM_ADDR biasGmAddr;      // Bias GM 起始地址（可选）
    GM_ADDR groupListGmAddr; // GroupList 地址（预留扩展）
    GM_ADDR workspaceGmAddr; // 工作空间地址（预留扩展）
};
```

说明：`Params` 同 `Arguments`，无 workspace 实际使用。

## 事件同步（Basic 特有）

| 事件 | 用途 |
|------|------|
| MTE1_MTE2 | L1 双缓冲同步（4 个标志） |
| FIX_M | L0C 双缓冲同步（2 个标志） |
| M_MTE1 | L0 双缓冲同步（2 个标志） |
| MTE2_MTE1 | GM→L1 完成同步 |
| MTE1_M | L1→L0 完成同步 |
| M_FIX | Mmad 计算完成同步 |

说明：事件数量可通过 Init 参数配置启用/禁用。

## 调用示例

### 组件组装
```
using AType = half;
using BType = half;
using CType = float;
using BiasType = float;
using LayoutA = AscendC::Te::Layout::RowMajor;
using LayoutB = AscendC::Te::Layout::ColMajor;
using LayoutC = AscendC::Te::Layout::RowMajor;
using LayoutBias = LayoutC;

using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<B_FULL_LOAD_MODE>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;
```

### 组件实例化
```
BlockMmad blockMmad;
```

### 组件初始化
```
TupleShape problemShape{m, n, k, batch};
TupleShape tileL1{mL1, nL1, kL1, 0, 0, 0};
TupleShape tileL0{baseM, baseN, baseK, 0, 0, 0};
bool isBias = true;
uint64_t l1BufNum = 2;  // L1 双缓冲
bool l0cDB = true;       // L0C 双缓冲
blockMmad.Init(problemShape, tileL1, tileL0, isBias, l1BufNum, l0cDB);
```

### 组件执行
```
// 准备 GM Tensor（已在 kernel 层创建）
auto gmA = AscendC::Te::MakeTensor(...);
auto gmB = AscendC::Te::MakeTensor(...);
auto gmC = AscendC::Te::MakeTensor(...);
auto gmBias = AscendC::Te::MakeTensor(...);

// Slice 到当前 block
auto gmBlockA = gmA.Slice(AscendC::MakeCoord(coordM, 0), AscendC::MakeShape(shapeM, shapeK));
auto gmBlockB = gmB.Slice(AscendC::MakeCoord(0, coordN), AscendC::MakeShape(shapeK, shapeN));
auto gmBlockC = gmC.Slice(AscendC::MakeCoord(coordM, coordN), AscendC::MakeShape(shapeM, shapeN));
auto gmBlockBias = gmBias.Slice(AscendC::MakeCoord(0, coordN), AscendC::MakeShape(1, shapeN));

// 执行矩阵乘
TupleL1L0Shape tileShape{shapeM, shapeN, shapeK, baseM, baseN, baseK};
blockMmad(gmBlockC, gmBlockA, gmBlockB, gmBlockBias, tileShape);
```

## 数据流

### 存储层次（Basic 特有）
```
GM → L1 (双缓冲) → L0A/L0B (双缓冲) → L0C (双缓冲) → GM
                    ↓
                  BIAS → L1 → BIAS Buffer
```

### 执行流程
```
K 轴外层循环：按 kL1 切分
    ↓
搬运 A、B、Bias 到 L1
    ↓
K 轴内层循环：按 baseK 切分
    ↓
搬运 A、B 到 L0
    ↓
Mmad 计算：C += A × B + Bias（首次迭代）
    ↓
结果搬出：L0C → GM
```

## 性能优化建议（Basic 特有）

### L1 缓冲配置
- 大矩阵场景：建议 `l1BufNum = 2` 最大化流水线并行度
- 小矩阵场景：可使用 `l1BufNum = 1` 减少缓冲开销

### L0C 双缓冲
- 启用 L0C 双缓冲（`l0cDB = true`）可隐藏搬出延迟
- 禁用时（`l0cDB = false`）可减少事件同步开销

### 全载模式选择
- **非全载模式**：每次迭代重新加载 A/B 块，适用于小 K 场景
- **B 全载模式**：B 矩阵常驻 L1，适用于大 K、小 N 场景
- **A 全载模式**：A 矩阵常驻 L1，适用于大 K、小 M 场景

### 适用场景
- Basic Kernel 的 BlockMmad 实现
- 不需要 workspace 中间结果
- 不需要 AIC-AIV 跨核同步