# Block Mmad Mx
> [代码位置](../../../include/blaze/block/block_mmad_mx.h)

## 功能说明
MX 量化矩阵乘 Block，基于 Tensor API 实现，仅支持 AIC 计算。支持 MxFP4/MxFP8 量化、Scale 因子处理（ScaleA + ScaleB）、L1/L0 双缓冲优化，适用于量化 Batch Matmul Kernel 场景。

**继承自**：[Block Mmad 基础框架](./block_mmad.md)

## 特殊约束

### 调度策略限制
仅支持以下调度策略：
- `MatmulWithScaleMx<>`（非全载模式）
- `MatmulWithScaleMx<A_FULL_LOAD_MODE>`（A 矩阵全载模式）

不支持 `MatmulMultiBlockBasic` 或 `MatmulMultiBlockWithStreamK`。

### 量化数据类型支持
| 数据类型 | 说明 | C0_SIZE |
|---------|------|---------|
| fp4x2_e2m1_t | MxFP4 E2M1 格式 | 64 |
| fp4x2_e1m2_t | MxFP4 E1M2 格式 | 64 |
| fp8_e5m2_t | MxFP8 E5M2 格式 | 32 |
| fp8_e4m3fn_t | MxFP8 E4M3FN 格式 | 32 |

说明：C0_SIZE 用于 L1/L0 layout 对齐，FP4 需要更大的对齐。

### Scale 因子类型
Scale 因子固定为 `fp8_e8m0_t`（E8M0 浮点格式），用于量化数据的反量化。

### 计算模式
仅支持 AIC 模式，不支持 AIV 计算。

### 输出目标
结果直接输出到 GM，不支持 workspace。

### MXFP 对齐要求
- K 轴需对齐到 `MXFP_DIVISOR_SIZE`（64）
- Scale K 轴需对齐到 `MXFP_DIVISOR_SIZE × MXFP_MULTI_BASE_SIZE`（128）
- L0C 布局固定使用 NZLayoutPtn

### Mmad 计算模式
使用 `MmadTraitMX` trait，支持量化数据的自动反量化：
```
AscendC::Te::Mmad(
    AscendC::Te::MmadAtom<
        AscendC::Te::MmadTraits<
            AscendC::Te::MmadOperation,
            AscendC::Te::MmadTraitMX>>{},
    tensorL0C, tensorAL0, tensorBL0);
```

## 特殊静态常量

| 常量 | 说明 |
|------|------|
| weightNz | B 矩阵是否为 NZ 格式 |
| transA | A 矩阵是否转置 |
| transB | B 矩阵是否转置 |
| C0_SIZE | C0 对齐大小（FP4: 64，FP8: 32） |
| SCALE_C0 | Scale C0 对齐大小（固定为 2） |
| SCALE_BUFFER_NUM | Scale 缓冲数量（固定为 2） |
| HALF_L0_SIZE | L0 缓冲区半大小 |
| HALF_L0C_SIZE | L0C 缓冲区半大小 |

## 特殊类型别名

| 类型 | 说明 |
|------|------|
| MxL0AType | A 矩阵 L0 数据类型（通过 GetL0DataType 获取） |
| MxL0BType | B 矩阵 L0 数据类型（通过 GetL0DataType 获取） |
| MakeLayoutScaleA | ScaleA Layout 构建器（根据 transA 选择 ScaleADN/ScaleAND） |
| MakeLayoutScaleB | ScaleB Layout 构建器（根据 transB 选择 ScaleBDN/ScaleBND） |

## 特殊数据结构

### Params
```
struct Params {
    GM_ADDR aGmAddr;             // A 矩阵 GM 起始地址
    GM_ADDR bGmAddr;             // B 矩阵 GM 起始地址
    GM_ADDR cGmAddr;             // C 矩阵 GM 起始地址
    GM_ADDR biasGmAddr;          // Bias GM 起始地址（可选）
    GM_ADDR pertokenScaleGmAddr; // A 矩阵 Scale GM 地址
    GM_ADDR scaleGmAddr;         // B 矩阵 Scale GM 地址
};
```

### L1Params
```
struct L1Params {
    uint64_t kL1;        // L1 K 轴切分大小
    uint64_t scaleKL1;   // Scale K 轴切分大小
    uint64_t l1BufNum;   // L1 缓冲数量（2 或 4）
};
```

### TileL1L0Param
```
struct TileL1L0Param {
    uint64_t curM;        // 当前 M 维大小
    uint64_t curN;        // 当前 N 维大小
    uint64_t curGmAKL1;   // A 矩阵 GM K 维大小
    uint64_t curGmBKL1;   // B 矩阵 GM K 维大小
    uint64_t curPadAKL1;  // A 矩阵对齐后的 K 维大小（对齐到 64）
    uint64_t curPadBKL1;  // B 矩阵对齐后的 K 维大小（对齐到 64）
};
```

## 特殊成员方法

### 构造函数
```
__aicore__ inline BlockMmad()
```
功能：构造 BlockMmad 对象，初始化硬件事件标志。

### 析构函数
```
__aicore__ inline ~BlockMmad()
```
功能：析构 BlockMmad 对象，等待硬件事件完成。

### Init函数
```
__aicore__ inline void Init(
    const TupleShape& problemShape,   // 问题规模 (m, n, k)
    const BlockShape& l0TileShape,    // L0 tile 形状 (baseM, baseN, baseK)
    const L1Params& l1Params,         // L1 参数 (kL1, scaleKL1, l1BufNum)
    bool isBias,                      // 是否启用 bias
    bool dbL0C)                       // 是否启用 L0C 双缓冲
```
功能：初始化 BlockMmadMX 组件。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| problemShape | TupleShape | 问题规模 |
| l0TileShape | BlockShape | L0 tile 形状 |
| l1Params | L1Params | L1 参数（kL1、scaleKL1、l1BufNum） |
| isBias | bool | 是否包含 bias 计算 |
| dbL0C | bool | 是否启用 L0C 双缓冲 |

说明：
- `l1BufNum` 支持 2 或 4 缓冲
- `scaleKL1` 应为 `kL1` 的整数倍（建议 2×kL1）
- K 轴需对齐到 MXFP_DIVISOR_SIZE（64）

### CopyScalesInL1函数
```
template <typename TensorScaleA, typename TensorScaleB>
__aicore__ inline void CopyScalesInL1(
    TensorScaleA const& gmScaleA,    // ScaleA GM Tensor
    TensorScaleB const& gmScaleB,    // ScaleB GM Tensor
    TileL1L0Param& tileL1L0Param,    // Tile 参数
    uint64_t scaleL1BufId,           // Scale L1 缓冲 ID
    uint64_t iter0)                  // K 轴迭代索引
```
功能：搬运 ScaleA 和 ScaleB 到 L1。
说明：
- Scale 仅在 `iter0 % (scaleKL1 / kL1) == 0` 时搬运（复用）
- A 全载模式：ScaleA 仅首次搬运，常驻 L1

### CopyAInL1函数
```
template <typename TensorA>
__aicore__ inline void CopyAInL1(
    TensorA const& gmA,              // A 矩阵 GM Tensor
    TileL1L0Param& tileL1L0Param,    // Tile 参数
    uint64_t l1BufId,                // L1 缓冲 ID
    uint64_t iter0)                  // K 轴迭代索引
```
功能：搬运 A 矩阵到 L1，包含 MXFP K 轴 Padding。
说明：
- 使用 `PadMxKAL1::PadZero` 进行 K 轴 Padding（对齐到 64）
- A 全载模式：A 仅首次搬运，常驻 L1

### Iterate函数
```
__aicore__ inline void Iterate(
    TileL1L0Param& tileL1L0Param,    // Tile 参数
    uint64_t iter0)                  // K 轴迭代索引
```
功能：执行 K 轴内层循环，搬运数据到 L0 并执行 Mmad。
执行流程：
1. K 轴内层循环（按 baseK 切分）
2. 搬运 A、ScaleA、B、ScaleB、Bias 到 L0
3. 执行 MX Mmad 计算（MmadTraitMX）
4. 事件同步（M_MTE1）

### operator函数
```
template <
    typename TensorA, typename TensorB, typename TensorScaleA,
    typename TensorScaleB, typename TensorBias, typename TensorC>
__aicore__ inline void operator()(
    TensorA const& gmA,              // A 矩阵 GM Tensor
    TensorB const& gmB,              // B 矩阵 GM Tensor
    TensorScaleA const& gmScaleA,    // ScaleA GM Tensor
    TensorScaleB const& gmScaleB,    // ScaleB GM Tensor
    TensorBias const& gmBias,        // Bias GM Tensor
    TensorC const& gmC,              // C 矩阵输出 GM Tensor
    BlockShape const& singleShape)   // Tile 形状
```
功能：执行单个 block 的量化矩阵乘计算。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| gmA | TensorA | A 矩阵输入 Tensor（量化数据） |
| gmB | TensorB | B 矩阵输入 Tensor（量化数据） |
| gmScaleA | TensorScaleA | A 矩阵 Scale Tensor（fp8_e8m0_t） |
| gmScaleB | TensorScaleB | B 矩阵 Scale Tensor（fp8_e8m0_t） |
| gmBias | TensorBias | Bias 输入 Tensor |
| gmC | TensorC | C 矩阵输出 Tensor（float） |
| singleShape | BlockShape | Tile 形状 |

## 事件同步（MX 特有）

| 事件 | 用途 |
|------|------|
| MTE1_MTE2 (0-3) | 数据缓冲双缓冲同步（4 个标志） |
| MTE1_MTE2 (4-5) | Scale 缓冲双缓冲同步（2 个标志） |
| FIX_M (0-1) | L0C 双缓冲同步（2 个标志） |
| M_MTE1 | L0 双缓冲同步 |

说明：
- Scale 使用独立的缓冲区（SCALE_BUFFER_FLAG_0、SCALE_BUFFER_FLAG_1）
- 数据和 Scale 分别管理双缓冲

## 调用示例

### 组件组装
```
// 定义量化数据类型
using AType = fp4x2_e2m1_t;  // 或 fp8_e5m2_t
using BType = fp4x2_e2m1_t;  // 或 fp8_e5m2_t
using CType = float;
using BiasType = float;
using ScaleType = fp8_e8m0_t;

// 定义 Layout
using LayoutA = AscendC::Te::NDExtLayoutPtn;
using LayoutB = AscendC::Te::NZLayoutPtn;
using LayoutC = AscendC::Te::NDExtLayoutPtn;
using LayoutBias = AscendC::Te::NDExtLayoutPtn;

// 定义调度策略
using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<A_FULL_LOAD_MODE>;

// 定义 BlockMmadMX
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;
```

### 组件初始化
```
BlockMmad blockMmad;
TupleShape problemShape{m, n, k};
BlockShape l0TileShape{baseM, baseN, baseK, 0};
BlockMmad::L1Params l1Params{kL1, scaleKL1, l1BufNum};
bool isBias = true;
bool dbL0C = true;
blockMmad.Init(problemShape, l0TileShape, l1Params, isBias, dbL0C);
```

### 组件执行
```
// 准备 GM Tensor（量化数据）
auto gmA = AscendC::Te::MakeTensor(...);      // 量化数据
auto gmB = AscendC::Te::MakeTensor(...);      // 量化数据
auto gmScaleA = AscendC::Te::MakeTensor(...); // fp8_e8m0_t
auto gmScaleB = AscendC::Te::MakeTensor(...); // fp8_e8m0_t
auto gmBias = AscendC::Te::MakeTensor(...);
auto gmC = AscendC::Te::MakeTensor(...);      // float 输出

// Slice 到当前 tile
auto gmBlockA = gmA.Slice(...);
auto gmBlockB = gmB.Slice(...);
auto gmBlockScaleA = gmScaleA.Slice(...);
auto gmBlockScaleB = gmScaleB.Slice(...);
auto gmBlockBias = gmBias.Slice(...);
auto gmBlockC = gmC.Slice(...);

// 执行量化矩阵乘
BlockShape singleShape{shapeM, shapeN, shapeK, 0};
blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleShape);
```

## 数据流

### 存储层次（MX 特有）
```
GM (量化A/B) + GM (ScaleA/ScaleB)
    ↓
L1 (量化数据缓冲 + Scale 缓冲)
    ↓
L0A/L0B (量化数据) + L0A/L0B (Scale)
    ↓
L0C (float 结果)
    ↓
GM (float 输出)
```

### L1 缓冲布局（2 buffer）
```
非全载：A0|B0|ScaleA0|ScaleB0|Bias0|...|A1|B1|ScaleA1|ScaleB1|Bias1|
A全载：B0|ScaleB0|Bias0|A|ScaleA|...|B1|ScaleB1|Bias1|
```

### 执行流程
```
K 轴外层循环：按 kL1 切分
    ↓
搬运 ScaleA/ScaleB 到 L1（复用策略）
    ↓
搬运 A/B/Bias 到 L1（MXFP Padding）
    ↓
K 轴内层循环：按 baseK 切分（Iterate）
    ↓
搬运 A/B/ScaleA/ScaleB 到 L0
    ↓
Mmad 计算（MX 模式：自动反量化）
    ↓
结果搬出：L0C → GM
```

## 性能优化建议（MX 特有）

### Scale KL1 配置
- `scaleKL1` 建议为 `kL1` 的 2-4 倍
- Scale 复用：减少 Scale 搬运次数，提升性能

### L1 缓冲数量
- `l1BufNum = 4`：最大化流水线并行度（推荐）
- `l1BufNum = 2`：减少 L1 占用（小矩阵场景）

### K 轴 Padding
- K 轴自动 Padding 到 MXFP_DIVISOR_SIZE（64）
- Padding 数据为 0，不影响计算结果

### 全载模式选择
- **非全载模式**：每次迭代重新加载 A/B 块
- **A 全载模式**：A 矩阵常驻 L1，ScaleA 常驻 L1，适用于大 K、小 M 场景

### Scale 复用策略
- Scale 在 `iter0 % (scaleKL1 / kL1) == 0` 时搬运
- 每 `scaleKL1 / kL1` 次迭代复用同一 Scale 数据

### 适用场景
- **量化推理**：MxFP4/MxFP8 量化矩阵乘
- **Batch Matmul**：多 Batch 维度支持（在 Kernel 层处理）
- **Scale 因子处理**：per-token 和 per-group scale
- **高性能场景**：L1 缓冲优化、Scale 复用、全载模式