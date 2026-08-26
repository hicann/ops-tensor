# Kernel Tqbmm Mx
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_tqbmm_mx.h)

## 功能说明
TQBMM MX 量化 Transpose Batch Matmul Kernel，仅支持 AIC 计算，支持 MxFP4/MxFP8 量化格式。集成 Scale 因子（x1Scale、x2Scale）处理、多 Batch 维度支持、L2 Cache 动态配置，适用于 TQBMM（Transpose Quant Batch MatMul）的 MX 量化推理场景。

> **注意**：TQBMM 与 QBMM 的核心区别在于输入/输出布局。TQBMM 使用 `perm_x1=[1,0,2]`（A 物理布局 `[m, batch, k]`），C 输出物理布局为 `[m, batch, n]`（即 `[batch, m, n]` 累加器经 `perm_y=[1,0,2]` 后存储）。

## 布局约定（与本 Kernel 实现一致）

- **A（x1）**：`perm_x1=[1,0,2]` 时物理存储 `[M, B, K]`（M 轴非连续）。
  - M 步长 `aMStride = batch * k`，Batch 步长 `aBatchStride = k`。
  - MxFP4 时 stride 由 Te 布局直接按 fp4 元素（packed `fp4x2`）解释，**不得再额外右移**（历史 bug 曾对 fp4 多一次 `>>1` 造成 M>=1 错行）。
- **C（y）**：物理存储 `[M, B, N]`（`perm_y=[1,0,2]`）。
  - Batch（B）嵌在 M 平面内：B 步长 `= N`，M 步长 `= B * N`。
  - 历史 bug 曾按 `[B,M,N]` 布局（B 步长 `= M*N`）写 C，导致 B>=2 时输出 m↔2m 交错。
- **x1Scale**：物理布局 `[M, B, scaleKLen]`（M 步长 `= B * scaleKLen`，Batch 步长 `= scaleKLen`；`scaleKLen = ceil(k/64)*2`）。
- **x2Scale**：物理布局 `[B, scaleKLen, N]`。

**参考**：[Kernel Matmul 基础框架](./kernel.md)、[Kernel Qbmm Mx](./kernel_qbmm_mx.md)

## 特殊约束

### 量化格式支持
支持以下量化数据类型：
- **MxFP4**：`fp4x2_e2m1_t`（4-bit 浮点）
- **MxFP8**：`fp8_e4m3fn_t`（8-bit 浮点）

### Scale 因子要求
必须提供两个 Scale 因子：
- `scaleAGmAddr`：A 矩阵的缩放因子（`fp8_e8m0_t` 类型），期望物理布局为 `[M, B, scaleKLen]`（见上文"布局约定"）
- `scaleBGmAddr`：B 矩阵的缩放因子（`fp8_e8m0_t` 类型），期望物理布局为 `[B, scaleKLen, N]`

### 计算模式
仅支持 AIC 模式，不支持 AIV 计算（AIV 核直接返回）。

### BlockScheduler
使用 `BlockSchedulerQuantBatchMatmulV3` 调度器，支持多 Batch 维度切分。

### Batch 维度
支持 4 维 Batch（batchC1/C2/C3/C4），需满足广播规则。x1Scale/x2Scale 的 Batch 维度须分别与 A 矩阵/B 矩阵一致。

## 类型定义

### GemmUniversal (TQBMM MX 特化)
```cpp
template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
class GemmUniversal  // 当 BlockMmad::DispatchPolicy::ScheduleType 为 KernelMmadMultiBlockTQBMM 时特化
```

### TQBMMTiling
```cpp
struct TQBMMTiling {
    uint32_t batchA1, batchA2, batchA3, batchA4;
    uint32_t batchB1, batchB2, batchB3, batchB4;
    uint32_t batchC1, batchC2, batchC3, batchC4;
    uint32_t biasThreeDim;
    uint32_t baseM, baseN, baseK;
    uint32_t isBias;
    uint32_t dbL0C;
    uint32_t bMustHitL2 = 1U;
};
```

### Params
```cpp
struct Params {
    ProblemShape problemShape;
    BlockMmadParams mmadParams;
    L1Params l1Params;
    BlockSchedulerParams schParams;
    TQBMMTiling tqbmmParams;
};
```

## 主要方法

### Run
执行量化 Batch Matmul Kernel 计算。包含构建 Layout、创建 GM Tensor、按 Batch 循环 Tile 切分。

### Init
初始化 GM 地址指针（A、B、C、ScaleA、ScaleB、Bias）。

## 调用示例

```cpp
using AType = fp4x2_e2m1_t;
using BType = fp4x2_e2m1_t;
using CType = half;
using BiasType = float;
using LayoutA = AscendC::Te::NDExtLayoutPtn;
using LayoutB = AscendC::Te::NDExtLayoutPtn;
using LayoutC = AscendC::Te::NDExtLayoutPtn;

using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<
    0, false, Blaze::Gemm::KernelMmadMultiBlockTQBMM>;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
    ProblemShape, 0, LayoutA, LayoutB, AType>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutC>;
using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
using TQBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

TQBMMKernel kernel;
kernel(params);
```

## 与 Kernel Qbmm Mx 的区别

| 项目 | QBMM MX | TQBMM MX |
|------|---------|----------|
| A 布局（permX1） | 无（`[batch, m, k]` 连续） | `[m, batch, k]`（M 轴非连续） |
| C 输出布局 | `[batch, m, n]` | `[m, batch, n]`（B 内嵌 M 平面，B 步长=N, M 步长=B*N） |
| x1Scale 布局 | `[batch, m, scaleKLen]` | `[m, batch, scaleKLen]`（B 内嵌 M 平面） |
| Kernel 类名 | `GemmUniversal` | `GemmUniversal`（不同特化） |
| ScheduleType | `KernelMmadWithScaleMx` | `KernelMmadMultiBlockTQBMM` |
| Tiling 结构 | `QBMMTiling` | `TQBMMTiling` |
| BlockMmad | block_mmad_qbmm_mx.h | block_mmad_qbmm_mx.h（复用） |
