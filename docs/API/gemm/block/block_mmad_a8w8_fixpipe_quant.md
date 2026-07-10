# Block Mmad A8W8 Fixpipe Quant
> [代码位置](../../../../include/blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h)

## 功能说明
Fixpipe 量化矩阵乘 Block，基于 Tensor API 实现，仅支持 AIC 计算。该组件执行由 `AType/BType` 指定的量化 A/B Cube Mmad 累加，并在 L0C 搬出阶段通过 Fixpipe 完成反量化，适用于 QBMM Cube Kernel 场景。文件名沿用 A8W8 路径命名，但实际输入类型不限定为 `int8_t`。

**继承自**：[Block Mmad 基础框架](./block_mmad.md)

## 特殊约束

### 调度策略限制
仅支持以下调度策略：
- `MatmulWithScaleFixpipeQuant<>`（非全载模式）
- `MatmulWithScaleFixpipeQuant<A_FULL_LOAD_MODE>`（A 矩阵全载模式）
- `MatmulWithScaleFixpipeQuant<0, true>`（Atomic Add 非全载模式）
- `MatmulWithScaleFixpipeQuant<A_FULL_LOAD_MODE, true>`（Atomic Add A 矩阵全载模式）

`MatmulWithScaleFixpipeQuant` 的 `ScheduleType` 固定为 `KernelMmadWithScaleFixpipeQuant`。Atomic Add 标志由 Kernel 层读取并配置，Block 内部不直接设置 atomic 状态。

不支持 `MatmulWithScaleMx`、`GroupedMatmulWithScaleMx`、`MatmulMultiBlockBasic` 或 `MatmulMultiBlockWithStreamK`。

### 量化数据类型支持
| 数据类型 | 说明 | C0_SIZE |
|---------|------|---------|
| int8_t | A8W8 量化输入 | 由 `C0_ELEMENT<AType>` 决定，通常为 32 |
| hifloat8_t | HiFloat8 量化输入 | 由 `C0_ELEMENT<AType>` 决定，通常为 32 |
| fp8_e5m2_t | FP8 E5M2 量化输入 | 由 `C0_ELEMENT<AType>` 决定，通常为 32 |
| fp8_e4m3fn_t | FP8 E4M3FN 量化输入 | 由 `C0_ELEMENT<AType>` 决定，通常为 32 |

说明：
- A 矩阵类型由 `AType` 指定。
- B 矩阵模板参数为 `BTypeTuple`，其中第 0 个类型为 B 数据类型，第 1 个类型为 X2 scale GM 类型。
- 典型 A/B 组合包括 `int8_t/int8_t`、`hifloat8_t/hifloat8_t`、`fp8_e4m3fn_t/fp8_e4m3fn_t`、`fp8_e4m3fn_t/fp8_e5m2_t`、`fp8_e5m2_t/fp8_e4m3fn_t` 和 `fp8_e5m2_t/fp8_e5m2_t`，最终以 Tensor API Mmad 静态检查为准。
- L0C 累加类型由 `AscendC::GetMmDstType<AType>::Type` 推导，`int8_t` 输入通常累加为 `int32_t`，HiFloat8/FP8 输入通常累加为 `float`。

### Scale 因子类型
Fixpipe 反量化使用 X2 scale，支持两类输入方式：
- **Scalar scale**：`operator()` 的 `scaleGlobal` 为 `uint64_t`，Fixpipe 搬出时使用标量 scale。
- **Per-channel scale**：`scaleGlobal` 为 Tensor，Block 将当前 N 分片的 scale 搬入 L1，并在 Fixpipe 搬出时使用 L1 scale Tensor。

当 `CType` 为 `int32_t` 时，结果直接以累加值搬出，不使用 scale 反量化。

### 计算模式
仅支持 AIC 模式，不支持 AIV 计算。

### 输出目标
结果直接输出到 GM，不支持 workspace。输出类型由 `CType` 指定：
- `CType = int32_t`：L0C 累加结果直接搬出。
- `CType = half/bfloat16_t/float`：通过 Fixpipe 搬出并应用 scalar/per-channel scale。

### L1 切分要求
- `kAL1`：A 矩阵 L1 K 轴切分大小。
- `kBL1`：B 矩阵 L1 K 轴切分大小。
- `l1BufferNum` 支持 2 或 4 缓冲。
- 非全载模式下，当 `l1BufferNum == 2` 时支持 `kAL1` 与 `kBL1` 不同，并按 A/B 的 K-L1 大小选择复用策略。
- A 全载模式下，A 常驻 L1，`kAL1` 跟随 `kBL1`，适用于 A 分片复用收益明显的场景。
- 注意：kAL1与kBL1不相等时必须满足整数倍关系。

### Mmad 计算模式
使用默认 Mmad trait，执行量化 A/B Cube Mmad：
```
using MmadAtomT = AscendC::Te::MmadAtom<
    AscendC::Te::MmadTraits<
        AscendC::Te::MmadOperation,
        AscendC::Te::MmadTraitDefault>>;

AscendC::Te::Mmad(MmadAtomT{}.with(mmadParams), c1Local, l0aLocal, l0bLocal);
```

Bias 仅在首个 K-L1/L0 迭代搬入 BT 并参与 Mmad。

## 特殊静态常量

| 常量 | 说明 |
|------|------|
| WEIGHT_NZ | B 矩阵是否为 NZ 格式 |
| TRANS_A | A 矩阵是否转置 |
| TRANS_B | B 矩阵是否转置 |
| C0_SIZE | A/B C0 对齐大小，由 `AscendC::Te::C0_ELEMENT<AType>` 推导 |
| L0C_C0 | L0C C0 对齐大小，固定为 16 |
| SCALE_BUFFER_NUM | X2 scale L1 缓冲数量，固定为 2 |
| AB_L1_TWO_BUFFER | A/B L1 双缓冲标志，固定为 2 |

## 特殊类型别名

| 类型 | 说明 |
|------|------|
| BType | `BTypeTuple` 的第 0 个类型，表示 B 矩阵数据类型 |
| X2ScaleType | `BTypeTuple` 的第 1 个类型，表示 scale GM 类型 |
| L0CType | Mmad 累加输出类型，由 `AType` 推导 |
| MakeLayoutAL1 | A 矩阵 L1 Layout 构建器（根据 TRANS_A 选择 ZN/NZ） |
| MakeLayoutBL1 | B 矩阵 L1 Layout 构建器（根据 TRANS_B 选择 ZN/NZ） |

## 特殊数据结构

### Params
```
struct Params {
    GM_ADDR aGmAddr{nullptr};      // A 矩阵 GM 地址
    GM_ADDR bGmAddr{nullptr};      // B 矩阵 GM 地址
    GM_ADDR cGmAddr{nullptr};      // C 矩阵 GM 地址
    GM_ADDR biasGmAddr{nullptr};   // Bias GM 地址
    GM_ADDR scaleAGmAddr{nullptr}; // A 矩阵 Scale GM 地址
    GM_ADDR scaleBGmAddr{nullptr}; // B 矩阵 Scale GM 地址
};
```

说明：
- `scaleAGmAddr` 由 Kernel 层用于 per-tensor scale 融合，Block 主要消费传入的 scalar scale 或 `scaleBGmAddr` 对应的 per-channel scale Tensor。
- `biasGmAddr` 可选；无 bias 时 Kernel 会传入占位 Tensor Slice。

## 接口概要

Block 对外主要通过初始化接口和调用接口完成单 block 计算：
- 初始化阶段配置问题规模、L0 tile、`kAL1/kBL1`、L1 缓冲数量、X2 scale 模式、Bias 标志和 L0C 双缓冲标志。
- 执行阶段接收当前 block 的 A/B/C/Bias Tensor，以及 scalar scale 或 per-channel scale Tensor，完成量化 Mmad 累加和 Fixpipe 搬出。

非全载模式下，`l1BufferNum == 2` 时可分别使用 `kAL1/kBL1`；`l1BufferNum != 2` 时使用统一 K-L1 窗口。A 全载模式下，A 常驻 L1，适用于 A 分片复用收益明显的场景。

## 事件同步

| 事件 | 用途 |
|------|------|
| MTE1_MTE2 (0-3) | A/B 数据 L1 缓冲同步 |
| MTE1_MTE2 (4-5) | Bias L1 双缓冲同步 |
| FIX_MTE2 (0-1) | X2 scale L1 双缓冲同步 |
| MTE2_FIX (0-1) | scale GM 到 L1 后通知 Fixpipe |
| MTE2_MTE1 | GM 到 L1 搬运完成后触发 L1 到 L0 |
| MTE1_M | L1 到 L0 搬运完成后触发 Mmad |
| M_MTE1 | Mmad 与下一轮 L0 搬运同步 |

## 调用示例

### 组件组装
```
// 以下以 int8_t A/B 为例，可按 Tensor API 支持组合替换为其他类型。
using AType = int8_t;
using BType = int8_t;
using CType = bfloat16_t;
using BiasType = int32_t;
using X2ScaleType = uint64_t;

using LayoutA = AscendC::Te::NDExtLayoutPtn;
using LayoutB = AscendC::Te::NDExtLayoutPtn;
using LayoutC = AscendC::Te::NDExtLayoutPtn;
using LayoutBias = AscendC::Te::NDExtLayoutPtn;

using BTypeTuple = AscendC::Std::tuple<BType, X2ScaleType>;
using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<0, false>;

using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BTypeTuple, LayoutB, CType, LayoutC, BiasType, LayoutBias>;
```

### 组件初始化
```
BlockMmad blockMmad;
BlockMmad::ProblemShape problemShape{m, n, k, batch};
BlockMmad::BlockShape l0TileShape{baseM, baseN, baseK, 0};

blockMmad.Init(
    problemShape, l0TileShape,
    kAL1, kBL1, l1BufferNum,
    QuantMode::PERTENSOR_MODE,
    isBias, dbL0C);
```

### 组件执行
```
auto gmBlockA = gmA.Slice(...);
auto gmBlockB = gmB.Slice(...);
auto gmBlockBias = gmBias.Slice(...);
auto gmBlockC = gmC.Slice(...);

uint64_t scalarScale = ...;
BlockMmad::BlockShape singleShape{curM, curN, k, 0};

blockMmad(gmBlockA, gmBlockB, scalarScale, gmBlockBias, gmBlockC, singleShape);
```

## 数据流

### 存储层次
```
GM (量化 A/B) + GM (scale/bias)
    ↓
L1 (A/B 数据缓冲 + X2 scale + Bias)
    ↓
L0A/L0B (量化数据) + BT (Bias)
    ↓
L0C (由 L0CType 决定的累加类型)
    ↓
Fixpipe (scale 反量化)
    ↓
GM (C 输出)
```

### L1 缓冲布局
```
非全载：
A0|B0|A1|B1|...|Scale0|Bias0|Scale1|Bias1

A 全载：
B0|Scale0|Bias0|A|...|B1|Scale1|Bias1
```

### 执行流程
```
初始化 L1/L0 tile 和 scale/bias 缓冲
    ↓
搬运 scalar 或 per-channel scale
    ↓
搬运 Bias 到 L1（可选）
    ↓
根据 kAL1/kBL1 选择 K-L1 主循环
    ↓
K-L0 循环执行量化 Mmad
    ↓
Fixpipe 搬出并完成反量化
```

## 性能优化建议

### L1 K 轴配置
- `kAL1 == kBL1`：A/B 同步推进，控制逻辑最简单。
- `kAL1 > kBL1`：复用 A，适合 A 搬运压力较大或 A 全载收益明显的场景。
- `kBL1 > kAL1`：复用 B，适合 B 搬运压力较大或 B 数据复用较高的场景。

### L1 缓冲数量
- `l1BufferNum = 2`：支持 A/B 不同 K-L1 窗口。
- `l1BufferNum = 4`：提高数据搬运流水并行度，但内部统一使用 `min(kAL1, kBL1)`。

### Scale 模式选择
- per-tensor scale 使用 scalar 传入，搬运开销最低。
- per-channel scale 需要将当前 N 分片 scale 搬入 L1，适合精度要求更高的场景。

### L0C 双缓冲
- `dbL0C > 1` 时启用 L0C ping-pong，可提升连续 block 搬出效率。
- `CType = int32_t` 时无反量化 scale 搬出，适合调试或保留累加结果的场景。
